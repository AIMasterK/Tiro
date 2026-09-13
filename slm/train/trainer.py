"""
SLM 학습 루프
- Cosine LR 스케줄 + Warmup
- Gradient clipping
- 체크포인트 저장/복원
- 학습 로그 (W&B 선택)
"""

import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


# ── 학습 설정 ──────────────────────────────────

@dataclass
class TrainConfig:
    # 경로
    out_dir:        str   = "checkpoints"
    data_path:      str   = "data/train.jsonl"

    # 학습 하이퍼파라미터
    max_steps:      int   = 10_000
    eval_interval:  int   = 500
    save_interval:  int   = 1_000
    batch_size:     int   = 32
    grad_accum:     int   = 4          # effective batch = batch_size * grad_accum
    max_seq_len:    int   = 256

    # 옵티마이저
    lr:             float = 3e-4
    min_lr:         float = 3e-5
    weight_decay:   float = 0.1
    grad_clip:      float = 1.0
    beta1:          float = 0.9
    beta2:          float = 0.95

    # LR 스케줄
    warmup_steps:   int   = 500

    # 기타
    device:         str   = "cuda" if torch.cuda.is_available() else "cpu"
    dtype:          str   = "bfloat16"   # bfloat16 / float32
    compile:        bool  = False         # torch.compile (PyTorch 2.0+)
    seed:           int   = 42

    # Loss 가중치 (Goal/Emotion head는 레이블 있을 때만)
    lm_weight:      float = 1.0
    goal_weight:    float = 0.1          # goal_logits에 대한 보조 loss 가중치
    emo_weight:     float = 0.1


# ── 데이터셋 ───────────────────────────────────

class ConversationDataset(Dataset):
    """
    JSONL 포맷: {"input": "...", "output": "...", "goal": 0, "emotion": 2}
    goal / emotion 레이블은 선택 (없으면 None)
    """
    def __init__(self, path: str, tokenizer, max_len: int = 256):
        import json
        self.samples = []
        self.tokenizer = tokenizer
        self.max_len = max_len

        with open(path, encoding="utf-8") as f:
            for line in f:
                item = json.loads(line.strip())
                self.samples.append(item)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        usr = self.tokenizer.special_tokens["<usr>"]
        bot = self.tokenizer.special_tokens["<bot>"]
        bos = self.tokenizer.special_tokens["<bos>"]
        eos = self.tokenizer.special_tokens["<eos>"]
        pad = self.tokenizer.special_tokens["<pad>"]

        # [BOS, <usr>, ...input tokens..., <bot>, ...output tokens..., EOS]
        inp_ids = self.tokenizer.encode(item["input"],  add_special=False)
        out_ids = self.tokenizer.encode(item["output"], add_special=False)
        ids = [bos, usr] + inp_ids + [bot] + out_ids + [eos]

        # 레이블: input 부분은 -100 (loss 제외), output 부분만 학습
        n_input = len([bos, usr] + inp_ids + [bot])
        labels = [-100] * n_input + out_ids + [eos]

        # 자르기 / 패딩
        ids    = ids[:self.max_len]
        labels = labels[:self.max_len]
        pad_len = self.max_len - len(ids)
        padding_mask = [1] * len(ids) + [0] * pad_len
        ids    += [pad] * pad_len
        labels += [-100] * pad_len

        return {
            "input_ids":    torch.tensor(ids,          dtype=torch.long),
            "labels":       torch.tensor(labels,       dtype=torch.long),
            "padding_mask": torch.tensor(padding_mask, dtype=torch.long),
            "goal_label":   torch.tensor(item.get("goal", -1),    dtype=torch.long),
            "emo_label":    torch.tensor(item.get("emotion", -1), dtype=torch.long),
        }


# ── LR 스케줄 ──────────────────────────────────

def get_lr(step: int, cfg: TrainConfig) -> float:
    if step < cfg.warmup_steps:
        return cfg.lr * step / cfg.warmup_steps
    progress = (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return cfg.min_lr + coeff * (cfg.lr - cfg.min_lr)


# ── Trainer ────────────────────────────────────

class Trainer:
    def __init__(self, model, tokenizer, cfg: TrainConfig):
        self.model     = model
        self.tokenizer = tokenizer
        self.cfg       = cfg

        torch.manual_seed(cfg.seed)
        self.device = torch.device(cfg.device)
        self.model.to(self.device)

        if cfg.compile:
            print("torch.compile 활성화...")
            self.model = torch.compile(self.model)

        self.scaler = torch.cuda.amp.GradScaler(enabled=(cfg.dtype == "float16"))
        self.dtype  = torch.bfloat16 if cfg.dtype == "bfloat16" else torch.float32

        self.optimizer = self._build_optimizer()
        Path(cfg.out_dir).mkdir(parents=True, exist_ok=True)

    def _build_optimizer(self):
        # weight decay는 2D 파라미터(행렬)에만
        decay, no_decay = [], []
        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            if p.dim() >= 2:
                decay.append(p)
            else:
                no_decay.append(p)
        return torch.optim.AdamW(
            [{"params": decay, "weight_decay": self.cfg.weight_decay},
             {"params": no_decay, "weight_decay": 0.0}],
            lr=self.cfg.lr,
            betas=(self.cfg.beta1, self.cfg.beta2),
        )

    def _compute_loss(self, batch) -> torch.Tensor:
        cfg = self.cfg
        out = self.model(
            input_ids    = batch["input_ids"].to(self.device),
            padding_mask = batch["padding_mask"].to(self.device),
            labels       = batch["labels"].to(self.device),
        )
        loss = cfg.lm_weight * out["lm_loss"]

        # 보조 loss: goal
        goal_labels = batch["goal_label"].to(self.device)
        valid_goal  = (goal_labels >= 0)
        if valid_goal.any():
            goal_loss = nn.functional.cross_entropy(
                out["goal_logits"][valid_goal], goal_labels[valid_goal]
            )
            loss = loss + cfg.goal_weight * goal_loss

        # 보조 loss: emotion
        emo_labels = batch["emo_label"].to(self.device)
        valid_emo  = (emo_labels >= 0)
        if valid_emo.any():
            emo_loss = nn.functional.cross_entropy(
                out["emo_logits"][valid_emo], emo_labels[valid_emo]
            )
            loss = loss + cfg.emo_weight * emo_loss

        return loss

    def train(self, train_dataset, val_dataset=None):
        cfg = self.cfg
        loader = DataLoader(
            train_dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
        )

        step = 0
        self.optimizer.zero_grad()
        t0 = time.time()

        print(f"\n🚀 학습 시작 | device={cfg.device} | dtype={cfg.dtype}")
        print(f"   모델: {self.model.num_params()/1e6:.2f}M params")
        print(f"   스텝: {cfg.max_steps} | LR: {cfg.lr} | batch: {cfg.batch_size}×{cfg.grad_accum}\n")

        while step < cfg.max_steps:
            for batch in loader:
                if step >= cfg.max_steps:
                    break

                # LR 업데이트
                lr = get_lr(step, cfg)
                for pg in self.optimizer.param_groups:
                    pg["lr"] = lr

                # Forward
                with torch.autocast(device_type=cfg.device.split(":")[0], dtype=self.dtype):
                    loss = self._compute_loss(batch) / cfg.grad_accum

                self.scaler.scale(loss).backward()

                if (step + 1) % cfg.grad_accum == 0:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()

                # 로깅
                if step % 100 == 0:
                    elapsed = time.time() - t0
                    print(f"  step {step:5d} | loss {loss.item()*cfg.grad_accum:.4f} | "
                          f"lr {lr:.2e} | {elapsed:.1f}s")
                    t0 = time.time()

                # 평가
                if val_dataset and step % cfg.eval_interval == 0 and step > 0:
                    val_loss = self.evaluate(val_dataset)
                    print(f"  ── val_loss: {val_loss:.4f} ──")

                # 체크포인트
                if step % cfg.save_interval == 0 and step > 0:
                    self.save(step)

                step += 1

        self.save(step, final=True)
        print("\n✅ 학습 완료!")

    @torch.no_grad()
    def evaluate(self, dataset) -> float:
        self.model.eval()
        loader = DataLoader(dataset, batch_size=self.cfg.batch_size)
        total, count = 0.0, 0
        for batch in loader:
            with torch.autocast(device_type=self.cfg.device.split(":")[0], dtype=self.dtype):
                loss = self._compute_loss(batch)
            total += loss.item()
            count += 1
        self.model.train()
        return total / max(count, 1)

    def save(self, step: int, final: bool = False):
        tag  = "final" if final else f"step_{step}"
        path = os.path.join(self.cfg.out_dir, f"ckpt_{tag}.pt")
        torch.save({
            "step":        step,
            "model":       self.model.state_dict(),
            "optimizer":   self.optimizer.state_dict(),
            "config":      self.cfg,
        }, path)
        print(f"  💾 저장: {path}")

    @classmethod
    def load_checkpoint(cls, path: str, model, tokenizer, cfg: TrainConfig):
        ckpt = torch.load(path, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        trainer = cls(model, tokenizer, cfg)
        trainer.optimizer.load_state_dict(ckpt["optimizer"])
        print(f"✓ 체크포인트 로드: step {ckpt['step']}")
        return trainer, ckpt["step"]
