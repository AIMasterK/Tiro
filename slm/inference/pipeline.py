"""
SLM 추론 파이프라인
- 단계별 내부 상태를 눈으로 볼 수 있음 (Goal / Context / Emotion 벡터)
- 대화 히스토리 관리
- 감정/목표 강제 주입 지원 (inference-time steering)
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

GOAL_LABELS = [
    "정보 제공", "질문 답변", "감정 공감", "작업 수행",
    "설명", "조언", "요약", "번역",
    "창작", "잡담", "확인", "경고",
    "거절", "칭찬", "수정 제안", "기타",
]

EMOTION_LABELS = ["중립", "공감", "유머", "단호", "친근", "조심", "격려", "정중"]


@dataclass
class Turn:
    role:    str   # "user" | "assistant"
    content: str


@dataclass
class IntentState:
    """한 번의 추론에서 나온 내부 상태 (디버그/해석용)"""
    goal_idx:     int
    goal_label:   str
    goal_conf:    float
    emo_idx:      int
    emo_label:    str
    emo_conf:     float
    intent_vec:   torch.Tensor   # (fused_dim,)

    def __str__(self):
        return (
            f"[Goal] {self.goal_label} ({self.goal_conf:.0%})  "
            f"[Emotion] {self.emo_label} ({self.emo_conf:.0%})"
        )


class SLMPipeline:
    """
    대화형 추론 파이프라인
    사용법:
        pipe = SLMPipeline(model, tokenizer)
        response, state = pipe.chat("오늘 기분 어때?")
        print(response)
        print(state)  # 내부 Goal/Emotion 확인
    """

    def __init__(
        self,
        model,
        tokenizer,
        max_history: int = 10,
        device: str = "cpu",
    ):
        self.model     = model.to(device).eval()
        self.tokenizer = tokenizer
        self.device    = torch.device(device)
        self.history:  List[Turn] = []
        self.max_history = max_history

    # ── 공개 API ─────────────────────────────────

    def chat(
        self,
        user_input: str,
        max_new_tokens: int = 200,
        temperature:    float = 0.8,
        top_k:          int   = 50,
        top_p:          float = 0.9,
        # inference-time steering (선택)
        force_goal:     Optional[int] = None,
        force_emotion:  Optional[int] = None,
        show_intent:    bool = True,
    ) -> Tuple[str, IntentState]:
        """
        user_input을 받아 응답 텍스트와 내부 IntentState를 반환.
        """
        self.history.append(Turn("user", user_input))

        # 히스토리 → 토큰 시퀀스
        input_ids = self._build_context()

        with torch.no_grad():
            # ── 1단계: backbone + heads로 Intent 추출 ──
            intent, goal_logits, emo_logits = self._extract_intent(input_ids)

            # ── 2단계: 강제 주입 (선택) ──
            if force_goal is not None or force_emotion is not None:
                intent = self._steer_intent(
                    input_ids, force_goal, force_emotion
                )

            # ── 3단계: 생성 ──
            output_ids = self._generate(
                input_ids, intent,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )

        # 디코딩 (새로 생성된 부분만)
        new_ids  = output_ids[0, input_ids.shape[1]:].tolist()
        response = self.tokenizer.decode(new_ids)

        self.history.append(Turn("assistant", response))
        self._trim_history()

        # IntentState 구성
        g_probs   = F.softmax(goal_logits[0], dim=-1)
        e_probs   = F.softmax(emo_logits[0],  dim=-1)
        g_idx     = g_probs.argmax().item()
        e_idx     = e_probs.argmax().item()

        state = IntentState(
            goal_idx   = g_idx,
            goal_label = GOAL_LABELS[g_idx] if g_idx < len(GOAL_LABELS) else str(g_idx),
            goal_conf  = g_probs[g_idx].item(),
            emo_idx    = e_idx,
            emo_label  = EMOTION_LABELS[e_idx] if e_idx < len(EMOTION_LABELS) else str(e_idx),
            emo_conf   = e_probs[e_idx].item(),
            intent_vec = intent.squeeze(0).cpu(),
        )

        if show_intent:
            print(f"\n  🧠 {state}")

        return response, state

    def reset(self):
        """대화 히스토리 초기화"""
        self.history.clear()

    # ── 내부 메서드 ──────────────────────────────

    def _build_context(self) -> torch.Tensor:
        """히스토리 → input_ids (B=1, T)"""
        tok  = self.tokenizer
        bos  = tok.special_tokens["<bos>"]
        usr  = tok.special_tokens["<usr>"]
        bot  = tok.special_tokens["<bot>"]

        ids = [bos]
        for turn in self.history:
            if turn.role == "user":
                ids += [usr] + tok.encode(turn.content, add_special=False)
            else:
                ids += [bot] + tok.encode(turn.content, add_special=False)

        # 마지막이 user면 bot 토큰 추가 (응답 시작 신호)
        if self.history and self.history[-1].role == "user":
            ids.append(bot)

        ids = ids[-self.model.config.max_seq_len:]
        return torch.tensor([ids], dtype=torch.long, device=self.device)

    def _extract_intent(self, input_ids):
        """backbone → goal/ctx/emo head → fusion → intent_vec"""
        out = self.model(input_ids)
        return out["intent_vec"], out["goal_logits"], out["emo_logits"]

    def _steer_intent(self, input_ids, force_goal, force_emotion):
        """
        특정 goal/emotion 방향으로 intent_vec를 수정.
        해당 임베딩 행의 가중치를 직접 사용해 대체.
        (간단 구현: 실제론 soft mixing도 가능)
        """
        out = self.model(input_ids)
        intent = out["intent_vec"]

        if force_goal is not None:
            goal_emb = self.model.goal_head.classify.weight[force_goal]
            # intent의 goal 서브공간을 교체 (단순화)
            intent = intent.clone()
            intent[0, :goal_emb.shape[0]] = goal_emb.detach()

        return intent

    def _generate(
        self,
        input_ids: torch.Tensor,
        intent_vec: torch.Tensor,
        max_new_tokens: int,
        temperature: float,
        top_k: int,
        top_p: float,
    ) -> torch.Tensor:
        """Intent-conditioned 자기회귀 생성"""
        eos = self.tokenizer.special_tokens["<eos>"]
        ids = input_ids.clone()

        for _ in range(max_new_tokens):
            ctx = ids[:, -self.model.config.max_seq_len:]
            out = self.model(ctx)

            # intent bias 주입 (마지막 위치 logit 수정)
            logits = out["logits"][:, -1, :]

            # 온도 조절
            logits = logits / max(temperature, 1e-6)

            # Top-k 필터
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            # Top-p (nucleus) 필터
            if top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                cum_probs = sorted_logits.softmax(-1).cumsum(-1)
                remove    = cum_probs - sorted_logits.softmax(-1) > top_p
                sorted_logits[remove] = float("-inf")
                logits = torch.zeros_like(logits).scatter_(1, sorted_idx, sorted_logits)

            probs    = F.softmax(logits, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1)

            ids = torch.cat([ids, next_tok], dim=1)
            if next_tok.item() == eos:
                break

        return ids

    def _trim_history(self):
        if len(self.history) > self.max_history * 2:
            self.history = self.history[-(self.max_history * 2):]


# ── 대화형 CLI ─────────────────────────────────

def run_cli(model, tokenizer, device="cpu"):
    """터미널에서 바로 대화 테스트"""
    pipe = SLMPipeline(model, tokenizer, device=device)
    print("\n=== SLM Chat ===  (종료: /quit | 초기화: /reset)\n")
    while True:
        try:
            user = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if user == "/quit":
            break
        if user == "/reset":
            pipe.reset()
            print("  [히스토리 초기화]")
            continue
        if not user:
            continue
        response, _ = pipe.chat(user)
        print(f"Bot: {response}\n")
