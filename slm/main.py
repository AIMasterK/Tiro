"""
SLM 메인 진입점
사용법:
    python main.py check          # 모델 구조 확인 + 더미 forward pass
    python main.py train          # 학습 (data/train.jsonl 필요)
    python main.py chat           # 터미널 대화 (체크포인트 필요)
    python main.py chat --ckpt checkpoints/ckpt_final.pt
"""

import sys
import torch

# 경로 설정
sys.path.insert(0, ".")

from model.architecture import SLM, SLMConfig
from model.tokenizer    import BPETokenizer
from train.trainer      import Trainer, TrainConfig
from inference.pipeline import SLMPipeline, run_cli


def cmd_check():
    """모델 구조 확인 + 더미 forward pass"""
    print("=" * 50)
    print("  SLM 구조 확인")
    print("=" * 50)

    cfg   = SLMConfig()
    model = SLM(cfg)
    print(f"\n  {model}")
    print(f"  파라미터 수: {model.num_params()/1e6:.2f}M\n")

    # 레이어별 파라미터
    print("  [레이어별 파라미터]")
    for name, mod in model.named_children():
        n = sum(p.numel() for p in mod.parameters())
        print(f"    {name:20s}  {n/1e6:.3f}M")

    # 더미 forward
    print("\n  [더미 Forward Pass]")
    B, T = 2, 64
    dummy_ids = torch.randint(0, cfg.vocab_size, (B, T))
    dummy_lbl = dummy_ids.clone()
    dummy_lbl[:, :10] = -100   # 앞부분은 loss 제외

    with torch.no_grad():
        out = model(dummy_ids, labels=dummy_lbl)

    print(f"    logits:       {out['logits'].shape}")
    print(f"    goal_logits:  {out['goal_logits'].shape}")
    print(f"    emo_logits:   {out['emo_logits'].shape}")
    print(f"    intent_vec:   {out['intent_vec'].shape}")
    print(f"    lm_loss:      {out['lm_loss'].item():.4f}")

    # 생성 테스트
    print("\n  [생성 테스트 (랜덤 가중치)]")
    prompt = torch.randint(0, cfg.vocab_size, (1, 10))
    generated = model.generate(prompt, max_new_tokens=20, temperature=1.0)
    print(f"    입력 길이: {prompt.shape[1]} → 출력 길이: {generated.shape[1]}")
    print("\n✅ 구조 확인 완료!\n")


def cmd_train():
    import os
    from model.tokenizer import BPETokenizer
    from train.trainer   import TrainConfig, Trainer, ConversationDataset

    # 토크나이저 준비 (없으면 더미로 생성)
    tok_path = "checkpoints/tokenizer"
    if not os.path.exists(os.path.join(tok_path, "tokenizer.json")):
        print("토크나이저가 없어서 더미 데이터로 학습합니다...")
        tok = BPETokenizer(vocab_size=8000)
        dummy_texts = ["안녕하세요", "오늘 날씨가 좋네요", "파이썬 언어 모델", "hello world"]
        tok.train(dummy_texts, verbose=False)
        tok.save(tok_path)
    else:
        tok = BPETokenizer.load(tok_path)

    model = SLM(SLMConfig())
    cfg   = TrainConfig()

    if not os.path.exists(cfg.data_path):
        print(f"⚠️  학습 데이터가 없습니다: {cfg.data_path}")
        print("   data/train.jsonl 포맷: {\"input\": \"...\", \"output\": \"...\", \"goal\": 0, \"emotion\": 2}")
        return

    dataset = ConversationDataset(cfg.data_path, tok, max_len=cfg.max_seq_len)
    trainer = Trainer(model, tok, cfg)
    trainer.train(dataset)


def cmd_chat(ckpt_path: str = None):
    from model.tokenizer import BPETokenizer

    tok_path = "checkpoints/tokenizer"
    try:
        tok = BPETokenizer.load(tok_path)
    except FileNotFoundError:
        print("토크나이저가 없습니다. 먼저 python main.py train을 실행하세요.")
        return

    model = SLM(SLMConfig())
    if ckpt_path:
        ckpt = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        print(f"체크포인트 로드: {ckpt_path}")
    else:
        print("⚠️  체크포인트 없이 랜덤 가중치로 실행합니다 (의미없는 출력 나옴)")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_cli(model, tok, device=device)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "check"
    ckpt = None
    if "--ckpt" in sys.argv:
        ckpt = sys.argv[sys.argv.index("--ckpt") + 1]

    if cmd == "check":
        cmd_check()
    elif cmd == "train":
        cmd_train()
    elif cmd == "chat":
        cmd_chat(ckpt)
    else:
        print(f"알 수 없는 명령: {cmd}")
        print("사용법: python main.py [check|train|chat] [--ckpt path]")
