"""
SLM (Small Language Model) - Multi-Head Intent Architecture
흐름: Input → Backbone → [Goal | Context | Emotion] Heads → Fusion → Decoder → Output
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ────────────────────────────────────────────────
# 기본 빌딩블록
# ────────────────────────────────────────────────

class RMSNorm(nn.Module):
    """LayerNorm보다 가벼운 정규화"""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        norm = x.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt()
        return self.weight * x / norm


class RotaryEmbedding(nn.Module):
    """RoPE: 상대적 위치 인코딩 (절대 위치보다 일반화 성능 좋음)"""
    def __init__(self, dim: int, max_seq_len: int = 512):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.max_seq_len = max_seq_len

    def forward(self, seq_len: int, device):
        t = torch.arange(seq_len, device=device).float()
        freqs = torch.outer(t, self.inv_freq)
        return torch.cat([freqs, freqs], dim=-1)  # (seq_len, dim)


def rotate_half(x):
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary(q, k, rope):
    cos = rope.cos()[None, None, :q.size(2), :]
    sin = rope.sin()[None, None, :q.size(2), :]
    q = q * cos + rotate_half(q) * sin
    k = k * cos + rotate_half(k) * sin
    return q, k


class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert dim % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.rope = RotaryEmbedding(self.head_dim)

    def forward(self, x, mask=None):
        B, T, C = x.shape
        qkv = self.qkv(x).chunk(3, dim=-1)
        q, k, v = [t.view(B, T, self.n_heads, self.head_dim).transpose(1, 2) for t in qkv]

        rope = self.rope(T, x.device)
        q, k = apply_rotary(q, k, rope)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        if mask is not None:
            attn = attn.masked_fill(mask[:, :, :T, :T] == 0, float("-inf"))
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.out(out)


class FeedForward(nn.Module):
    """SwiGLU activation (LLaMA 스타일)"""
    def __init__(self, dim: int, expand: int = 4, dropout: float = 0.1):
        super().__init__()
        hidden = int(dim * expand * 2 / 3)
        self.gate = nn.Linear(dim, hidden, bias=False)
        self.up   = nn.Linear(dim, hidden, bias=False)
        self.down = nn.Linear(hidden, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.down(F.silu(self.gate(x)) * self.up(x)))


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn  = CausalSelfAttention(dim, n_heads, dropout)
        self.norm2 = RMSNorm(dim)
        self.ff    = FeedForward(dim, dropout=dropout)

    def forward(self, x, mask=None):
        x = x + self.attn(self.norm1(x), mask)
        x = x + self.ff(self.norm2(x))
        return x


# ────────────────────────────────────────────────
# 전용 Head들
# ────────────────────────────────────────────────

class GoalHead(nn.Module):
    """
    목표 벡터 생성: "이 입력에 대해 뭘 해야 하지?"
    출력: (B, goal_dim) — 목표를 나타내는 연속 벡터
    """
    def __init__(self, dim: int, goal_dim: int, n_goals: int = 16):
        super().__init__()
        self.pool = nn.Linear(dim, goal_dim)          # 시퀀스 압축
        self.classify = nn.Linear(goal_dim, n_goals)  # 목표 분류 (보조 loss용)
        self.norm = RMSNorm(goal_dim)

    def forward(self, hidden):
        # hidden: (B, T, dim) → 평균 풀링으로 시퀀스 요약
        pooled = hidden.mean(dim=1)          # (B, dim)
        goal_vec = self.norm(self.pool(pooled))  # (B, goal_dim)
        goal_logits = self.classify(goal_vec)    # (B, n_goals) — 학습용
        return goal_vec, goal_logits


class ContextHead(nn.Module):
    """
    맥락 압축: 대화 히스토리나 현재 시퀀스의 핵심 정보 추출
    출력: (B, ctx_dim)
    """
    def __init__(self, dim: int, ctx_dim: int):
        super().__init__()
        # Attention-based pooling: 어떤 토큰이 맥락에서 중요한지 학습
        self.attn_pool = nn.Linear(dim, 1)
        self.proj = nn.Linear(dim, ctx_dim)
        self.norm = RMSNorm(ctx_dim)

    def forward(self, hidden, padding_mask=None):
        # hidden: (B, T, dim)
        scores = self.attn_pool(hidden).squeeze(-1)  # (B, T)
        if padding_mask is not None:
            scores = scores.masked_fill(padding_mask == 0, float("-inf"))
        weights = F.softmax(scores, dim=-1).unsqueeze(-1)  # (B, T, 1)
        ctx = (hidden * weights).sum(dim=1)              # (B, dim)
        return self.norm(self.proj(ctx))                 # (B, ctx_dim)


class EmotionHead(nn.Module):
    """
    감정 톤 벡터: 응답의 감정적 색채를 결정
    출력: (B, emo_dim) + (B, n_emotions) logits
    기본 감정 레이블: [중립, 공감, 유머, 단호, 친근, 조심, 격려, 정중]
    """
    N_EMOTIONS = 8

    def __init__(self, dim: int, emo_dim: int):
        super().__init__()
        self.pool = nn.Linear(dim, emo_dim)
        self.classify = nn.Linear(emo_dim, self.N_EMOTIONS)
        self.norm = RMSNorm(emo_dim)

    def forward(self, hidden):
        pooled = hidden.mean(dim=1)
        emo_vec = self.norm(F.gelu(self.pool(pooled)))   # (B, emo_dim)
        emo_logits = self.classify(emo_vec)               # (B, N_EMOTIONS)
        return emo_vec, emo_logits


# ────────────────────────────────────────────────
# Fusion Layer
# ────────────────────────────────────────────────

class IntentFusion(nn.Module):
    """
    Goal + Context + Emotion → 단일 "의도 벡터"
    이 벡터가 디코더의 초기 컨디셔닝이 됨
    """
    def __init__(self, goal_dim: int, ctx_dim: int, emo_dim: int, fused_dim: int):
        super().__init__()
        in_dim = goal_dim + ctx_dim + emo_dim
        self.fusion = nn.Sequential(
            nn.Linear(in_dim, fused_dim * 2),
            nn.SiLU(),
            nn.Linear(fused_dim * 2, fused_dim),
            RMSNorm(fused_dim),
        )

    def forward(self, goal_vec, ctx_vec, emo_vec):
        combined = torch.cat([goal_vec, ctx_vec, emo_vec], dim=-1)
        return self.fusion(combined)  # (B, fused_dim)


# ────────────────────────────────────────────────
# 메인 모델
# ────────────────────────────────────────────────

class SLMConfig:
    """모델 하이퍼파라미터"""
    vocab_size:  int = 8000
    max_seq_len: int = 512
    dim:         int = 256        # backbone hidden dim
    n_layers:    int = 6          # transformer 블록 수
    n_heads:     int = 8
    dropout:     float = 0.1

    # Head dims
    goal_dim:    int = 128
    n_goals:     int = 16         # 목표 분류 클래스 수
    ctx_dim:     int = 128
    emo_dim:     int = 64
    fused_dim:   int = 256        # 융합 후 dim (= backbone dim과 맞춤)

    def __repr__(self):
        total = self._count_params()
        return f"SLMConfig(dim={self.dim}, layers={self.n_layers}, ~{total/1e6:.1f}M params)"

    def _count_params(self):
        # 대략적인 파라미터 수 추정
        emb = self.vocab_size * self.dim
        backbone = self.n_layers * (4 * self.dim**2 + 3 * self.dim * int(self.dim * 8/3))
        heads = (self.dim * self.goal_dim + self.goal_dim * self.n_goals +
                 self.dim * self.ctx_dim + self.dim * self.emo_dim +
                 (self.goal_dim + self.ctx_dim + self.emo_dim) * self.fused_dim)
        lm_head = self.dim * self.vocab_size
        return emb + backbone + heads + lm_head


class SLM(nn.Module):
    def __init__(self, config: SLMConfig = None):
        super().__init__()
        self.config = config or SLMConfig()
        cfg = self.config

        # ── 입력 임베딩 ──
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.emb_drop  = nn.Dropout(cfg.dropout)

        # ── Shared Backbone ──
        self.blocks = nn.ModuleList([
            TransformerBlock(cfg.dim, cfg.n_heads, cfg.dropout)
            for _ in range(cfg.n_layers)
        ])
        self.norm_out = RMSNorm(cfg.dim)

        # ── Intent Heads ──
        self.goal_head    = GoalHead(cfg.dim, cfg.goal_dim, cfg.n_goals)
        self.context_head = ContextHead(cfg.dim, cfg.ctx_dim)
        self.emotion_head = EmotionHead(cfg.dim, cfg.emo_dim)

        # ── Fusion ──
        self.fusion = IntentFusion(cfg.goal_dim, cfg.ctx_dim, cfg.emo_dim, cfg.fused_dim)

        # ── Intent → 토큰 공간으로 투영 (디코더 컨디셔닝) ──
        self.intent_proj = nn.Linear(cfg.fused_dim, cfg.dim)

        # ── LM Head (출력 생성) ──
        self.lm_head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)
        # weight tying
        self.lm_head.weight = self.token_emb.weight

        # 인과적 마스크 등록
        mask = torch.tril(torch.ones(cfg.max_seq_len, cfg.max_seq_len)).view(
            1, 1, cfg.max_seq_len, cfg.max_seq_len
        )
        self.register_buffer("causal_mask", mask)

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)

    def forward(self, input_ids, padding_mask=None, labels=None):
        """
        input_ids:    (B, T)
        padding_mask: (B, T) — 1=유효 토큰, 0=패딩
        labels:       (B, T) — language modeling 타겟

        반환: dict with keys
          logits, goal_logits, emo_logits, intent_vec, loss (labels 있을 때)
        """
        B, T = input_ids.shape
        device = input_ids.device

        # ── 1. 임베딩 ──
        x = self.emb_drop(self.token_emb(input_ids))  # (B, T, dim)

        # ── 2. Backbone ──
        mask = self.causal_mask[:, :, :T, :T]
        for block in self.blocks:
            x = block(x, mask)
        hidden = self.norm_out(x)  # (B, T, dim)

        # ── 3. Intent Heads (입력 이해) ──
        goal_vec,  goal_logits = self.goal_head(hidden)
        ctx_vec                = self.context_head(hidden, padding_mask)
        emo_vec,   emo_logits  = self.emotion_head(hidden)

        # ── 4. Fusion → 의도 벡터 ──
        intent_vec = self.fusion(goal_vec, ctx_vec, emo_vec)   # (B, fused_dim)

        # ── 5. 의도를 hidden state에 주입 ──
        intent_bias = self.intent_proj(intent_vec).unsqueeze(1)  # (B, 1, dim)
        conditioned  = hidden + intent_bias                       # broadcast over T

        # ── 6. LM 출력 ──
        logits = self.lm_head(conditioned)  # (B, T, vocab_size)

        out = dict(
            logits=logits,
            goal_logits=goal_logits,
            emo_logits=emo_logits,
            intent_vec=intent_vec,
        )

        # ── 7. Loss 계산 (학습 시) ──
        if labels is not None:
            # LM loss
            lm_loss = F.cross_entropy(
                logits[:, :-1].contiguous().view(-1, self.config.vocab_size),
                labels[:, 1:].contiguous().view(-1),
                ignore_index=-100,
            )
            out["lm_loss"] = lm_loss
            out["loss"]    = lm_loss   # 추후 goal/emo loss 추가 가능

        return out

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=100, temperature=0.8, top_k=50):
        """간단한 greedy/top-k 생성"""
        self.eval()
        for _ in range(max_new_tokens):
            ctx = input_ids[:, -self.config.max_seq_len:]
            out = self(ctx)
            logits = out["logits"][:, -1, :] / temperature
            if top_k:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_tok], dim=1)
        return input_ids

    def num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self):
        return f"SLM({self.num_params()/1e6:.2f}M params)"
