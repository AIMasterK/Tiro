"""
경량 BPE 토크나이저
학습 데이터가 없을 땐 sentencepiece를 래핑해서 쓰는 버전도 지원
"""

import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import List, Optional


SPECIAL_TOKENS = {
    "<pad>": 0,
    "<unk>": 1,
    "<bos>": 2,
    "<eos>": 3,
    "<sep>": 4,   # 발화 구분
    "<usr>": 5,   # 사용자 턴
    "<bot>": 6,   # 봇 턴
}


class BPETokenizer:
    """
    간단한 BPE 토크나이저.
    - 소규모 데이터에서 직접 학습 가능
    - JSON으로 저장/로드
    """

    def __init__(self, vocab_size: int = 8000):
        self.vocab_size = vocab_size
        self.vocab: dict[str, int] = {}
        self.inv_vocab: dict[int, str] = {}
        self.merges: list[tuple[str, str]] = []
        self.special_tokens = SPECIAL_TOKENS.copy()

    # ── 학습 ────────────────────────────────────

    def train(self, texts: List[str], verbose: bool = True):
        """텍스트 리스트로 BPE 학습"""
        # 1. 초기 vocab: 특수 토큰 + 캐릭터
        vocab = dict(self.special_tokens)
        char_set = set()
        for text in texts:
            char_set.update(text)
        for ch in sorted(char_set):
            if ch not in vocab:
                vocab[ch] = len(vocab)

        # 2. 단어 → 캐릭터 시퀀스 (word-level BPE)
        word_freq = Counter()
        for text in texts:
            for word in self._pre_tokenize(text):
                word_freq[word] += 1

        # BPE 형식: 각 word를 캐릭터 + </w> 로 분해
        corpus = {}
        for word, freq in word_freq.items():
            chars = list(word) + ["</w>"]
            corpus[tuple(chars)] = freq

        # 3. 반복 merge
        n_merges = self.vocab_size - len(vocab)
        for step in range(n_merges):
            pairs = self._get_pairs(corpus)
            if not pairs:
                break
            best = max(pairs, key=pairs.get)
            corpus = self._merge_vocab(best, corpus)
            merged = "".join(best)
            if merged not in vocab:
                vocab[merged] = len(vocab)
            self.merges.append(best)
            if verbose and step % 500 == 0:
                print(f"  merge {step}/{n_merges}: {best} → '{merged}'")

        self.vocab = vocab
        self.inv_vocab = {v: k for k, v in vocab.items()}
        print(f"✓ 학습 완료 | vocab size: {len(self.vocab)}")

    def _pre_tokenize(self, text: str) -> List[str]:
        """공백/구두점 기준 사전 분리"""
        return re.findall(r"\w+|[^\w\s]", text.lower())

    def _get_pairs(self, corpus):
        pairs = Counter()
        for seq, freq in corpus.items():
            for i in range(len(seq) - 1):
                pairs[(seq[i], seq[i+1])] += freq
        return pairs

    def _merge_vocab(self, pair, corpus):
        new_corpus = {}
        bigram = " ".join(pair)
        replacement = "".join(pair)
        pattern = re.compile(r"(?<!\S)" + re.escape(bigram) + r"(?!\S)")
        for seq, freq in corpus.items():
            new_seq = pattern.sub(replacement, " ".join(seq)).split()
            new_corpus[tuple(new_seq)] = freq
        return new_corpus

    # ── 인코딩 / 디코딩 ─────────────────────────

    def encode(self, text: str, add_special: bool = True) -> List[int]:
        tokens = []
        if add_special:
            tokens.append(self.special_tokens["<bos>"])

        for word in self._pre_tokenize(text):
            word_tokens = self._bpe(word)
            for tok in word_tokens:
                tokens.append(self.vocab.get(tok, self.special_tokens["<unk>"]))

        if add_special:
            tokens.append(self.special_tokens["<eos>"])
        return tokens

    def decode(self, ids: List[int], skip_special: bool = True) -> str:
        special_ids = set(self.special_tokens.values())
        tokens = []
        for i in ids:
            if skip_special and i in special_ids:
                continue
            tokens.append(self.inv_vocab.get(i, "<unk>"))
        text = "".join(tokens).replace("</w>", " ").strip()
        return text

    def _bpe(self, word: str) -> List[str]:
        """단어에 학습된 merge 적용"""
        chars = list(word) + ["</w>"]
        for pair in self.merges:
            i = 0
            new_chars = []
            while i < len(chars):
                if i < len(chars) - 1 and (chars[i], chars[i+1]) == pair:
                    new_chars.append("".join(pair))
                    i += 2
                else:
                    new_chars.append(chars[i])
                    i += 1
            chars = new_chars
            if len(chars) == 1:
                break
        return chars

    # ── 저장 / 로드 ─────────────────────────────

    def save(self, path: str):
        Path(path).mkdir(parents=True, exist_ok=True)
        data = {
            "vocab_size": self.vocab_size,
            "vocab": self.vocab,
            "merges": self.merges,
            "special_tokens": self.special_tokens,
        }
        with open(os.path.join(path, "tokenizer.json"), "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"✓ 저장 완료: {path}/tokenizer.json")

    @classmethod
    def load(cls, path: str) -> "BPETokenizer":
        with open(os.path.join(path, "tokenizer.json"), encoding="utf-8") as f:
            data = json.load(f)
        tok = cls(vocab_size=data["vocab_size"])
        tok.vocab = data["vocab"]
        tok.inv_vocab = {int(v): k for k, v in tok.vocab.items()}
        tok.merges = [tuple(m) for m in data["merges"]]
        tok.special_tokens = data["special_tokens"]
        return tok

    def __len__(self):
        return len(self.vocab)


# ── 빠른 테스트 ────────────────────────────────

if __name__ == "__main__":
    sample = [
        "안녕하세요! 오늘 날씨가 좋네요.",
        "파이썬으로 언어 모델을 만들고 있어요.",
        "자연어 처리는 재미있는 분야입니다.",
        "hello world, this is a test sentence.",
        "language models are fascinating to build.",
    ]
    tok = BPETokenizer(vocab_size=200)
    tok.train(sample, verbose=False)

    test = "오늘 날씨 어때요?"
    ids = tok.encode(test)
    decoded = tok.decode(ids)
    print(f"원문:   {test}")
    print(f"ids:    {ids}")
    print(f"복원:   {decoded}")
