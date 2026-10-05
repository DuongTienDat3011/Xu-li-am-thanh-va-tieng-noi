"""
lab2_part_lm.py
===============
CSE457 Lab 2 – Mục 3.4: Mô hình ngôn ngữ (Language Modeling)
Sinh viên: Dương Tiến Đạt – MSSV: 2251172268

Nội dung theo Chương 3:
  3.4 Language Modeling: N-gram, MLE, Perplexity, Smoothing
    - Unigram (N=1) và Bigram (N=2)
    - Maximum Likelihood Estimation (MLE)
    - Add-1 (Laplace) smoothing
    - Backoff smoothing
    - Perplexity đánh giá chất lượng mô hình

Tài liệu:
  - Huang et al., Ch.11: Language Modeling
  - Jurafsky & Martin, Ch.3: N-gram Language Models
"""

import os, math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict

from lab2_utils import LABELS, LABEL_VI, FIGURES_DIR, savefig, style_ax

# ════════════════════════════════════════════════════════════
# Corpus tiếng Việt đơn giản (câu lệnh giả định cho hệ thống)
# Từ vựng: khong, mot, hai, ba, bon (và các từ mở rộng)
# Câu lệnh điều khiển đơn giản
# ════════════════════════════════════════════════════════════

CORPUS_SENTENCES = [
    # Mỗi câu là 1 list các token (đã chuyển về ASCII)
    ["<s>", "mot",   "hai",   "ba",    "</s>"],
    ["<s>", "khong", "mot",   "</s>"],
    ["<s>", "hai",   "ba",    "bon",   "</s>"],
    ["<s>", "mot",   "bon",   "ba",    "</s>"],
    ["<s>", "khong", "hai",   "</s>"],
    ["<s>", "ba",    "bon",   "</s>"],
    ["<s>", "mot",   "hai",   "khong", "</s>"],
    ["<s>", "hai",   "mot",   "ba",    "</s>"],
    ["<s>", "khong", "khong", "mot",   "</s>"],
    ["<s>", "bon",   "ba",    "hai",   "</s>"],
    ["<s>", "mot",   "mot",   "hai",   "</s>"],
    ["<s>", "ba",    "hai",   "mot",   "</s>"],
    ["<s>", "khong", "ba",    "bon",   "</s>"],
    ["<s>", "hai",   "khong", "</s>"],
    ["<s>", "bon",   "mot",   "hai",   "ba",  "</s>"],
    ["<s>", "mot",   "ba",    "khong", "</s>"],
    ["<s>", "hai",   "hai",   "ba",    "</s>"],
    ["<s>", "khong", "bon",   "</s>"],
    ["<s>", "ba",    "mot",   "hai",   "bon", "</s>"],
    ["<s>", "bon",   "hai",   "khong", "ba",  "</s>"],
]

TEST_SENTENCES = [
    ["<s>", "mot",   "hai",   "</s>"],
    ["<s>", "khong", "ba",    "</s>"],
    ["<s>", "bon",   "mot",   "</s>"],
    ["<s>", "hai",   "khong", "ba",   "</s>"],
    ["<s>", "ba",    "hai",   "</s>"],
]

VOCAB = sorted(set(
    w for sent in CORPUS_SENTENCES for w in sent
))


# ════════════════════════════════════════════════════════════
# N-GRAM LANGUAGE MODEL
# ════════════════════════════════════════════════════════════

class NgramLM:
    """
    N-gram Language Model với các phương pháp smoothing.

    P_MLE(wₙ|w₁..wₙ₋₁) = C(w₁..wₙ) / C(w₁..wₙ₋₁)

    Smoothing methods:
      - MLE: Maximum Likelihood Estimation (không smooth)
      - Laplace (Add-1): P = (C+1) / (C_ctx + |V|)
      - Backoff: dùng (N-1)-gram khi N-gram count = 0
    """

    def __init__(self, n: int = 2, smoothing: str = "laplace"):
        """
        n         : bậc N-gram (1=unigram, 2=bigram)
        smoothing : 'mle' | 'laplace' | 'backoff'
        """
        self.n          = n
        self.smoothing  = smoothing
        self.ngram_counts = defaultdict(int)   # C(w₁..wₙ)
        self.context_counts = defaultdict(int) # C(w₁..wₙ₋₁)
        self.vocab      = set()
        self.total_words = 0

    def _get_ngrams(self, sentence: list) -> list:
        """Trích xuất tất cả N-gram từ 1 câu."""
        return [tuple(sentence[i:i+self.n])
                for i in range(len(sentence) - self.n + 1)]

    def fit(self, corpus: list) -> None:
        """
        Ước lượng xác suất N-gram từ corpus (MLE).
        corpus: list of list of str (token)
        """
        for sent in corpus:
            # Đếm N-gram
            for ngram in self._get_ngrams(sent):
                self.ngram_counts[ngram] += 1
                ctx = ngram[:-1]                  # context = N-1 đầu
                self.context_counts[ctx] += 1
                self.vocab.update(ngram)
                self.total_words += 1

        # Cũng đếm unigram riêng nếu N>1 (để backoff)
        if self.n > 1:
            for sent in corpus:
                for w in sent:
                    self.vocab.add(w)

        self.V = len(self.vocab)

    def log_prob(self, ngram: tuple) -> float:
        """
        Tính log P(wₙ | w₁..wₙ₋₁) theo smoothing method.

        MLE    : log(C(ngram) / C(context))
        Laplace: log((C+1) / (C_ctx + |V|))
        Backoff: dùng (N-1)-gram nếu ngram chưa thấy
        """
        ctx   = ngram[:-1]
        word  = ngram[-1]
        c_ngram = self.ngram_counts.get(ngram, 0)
        c_ctx   = self.context_counts.get(ctx, 0)

        if self.smoothing == "mle":
            if c_ngram == 0 or c_ctx == 0:
                return -math.inf
            return math.log(c_ngram / c_ctx)

        elif self.smoothing == "laplace":
            # P_Laplace = (C + 1) / (C_ctx + |V|)
            return math.log((c_ngram + 1) / (c_ctx + self.V + 1e-10))

        elif self.smoothing == "backoff":
            if c_ngram > 0:
                # Có đủ count → dùng MLE
                return math.log(c_ngram / (c_ctx + 1e-10))
            elif self.n > 1:
                # Backoff sang (N-1)-gram với discount α=0.4
                lower_ngram = ngram[1:]  # bỏ token đầu
                lower_ctx   = lower_ngram[:-1]
                c_lo = self.ngram_counts.get(lower_ngram, 0)
                c_lo_ctx = self.context_counts.get(lower_ctx, 0)
                if c_lo > 0 and c_lo_ctx > 0:
                    return math.log(0.4 * c_lo / c_lo_ctx)
                else:
                    # Fallback unigram với Add-1
                    ug = self.ngram_counts.get((word,), 0)
                    return math.log((ug + 1) / (self.total_words + self.V))
            else:
                # Unigram với Laplace
                ug = self.ngram_counts.get(ngram, 0)
                return math.log((ug + 1) / (self.total_words + self.V))

        else:
            raise ValueError(f"Smoothing '{self.smoothing}' không hợp lệ")

    def sentence_log_prob(self, sentence: list) -> float:
        """
        Log P(sentence) = Σ log P(wₙ | w₁..wₙ₋₁)
        """
        lp = 0.0
        for ngram in self._get_ngrams(sentence):
            lp += self.log_prob(ngram)
        return lp

    def perplexity(self, test_corpus: list) -> float:
        """
        Perplexity = exp(-1/N · Σ log P(sentence))

        Đo độ "bất ngờ" trung bình của mô hình với dữ liệu test.
        PP thấp → mô hình dự đoán tốt.
        PP(random) = |V|; PP tốt << |V|.

        PP(W) = P(w₁w₂..wₙ)^(-1/N)
              = 2^(-1/N · log₂P(W))
        """
        total_ll  = 0.0
        total_cnt = 0

        for sent in test_corpus:
            ll = self.sentence_log_prob(sent)
            if ll == -math.inf:
                ll = -100.0  # phạt cứng nếu MLE gặp OOV
            total_ll  += ll
            total_cnt += max(len(sent) - self.n + 1, 1)

        avg_ll = total_ll / total_cnt
        return math.exp(-avg_ll)

    def next_word_probs(self, context: tuple) -> dict:
        """
        Dự đoán xác suất cho tất cả từ tiếp theo.
        Dùng trong demo Language Model.
        """
        return {
            w: math.exp(self.log_prob(context + (w,)))
            for w in sorted(self.vocab)
            if w not in ("<s>",)
        }


# ════════════════════════════════════════════════════════════
# DEMO & VẼ HÌNH
# ════════════════════════════════════════════════════════════

def _print_ngram_table(model: NgramLM, context: tuple, top_k: int = 6) -> None:
    """In bảng xác suất N-gram cho 1 context."""
    probs = model.next_word_probs(context)
    # Sắp xếp giảm dần
    probs_sorted = sorted(probs.items(), key=lambda kv: -kv[1])
    ctx_str = " ".join(context) if context else "(start)"
    print(f"\n    P(w | {ctx_str}) [{model.smoothing}]:")
    print(f"    {'Từ':<10} {'Xác suất':>10}")
    print("    " + "─"*22)
    for w, p in probs_sorted[:top_k]:
        bar = "█" * int(p * 30)
        print(f"    {LABEL_VI.get(w,w):<10} {p:>10.4f}  {bar}")


def plot_lm_comparison(models_by_smooth: dict,
                       test_corpus: list) -> None:
    """
    Vẽ:
      (a) Heatmap xác suất bigram P(w_j | w_i) cho 3 smoothing methods
      (b) Perplexity trên test set cho các phương pháp
    """
    words   = LABELS   # khong, mot, hai, ba, bon
    vi_words = [LABEL_VI[w] for w in words]

    fig, axes = plt.subplots(2, len(models_by_smooth),
                              figsize=(6 * len(models_by_smooth), 11))
    if len(models_by_smooth) == 1:
        axes = axes.reshape(2, 1)

    perplexities = {}

    for col, (tag, model) in enumerate(models_by_smooth.items()):
        # ── Heatmap P(word_j | word_i) ──
        P = np.zeros((len(words), len(words)))
        for i, wi in enumerate(words):
            row_probs = model.next_word_probs(("<s>",))
            row_probs.update(model.next_word_probs((wi,)))
            for j, wj in enumerate(words):
                p = math.exp(model.log_prob((wi, wj)))
                P[i, j] = max(p, 0)

        im = axes[0, col].imshow(P, cmap="Blues", vmin=0, aspect="auto")
        plt.colorbar(im, ax=axes[0, col], fraction=0.046)
        for i in range(len(words)):
            for j in range(len(words)):
                axes[0, col].text(j, i, f"{P[i,j]:.2f}",
                                  ha="center", va="center", fontsize=8)
        axes[0, col].set_xticks(range(len(words)))
        axes[0, col].set_xticklabels(vi_words, fontsize=8)
        axes[0, col].set_yticks(range(len(words)))
        axes[0, col].set_yticklabels(vi_words, fontsize=8)
        axes[0, col].set_title(f"Bigram P(w_j | w_i)\n[{tag}]",
                                fontsize=9, fontweight="bold")
        axes[0, col].set_xlabel("w_j (tiếp theo)")
        axes[0, col].set_ylabel("w_i (hiện tại)")

        # ── Perplexity ──
        pp = model.perplexity(test_corpus)
        perplexities[tag] = pp
        axes[1, col].bar([tag], [pp], color=["steelblue","darkorange","seagreen"][col % 3],
                          edgecolor="black", width=0.4)
        axes[1, col].text(0, pp + 0.5, f"{pp:.2f}", ha="center", fontsize=10)
        axes[1, col].set_ylabel("Perplexity")
        axes[1, col].set_title(f"PP = {pp:.2f}", fontsize=10, fontweight="bold")
        axes[1, col].grid(True, axis="y", alpha=0.3)
        axes[1, col].set_ylim([0, max(perplexities.values(), default=1) * 1.3 + 2])

    fig.suptitle("3.4 Mô hình ngôn ngữ – Bigram heatmap & Perplexity\n"
                 "MLE: zero-probability OOV | Laplace: smooth đều | Backoff: hạ bậc",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    savefig(fig, "L1_language_model.png")

    print(f"\n  Perplexity trên test set:")
    for tag, pp in perplexities.items():
        print(f"    {tag:<10}: PP = {pp:.4f}  (thấp hơn = dự đoán tốt hơn)")
    print(f"  Vocab size |V| = {len(VOCAB)}  (PP random ≈ {len(VOCAB):.0f})")


def run_part_lm() -> dict:
    """Chạy toàn bộ mục 3.4 Language Modeling."""
    print("\n" + "═"*60)
    print("  MỤC 3.4 – MÔ HÌNH NGÔN NGỮ (N-gram LM)")
    print("═"*60)

    # Huấn luyện 3 phương pháp
    models = {}
    for smooth in ["mle", "laplace", "backoff"]:
        lm = NgramLM(n=2, smoothing=smooth)
        lm.fit(CORPUS_SENTENCES)
        models[smooth] = lm

    also_unigram = NgramLM(n=1, smoothing="laplace")
    also_unigram.fit(CORPUS_SENTENCES)

    print(f"\n  Corpus: {len(CORPUS_SENTENCES)} câu | Test: {len(TEST_SENTENCES)} câu")
    print(f"  Vocab: {len(VOCAB)} tokens: {sorted(VOCAB)}")

    # In bảng ví dụ
    for tag, model in models.items():
        _print_ngram_table(model, ("mot",))

    # Tính Perplexity
    print("\n  Perplexity (Bigram):")
    for tag, model in models.items():
        pp = model.perplexity(TEST_SENTENCES)
        print(f"    {tag:<10}: {pp:.4f}")

    pp_ug = also_unigram.perplexity(TEST_SENTENCES)
    print(f"    {'unigram':<10}: {pp_ug:.4f}")

    # Vẽ hình
    plot_lm_comparison(models, TEST_SENTENCES)

    # Demo: xác suất câu test
    print("\n  Demo: log P(câu | Laplace Bigram):")
    for sent in TEST_SENTENCES[:3]:
        lp = models["laplace"].sentence_log_prob(sent)
        pp = math.exp(-lp / max(len(sent)-1, 1))
        sent_str = " ".join(LABEL_VI.get(w, w) for w in sent)
        print(f"    {sent_str:<35} logP={lp:.3f}  PP={pp:.3f}")

    print("\n  Nhận xét:")
    print("    - MLE: xác suất = 0 nếu bigram chưa thấy → PP = ∞ → không dùng được")
    print("    - Laplace: thêm 1 vào mọi count → tránh zero-prob nhưng redistribute nhiều")
    print("    - Backoff: dùng bigram khi có, hạ về unigram khi không → cân bằng tốt hơn")
    print("    - Perplexity PP = exp(-1/N Σ log P) đo độ 'bất ngờ' trung bình")
    print("    - PP thấp = mô hình dự đoán tốt; PP(random) ≈ |V|")
    print("\n  ✓ Mục 3.4 hoàn thành.")

    return {"lm_models": models, "lm_unigram": also_unigram}


if __name__ == "__main__":
    run_part_lm()
