"""
lab2_part_hmm.py
================
CSE457 Lab 2 – Bổ sung mục 3.1 + 3.3: Nhận dạng tiếng nói dùng HMM Gaussian
Sinh viên: Dương Tiến Đạt – MSSV: 2251172268

Nội dung theo Chương 3:
  3.1 HMM: Chuỗi Markov ẩn, thuật toán Forward, Viterbi decoding, Baum-Welch
  3.3 Acoustic Modeling: Whole-word Gaussian HMM (đơn vị = từ), so sánh với DTW

Tài liệu tham chiếu:
  - Huang et al., Ch.8: Hidden Markov Models
  - Rabiner & Schafer, Ch.14 Mục 14.2–14.5: Công thức ASR cơ bản
"""

import os, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from lab2_utils import (
    FS, WIN_LEN, HOP_LEN, LABELS, LABEL_VI,
    DATASET_DIR, FIGURES_DIR,
    read_wav, savefig, style_ax, get_train_test_files
)
from lab2_part_abcd import compute_mfcc, endpoint_detection

# ════════════════════════════════════════════════════════════
# 3.1 – MÔ HÌNH MARKOV ẨN (Hidden Markov Model)
#
# Định nghĩa HMM λ = (A, B, π):
#   A[i,j] : xác suất chuyển trạng thái i → j
#   B(o|j) : xác suất phát sinh quan sát o tại trạng thái j  (Gaussian)
#   π[i]   : xác suất trạng thái ban đầu
#
# Với whole-word HMM 5 trạng thái/từ (Ergodic → left-to-right):
#   Trạng thái 0→4 tương ứng với các đoạn âm tiết của từ
# ════════════════════════════════════════════════════════════

class GaussianHMM:
    """
    Whole-word HMM với phân phối Gaussian đơn (diagonal covariance).
    Mỗi trạng thái phát sinh một vector MFCC theo phân phối N(μ, Σ).

    Tham số:
        n_states  : số trạng thái (thường 3–5 cho isolated word)
        n_dim     : số chiều đặc trưng (13 với MFCC)
        label     : nhãn từ tương ứng
    """

    def __init__(self, n_states: int = 5, n_dim: int = 13, label: str = ""):
        self.n_states = n_states
        self.n_dim    = n_dim
        self.label    = label

        # ── Ma trận chuyển trạng thái A (left-to-right) ──
        # Left-to-right: chỉ cho phép ở lại (self-loop) hoặc tiến 1 bước
        self.A  = np.zeros((n_states, n_states))
        for i in range(n_states):
            if i < n_states - 1:
                self.A[i, i]   = 0.6   # self-loop
                self.A[i, i+1] = 0.4   # tiến 1 bước
            else:
                self.A[i, i] = 1.0    # trạng thái cuối: chỉ self-loop

        # ── Xác suất ban đầu π ──
        self.pi = np.zeros(n_states)
        self.pi[0] = 1.0   # luôn bắt đầu ở trạng thái 0

        # ── Tham số Gaussian: mean μ và variance σ² (diagonal) ──
        self.means = np.zeros((n_states, n_dim))
        self.vars  = np.ones( (n_states, n_dim))   # khởi tạo phương sai = 1

        self.log_norm_const = -0.5 * n_dim * np.log(2 * np.pi)

    # ── Tính log-likelihood phát sinh p(o | state j) ──
    def _log_emission(self, obs: np.ndarray) -> np.ndarray:
        """
        Tính log B(o|j) = log N(o; μ_j, Σ_j) cho tất cả trạng thái j.

        Gaussian diagonal:
          log N(o; μ, σ²) = -0.5·Σ_d [(o_d-μ_d)²/σ²_d + log(σ²_d)] - D/2·log(2π)

        obs: (D,) hoặc (T, D)
        returns: (n_states,) hoặc (T, n_states)
        """
        # obs shape: (T, D) hoặc (D,)
        single = obs.ndim == 1
        if single:
            obs = obs[np.newaxis, :]   # (1, D)

        T = obs.shape[0]
        log_b = np.zeros((T, self.n_states))

        for j in range(self.n_states):
            diff   = obs - self.means[j]           # (T, D)
            log_b[:, j] = (
                self.log_norm_const
                - 0.5 * np.sum(
                    diff ** 2 / self.vars[j]
                    + np.log(self.vars[j]), axis=1
                )
            )

        return log_b[0] if single else log_b

    # ────────────────────────────────────────────────────────
    # THUẬT TOÁN FORWARD (α)
    # Tính P(O₁..Oₜ, qₜ=j | λ) = α_t(j)
    #
    # Khởi tạo: α_1(j) = π_j · B(o₁|j)
    # Quy nạp : α_t(j) = [Σ_i α_{t-1}(i) · A[i,j]] · B(oₜ|j)
    # Kết thúc: P(O|λ) = Σ_j α_T(j)
    # ────────────────────────────────────────────────────────
    def forward(self, obs: np.ndarray) -> tuple:
        """
        Thuật toán Forward (scaled) để tránh underflow số học.

        Returns:
            log_prob : log P(O|λ)
            alpha    : ndarray (T, n_states) – scaled forward variables
            scales   : ndarray (T,) – scaling factors
        """
        T       = len(obs)
        log_b   = self._log_emission(obs)    # (T, n_states)

        alpha   = np.zeros((T, self.n_states))
        scales  = np.zeros(T)

        # ── Khởi tạo t=0 ──
        alpha[0] = self.pi * np.exp(log_b[0])
        scales[0] = alpha[0].sum() + 1e-300
        alpha[0] /= scales[0]

        # ── Quy nạp t=1..T-1 ──
        for t in range(1, T):
            # α_t(j) = Σ_i α_{t-1}(i)·A[i,j] · B(oₜ|j)
            alpha[t] = (alpha[t-1] @ self.A) * np.exp(log_b[t])
            scales[t] = alpha[t].sum() + 1e-300
            alpha[t] /= scales[t]

        log_prob = np.sum(np.log(scales + 1e-300))
        return log_prob, alpha, scales

    # ────────────────────────────────────────────────────────
    # THUẬT TOÁN VITERBI
    # Tìm chuỗi trạng thái tối ưu q* = argmax P(O, q | λ)
    #
    # δ_t(j) = max_{q1..qt-1} P(O₁..Oₜ, qₜ=j | λ)
    # ψ_t(j) = argmax_i δ_{t-1}(i)·A[i,j]
    # Backtrack từ t=T → t=1 để tìm optimal state sequence
    # ────────────────────────────────────────────────────────
    def viterbi(self, obs: np.ndarray) -> tuple:
        """
        Viterbi decoding.

        Returns:
            best_path  : list – chuỗi trạng thái tối ưu
            best_score : float – log P của đường tối ưu
            delta      : ndarray (T, n_states)
        """
        T       = len(obs)
        log_b   = self._log_emission(obs)
        log_A   = np.log(self.A + 1e-300)
        log_pi  = np.log(self.pi + 1e-300)

        delta = np.full((T, self.n_states), -np.inf)
        psi   = np.zeros((T, self.n_states), dtype=int)

        # ── Khởi tạo t=0 ──
        delta[0] = log_pi + log_b[0]

        # ── Quy nạp t=1..T-1 ──
        for t in range(1, T):
            for j in range(self.n_states):
                # max_i [δ_{t-1}(i) + log A[i,j]] + log B(oₜ|j)
                scores   = delta[t-1] + log_A[:, j]
                best_i   = int(np.argmax(scores))
                delta[t, j] = scores[best_i] + log_b[t, j]
                psi[t, j]   = best_i

        # ── Backtrack ──
        best_path = [0] * T
        best_path[T-1] = int(np.argmax(delta[T-1]))
        for t in range(T-2, -1, -1):
            best_path[t] = psi[t+1, best_path[t+1]]

        best_score = np.max(delta[T-1])
        return best_path, best_score, delta

    # ────────────────────────────────────────────────────────
    # BAUM-WELCH (EM cho HMM)
    # Ước lượng tham số λ = (A, B, π) từ dữ liệu training.
    #
    # E-step: tính γ_t(j) = P(qₜ=j|O,λ) và ξ_t(i,j) = P(qₜ=i,qₜ₊₁=j|O,λ)
    # M-step: cập nhật A, μ, σ²
    # ────────────────────────────────────────────────────────
    def _backward(self, obs: np.ndarray, scales: np.ndarray) -> np.ndarray:
        """Thuật toán Backward (scaled) – tính β_t(j)."""
        T     = len(obs)
        log_b = self._log_emission(obs)

        beta = np.zeros((T, self.n_states))
        beta[T-1] = 1.0 / (scales[T-1] + 1e-300)

        for t in range(T-2, -1, -1):
            # β_t(i) = Σ_j A[i,j] · B(o_{t+1}|j) · β_{t+1}(j)
            beta[t] = (self.A * np.exp(log_b[t+1]) * beta[t+1]).sum(axis=1)
            beta[t] /= (scales[t] + 1e-300)

        return beta

    def fit(self, obs_list: list, n_iter: int = 20, tol: float = 1e-4) -> list:
        """
        Baum-Welch EM: ước lượng tham số từ danh sách chuỗi quan sát.

        obs_list: list of ndarray (T_i, D)
        n_iter  : số vòng lặp EM tối đa
        tol     : ngưỡng hội tụ (thay đổi log-likelihood)

        Returns: list of log-likelihood theo từng vòng
        """
        # Khởi tạo μ bằng K-means đơn giản trên tất cả dữ liệu
        all_obs = np.concatenate(obs_list, axis=0)
        idx     = np.linspace(0, len(all_obs)-1, self.n_states, dtype=int)
        self.means = all_obs[idx].copy()
        self.vars  = np.full_like(self.means,
                                   np.var(all_obs, axis=0).mean() + 1e-6)

        log_likelihoods = []

        for it in range(n_iter):
            # ── Accumulators ──
            A_num   = np.zeros_like(self.A)
            A_den   = np.zeros(self.n_states)
            pi_acc  = np.zeros(self.n_states)
            mu_num  = np.zeros_like(self.means)
            var_num = np.zeros_like(self.vars)
            gamma_sum = np.zeros(self.n_states)

            total_ll = 0.0

            for obs in obs_list:
                T     = len(obs)
                log_b = self._log_emission(obs)
                log_p, alpha, scales = self.forward(obs)
                beta  = self._backward(obs, scales)
                total_ll += log_p

                # ── γ_t(j) = P(qₜ=j|O,λ) ──
                gamma = alpha * beta
                gamma_sum_t = gamma.sum(axis=1, keepdims=True)
                gamma_sum_t = np.where(gamma_sum_t == 0, 1e-300, gamma_sum_t)
                gamma = gamma / gamma_sum_t        # (T, S)

                # ── ξ_t(i,j) = P(qₜ=i, qₜ₊₁=j|O,λ) ──
                # Tổng hợp theo thời gian để cập nhật A
                for t in range(T-1):
                    xi_t = (alpha[t, :, None]
                             * self.A
                             * np.exp(log_b[t+1])
                             * beta[t+1])
                    xi_norm = xi_t.sum() + 1e-300
                    A_num  += xi_t / xi_norm
                    A_den  += gamma[t]

                # Accumulators cho μ, σ²
                pi_acc   += gamma[0]
                for j in range(self.n_states):
                    w_j = gamma[:, j]          # (T,)
                    mu_num[j]  += (w_j[:, None] * obs).sum(axis=0)
                    diff = obs - self.means[j]
                    var_num[j] += (w_j[:, None] * diff**2).sum(axis=0)
                    gamma_sum[j] += w_j.sum()

            # ── M-step: cập nhật tham số ──
            # Cập nhật A
            for i in range(self.n_states):
                denom = A_den[i] + 1e-300
                self.A[i] = A_num[i] / denom
                row_sum   = self.A[i].sum()
                self.A[i] /= (row_sum + 1e-300)

            # Cập nhật π
            self.pi = pi_acc / (pi_acc.sum() + 1e-300)

            # Cập nhật μ và σ²
            for j in range(self.n_states):
                g = gamma_sum[j] + 1e-300
                self.means[j] = mu_num[j] / g
                self.vars[j]  = np.maximum(var_num[j] / g, 1e-6)

            log_likelihoods.append(total_ll)

            # Kiểm tra hội tụ
            if (it > 0 and
                    abs(log_likelihoods[-1] - log_likelihoods[-2]) < tol):
                print(f"      Hội tụ tại vòng {it+1}: ΔLL={abs(log_likelihoods[-1]-log_likelihoods[-2]):.6f}")
                break

        return log_likelihoods

    def score(self, obs: np.ndarray) -> float:
        """Trả về log P(O|λ) – dùng để nhận dạng."""
        log_p, _, _ = self.forward(obs)
        return log_p


# ════════════════════════════════════════════════════════════
# 3.3 – ACOUSTIC MODELING: Whole-word Gaussian HMM
# Huấn luyện 1 HMM/từ, sau đó nhận dạng bằng argmax log P(O|λ_w)
# So sánh với DTW để thấy ưu/nhược điểm
# ════════════════════════════════════════════════════════════

def train_hmm_models(train_files: dict,
                     n_states: int = 5,
                     n_iter:   int = 25) -> dict:
    """
    Huấn luyện 1 HMM Gaussian/từ bằng Baum-Welch.

    Returns: dict {label: GaussianHMM}
    """
    from lab2_part_abcd import N_MFCC
    models = {}
    print("\n  Huấn luyện HMM Gaussian (Baum-Welch):")
    print(f"    n_states={n_states} | n_mfcc={N_MFCC} | n_iter={n_iter}")
    print("  " + "─"*55)

    for lab in LABELS:
        print(f"  [{lab}] ...", end="", flush=True)
        obs_list = []
        for fpath in train_files[lab]:
            x    = read_wav(fpath)
            xt, _, _ = endpoint_detection(x)
            mfcc = compute_mfcc(xt)
            obs_list.append(mfcc)

        hmm = GaussianHMM(n_states=n_states, n_dim=N_MFCC, label=lab)
        ll_hist = hmm.fit(obs_list, n_iter=n_iter)
        models[lab] = hmm

        ll_final = ll_hist[-1] if ll_hist else float("-inf")
        print(f" LL_final={ll_final:.2f}  ({len(ll_hist)} vòng)")

    return models


def recognize_hmm(x: np.ndarray, models: dict) -> tuple:
    """
    Nhận dạng 1 utterance bằng argmax log P(O|λ_w).
    Returns: (predicted_label, scores_dict)
    """
    xt, _, _ = endpoint_detection(x)
    mfcc     = compute_mfcc(xt)

    scores = {lab: models[lab].score(mfcc) for lab in LABELS}
    pred   = max(scores, key=scores.get)
    return pred, dict(sorted(scores.items(), key=lambda kv: -kv[1]))


def evaluate_hmm(models: dict, test_files: dict) -> tuple:
    """Đánh giá HMM trên tập test. Returns (y_true, y_pred, accuracy)."""
    from sklearn.metrics import accuracy_score
    y_true, y_pred = [], []
    for lab in LABELS:
        for fpath in test_files[lab]:
            x    = read_wav(fpath)
            pred, scores = recognize_hmm(x, models)
            y_true.append(lab); y_pred.append(pred)
    acc = accuracy_score(y_true, y_pred)
    return y_true, y_pred, acc


def plot_hmm_training(models: dict, train_files: dict) -> None:
    """
    Vẽ: (1) Log-likelihood hội tụ; (2) Viterbi path cho 1 file;
         (3) So sánh ma trận chuyển trạng thái A của 5 từ
    """
    # ── Plot 1: Ma trận chuyển trạng thái A của 5 models ──
    fig, axes = plt.subplots(1, len(LABELS), figsize=(14, 4))
    for ax, lab in zip(axes, LABELS):
        im = ax.imshow(models[lab].A, cmap="Blues", vmin=0, vmax=1,
                       aspect="auto")
        plt.colorbar(im, ax=ax, fraction=0.046)
        ax.set_title(f"[{LABEL_VI[lab]}]\nA matrix", fontsize=9, fontweight="bold")
        ax.set_xlabel("State j"); ax.set_ylabel("State i")
        ax.set_xticks(range(models[lab].n_states))
        ax.set_yticks(range(models[lab].n_states))
    fig.suptitle("3.1 HMM – Ma trận chuyển trạng thái A (Left-to-right)\n"
                 "Đường chéo = self-loop; đường chéo+1 = tiến 1 trạng thái",
                 fontsize=10, fontweight="bold")
    plt.tight_layout()
    savefig(fig, "H1_hmm_transition_matrix.png")

    # ── Plot 2: Viterbi path cho 1 từ ──
    fig, axes = plt.subplots(2, 1, figsize=(13, 8))
    lab   = "khong"
    fpath = train_files[lab][0]
    x     = read_wav(fpath)
    xt, _, _ = endpoint_detection(x)
    mfcc  = compute_mfcc(xt)

    best_path, best_score, delta = models[lab].viterbi(mfcc)
    t_axis = np.arange(len(mfcc)) * HOP_LEN / FS * 1000

    im = axes[0].imshow(delta.T, aspect="auto", origin="lower",
                         cmap="YlOrRd",
                         extent=[0, t_axis[-1], -0.5, models[lab].n_states-0.5])
    plt.colorbar(im, ax=axes[0], label="log δ")
    axes[0].plot(t_axis, best_path, "b.-", lw=1.2, ms=3, label="Viterbi path")
    axes[0].set_title(f"3.1 Viterbi Decoding – [{LABEL_VI[lab]}]\n"
                      f"log P* = {best_score:.2f}", fontsize=10, fontweight="bold")
    axes[0].set_ylabel("Trạng thái HMM"); axes[0].set_xlabel("Thời gian (ms)")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    # MFCC heatmap cùng trục x
    im2 = axes[1].imshow(mfcc.T, aspect="auto", origin="lower", cmap="RdYlBu_r",
                          extent=[0, t_axis[-1], 0.5, 13.5])
    plt.colorbar(im2, ax=axes[1], label="MFCC value")
    axes[1].set_title("MFCC (input feature sequence)")
    axes[1].set_ylabel("Hệ số MFCC"); axes[1].set_xlabel("Thời gian (ms)")

    fig.suptitle("3.1 + 3.3 HMM – Viterbi Decoding và MFCC Input",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    savefig(fig, "H2_hmm_viterbi_path.png")


def plot_hmm_comparison(hmm_results: tuple, dtw_results: tuple) -> None:
    """So sánh confusion matrix HMM vs DTW."""
    from sklearn.metrics import confusion_matrix, accuracy_score
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    for ax, (yt, yp, title) in zip(axes, [
        (*hmm_results[:2], f"HMM Gaussian\nAccuracy={hmm_results[2]*100:.1f}%"),
        (*dtw_results[:2],  f"DTW Nearest-Template\nAccuracy={dtw_results[2]*100:.1f}%"),
    ]):
        cm  = confusion_matrix(yt, yp, labels=LABELS)
        im  = ax.imshow(cm, cmap="Blues", vmin=0)
        plt.colorbar(im, ax=ax)
        for i in range(len(LABELS)):
            for j in range(len(LABELS)):
                clr = "white" if cm[i,j] > cm.max()*.6 else "black"
                ax.text(j, i, str(cm[i,j]), ha="center", va="center",
                        fontsize=12, fontweight="bold", color=clr)
        vil = [LABEL_VI[l] for l in LABELS]
        ax.set_xticks(range(len(LABELS))); ax.set_xticklabels(vil, fontsize=9)
        ax.set_yticks(range(len(LABELS))); ax.set_yticklabels(vil, fontsize=9)
        ax.set_xlabel("Dự đoán"); ax.set_ylabel("Nhãn thật")
        ax.set_title(title, fontsize=10, fontweight="bold")

    fig.suptitle("3.1 + 3.3 – So sánh HMM Gaussian vs DTW Nearest-Template",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    savefig(fig, "H3_hmm_vs_dtw.png")


def run_part_hmm() -> dict:
    """Chạy toàn bộ phần HMM (3.1 + 3.3)."""
    print("\n" + "═"*60)
    print("  MỤC 3.1 + 3.3 – HMM GAUSSIAN & ACOUSTIC MODELING")
    print("═"*60)

    train_files, test_files = get_train_test_files()

    # Huấn luyện
    models = train_hmm_models(train_files, n_states=5, n_iter=30)

    # Đánh giá
    y_true_hmm, y_pred_hmm, acc_hmm = evaluate_hmm(models, test_files)
    print(f"\n  HMM Accuracy = {sum(t==p for t,p in zip(y_true_hmm,y_pred_hmm))}"
          f"/{len(y_true_hmm)} = {acc_hmm*100:.1f}%")

    # DTW để so sánh
    from lab2_part_efg import build_templates, dtw_distance
    from sklearn.metrics import accuracy_score

    templates = build_templates(train_files, use_trim=True)
    y_true_dtw, y_pred_dtw = [], []
    for lab in LABELS:
        for fpath in test_files[lab]:
            x    = read_wav(fpath)
            xt, _, _ = endpoint_detection(x)
            mfcc = compute_mfcc(xt)
            sc = {l: min(dtw_distance(mfcc, R)[0] for R in templates[l])
                  for l in LABELS}
            y_true_dtw.append(lab)
            y_pred_dtw.append(min(sc, key=sc.get))
    acc_dtw = accuracy_score(y_true_dtw, y_pred_dtw)

    # Vẽ hình
    plot_hmm_training(models, train_files)
    plot_hmm_comparison(
        (y_true_hmm, y_pred_hmm, acc_hmm),
        (y_true_dtw, y_pred_dtw, acc_dtw)
    )

    print("\n  Nhận xét:")
    print("    - HMM Gaussian mô hình hóa phân phối xác suất của MFCC mỗi trạng thái")
    print("    - Viterbi path cho biết chuỗi trạng thái tối ưu theo thời gian")
    print("    - Baum-Welch (EM) tự động ước lượng A, μ, σ² từ dữ liệu training")
    print("    - DTW deterministic; HMM probabilistic → HMM generalize tốt hơn với dữ liệu lớn")
    print("    - Với dataset nhỏ (3 file/từ), cả hai phương pháp cho kết quả tương đương")
    print("\n  ✓ Mục 3.1 + 3.3 hoàn thành.")

    return {"hmm_models": models,
            "hmm_acc": acc_hmm, "dtw_acc": acc_dtw,
            "y_true_hmm": y_true_hmm, "y_pred_hmm": y_pred_hmm}


if __name__ == "__main__":
    run_part_hmm()
