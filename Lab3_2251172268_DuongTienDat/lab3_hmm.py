"""
lab3_hmm.py
===========
CSE457 Lab 3 – Mục B, C, D: HMM trái-sang-phải Gaussian
Sinh viên: Dương Tiến Đạt – MSSV: 2251172268

Nội dung:
  B. Khởi tạo HMM left-to-right 5 state, Gaussian diagonal
  C. Huấn luyện Baum-Welch (tự cài + wrapper tương thích hmmlearn API)
  D. Nhận dạng maximum likelihood + Viterbi alignment
"""

import os, pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from lab3_utils import (
    FS, LABELS, LABEL_VI, FIGURES_DIR, MODELS_DIR,
    savefig, style_ax, get_files
)
from lab3_features import extract_feat, HOP_LEN, N_FEAT


# ═══════════════════════════════════════════════════════════
# B. LEFT-TO-RIGHT GAUSSIAN HMM
# ═══════════════════════════════════════════════════════════

class GaussianHMM_LR:
    """
    Whole-word Left-to-Right Gaussian HMM.
    Topology: self-loop + forward 1 step; state cuối chỉ self-loop.

    Tham số λ = (A, B={μ,Σ}, π):
      A[i,j]  : ma trận chuyển trạng thái (left-to-right)
      μᵢ      : mean vector state i  (D chiều)
      σ²ᵢ     : diagonal variance state i
      π       : xác suất ban đầu [1,0,...,0]

    Thuật toán:
      - Forward (scaled): tính P(X|λ)
      - Backward (scaled): tính β
      - Baum-Welch EM: cập nhật A, μ, σ²
      - Viterbi: tìm chuỗi trạng thái tối ưu q*
    """

    def __init__(self, n_states=5, n_dim=39, label="",
                 self_loop_prob=0.7):
        self.n_states     = n_states
        self.n_dim        = n_dim
        self.label        = label
        self._lnc         = -0.5 * n_dim * np.log(2 * np.pi)

        # ── Xác suất ban đầu π ──────────────────────────
        # π = [1,0,0,...,0]: luôn bắt đầu từ state 0
        self.pi = np.zeros(n_states)
        self.pi[0] = 1.0

        # ── Ma trận chuyển A (left-to-right) ──────────────
        # aᵢᵢ  = self_loop_prob  (ở lại)
        # aᵢ,ᵢ₊₁= 1 - self_loop_prob  (tiến)
        # a_last,last = 1.0
        self.A = np.zeros((n_states, n_states))
        for i in range(n_states - 1):
            self.A[i, i]   = self_loop_prob
            self.A[i, i+1] = 1.0 - self_loop_prob
        self.A[-1, -1] = 1.0

        # ── Tham số Gaussian B={μ,σ²} ─────────────────────
        self.means = np.zeros((n_states, n_dim))
        self.vars  = np.ones( (n_states, n_dim))   # diagonal variance
        self.converged_ = False
        self.monitor_ll = []

    # ── Log B(o|j) ──────────────────────────────────────
    def _log_b(self, obs):
        """
        log N(o; μⱼ, Σⱼ) cho mọi j.
        Gaussian diagonal:
          log b_j(o) = const - 0.5·Σ_d [(o_d-μ_d)²/σ²_d + log σ²_d]
        obs: (T,D) → returns (T, S)
        """
        single = obs.ndim == 1
        if single: obs = obs[np.newaxis,:]
        T = obs.shape[0]
        lb = np.zeros((T, self.n_states))
        for j in range(self.n_states):
            d = obs - self.means[j]
            lb[:,j] = self._lnc - 0.5*np.sum(
                d**2/self.vars[j] + np.log(self.vars[j]), axis=1)
        return lb[0] if single else lb

    # ════════════════════════════════════════════════════
    # THUẬT TOÁN FORWARD (scaled)
    # Tính P(X|λ) = Σⱼ αT(j)
    # α₁(j) = πⱼ·Bⱼ(x₁)
    # αₜ(j) = Bⱼ(xₜ)·Σᵢ αₜ₋₁(i)·Aᵢⱼ
    # ════════════════════════════════════════════════════
    def forward(self, obs):
        T  = len(obs); lb = self._log_b(obs)
        al = np.zeros((T, self.n_states)); sc = np.zeros(T)
        al[0] = self.pi * np.exp(lb[0])
        sc[0] = al[0].sum() + 1e-300; al[0] /= sc[0]
        for t in range(1, T):
            al[t] = (al[t-1] @ self.A) * np.exp(lb[t])
            sc[t] = al[t].sum() + 1e-300; al[t] /= sc[t]
        log_p = np.sum(np.log(sc + 1e-300))
        return log_p, al, sc

    # ════════════════════════════════════════════════════
    # THUẬT TOÁN BACKWARD (scaled)
    # βT(i) = 1
    # βₜ(i) = Σⱼ Aᵢⱼ·Bⱼ(xₜ₊₁)·βₜ₊₁(j)
    # ════════════════════════════════════════════════════
    def backward(self, obs, scales):
        T  = len(obs); lb = self._log_b(obs)
        bt = np.zeros((T, self.n_states))
        bt[T-1] = 1.0 / (scales[T-1] + 1e-300)
        for t in range(T-2, -1, -1):
            bt[t] = (self.A * np.exp(lb[t+1]) * bt[t+1]).sum(axis=1)
            bt[t] /= (scales[t] + 1e-300)
        return bt

    # ════════════════════════════════════════════════════
    # BAUM-WELCH EM
    # E-step: γₜ(i) = αₜ(i)βₜ(i)/P(X|λ)
    #         ξₜ(i,j) = αₜ(i)·Aᵢⱼ·Bⱼ(xₜ₊₁)·βₜ₊₁(j)/P(X|λ)
    # M-step: cập nhật π, A, μ, σ²
    # ════════════════════════════════════════════════════
    def fit(self, obs_list, n_iter=80, tol=1e-3):
        """
        Baum-Welch với nhiều chuỗi quan sát.
        obs_list: list of ndarray (Tᵢ, D)
        """
        # Khởi tạo μ bằng k-means đơn giản (chia đều frame)
        all_obs = np.concatenate(obs_list, axis=0)
        idx     = np.linspace(0, len(all_obs)-1, self.n_states, dtype=int)
        self.means = all_obs[idx].copy()
        self.vars  = np.full_like(self.means, np.var(all_obs, axis=0).mean()+1e-4)

        self.monitor_ll = []; prev_ll = -np.inf

        for it in range(n_iter):
            # Accumulators
            A_num   = np.zeros_like(self.A)
            A_den   = np.zeros(self.n_states)
            pi_acc  = np.zeros(self.n_states)
            mu_num  = np.zeros_like(self.means)
            var_num = np.zeros_like(self.vars)
            g_sum   = np.zeros(self.n_states)
            total_ll = 0.0

            for obs in obs_list:
                T  = len(obs); lb = self._log_b(obs)
                lp, al, sc = self.forward(obs)
                bt          = self.backward(obs, sc)
                total_ll   += lp

                # γₜ(j) = αₜ(j)·βₜ(j) / P(X|λ)
                gm   = al * bt
                gn   = gm.sum(axis=1, keepdims=True)
                gm  /= np.where(gn==0, 1e-300, gn)  # (T,S)

                # ξₜ(i,j) accumulate → A_num
                for t in range(T-1):
                    xi   = (al[t,:,None] * self.A *
                            np.exp(lb[t+1]) * bt[t+1])
                    xn   = xi.sum() + 1e-300
                    A_num += xi / xn
                    A_den += gm[t]

                pi_acc  += gm[0]
                for j in range(self.n_states):
                    wj = gm[:, j]
                    mu_num[j]  += (wj[:,None] * obs).sum(axis=0)
                    var_num[j] += (wj[:,None] * (obs-self.means[j])**2).sum(axis=0)
                    g_sum[j]   += wj.sum()

            # M-step
            for i in range(self.n_states):
                d = A_den[i]+1e-300
                row = A_num[i]/d
                self.A[i] = row / (row.sum()+1e-300)
                # Enforce left-to-right: giữ mask
                mask = (np.triu(np.ones_like(self.A),0) -
                        np.triu(np.ones_like(self.A),2)) > 0
                self.A *= mask
                self.A[i] /= (self.A[i].sum()+1e-300)

            self.pi = pi_acc / (pi_acc.sum()+1e-300)
            for j in range(self.n_states):
                g = g_sum[j]+1e-300
                self.means[j] = mu_num[j]/g
                self.vars[j]  = np.maximum(var_num[j]/g, 1e-4)

            self.monitor_ll.append(total_ll)

            # Kiểm tra hội tụ
            if it > 2 and abs(total_ll - prev_ll) < tol:
                self.converged_ = True
                break
            prev_ll = total_ll

        return self

    # ════════════════════════════════════════════════════
    # VITERBI DECODING
    # δₜ(j) = Bⱼ(xₜ)·maxᵢ[δₜ₋₁(i)·Aᵢⱼ]
    # ψₜ(j) = argmaxᵢ[δₜ₋₁(i)·Aᵢⱼ]
    # ════════════════════════════════════════════════════
    def viterbi(self, obs):
        """
        Tìm chuỗi trạng thái tối ưu q* bằng Viterbi.
        Làm việc trong log-domain để tránh underflow.

        Returns: (log_prob, state_path, delta_matrix)
        """
        T  = len(obs); lb = self._log_b(obs)
        lA = np.log(self.A + 1e-300)
        lp = np.log(self.pi + 1e-300)

        delta = np.full((T, self.n_states), -np.inf)
        psi   = np.zeros((T, self.n_states), dtype=int)
        delta[0] = lp + lb[0]

        for t in range(1, T):
            for j in range(self.n_states):
                sc       = delta[t-1] + lA[:,j]
                bi       = int(np.argmax(sc))
                delta[t,j] = sc[bi] + lb[t,j]
                psi[t,j]   = bi

        # Backtrack
        path = [0]*T
        path[T-1] = int(np.argmax(delta[T-1]))
        for t in range(T-2, -1, -1):
            path[t] = psi[t+1, path[t+1]]

        return float(np.max(delta[T-1])), path, delta

    def score(self, obs):
        """log P(X|λ) – dùng để nhận dạng."""
        lp, _, _ = self.forward(obs); return lp

    def decode(self, obs, algorithm="viterbi"):
        """API tương thích hmmlearn: trả về (logp, state_seq)."""
        lp, path, _ = self.viterbi(obs)
        return lp, np.array(path)

    def save(self, path):
        with open(path,"wb") as f: pickle.dump(self.__dict__, f)

    @classmethod
    def load(cls, path):
        obj = cls.__new__(cls)
        with open(path,"rb") as f: obj.__dict__.update(pickle.load(f))
        return obj


# ═══════════════════════════════════════════════════════════
# C. HUẤN LUYỆN HMM
# ═══════════════════════════════════════════════════════════

def train_models(train_files, n_states=5, n_iter=80,
                 apply_trim=True) -> dict:
    """
    Huấn luyện 1 GaussianHMM_LR/từ bằng Baum-Welch.
    Nối tất cả utterances của cùng 1 lớp → fit với lengths.
    """
    models = {}
    print(f"\n  Huấn luyện HMM (n_states={n_states}, n_iter={n_iter}):")
    print("  " + "─"*55)

    for lab in LABELS:
        obs_list = []
        for fp in train_files[lab]:
            feat = extract_feat(fp, apply_trim=apply_trim)
            obs_list.append(feat)

        hmm = GaussianHMM_LR(n_states=n_states, n_dim=N_FEAT, label=lab)
        hmm.fit(obs_list, n_iter=n_iter)
        models[lab] = hmm

        ll = hmm.monitor_ll[-1] if hmm.monitor_ll else float("-inf")
        cvg = "✓" if hmm.converged_ else "~"
        print(f"  [{lab}] {cvg} LL_final={ll:.2f}  ({len(hmm.monitor_ll)} vòng)")
        print(f"    transmat (diag)=[{', '.join(f'{hmm.A[i,i]:.3f}' for i in range(n_states))}]")

    return models


def plot_training_convergence(models_dict: dict, tag="") -> None:
    """Vẽ đường hội tụ log-likelihood Baum-Welch cho tất cả model."""
    fig, ax = plt.subplots(figsize=(10,5))
    colors = plt.cm.tab10(np.linspace(0,1,len(LABELS)))
    for lab, clr in zip(LABELS, colors):
        ll_hist = models_dict[lab].monitor_ll
        ax.plot(ll_hist, color=clr, lw=1.5, marker=".", ms=4,
                label=f"[{LABEL_VI[lab]}] final={ll_hist[-1]:.1f}")
    style_ax(ax, xlabel="Vòng lặp EM", ylabel="Log-likelihood",
             title="Phần C – Convergence Baum-Welch (Forward-Backward EM)\n"
                   "Log-likelihood tăng đơn điệu mỗi vòng đến cực đại cục bộ")
    ax.legend(fontsize=9)
    plt.tight_layout()
    savefig(fig, f"C_baum_welch_convergence{tag}.png")


def plot_transition_matrices(models_dict: dict, tag="") -> None:
    """Vẽ ma trận chuyển trạng thái A sau huấn luyện."""
    n = len(LABELS)
    fig, axes = plt.subplots(1, n, figsize=(4*n, 4))
    for ax, lab in zip(axes, LABELS):
        A = models_dict[lab].A
        im = ax.imshow(A, cmap="Blues", vmin=0, vmax=1)
        plt.colorbar(im, ax=ax, fraction=0.046)
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                if A[i,j] > 0.01:
                    ax.text(j,i,f"{A[i,j]:.2f}",ha="center",va="center",
                            fontsize=7, color="white" if A[i,j]>0.5 else "black")
        style_ax(ax, xlabel="State j (to)", ylabel="State i (from)",
                 title=f"[{LABEL_VI[lab]}]\nTransition A")
        ax.set_xticks(range(A.shape[0])); ax.set_yticks(range(A.shape[0]))
    fig.suptitle("Phần B – Ma trận chuyển trạng thái A (Left-to-Right)\n"
                 "Đường chéo=self-loop | Đường chéo+1=tiến | Giá trị 0=cấm",
                 fontsize=10, fontweight="bold")
    plt.tight_layout()
    savefig(fig, f"B_transition_matrices{tag}.png")


# ═══════════════════════════════════════════════════════════
# D. NHẬN DẠNG VÀ VITERBI ALIGNMENT
# ═══════════════════════════════════════════════════════════

def recognize(feat: np.ndarray, models: dict) -> tuple:
    """
    Nhận dạng Maximum Likelihood:
      ŵ = argmax_w log P(X|λ_w)

    Returns: (pred_label, scores_dict)
    """
    scores = {lab: models[lab].score(feat) for lab in LABELS}
    pred   = max(scores, key=scores.get)
    return pred, dict(sorted(scores.items(), key=lambda kv: -kv[1]))


def plot_viterbi_alignment(models: dict, train_files: dict,
                            words=None) -> None:
    """
    Vẽ Viterbi state path + MFCC heatmap cho 2+ utterances.
    Phân tích: duration của mỗi state phản ánh độ dài phoneme.
    """
    if words is None:
        words = ["mo", "dong"]

    fig, axes = plt.subplots(len(words), 2, figsize=(14, 5*len(words)))
    if len(words)==1: axes=axes.reshape(1,-1)

    for row, word in enumerate(words):
        fpath = train_files[word][0]
        feat  = extract_feat(fpath)
        T     = len(feat)
        t_ms  = np.arange(T)*HOP_LEN/FS*1000

        # Viterbi path và delta matrix
        lp_v, path, delta = models[word].viterbi(feat)

        # Subplot: delta trellis + path
        im0 = axes[row,0].imshow(delta.T, aspect="auto", origin="lower",
                                  cmap="YlOrRd",
                                  extent=[0,t_ms[-1],-0.5,models[word].n_states-0.5])
        plt.colorbar(im0, ax=axes[row,0], label="log δ", fraction=0.04)
        axes[row,0].plot(t_ms, path, "b.-", lw=1.5, ms=4,
                         label=f"Viterbi path (log P*={lp_v:.1f})")
        style_ax(axes[row,0], xlabel="Thời gian (ms)",
                 ylabel="Trạng thái",
                 title=f"Phần D – Viterbi Trellis [{LABEL_VI[word]}]")
        axes[row,0].legend(fontsize=8); axes[row,0].grid(True,alpha=.3)

        # State duration analysis
        from collections import Counter
        cnt = Counter(path)
        dur_str = "  ".join(f"S{s}:{cnt[s]*HOP_LEN/FS*1000:.0f}ms" for s in sorted(cnt))

        # Subplot: MFCC heatmap + state boundaries
        im1 = axes[row,1].imshow(feat[:,:13].T, aspect="auto", origin="lower",
                                  cmap="RdYlBu_r",
                                  extent=[0,t_ms[-1],0.5,13.5])
        plt.colorbar(im1, ax=axes[row,1], label="MFCC", fraction=0.04)

        # Vẽ biên state
        prev_s = path[0]
        for t in range(1,T):
            if path[t] != prev_s:
                axes[row,1].axvline(t_ms[t], color="white", lw=1.0, alpha=0.8)
                prev_s = path[t]

        style_ax(axes[row,1], xlabel="Thời gian (ms)",
                 ylabel="MFCC (1-13)",
                 title=f"MFCC + State boundaries [{LABEL_VI[word]}]\n{dur_str}")

    fig.suptitle("Phần D – Viterbi Alignment: Trellis và MFCC\n"
                 "State tiến đều theo thời gian (left-to-right); "
                 "self-loop = phần kéo dài âm thanh",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    savefig(fig, "D_viterbi_alignment.png")


def print_score_table(test_files: dict, models: dict,
                       n_examples: int = 3) -> None:
    """In bảng acoustic scores của n_examples test utterances."""
    print("\n  Bảng acoustic scores (log P(X|λ_w)):")
    print(f"  {'File':<22} {'True':<7} {'Pred':<7}", end="")
    for lab in LABELS:
        print(f" {LABEL_VI[lab]:>6}", end="")
    print()
    print("  " + "─"*70)

    count = 0
    for true_lab in LABELS:
        for fpath in test_files[true_lab][:1]:
            feat = extract_feat(fpath)
            pred, scores = recognize(feat, models)
            ok = "✓" if pred==true_lab else "✗"
            fname = os.path.basename(fpath)
            print(f"  {fname:<22} {true_lab:<7} {pred+ok:<7}", end="")
            for lab in LABELS:
                print(f" {scores[lab]:>6.0f}", end="")
            print()
            count += 1
            if count >= n_examples:
                break
        if count >= n_examples:
            break


def run_part_bcd(data: dict = None, n_states: int = 5) -> dict:
    """Chạy phần B, C, D: HMM training + Viterbi."""
    if data is None:
        train_files, test_files = get_files()
    else:
        train_files = data["train_files"]
        test_files  = data["test_files"]

    print("\n"+"═"*60)
    print(f"  PHẦN B – KHỞI TẠO HMM ({n_states} STATES)")
    print("═"*60)
    print(f"  Topology: Left-to-right | self_loop=0.7, forward=0.3")
    print(f"  π = [1,0,...,0] | covariance = diagonal")
    print(f"  Diagonal covariance: ít tham số hơn → ổn định với ít dữ liệu")
    print(f"  Full covariance: D²/2 tham số/state → cần nhiều dữ liệu hơn")

    print("\n"+"═"*60)
    print("  PHẦN C – HUẤN LUYỆN BAUM-WELCH")
    print("═"*60)

    models = train_models(train_files, n_states=n_states, n_iter=80)
    plot_training_convergence(models)
    plot_transition_matrices(models)

    print("\n"+"═"*60)
    print("  PHẦN D – NHẬN DẠNG VÀ VITERBI ALIGNMENT")
    print("═"*60)

    print_score_table(test_files, models, n_examples=5)
    plot_viterbi_alignment(models, train_files, words=["mo","dong","len"])

    return {"models": models, "train_files": train_files, "test_files": test_files}


if __name__ == "__main__":
    from lab3_features import run_part_a
    data = run_part_a()
    run_part_bcd(data)
