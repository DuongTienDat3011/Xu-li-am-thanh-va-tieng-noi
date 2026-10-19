"""
lab3_features.py
================
CSE457 Lab 3 – Mục A: Trích đặc trưng 39 chiều MFCC + Δ + Δ²
Sinh viên: Dương Tiến Đạt – MSSV: 2251172268

Theo đề bài Lab 3:
  - 13 MFCC tĩnh  (mô tả hình dạng phổ ngắn hạn)
  - 13 Δ-MFCC     (mô tả xu hướng thay đổi theo thời gian)
  - 13 Δ²-MFCC    (mô tả thay đổi của xu hướng)
  → xₜ = [cₜ, Δcₜ, Δ²cₜ] ∈ R³⁹

  CMVN: Cepstral Mean-Variance Normalization (theo utterance)
"""

import os, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from lab3_utils import (
    FS, LABELS, LABEL_VI, FIGURES_DIR,
    read_wav, savefig, style_ax, get_files
)

# ═══════════════════════════════════════════════════════════
# THAM SỐ MFCC
# ═══════════════════════════════════════════════════════════
FRAME_MS  = 25      # ms – độ dài frame
HOP_MS    = 10      # ms – bước nhảy
WIN_LEN   = int(FS * FRAME_MS / 1000)   # 400 mẫu
HOP_LEN   = int(FS * HOP_MS   / 1000)   # 160 mẫu
N_FFT     = 512
N_MELS    = 26      # Số Mel filters (Lab 3 dùng 26, tốt hơn 24)
N_MFCC    = 13
PRE_EMPH  = 0.97
N_FEAT    = 39      # 13 + 13 + 13


# ═══════════════════════════════════════════════════════════
# CÁC HÀM TÍNH MFCC VÀ ĐẶC TRƯNG ĐỘNG
# ═══════════════════════════════════════════════════════════

def pre_emphasis(x, alpha=PRE_EMPH):
    """y[n] = x[n] - α·x[n-1] – tăng tần số cao."""
    y = np.empty_like(x); y[0] = x[0]; y[1:] = x[1:] - alpha*x[:-1]
    return y

def endpoint_trim(x, fs=FS, top_db=30, margin_ms=40):
    """Cắt silence đầu/cuối bằng ngưỡng energy tương đối."""
    from scipy.signal import resample_poly
    from math import gcd
    WIN = int(0.025*fs); HOP = int(0.010*fs)
    N = len(x); n_fr = max(1, 1+(N-WIN)//HOP)
    ham = np.hamming(WIN); le = []
    for i in range(n_fr):
        s=i*HOP; seg=x[s:min(s+WIN,N)]
        if len(seg)<WIN: seg=np.pad(seg,(0,WIN-len(seg)))
        le.append(10*np.log10(np.sum((seg*ham)**2)+1e-10))
    le = np.array(le); thr = le.max() - top_db
    idx = np.where(le >= thr)[0]
    if len(idx)==0: return x
    mg = max(1, int(margin_ms*1e-3*fs/HOP))
    sf = max(0, int(idx[0])-mg)*HOP
    ef = min(N, (min(len(le)-1, int(idx[-1])+mg)+1)*HOP+WIN)
    return x[sf:ef]

def mel_filterbank(n_fft=N_FFT, n_mels=N_MELS, fs=FS,
                   fmin=20.0, fmax=None):
    """
    Mel filterbank theo Huang-Acero-Hon:
      B(f) = 1125·ln(1+f/700)
    Trả về H shape (n_mels, n_fft//2+1)
    """
    if fmax is None: fmax = fs/2.0
    n_freqs = n_fft//2+1
    freqs   = np.linspace(0, fs/2, n_freqs)
    mel_min = 1125*np.log(1+fmin/700)
    mel_max = 1125*np.log(1+fmax/700)
    mel_pts = np.linspace(mel_min, mel_max, n_mels+2)
    hz_pts  = 700*(np.exp(mel_pts/1125)-1)
    H = np.zeros((n_mels, n_freqs))
    for m in range(n_mels):
        fl,fc,fr = hz_pts[m],hz_pts[m+1],hz_pts[m+2]
        for k,f in enumerate(freqs):
            if   fl<=f<=fc: H[m,k]=(f-fl)/(fc-fl+1e-10)
            elif fc<f<=fr:  H[m,k]=(fr-f)/(fr-fc+1e-10)
    return H

def compute_mfcc_static(x, alpha=PRE_EMPH,
                         win_len=WIN_LEN, hop_len=HOP_LEN,
                         n_fft=N_FFT, n_mels=N_MELS, n_mfcc=N_MFCC,
                         fs=FS):
    """
    Tính 13 MFCC tĩnh từ tín hiệu:
      pre-emphasis → framing+Hamming → FFT → power → Mel → log → DCT
    Trả về ndarray shape (T, 13)
    """
    # Pre-emphasis
    y = pre_emphasis(x, alpha)
    # Framing
    N = len(y); n_fr = max(1,1+(N-win_len)//hop_len)
    ham = np.hamming(win_len); frames = np.zeros((n_fr,win_len))
    for i in range(n_fr):
        s=i*hop_len; seg=y[s:min(s+win_len,N)]
        if len(seg)<win_len: seg=np.pad(seg,(0,win_len-len(seg)))
        frames[i]=seg*ham
    # Power spectrum
    n_freqs=n_fft//2+1
    psp=np.array([(np.abs(np.fft.rfft(f,n=n_fft))**2)/n_fft for f in frames])
    # Mel filterbank → log
    H  = mel_filterbank(n_fft,n_mels,fs)
    fb = np.log(np.dot(psp,H.T)+1e-10)   # (T,n_mels)
    # DCT: c[n] = Σ S[m]·cos(π·n·(m+0.5)/M)
    M = n_mels
    mfcc = np.array([[np.sum(fb[t]*np.cos(np.pi*n*(np.arange(M)+.5)/M))
                      for n in range(n_mfcc)] for t in range(n_fr)])
    return mfcc  # (T,13)

def compute_delta(c, win=2):
    """
    Delta (đạo hàm bậc 1) theo hồi quy tuyến tính:
      Δcₜ = Σ_{k=1}^{win} k·(cₜ₊ₖ − cₜ₋ₖ) / (2·Σ k²)
    Ổn định hơn sai phân đơn giản.
    """
    T,D = c.shape; delta = np.zeros_like(c)
    denom = 2*sum(k**2 for k in range(1,win+1))
    for t in range(T):
        for k in range(1,win+1):
            delta[t] += k*(c[min(t+k,T-1)]-c[max(t-k,0)])
    return delta/(denom+1e-10)

def cmvn(feat):
    """
    Cepstral Mean-Variance Normalization (utterance-level):
      x' = (x - μ) / σ
    Đảm bảo mỗi chiều có mean≈0 và std≈1.
    """
    mu = feat.mean(axis=0,keepdims=True)
    sd = feat.std(axis=0,keepdims=True)+1e-8
    return (feat-mu)/sd

def extract_feat(path, apply_trim=True, apply_cmvn=True):
    """
    Pipeline đầy đủ:
      WAV → normalize → [trim] → MFCC 13 → Δ 13 → Δ² 13 → [CMVN] → (T,39)

    Parameters
    ----------
    path       : đường dẫn WAV
    apply_trim : có cắt silence không
    apply_cmvn : có áp CMVN không

    Returns
    -------
    feat : ndarray (T, 39)
    """
    x = read_wav(path)
    if apply_trim:
        x = endpoint_trim(x)
    if len(x) < WIN_LEN:
        x = np.pad(x,(0,WIN_LEN-len(x)))
    c   = compute_mfcc_static(x)
    d1  = compute_delta(c, win=2)
    d2  = compute_delta(d1, win=2)
    feat = np.concatenate([c,d1,d2], axis=1)   # (T,39)
    if apply_cmvn:
        feat = cmvn(feat)
    return feat


# ═══════════════════════════════════════════════════════════
# VẼ HÌNH
# ═══════════════════════════════════════════════════════════

def plot_feat_heatmap(train_files):
    """
    Vẽ heatmap MFCC / Δ / Δ² cho 2 từ khác nhau.
    Mục tiêu: so sánh cấu trúc đặc trưng giữa lệnh ngắn và lệnh dài hơn.
    """
    words = ["mo", "dong"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    cmap_list = ["RdYlBu_r", "PuOr", "PiYG"]
    titles_col = ["MFCC (13D)", "Δ-MFCC (13D)", "Δ²-MFCC (13D)"]

    for row, word in enumerate(words):
        fpath = train_files[word][0]
        x = read_wav(fpath)
        x = endpoint_trim(x)
        c   = compute_mfcc_static(x)
        d1  = compute_delta(c)
        d2  = compute_delta(d1)
        T   = len(c)
        t_ms= np.arange(T)*HOP_LEN/FS*1000

        for col, (mat, cmap) in enumerate(zip([c,d1,d2], cmap_list)):
            ax = axes[row,col]
            im = ax.imshow(mat.T, aspect="auto", origin="lower",
                           cmap=cmap,
                           extent=[0,t_ms[-1],0.5,N_MFCC+0.5])
            plt.colorbar(im, ax=ax, fraction=0.04)
            style_ax(ax,
                     xlabel="Thời gian (ms)" if row==1 else "",
                     ylabel=f"Hệ số (1–{N_MFCC})" if col==0 else "",
                     title=f"[{LABEL_VI[word]}] {titles_col[col]}")
            ax.set_yticks([1,5,9,13])

    fig.suptitle("Phần A – Heatmap đặc trưng 39D: MFCC, Δ, Δ²\n"
                 "MFCC mô tả hình dạng phổ; Δ mô tả xu hướng; Δ² mô tả gia tốc",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    savefig(fig, "A_feat_heatmap_mfcc_delta.png")


def plot_feat_comparison(train_files):
    """Vẽ shape (T,39) cho tất cả 5 lệnh để so sánh số frame và cấu trúc."""
    fig, axes = plt.subplots(1, len(LABELS), figsize=(16,5))
    for ax, lab in zip(axes, LABELS):
        feat = extract_feat(train_files[lab][0])
        im   = ax.imshow(feat.T, aspect="auto", origin="lower",
                         cmap="RdYlBu_r",
                         extent=[0, feat.shape[0]*HOP_LEN/FS*1000, 0.5, 39.5])
        plt.colorbar(im, ax=ax, fraction=0.04)
        style_ax(ax, xlabel="ms", ylabel="Feature dim",
                 title=f"[{LABEL_VI[lab]}]\n{feat.shape[0]}fr×{feat.shape[1]}D")
        ax.axhline(13.5, color="white", lw=0.8, ls="--", alpha=0.7)
        ax.axhline(26.5, color="white", lw=0.8, ls="--", alpha=0.7)
    fig.suptitle("Phần A – 39D Feature Matrix (T×39) cho 5 lệnh\n"
                 "Đường trắng phân chia: MFCC(1-13) | Δ(14-26) | Δ²(27-39)",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    savefig(fig, "A_feat_all5words.png")


def run_part_a():
    """Chạy phần A: trích đặc trưng và vẽ hình."""
    print("\n"+"═"*60)
    print("  PHẦN A – TRÍCH ĐẶC TRƯNG 39D (MFCC + Δ + Δ²)")
    print("═"*60)
    print(f"  Tham số: Fs={FS}Hz | Frame={FRAME_MS}ms | Hop={HOP_MS}ms")
    print(f"           N_FFT={N_FFT} | N_MELS={N_MELS} | N_MFCC={N_MFCC}")
    print(f"           α_preemph={PRE_EMPH} | CMVN=utterance")

    train_files, test_files = get_files()

    # Kiểm tra shape
    print("\n  Kiểm tra shape đặc trưng (39D):")
    for lab in LABELS:
        feat = extract_feat(train_files[lab][0])
        fname = os.path.basename(train_files[lab][0])
        print(f"  [{lab}] {fname} → shape={feat.shape}  "
              f"(T={feat.shape[0]} frames, D={feat.shape[1]})")
        assert feat.shape[1] == N_FEAT, f"Chiều phải = {N_FEAT}!"

    print("\n  Vẽ heatmap MFCC/Δ/Δ²...")
    plot_feat_heatmap(train_files)
    plot_feat_comparison(train_files)

    print("\n  Giải thích tại sao cần 39D:")
    print("    - 13 MFCC tĩnh: mô tả 'hình dạng' phổ tại frame t")
    print("    - 13 Δ-MFCC   : xu hướng thay đổi → phân biệt phụ âm đầu/cuối")
    print("    - 13 Δ²-MFCC  : gia tốc thay đổi → phát hiện burst, transition nhanh")
    print("    - HMM chuẩn giả định output independence → Δ,Δ² đưa")
    print("      thông tin thời gian VÀO mỗi frame, không vi phạm giả thiết")

    print("\n  ✓ Phần A hoàn thành.")
    return {"train_files": train_files, "test_files": test_files}


if __name__ == "__main__":
    run_part_a()
