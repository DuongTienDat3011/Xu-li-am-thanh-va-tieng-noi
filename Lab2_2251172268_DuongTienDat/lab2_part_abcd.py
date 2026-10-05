"""
lab2_part_abcd.py
=================
CSE457 Lab 2 – Phần A, B, C, D
Sinh viên: Dương Tiến Đạt – MSSV: 2251172268

A. Thu dữ liệu & kiểm tra chất lượng
B. Đặc trưng miền thời gian (Energy, RMS, ZCR)
C. Endpoint detection (log-energy + ZCR)
D. MFCC (pre-emphasis → framing → FFT → Mel → log → DCT)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import lab2_utils as U
from lab2_utils import (
    FS, WIN_LEN, HOP_LEN, PRE_EMPH, N_FFT, N_MELS, N_MFCC,
    LABELS, LABEL_VI, DATASET_DIR, FIGURES_DIR, TRIM_DIR,
    read_wav, write_wav, pre_emphasis, framing,
    short_time_energy, short_time_rms, log_energy,
    zero_crossing_rate, short_time_autocorr, estimate_pitch,
    savefig, style_ax, db
)


# ════════════════════════════════════════════════════════════
# PHẦN A – THU DỮ LIỆU & KIỂM TRA
# ════════════════════════════════════════════════════════════

def run_part_a() -> dict:
    """
    Phần A: Đọc tất cả file WAV, kiểm tra:
      - Sampling rate, số kênh, thời lượng
      - Clipping (|x| >= 0.999)
      - Độ dài silence đầu/cuối
    Vẽ waveform của 3 từ khác nhau.
    """
    print("\n" + "═" * 60)
    print("  PHẦN A – THU DỮ LIỆU & KIỂM TRA CHẤT LƯỢNG")
    print("═" * 60)

    U.dataset_info()

    train_files, test_files = U.get_train_test_files()

    # Vẽ waveform 3 từ (1 file/từ)
    sample_words  = ["khong", "hai", "bon"]
    fig, axes = plt.subplots(3, 1, figsize=(13, 8), sharex=False)

    for ax, word in zip(axes, sample_words):
        fpath = train_files[word][0]
        x     = read_wav(fpath)
        t     = np.arange(len(x)) / FS
        peak  = float(np.max(np.abs(x)))
        rms   = float(np.sqrt(np.mean(x ** 2)))
        clip  = int(np.sum(np.abs(x) >= 0.999))

        ax.plot(t, x, color="steelblue", lw=0.5, alpha=0.85)
        ax.axhline( rms, color="orange", lw=1.0, ls="-.",
                    label=f"RMS={rms:.4f} ({db(rms):.1f}dBFS)")
        ax.axhline(-rms, color="orange", lw=1.0, ls="-.")
        ax.axhline( 1.0, color="red",    lw=0.6, ls="--", alpha=0.5)
        ax.axhline(-1.0, color="red",    lw=0.6, ls="--", alpha=0.5)
        style_ax(ax, xlabel="Thời gian (s)", ylabel="Biên độ",
                 title=f"[{LABEL_VI[word]}] | {os.path.basename(fpath)} | "
                       f"Peak={peak:.4f} ({db(peak):.1f}dBFS) | Clipping={clip}")
        ax.legend(fontsize=8, loc="upper right")
        ax.set_ylim([-1.15, 1.15])

    fig.suptitle("Phần A – Waveform 3 từ mẫu (chuẩn hóa full-scale)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    savefig(fig, "A_waveform_3words.png")

    # Bảng tóm tắt tất cả file
    print("\n  ┌──────┬───────┬──────────┬───────────┬─────────┬─────────┐")
    print("  │ Từ   │ File  │ Dur (s)  │ Peak (dBFS)│ Clip    │ Train?  │")
    print("  ├──────┼───────┼──────────┼───────────┼─────────┼─────────┤")
    all_info = {}
    for lab in LABELS:
        for is_train, fdict in [(True, train_files), (False, test_files)]:
            for fpath in fdict[lab]:
                x    = read_wav(fpath)
                dur  = len(x) / FS
                peak = float(np.max(np.abs(x)))
                clip = int(np.sum(np.abs(x) >= 0.999))
                tag  = "Train" if is_train else "Test"
                fname = os.path.basename(fpath)
                print(f"  │ {lab:<4} │ {fname:<5} │ {dur:>7.3f}s │ {db(peak):>9.2f}dB │ {clip:>7} │ {tag:<7} │")
                all_info[fpath] = dict(x=x, dur=dur, peak=peak, clip=clip,
                                       label=lab, is_train=is_train)
    print("  └──────┴───────┴──────────┴───────────┴─────────┴─────────┘")
    print(f"\n  Train: {sum(1 for v in all_info.values() if v['is_train'])} file  |  "
          f"Test: {sum(1 for v in all_info.values() if not v['is_train'])} file")
    print("  ✓ Phần A hoàn thành.")
    return {"train_files": train_files, "test_files": test_files, "all_info": all_info}


# ════════════════════════════════════════════════════════════
# PHẦN B – ĐẶC TRƯNG MIỀN THỜI GIAN
# ════════════════════════════════════════════════════════════

def run_part_b(data: dict) -> None:
    """
    Phần B: Tính và vẽ Energy, RMS, ZCR theo thời gian.
    Phân tích silence/voiced/unvoiced.
    """
    print("\n" + "═" * 60)
    print("  PHẦN B – ĐẶC TRƯNG MIỀN THỜI GIAN")
    print("═" * 60)

    train_files = data["train_files"]
    sample_words = ["khong", "hai", "ba"]  # 3 từ có đặc điểm âm học khác nhau

    fig = plt.figure(figsize=(14, 12))
    gs  = gridspec.GridSpec(len(sample_words), 3, figure=fig,
                             hspace=0.55, wspace=0.35)

    for row, word in enumerate(sample_words):
        fpath  = train_files[word][0]
        x      = read_wav(fpath)
        t_samp = np.arange(len(x)) / FS

        # Framing
        frames, t_frames = framing(x, WIN_LEN, HOP_LEN)

        # Tính các đặc trưng
        le  = log_energy(frames)        # Log-energy (dB)
        rms = short_time_rms(frames)    # RMS
        zcr = zero_crossing_rate(frames)# ZCR

        # Cột 1: Waveform + RMS line
        ax1 = fig.add_subplot(gs[row, 0])
        ax1.plot(t_samp, x, color="steelblue", lw=0.4, alpha=0.85)
        ax1.set_xlim([0, t_samp[-1]])
        ax1.set_ylim([-1.15, 1.15])
        style_ax(ax1, xlabel="s", ylabel="Biên độ",
                 title=f"[{LABEL_VI[word]}] Waveform")

        # Cột 2: Log-energy
        ax2 = fig.add_subplot(gs[row, 1])
        ax2.plot(t_frames, le, color="darkorange", lw=1.0)
        ax2.fill_between(t_frames, le, le.min(), alpha=0.25, color="darkorange")
        style_ax(ax2, xlabel="s", ylabel="Log-energy (dB)",
                 title=f"[{LABEL_VI[word]}] Log-Energy")
        ax2.set_xlim([0, t_samp[-1]])

        # Cột 3: ZCR
        ax3 = fig.add_subplot(gs[row, 2])
        ax3.plot(t_frames, zcr, color="seagreen", lw=1.0)
        ax3.fill_between(t_frames, zcr, 0, alpha=0.25, color="seagreen")
        style_ax(ax3, xlabel="s", ylabel="ZCR (crossings/sample)",
                 title=f"[{LABEL_VI[word]}] ZCR")
        ax3.set_xlim([0, t_samp[-1]])

        # In thống kê
        print(f"\n  [{LABEL_VI[word]}] {os.path.basename(fpath)}")
        print(f"    N frames = {len(frames)}")
        print(f"    Log-energy: min={le.min():.1f}  max={le.max():.1f}  "
              f"mean={le.mean():.1f} dB")
        print(f"    ZCR      : min={zcr.min():.4f}  max={zcr.max():.4f}  "
              f"mean={zcr.mean():.4f}")

        # Phân tích voiced/unvoiced/silence
        le_thresh  = le.max() - 35        # Ngưỡng silence: 35 dB dưới max
        zcr_thresh = 0.08                  # Ngưỡng voiced/unvoiced
        n_sil  = np.sum(le < le_thresh)
        n_voiced   = np.sum((le >= le_thresh) & (zcr < zcr_thresh))
        n_unvoiced = np.sum((le >= le_thresh) & (zcr >= zcr_thresh))
        print(f"    Silence (le<{le_thresh:.0f}dB): {n_sil} frames "
              f"({n_sil/len(frames)*100:.0f}%)")
        print(f"    Voiced (zcr<{zcr_thresh:.2f}): {n_voiced} frames "
              f"({n_voiced/len(frames)*100:.0f}%)")
        print(f"    Unvoiced (zcr≥{zcr_thresh:.2f}): {n_unvoiced} frames "
              f"({n_unvoiced/len(frames)*100:.0f}%)")

    fig.suptitle("Phần B – Waveform, Log-Energy, ZCR theo thời gian\n"
                 "(Voiced: energy cao + ZCR thấp | Unvoiced: ZCR cao | "
                 "Silence: energy thấp)",
                 fontsize=10, fontweight="bold")
    savefig(fig, "B_energy_zcr.png")

    # Vẽ thêm autocorrelation + pitch cho 1 frame voiced vs 1 frame unvoiced
    _plot_autocorr_analysis(train_files)

    print("\n  ✓ Phần B hoàn thành.")


def _plot_autocorr_analysis(train_files: dict) -> None:
    """Vẽ autocorrelation của frame voiced vs unvoiced để ước lượng pitch."""
    word  = "khong"
    fpath = train_files[word][0]
    x     = read_wav(fpath)
    frames, t_frames = framing(x, WIN_LEN, HOP_LEN)
    le    = log_energy(frames)
    zcr   = zero_crossing_rate(frames)

    # Tìm frame voiced (energy cao, ZCR thấp) và unvoiced (energy cao, ZCR cao)
    le_thresh   = le.max() - 30
    voiced_idx  = np.where((le > le_thresh) & (zcr < 0.08))[0]
    unvoiced_idx= np.where((le > le_thresh - 5) & (zcr >= 0.10))[0]

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))

    for col, (idx_arr, lbl, clr) in enumerate([
        (voiced_idx,   "Voiced (hữu thanh)",   "steelblue"),
        (unvoiced_idx, "Unvoiced (vô thanh)",  "tomato"),
    ]):
        if len(idx_arr) == 0:
            continue
        fi    = idx_arr[len(idx_arr) // 2]
        frame = frames[fi]
        t_f   = np.arange(WIN_LEN) / FS * 1000   # ms

        # Time domain của frame
        axes[0, col].plot(t_f, frame, color=clr, lw=0.7)
        style_ax(axes[0, col], xlabel="ms", ylabel="Biên độ",
                 title=f"{lbl} | Frame #{fi} | t={t_frames[fi]:.3f}s")

        # Autocorrelation
        r   = short_time_autocorr(frame, max_lag=int(FS / 60))
        lag = np.arange(len(r)) / FS * 1000   # ms
        axes[1, col].plot(lag, r, color=clr, lw=0.8)
        axes[1, col].axhline(0, color="black", lw=0.5)

        f0 = estimate_pitch(frame, FS)
        if f0 > 0:
            axes[1, col].axvline(1000.0 / f0, color="purple", lw=1.0,
                                  ls="--", label=f"F0≈{f0:.0f}Hz")
            axes[1, col].legend(fontsize=8)
            print(f"  [{lbl}] Frame #{fi}: F0 ≈ {f0:.1f} Hz")
        else:
            print(f"  [{lbl}] Frame #{fi}: Không tìm thấy F0 (unvoiced)")

        style_ax(axes[1, col], xlabel="Lag (ms)", ylabel="R[k] (normalized)",
                 title=f"Autocorrelation – {lbl}")

    fig.suptitle("Phần B – Short-time autocorrelation và ước lượng F0\n"
                 "Voiced: đỉnh rõ tại lag=T0 | Unvoiced: không có đỉnh",
                 fontsize=10, fontweight="bold")
    plt.tight_layout()
    savefig(fig, "B_autocorrelation.png")


# ════════════════════════════════════════════════════════════
# PHẦN C – ENDPOINT DETECTION
# ════════════════════════════════════════════════════════════

def endpoint_detection(x: np.ndarray,
                        top_db: float = 30.0,
                        zcr_thresh: float = 0.15,
                        margin_ms: float = 40.0) -> tuple:
    """
    Tìm start/end frame dựa trên log-energy và ZCR.

    Thuật toán:
      1. Tính log-energy và ZCR theo frame
      2. Ngưỡng energy = max_energy - top_db (tương đối)
      3. Tìm vùng liên tục có energy >= ngưỡng
      4. Dùng ZCR để mở rộng biên nếu từ bắt đầu/kết thúc bằng fricative
      5. Thêm margin để không cắt mất phụ âm đầu/cuối

    Parameters
    ----------
    x          : tín hiệu 1D float64
    top_db     : dải động (dB) – frame dưới max_energy - top_db = silence
    zcr_thresh : ZCR cao → có thể là unvoiced speech (fricative)
    margin_ms  : margin thêm vào trước/sau biên (ms)

    Returns
    -------
    x_trimmed  : tín hiệu đã cắt
    (start_samp, end_samp): vị trí mẫu
    (start_frame, end_frame): chỉ số frame
    """
    frames, t_frames = framing(x, WIN_LEN, HOP_LEN)
    le  = log_energy(frames)
    zcr = zero_crossing_rate(frames)

    # ── Bước 1: Ngưỡng energy tương đối ──
    le_thresh = le.max() - top_db

    # ── Bước 2: Mask frame "có speech" dựa trên energy ──
    speech_mask = le >= le_thresh

    # ── Bước 3: Mở rộng mask bằng ZCR (bắt fricative) ──
    # Nếu frame ngay trước/sau vùng speech có ZCR cao → có thể là phụ âm
    for i in range(1, len(speech_mask)):
        if speech_mask[i] and not speech_mask[i-1] and zcr[i-1] >= zcr_thresh:
            speech_mask[i-1] = True
    for i in range(len(speech_mask)-2, -1, -1):
        if speech_mask[i] and not speech_mask[i+1] and zcr[i+1] >= zcr_thresh:
            speech_mask[i+1] = True

    # ── Bước 4: Tìm start/end frame ──
    speech_idx = np.where(speech_mask)[0]
    if len(speech_idx) == 0:
        return x, (0, len(x)), (0, len(frames)-1)

    start_frame = int(speech_idx[0])
    end_frame   = int(speech_idx[-1])

    # ── Bước 5: Thêm margin ──
    margin_frames = max(1, int(margin_ms * 1e-3 * FS / HOP_LEN))
    start_frame   = max(0, start_frame - margin_frames)
    end_frame     = min(len(frames) - 1, end_frame + margin_frames)

    # Chuyển về mẫu
    start_samp = start_frame * HOP_LEN
    end_samp   = min(len(x), end_frame * HOP_LEN + WIN_LEN)

    x_trimmed = x[start_samp:end_samp]
    return x_trimmed, (start_samp, end_samp), (start_frame, end_frame)


def run_part_c(data: dict) -> dict:
    """
    Phần C: Áp dụng endpoint detection, vẽ minh họa, lưu file trim.
    """
    print("\n" + "═" * 60)
    print("  PHẦN C – ENDPOINT DETECTION")
    print("═" * 60)

    train_files = data["train_files"]
    TOP_DB      = 30.0     # Ngưỡng: frame dưới max-30dB → silence
    ZCR_THRESH  = 0.15     # ZCR > 0.15 → có thể fricative
    MARGIN_MS   = 40.0     # Margin 40ms

    print(f"  Tham số: top_db={TOP_DB}dB | zcr_thresh={ZCR_THRESH} | margin={MARGIN_MS}ms")

    fig, axes = plt.subplots(5, 1, figsize=(13, 14), sharex=False)
    trim_info  = {}

    for ax, lab in zip(axes, LABELS):
        fpath = train_files[lab][0]
        x     = read_wav(fpath)

        x_trim, (s_samp, e_samp), (s_fr, e_fr) = endpoint_detection(
            x, top_db=TOP_DB, zcr_thresh=ZCR_THRESH, margin_ms=MARGIN_MS)

        t_full = np.arange(len(x)) / FS
        frames, t_frames = framing(x, WIN_LEN, HOP_LEN)
        le  = log_energy(frames)
        zcr = zero_crossing_rate(frames)

        # Vẽ waveform với vùng speech được tô màu
        ax.plot(t_full, x, color="lightgray", lw=0.4, label="Toàn bộ")
        ax.axvspan(s_samp / FS, e_samp / FS,
                   color="steelblue", alpha=0.25, label="Vùng speech")
        ax.plot(t_full[s_samp:e_samp],
                x[s_samp:e_samp], color="steelblue", lw=0.6)
        ax.axvline(s_samp / FS, color="green", lw=1.2, ls="--", label="Start")
        ax.axvline(e_samp / FS, color="red",   lw=1.2, ls="--", label="End")

        dur_before = len(x) / FS
        dur_after  = len(x_trim) / FS
        style_ax(ax, xlabel="Thời gian (s)", ylabel="Biên độ",
                 title=f"[{LABEL_VI[lab]}] Trước={dur_before:.3f}s → "
                       f"Sau={dur_after:.3f}s "
                       f"(cắt {(dur_before-dur_after)*1000:.0f}ms)")
        ax.legend(fontsize=7, loc="upper right", ncol=4)
        ax.set_ylim([-1.15, 1.15])

        print(f"  [{lab:<5}] {os.path.basename(fpath)}  "
              f"Trước={dur_before:.3f}s → Sau={dur_after:.3f}s  "
              f"Cắt={((dur_before-dur_after)*1000):.0f}ms  "
              f"Frame {s_fr}→{e_fr}")

        # Lưu file trim
        trim_path = os.path.join(TRIM_DIR, f"{lab}_trimmed.wav")
        write_wav(x_trim, FS, trim_path)
        trim_info[lab] = {
            "x_trim": x_trim,
            "dur_before": dur_before, "dur_after": dur_after,
            "start_samp": s_samp, "end_samp": e_samp
        }

    fig.suptitle("Phần C – Endpoint Detection\n"
                 f"top_db={TOP_DB}dB | ZCR_thresh={ZCR_THRESH} | margin={MARGIN_MS}ms",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    savefig(fig, "C_endpoint_detection.png")

    print("\n  ✓ Phần C hoàn thành.")
    return {"trim_info": trim_info, "TOP_DB": TOP_DB,
            "ZCR_THRESH": ZCR_THRESH, "MARGIN_MS": MARGIN_MS}


# ════════════════════════════════════════════════════════════
# PHẦN D – MFCC
# ════════════════════════════════════════════════════════════

def mel_filterbank(n_fft: int, n_mels: int, fs: int,
                   fmin: float = 0.0, fmax: float = None) -> np.ndarray:
    """
    Tạo Mel filterbank gồm n_mels bộ lọc tam giác.

    Thang Mel theo Huang-Acero-Hon:
      B(f)   = 1125 · ln(1 + f/700)
      B⁻¹(b) = 700 · (exp(b/1125) - 1)

    Các bộ lọc phân bố đều trên thang Mel → dày hơn ở tần số thấp.

    Returns
    -------
    H : ndarray shape (n_mels, n_fft//2 + 1)
        Trọng số bộ lọc; mỗi hàng là 1 bộ lọc tam giác
    """
    if fmax is None:
        fmax = fs / 2.0

    # Tần số của các bin FFT
    n_freqs  = n_fft // 2 + 1
    freqs    = np.linspace(0, fs / 2, n_freqs)

    # Chuyển sang thang Mel
    mel_min  = 1125.0 * np.log(1 + fmin / 700.0)
    mel_max  = 1125.0 * np.log(1 + fmax / 700.0)

    # n_mels + 2 điểm đều trên thang Mel (gồm 2 biên)
    mel_pts  = np.linspace(mel_min, mel_max, n_mels + 2)
    # Chuyển ngược về Hz
    hz_pts   = 700.0 * (np.exp(mel_pts / 1125.0) - 1.0)

    H = np.zeros((n_mels, n_freqs))
    for m in range(n_mels):
        f_left   = hz_pts[m]
        f_center = hz_pts[m + 1]
        f_right  = hz_pts[m + 2]

        for k in range(n_freqs):
            f = freqs[k]
            if f_left <= f <= f_center:
                H[m, k] = (f - f_left) / (f_center - f_left + 1e-10)
            elif f_center < f <= f_right:
                H[m, k] = (f_right - f) / (f_right - f_center + 1e-10)

    return H


def compute_mfcc(x: np.ndarray,
                 alpha: float   = PRE_EMPH,
                 win_len: int   = WIN_LEN,
                 hop_len: int   = HOP_LEN,
                 n_fft: int     = N_FFT,
                 n_mels: int    = N_MELS,
                 n_mfcc: int    = N_MFCC,
                 fs: int        = FS,
                 cmn: bool      = True) -> np.ndarray:
    """
    Trích MFCC từ tín hiệu tiếng nói.

    Pipeline đầy đủ theo đề bài:
      pre-emphasis → frame+Hamming → FFT → power spectrum
      → Mel filterbank → log → DCT → [CMN]

    Parameters
    ----------
    x       : tín hiệu 1D float64 [-1,1]
    alpha   : pre-emphasis coefficient (0.97)
    cmn     : Cepstral Mean Normalization (áp dụng nhất quán train+test)

    Returns
    -------
    mfcc : ndarray shape (T_frames, n_mfcc)
           Mỗi hàng là 1 vector MFCC 13 chiều của 1 frame
    """
    # ── Bước 1: Pre-emphasis ──
    y = pre_emphasis(x, alpha)

    # ── Bước 2: Framing + Hamming window ──
    frames, _ = framing(y, win_len, hop_len)
    T = len(frames)

    # ── Bước 3: FFT → Power spectrum ──
    NFFT     = n_fft
    n_freqs  = NFFT // 2 + 1
    power_sp = np.zeros((T, n_freqs))
    for i in range(T):
        X         = np.fft.rfft(frames[i], n=NFFT)
        power_sp[i] = (np.abs(X) ** 2) / NFFT

    # ── Bước 4: Mel filterbank → log-energy ──
    H  = mel_filterbank(NFFT, n_mels, fs)  # (n_mels, n_freqs)
    # S[t, m] = ln(Σ_k P[t,k] · H[m,k] + ε)
    fb = np.dot(power_sp, H.T)            # (T, n_mels)
    log_fb = np.log(fb + 1e-10)            # (T, n_mels)

    # ── Bước 5: DCT → MFCC ──
    # c[n] = Σ_{m=0}^{M-1} S[m] · cos(π·n·(m+0.5)/M)
    M    = n_mels
    mfcc = np.zeros((T, n_mfcc))
    for n in range(n_mfcc):
        mfcc[:, n] = np.sum(
            log_fb * np.cos(np.pi * n *
                            (np.arange(M) + 0.5) / M)[np.newaxis, :],
            axis=1
        )

    # ── Bước 6: Cepstral Mean Normalization (CMN) ──
    if cmn and T > 1:
        mfcc -= np.mean(mfcc, axis=0, keepdims=True)

    return mfcc


def run_part_d(data: dict) -> dict:
    """
    Phần D: Trích MFCC, vẽ heatmap, vẽ Mel filterbank.
    """
    print("\n" + "═" * 60)
    print("  PHẦN D – MFCC")
    print("═" * 60)

    train_files = data["train_files"]

    print(f"  Tham số MFCC:")
    print(f"    Fs={FS}Hz | Frame={WIN_LEN}mẫu({WIN_LEN/FS*1000:.0f}ms) | "
          f"Hop={HOP_LEN}mẫu({HOP_LEN/FS*1000:.0f}ms)")
    print(f"    pre-emphasis α={PRE_EMPH} | NFFT={N_FFT} | "
          f"n_mels={N_MELS} | n_mfcc={N_MFCC} | CMN=True")

    # ── Vẽ Mel filterbank ──
    H    = mel_filterbank(N_FFT, N_MELS, FS)
    freqs = np.linspace(0, FS / 2, N_FFT // 2 + 1)

    fig_fb, ax_fb = plt.subplots(figsize=(12, 4))
    for m in range(N_MELS):
        ax_fb.plot(freqs, H[m], lw=0.8, alpha=0.7)
    style_ax(ax_fb,
             xlabel="Tần số (Hz)", ylabel="Trọng số",
             title=f"Mel Filterbank – {N_MELS} bộ lọc tam giác, Fs={FS}Hz\n"
                   "Mật độ dày ở tần số thấp (theo thang Mel)")
    ax_fb.set_xlim([0, FS / 2])
    plt.tight_layout()
    savefig(fig_fb, "D_mel_filterbank.png")

    # ── Trích MFCC cho tất cả file ──
    all_mfcc = {}
    for lab in LABELS:
        all_mfcc[lab] = []
        for fpath in train_files[lab]:
            x    = read_wav(fpath)
            mfcc = compute_mfcc(x)
            all_mfcc[lab].append(mfcc)
            print(f"  [{lab}] {os.path.basename(fpath)} → "
                  f"MFCC shape: {mfcc.shape}  "
                  f"({mfcc.shape[0]} frames × {mfcc.shape[1]} coefficients)")

    # ── Vẽ MFCC heatmap cho 2 từ khác nhau ──
    words_to_plot = ["khong", "hai", "ba", "bon"]
    fig_h, axes = plt.subplots(2, 2, figsize=(14, 9))
    axes_flat = axes.flatten()

    for ax, word in zip(axes_flat, words_to_plot):
        mfcc = all_mfcc[word][0]   # Lấy file đầu tiên
        im   = ax.imshow(mfcc.T, aspect="auto", origin="lower",
                         cmap="RdYlBu_r",
                         extent=[0, mfcc.shape[0] * HOP_LEN / FS * 1000,
                                 0.5, N_MFCC + 0.5])
        plt.colorbar(im, ax=ax, label="Giá trị MFCC")
        style_ax(ax, xlabel="Thời gian (ms)", ylabel="Hệ số MFCC (1–13)",
                 title=f"MFCC – [{LABEL_VI[word]}] | "
                       f"{mfcc.shape[0]} frames × {N_MFCC} coefficients")
        ax.set_yticks(range(1, N_MFCC + 1))

    fig_h.suptitle("Phần D – MFCC Heatmap: 4 từ khác nhau\n"
                   "Mỗi file có số frame khác nhau nhưng luôn có 13 hệ số/frame",
                   fontsize=11, fontweight="bold")
    plt.tight_layout()
    savefig(fig_h, "D_mfcc_heatmap.png")

    # In giải thích số frame
    print(f"\n  Giải thích số frame khác nhau:")
    print(f"    Số frame T = ceil((N - {WIN_LEN}) / {HOP_LEN}) + 1")
    print(f"    N = số mẫu, phụ thuộc thời lượng từng utterance")
    print(f"    Số hệ số MFCC/frame = {N_MFCC} = cố định (không phụ thuộc N)")

    print("\n  ✓ Phần D hoàn thành.")
    return {"all_mfcc": all_mfcc}


# ════════════════════════════════════════════════════════════
# CHẠY ĐẦY ĐỦ PHẦN A → D
# ════════════════════════════════════════════════════════════

def run_all_abcd():
    data_a = run_part_a()
    run_part_b(data_a)
    data_c = run_part_c(data_a)
    data_d = run_part_d(data_a)
    return {**data_a, **data_c, **data_d}


if __name__ == "__main__":
    results = run_all_abcd()
    print("\n  ✓ Phần A–D hoàn thành.\n")
