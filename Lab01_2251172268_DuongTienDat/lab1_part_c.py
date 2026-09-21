"""
lab1_part_c.py
==============
CSE457 Lab 1 – Phần C: Phân tích miền tần số bằng FFT
-------------------------------------------------------
Yêu cầu:
  - Chọn đoạn ổn định 0.5–1s, áp cửa sổ Hamming, tính FFT
  - Vẽ magnitude spectrum (tuyến tính và dB)
  - Chỉ ra ít nhất 3 đỉnh nổi bật với tần số và biên độ
  - So sánh 2 giá trị NFFT: phân biệt frequency-bin spacing vs true resolution
  - Giải thích Δf = Fs/NFFT và true resolution = Fs/N_window
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

from lab1_utils import (
    read_wav, choose_audio_file, savefig, style_ax, db,
    FIGURES_DIR
)


# ════════════════════════════════════════════════════════════
# C1 – Tính FFT chuẩn (normalized, one-sided)
# Quy trình:
#   1. Áp cửa sổ lên frame để giảm spectral leakage
#   2. Tính FFT bằng numpy.fft.rfft (chỉ phần tần số dương)
#   3. Normalize: chia N·coherent_gain, nhân 2 (one-sided correction)
#   4. Tính magnitude dB, power dB và phase
# ════════════════════════════════════════════════════════════

def compute_fft(x: np.ndarray, fs: int,
                nfft: int = None,
                window: str = "hamming") -> dict:
    """
    Tính one-sided FFT (phổ một phía) của đoạn tín hiệu x.

    Lý do normalize:
      - Chia N·CG: loại ảnh hưởng của độ dài frame và coherent gain cửa sổ
      - Nhân 2: vì one-sided bỏ phần tần số âm (đối xứng với phần dương)
      - Không nhân 2 cho DC (k=0) và Nyquist (k=N/2) vì không có đối xứng

    Parameters
    ----------
    x      : tín hiệu 1D float64 [-1,1]
    fs     : sampling rate (Hz)
    nfft   : số điểm FFT (zero-padding nếu nfft > len(x))
    window : loại cửa sổ ('hamming'|'hann'|'blackman'|'rectangular')

    Returns – dict chứa:
      f          : mảng tần số Hz [0…Fs/2]
      mag_norm   : biên độ normalized [0…1]
      mag_dB     : 20·log10(mag_norm) [dBFS]
      power_dB   : 10·log10(mag_norm²)
      phase      : góc pha (radians)
      delta_f    : Fs/NFFT – khoảng cách giữa các bin (Hz)
      true_res   : Fs/N_window – độ phân giải thực (Hz)
    """
    N = len(x)

    # Tự động chọn NFFT = lũy thừa 2 lớn nhất khi không chỉ định
    if nfft is None:
        nfft = int(2 ** np.ceil(np.log2(N)))

    # Tạo cửa sổ theo tên
    win_funcs = {
        "hamming"    : np.hamming,
        "hann"       : np.hanning,
        "blackman"   : np.blackman,
        "rectangular": np.ones,
        "rect"       : np.ones,
    }
    w_func = win_funcs.get(window.lower(), np.hamming)
    win    = w_func(N)

    # Coherent gain (CG): trung bình của các hệ số cửa sổ
    # Hamming CG ≈ 0.54, Rectangular CG = 1.0
    cg = float(np.sum(win)) / N

    # Nhân cửa sổ vào tín hiệu để giảm spectral leakage
    xw = x * win

    # Tính FFT (chỉ phần không âm: 0 đến Fs/2)
    X = np.fft.rfft(xw, n=nfft)
    f = np.fft.rfftfreq(nfft, d=1.0 / fs)

    # Normalize biên độ
    mag_norm = np.abs(X) / (N * cg)
    mag_norm[1:-1] *= 2.0           # One-sided correction (nhân 2 trừ DC và Nyquist)

    mag_dB   = db(mag_norm)
    power    = mag_norm ** 2
    power_dB = np.maximum(10 * np.log10(power + 1e-30), -120.0)
    phase    = np.angle(X)

    return dict(
        f        = f,
        mag      = np.abs(X),       # Biên độ thô (chưa normalize)
        mag_norm = mag_norm,        # Biên độ normalized
        mag_dB   = mag_dB,
        power    = power,
        power_dB = power_dB,
        phase    = phase,
        delta_f  = fs / nfft,       # Khoảng cách bin (Hz) – phụ thuộc NFFT
        true_res = fs / N,          # Độ phân giải thực (Hz) – phụ thuộc N_window
        nfft     = nfft,
        n_window = N,
        window   = window,
        win_gain = cg,
    )


# ════════════════════════════════════════════════════════════
# C2 – Tìm các đỉnh nổi bật trên phổ dB
# Dùng scipy.signal.find_peaks với các điều kiện:
#   - Cao hơn ngưỡng min_dB
#   - Cách nhau ít nhất 20 Hz
#   - Có prominence (độ nổi) >= 3 dB
# ════════════════════════════════════════════════════════════

def find_spectral_peaks(r: dict, n_peaks: int = 6,
                        min_dB: float = -60.0,
                        min_freq: float = 20.0) -> list:
    """
    Tìm n_peaks đỉnh lớn nhất trên phổ dB.
    Bỏ qua tần số < min_freq (DC offset, hum).
    Trả về list[(freq_Hz, magnitude_dB)] sắp theo biên độ giảm dần.
    """
    f       = r["f"]
    mag_db  = r["mag_dB"]
    delta_f = r["delta_f"]

    # Chỉ tìm đỉnh ở tần số > min_freq
    valid      = f > min_freq
    idx_offset = int(np.argmax(valid))

    # Khoảng cách tối thiểu giữa các đỉnh: tương đương 20 Hz
    min_dist = max(3, int(20 / delta_f))

    peaks_rel, props = find_peaks(
        mag_db[valid],
        height    = min_dB,        # Ngưỡng biên độ tối thiểu
        distance  = min_dist,      # Khoảng cách bin tối thiểu
        prominence = 3.0,          # Độ nổi tối thiểu 3 dB
    )
    peaks_abs = peaks_rel + idx_offset

    if len(peaks_abs) == 0:
        return []

    # Sắp xếp theo biên độ giảm dần, lấy top n_peaks
    order = np.argsort(mag_db[peaks_abs])[::-1][:n_peaks]
    return [(float(f[peaks_abs[i]]), float(mag_db[peaks_abs[i]]))
            for i in order]


# ════════════════════════════════════════════════════════════
# C3 – Vẽ phổ FFT (2 subplot: tuyến tính + dB)
# Subplot 1: |X(f)| tuyến tính – thể hiện biên độ tuyệt đối
# Subplot 2: |X(f)| dB – thể hiện dải động, dễ thấy các thành phần nhỏ
# Đánh dấu top 5 đỉnh bằng màu sắc khác nhau với annotation
# ════════════════════════════════════════════════════════════

def plot_fft(r: dict, fname: str = "",
             freq_max: float = None,
             save_name: str = "C_fft.png") -> None:
    f      = r["f"]
    fs_nyq = float(f[-1])
    if freq_max is None:
        freq_max = min(8000.0, fs_nyq)

    mask  = f <= freq_max
    peaks = find_spectral_peaks(r, n_peaks=5)

    fig, axes = plt.subplots(2, 1, figsize=(13, 9))

    # ── Subplot 1: biên độ tuyến tính ──
    axes[0].plot(f[mask], r["mag_norm"][mask], color="steelblue", lw=0.7)
    axes[0].fill_between(f[mask], r["mag_norm"][mask], alpha=0.25, color="steelblue")
    style_ax(axes[0], ylabel="|X(f)| (normalized)",
             title=f"Phổ biên độ tuyến tính  |  NFFT={r['nfft']:,}  |  Δf={r['delta_f']:.4f} Hz")

    # ── Subplot 2: biên độ dB ──
    axes[1].plot(f[mask], r["mag_dB"][mask], color="darkorange", lw=0.8)
    style_ax(axes[1], xlabel="Tần số (Hz)", ylabel="Biên độ (dBFS)",
             title=f"Phổ dB  |  Cửa sổ {r['window'].capitalize()}  |  "
                   f"True resolution = {r['true_res']:.3f} Hz  (Fs/N_window)")
    axes[1].set_ylim([-90, 5])
    axes[1].set_xlim([0, freq_max])

    # Đánh dấu và ghi nhãn các đỉnh phổ
    clrs = ["red", "limegreen", "blueviolet", "brown", "deeppink"]
    for i, (fp, fdb) in enumerate(peaks):
        if fp <= freq_max:
            c = clrs[i % len(clrs)]
            axes[1].axvline(fp, color=c, lw=0.8, ls="--", alpha=0.7)
            axes[1].annotate(
                f"#{i+1}\n{fp:.1f}Hz\n{fdb:.1f}dB",
                xy=(fp, fdb),
                xytext=(fp + freq_max * 0.012, fdb - 8),
                fontsize=7, color=c,
                arrowprops=dict(arrowstyle="->", color=c, lw=0.5),
            )

    axes[1].set_xlim([0, freq_max])
    axes[0].set_xlim([0, freq_max])

    # In bảng đỉnh phổ và thông số FFT
    if peaks:
        print(f"\n  Top đỉnh phổ (NFFT={r['nfft']:,}):")
        print(f"  {'#':>2}  {'Tần số (Hz)':>14}  {'Biên độ (dBFS)':>16}")
        print(f"  {'─'*2}  {'─'*14}  {'─'*16}")
        for i, (fp, fdb) in enumerate(peaks, 1):
            print(f"  {i:>2}  {fp:>14.3f}  {fdb:>16.2f}")

    print(f"\n  Thông tin FFT:")
    print(f"    NFFT        = {r['nfft']:,}")
    print(f"    N window    = {r['n_window']:,} mẫu")
    print(f"    Δf (bin)    = {r['delta_f']:.5f} Hz  ← phụ thuộc NFFT")
    print(f"    True resol. = {r['true_res']:.5f} Hz  ← phụ thuộc N_window (quan trọng)")
    print(f"    Window gain = {r['win_gain']:.5f}  ({r['window']})")

    fig.suptitle(f"Phần C – Phân tích FFT: {fname}",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    savefig(fig, save_name)


# ════════════════════════════════════════════════════════════
# C4 – So sánh NFFT nhỏ vs NFFT lớn trên CÙNG frame
# Mục đích: chứng minh rằng:
#   - NFFT lớn hơn → Δf nhỏ hơn → đường phổ mịn hơn (nhiều điểm hơn)
#   - Nhưng TRUE RESOLUTION = Fs/N_window KHÔNG ĐỔI vì frame giữ nguyên
#   - Zero-padding chỉ là nội suy, không tạo thêm thông tin
# ════════════════════════════════════════════════════════════

def plot_compare_nfft(x: np.ndarray, fs: int,
                      nfft_small: int, nfft_large: int,
                      freq_max: float = 6000.0,
                      fname: str = "") -> None:
    """
    Tính FFT cho cùng 1 frame với 2 NFFT khác nhau, vẽ cạnh nhau.
    """
    r_s      = compute_fft(x, fs, nfft=nfft_small)
    r_l      = compute_fft(x, fs, nfft=nfft_large)
    true_res = fs / len(x)    # True resolution = Fs/N – như nhau cho cả 2

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    axes = axes.flatten()

    # Cấu hình vẽ: (dict kết quả, màu linear, màu dB, nfft, vị trí linear, vị trí dB)
    configs = [
        (r_s, "steelblue",  "darkorange", nfft_small, 0, 2),
        (r_l, "seagreen",   "tomato",     nfft_large, 1, 3),
    ]

    for r, c1, c2, nf, col_lin, col_db in configs:
        mask = r["f"] <= freq_max

        # Subplot tuyến tính
        axes[col_lin].plot(r["f"][mask], r["mag_norm"][mask], color=c1, lw=0.6)
        style_ax(axes[col_lin], ylabel="|X(f)|",
                 title=f"NFFT={nf:,} | Δf={r['delta_f']:.3f}Hz (linear)")

        # Subplot dB
        axes[col_db].plot(r["f"][mask], r["mag_dB"][mask], color=c2, lw=0.7)
        style_ax(axes[col_db], xlabel="Tần số (Hz)", ylabel="dBFS",
                 title=f"NFFT={nf:,} | Δf={r['delta_f']:.3f}Hz (dB)")
        axes[col_db].set_ylim([-90, 5])

    for ax in axes:
        ax.set_xlim([0, freq_max])

    # In kết luận quan trọng
    print(f"\n  ── So sánh NFFT ─────────────────────────────────────")
    print(f"  Frame = {len(x):,} mẫu  |  True resolution = {true_res:.4f} Hz (cả 2 giống nhau!)")
    print(f"  NFFT={nfft_small:,}  → Δf={fs/nfft_small:.4f} Hz  (phổ thô, ít điểm)")
    print(f"  NFFT={nfft_large:,}  → Δf={fs/nfft_large:.4f} Hz  (phổ mịn, nhiều điểm)")
    print(f"  ⚠ Zero-padding KHÔNG tăng true resolution; chỉ nội suy trên trục tần số.")

    fig.suptitle(
        f"Phần C – So sánh NFFT: {nfft_small:,} vs {nfft_large:,}\n"
        f"True resolution = Fs/N_window = {true_res:.3f} Hz (không đổi) | {fname}",
        fontsize=11, fontweight="bold",
    )
    plt.tight_layout()
    savefig(fig, "C_compare_nfft.png")


# ════════════════════════════════════════════════════════════
# HÀM CHẠY PHẦN C
# Chọn đoạn ổn định 1s ở giữa file → FFT chuẩn → so sánh NFFT
# ════════════════════════════════════════════════════════════

def run(info: dict = None) -> None:
    if info is None:
        info = read_wav(choose_audio_file())

    x   = info["x_mono"]
    fs  = info["fs"]
    dur = info["duration"]

    print("\n" + "═" * 60)
    print("  PHẦN C – PHÂN TÍCH FFT")
    print("═" * 60)

    # Chọn đoạn ổn định: từ 30% đến 80% tổng duration, tối đa 1 giây
    t_s = dur * 0.3
    t_e = min(t_s + 1.0, dur * 0.8)
    seg = x[int(t_s * fs): int(t_e * fs)]
    print(f"  Đoạn phân tích: {t_s:.3f}s – {t_e:.3f}s  ({len(seg):,} mẫu)")
    print(f"  True resolution = Fs/N = {fs}/{len(seg)} = {fs/len(seg):.4f} Hz")

    freq_max = float(min(8000, fs // 2))

    # C1: FFT chuẩn với NFFT = lũy thừa 2 >= len(seg)
    nfft_std = int(2 ** np.ceil(np.log2(max(len(seg), 2048))))
    r_std    = compute_fft(seg, fs, nfft=nfft_std, window="hamming")
    print(f"\n  [C1] FFT chuẩn (NFFT={nfft_std:,}, Hamming):")
    plot_fft(r_std, info["filename"], freq_max=freq_max,
             save_name="C_fft_standard.png")

    # C2: So sánh NFFT nhỏ và lớn trên cùng frame
    nfft_small = max(256, int(2 ** np.floor(np.log2(len(seg)))) // 2)
    nfft_large = min(262144, nfft_std * 8)
    print(f"\n  [C2] So sánh NFFT={nfft_small:,} vs {nfft_large:,}:")
    plot_compare_nfft(seg, fs, nfft_small, nfft_large,
                      freq_max=freq_max, fname=info["filename"])

    print("\n  ✓ Phần C hoàn thành.\n")


if __name__ == "__main__":
    run()
