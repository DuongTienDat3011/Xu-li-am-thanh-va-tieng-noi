"""
lab1_part_d.py
==============
CSE457 Lab 1 – Phần D: STFT và Spectrogram
-------------------------------------------
Yêu cầu:
  - Spectrogram chuẩn: frame≈25ms, hop≈10ms, cửa sổ Hamming
  - So sánh 3 frame length: 10ms / 25ms / 50ms
  - Giải thích time–frequency resolution trade-off
  - Phát hiện vùng năng lượng ổn định và transient
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.signal import spectrogram as _spectrogram

from lab1_utils import (
    read_wav, choose_audio_file, savefig, style_ax,
    FIGURES_DIR
)


# ════════════════════════════════════════════════════════════
# D1 – Tính STFT / Spectrogram
# STFT (Short-Time Fourier Transform) chia tín hiệu thành các frame
# rồi tính FFT cho từng frame, tạo ra biểu diễn 2D: thời gian × tần số
#
# Công thức: X[m,k] = Σ x[n] w[n−mH] e^(−j2πkn/NFFT)
#   m: chỉ số frame theo thời gian
#   k: chỉ số bin tần số
#   H: hop size (bước nhảy)
#   w: cửa sổ (Hamming)
# ════════════════════════════════════════════════════════════

def compute_stft(x: np.ndarray, fs: int,
                 frame_ms: float = 25.0,
                 hop_ms:   float = 10.0,
                 nfft:     int   = None,
                 window:   str   = "hamming") -> dict:
    """
    Tính spectrogram (STFT magnitude) bằng scipy.signal.spectrogram.

    Parameters:
        frame_ms : độ dài frame (ms) – ảnh hưởng đến Δf
        hop_ms   : bước nhảy (ms) – ảnh hưởng đến Δt
        nfft     : số điểm FFT (zero-padding nếu > nperseg)
        window   : loại cửa sổ

    Trade-off quan trọng:
        Frame dài  → nhiều mẫu/frame → Δf nhỏ  (tần số tốt) | Δt lớn  (thời gian kém)
        Frame ngắn → ít mẫu/frame   → Δf lớn  (tần số kém)  | Δt nhỏ  (thời gian tốt)
    """
    nperseg  = round(frame_ms * 1e-3 * fs)    # Số mẫu mỗi frame
    hop_n    = round(hop_ms   * 1e-3 * fs)    # Số mẫu bước nhảy
    noverlap = nperseg - hop_n                 # Số mẫu overlap giữa các frame
    noverlap = max(0, min(noverlap, nperseg - 1))   # Đảm bảo hợp lệ

    # NFFT: tăng lên lũy thừa 2 để FFT nhanh hơn
    if nfft is None:
        nfft = int(2 ** np.ceil(np.log2(max(nperseg, 32))))

    # Gọi scipy.signal.spectrogram với mode='magnitude'
    f, t, Sxx = _spectrogram(
        x, fs=fs,
        window   = window,
        nperseg  = nperseg,
        noverlap = noverlap,
        nfft     = nfft,
        scaling  = "spectrum",
        mode     = "magnitude",
    )

    # Chuyển sang dB (tránh log(0) bằng clip dưới)
    S_dB       = 20 * np.log10(np.maximum(Sxx, 1e-10))
    S_power_dB = 10 * np.log10(np.maximum(Sxx ** 2, 1e-20))

    return dict(
        f          = f,
        t          = t,
        S_mag      = Sxx,
        S_dB       = S_dB,
        S_power_dB = S_power_dB,
        nperseg    = nperseg,           # Số mẫu/frame
        noverlap   = noverlap,          # Số mẫu overlap
        hop_n      = hop_n,             # Số mẫu hop
        nfft       = nfft,
        frame_ms   = frame_ms,
        hop_ms     = hop_ms,
        delta_f    = fs / nfft,         # Δf = Fs/NFFT (Hz/bin)
        true_res   = fs / nperseg,      # Độ phân giải tần số thực = Fs/N_frame
        delta_t_ms = hop_n / fs * 1000, # Δt theo ms
        overlap_pct= noverlap / nperseg * 100,
        window     = window,
        n_frames   = Sxx.shape[1],
        n_bins     = Sxx.shape[0],
    )


# ════════════════════════════════════════════════════════════
# D2 – Vẽ spectrogram 2D (thời gian × tần số)
# Trục X: thời gian (giây)
# Trục Y: tần số (Hz)
# Màu sắc: cường độ âm thanh (dB)
# Dùng colormap 'inferno': đen=yên tĩnh, đỏ/vàng=năng lượng cao
# ════════════════════════════════════════════════════════════

def plot_spectrogram(sg: dict, title: str = "",
                     freq_max: float = None,
                     vmin_dB: float = -90, vmax_dB: float = 0,
                     save_name: str = "D_spectrogram.png",
                     cmap: str = "inferno") -> None:
    """
    Vẽ spectrogram 2D với color bar, in thông số cấu hình.
    vmin_dB/vmax_dB: giới hạn dải động màu sắc (dB).
    """
    f = sg["f"]
    t = sg["t"]
    S = sg["S_dB"]

    if freq_max is None:
        freq_max = min(8000.0, float(f[-1]))

    fmask = f <= freq_max   # Chỉ hiển thị đến freq_max

    fig, ax = plt.subplots(figsize=(13, 5))
    # pcolormesh vẽ heatmap; shading='gouraud' làm mịn màu
    im = ax.pcolormesh(t, f[fmask], S[fmask, :],
                       shading="gouraud",
                       cmap=cmap,
                       vmin=vmin_dB, vmax=vmax_dB)
    cbar = plt.colorbar(im, ax=ax, pad=0.01)
    cbar.set_label("Biên độ (dB)", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    # Format trục tần số: hiển thị kHz nếu >= 1000 Hz
    style_ax(ax, xlabel="Thời gian (s)", ylabel="Tần số (Hz)",
             title=f"{title}\n"
                   f"Frame={sg['frame_ms']:.0f}ms | Hop={sg['hop_ms']:.0f}ms | "
                   f"Δf={sg['delta_f']:.2f}Hz | Δt={sg['delta_t_ms']:.1f}ms | "
                   f"Overlap={sg['overlap_pct']:.0f}%")
    ax.set_ylim([0, freq_max])
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(
        lambda v, _: f"{v/1000:.1f}kHz" if v >= 1000 else f"{v:.0f}Hz"
    ))
    plt.tight_layout()
    savefig(fig, save_name)

    # In thông số để sinh viên kiểm tra
    print(f"  Spectrogram ({sg['frame_ms']:.0f}ms/{sg['hop_ms']:.0f}ms):")
    print(f"    nperseg     = {sg['nperseg']} mẫu")
    print(f"    hop         = {sg['hop_n']} mẫu")
    print(f"    NFFT        = {sg['nfft']}")
    print(f"    Δf          = {sg['delta_f']:.4f} Hz")
    print(f"    True resol. = {sg['true_res']:.4f} Hz")
    print(f"    Δt          = {sg['delta_t_ms']:.2f} ms")
    print(f"    Shape       = {sg['n_bins']} bins × {sg['n_frames']} frames")


# ════════════════════════════════════════════════════════════
# D3 – So sánh 3 frame length: 10ms / 25ms / 50ms
# Hop cố định = 10ms để chỉ thay đổi frame length
# Quan sát:
#   - Frame 10ms: thấy rõ các biến đổi nhanh (transient) nhưng tần số mờ
#   - Frame 25ms: cân bằng (cấu hình thực hành chuẩn)
#   - Frame 50ms: thấy rõ harmonic series nhưng transient bị làm mờ
# ════════════════════════════════════════════════════════════

def plot_compare_frames(x: np.ndarray, fs: int,
                        frame_lengths_ms: list = None,
                        freq_max: float = None,
                        fname: str = "") -> None:
    if frame_lengths_ms is None:
        frame_lengths_ms = [10.0, 25.0, 50.0]

    if freq_max is None:
        freq_max = min(8000.0, fs / 2)

    n   = len(frame_lengths_ms)
    fig, axes = plt.subplots(n, 1, figsize=(13, 4 * n), sharex=True)
    if n == 1:
        axes = [axes]

    summary = []
    for ax, fl in zip(axes, frame_lengths_ms):
        # Hop cố định 10ms để so sánh công bằng về Δt
        sg    = compute_stft(x, fs, frame_ms=fl, hop_ms=10.0)
        fmask = sg["f"] <= freq_max

        im = ax.pcolormesh(sg["t"], sg["f"][fmask], sg["S_dB"][fmask, :],
                           shading="gouraud", cmap="magma",
                           vmin=-90, vmax=0)
        plt.colorbar(im, ax=ax, label="dB", pad=0.01).ax.tick_params(labelsize=7)
        ax.set_ylabel("Tần số (Hz)", fontsize=9)
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(
            lambda v, _: f"{int(v/1000)}kHz" if v >= 1000 else f"{int(v)}Hz"
        ))
        ax.set_title(
            f"Frame = {fl:.0f} ms  |  Δt = {sg['delta_t_ms']:.1f} ms  |  "
            f"True Δf = {sg['true_res']:.2f} Hz  |  NFFT-Δf = {sg['delta_f']:.2f} Hz",
            fontsize=9, fontweight="bold",
        )
        ax.set_ylim([0, freq_max])
        summary.append((fl, sg["delta_t_ms"], sg["true_res"], sg["delta_f"]))

    axes[-1].set_xlabel("Thời gian (s)", fontsize=9)

    fig.suptitle(
        f"Phần D – Trade-off Time/Frequency Resolution  |  {fname}\n"
        "Hop = 10ms cố định. Frame ngắn → Δt tốt, Δf kém. Frame dài → Δf tốt, Δt kém.",
        fontsize=10, fontweight="bold",
    )
    plt.tight_layout()
    savefig(fig, "D_compare_frames.png")

    # In bảng tóm tắt trade-off
    print("\n  ┌──────────┬───────────┬────────────┬──────────────┐")
    print("  │ Frame(ms)│ Δt (ms)   │ True Δf(Hz)│ NFFT Δf(Hz)  │")
    print("  ├──────────┼───────────┼────────────┼──────────────┤")
    for fl, dt, tdf, ndf in summary:
        print(f"  │ {fl:>7.1f}  │ {dt:>8.2f}  │ {tdf:>9.3f}  │ {ndf:>11.3f}  │")
    print("  └──────────┴───────────┴────────────┴──────────────┘")
    print("  → Frame ngắn: Δt nhỏ ↔ phân giải thời gian tốt → rõ transient")
    print("  → Frame dài : Δf nhỏ ↔ phân giải tần số tốt   → rõ harmonic")


# ════════════════════════════════════════════════════════════
# D4 – Spectrogram có annotation vùng transient
# Phát hiện transient bằng cách tính:
#   1. Energy mỗi frame = tổng |S[f,t]|² theo trục tần số
#   2. Làm mịn energy bằng bộ lọc trung bình động
#   3. Tính đạo hàm energy (biến đổi đột ngột = transient)
#   4. Đánh dấu bằng đường dọc cyan trên spectrogram
# ════════════════════════════════════════════════════════════

def plot_spectrogram_annotated(x: np.ndarray, fs: int, fname: str = "") -> None:
    """
    Spectrogram 25ms/5ms với annotation phát hiện transient.
    Subplot dưới: biểu đồ energy/frame và ngưỡng phát hiện.
    """
    sg      = compute_stft(x, fs, frame_ms=25, hop_ms=5)
    freq_max = min(8000.0, float(sg["f"][-1]))
    fmask   = sg["f"] <= freq_max

    # Tính energy mỗi frame: tổng theo trục tần số
    energy_per_frame = np.sum(sg["S_mag"][fmask, :] ** 2, axis=0)

    # Làm mịn energy bằng moving average 5 frame
    energy_smooth = np.convolve(energy_per_frame, np.ones(5) / 5, mode="same")

    # Đạo hàm bậc 1: biến đổi đột ngột = transient
    energy_diff = np.abs(np.diff(energy_smooth, prepend=energy_smooth[0]))

    # Ngưỡng: percentile 92% → chỉ bắt các thay đổi lớn nhất
    threshold       = np.percentile(energy_diff, 92)
    transient_frames = sg["t"][energy_diff > threshold]

    fig, axes = plt.subplots(2, 1, figsize=(13, 8),
                              gridspec_kw={"height_ratios": [3, 1]})

    # Vẽ spectrogram chính
    im = axes[0].pcolormesh(sg["t"], sg["f"][fmask], sg["S_dB"][fmask, :],
                             shading="gouraud", cmap="inferno",
                             vmin=-90, vmax=0)
    plt.colorbar(im, ax=axes[0], label="dB", pad=0.01)

    # Đánh dấu tối đa 15 vị trí transient bằng đường đứng màu cyan
    for tf in transient_frames[:15]:
        axes[0].axvline(tf, color="cyan", lw=0.7, alpha=0.6)

    axes[0].set_ylabel("Tần số (Hz)", fontsize=9)
    axes[0].set_title(
        f"Phần D – Spectrogram (25ms/5ms) + Transient detection  |  {fname}",
        fontsize=10, fontweight="bold",
    )
    axes[0].set_ylim([0, freq_max])

    # Subplot dưới: energy/frame và ngưỡng
    axes[1].fill_between(sg["t"],
                          energy_per_frame / (np.max(energy_per_frame) + 1e-9),
                          alpha=0.6, color="steelblue", label="Energy/frame")
    axes[1].axhline(threshold / (np.max(energy_per_frame) + 1e-9),
                    color="red", lw=0.8, ls="--", label="Transient threshold")
    style_ax(axes[1], xlabel="Thời gian (s)", ylabel="Năng lượng (norm)",
             title="Short-time energy per frame")
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    savefig(fig, "D_spectrogram_annotated.png")


# ════════════════════════════════════════════════════════════
# HÀM CHẠY PHẦN D
# Cắt 10s để vẽ nhanh → 3 bước: standard, compare frames, annotated
# ════════════════════════════════════════════════════════════

def run(info: dict = None) -> None:
    if info is None:
        info = read_wav(choose_audio_file())

    x   = info["x_mono"]
    fs  = info["fs"]
    dur = info["duration"]

    print("\n" + "═" * 60)
    print("  PHẦN D – STFT VÀ SPECTROGRAM")
    print("═" * 60)

    # Cắt tối đa 10s bắt đầu từ 10% duration để vẽ nhanh
    t_s = dur * 0.1
    t_e = min(t_s + 10.0, dur)
    seg = x[int(t_s * fs): int(t_e * fs)]
    freq_max = float(min(8000, fs // 2))

    print(f"  Đoạn phân tích: {t_s:.2f}s – {t_e:.2f}s  ({len(seg):,} mẫu)")

    # D1: Spectrogram chuẩn (frame=25ms, hop=10ms)
    print("\n  [D1] Spectrogram chuẩn 25ms/10ms...")
    sg_std = compute_stft(seg, fs, frame_ms=25, hop_ms=10)
    plot_spectrogram(sg_std,
                     title=f"Spectrogram chuẩn: {info['filename']}",
                     freq_max=freq_max,
                     save_name="D_spectrogram_standard.png")

    # D2: So sánh 3 frame length
    print("\n  [D2] So sánh 3 frame length (10/25/50 ms)...")
    plot_compare_frames(seg, fs,
                        frame_lengths_ms=[10.0, 25.0, 50.0],
                        freq_max=freq_max,
                        fname=info["filename"])

    # D3: Spectrogram với annotation transient
    print("\n  [D3] Spectrogram với annotation transient...")
    plot_spectrogram_annotated(seg, fs, fname=info["filename"])

    print("\n  ✓ Phần D hoàn thành.\n")


if __name__ == "__main__":
    run()
