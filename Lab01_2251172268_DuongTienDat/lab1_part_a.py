"""
lab1_part_a.py
==============
CSE457 Lab 1 – Phần A: Đọc và kiểm tra dữ liệu âm thanh
---------------------------------------------------------
Yêu cầu:
  - Đọc file WAV, hiển thị đầy đủ metadata
  - Kiểm tra Fs, channels, duration, bit depth, file size, bit rate
  - Nếu stereo → tạo bản mono (trung bình 2 kênh), so sánh waveform + RMS
  - Chuẩn hóa về [-1, 1] (full-scale normalization)
"""

import os
import wave
import numpy as np
import matplotlib.pyplot as plt

from lab1_utils import (
    read_wav, write_wav, choose_audio_file,
    savefig, style_ax, db,
    FIGURES_DIR, AUDIO_DIR
)


# ════════════════════════════════════════════════════════════
# A1 – In bảng metadata đầy đủ ra terminal
# Hiển thị tất cả thông số kỹ thuật của file âm thanh:
#   - Sampling rate và tần số Nyquist
#   - Số kênh, bit depth, số frame
#   - PCM bit rate (lý thuyết) và kích thước file thực tế
#   - Peak và RMS của tín hiệu mono
# ════════════════════════════════════════════════════════════

def print_metadata(info: dict) -> None:
    # Lấy các trường từ dict info do read_wav() trả về
    fs       = info["fs"]
    channels = info["channels"]
    bd       = info["bit_depth"]
    dur      = info["duration"]
    nfr      = info["n_frames"]
    fsz      = info["file_size"]
    fname    = info["filename"]

    # Tính các đại lượng dẫn xuất
    pcm_bitrate  = fs * bd * channels       # Bit rate PCM lý thuyết (bit/s)
    pcm_size_est = pcm_bitrate * dur / 8    # Kích thước PCM lý thuyết (bytes)
    nyquist      = fs / 2                   # Tần số Nyquist = Fs/2

    # In bảng đóng khung
    print("\n" + "╔" + "═" * 56 + "╗")
    print(f"║  METADATA: {fname:<44}║")
    print("╠" + "═" * 56 + "╣")
    print(f"║  Sampling rate  (Fs)  : {fs:>10,} Hz                   ║")
    print(f"║  Nyquist freq         : {nyquist:>10,.1f} Hz                   ║")
    print(f"║  Channels             : {channels:>10}                        ║")
    print(f"║  Bit depth            : {bd:>10} bit/sample               ║")
    print(f"║  Sample width         : {info['sample_width']:>10} bytes/sample            ║")
    print(f"║  Frames               : {nfr:>10,}                        ║")
    print(f"║  Duration             : {dur:>10.4f} s                     ║")
    print(f"║  PCM bit rate         : {pcm_bitrate:>10,} bit/s                ║")
    print(f"║  PCM bit rate         : {pcm_bitrate/1000:>10.1f} kbps                ║")
    print(f"║  PCM size (est.)      : {pcm_size_est/1024/1024:>10.3f} MB                   ║")
    print(f"║  File size (actual)   : {fsz:>10,} bytes                ║")
    print(f"║  File size (actual)   : {fsz/1024/1024:>10.4f} MB                   ║")

    # Compression ratio: PCM lý thuyết / kích thước file thực
    if pcm_size_est > 0:
        ratio = pcm_size_est / fsz
        print(f"║  Comp. ratio (est.)   : {ratio:>10.2f} : 1                  ║")

    # Peak và RMS của kênh mono
    peak = float(np.max(np.abs(info["x_mono"])))
    rms  = float(np.sqrt(np.mean(info["x_mono"] ** 2)))
    print(f"║  Peak amplitude       : {peak:>10.6f}  ({db(peak):>7.2f} dBFS)     ║")
    print(f"║  RMS amplitude        : {rms:>10.6f}  ({db(rms):>7.2f} dBFS)     ║")
    print("╚" + "═" * 56 + "╝")


# ════════════════════════════════════════════════════════════
# A2 – Vẽ waveform theo kênh + so sánh stereo vs mono
# Nếu stereo: vẽ riêng kênh L, kênh R và bản mono (3 subplot)
# Nếu mono  : chỉ vẽ 1 subplot duy nhất
# Mục đích: kiểm tra cân bằng L/R và so sánh RMS
# ════════════════════════════════════════════════════════════

def plot_channels(info: dict) -> None:
    """
    Vẽ waveform từng kênh, kèm đường RMS (màu cam) và ngưỡng clipping (màu đỏ).
    Nếu stereo: in thêm bảng so sánh RMS giữa L, R và mono.
    """
    x_full = info["x_norm"]     # Tín hiệu gốc đã chuẩn hóa (có thể multi-channel)
    x_mono = info["x_mono"]     # Tín hiệu mono
    fs     = info["fs"]
    ch     = info["channels"]
    fname  = info["filename"]
    t      = np.arange(len(x_mono)) / fs   # Trục thời gian (giây)

    # Tạo figure với số hàng subplot = số kênh + 1 (nếu stereo)
    n_rows = ch + (1 if ch > 1 else 0)
    fig, axes = plt.subplots(n_rows, 1, figsize=(13, 2.8 * n_rows), sharex=True)
    if n_rows == 1:
        axes = [axes]

    colors = ["steelblue", "tomato", "seagreen"]
    labels = ["Kênh trái (L)", "Kênh phải (R)"]

    # Vẽ từng kênh (L và R nếu stereo)
    for i in range(ch):
        sig_i = x_full[:, i] if ch > 1 else x_full.flatten()
        rms_i = float(np.sqrt(np.mean(sig_i ** 2)))
        axes[i].plot(t, sig_i, color=colors[i], lw=0.35, alpha=0.9)
        # Đường RMS nằm ngang
        axes[i].axhline( rms_i, color="orange", lw=0.9, ls="-.",
                         label=f"RMS = {rms_i:.5f}  ({db(rms_i):.1f} dBFS)")
        axes[i].axhline(-rms_i, color="orange", lw=0.9, ls="-.")
        # Đường clipping ±1
        axes[i].axhline( 1.0,   color="red",    lw=0.6, ls="--", alpha=0.6)
        axes[i].axhline(-1.0,   color="red",    lw=0.6, ls="--", alpha=0.6)
        style_ax(axes[i], ylabel="Biên độ",
                 title=labels[i] if ch > 1 else f"Waveform – {fname}")
        axes[i].legend(fontsize=8, loc="upper right")
        axes[i].set_ylim([-1.15, 1.15])

    # Nếu stereo: thêm subplot mono ở cuối và so sánh RMS
    if ch > 1:
        rms_m = float(np.sqrt(np.mean(x_mono ** 2)))
        rms_l = float(np.sqrt(np.mean(x_full[:, 0] ** 2)))
        rms_r = float(np.sqrt(np.mean(x_full[:, 1] ** 2)))
        axes[-1].plot(t, x_mono, color="seagreen", lw=0.35)
        axes[-1].axhline( rms_m, color="orange", lw=0.9, ls="-.",
                          label=f"RMS_mono = {rms_m:.5f}  ({db(rms_m):.1f} dBFS)")
        axes[-1].axhline(-rms_m, color="orange", lw=0.9, ls="-.")
        style_ax(axes[-1], xlabel="Thời gian (s)", ylabel="Biên độ",
                 title=f"Mono (mean L+R) | RMS_L={db(rms_l):.1f}  RMS_R={db(rms_r):.1f}  RMS_M={db(rms_m):.1f} dBFS")
        axes[-1].legend(fontsize=8, loc="upper right")
        axes[-1].set_ylim([-1.15, 1.15])

        # In bảng so sánh RMS
        print("\n  ── So sánh RMS các kênh ──────────────────────────")
        print(f"  RMS kênh L : {rms_l:.6f}  ({db(rms_l):.2f} dBFS)")
        print(f"  RMS kênh R : {rms_r:.6f}  ({db(rms_r):.2f} dBFS)")
        print(f"  RMS mono   : {rms_m:.6f}  ({db(rms_m):.2f} dBFS)")
        diff_lr = abs(rms_l - rms_r) / max(rms_l, rms_r) * 100
        print(f"  Chênh lệch L/R: {diff_lr:.2f}%")
    else:
        axes[-1].set_xlabel("Thời gian (s)")

    fig.suptitle(f"Phần A – Kiểm tra kênh audio: {fname}",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    savefig(fig, "A_channels.png")


# ════════════════════════════════════════════════════════════
# A3 – So sánh 2 phương pháp chuẩn hóa biên độ
# Mục đích: giúp sinh viên hiểu sự khác nhau giữa:
#   - Full-scale norm: Peak luôn = 0 dBFS, nhưng RMS tuỳ tín hiệu
#   - RMS norm: RMS cố định tại target (ví dụ -20 dBFS), nhưng Peak có thể vượt 0 dBFS
# ════════════════════════════════════════════════════════════

def plot_normalization(info: dict) -> None:
    """
    Vẽ 3 subplot so sánh:
      1. Tín hiệu gốc (chưa chuẩn hóa)
      2. Full-scale normalization: x_fs = x / max|x|  → Peak = 0 dBFS
      3. RMS normalization: target RMS = -20 dBFS (RMS = 0.1)
    """
    x_raw = info["x_mono"]
    fs    = info["fs"]
    t     = np.arange(len(x_raw)) / fs

    # Phương pháp 1: Full-scale normalization
    # Chia tất cả mẫu cho giá trị peak lớn nhất → peak sẽ là ±1.0
    peak = np.max(np.abs(x_raw))
    x_fs = x_raw / peak if peak > 0 else x_raw.copy()

    # Phương pháp 2: RMS normalization → target -20 dBFS (RMS = 0.1)
    # Nhân tín hiệu với (target_rms / rms_gốc) để đạt mức RMS mong muốn
    target_rms = 0.1    # -20 dBFS
    rms_raw    = np.sqrt(np.mean(x_raw ** 2))
    x_rms      = x_raw * (target_rms / rms_raw) if rms_raw > 0 else x_raw.copy()
    x_rms      = np.clip(x_rms, -1.0, 1.0)   # Clip nếu peak vượt 0 dBFS

    # Vẽ 3 subplot với cùng trục x
    fig, axes = plt.subplots(3, 1, figsize=(13, 8), sharex=True)
    plots = [
        (x_raw, "steelblue",  f"Gốc  | Peak={db(np.max(np.abs(x_raw))):.1f} dBFS  RMS={db(np.sqrt(np.mean(x_raw**2))):.1f} dBFS"),
        (x_fs,  "darkorange", f"Full-scale norm | Peak=0 dBFS  RMS={db(np.sqrt(np.mean(x_fs**2))):.1f} dBFS"),
        (x_rms, "seagreen",   f"RMS norm (-20 dBFS target) | Peak={db(np.max(np.abs(x_rms))):.1f} dBFS"),
    ]
    for ax, (sig, clr, ttl) in zip(axes, plots):
        ax.plot(t, sig, color=clr, lw=0.35)
        rms = np.sqrt(np.mean(sig ** 2))
        ax.axhline( rms, color="orange", lw=0.9, ls="-.", label=f"RMS={rms:.4f}")
        ax.axhline(-rms, color="orange", lw=0.9, ls="-.")
        ax.axhline( 1.0, color="red",    lw=0.6, ls="--", alpha=0.5)  # Ngưỡng clipping trên
        ax.axhline(-1.0, color="red",    lw=0.6, ls="--", alpha=0.5)  # Ngưỡng clipping dưới
        style_ax(ax, ylabel="Biên độ", title=ttl)
        ax.legend(fontsize=8, loc="upper right")
        ax.set_ylim([-1.15, 1.15])

    axes[-1].set_xlabel("Thời gian (s)")
    fig.suptitle("Phần A – So sánh phương pháp chuẩn hóa",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    savefig(fig, "A_normalization.png")


# ════════════════════════════════════════════════════════════
# HÀM CHẠY PHẦN A
# Gọi lần lượt: chọn file → đọc → in metadata → vẽ hình
# Trả về info dict để phần B–G dùng tiếp
# ════════════════════════════════════════════════════════════

def run(filepath: str = None) -> dict:
    """
    Chạy toàn bộ phần A.
    filepath: nếu None → hiện menu chọn file.
    Trả về dict info để truyền sang phần B, C, ... G.
    """
    if filepath is None:
        filepath = choose_audio_file()

    print(f"\n  Đang đọc: {filepath}")
    info = read_wav(filepath)    # Đọc và chuẩn hóa tín hiệu
    print_metadata(info)         # In bảng thông số kỹ thuật
    plot_channels(info)          # Vẽ waveform theo kênh
    plot_normalization(info)     # So sánh các phương pháp chuẩn hóa
    print("\n  ✓ Phần A hoàn thành.\n")
    return info


if __name__ == "__main__":
    run()
