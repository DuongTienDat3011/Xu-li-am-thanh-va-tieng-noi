"""
lab1_part_e.py
==============
CSE457 Lab 1 – Phần E: Thí nghiệm cửa sổ
------------------------------------------
Yêu cầu:
  - So sánh Rectangular và Hamming trên cùng một frame (cùng NFFT)
  - Vẽ log-spectrum, đo main-lobe width và side-lobe level
  - Phân tích spectral leakage
  - Mở rộng: thêm Hann và Blackman
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from lab1_utils import (
    read_wav, choose_audio_file, savefig, style_ax, db,
    FIGURES_DIR
)


# ════════════════════════════════════════════════════════════
# E1 – Thư viện cửa sổ
# Cửa sổ (window function) là hàm nhân với tín hiệu trước FFT
# để giảm spectral leakage (rò phổ).
# Leakage xảy ra vì FFT giả định tín hiệu là tuần hoàn trong frame,
# nhưng thực tế thường không phải vậy → cắt đột ngột ở biên frame
# ════════════════════════════════════════════════════════════

def get_window(name: str, N: int) -> np.ndarray:
    """
    Tạo mảng cửa sổ độ dài N theo tên.
    Hỗ trợ: rectangular, hamming, hann, blackman, bartlett.
    """
    name = name.lower()
    if name in ("rectangular", "rect", "boxcar"):
        return np.ones(N)           # Cửa sổ chữ nhật: tất cả hệ số = 1
    elif name == "hamming":
        return np.hamming(N)        # w[n] = 0.54 - 0.46·cos(2πn/(N-1))
    elif name in ("hann", "hanning"):
        return np.hanning(N)        # w[n] = 0.5·(1 - cos(2πn/(N-1)))
    elif name == "blackman":
        return np.blackman(N)       # Cửa sổ Blackman: side-lobe thấp hơn
    elif name == "bartlett":
        return np.bartlett(N)       # Cửa sổ tam giác
    else:
        raise ValueError(f"Cửa sổ '{name}' không nhận ra.")


def window_properties(w: np.ndarray, fs: int = 1, nfft: int = 8192) -> dict:
    """
    Đo các thuộc tính kỹ thuật của cửa sổ trong miền tần số:

    Coherent Gain (CG): trung bình các hệ số cửa sổ
      → Ảnh hưởng đến biên độ phổ (cần chia khi normalize)

    ENBW (Equivalent Noise BandWidth): băng thông nhiễu tương đương
      = incoherent_gain / CG²  (đo bằng bins)
      → ENBW nhỏ = bộ lọc hẹp = phân tách tần số tốt hơn

    Main-lobe width: số bins từ DC đến điểm -3dB đầu tiên
      → Hẹp = phân tách 2 tần số gần nhau tốt hơn

    Side-lobe level (dB): mức suy giảm của thùy phụ so với thùy chính
      → Thấp = spectral leakage ít
    """
    N  = len(w)
    cg = float(np.mean(w))          # Coherent gain
    ig = float(np.mean(w ** 2))     # Incoherent gain (power gain)

    # ENBW tính theo bins
    enbw_bins = ig / (cg ** 2)

    # Tính phổ của cửa sổ (normalized)
    W     = np.fft.rfft(w / N, n=nfft)
    W_db  = 20 * np.log10(np.abs(W) + 1e-15)
    freqs = np.arange(len(W))

    # Tìm đỉnh chính (main lobe peak)
    peak_idx = int(np.argmax(W_db))
    peak_db  = float(W_db[peak_idx])

    # Đo main-lobe width: đi từ peak ra hai phía đến điểm < peak - 3dB
    left = peak_idx
    while left > 0 and W_db[left] > peak_db - 3:
        left -= 1
    right = peak_idx
    while right < len(W_db) - 1 and W_db[right] > peak_db - 3:
        right += 1
    main_lobe_bins = right - left

    # Đo side-lobe: tìm max ở ngoài vùng main-lobe (margin ±5 bins)
    side = W_db.copy()
    side[max(0, left - 5): right + 5] = -200   # Che main-lobe
    first_sl_db = float(np.max(side)) - peak_db  # Tương đối so với peak

    return dict(
        coherent_gain      = cg,
        incoherent_gain    = ig,
        enbw_bins          = enbw_bins,
        main_lobe_bins     = main_lobe_bins,
        first_side_lobe_dB = first_sl_db,
        peak_dB            = peak_db,
        W_db               = W_db,
        freqs              = freqs,
        nfft               = nfft,
    )


# ════════════════════════════════════════════════════════════
# E2 – Vẽ tổng quan 4 cửa sổ phổ biến
# Hàng trên: dạng sóng cửa sổ w[n] trong miền thời gian
# Hàng dưới: đáp ứng tần số |W(f)| trong miền tần số (dB)
# ════════════════════════════════════════════════════════════

def plot_window_overview(N: int = 1024) -> None:
    """
    Vẽ và đo 4 cửa sổ: Rectangular, Hamming, Hann, Blackman.
    In bảng so sánh thuộc tính: CG, ENBW, Main-lobe, Side-lobe.
    """
    windows = ["rectangular", "hamming", "hann", "blackman"]
    colors  = ["steelblue", "darkorange", "seagreen", "tomato"]
    nfft_w  = 8192   # NFFT lớn để đáp ứng tần số mịn

    fig, axes = plt.subplots(2, 4, figsize=(16, 7))

    print("\n  Thuộc tính cửa sổ (N={}):".format(N))
    print(f"  {'Cửa sổ':<12} {'CG':>8} {'ENBW(bins)':>12} {'Main-lobe(bins)':>17} {'Side-lobe(dB)':>14}")
    print("  " + "─" * 65)

    for col, (wname, clr) in enumerate(zip(windows, colors)):
        w    = get_window(wname, N)
        prop = window_properties(w, nfft=nfft_w)
        t_n  = np.arange(N)

        print(f"  {wname:<12} {prop['coherent_gain']:>8.4f} {prop['enbw_bins']:>12.3f} "
              f"{prop['main_lobe_bins']:>17d} {prop['first_side_lobe_dB']:>14.1f}")

        # Vẽ dạng sóng thời gian
        axes[0, col].plot(t_n, w, color=clr, lw=1.0)
        axes[0, col].set_title(f"{wname.capitalize()}", fontweight="bold")
        axes[0, col].set_ylim([-0.1, 1.15])
        style_ax(axes[0, col], xlabel="Mẫu n", ylabel="w[n]")

        # Vẽ đáp ứng tần số (chỉ hiển thị phần main-lobe và một vài side-lobe đầu)
        f_disp = prop["freqs"][:nfft_w // 32]
        W_disp = prop["W_db"][:nfft_w // 32]
        axes[1, col].plot(f_disp, W_disp, color=clr, lw=0.8)
        axes[1, col].axhline(-3,  color="gray",  lw=0.6, ls="--")   # -3 dB
        axes[1, col].axhline(-13, color="black", lw=0.5, ls=":", alpha=0.5)   # Ref
        axes[1, col].axhline(-43, color="black", lw=0.5, ls=":", alpha=0.5)
        axes[1, col].set_ylim([-100, 5])
        style_ax(axes[1, col], xlabel="Bin (norm.)", ylabel="|W| (dB)")
        axes[1, col].set_title(
            f"SL={prop['first_side_lobe_dB']:.1f}dB | ML={prop['main_lobe_bins']}bins",
            fontsize=8
        )

    fig.suptitle(f"Phần E – Hình dạng và đáp ứng tần số các cửa sổ (N={N})",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    savefig(fig, "E_window_overview.png")


# ════════════════════════════════════════════════════════════
# E3 – So sánh Rectangular vs Hamming trên frame dữ liệu THỰC
# Điều kiện so sánh công bằng: cùng frame, cùng NFFT
# Vẽ 3 hàng:
#   Hàng 0: frame sau khi nhân cửa sổ (time domain)
#   Hàng 1: phổ dB riêng từng cửa sổ
#   Hàng 2a: chồng 2 phổ lên nhau để so sánh trực tiếp
#   Hàng 2b: biểu đồ chênh lệch (Rectangular − Hamming) dB
# ════════════════════════════════════════════════════════════

def plot_window_on_frame(x_frame: np.ndarray, fs: int,
                         nfft: int,
                         freq_max: float = 6000.0,
                         fname: str = "") -> None:
    """
    So sánh Rectangular và Hamming trên cùng 1 frame dữ liệu thực.
    Kết quả cho thấy Hamming giảm side-lobe nhưng main-lobe rộng hơn.
    """
    N        = len(x_frame)
    win_rect = get_window("rectangular", N)
    win_ham  = get_window("hamming",     N)

    windows = {"Rectangular": win_rect, "Hamming": win_ham}
    colors  = {"Rectangular": "steelblue", "Hamming": "darkorange"}

    # Tính phổ cho từng cửa sổ (với NFFT cố định → so sánh công bằng)
    results = {}
    for wname, w in windows.items():
        xw    = x_frame * w                    # Nhân cửa sổ vào tín hiệu
        X     = np.fft.rfft(xw, n=nfft)       # Tính FFT
        cg    = float(np.mean(w))              # Coherent gain để normalize
        mag   = np.abs(X) / (N * cg)
        mag[1:-1] *= 2.0                       # One-sided correction
        f_ax  = np.fft.rfftfreq(nfft, 1.0 / fs)
        mag_dB = 20 * np.log10(mag + 1e-15)
        results[wname] = dict(f=f_ax, xw=xw, mag_dB=mag_dB, w=w)

    t_ms = np.arange(N) / fs * 1000    # Trục thời gian tính bằng ms
    fig  = plt.figure(figsize=(14, 12))
    gs   = gridspec.GridSpec(3, 2, hspace=0.4, wspace=0.3)

    # ── Hàng 0: frame sau nhân cửa sổ ──
    for col, (wname, res) in enumerate(results.items()):
        ax = fig.add_subplot(gs[0, col])
        ax.plot(t_ms, res["xw"], color=colors[wname], lw=0.6)
        # Vẽ hình dạng cửa sổ (scaled để thấy biên)
        ax.plot(t_ms, res["w"] * np.max(np.abs(x_frame)),
                color="gray", lw=0.7, ls="--", alpha=0.5, label="Cửa sổ (scaled)")
        style_ax(ax, xlabel="ms", ylabel="Biên độ",
                 title=f"Frame × {wname}")
        ax.legend(fontsize=7)

    # ── Hàng 1: phổ dB riêng từng cửa sổ ──
    for col, (wname, res) in enumerate(results.items()):
        ax   = fig.add_subplot(gs[1, col])
        mask = res["f"] <= freq_max
        ax.plot(res["f"][mask], res["mag_dB"][mask],
                color=colors[wname], lw=0.7)
        ax.set_ylim([-90, 5])
        style_ax(ax, xlabel="Hz", ylabel="dBFS", title=f"Phổ dB – {wname}")

        # Tự động đo và vẽ side-lobe
        peak_idx = int(np.argmax(res["mag_dB"][mask]))
        sl_arr   = res["mag_dB"][mask].copy()
        sl_arr[max(0, peak_idx-10): peak_idx+10] = -200   # Che main-lobe
        sl_v = float(np.max(sl_arr))
        ax.axhline(sl_v, color="red", lw=0.6, ls="--",
                   label=f"Side-lobe ≈ {sl_v:.1f}dB")
        ax.legend(fontsize=7)

    # ── Hàng 2a: chồng 2 phổ ──
    ax_overlay = fig.add_subplot(gs[2, 0])
    for wname, res in results.items():
        mask = res["f"] <= freq_max
        ax_overlay.plot(res["f"][mask], res["mag_dB"][mask],
                        color=colors[wname], lw=0.7, alpha=0.85, label=wname)
    ax_overlay.set_ylim([-90, 5])
    style_ax(ax_overlay, xlabel="Hz", ylabel="dBFS",
             title="So sánh Rectangular vs Hamming")
    ax_overlay.legend(fontsize=8)

    # ── Hàng 2b: chênh lệch dB ──
    ax_diff = fig.add_subplot(gs[2, 1])
    f_r  = results["Rectangular"]["f"]
    mask = f_r <= freq_max
    diff = results["Rectangular"]["mag_dB"][mask] - results["Hamming"]["mag_dB"][mask]
    # Màu xanh: vùng Rectangular > Hamming (leakage nhiều hơn)
    ax_diff.fill_between(f_r[mask], diff, 0, where=diff > 0,
                         color="steelblue", alpha=0.5, label="Rect > Ham")
    # Màu cam: vùng Hamming > Rectangular (thường ở main-lobe)
    ax_diff.fill_between(f_r[mask], diff, 0, where=diff < 0,
                         color="darkorange", alpha=0.5, label="Ham > Rect")
    ax_diff.axhline(0, color="black", lw=0.5)
    style_ax(ax_diff, xlabel="Hz", ylabel="dB",
             title="Chênh lệch (Rectangular − Hamming)")
    ax_diff.legend(fontsize=8)

    fig.suptitle(
        f"Phần E – So sánh cửa sổ trên frame thực\n"
        f"N={N} mẫu | NFFT={nfft:,} | Δf={fs/nfft:.3f}Hz | {fname}",
        fontsize=11, fontweight="bold",
    )
    savefig(fig, "E_window_frame_comparison.png")

    # In bảng đặc tính so sánh
    print("\n  ── Đặc tính so sánh ────────────────────────────────")
    print(f"  {'Cửa sổ':<14} {'Coherent Gain':>14} {'Main-lobe(bins)':>16} {'Side-lobe(dB)':>14}")
    print("  " + "─" * 60)
    for wname, w in windows.items():
        prop = window_properties(w, nfft=nfft)
        print(f"  {wname:<14} {prop['coherent_gain']:>14.4f} {prop['main_lobe_bins']:>16d} "
              f"{prop['first_side_lobe_dB']:>14.1f}")
    print("\n  Nhận xét kỹ thuật:")
    print("    - Rectangular: main-lobe hẹp → phân tách 2 tần số gần nhau tốt hơn")
    print("    - Hamming    : side-lobe thấp hơn ~42dB → spectral leakage ít hơn nhiều")
    print("    - Đánh đổi   : Hamming có main-lobe rộng hơn ~2× → khó tách peak gần nhau")


# ════════════════════════════════════════════════════════════
# HÀM CHẠY PHẦN E
# E1: tổng quan 4 cửa sổ trên N=1024 mẫu lý thuyết
# E2: so sánh Rectangular vs Hamming trên frame dữ liệu thực (25ms)
# ════════════════════════════════════════════════════════════

def run(info: dict = None) -> None:
    if info is None:
        info = read_wav(choose_audio_file())

    x   = info["x_mono"]
    fs  = info["fs"]
    dur = info["duration"]

    print("\n" + "═" * 60)
    print("  PHẦN E – THÍ NGHIỆM CỬA SỔ")
    print("═" * 60)

    # E1: Vẽ tổng quan 4 cửa sổ
    print("\n  [E1] Vẽ tổng quan 4 cửa sổ...")
    plot_window_overview(N=1024)

    # Lấy frame 25ms tại 40% duration để thực nghiệm
    frame_ms = 25.0
    N_frame  = round(frame_ms * 1e-3 * fs)   # Số mẫu trong frame 25ms
    i0       = int(dur * 0.4 * fs)            # Vị trí bắt đầu frame
    i0       = max(0, min(i0, len(x) - N_frame))
    frame    = x[i0: i0 + N_frame]
    # Padding zeros nếu frame quá ngắn (xảy ra với file rất ngắn)
    if len(frame) < N_frame:
        frame = np.concatenate([frame, np.zeros(N_frame - len(frame))])

    # NFFT = 4× độ dài frame để có zero-padding và đường phổ mịn
    nfft     = int(2 ** np.ceil(np.log2(N_frame))) * 4
    freq_max = float(min(6000, fs // 2))

    print(f"\n  [E2] So sánh Rectangular vs Hamming trên frame thực")
    print(f"  Frame tại t={i0/fs:.3f}s | N={N_frame} | NFFT={nfft:,}")
    plot_window_on_frame(frame, fs, nfft=nfft,
                         freq_max=freq_max, fname=info["filename"])

    print("\n  ✓ Phần E hoàn thành.\n")


if __name__ == "__main__":
    run()
