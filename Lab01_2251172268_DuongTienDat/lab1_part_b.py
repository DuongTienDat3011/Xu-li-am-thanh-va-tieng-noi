"""
lab1_part_b.py
==============
CSE457 Lab 1 – Phần B: Phân tích miền thời gian
-------------------------------------------------
Yêu cầu:
  - Vẽ waveform toàn tệp và đoạn 0.5–1 s
  - Tính Peak, RMS, Energy, CREST factor, ZCR, kiểm tra clipping
  - Chọn ít nhất 2 đoạn đặc tính khác nhau, so sánh và giải thích
  - Short-time energy (STE) để phát hiện vùng năng lượng cao/thấp
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from lab1_utils import (
    read_wav, choose_audio_file, savefig, style_ax, db,
    FIGURES_DIR, AUDIO_DIR
)


# ════════════════════════════════════════════════════════════
# B1 – Tính các đại lượng đặc trưng miền thời gian
# Công thức:
#   Peak  = max|x[n]|
#   RMS   = sqrt(mean(x[n]²))
#   E     = sum(x[n]²)
#   Crest = Peak / RMS  (đo độ nhọn, cao = nhiều transient)
#   ZCR   = tỷ lệ lần tín hiệu cắt qua 0 (cao = tín hiệu tần số cao)
# ════════════════════════════════════════════════════════════

def time_domain_stats(x: np.ndarray, label: str = "") -> dict:
    """
    Tính và in đầy đủ các đại lượng miền thời gian.
    Trả về dict để dùng trong so sánh và vẽ hình.
    """
    # Peak: giá trị tuyệt đối lớn nhất
    peak = float(np.max(np.abs(x)))

    # RMS (Root Mean Square): mức năng lượng hiệu dụng
    rms  = float(np.sqrt(np.mean(x ** 2)))

    # Energy: tổng bình phương tất cả mẫu
    energy = float(np.sum(x ** 2))

    # Crest factor: tỷ số Peak/RMS – cao khi tín hiệu có nhiều transient
    crest    = peak / (rms + 1e-15)
    crest_db = 20 * np.log10(crest + 1e-15)

    # Zero Crossing Rate: tỷ lệ mẫu có dấu thay đổi (tín hiệu cắt qua 0)
    # Tín hiệu tần số cao → ZCR cao; tín hiệu tần số thấp → ZCR thấp
    zcr = float(np.mean(np.abs(np.diff(np.sign(x))) > 0))

    # Kiểm tra clipping: mẫu nào có |x[n]| >= 0.999 là đã clip
    n_clip   = int(np.sum(np.abs(x) >= 0.999))
    clip_pct = n_clip / len(x) * 100.0

    stats = dict(
        label     = label,
        n_samples = len(x),
        peak      = peak,
        peak_dBFS = db(peak),
        rms       = rms,
        rms_dBFS  = db(rms),
        energy    = energy,
        crest     = crest,
        crest_db  = crest_db,
        zcr       = zcr,
        n_clipped = n_clip,
        clip_pct  = clip_pct,
    )

    # In kết quả ra console
    tag = f"[{label}]" if label else ""
    print(f"  {tag}")
    print(f"    N       = {len(x):,} mẫu")
    print(f"    Peak    = {peak:.6f}  ({db(peak):.2f} dBFS)")
    print(f"    RMS     = {rms:.6f}  ({db(rms):.2f} dBFS)")
    print(f"    Energy  = {energy:.4f}")
    print(f"    Crest   = {crest:.4f}  ({crest_db:.2f} dB)")
    print(f"    ZCR     = {zcr:.4f}  (zero-crossing rate)")
    status = f"⚠ {n_clip} mẫu ({clip_pct:.4f}%)" if n_clip > 0 else "✓ Không có"
    print(f"    Clipping: {status}")

    return stats


# ════════════════════════════════════════════════════════════
# B2 – Short-Time Energy (STE)
# Chia tín hiệu thành các frame ngắn, tính năng lượng từng frame.
# Mục đích: theo dõi biến đổi năng lượng theo thời gian
#   → Phát hiện đoạn im lặng, nhịp, attack của âm thanh
# ════════════════════════════════════════════════════════════

def short_time_energy(x: np.ndarray, fs: int,
                      frame_ms: float = 20.0,
                      hop_ms:   float = 10.0) -> tuple:
    """
    Tính Short-Time Energy (STE) theo từng frame.

    Thuật toán:
      - Chia tín hiệu thành các frame dài frame_ms ms
      - Bước nhảy (hop) là hop_ms ms (có thể overlap giữa các frame)
      - STE[m] = (1/N) * sum(x[n]² trong frame m)

    Returns:
        (t_ste, ste) – mảng thời gian và mảng năng lượng tương ứng
    """
    frame_n = int(frame_ms * 1e-3 * fs)    # Số mẫu mỗi frame
    hop_n   = int(hop_ms   * 1e-3 * fs)    # Số mẫu bước nhảy
    N       = len(x)

    ste_vals = []
    t_vals   = []

    start = 0
    while start + frame_n <= N:
        frame = x[start: start + frame_n]
        # Năng lượng trung bình trong frame (normalized by frame length)
        ste_vals.append(float(np.sum(frame ** 2)) / frame_n)
        # Thời gian tại tâm frame
        t_vals.append((start + frame_n / 2) / fs)
        start += hop_n

    return np.array(t_vals), np.array(ste_vals)


# ════════════════════════════════════════════════════════════
# B3 – Vẽ waveform toàn tệp kết hợp Short-Time Energy
# Subplot trên: waveform với đường RMS, Peak, ngưỡng clipping
# Subplot dưới: biểu đồ STE cho thấy biến động năng lượng
# ════════════════════════════════════════════════════════════

def plot_full_waveform(x: np.ndarray, fs: int, fname: str = "") -> None:
    t    = np.arange(len(x)) / fs
    rms  = float(np.sqrt(np.mean(x ** 2)))
    peak = float(np.max(np.abs(x)))

    # Tính STE và normalize để hiển thị cùng trục dọc [0,1]
    t_ste, ste = short_time_energy(x, fs)
    ste_norm   = ste / (np.max(ste) + 1e-15)

    # Tạo layout với GridSpec: subplot waveform cao gấp 3 subplot STE
    fig = plt.figure(figsize=(13, 6))
    gs  = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.12)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)    # Chia sẻ trục x với subplot trên

    # ── Subplot 1: Waveform ──
    ax1.plot(t, x, color="steelblue", lw=0.3, alpha=0.85)
    # Đường RMS nằm ngang (năng lượng hiệu dụng)
    ax1.axhline( rms,  color="orange", lw=1.0, ls="-.",
                 label=f"RMS = {rms:.4f}  ({db(rms):.1f} dBFS)")
    ax1.axhline(-rms,  color="orange", lw=1.0, ls="-.")
    # Đường Peak (giá trị lớn nhất)
    ax1.axhline( peak, color="purple", lw=0.7, ls=":",
                 label=f"Peak = {peak:.4f}  ({db(peak):.1f} dBFS)")
    ax1.axhline(-peak, color="purple", lw=0.7, ls=":")
    # Ngưỡng clipping ±1.0
    ax1.axhline( 1.0,  color="red",    lw=0.6, ls="--", alpha=0.5, label="Clipping ±1")
    ax1.axhline(-1.0,  color="red",    lw=0.6, ls="--", alpha=0.5)
    style_ax(ax1, ylabel="Biên độ chuẩn hóa",
             title=f"Phần B – Waveform toàn tệp: {fname}")
    ax1.legend(fontsize=8, loc="upper right")
    ax1.set_ylim([-1.15, 1.15])
    plt.setp(ax1.get_xticklabels(), visible=False)  # Ẩn nhãn trục x

    # ── Subplot 2: Short-Time Energy ──
    ax2.fill_between(t_ste, ste_norm, alpha=0.55, color="darkorange")
    ax2.plot(t_ste, ste_norm, color="darkorange", lw=0.7)
    style_ax(ax2, xlabel="Thời gian (s)", ylabel="STE (norm)",
             title="Short-Time Energy (frame=20ms, hop=10ms)")
    ax2.set_ylim([0, 1.05])

    fig.tight_layout(rect=[0, 0, 1, 1])
    savefig(fig, "B_waveform_full.png")


# ════════════════════════════════════════════════════════════
# B4 – Vẽ đoạn ngắn (zoom vào một đoạn cụ thể)
# Mục đích: quan sát chi tiết dạng sóng ở mức mẫu
# Đánh dấu các điểm clipping bằng chấm đỏ (nếu có)
# ════════════════════════════════════════════════════════════

def plot_segment(x: np.ndarray, fs: int,
                 t_start: float, t_end: float,
                 label: str = "", color: str = "steelblue",
                 save_name: str = "B_segment.png") -> dict:
    """
    Vẽ và phân tích một đoạn tín hiệu từ t_start đến t_end (giây).
    Hiển thị RMS, Peak và đánh dấu điểm clipping.
    Trả về dict stats để dùng trong so sánh.
    """
    # Cắt đoạn tín hiệu cần xem
    i0  = max(0, int(t_start * fs))
    i1  = min(len(x), int(t_end   * fs))
    seg = x[i0:i1]
    t   = np.linspace(t_start, t_end, len(seg))

    # Tính thống kê cho đoạn này
    stats = time_domain_stats(seg, label)

    fig, ax = plt.subplots(figsize=(13, 3.5))
    ax.fill_between(t, seg, alpha=0.25, color=color)
    ax.plot(t, seg, color=color, lw=0.6)

    # Tìm và đánh dấu các mẫu bị clipping (|x| >= 0.999)
    clipped = np.abs(seg) >= 0.999
    if np.any(clipped):
        ax.scatter(t[clipped], seg[clipped],
                   color="red", s=18, zorder=5, label="Clipping")

    rms = stats["rms"]
    # Đường RMS, Peak, ngưỡng clipping
    ax.axhline( rms,           color="orange", lw=1.0, ls="-.",
                label=f"RMS={rms:.5f}  ({stats['rms_dBFS']:.1f}dBFS)")
    ax.axhline(-rms,           color="orange", lw=1.0, ls="-.")
    ax.axhline( stats["peak"], color="purple", lw=0.7, ls=":",
                label=f"Peak={stats['peak']:.5f}  ({stats['peak_dBFS']:.1f}dBFS)")
    ax.axhline(-stats["peak"], color="purple", lw=0.7, ls=":")
    ax.axhline( 1.0,           color="red",    lw=0.6, ls="--", alpha=0.5)
    ax.axhline(-1.0,           color="red",    lw=0.6, ls="--", alpha=0.5)

    style_ax(ax, xlabel="Thời gian (s)", ylabel="Biên độ",
             title=f"Phần B – {label}  ({t_start:.3f}s – {t_end:.3f}s)  "
                   f"E={stats['energy']:.3f}  Crest={stats['crest_db']:.1f}dB")
    ax.legend(fontsize=8, loc="upper right")
    ax.set_xlim([t_start, t_end])
    ax.set_ylim([-1.15, 1.15])
    plt.tight_layout()
    savefig(fig, save_name)
    return stats


# ════════════════════════════════════════════════════════════
# B5 – So sánh 2 đoạn tín hiệu có đặc tính khác nhau
# Mục đích: thể hiện sự khác biệt giữa các đoạn
#   (ví dụ: đoạn yên tĩnh vs đoạn có nhịp mạnh)
# In bảng so sánh Peak, RMS, Energy, Crest, ZCR, Clipping
# ════════════════════════════════════════════════════════════

def plot_compare_segments(x: np.ndarray, fs: int,
                          seg1: tuple, seg2: tuple) -> None:
    """
    So sánh 2 đoạn tín hiệu đặt cạnh nhau.
    seg1, seg2: tuple (t_start, t_end, label)
    """
    (t1s, t1e, lbl1) = seg1
    (t2s, t2e, lbl2) = seg2

    # Cắt 2 đoạn
    s1  = x[int(t1s*fs):int(t1e*fs)]
    s2  = x[int(t2s*fs):int(t2e*fs)]
    t_1 = np.linspace(t1s, t1e, len(s1))
    t_2 = np.linspace(t2s, t2e, len(s2))

    # Tính thống kê cho cả 2 đoạn
    st1 = time_domain_stats(s1, lbl1)
    st2 = time_domain_stats(s2, lbl2)

    fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=False)

    # Vẽ 2 đoạn với màu khác nhau + đường RMS
    for ax, sig, t_ax, st, clr, lbl in [
        (axes[0], s1, t_1, st1, "steelblue", lbl1),
        (axes[1], s2, t_2, st2, "tomato",    lbl2),
    ]:
        ax.plot(t_ax, sig, color=clr, lw=0.5)
        ax.fill_between(t_ax, sig, alpha=0.2, color=clr)
        # Đường RMS kèm thống kê tóm tắt trong legend
        ax.axhline( st["rms"], color="orange", lw=1.0, ls="-.",
                    label=f"RMS={st['rms']:.5f} ({st['rms_dBFS']:.1f}dBFS)  "
                          f"E={st['energy']:.3f}  Crest={st['crest_db']:.1f}dB  ZCR={st['zcr']:.3f}")
        ax.axhline(-st["rms"], color="orange", lw=1.0, ls="-.")
        style_ax(ax, ylabel="Biên độ",
                 title=f"{lbl}  Peak={st['peak_dBFS']:.1f}dBFS")
        ax.legend(fontsize=8)
        ax.set_ylim([-1.15, 1.15])

    axes[0].set_xlabel("")
    axes[1].set_xlabel("Thời gian (s)")

    # In bảng so sánh trực quan
    print("\n  ┌─────────────────┬──────────────────┬──────────────────┐")
    print(f"  │ Chỉ số          │ {lbl1[:16]:<16} │ {lbl2[:16]:<16} │")
    print("  ├─────────────────┼──────────────────┼──────────────────┤")
    rows = [
        ("Peak (dBFS)",  f"{st1['peak_dBFS']:>12.2f} dB", f"{st2['peak_dBFS']:>12.2f} dB"),
        ("RMS  (dBFS)",  f"{st1['rms_dBFS']:>12.2f} dB",  f"{st2['rms_dBFS']:>12.2f} dB"),
        ("Energy",       f"{st1['energy']:>15.3f}",        f"{st2['energy']:>15.3f}"),
        ("Crest factor", f"{st1['crest_db']:>12.2f} dB",  f"{st2['crest_db']:>12.2f} dB"),
        ("ZCR",          f"{st1['zcr']:>15.4f}",           f"{st2['zcr']:>15.4f}"),
        ("Clipped",      f"{st1['n_clipped']:>15,}",       f"{st2['n_clipped']:>15,}"),
    ]
    for r, v1, v2 in rows:
        print(f"  │ {r:<15}  │ {v1:>16} │ {v2:>16} │")
    print("  └─────────────────┴──────────────────┴──────────────────┘")

    fig.suptitle("Phần B – So sánh 2 đoạn tín hiệu",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    savefig(fig, "B_compare_segments.png")


# ════════════════════════════════════════════════════════════
# HÀM CHẠY PHẦN B
# Thứ tự: thống kê toàn tệp → waveform + STE → đoạn đầu → đoạn giữa → so sánh
# ════════════════════════════════════════════════════════════

def run(info: dict = None) -> None:
    """
    Chạy toàn bộ phần B. Nhận info dict từ phần A.
    Nếu info=None sẽ hiện menu chọn file.
    """
    if info is None:
        info = read_wav(choose_audio_file())

    x   = info["x_mono"]
    fs  = info["fs"]
    dur = info["duration"]

    print("\n" + "═" * 60)
    print("  PHẦN B – PHÂN TÍCH MIỀN THỜI GIAN")
    print("═" * 60)

    # B1: Thống kê cho toàn bộ tín hiệu
    print("\n  [B1] Thống kê toàn tệp:")
    time_domain_stats(x, f"Toàn tệp ({dur:.2f}s)")

    # B2: Vẽ waveform + Short-Time Energy
    print("\n  [B2] Waveform toàn tệp + Short-Time Energy...")
    plot_full_waveform(x, fs, info["filename"])

    # B3: Phân tích đoạn đầu (0 → 0.5s)
    t1s, t1e = 0.0, min(0.5, dur)
    print(f"\n  [B3] Đoạn đầu {t1s:.2f}–{t1e:.2f}s:")
    plot_segment(x, fs, t1s, t1e, "Đoạn đầu", "steelblue", "B_segment_1.png")

    # B4: Phân tích đoạn giữa (40%–90% duration, dài 0.5s)
    t2s = dur * 0.4
    t2e = min(t2s + 0.5, dur)
    print(f"\n  [B4] Đoạn giữa {t2s:.2f}–{t2e:.2f}s:")
    plot_segment(x, fs, t2s, t2e, "Đoạn giữa", "tomato", "B_segment_2.png")

    # B5: So sánh 2 đoạn đặc trưng
    print("\n  [B5] So sánh 2 đoạn:")
    plot_compare_segments(x, fs,
                          (t1s, t1e, "Đoạn đầu"),
                          (t2s, t2e, "Đoạn giữa"))

    print("\n  ✓ Phần B hoàn thành.\n")


if __name__ == "__main__":
    run()
