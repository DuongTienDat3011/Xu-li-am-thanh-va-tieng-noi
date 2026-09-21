"""
lab1_part_f.py
==============
CSE457 Lab 1 – Phần F: Lọc số FIR
------------------------------------
Yêu cầu:
  - Thiết kế FIR low-pass (LP), high-pass (HP), band-pass (BP)
  - Vẽ frequency response: magnitude dB, phase, group delay
  - Áp dụng filter, bù group delay đúng cách
  - So sánh phổ trước/sau lọc → liên hệ với H(f)
  - Xuất file WAV để nghe so sánh
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.signal import firwin, lfilter, freqz, group_delay

from lab1_utils import (
    read_wav, write_wav, choose_audio_file,
    savefig, style_ax, db,
    FIGURES_DIR, AUDIO_DIR
)


# ════════════════════════════════════════════════════════════
# F1 – Thiết kế FIR filter bằng phương pháp cửa sổ
# Nguyên lý: nhân đáp ứng xung lý tưởng (sinc) với cửa sổ
# để tạo FIR filter có độ dài hữu hạn.
#
# Công thức bậc FIR M (số taps = M+1):
#   y[n] = Σ(r=0..M) b_r · x[n-r]
#   H(e^jω) = Σ(r=0..M) b_r · e^(-jωr)
#
# Group delay của FIR đối xứng (linear phase):
#   τ = (M)/2 mẫu (hằng số, tất cả tần số bị trễ như nhau)
# ════════════════════════════════════════════════════════════

def design_fir(filter_type: str, cutoff, fs: int,
               numtaps: int = 201, window: str = "hamming") -> np.ndarray:
    """
    Thiết kế FIR filter dùng scipy.signal.firwin (window method).

    Parameters
    ----------
    filter_type : 'lowpass' | 'highpass' | 'bandpass' | 'bandstop'
    cutoff      : tần số cắt (Hz) – scalar hoặc [f_low, f_high]
    fs          : sampling rate (Hz)
    numtaps     : số hệ số (M+1); nên là số lẻ với HP/BS
    window      : cửa sổ thiết kế ('hamming'|'hann'|'blackman')

    Lưu ý:
      numtaps lớn → sườn dốc hơn nhưng group delay lớn hơn
      numtaps = 201 → group delay = 100 mẫu ≈ 4.5ms ở Fs=22050Hz
    """
    # High-pass và Band-stop cần numtaps lẻ để đối xứng đúng
    if filter_type in ("highpass", "bandstop") and numtaps % 2 == 0:
        numtaps += 1

    # pass_zero=True cho LP/BS (thông tần thấp), False cho HP/BP
    pass_zero = (filter_type in ("lowpass", "bandstop"))

    b = firwin(
        numtaps   = numtaps,
        cutoff    = cutoff,
        pass_zero = pass_zero,
        fs        = fs,
        window    = window,
    )
    return b


def filter_info(b: np.ndarray, fs: int, label: str = "") -> None:
    """In thông tin của bộ lọc: số taps, group delay, kiểm tra tính đối xứng."""
    taps     = len(b)
    gd_samps = (taps - 1) / 2         # Group delay = (M)/2 mẫu
    gd_ms    = gd_samps / fs * 1000   # Chuyển sang ms
    print(f"  [{label}]")
    print(f"    Taps (M+1)  = {taps}")
    print(f"    Group delay = {gd_samps:.1f} mẫu = {gd_ms:.3f} ms  (tại Fs={fs}Hz)")
    print(f"    b[0..4]     = {b[:5].round(6)}")
    # FIR tuyến tính phải có hệ số đối xứng: b[k] = b[M-k]
    print(f"    b là đối xứng: {np.allclose(b, b[::-1])}")


# ════════════════════════════════════════════════════════════
# F2 – Vẽ frequency response (magnitude + phase + group delay)
# 3 hàng cho mỗi filter:
#   Hàng 0: |H(f)| dB – cho thấy passband và stopband
#   Hàng 1: Phase (°) – mong đợi tuyến tính (FIR linear phase)
#   Hàng 2: Group delay (ms) – mong đợi hằng số trong passband
# ════════════════════════════════════════════════════════════

def plot_filter_responses_fixed(filters: list, fs: int, fname: str = "") -> None:
    """
    Vẽ frequency response cho nhiều filter cạnh nhau.
    filters: list of (b, label, color)
    """
    n    = len(filters)
    fig, axes = plt.subplots(3, n, figsize=(6 * n, 10))
    if n == 1:
        axes = axes.reshape(3, 1)

    worN = 8192   # Số điểm tính frequency response

    for col, (b, label, clr) in enumerate(filters):
        # Tính frequency response H(f) = scipy.signal.freqz
        w, H = freqz(b, worN=worN, fs=fs)
        H_mag_dB = 20 * np.log10(np.maximum(np.abs(H), 1e-15))  # Magnitude dB
        H_phase  = np.unwrap(np.angle(H)) * (180 / np.pi)        # Phase (degrees)

        # Tính group delay (derivative của phase)
        w_gd, gd_samp = group_delay((b, [1.0]), w=worN, fs=fs)
        gd_ms = gd_samp / fs * 1000   # Chuyển từ mẫu sang ms

        # ── Hàng 0: Magnitude response ──
        ax = axes[0, col]
        ax.plot(w, H_mag_dB, color=clr, lw=1.2)
        ax.axhline(-3,  color="black", lw=0.7, ls="--", alpha=0.8, label="-3 dB")
        ax.axhline(-40, color="gray",  lw=0.6, ls=":",  alpha=0.7, label="-40 dB")

        # Tìm và đánh dấu tần số -3dB (cutoff)
        idx3 = np.where(H_mag_dB <= -3)[0]
        if len(idx3) > 0:
            f3 = w[idx3[0]]
            ax.axvline(f3, color="red", lw=0.8, ls="-.", alpha=0.8)
            ax.text(f3 + fs * 0.005, -12, f"{f3:.0f}Hz\n-3dB",
                    fontsize=7, color="red")
        style_ax(ax, ylabel="|H(f)| (dB)",
                 title=f"{label}\n{len(b)} taps")
        ax.set_ylim([-90, 5])
        ax.legend(fontsize=7)

        # ── Hàng 1: Phase response ──
        ax = axes[1, col]
        ax.plot(w, H_phase, color=clr, lw=0.9)
        style_ax(ax, ylabel="Phase (°)", title=f"Phase – {label}")
        # FIR linear phase → phase tuyến tính theo tần số

        # ── Hàng 2: Group delay ──
        ax = axes[2, col]
        ax.plot(w_gd, gd_ms, color=clr, lw=0.9)
        # Giá trị nominal = (numtaps-1)/2 / fs * 1000 ms
        gd_nom = (len(b) - 1) / 2 / fs * 1000
        ax.axhline(gd_nom, color="red", lw=0.8, ls="--",
                   label=f"Nominal {gd_nom:.2f}ms")
        style_ax(ax, xlabel="Tần số (Hz)", ylabel="Group delay (ms)",
                 title=f"Group delay – {label}")
        ax.legend(fontsize=7)
        # FIR linear phase → group delay phẳng (hằng số) trong passband

    fig.suptitle(f"Phần F – Đáp ứng tần số FIR  |  {fname}",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    savefig(fig, "F_filter_responses.png")


# Giữ tên cũ để không break code gọi
plot_filter_responses = plot_filter_responses_fixed


# ════════════════════════════════════════════════════════════
# F3 – Áp dụng FIR filter với bù group delay
# Vấn đề: lfilter() trả về output bị trễ (group_delay) mẫu so với input
# → Tín hiệu lọc sẽ lệch thời gian so với gốc
# Giải pháp: dịch output về trái delay mẫu, pad zeros ở cuối
# Kết quả: output căn chỉnh đúng về thời gian với input
# ════════════════════════════════════════════════════════════

def apply_fir(x: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Áp dụng FIR filter và bù group delay để căn chỉnh thời gian.

    FIR linear phase group delay = (len(b)-1)/2 mẫu.
    Sau lfilter(), output bị trễ đúng delay mẫu.
    → Dịch output[delay:] về đầu mảng, điền zeros ở cuối.
    """
    delay = (len(b) - 1) // 2    # Group delay tính bằng mẫu

    y_raw = lfilter(b, [1.0], x)  # Lọc: y = b * x (tích chập)

    # Bù group delay: dịch trái delay mẫu
    y = np.empty_like(x)
    y[:len(x) - delay] = y_raw[delay:]   # Phần đã lọc (đã bù delay)
    y[len(x) - delay:] = 0.0             # Pad zeros cuối (transient tail)
    return y


# ════════════════════════════════════════════════════════════
# F4 – So sánh phổ trước và sau lọc
# 3 subplot:
#   Subplot 0: phổ gốc và phổ đã lọc chồng lên nhau
#   Subplot 1: attenuation = phổ gốc - phổ đã lọc (dB)
#   Subplot 2: |H(f)| của filter để đối chiếu
# ════════════════════════════════════════════════════════════

def plot_before_after(x: np.ndarray, y: np.ndarray, fs: int,
                      b: np.ndarray,
                      label: str = "Filter",
                      freq_max: float = None,
                      save_suffix: str = "filter") -> None:
    """
    Vẽ so sánh phổ tín hiệu trước (x) và sau (y) khi áp dụng filter b.
    Dùng cùng 1 giây đầu với cửa sổ Hamming để tính phổ.
    """
    if freq_max is None:
        freq_max = float(min(8000, fs // 2))

    # Lấy 1s đầu để tính phổ (đủ dài, đủ nhanh)
    n_seg = min(len(x), fs)
    n_fft = int(2 ** np.ceil(np.log2(n_seg))) * 2   # Zero-pad 2× cho phổ mịn
    win   = np.hamming(n_seg)

    def spectrum_db(sig):
        """Tính phổ dB normalized cho 1 đoạn tín hiệu."""
        S  = np.fft.rfft(sig[:n_seg] * win, n=n_fft)
        cg = float(np.mean(win))
        m  = np.abs(S) / (n_seg * cg)
        m[1:-1] *= 2
        return np.fft.rfftfreq(n_fft, 1.0 / fs), 20 * np.log10(m + 1e-15)

    f_x, Xdb = spectrum_db(x)
    f_y, Ydb = spectrum_db(y)

    # Tính frequency response của filter để vẽ đối chiếu
    w_fr, H = freqz(b, worN=n_fft // 2, fs=fs)
    H_dB    = 20 * np.log10(np.maximum(np.abs(H), 1e-15))

    mask_x = f_x  <= freq_max
    mask_h = w_fr <= freq_max

    fig, axes = plt.subplots(3, 1, figsize=(13, 10))

    # ── Subplot 0: Phổ gốc vs phổ đã lọc ──
    axes[0].plot(f_x[mask_x], Xdb[mask_x],
                 color="steelblue", lw=0.8, alpha=0.85, label="Trước lọc")
    axes[0].plot(f_y[mask_x], Ydb[mask_x],
                 color="darkorange", lw=0.8, alpha=0.90, label=f"Sau {label}")
    # Tô màu vùng bị thay đổi
    axes[0].fill_between(f_x[mask_x],
                         np.minimum(Xdb[mask_x], Ydb[mask_x]),
                         np.maximum(Xdb[mask_x], Ydb[mask_x]),
                         alpha=0.15, color="gray")
    axes[0].set_ylim([-90, 5])
    style_ax(axes[0], ylabel="dBFS", title=f"Phổ trước và sau lọc – {label}")
    axes[0].legend(fontsize=9)

    # ── Subplot 1: Attenuation của filter ──
    # Attenuation (dB) = phổ gốc - phổ đã lọc
    # Giá trị dương = tần số bị suy giảm; âm = tần số được khuếch đại (không mong muốn)
    atten = Xdb[mask_x] - Ydb[mask_x]
    axes[1].plot(f_x[mask_x], atten, color="seagreen", lw=0.8)
    axes[1].axhline(0,   color="black", lw=0.5)
    axes[1].axhline(3,   color="red",   lw=0.7, ls="--", label="3 dB")
    axes[1].axhline(40,  color="gray",  lw=0.6, ls=":",  label="40 dB")
    style_ax(axes[1], ylabel="Attenuation (dB)",
             title="Suy giảm (Trước − Sau): dương = bị lọc bỏ")
    axes[1].legend(fontsize=8)

    # ── Subplot 2: Frequency response H(f) để đối chiếu ──
    axes[2].plot(w_fr[mask_h], H_dB[mask_h],
                 color="tomato", lw=1.0, label="|H(f)| của filter")
    axes[2].axhline(-3, color="black", lw=0.7, ls="--", label="-3 dB")
    axes[2].set_ylim([-90, 5])
    style_ax(axes[2], xlabel="Tần số (Hz)", ylabel="|H(f)| (dB)",
             title=f"Đáp ứng tần số – {label}")
    axes[2].legend(fontsize=8)

    fig.suptitle(f"Phần F – {label}  |  {freq_max:.0f} Hz display",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    savefig(fig, f"F_before_after_{save_suffix}.png")


# ════════════════════════════════════════════════════════════
# HÀM CHẠY PHẦN F
# F1: thiết kế 3 filter (LP, HP, BP)
# F2: vẽ frequency response
# F3–F5: áp dụng từng filter, lưu WAV, vẽ so sánh phổ
# ════════════════════════════════════════════════════════════

def run(info: dict = None) -> dict:
    if info is None:
        info = read_wav(choose_audio_file())

    # Xác định prefix tên file (speech_ hoặc music_) dựa vào tên file đầu vào
    fname_base = os.path.splitext(info["filename"])[0]
    if any(k in fname_base.lower() for k in ("speech","voice","tts","dialogue","mono")):
        prefix = "speech"
    else:
        prefix = "music"

    x  = info["x_mono"]
    fs = info["fs"]

    print("\n" + "═" * 60)
    print("  PHẦN F – LỌC SỐ FIR")
    print("═" * 60)

    # Xác định tần số cutoff phù hợp với Fs của file
    nyq    = fs / 2.0
    taps   = 201
    lp_cut = min(2000, int(nyq * 0.40))    # LP: giữ dưới 2kHz (giọng nói)
    hp_cut = min(3000, int(nyq * 0.55))    # HP: giữ trên 3kHz (âm sắc cao)
    bp_lo  = min(500,  int(nyq * 0.08))    # BP lo: 500Hz
    bp_hi  = min(3500, int(nyq * 0.65))    # BP hi: 3.5kHz

    print(f"  Fs={fs}Hz | Nyquist={nyq:.0f}Hz | Taps={taps}")
    print(f"  LP cutoff : {lp_cut} Hz   → giữ tiếng nói, loại tiếng ồn cao")
    print(f"  HP cutoff : {hp_cut} Hz   → giữ âm sắc cao, loại rumble")
    print(f"  BP passband: {bp_lo}–{bp_hi} Hz → dải thoại chính")
    gd_ms = (taps - 1) / 2 / fs * 1000
    print(f"  Group delay: {(taps-1)//2} mẫu = {gd_ms:.3f} ms")

    # Thiết kế 3 filter
    b_lp = design_fir("lowpass",  lp_cut,          fs, taps)
    b_hp = design_fir("highpass", hp_cut,           fs, taps)
    b_bp = design_fir("bandpass", [bp_lo, bp_hi],   fs, taps)

    # In thông tin filter
    print("\n  [F1] Thông tin bộ lọc:")
    filter_info(b_lp, fs, f"Low-pass {lp_cut}Hz")
    filter_info(b_hp, fs, f"High-pass {hp_cut}Hz")
    filter_info(b_bp, fs, f"Band-pass {bp_lo}-{bp_hi}Hz")

    # Vẽ frequency response của cả 3 filter
    print("\n  [F2] Vẽ frequency response...")
    plot_filter_responses_fixed(
        [(b_lp, f"Low-pass {lp_cut}Hz",         "steelblue"),
         (b_hp, f"High-pass {hp_cut}Hz",        "tomato"),
         (b_bp, f"Band-pass {bp_lo}-{bp_hi}Hz", "seagreen")],
        fs, fname=info["filename"]
    )

    freq_max = float(min(8000, int(nyq)))

    # Áp dụng và lưu – tên file thể hiện rõ loại tín hiệu + tham số filter
    print("\n  [F3] Áp dụng Low-pass...")
    y_lp    = apply_fir(x, b_lp)
    path_lp = write_wav(y_lp, fs, f"{prefix}_lpf_{lp_cut}hz.wav")
    plot_before_after(x, y_lp, fs, b_lp,
                      label=f"Low-pass {lp_cut}Hz",
                      freq_max=freq_max, save_suffix="lowpass")
    print(f"  → {path_lp}")

    # Áp dụng High-pass và lưu kết quả
    print("\n  [F4] Áp dụng High-pass...")
    y_hp    = apply_fir(x, b_hp)
    path_hp = write_wav(y_hp, fs, f"{prefix}_hpf_{hp_cut}hz.wav")
    plot_before_after(x, y_hp, fs, b_hp,
                      label=f"High-pass {hp_cut}Hz",
                      freq_max=freq_max, save_suffix="highpass")
    print(f"  → {path_hp}")

    # Áp dụng Band-pass và lưu kết quả
    print("\n  [F5] Áp dụng Band-pass...")
    y_bp    = apply_fir(x, b_bp)
    path_bp = write_wav(y_bp, fs, f"{prefix}_bpf_{bp_lo}_{bp_hi}hz.wav")
    plot_before_after(x, y_bp, fs, b_bp,
                      label=f"Band-pass {bp_lo}-{bp_hi}Hz",
                      freq_max=freq_max, save_suffix="bandpass")
    print(f"  → {path_bp}")

    print("\n  ✓ Phần F hoàn thành.\n")
    return dict(b_lp=b_lp, b_hp=b_hp, b_bp=b_bp,
                y_lp=y_lp, y_hp=y_hp, y_bp=y_bp)


if __name__ == "__main__":
    run()
