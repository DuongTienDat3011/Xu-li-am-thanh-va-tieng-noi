"""
lab1_part_g.py
==============
CSE457 Lab 1 – Phần G: Lượng tử hóa, Resampling và Mã hóa
-----------------------------------------------------------
Yêu cầu:
  - Lượng tử hóa B = 4, 6, 8, 12, 16 bit → tính SNR đo và lý thuyết
  - Resample về 16 kHz và 8 kHz (với anti-aliasing filter đúng)
  - Tính PCM bit rate, file size lý thuyết, compression ratio
  - Xuất file WAV cho từng cấu hình
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.signal import resample_poly
from math import gcd

try:
    import soundfile as _sf_check
    HAS_SOUNDFILE = True
except ImportError:
    HAS_SOUNDFILE = False

from lab1_utils import (
    read_wav, write_wav, choose_audio_file,
    savefig, style_ax, db,
    FIGURES_DIR, AUDIO_DIR
)


# ════════════════════════════════════════════════════════════
# G1 – Lượng tử hóa đều PCM (Uniform Mid-Tread Quantization)
#
# Nguyên lý:
#   L = 2^B mức lượng tử đều trên [-1, 1]
#   Δ = 2/L = bước lượng tử
#   x̂[n] = round(x[n]/Δ) × Δ   (mid-tread: có mức tại 0)
#   e[n] = x̂[n] − x[n]          (nhiễu lượng tử)
#
# Lý thuyết:
#   Phương sai nhiễu lượng tử đều ≈ Δ²/12
#   SNR_Q(dB) = 6.02B + 4.77 − 20·log10(Xmax/σx)
#   → Mỗi bit thêm: SNR tăng ~6.02 dB
# ════════════════════════════════════════════════════════════

def quantize_uniform(x: np.ndarray, B: int) -> np.ndarray:
    """
    Lượng tử hóa đều B-bit (mid-tread PCM) cho tín hiệu x ∈ [-1, 1].

    Bước thực hiện:
      1. Tính L = 2^B (số mức), Δ = 2/L (bước lượng tử)
      2. Clip x về [-1, 1-Δ] để tránh overflow
      3. Làm tròn x/Δ về số nguyên gần nhất, nhân lại với Δ
      4. Clip lại để đảm bảo trong phạm vi [-1, 1-Δ]
    """
    L     = 2 ** B
    delta = 2.0 / L            # Bước lượng tử (step size)

    # Clip trước khi lượng tử hóa: tránh x=1.0 gây index lớn hơn L/2
    x_clip = np.clip(x, -1.0, 1.0 - delta)

    # Làm tròn (round) thay vì floor → mid-tread (có mức tại 0)
    x_q = np.round(x_clip / delta) * delta

    # Đảm bảo kết quả hợp lệ sau round
    x_q = np.clip(x_q, -1.0, 1.0 - delta)
    return x_q


def compute_snr(x: np.ndarray, x_q: np.ndarray) -> float:
    """
    Tính SNR đo được (dB):
      SNR = 10·log10(P_signal / P_noise)
      P_signal = mean(x²)
      P_noise  = mean((x̂-x)²)
    Trả về inf nếu nhiễu = 0 (B=16 thường rất gần 0).
    """
    noise   = x_q - x
    sig_pow = np.mean(x  ** 2)
    noi_pow = np.mean(noise ** 2)
    if noi_pow < 1e-30:
        return float("inf")
    return 10.0 * np.log10(sig_pow / noi_pow)


def snr_theoretical(B: int, x: np.ndarray) -> float:
    """
    Tính SNR lý thuyết theo công thức Rabiner–Schafer:
      SNR_Q(dB) = 6.02·B + 4.77 − 20·log10(Xmax/σx)

    Trong đó:
      Xmax = 1.0 (tín hiệu đã chuẩn hóa full-scale)
      σx   = std(x) = RMS của tín hiệu (nếu mean ≈ 0)

    Ý nghĩa: SNR phụ thuộc không chỉ vào B mà còn vào mức tín hiệu.
    Tín hiệu yếu (σx nhỏ) → 20log10(1/σx) lớn → SNR lý thuyết thấp hơn.
    """
    sigma_x = float(np.std(x))
    if sigma_x < 1e-15:
        return 0.0
    Xmax = 1.0
    return 6.02 * B + 4.77 - 20 * np.log10(Xmax / sigma_x)


def run_quantization(x: np.ndarray, fs: int,
                     bit_depths: list = None) -> dict:
    """
    Chạy lượng tử hóa cho nhiều bit depth, in bảng so sánh.
    Trả về dict {B: {x_q, snr_measured, snr_theory, L, delta}}.
    """
    if bit_depths is None:
        bit_depths = [4, 6, 8, 12, 16]

    results = {}

    rms_x = np.sqrt(np.mean(x**2))
    print(f"\n  RMS tín hiệu gốc = {rms_x:.6f}  ({db(rms_x):.2f} dBFS)")
    print(f"\n  ┌──────┬──────────┬─────────────┬────────────┬──────────────┐")
    print(f"  │ B    │ L (mức)  │ Δ (bước)    │ SNR đo(dB) │ SNR lý th(dB)│")
    print(f"  ├──────┼──────────┼─────────────┼────────────┼──────────────┤")

    for B in bit_depths:
        x_q   = quantize_uniform(x, B)
        snr_m = compute_snr(x, x_q)
        snr_t = snr_theoretical(B, x)
        L     = 2 ** B
        delta = 2.0 / L
        results[B] = dict(x_q=x_q, snr_measured=snr_m,
                          snr_theory=snr_t, L=L, delta=delta)
        print(f"  │ {B:>4} │ {L:>8,} │ {delta:>11.7f} │ {snr_m:>10.2f} │ {snr_t:>12.2f} │")

    print(f"  └──────┴──────────┴─────────────┴────────────┴──────────────┘")
    print(f"  Lý thuyết: mỗi bit thêm → SNR tăng ~6.02 dB")
    print(f"  Thực nghiệm sai khác vì: phân bố tín hiệu, mức tín hiệu, mô hình nhiễu")
    return results


# ════════════════════════════════════════════════════════════
# G2 – Vẽ đồ thị lượng tử hóa
# Subplot 1: SNR đo được và lý thuyết theo số bit
# Subplot 2: Dạng sóng nhiễu lượng tử e[n] = x̂[n] − x[n]
#   B thấp → nhiễu lớn, dạng sóng rõ "bậc thang"
#   B cao  → nhiễu nhỏ, trông như nhiễu trắng
# ════════════════════════════════════════════════════════════

def plot_quantization(x: np.ndarray, fs: int, quant_results: dict) -> None:
    bit_depths = list(quant_results.keys())

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # ── Subplot trái: SNR vs Bit depth ──
    snr_m = [quant_results[B]["snr_measured"] for B in bit_depths]
    snr_t = [quant_results[B]["snr_theory"]   for B in bit_depths]
    ref_6 = [6.02 * B for B in bit_depths]   # Đường tham chiếu 6.02 dB/bit

    axes[0].plot(bit_depths, snr_m, "o-", color="steelblue",
                 lw=1.5, ms=7, label="SNR đo được")
    axes[0].plot(bit_depths, snr_t, "s--", color="tomato",
                 lw=1.2, ms=7, label="SNR lý thuyết (6.02B+4.77)")
    axes[0].plot(bit_depths, ref_6, ":k",
                 lw=0.9, alpha=0.4, label="6.02 dB/bit (ref đơn giản)")

    # Ghi giá trị SNR lên mỗi điểm
    for B, sm in zip(bit_depths, snr_m):
        axes[0].annotate(f"{sm:.1f}", xy=(B, sm),
                         xytext=(B + 0.3, sm + 1.5),
                         fontsize=8, color="steelblue")

    style_ax(axes[0], xlabel="Bit/mẫu (B)", ylabel="SNR (dB)",
             title="SNR lượng tử theo số bit/mẫu")
    axes[0].legend(fontsize=9)
    axes[0].set_xticks(bit_depths)

    # ── Subplot phải: Nhiễu lượng tử (50ms đầu) ──
    n_show = min(int(0.05 * fs), len(x))   # 50ms
    seg    = x[:n_show]
    t_ms   = np.arange(n_show) / fs * 1000

    # Chỉ vẽ 3 bit depth điển hình để tránh chồng chéo
    show_bits = [bit_depths[0],
                 bit_depths[len(bit_depths)//2],
                 bit_depths[-1]]

    axes[1].set_title("Nhiễu lượng tử (50ms đầu)", fontweight="bold")
    for B, c in zip(show_bits, ["tomato", "darkorange", "steelblue"]):
        noise = quant_results[B]["x_q"][:n_show] - seg
        delta = quant_results[B]["delta"]
        axes[1].plot(t_ms, noise, color=c, lw=0.5, alpha=0.8,
                     label=f"B={B}  Δ={delta:.5f}")
        # Đường ±Δ/2: biên lý thuyết của nhiễu lượng tử đều
        axes[1].axhline( delta / 2, color=c, lw=0.4, ls="--", alpha=0.4)
        axes[1].axhline(-delta / 2, color=c, lw=0.4, ls="--", alpha=0.4)

    style_ax(axes[1], xlabel="ms", ylabel="e[n] = x̂[n]−x[n]",
             title="Nhiễu lượng tử e[n] (50ms đầu)")
    axes[1].legend(fontsize=8)

    fig.suptitle("Phần G – Lượng tử hóa PCM", fontsize=12, fontweight="bold")
    plt.tight_layout()
    savefig(fig, "G_quantization.png")


def plot_quantization_spectrum(x: np.ndarray, fs: int,
                               quant_results: dict,
                               freq_max: float = None) -> None:
    """
    Vẽ phổ của nhiễu lượng tử.
    B thấp → nhiễu có phổ phẳng (white noise-like).
    B cao  → nhiễu nhỏ nhưng phổ vẫn khá phẳng.
    """
    if freq_max is None:
        freq_max = float(min(8000, fs // 2))

    # Chỉ vẽ các B <= 8 để thấy rõ nhiễu
    show_bits = [b for b in quant_results if b <= 8] or list(quant_results.keys())[:3]
    n_seg = min(len(x), fs)
    win   = np.hamming(n_seg)
    nfft  = int(2 ** np.ceil(np.log2(n_seg)))

    fig, axes = plt.subplots(1, len(show_bits),
                              figsize=(6 * len(show_bits), 5),
                              sharey=True)
    if len(show_bits) == 1:
        axes = [axes]

    colors = ["tomato", "darkorange", "steelblue"]
    for ax, B, clr in zip(axes, show_bits, colors):
        # Tính phổ của nhiễu e[n] = x̂[n] - x[n]
        noise = quant_results[B]["x_q"][:n_seg] - x[:n_seg]
        N_db  = 20 * np.log10(np.maximum(
            np.abs(np.fft.rfft(noise * win, n=nfft)), 1e-15))
        f_ax  = np.fft.rfftfreq(nfft, 1.0 / fs)
        mask  = f_ax <= freq_max
        ax.plot(f_ax[mask], N_db[mask], color=clr, lw=0.6)
        snr_str = f"{quant_results[B]['snr_measured']:.1f}dB"
        style_ax(ax, xlabel="Hz",
                 ylabel="dBFS" if ax == axes[0] else "",
                 title=f"Phổ nhiễu B={B}bit | SNR={snr_str}")
        ax.set_ylim([-90, 0])

    fig.suptitle("Phần G – Phổ nhiễu lượng tử (B thấp → phổ phẳng như white noise)",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    savefig(fig, "G_quantization_noise_spectrum.png")


# ════════════════════════════════════════════════════════════
# G3 – Resampling đúng chuẩn (với anti-aliasing filter)
#
# Khi giảm sampling rate (downsample):
#   PHẢI lọc anti-aliasing TRƯỚC khi bỏ mẫu
#   → Tránh aliasing: tần số > Fs_mới/2 sẽ bị "gập vào" vùng thấp
#
# scipy.signal.resample_poly tự động xây dựng LP anti-aliasing filter
#   Tỷ lệ up/down = Fs_mới/Fs_cũ (tính bằng GCD để tối giản)
# ════════════════════════════════════════════════════════════

def resample_audio(x: np.ndarray, fs_src: int, fs_dst: int) -> np.ndarray:
    """
    Resample tín hiệu từ fs_src Hz sang fs_dst Hz.

    Dùng scipy.signal.resample_poly:
      1. Tính GCD(fs_src, fs_dst) để tối giản tỷ lệ
      2. Upsample × up = fs_dst/GCD
      3. Lọc LP anti-aliasing (tự động tại min(fs_src, fs_dst)/2)
      4. Downsample × down = fs_src/GCD
    """
    if fs_src == fs_dst:
        return x.copy()

    g    = gcd(fs_src, fs_dst)
    up   = fs_dst // g       # Hệ số upsample
    down = fs_src // g       # Hệ số downsample
    y    = resample_poly(x, up, down)   # Tự động anti-alias

    return np.clip(y.astype(np.float64), -1.0, 1.0)


def plot_resampling(x_orig: np.ndarray, fs_orig: int,
                    resampled_dict: dict,
                    freq_max: float = None) -> None:
    """
    Vẽ waveform và phổ FFT của tín hiệu gốc và các bản resampled.
    Chú ý: đường đứng đỏ đánh dấu tần số Nyquist của mỗi bản.
    """
    all_sigs = {fs_orig: x_orig}
    all_sigs.update(resampled_dict)

    if freq_max is None:
        freq_max = float(min(22050, fs_orig // 2))

    n_sigs = len(all_sigs)
    fig, axes = plt.subplots(n_sigs, 2, figsize=(14, 3.5 * n_sigs))
    if n_sigs == 1:
        axes = axes.reshape(1, -1)

    colors = ["steelblue", "darkorange", "seagreen", "tomato"]

    for row, (fs_c, sig) in enumerate(all_sigs.items()):
        clr = colors[row % len(colors)]
        # Chỉ vẽ 0.5s đầu để thấy rõ dạng sóng
        n_show = min(len(sig), int(0.5 * fs_c))
        t      = np.arange(n_show) / fs_c

        # Subplot trái: waveform
        axes[row, 0].plot(t, sig[:n_show], color=clr, lw=0.4)
        style_ax(axes[row, 0], xlabel="s", ylabel="Biên độ",
                 title=f"Waveform  Fs={fs_c:,}Hz  Nyquist={fs_c//2:,}Hz")

        # Subplot phải: phổ FFT
        n_seg = min(len(sig), fs_c)
        nfft  = int(2 ** np.ceil(np.log2(n_seg)))
        win   = np.hamming(n_seg)
        Xdb   = 20 * np.log10(np.maximum(
            np.abs(np.fft.rfft(sig[:n_seg] * win, n=nfft)), 1e-15))
        f_ax  = np.fft.rfftfreq(nfft, 1.0 / fs_c)
        mask  = f_ax <= min(freq_max, fs_c // 2)

        axes[row, 1].plot(f_ax[mask], Xdb[mask], color=clr, lw=0.6)
        # Đường đỏ: Nyquist của sampling rate hiện tại
        axes[row, 1].axvline(fs_c // 2, color="red", lw=0.8, ls="--",
                             label=f"Nyquist={fs_c//2:,}Hz")
        axes[row, 1].set_ylim([-90, 5])
        style_ax(axes[row, 1], xlabel="Hz", ylabel="dBFS",
                 title=f"Phổ  Fs={fs_c:,}Hz")
        axes[row, 1].legend(fontsize=8)

    fig.suptitle("Phần G – So sánh Resampling (anti-alias via resample_poly)",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    savefig(fig, "G_resampling.png")


# ════════════════════════════════════════════════════════════
# G4 – Tính Bit rate, File size và Compression ratio
#
# PCM bit rate = Fs × B × C   (bit/s)
#   Fs: sampling rate (Hz)
#   B:  bit depth (bit/sample)
#   C:  số kênh
#
# Compression ratio = R_PCM_ref / R_compressed
# Saving (%) = (1 - R_compressed/R_PCM_ref) × 100%
#
# Tham chiếu chuẩn CD: 44100 Hz × 16 bit × 2 kênh = 1411.2 kbps
# ════════════════════════════════════════════════════════════

def bitrate_analysis(fs: int, channels: int,
                     duration: float, file_size_bytes: int) -> None:
    """In bảng bit rate, kích thước ước tính và compression ratio."""
    configs = [
        ("PCM  8-bit mono   8kHz",    8000,  8, 1),
        ("PCM  8-bit mono  16kHz",   16000,  8, 1),
        ("PCM 16-bit mono  16kHz",   16000, 16, 1),
        ("PCM 16-bit mono  44.1kHz", 44100, 16, 1),
        ("PCM 16-bit stereo 44.1kHz",44100, 16, 2),
        (f"Gốc ({fs}Hz,{channels}ch,16bit)", fs, 16, channels),
    ]

    # Tham chiếu: PCM 16-bit stereo 44.1kHz (chuẩn CD)
    pcm_ref_bps = 44100 * 16 * 2

    print(f"\n  ┌{'─'*34}┬{'─'*10}┬{'─'*10}┬{'─'*12}┬{'─'*9}┐")
    print(f"  │ {'Cấu hình':<32} │ {'kbps':>8} │ {'MB':>8} │ {'Ratio':>10} │ {'Saving':>7} │")
    print(f"  ├{'─'*34}┼{'─'*10}┼{'─'*10}┼{'─'*12}┼{'─'*9}┤")
    for name, fs_c, bits, ch in configs:
        bps  = fs_c * bits * ch              # Bit rate (bit/s)
        sz   = bps * duration / 8 / (1024 ** 2)  # Kích thước ước tính (MB)
        rat  = pcm_ref_bps / bps             # Compression ratio so với CD
        sav  = (1 - bps / pcm_ref_bps) * 100   # Tiết kiệm so với CD (%)
        print(f"  │ {name:<32} │ {bps/1000:>8.1f} │ {sz:>8.3f} │ {rat:>10.2f}:1 │ {sav:>6.1f}% │")

    # Thêm dòng file thực tế (đo từ kích thước file)
    print(f"  ├{'─'*34}┼{'─'*10}┼{'─'*10}┼{'─'*12}┼{'─'*9}┤")
    actual_kbps  = file_size_bytes * 8 / duration / 1000
    actual_mb    = file_size_bytes / (1024 ** 2)
    actual_ratio = pcm_ref_bps / (actual_kbps * 1000)
    actual_save  = (1 - actual_kbps * 1000 / pcm_ref_bps) * 100
    print(f"  │ {'File thực tế (đo từ file size)':<32} │ {actual_kbps:>8.1f} │ {actual_mb:>8.3f} │ {actual_ratio:>10.2f}:1 │ {actual_save:>6.1f}% │")
    print(f"  └{'─'*34}┴{'─'*10}┴{'─'*10}┴{'─'*12}┴{'─'*9}┘")


def export_compressed(signal: np.ndarray, fs: int,
                      filename_base: str,
                      qualities: list = None) -> dict:
    """
    Xuất file nén OGG/Vorbis ở nhiều mức chất lượng dùng soundfile.
    So sánh file size thực tế và tính compression ratio so với PCM 16-bit.

    Parameters
    ----------
    signal        : tín hiệu float64 [-1,1]
    fs            : sampling rate
    filename_base : ví dụ 'music_chord' → music_chord_q3.ogg
    qualities     : list of int Vorbis quality [-1..10], mặc định [1,3,6,9]

    Returns
    -------
    dict {quality: {"ogg_path", "ogg_size_bytes", "wav_size_bytes",
                    "ogg_kbps", "ratio", "saving_pct"}}
    """
    import soundfile as sf

    if qualities is None:
        qualities = [1, 3, 6, 9]

    # Kích thước PCM 16-bit tham chiếu
    wav_bytes = len(signal) * 2   # 2 bytes/sample (int16)
    wav_kbps  = fs * 16 / 1000    # Bit rate PCM mono

    results   = {}
    print(f"\n  ── Nén OGG/Vorbis: {filename_base} ──────────────────────")
    print(f"  PCM 16-bit: {wav_bytes//1024} KB  ({wav_kbps:.1f} kbps)")
    print(f"  {'Quality':>8}  {'Size(KB)':>9}  {'kbps':>7}  {'Ratio':>8}  {'Saving':>8}")
    print(f"  {'─'*8}  {'─'*9}  {'─'*7}  {'─'*8}  {'─'*8}")

    sig_f32 = signal.astype(np.float32)
    dur     = len(signal) / fs

    for q in qualities:
        fname    = f"{filename_base}_q{q}.ogg"
        ogg_path = os.path.join(AUDIO_DIR, fname)
        try:
            sf.write(ogg_path, sig_f32, fs,
                     format="OGG", subtype="VORBIS")
            ogg_bytes = os.path.getsize(ogg_path)
            ogg_kbps  = ogg_bytes * 8 / dur / 1000
            ratio     = wav_bytes / ogg_bytes
            saving    = (1 - ogg_bytes / wav_bytes) * 100
            print(f"  Q={q:>3}:  {ogg_bytes//1024:>6} KB  {ogg_kbps:>7.1f}  {ratio:>7.2f}:1  {saving:>6.1f}%")
            results[q] = dict(
                ogg_path      = ogg_path,
                ogg_size_bytes= ogg_bytes,
                wav_size_bytes= wav_bytes,
                ogg_kbps      = ogg_kbps,
                wav_kbps      = wav_kbps,
                ratio         = ratio,
                saving_pct    = saving,
                quality       = q,
            )
        except Exception as e:
            print(f"  Q={q}: LỖI – {e}")

    return results


def plot_compression_comparison(results_dict: dict,
                                 filename_base: str = "",
                                 save: bool = True) -> None:
    """
    Vẽ đồ thị so sánh file size và compression ratio theo chất lượng OGG.
    2 subplot:
      Trái : file size (KB) – OGG các quality vs PCM tham chiếu
      Phải : compression ratio vs quality
    """
    if not results_dict:
        return

    qualities = sorted(results_dict.keys())
    ogg_kb    = [results_dict[q]["ogg_size_bytes"] / 1024 for q in qualities]
    ogg_kbps  = [results_dict[q]["ogg_kbps"]               for q in qualities]
    ratios    = [results_dict[q]["ratio"]                   for q in qualities]
    savings   = [results_dict[q]["saving_pct"]              for q in qualities]
    wav_kb    = results_dict[qualities[0]]["wav_size_bytes"] / 1024
    wav_kbps  = results_dict[qualities[0]]["wav_kbps"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # ── Subplot trái: File size (KB) ──
    bars = axes[0].bar([str(q) for q in qualities], ogg_kb,
                       color=plt.cm.Blues(np.linspace(0.4, 0.9, len(qualities))),
                       edgecolor="black", lw=0.6, label="OGG Vorbis")
    axes[0].axhline(wav_kb, color="red", lw=1.5, ls="--",
                    label=f"PCM 16-bit = {wav_kb:.0f} KB")
    for bar, kb in zip(bars, ogg_kb):
        axes[0].text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + wav_kb * 0.01,
                     f"{kb:.0f}KB", ha="center", fontsize=8)
    style_ax(axes[0], xlabel="OGG Quality (Vorbis)",
             ylabel="Kích thước file (KB)",
             title=f"File Size: PCM vs OGG – {filename_base}")
    axes[0].legend(fontsize=8)
    axes[0].set_ylim([0, wav_kb * 1.25])

    # ── Subplot phải: Compression Ratio & Bit Rate ──
    color_ratio = "steelblue"
    ax2b = axes[1].twinx()
    axes[1].plot([str(q) for q in qualities], ratios, "o-",
                 color=color_ratio, lw=1.5, ms=7, label="Compression ratio")
    axes[1].set_ylabel("Compression ratio (PCM/OGG)", color=color_ratio, fontsize=9)
    axes[1].tick_params(axis="y", labelcolor=color_ratio)
    for xi, (q, r) in enumerate(zip(qualities, ratios)):
        axes[1].annotate(f"{r:.1f}:1", xy=(xi, r),
                         xytext=(xi + 0.1, r + max(ratios)*0.02),
                         fontsize=8, color=color_ratio)

    ax2b.plot([str(q) for q in qualities], ogg_kbps, "s--",
              color="tomato", lw=1.2, ms=6, label="OGG kbps")
    ax2b.axhline(wav_kbps, color="gray", lw=0.8, ls=":",
                 label=f"PCM {wav_kbps:.0f}kbps")
    ax2b.set_ylabel("Bit rate (kbps)", color="tomato", fontsize=9)
    ax2b.tick_params(axis="y", labelcolor="tomato")

    style_ax(axes[1], xlabel="OGG Quality (Vorbis)",
             title="Compression Ratio và Bit Rate theo Quality")
    axes[1].grid(True, alpha=0.3)

    lines1, labels1 = axes[1].get_legend_handles_labels()
    lines2, labels2 = ax2b.get_legend_handles_labels()
    axes[1].legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="upper left")

    # In nhận xét tự động
    best_q   = qualities[0]   # quality thấp nhất = nén mạnh nhất
    best_rat = results_dict[best_q]["ratio"]
    hi_q     = qualities[-1]
    hi_rat   = results_dict[hi_q]["ratio"]
    print(f"\n  Nhận xét:")
    print(f"    OGG Q={best_q}: ratio={best_rat:.1f}:1  "
          f"({results_dict[best_q]['saving_pct']:.0f}% nhỏ hơn PCM) – chất lượng thấp")
    print(f"    OGG Q={hi_q} : ratio={hi_rat:.1f}:1  "
          f"({results_dict[hi_q]['saving_pct']:.0f}% nhỏ hơn PCM) – chất lượng cao")
    print(f"    PCM 16-bit   : ratio=1.0:1 – lossless reference")
    print(f"    → OGG là lossy: không thể phục hồi thông tin đã mất khi giải nén")

    fig.suptitle(f"Phần G – So sánh Mã hóa PCM vs OGG/Vorbis  |  {filename_base}",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    if save:
        savefig(fig, "G_compression_ogg.png")


def plot_bitrate_chart(fs: int, channels: int,
                       duration: float, file_size_bytes: int) -> None:
    """Vẽ biểu đồ cột so sánh bit rate các cấu hình (màu gradient xanh → đỏ)."""
    configs = [
        ("PCM 8b 8kHz",    8000,  8, 1),
        ("PCM 8b 16kHz",  16000,  8, 1),
        ("PCM 16b 16kHz", 16000, 16, 1),
        ("PCM 16b 44.1k", 44100, 16, 1),
        ("PCM 16b stereo",44100, 16, 2),
        (f"Gốc ({fs}Hz)",  fs,   16, channels),
    ]
    names  = [c[0] for c in configs]
    kbps   = [c[1] * c[2] * c[3] / 1000 for c in configs]

    # Thêm bit rate file thực tế
    actual = file_size_bytes * 8 / duration / 1000
    names.append("File thực tế")
    kbps.append(actual)

    # Màu gradient: xanh lá (thấp) → đỏ (cao)
    colors_bar = plt.cm.RdYlGn(np.linspace(0.8, 0.2, len(names)))

    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.bar(range(len(names)), kbps, color=colors_bar,
                  edgecolor="black", linewidth=0.5)

    # Ghi giá trị lên đầu mỗi cột
    for bar, k in zip(bars, kbps):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 5,
                f"{k:.0f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=18, ha="right", fontsize=8)
    style_ax(ax, ylabel="Bit rate (kbps)",
             title="Phần G – So sánh Bit Rate các cấu hình audio")
    ax.set_ylim([0, max(kbps) * 1.15])
    plt.tight_layout()
    savefig(fig, "G_bitrate_chart.png")


# ════════════════════════════════════════════════════════════
# HÀM CHẠY PHẦN G
# G1: lượng tử hóa B=[4,6,8,12,16] → SNR, đồ thị nhiễu, lưu WAV
# G2: resampling về 16kHz và 8kHz → vẽ phổ so sánh, lưu WAV
# G3: tính bit rate / compression ratio, vẽ bar chart
# ════════════════════════════════════════════════════════════

def run(info: dict = None) -> None:
    if info is None:
        info = read_wav(choose_audio_file())

    x   = info["x_mono"]
    fs  = info["fs"]
    dur = info["duration"]
    ch  = info["channels"]
    fsz = info["file_size"]

    print("\n" + "═" * 60)
    print("  PHẦN G – LƯỢNG TỬ HÓA, RESAMPLING VÀ MÃ HÓA")
    print("═" * 60)

    # ── G1: Lượng tử hóa ──
    bit_depths = [4, 6, 8, 12, 16]
    print(f"\n  [G1] Lượng tử hóa đều B = {bit_depths}...")
    qr = run_quantization(x, fs, bit_depths)

    print("\n  Vẽ đồ thị SNR và dạng sóng nhiễu...")
    plot_quantization(x, fs, qr)

    print("\n  Vẽ phổ của nhiễu lượng tử...")
    plot_quantization_spectrum(x, fs, qr, freq_max=float(min(8000, fs // 2)))

    # Lưu file WAV lượng tử với tên thể hiện cấu hình (speech_Xbit.wav / music_Xbit.wav)
    fname_base_g = os.path.splitext(info["filename"])[0]
    if any(k in fname_base_g.lower() for k in ("speech","voice","tts","dialogue","mono")):
        prefix_g = "speech"
    else:
        prefix_g = "music"

    print("\n  Lưu file audio lượng tử...")
    for B in [4, 8, 16]:
        path = write_wav(qr[B]["x_q"], fs, f"{prefix_g}_{B}bit.wav")
        print(f"  → {path}")

    # ── G2: Resampling ──
    # Chỉ resample nếu Fs hiện tại cao hơn target
    target_rates = []
    if fs > 16000:
        target_rates.append(16000)
    if fs > 8000:
        target_rates.append(8000)

    if target_rates:
        print(f"\n  [G2] Resampling {fs}Hz → {target_rates}Hz...")
        resampled = {}
        for fs_t in target_rates:
            print(f"  Đang resample → {fs_t}Hz ...", end=" ", flush=True)
            xr   = resample_audio(x, fs, fs_t)
            resampled[fs_t] = xr
            rname = f"{prefix_g}_resample_{fs_t}hz.wav"
            path  = write_wav(xr, fs_t, rname)
            print(f"OK  ({len(xr):,} mẫu)")
            print(f"  → {path}")

        print("\n  Vẽ so sánh phổ trước/sau resampling...")
        plot_resampling(x, fs, resampled,
                        freq_max=float(min(20000, fs // 2)))
    else:
        print(f"\n  [G2] Fs={fs}Hz ≤ 8kHz, bỏ qua resampling.")

    # ── G3: Bit rate và compression ratio lý thuyết ──
    print(f"\n  [G3] Tính bit rate và compression ratio:")
    print(f"  File: {info['filename']}  |  {dur:.3f}s  |  {fsz:,} bytes")
    bitrate_analysis(fs, ch, dur, fsz)
    plot_bitrate_chart(fs, ch, dur, fsz)

    # ── G4: Nén OGG/Vorbis thực tế ──
    print(f"\n  [G4] Nén OGG/Vorbis thực tế (soundfile)...")
    fname_base = os.path.splitext(info["filename"])[0]   # vd: music_chord_Cmaj
    # Phân loại speech hay music để đặt prefix tên file
    if any(k in fname_base.lower() for k in ("speech","voice","tts","dialogue","mono")):
        prefix = "speech"
    else:
        prefix = "music"
    # Tránh trùng prefix nếu tên file đã có sẵn
    if fname_base.startswith(prefix + "_"):
        ogg_base = fname_base
    else:
        ogg_base = f"{prefix}_{fname_base}"

    ogg_results = export_compressed(x, fs,
                                    filename_base=ogg_base,
                                    qualities=[1, 3, 6, 9])
    if ogg_results:
        plot_compression_comparison(ogg_results, filename_base=fname_base)
        # Lưu WAV gốc với tên chuẩn cho đối chiếu (speech_X.wav / music_X.wav)
        pcm_ref_name = f"{ogg_base}_pcm_ref.wav"
        _pcm_ref = write_wav(x, fs, pcm_ref_name)
        print(f"  → PCM ref: {_pcm_ref}")

    print("\n  ✓ Phần G hoàn thành.\n")


if __name__ == "__main__":
    run()
