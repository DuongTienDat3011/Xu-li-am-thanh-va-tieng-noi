"""
lab4_prosody.py
===============
Mô đun phân tích và biến điệu ngữ điệu (Prosody) & Mô hình Source-Filter cho Lab 4:
  - Đo tần số cơ bản F0 trên đoạn hữu thanh (Voiced segment) qua hàm tự tương quan (Autocorrelation).
  - Kiểm chứng công thức F0 ≈ Fs / Np trên thực nghiệm.
  - Điều chỉnh tốc độ / duration (0.8x, 1.2x) bằng thuật toán Time-Stretch (bảo toàn cao độ).
  - Biến điệu cao độ (Pitch Shift ±2 semitones) dựa trên tỷ số r = 2^(delta_s / 12) (bảo toàn độ dài và formant).
  - Phân tích và trực quan hóa Spectrogram theo lý thuyết Source-Filter.
"""

import sys
import os
import numpy as np
import soundfile as sf
import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from lab4_utils import load_audio, compute_log_spec

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
AUDIO_OUT_DIR = os.path.join(BASE_DIR, "audio_output")
FIG_DIR = os.path.join(BASE_DIR, "figures")
UNITS_DIR = os.path.join(BASE_DIR, "units")

def estimate_f0_autocorr(segment: np.ndarray, sr: int = 16000, fmin: float = 70.0, fmax: float = 400.0):
    """
    Ước lượng pitch period Np và F0 bằng hàm tự tương quan cực đại (Autocorrelation peak).
    Np = argmax_{k} R_xx[k] -> F0 = Fs / Np.
    """
    # Trừ DC offset và nhân cửa sổ Hanning
    x = segment - np.mean(segment)
    w = np.hanning(len(x))
    xw = x * w
    
    # Tính tự tương quan qua FFT
    nfft = 2 ** int(np.ceil(np.log2(2 * len(xw) - 1)))
    X = np.fft.rfft(xw, n=nfft)
    rxx = np.fft.irfft(np.abs(X) ** 2, n=nfft)[:len(xw)]
    rxx = rxx / (rxx[0] + 1e-12)  # Chuẩn hóa về [0, 1]
    
    # Giới hạn tìm kiếm theo dải tần fmin, fmax
    k_min = int(sr / fmax)
    k_max = int(sr / fmin)
    k_max = min(k_max, len(rxx) - 1)
    
    peak_idx = np.argmax(rxx[k_min:k_max]) + k_min
    np_samples = peak_idx
    f0_est = sr / np_samples if np_samples > 0 else 0.0
    
    return float(f0_est), int(np_samples), rxx, (k_min, k_max)

def track_f0_contour(audio: np.ndarray, sr: int = 16000):
    """Trích xuất F0 contour theo thời gian bằng thuật toán probabilistic YIN (pYIN)."""
    f0, voiced_flag, voiced_probs = librosa.pyin(
        audio,
        fmin=70,
        fmax=400,
        sr=sr,
        frame_length=1024,
        hop_length=160
    )
    times = librosa.times_like(f0, sr=sr, hop_length=160)
    return times, f0, voiced_flag

def run_prosody_experiments(sample_audio_path: str = None):
    """
    Thực hiện trọn vẹn thí nghiệm Prosody & Source-Filter của Lab 4:
      1. Phân tích F0 trên unit hữu thanh (ví dụ 'moi_01.wav' hoặc 'best_concat.wav')
      2. Time-stretch (0.8x, 1.2x)
      3. Pitch shift (-2, +2 semitones)
      4. Vẽ biểu đồ kiểm chứng và so sánh phổ
    """
    os.makedirs(AUDIO_OUT_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)
    
    # Sử dụng best_concat.wav hoặc moi_01.wav
    if sample_audio_path is None or not os.path.exists(sample_audio_path):
        sample_audio_path = os.path.join(AUDIO_OUT_DIR, "best_concat.wav")
        if not os.path.exists(sample_audio_path):
            sample_audio_path = os.path.join(UNITS_DIR, "moi_01.wav")
            
    signal = load_audio(sample_audio_path, sr=16000)
    sr = 16000
    
    print("=" * 60)
    print("THÍ NGHIỆM PROSODY & SOURCE-FILTER MODEL (F0, SPEED, PITCH)")
    print("=" * 60)
    
    # --- 1. Đo F0 trên đoạn nguyên âm hữu thanh ---
    # Chọn một đoạn nguyên âm ổn định dài ~50 ms (800 mẫu)
    # Lấy đoạn trong từ 'mời'
    voiced_unit_path = os.path.join(UNITS_DIR, "moi_01.wav")
    voiced_wave = load_audio(voiced_unit_path, sr=sr)
    start_samp = int(len(voiced_wave) * 0.4)
    frame_len = 800  # 50 ms
    voiced_seg = voiced_wave[start_samp : start_samp + frame_len]
    
    f0_est, np_samples, rxx, (k_min, k_max) = estimate_f0_autocorr(voiced_seg, sr=sr)
    print(f"[+] Phân tích chu kỳ pitch đoạn hữu thanh:")
    print(f"    - Tần số lấy mẫu Fs = {sr} Hz")
    print(f"    - Chu kỳ pitch Np = {np_samples} mẫu ({np_samples/sr*1000:.2f} ms)")
    print(f"    - F0 ước lượng = Fs / Np = {sr} / {np_samples} = {f0_est:.2f} Hz")
    
    # Trích xuất F0 contour cho toàn bộ câu
    times_f0, f0_track, _ = track_f0_contour(signal, sr=sr)
    valid_f0 = f0_track[~np.isnan(f0_track)]
    f0_mean = float(np.mean(valid_f0)) if len(valid_f0) > 0 else 0.0
    print(f"[+] F0 trung bình toàn câu: {f0_mean:.2f} Hz (Dải hữu thanh: {np.min(valid_f0):.1f} - {np.max(valid_f0):.1f} Hz)")
    
    # Vẽ Figure 5: F0 Autocorrelation và F0 Contour
    fig5_path = os.path.join(FIG_DIR, "fig5_prosody_pitch_f0.png")
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(11, 7.5))
    plt.subplots_adjust(hspace=0.35)
    
    # Panel 1: Dạng sóng đoạn hữu thanh
    t_seg = np.arange(len(voiced_seg)) / sr * 1000
    ax1.plot(t_seg, voiced_seg, color="#1f77b4", linewidth=1.2)
    ax1.set_title("Đoạn nguyên âm hữu thanh (Voiced Segment - 'mời')", fontsize=10, fontweight="bold")
    ax1.set_xlabel("Thời gian (ms)", fontsize=9)
    ax1.set_ylabel("Biên độ", fontsize=9)
    ax1.grid(True, linestyle=":", alpha=0.6)
    
    # Panel 2: Hàm tự tương quan Autocorrelation R_xx[k]
    lags = np.arange(len(rxx[:600]))
    ax2.plot(lags, rxx[:600], color="#2ca02c", linewidth=1.2)
    ax2.axvline(x=np_samples, color="#d62728", linestyle="--", linewidth=1.5,
                label=f"Đỉnh Pitch Np = {np_samples} mẫu (F0 = {f0_est:.1f} Hz)")
    ax2.axvspan(k_min, k_max, color="#ffeb3b", alpha=0.25, label="Khoảng tìm kiếm (70 - 400 Hz)")
    ax2.set_title("Hàm tự tương quan (Autocorrelation) và vị trí chu kỳ pitch Np", fontsize=10, fontweight="bold")
    ax2.set_xlabel("Độ trễ lag k (mẫu)", fontsize=9)
    ax2.set_ylabel("R_xx[k] chuẩn hóa", fontsize=9)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="upper right")
    
    # Panel 3: F0 contour của toàn câu
    ax3.plot(times_f0, f0_track, color="#9467bd", linewidth=1.8, marker=".", markersize=3, label="F0 Contour (Hz)")
    ax3.axhline(y=f0_mean, color="#e74c3c", linestyle=":", linewidth=1.2, label=f"F0 Mean = {f0_mean:.1f} Hz")
    ax3.set_title("Đường cong tần số cơ bản (F0 Contour) toàn câu", fontsize=10, fontweight="bold")
    ax3.set_xlabel("Thời gian (giây)", fontsize=9)
    ax3.set_ylabel("Tần số (Hz)", fontsize=9)
    ax3.set_ylim(50, 350)
    ax3.grid(True, linestyle=":", alpha=0.6)
    ax3.legend(loc="upper right")
    
    plt.tight_layout()
    plt.savefig(fig5_path, dpi=180)
    plt.close(fig)
    print(f"[+] Đã lưu biểu đồ F0 Tracking: {fig5_path}")
    
    # --- 2. Thí nghiệm Time-Stretch (Tốc độ 0.8x và 1.2x) ---
    # Tốc độ 0.8x -> nói chậm hơn, thời gian dài hơn: rate=0.8
    # Tốc độ 1.2x -> nói nhanh hơn, thời gian ngắn hơn: rate=1.2
    y_slow = librosa.effects.time_stretch(signal, rate=0.8)
    y_fast = librosa.effects.time_stretch(signal, rate=1.2)
    
    path_slow = os.path.join(AUDIO_OUT_DIR, "speed_0.8x.wav")
    path_fast = os.path.join(AUDIO_OUT_DIR, "speed_1.2x.wav")
    sf.write(path_slow, y_slow, sr)
    sf.write(path_fast, y_fast, sr)
    print(f"[+] Đã tạo audio speed_0.8x.wav ({len(y_slow)/sr:.2f}s) và speed_1.2x.wav ({len(y_fast)/sr:.2f}s)")
    
    # --- 3. Thí nghiệm Pitch Shift (±2 semitones) ---
    # delta_s = +2 semitone -> r = 2^(2/12) = 1.1225
    # delta_s = -2 semitone -> r = 2^(-2/12) = 0.8909
    y_pitch_up = librosa.effects.pitch_shift(signal, sr=sr, n_steps=2)
    y_pitch_down = librosa.effects.pitch_shift(signal, sr=sr, n_steps=-2)
    
    path_pup = os.path.join(AUDIO_OUT_DIR, "pitch_plus2.wav")
    path_pdown = os.path.join(AUDIO_OUT_DIR, "pitch_minus2.wav")
    sf.write(path_pup, y_pitch_up, sr)
    sf.write(path_pdown, y_pitch_down, sr)
    print(f"[+] Đã tạo audio pitch_plus2.wav và pitch_minus2.wav")
    
    # --- 4. Vẽ biểu đồ so sánh Spectrogram (Figure 6) ---
    fig6_path = os.path.join(FIG_DIR, "fig6_prosody_speed_pitch_spectrogram.png")
    fig, axes = plt.subplots(5, 1, figsize=(11, 10), sharex=False)
    plt.subplots_adjust(hspace=0.45)
    
    signals_map = [
        ("Gốc (Original 1.0x, 0 semitones)", signal),
        ("Tốc độ chậm (Speed 0.8x - Thời gian dài hơn 25%)", y_slow),
        ("Tốc độ nhanh (Speed 1.2x - Thời gian ngắn hơn 17%)", y_fast),
        ("Tăng cao độ (+2 Semitones, r = 1.1225)", y_pitch_up),
        ("Giảm cao độ (-2 Semitones, r = 0.8909)", y_pitch_down),
    ]
    
    for ax, (lbl, sig) in zip(axes, signals_map):
        log_sp = compute_log_spec(sig, sr=sr)
        img = librosa.display.specshow(log_sp, sr=sr, hop_length=160, x_axis="time", y_axis="hz", ax=ax, cmap="magma")
        ax.set_title(lbl, fontsize=10, fontweight="bold")
        ax.set_ylabel("Tần số (Hz)", fontsize=8)
        ax.set_ylim(0, 6000)
        
    axes[-1].set_xlabel("Thời gian (giây)", fontsize=10, fontweight="bold")
    plt.tight_layout()
    plt.savefig(fig6_path, dpi=180)
    plt.close(fig)
    print(f"[+] Đã lưu biểu đồ Spectrogram Biến điệu Prosody: {fig6_path}")
    
    return {
        "f0_estimated": f0_est,
        "np_samples": np_samples,
        "f0_mean": f0_mean,
        "path_slow": path_slow,
        "path_fast": path_fast,
        "path_pup": path_pup,
        "path_pdown": path_pdown
    }

if __name__ == "__main__":
    run_prosody_experiments()
