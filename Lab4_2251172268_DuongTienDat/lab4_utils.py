"""
lab4_utils.py
=============
Các hàm tiện ích dùng chung cho toàn bộ Lab 4:
  - Đọc/ghi audio chuẩn 16 kHz mono.
  - Chuẩn hóa RMS / biên độ.
  - Vẽ biểu đồ Waveform và Spectrogram có đánh dấu biên ghép nối (boundaries).
  - Trích xuất đặc trưng phổ và tính toán sai khác biên (boundary discontinuity).
"""

import sys
import os
import numpy as np
import soundfile as sf
import librosa
import matplotlib
matplotlib.use("Agg")  # Chế độ headless cho matplotlib
import matplotlib.pyplot as plt

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

def load_audio(path: str, sr: int = 16000) -> np.ndarray:
    """Đọc file audio, chuyển về mono và kiểm tra sampling rate."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Không tìm thấy file: {path}")
    x, fs = sf.read(path, dtype="float32")
    if x.ndim > 1:
        x = x.mean(axis=1)
    if fs != sr:
        x = librosa.resample(x, orig_sr=fs, target_sr=sr)
    return x

def calc_rms(x: np.ndarray) -> float:
    """Tính giá trị RMS (Root Mean Square) của tín hiệu."""
    return float(np.sqrt(np.mean(x * x) + 1e-12))

def match_rms(x: np.ndarray, target: float = 0.08) -> np.ndarray:
    """Điều chỉnh năng lượng của tín hiệu về mức RMS mục tiêu."""
    current_rms = calc_rms(x)
    if current_rms < 1e-7:
        return x
    scaled = x * (target / current_rms)
    return np.clip(scaled, -0.99, 0.99)

def compute_log_spec(x: np.ndarray, sr: int = 16000, n_fft: int = 512, hop_length: int = 160):
    """Tính log-magnitude spectrogram."""
    stft = librosa.stft(x, n_fft=n_fft, hop_length=hop_length, window="hann")
    mag = np.abs(stft)
    log_spec = librosa.amplitude_to_db(mag, ref=np.max)
    return log_spec

def plot_waveform_and_spectrogram(
    signal: np.ndarray,
    sr: int = 16000,
    boundaries_samples: list[int] = None,
    boundary_labels: list[str] = None,
    title: str = "Waveform and Spectrogram",
    save_path: str = None
):
    """
    Vẽ đồng bộ Waveform và Spectrogram, đánh dấu các đường biên nối đơn vị (vertical lines).
    """
    time_axis = np.arange(len(signal)) / sr
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    plt.subplots_adjust(hspace=0.25)
    
    # 1. Waveform
    ax1.plot(time_axis, signal, color="#1f77b4", linewidth=0.8, alpha=0.9)
    ax1.set_ylabel("Biên độ (Amplitude)", fontsize=10, fontweight="bold")
    ax1.set_title(f"{title} - Dạng sóng (Waveform)", fontsize=11, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.set_ylim(-1.05, 1.05)
    
    # 2. Spectrogram
    log_spec = compute_log_spec(signal, sr=sr)
    img = librosa.display.specshow(
        log_spec,
        sr=sr,
        hop_length=160,
        x_axis="time",
        y_axis="hz",
        ax=ax2,
        cmap="magma"
    )
    ax2.set_xlabel("Thời gian (giây)", fontsize=10, fontweight="bold")
    ax2.set_ylabel("Tần số (Hz)", fontsize=10, fontweight="bold")
    ax2.set_title("Phổ tần (Spectrogram)", fontsize=11, fontweight="bold")
    fig.colorbar(img, ax=ax2, format="%+2.0f dB", label="Cường độ (dB)")
    
    # Đánh dấu các biên ghép nối
    if boundaries_samples:
        for i, b_samp in enumerate(boundaries_samples):
            b_sec = b_samp / sr
            ax1.axvline(x=b_sec, color="#d62728", linestyle="--", linewidth=1.2, alpha=0.85)
            ax2.axvline(x=b_sec, color="#00ffcc", linestyle="--", linewidth=1.2, alpha=0.85)
            if boundary_labels and i < len(boundary_labels):
                ax1.text(b_sec + 0.01, 0.75, boundary_labels[i], color="#d62728", fontsize=9, fontweight="bold",
                         bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7, edgecolor="#d62728"))
    
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=180)
        plt.close(fig)
        print(f"[+] Đã lưu biểu đồ: {save_path}")
    else:
        plt.close(fig)

def calc_boundary_discontinuity(x1: np.ndarray, x2: np.ndarray, sr: int = 16000, ms: int = 25) -> float:
    """
    Tính độ chênh phổ (Log Spectral Distance - LSD) giữa 25 ms cuối của unit 1 và 25 ms đầu của unit 2.
    """
    n = int(sr * ms / 1000)
    if len(x1) < n or len(x2) < n:
        return 0.0
    w = np.hanning(n)
    fft_1 = np.fft.rfft(x1[-n:] * w, n=512)
    fft_2 = np.fft.rfft(x2[:n] * w, n=512)
    l1 = 20 * np.log10(np.maximum(np.abs(fft_1), 1e-8))
    l2 = 20 * np.log10(np.maximum(np.abs(fft_2), 1e-8))
    return float(np.mean((l1 - l2) ** 2))
