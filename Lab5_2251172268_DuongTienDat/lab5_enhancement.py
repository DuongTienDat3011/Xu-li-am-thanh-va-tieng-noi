"""
lab5_enhancement.py
===================
CSE457 Lab 5 – Khử nhiễu tiếng nói trong miền STFT:
- Phân tích STFT (win=25ms/400 mẫu, hop=10ms/160 mẫu, NFFT=512)
- Ước lượng phổ công suất nhiễu P_v[k]
- Khử nhiễu phổ (Spectral Subtraction) với tham số over-subtraction beta và spectral floor gamma
- Lọc Wiener (Wiener Filtering) dựa trên tỉ số công suất tiếng nói / (tiếng nói + nhiễu)
- Tái tạo tín hiệu miền thời gian bằng ISTFT và overlap-add

Sinh viên: Dương Tiến Đạt – MSSV: 2251172268
"""

import numpy as np
import librosa
from lab5_utils import snr_db

def compute_noise_psd(noise_signal, n_fft=512, win_length=400, hop_length=160):
    """
    Ước lượng phổ công suất nhiễu P_hat_v[k] từ đoạn mẫu nhiễu:
    P_hat_v[k] = (1/M) * sum_{m=1}^M |N[m, k]|^2
    Trả về mảng 2D shape (n_fft//2 + 1, 1).
    """
    N = librosa.stft(
        noise_signal,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window='hann'
    )
    Pn = np.mean(np.abs(N)**2, axis=1, keepdims=True)
    return Pn

def spectral_subtraction(y, noise_sample, sr=16000, beta=1.5, floor=0.02,
                         n_fft=512, win_length=400, hop_length=160):
    """
    Thuật toán Spectral Subtraction (Boll 1979):
    P_ss[m, k] = max( |Y[m, k]|^2 - beta * P_hat_v[k], floor * P_hat_v[k] )
    S_ss[m, k] = sqrt(P_ss[m, k]) * exp(j * phase(Y[m, k]))
    """
    y = np.asarray(y, dtype=np.float64)
    Y = librosa.stft(
        y,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window='hann'
    )
    Py = np.abs(Y)**2
    phase_Y = np.angle(Y)
    
    # Ước lượng PSD nhiễu
    Pn = compute_noise_psd(noise_sample, n_fft=n_fft, win_length=win_length, hop_length=hop_length)
    
    # Trừ phổ công suất với beta và floor gamma
    Pss = np.maximum(Py - beta * Pn, floor * Pn)
    
    # Tái tạo phổ biên độ và kết hợp pha gốc của noisy signal
    Sss = np.sqrt(Pss) * np.exp(1j * phase_Y)
    
    # ISTFT về miền thời gian
    x_ss = librosa.istft(
        Sss,
        hop_length=hop_length,
        win_length=win_length,
        window='hann',
        length=len(y)
    )
    return x_ss

def wiener_filter(y, noise_sample, sr=16000, n_fft=512, win_length=400, hop_length=160):
    """
    Bộ lọc Wiener không nhân quả (MMSE estimator):
    P_hat_s[m, k] = max( |Y[m, k]|^2 - P_hat_v[k], 0 )
    H[m, k] = P_hat_s[m, k] / ( P_hat_s[m, k] + P_hat_v[k] + epsilon )
    S_w[m, k] = H[m, k] * Y[m, k]
    """
    y = np.asarray(y, dtype=np.float64)
    Y = librosa.stft(
        y,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window='hann'
    )
    Py = np.abs(Y)**2
    
    # Ước lượng PSD nhiễu
    Pn = compute_noise_psd(noise_sample, n_fft=n_fft, win_length=win_length, hop_length=hop_length)
    
    # Ước lượng công suất tiếng nói Ps
    Ps = np.maximum(Py - Pn, 0.0)
    
    # Hàm truyền Wiener H[m, k]
    H = Ps / (Ps + Pn + 1e-12)
    
    # Lọc phổ
    Sw = H * Y
    
    # ISTFT về miền thời gian
    x_w = librosa.istft(
        Sw,
        hop_length=hop_length,
        win_length=win_length,
        window='hann',
        length=len(y)
    )
    return x_w, H

def enhance_stft(y, noise_sample, sr=16000, beta=1.5, floor=0.02,
                 n_fft=512, win_length=400, hop_length=160):
    """
    Hàm tổng hợp trả về đồng thời cả 2 kết quả Spectral Subtraction và Wiener.
    """
    x_ss = spectral_subtraction(y, noise_sample, sr=sr, beta=beta, floor=floor,
                                n_fft=n_fft, win_length=win_length, hop_length=hop_length)
    x_w, _ = wiener_filter(y, noise_sample, sr=sr,
                           n_fft=n_fft, win_length=win_length, hop_length=hop_length)
    return x_ss, x_w
