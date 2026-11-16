"""
lab5_utils.py
=============
CSE457 Lab 5 – Các hàm tiện ích toán học và xử lý âm thanh:
- Đọc/ghi WAV PCM 16-bit, 16kHz
- Tính công suất, RMS, căn chỉnh độ trễ
- Thêm nhiễu ở mức SNR định trước theo công thức alpha
- Đo SNR (dB) và cải thiện SNR (Delta SNR)

Sinh viên: Dương Tiến Đạt – MSSV: 2251172268
"""

import os
import numpy as np
import librosa
import soundfile as sf
from scipy.signal import correlate

def load16k(path):
    """
    Đọc tệp âm thanh và chuẩn hóa về mono 16 kHz, định dạng float64.
    """
    x, sr = librosa.load(path, sr=16000, mono=True)
    return x.astype(np.float64), sr

def save16k(path, x, sr=16000):
    """
    Lưu mảng numpy float thành tệp WAV PCM 16-bit tại sampling rate sr (16kHz).
    Tự động giới hạn (clip) và tránh overflow.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # Clip vào dải an toàn [-1.0, 1.0]
    x_clipped = np.clip(x, -1.0, 1.0)
    sf.write(path, x_clipped, sr, subtype='PCM_16')

def power(x):
    """
    Tính công suất trung bình tín hiệu: P_x = (1/N) * sum(x^2).
    Có epsilon 1e-12 để tránh chia cho 0.
    """
    return np.mean(np.asarray(x, dtype=np.float64)**2) + 1e-12

def rms(x):
    """
    Tính Root Mean Square (RMS): RMS(x) = sqrt(P_x).
    """
    return np.sqrt(power(x))

def rms_normalize(x):
    """
    Chuẩn hóa tín hiệu về RMS = 1.0.
    """
    return x / (rms(x) + 1e-12)

def peak_normalize(x, max_peak=0.95):
    """
    Chuẩn hóa theo biên độ đỉnh (peak normalization).
    """
    p = np.max(np.abs(x))
    if p > 0:
        return x / p * max_peak
    return x

def add_noise_at_snr(s, v, snr_db):
    """
    Mô hình hóa: y[n] = s[n] + alpha * v[n] tại target SNR (dB).
    Công thức: alpha = sqrt( P_s / (P_v * 10^(snr_db / 10)) )
    
    Lưu ý quan trọng:
    - Nếu độ dài nhiễu v ngắn hơn tiếng nói s, lặp lại (tile) nhiễu.
    - Cắt nhiễu đúng bằng độ dài của tiếng nói trước khi tính công suất và cộng.
    - Không peak-normalize ở đây để SNR mục tiêu được bảo toàn chính xác.
    """
    s = np.asarray(s, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    
    if len(v) < len(s):
        v = np.tile(v, int(np.ceil(len(s) / len(v))))
    v = v[:len(s)]
    
    P_s = power(s)
    P_v = power(v)
    
    alpha = np.sqrt(P_s / (P_v * (10.0 ** (snr_db / 10.0))))
    y = s + alpha * v
    return y, alpha

def align_signals(ref, test, max_lag_samples=400):
    """
    Căn chỉnh độ trễ (delay) giữa tín hiệu test và reference bằng cross-correlation.
    Giúp tránh sai số do trễ cửa sổ khi tính SNR_out.
    """
    ref = np.asarray(ref, dtype=np.float64)
    test = np.asarray(test, dtype=np.float64)
    L = min(len(ref), len(test))
    r = ref[:L]
    t = test[:L]
    
    corr = correlate(t, r, mode='full')
    lags = np.arange(-len(r) + 1, len(r))
    
    # Giới hạn tìm kiếm trong khoảng [-max_lag_samples, max_lag_samples]
    valid_mask = (lags >= -max_lag_samples) & (lags <= max_lag_samples)
    corr_subset = corr[valid_mask]
    lags_subset = lags[valid_mask]
    
    best_lag = lags_subset[np.argmax(corr_subset)]
    
    if best_lag > 0:
        # test bị trễ sau ref
        aligned_test = t[best_lag:]
        aligned_ref = r[:len(aligned_test)]
    elif best_lag < 0:
        # test đến sớm hơn ref
        aligned_ref = r[-best_lag:]
        aligned_test = t[:len(aligned_ref)]
    else:
        aligned_ref = r
        aligned_test = t
        
    return aligned_ref, aligned_test, best_lag

def snr_db(ref, test, align=False):
    """
    Đo tỉ số tín hiệu trên nhiễu:
    SNR (dB) = 10 * log10( sum(s^2) / sum((s - s_hat)^2) )
    """
    if align:
        ref, test, _ = align_signals(ref, test)
    else:
        L = min(len(ref), len(test))
        ref = ref[:L]
        test = test[:L]
        
    err = test - ref
    P_ref = np.sum(ref**2)
    P_err = np.sum(err**2) + 1e-12
    return float(10.0 * np.log10(P_ref / P_err))
