"""
generate_dataset.py
===================
CSE457 Lab 2 – Tạo dataset 5 từ tiếng Việt (tổng hợp bằng source-filter model)
Vocabulary: khong (không), mot (một), hai (hai), ba (ba), bon (bốn)

Mỗi từ ghi 5 lần với biến thể tự nhiên về:
  - Pitch F0 (±15%)
  - Tốc độ nói (±20%)
  - Biên độ (±10%)
  - Jitter/shimmer
  - Noise nền nhỏ

Phục vụ Lab 2: MFCC + DTW nhận dạng từ đơn
Sinh viên: Dương Tiến Đạt – MSSV: 2251172268
"""

import os, wave, struct
import numpy as np
from scipy.signal import lfilter, resample_poly
from math import gcd

FS      = 16000   # 16 kHz – chuẩn Lab 2
AUDIO   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset")

# ═══════════════════════════════════════════════════════════════
# ĐỊNH NGHĨA PHONEME VÀ ĐẶC TRƯNG ÂM HỌC CHO 5 TỪ TIẾNG VIỆT
# ═══════════════════════════════════════════════════════════════
# Mỗi từ được mô hình hóa như chuỗi phoneme:
#   (loại, f0_Hz, dur_s, [(f1,bw1),(f2,bw2),(f3,bw3)], pre_pause, post_pause)
# loại: 'voiced' | 'unvoiced' | 'nasal' | 'stop_voiced' | 'stop_unvoiced'

WORDS = {
    # "không" = /kh/ (fricative) + /ô/ (vowel rounded) + /ng/ (nasal)
    "khong": [
        ("unvoiced",  0,   0.06, [(2500,600),(4500,800),(6000,900)],  0.28, 0.0),
        ("voiced",    130, 0.18, [(450, 80), (700, 90), (2400,180)],  0.0,  0.0),
        ("nasal",     130, 0.12, [(250, 60), (2200,200),(3200,250)],  0.0,  0.30),
    ],
    # "một" = /m/ (nasal) + /ô/ (vowel) + /t/ (stop, closure → burst)
    "mot": [
        ("nasal",     140, 0.10, [(250, 60), (2200,200),(3200,250)],  0.28, 0.0),
        ("voiced",    140, 0.16, [(450, 80), (700, 90), (2400,180)],  0.0,  0.0),
        ("stop_unvoiced", 0, 0.06, [(1500,600),(3500,700),(6000,800)], 0.0, 0.28),
    ],
    # "hai" = /h/ (fricative) + /a/ (open vowel) + /i/ (high front vowel)
    "hai": [
        ("unvoiced",  0,   0.05, [(2000,500),(4000,700),(6000,900)],  0.28, 0.0),
        ("voiced",    135, 0.16, [(730, 90), (1090,110),(2440,170)],  0.0,  0.0),
        ("voiced",    135, 0.12, [(270, 80), (2290,200),(3010,250)],  0.0,  0.28),
    ],
    # "ba" = /b/ (voiced stop) + /a/ (open vowel)
    "ba": [
        ("stop_voiced", 120, 0.06, [(200,200),(1000,300),(2500,400)], 0.28, 0.0),
        ("voiced",    120, 0.24, [(730, 90), (1090,110),(2440,170)],  0.0,  0.30),
    ],
    # "bốn" = /b/ (voiced stop) + /ô/ (rounded) + /n/ (nasal alveolar)
    "bon": [
        ("stop_voiced", 125, 0.06, [(200,200),(1000,300),(2500,400)], 0.28, 0.0),
        ("voiced",    125, 0.15, [(450, 80), (700, 90), (2400,180)],  0.0,  0.0),
        ("nasal",     125, 0.12, [(280, 70), (2400,180),(3100,240)],  0.0,  0.30),
    ],
}

# ═══════════════════════════════════════════════════════════════
# HÀM TỔNG HỢP ÂM THANH
# ═══════════════════════════════════════════════════════════════

def _glottal_source(f0: float, n: int, fs: int, rng) -> np.ndarray:
    """Tạo nguồn glottal (chuỗi xung) với jitter và shimmer."""
    src    = np.zeros(n)
    period = fs / f0
    t      = 0.0
    while t < n:
        idx = int(t)
        if idx < n:
            jitter  = rng.uniform(-0.02, 0.02)   # ±2% pitch jitter
            shimmer = rng.uniform(0.95,  1.05)    # ±5% amplitude shimmer
            src[idx] += shimmer
        t += period * (1.0 + jitter)
    return src


def _formant_filter(src: np.ndarray, formants: list, fs: int) -> np.ndarray:
    """Lọc nguồn qua chuỗi cộng hưởng formant (IIR bậc 2)."""
    out = np.zeros(len(src))
    for (fc, bw) in formants:
        if fc >= fs / 2:
            continue
        r     = np.exp(-np.pi * bw / fs)
        theta = 2.0 * np.pi * fc / fs
        b     = np.array([1.0 - r])
        a     = np.array([1.0, -2.0 * r * np.cos(theta), r * r])
        out  += lfilter(b, a, src)
    return out


def _synthesis_noise(n: int, formants: list, fs: int, rng) -> np.ndarray:
    """Tổng hợp âm vô thanh (nhiễu qua bandpass)."""
    noise = rng.standard_normal(n) * 0.5
    return _formant_filter(noise, formants, fs)


def _fade(sig: np.ndarray, fs: int,
          fade_in_ms: float = 10.0, fade_out_ms: float = 15.0) -> np.ndarray:
    """Áp fade-in/out để tránh click."""
    fi = min(int(fade_in_ms  * 1e-3 * fs), len(sig) // 4)
    fo = min(int(fade_out_ms * 1e-3 * fs), len(sig) // 4)
    env = np.ones(len(sig))
    env[:fi]           = np.linspace(0, 1, fi)
    env[len(sig)-fo:]  = np.linspace(1, 0, fo)
    return sig * env


def synthesize_word(phonemes: list, fs: int,
                    speed: float = 1.0,
                    f0_scale: float = 1.0,
                    amp_scale: float = 1.0,
                    room_snr_db: float = 35.0,
                    rng=None) -> np.ndarray:
    """
    Tổng hợp 1 utterance từ danh sách phoneme.

    Parameters
    ----------
    phonemes   : list of (type, f0, dur_s, formants, pre_pause, post_pause)
    speed      : hệ số tốc độ (< 1 = chậm hơn, > 1 = nhanh hơn)
    f0_scale   : nhân vào f0 (mô phỏng biến đổi pitch giữa lần nói)
    amp_scale  : nhân vào biên độ
    room_snr_db: mức nhiễu nền (dB)
    """
    if rng is None:
        rng = np.random.default_rng(42)

    segments = []

    for (ptype, f0, dur_s, formants, pre_pause, post_pause) in phonemes:
        # Thêm khoảng lặng trước phoneme (silence đầu từ)
        if pre_pause > 0:
            n_sil = int(pre_pause * fs)
            segments.append(np.zeros(n_sil))

        # Số mẫu phoneme (điều chỉnh theo tốc độ nói)
        n = max(32, int(dur_s / speed * fs))

        # Tổng hợp theo loại phoneme
        if ptype == "voiced":
            f0_cur = f0 * f0_scale
            src    = _glottal_source(f0_cur, n, fs, rng)
            seg    = _formant_filter(src, formants, fs)

        elif ptype == "nasal":
            f0_cur = f0 * f0_scale
            src    = _glottal_source(f0_cur, n, fs, rng)
            # Nasal: cộng hưởng thấp + anti-formant mũi (giả lập đơn giản)
            seg    = _formant_filter(src, formants, fs)
            seg   *= 0.6   # nasal thường có biên độ thấp hơn

        elif ptype == "stop_voiced":
            # Voiced stop: burst ngắn + giọng
            burst_n = int(0.02 * fs)
            burst   = rng.standard_normal(burst_n) * 0.3
            burst   = _formant_filter(burst, [(1200,800),(2800,1000)], fs)
            f0_cur  = f0 * f0_scale
            voiced  = _glottal_source(f0_cur, n - burst_n, fs, rng)
            voiced  = _formant_filter(voiced, formants, fs)
            seg     = np.concatenate([_fade(burst, fs, 5, 5), voiced])

        elif ptype in ("unvoiced", "stop_unvoiced"):
            seg = _synthesis_noise(n, formants, fs, rng)
            if ptype == "stop_unvoiced":
                # closure: giảm năng lượng dần rồi burst nhỏ
                closure_n = n // 2
                burst_n   = n - closure_n
                closure   = np.zeros(closure_n)
                burst     = _synthesis_noise(burst_n, formants, fs, rng) * 1.2
                seg       = np.concatenate([closure, burst])

        else:
            seg = np.zeros(n)

        # Fade in/out cho phoneme
        seg = _fade(seg, fs, fade_in_ms=8, fade_out_ms=12)
        segments.append(seg)

        # Khoảng lặng sau phoneme (silence cuối từ)
        if post_pause > 0:
            n_sil = int(post_pause * fs)
            segments.append(np.zeros(n_sil))

    if not segments:
        return np.zeros(int(0.5 * fs))

    signal = np.concatenate(segments).astype(np.float64)

    # Thêm room noise
    sig_power  = np.mean(signal ** 2) + 1e-10
    noise_std  = np.sqrt(sig_power / (10 ** (room_snr_db / 10)))
    signal    += rng.standard_normal(len(signal)) * noise_std

    # Nhân biên độ và normalize
    signal *= amp_scale
    peak    = np.max(np.abs(signal))
    if peak > 0:
        signal = signal / peak * 0.85

    return signal


def save_wav_16k(signal: np.ndarray, fs: int, path: str) -> None:
    """Lưu WAV 16-bit mono 16 kHz."""
    # Resample nếu cần
    if fs != FS:
        g      = gcd(fs, FS)
        signal = resample_poly(signal, FS // g, fs // g).astype(np.float64)

    s16 = (np.clip(signal, -1.0, 1.0) * 32767).astype(np.int16)
    with wave.open(path, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(FS)
        wf.writeframes(struct.pack(f"<{len(s16)}h", *s16))


# ═══════════════════════════════════════════════════════════════
# SINH 5 LẦN LẶP CHO MỖI TỪ VỚI BIẾN THỂ TỰ NHIÊN
# ═══════════════════════════════════════════════════════════════

# Biến thể cho 5 lần lặp – mô phỏng tốc độ và pitch tự nhiên
VARIATIONS = [
    # (speed, f0_scale, amp_scale, room_snr_db, seed)
    (1.00, 1.00, 1.00, 38, 101),   # lần 1: chuẩn
    (0.90, 1.08, 0.95, 35, 202),   # lần 2: chậm hơn, pitch cao hơn
    (1.12, 0.94, 1.05, 40, 303),   # lần 3: nhanh hơn, pitch thấp hơn
    (0.95, 1.05, 0.98, 33, 404),   # lần 4: pitch cao nhẹ, noise nhiều hơn
    (1.08, 0.97, 1.02, 42, 505),   # lần 5: nhanh nhẹ
]


def generate_all() -> None:
    print("=" * 60)
    print("  CSE457 Lab 2 – Tạo dataset 5 từ × 5 lần")
    print("  Vocabulary: khong, mot, hai, ba, bon")
    print("=" * 60)

    total = 0
    for word, phonemes in WORDS.items():
        folder = os.path.join(AUDIO, word)
        os.makedirs(folder, exist_ok=True)
        print(f"\n  [{word.upper()}]")

        for rep, (speed, f0s, amps, snr, seed) in enumerate(VARIATIONS, 1):
            rng      = np.random.default_rng(seed)
            signal   = synthesize_word(phonemes, FS,
                                       speed=speed, f0_scale=f0s,
                                       amp_scale=amps, room_snr_db=snr,
                                       rng=rng)
            fname    = f"{word}_{rep:02d}.wav"
            fpath    = os.path.join(folder, fname)
            save_wav_16k(signal, FS, fpath)
            dur      = len(signal) / FS
            fsize    = os.path.getsize(fpath)
            print(f"    {fname}  {dur:.3f}s  {fsize//1024}KB")
            total   += 1

    print(f"\n  ✓ Tổng: {total} file WAV trong {AUDIO}")
    print("  Train: file _01 _02 _03 → template (3/từ)")
    print("  Test : file _04 _05     → evaluation (2/từ)")


if __name__ == "__main__":
    generate_all()
