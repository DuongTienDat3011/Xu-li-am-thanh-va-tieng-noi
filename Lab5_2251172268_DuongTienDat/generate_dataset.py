"""
generate_dataset.py
===================
CSE457 Lab 5 – Tạo tập dữ liệu thực nghiệm âm thanh và tiếng nói:
- 3 Người nói (spk01, spk02, spk03): 4 enrollment, 1 dev, 2 test cho mỗi người (tổng 21 file clean speech)
- 2 Tín hiệu nhiễu: fan.wav (nhiễu dừng) và cafeteria.wav (nhiễu không dừng / babble)
- 2 Nguồn cho BSS/ICA: source_A.wav và source_B.wav (cùng Fs=16kHz, cùng độ dài)

Sinh viên: Dương Tiến Đạt – MSSV: 2251172268
"""

import os
import wave
import struct
import numpy as np
import pyttsx3
import librosa
import soundfile as sf
from scipy.signal import butter, filtfilt, resample_poly

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "lab5_data")
CLEAN_DIR = os.path.join(DATA_DIR, "clean")
NOISE_DIR = os.path.join(DATA_DIR, "noise")
SEP_DIR = os.path.join(DATA_DIR, "separation")

for d in [CLEAN_DIR, NOISE_DIR, SEP_DIR]:
    os.makedirs(d, exist_ok=True)

FS = 16000

# ═══════════════════════════════════════════════════════════════════════
# 1. KỊCH BẢN VÀ THAM SỐ 3 NGƯỜI NÓI
# ═══════════════════════════════════════════════════════════════════════
SPEAKER_CONFIGS = {
    "spk01": {
        "voice_idx": 0,    # Microsoft David (Male)
        "rate": 140,
        "pitch_factor": 1.0,
        "label": "Speaker 1 (David - Male)"
    },
    "spk02": {
        "voice_idx": 1,    # Microsoft Zira (Female)
        "rate": 155,
        "pitch_factor": 1.0,
        "label": "Speaker 2 (Zira - Female)"
    },
    "spk03": {
        "voice_idx": 0,    # David modified (Youth/Bright Male)
        "rate": 165,
        "pitch_factor": 1.18,
        "label": "Speaker 3 (David Mod - Higher Pitch)"
    }
}

UTTERANCES = {
    "enroll_01": "Artificial intelligence and digital speech processing are transforming our modern world.",
    "enroll_02": "Speech enhancement removes background noise while preserving acoustic speech quality.",
    "enroll_03": "Speaker recognition systems can identify who is speaking from unique voice characteristics.",
    "enroll_04": "Blind source separation separates overlapping speech voices using statistical independence.",
    "dev_01":    "The acoustic laboratory tests speech enhancement models in noisy conference rooms.",
    "test_01":   "Welcome to our speech enhancement and speaker recognition laboratory demonstration.",
    "test_02":   "Independent component analysis successfully separates two simultaneous conversations.",
}


def tts_to_file(text, out_path, voice_idx, rate):
    """Tổng hợp 1 câu đơn lẻ và giải phóng SAPI COM an toàn."""
    eng = pyttsx3.init()
    voices = eng.getProperty('voices')
    eng.setProperty('voice', voices[min(voice_idx, len(voices) - 1)].id)
    eng.setProperty('rate', rate)
    eng.setProperty('volume', 1.0)
    eng.save_to_file(text, out_path)
    eng.runAndWait()
    eng.stop()
    del eng


def synthesize_speech():
    """Tạo tập dữ liệu clean speech chuẩn 16 kHz mono cho cả 3 người nói."""
    print("=== [1/3] Đang tạo dữ liệu clean speech (21 tệp WAV) ===")
    temp_wav = os.path.join(BASE_DIR, "temp_tts.wav")

    for spk_id, cfg in SPEAKER_CONFIGS.items():
        print(f"\n[+] Khởi tạo {cfg['label']}:")
        for utt_key, text in UTTERANCES.items():
            out_filename = f"{spk_id}_{utt_key}.wav"
            final_path = os.path.join(CLEAN_DIR, out_filename)

            # Gọi TTS
            tts_to_file(text, temp_wav, cfg["voice_idx"], cfg["rate"])

            # Đọc lại và resample sang 16kHz
            y, orig_sr = sf.read(temp_wav)
            if orig_sr != FS:
                y = librosa.resample(y, orig_sr=orig_sr, target_sr=FS)

            # Nếu có pitch/vocal tract scaling cho spk03
            if cfg["pitch_factor"] != 1.0:
                # Resample để đổi pitch và timbre
                up = int(round(cfg["pitch_factor"] * 100))
                down = 100
                y = resample_poly(y, up, down)
                # Resample lại về đúng 16kHz
                if len(y) > 0:
                    y = librosa.resample(y, orig_sr=int(FS * cfg["pitch_factor"]), target_sr=FS)

            # Đảm bảo mono float64 và chuẩn hóa nhẹ
            if y.ndim > 1:
                y = np.mean(y, axis=1)
            y = y.astype(np.float64)
            # Peak normalize về 0.85 để giữ headroom
            max_v = np.max(np.abs(y))
            if max_v > 0:
                y = y / max_v * 0.85

            sf.write(final_path, y, FS, subtype='PCM_16')
            dur = len(y) / FS
            print(f"  ✓ {out_filename:<22} | Thời lượng: {dur:.2f}s | Fs: {FS}Hz")

    if os.path.exists(temp_wav):
        os.remove(temp_wav)


def synthesize_noise():
    """Tạo 2 tín hiệu nhiễu: fan.wav (dừng) và cafeteria.wav (không dừng)."""
    print("\n=== [2/3] Đang tạo dữ liệu nhiễu (fan.wav & cafeteria.wav) ===")
    dur = 6.0
    N = int(FS * dur)
    t = np.arange(N) / FS
    rng = np.random.default_rng(seed=42)

    # 1. fan.wav: Nhiễu quạt máy (stationary)
    # Hài tần động cơ: 50, 100, 150, 200 Hz
    fan_hum = (
        0.35 * np.sin(2 * np.pi * 50.0 * t) +
        0.20 * np.sin(2 * np.pi * 100.0 * t + 0.5) +
        0.12 * np.sin(2 * np.pi * 150.0 * t + 1.2) +
        0.08 * np.sin(2 * np.pi * 200.0 * t + 2.0)
    )
    # Tiếng gió dải rộng qua bộ lọc thông thấp (fc=700Hz)
    b_lp, a_lp = butter(4, 700.0 / (FS / 2), btype='low')
    wind = filtfilt(b_lp, a_lp, rng.normal(0, 1, N))
    wind = wind / (np.std(wind) + 1e-6) * 0.45

    fan = fan_hum + wind
    fan = fan / (np.max(np.abs(fan)) + 1e-6) * 0.85
    fan_path = os.path.join(NOISE_DIR, "fan.wav")
    sf.write(fan_path, fan, FS, subtype='PCM_16')
    print(f"  ✓ fan.wav                 | {dur:.2f}s | Stationary Low-frequency Fan Noise")

    # 2. cafeteria.wav: Nhiễu quán cà phê / môi trường tiệc (non-stationary babble)
    b_bp, a_bp = butter(3, [300.0 / (FS / 2), 3500.0 / (FS / 2)], btype='band')
    babble = filtfilt(b_bp, a_bp, rng.normal(0, 1, N))
    # Điều chế nhịp nói tự nhiên (modulating envelope)
    env = 0.5 + 0.4 * np.abs(np.sin(2 * np.pi * 1.8 * t) * np.cos(2 * np.pi * 0.6 * t))
    cafe = babble * env

    # Thêm tiếng va chạm bát đĩa thìa dĩa ngẫu nhiên (transient clinks cao tần)
    b_hi, a_hi = butter(2, [3000.0 / (FS / 2), 6000.0 / (FS / 2)], btype='band')
    clinks = np.zeros(N)
    for _ in range(12):
        idx = rng.integers(int(0.15 * N), int(0.95 * N))
        L_c = int(0.03 * FS)
        if idx + L_c < N:
            clinks[idx:idx + L_c] += rng.normal(0, 0.6, L_c) * np.exp(-np.linspace(0, 6, L_c))
    clinks = filtfilt(b_hi, a_hi, clinks)

    cafeteria = cafe + 0.35 * clinks
    cafeteria = cafeteria / (np.max(np.abs(cafeteria)) + 1e-6) * 0.85
    cafe_path = os.path.join(NOISE_DIR, "cafeteria.wav")
    sf.write(cafe_path, cafeteria, FS, subtype='PCM_16')
    print(f"  ✓ cafeteria.wav           | {dur:.2f}s | Non-stationary Cafeteria Babble Noise")


def synthesize_separation_sources():
    """Tạo 2 nguồn tiếng nói độc lập cho thí nghiệm tách nguồn BSS/ICA."""
    print("\n=== [3/3] Đang chuẩn bị nguồn cho BSS/ICA (source_A.wav & source_B.wav) ===")
    src1_path = os.path.join(CLEAN_DIR, "spk01_test_01.wav")
    src2_path = os.path.join(CLEAN_DIR, "spk02_test_01.wav")

    s1, _ = sf.read(src1_path)
    s2, _ = sf.read(src2_path)

    # Cắt về cùng độ dài
    L = min(len(s1), len(s2))
    s1 = s1[:L]
    s2 = s2[:L]

    s1 = s1 / (np.max(np.abs(s1)) + 1e-6) * 0.85
    s2 = s2 / (np.max(np.abs(s2)) + 1e-6) * 0.85

    out_a = os.path.join(SEP_DIR, "source_A.wav")
    out_b = os.path.join(SEP_DIR, "source_B.wav")
    sf.write(out_a, s1, FS, subtype='PCM_16')
    sf.write(out_b, s2, FS, subtype='PCM_16')
    print(f"  ✓ source_A.wav            | {L} samples ({L/FS:.2f}s) | Speaker 1")
    print(f"  ✓ source_B.wav            | {L} samples ({L/FS:.2f}s) | Speaker 2")


if __name__ == "__main__":
    synthesize_speech()
    synthesize_noise()
    synthesize_separation_sources()
    print("\n✓ Hoàn tất 100% dữ liệu Lab 5 theo đúng quy chuẩn!")
