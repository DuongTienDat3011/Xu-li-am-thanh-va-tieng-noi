"""
generate_inventory.py
=====================
Tạo và chuẩn hóa Unit Inventory cho hệ thống TTS ghép nối miền hẹp (Xếp hàng ngân hàng/bệnh viện).
Bao gồm 16 từ: mời, khách, hàng, số, không -> chín, đến, quầy.
Mỗi từ có 2 ứng viên (candidates: _01 và _02) để phục vụ giải thuật Unit Selection.
Định dạng: WAV mono, Fs = 16,000 Hz, 16-bit PCM, cắt silence sạch, chuẩn hóa RMS = 0.08.
"""

import os
import sys
import asyncio
import numpy as np
import soundfile as sf
import librosa
import pandas as pd

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# Đường dẫn làm việc
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UNITS_DIR = os.path.join(BASE_DIR, "units")
METADATA_PATH = os.path.join(BASE_DIR, "metadata.csv")

# Danh sách 16 từ vựng miền hẹp
WORDS = [
    ("mời", "moi"),
    ("khách", "khach"),
    ("hàng", "hang"),
    ("số", "so"),
    ("không", "khong"),
    ("một", "mot"),
    ("hai", "hai"),
    ("ba", "ba"),
    ("bốn", "bon"),
    ("năm", "nam"),
    ("sáu", "sau"),
    ("bảy", "bay"),
    ("tám", "tam"),
    ("chín", "chin"),
    ("đến", "den"),
    ("quầy", "quay"),
]

TARGET_SR = 16000
TARGET_RMS = 0.08

def calc_rms(x):
    """Tính Root Mean Square của tín hiệu."""
    return float(np.sqrt(np.mean(x * x) + 1e-12))

def match_rms(x, target=TARGET_RMS):
    """Chuẩn hóa biên độ tín hiệu theo RMS mục tiêu."""
    current_rms = calc_rms(x)
    if current_rms < 1e-6:
        return x
    y = x * (target / current_rms)
    return np.clip(y, -0.99, 0.99)

async def synthesize_word_edge(word, voice, rate="+0%", pitch="+0Hz", out_path="temp.mp3"):
    """Tổng hợp 1 từ tiếng Việt bằng Edge Neural TTS cực kỳ tự nhiên."""
    import edge_tts
    comm = edge_tts.Communicate(word, voice=voice, rate=rate, pitch=pitch)
    await comm.save(out_path)

def generate_fallback_signal(word_vi, variant=1, sr=TARGET_SR):
    """Fallback tổng hợp bằng bộ lọc nguồn (Source-Filter) nếu offline."""
    duration = 0.35 + (0.05 if variant == 2 else 0.0)
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    f0 = 160.0 if variant == 1 else 175.0
    
    # Nguồn xung thanh đới (glottal pulses)
    excitation = np.zeros_like(t)
    period = int(sr / f0)
    excitation[::period] = 1.0
    # Thêm chút unvoiced noise
    noise = np.random.normal(0, 0.05, len(t))
    signal = excitation + noise
    
    # Bộ lọc cộng hưởng (Formant filter xấp xỉ)
    from scipy.signal import butter, lfilter
    b, a = butter(2, [300 / (sr/2), 3400 / (sr/2)], btype='band')
    filtered = lfilter(b, a, signal)
    
    # Cửa sổ envelope làm mượt đầu/cuối
    window = np.hanning(len(filtered))
    return filtered * window

def create_inventory(force_regenerate=False):
    os.makedirs(UNITS_DIR, exist_ok=True)
    metadata_records = []
    
    print("=" * 60)
    print("KHỞI TẠO UNIT INVENTORY CHO LAB 4 (16 KHZ MONO, RMS = 0.08)")
    print("=" * 60)

    # Thử kiểm tra edge-tts có khả dụng không
    use_edge = True
    try:
        import edge_tts
    except ImportError:
        use_edge = False
        print("[-] edge-tts chưa cài đặt, chuyển sang fallback parametric.")

    for vi_word, ascii_label in WORDS:
        for cand in [1, 2]:
            unit_id = f"{ascii_label}_0{cand}"
            wav_filename = f"{unit_id}.wav"
            wav_path = os.path.join(UNITS_DIR, wav_filename)
            
            if os.path.exists(wav_path) and not force_regenerate:
                # Đã tồn tại, đọc lại thông số
                y, sr = sf.read(wav_path)
                dur = len(y) / sr
                cur_rms = calc_rms(y)
                speaker = "vi-VN-HoaiMyNeural" if cand == 1 else "vi-VN-HoaiMyNeural_v2"
                metadata_records.append({
                    "unit_id": unit_id,
                    "label": vi_word,
                    "ascii": ascii_label,
                    "candidate": cand,
                    "speaker": speaker,
                    "duration_s": round(dur, 4),
                    "rms": round(cur_rms, 4),
                    "path": os.path.relpath(wav_path, BASE_DIR).replace("\\", "/")
                })
                continue

            audio_data = None
            speaker_name = "vi-VN-HoaiMyNeural" if cand == 1 else "vi-VN-NamMinhNeural"
            
            if use_edge:
                temp_mp3 = os.path.join(UNITS_DIR, f"temp_{unit_id}.mp3")
                voice = "vi-VN-HoaiMyNeural" if cand == 1 else "vi-VN-NamMinhNeural"
                
                try:
                    asyncio.run(synthesize_word_edge(vi_word, voice, rate="+0%", pitch="+0Hz", out_path=temp_mp3))
                    y, sr = librosa.load(temp_mp3, sr=TARGET_SR, mono=True)
                    yt, _ = librosa.effects.trim(y, top_db=20)
                    audio_data = yt
                    if os.path.exists(temp_mp3):
                        os.remove(temp_mp3)
                except Exception as e:
                    print(f"[!] Lỗi khi gọi edge-tts cho từ '{vi_word}' cand {cand}: {e}. Dùng fallback.")
                    audio_data = None

            if audio_data is None:
                audio_data = generate_fallback_signal(vi_word, variant=cand, sr=TARGET_SR)
                speaker_name = "SourceFilter_Synth"

            # Chuẩn hóa RMS = 0.08
            norm_audio = match_rms(audio_data, target=TARGET_RMS)
            
            # Ghi file WAV 16-bit PCM
            sf.write(wav_path, norm_audio.astype(np.float32), TARGET_SR, subtype="PCM_16")
            
            dur = len(norm_audio) / TARGET_SR
            actual_rms = calc_rms(norm_audio)
            
            metadata_records.append({
                "unit_id": unit_id,
                "label": vi_word,
                "ascii": ascii_label,
                "candidate": cand,
                "speaker": speaker_name,
                "duration_s": round(dur, 4),
                "rms": round(actual_rms, 4),
                "path": os.path.relpath(wav_path, BASE_DIR).replace("\\", "/")
            })
            print(f"[+] Tạo thành công: {wav_filename} ({vi_word}) | dur={dur:.3f}s | rms={actual_rms:.4f}")

    # Ghi metadata.csv
    df = pd.DataFrame(metadata_records)
    df.to_csv(METADATA_PATH, index=False, encoding="utf-8")
    print("\n" + "=" * 60)
    print(f"Đã lưu Metadata ({len(df)} units) vào: {METADATA_PATH}")
    print("=" * 60)
    return df

if __name__ == "__main__":
    create_inventory(force_regenerate=True)
