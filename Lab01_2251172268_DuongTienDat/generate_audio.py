"""
generate_audio.py
=================
CSE457 Lab 1 – Tạo file âm thanh từ văn bản (TTS) và tín hiệu tổng hợp.
Sinh viên: Dương Tiến Đạt – MSSV: 2251172268

Chạy:  python generate_audio.py

Các chế độ:
  [1] TEXT-TO-SPEECH  : Nhập văn bản → WAV tiếng nói (pyttsx3 offline)
  [2] HỘI THOẠI 2 NGƯỜI : Tạo đoạn hội thoại có 2 giọng khác nhau
  [3] SYNTHETIC SIGNALS : Tín hiệu tổng hợp (sine, chord, speech-like)
  [0] Thoát
"""

import os, wave, struct, time
import numpy as np
from scipy.signal import lfilter, resample_poly
from math import gcd

# ── Thư mục đầu ra ──────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
AUDIO_DIR = os.path.join(BASE_DIR, "audio")
os.makedirs(AUDIO_DIR, exist_ok=True)

FS        = 22050   # Hz – sampling rate chuẩn speech
DURATION  = 5.0     # s  – cho tín hiệu tổng hợp


# ════════════════════════════════════════════════════════════
# TIỆN ÍCH LƯU WAV
# ════════════════════════════════════════════════════════════

def save_wav_float(signal: np.ndarray, fs: int, filename: str) -> str:
    """Lưu ndarray float64 [-1,1] → WAV PCM 16-bit mono."""
    sig  = np.clip(signal, -1.0, 1.0)
    s16  = (sig * 32767).astype(np.int16)
    path = os.path.join(AUDIO_DIR, filename)
    with wave.open(path, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(fs)
        wf.writeframes(struct.pack(f"<{len(s16)}h", *s16))
    sz  = os.path.getsize(path)
    dur = len(signal) / fs
    print(f"  ✓ {filename:<48}  {dur:.2f}s  {sz//1024}KB")
    return path


# ════════════════════════════════════════════════════════════
# PHẦN 1 – TEXT-TO-SPEECH (pyttsx3 offline)
# ════════════════════════════════════════════════════════════

def tts_pyttsx3(text: str, out_path: str,
                rate: int = 145, voice_index: int = 0) -> bool:
    """
    Tổng hợp giọng nói offline bằng pyttsx3 (Windows SAPI5).
    rate        : tốc độ đọc (từ/phút) – 120=chậm, 145=bình thường, 180=nhanh
    voice_index : 0=David (nam), 1=Zira (nữ)
    """
    try:
        import pyttsx3
        engine = pyttsx3.init()
        voices = engine.getProperty("voices")
        if voices:
            engine.setProperty("voice", voices[min(voice_index, len(voices)-1)].id)
        engine.setProperty("rate",   rate)
        engine.setProperty("volume", 1.0)
        engine.save_to_file(text, out_path)
        engine.runAndWait()
        engine.stop()
        ok = os.path.isfile(out_path) and os.path.getsize(out_path) > 1000
        return ok
    except Exception as e:
        print(f"  [pyttsx3 lỗi] {e}")
        return False


def text_to_speech(text: str, filename: str = None,
                   voice_index: int = 0, rate: int = 145) -> str | None:
    """Chuyển text → WAV. Trả về đường dẫn hoặc None nếu thất bại."""
    if not text.strip():
        return None
    if filename is None:
        safe     = "".join(c if c.isalnum() else "_" for c in text[:25]).strip("_")
        filename = f"speech_tts_{safe}.wav"
    out_path = os.path.join(AUDIO_DIR, filename)
    print(f"  TTS → {filename} ...")
    ok = tts_pyttsx3(text, out_path, rate=rate, voice_index=voice_index)
    if ok:
        fsize = os.path.getsize(out_path)
        with wave.open(out_path, "r") as wf:
            dur = wf.getnframes() / wf.getframerate()
        print(f"  ✓ {filename}  {dur:.2f}s  {fsize//1024}KB")
        return out_path
    print("  ✗ Không tạo được file TTS.")
    return None


def interactive_tts() -> str | None:
    """Menu nhập văn bản → tạo WAV tiếng nói."""
    print("\n" + "═"*60)
    print("  TEXT-TO-SPEECH – Nhập văn bản để tạo file WAV")
    print("═"*60)

    text = input("  Nhập văn bản (Enter = dùng mặc định): ").strip()
    if not text:
        text = ("Hello, this is a test of the text to speech system "
                "for CSE457 Lab 1 audio signal processing.")
        print(f"  (Mặc định): \"{text}\"")

    fname_raw = input("  Tên file đầu ra (Enter = tự động): ").strip()
    filename  = fname_raw if fname_raw.endswith(".wav") else (
        fname_raw + ".wav" if fname_raw else None)

    vi_raw  = input("  Giọng [0=nam David, 1=nữ Zira, Enter=0]: ").strip()
    vi_idx  = int(vi_raw) if vi_raw in ("0","1") else 0

    rate_raw = input("  Tốc độ [120=chậm, 145=thường, 180=nhanh, Enter=145]: ").strip()
    rate     = int(rate_raw) if rate_raw.isdigit() else 145

    return text_to_speech(text, filename=filename,
                          voice_index=vi_idx, rate=rate)


# ════════════════════════════════════════════════════════════
# PHẦN 2 – HỘI THOẠI 2 NGƯỜI
# Tạo file âm thanh hội thoại giữa 2 giọng nói (nam + nữ)
# Mỗi lượt nói được TTS riêng rồi ghép lại với khoảng lặng
# ════════════════════════════════════════════════════════════

# Kịch bản hội thoại: chủ đề âm thanh / xử lý tín hiệu
DIALOGUE_SCRIPT = [
    # (speaker_idx, rate, text_en, label)
    (0, 140, "Hello, today we will discuss digital audio signal processing.", "A"),
    (1, 145, "Great. Can you explain what sampling rate means?", "B"),
    (0, 140, "Sure. Sampling rate is the number of samples taken per second. "
             "For example, CD quality audio uses 44,100 samples per second.", "A"),
    (1, 145, "I see. So a higher sampling rate means better audio quality?", "B"),
    (0, 140, "That is correct, up to the Nyquist limit. The Nyquist theorem says "
             "you need at least twice the frequency you want to capture.", "A"),
    (1, 145, "What about bit depth? How does that affect quality?", "B"),
    (0, 140, "Bit depth determines quantization levels. More bits means less "
             "quantization noise and higher signal to noise ratio.", "A"),
    (1, 145, "So 16 bit audio is better than 8 bit audio.", "B"),
    (0, 140, "Exactly. 16 bit gives about 96 decibels of dynamic range, "
             "while 8 bit only gives around 48 decibels.", "A"),
    (1, 145, "Thank you. That is very helpful for our lab assignment.", "B"),
]

# Câu nói tiếng Việt dùng cho speech_like
VIETNAMESE_LINES = [
    "Xin chào, hôm nay chúng ta học về xử lý âm thanh số.",
    "Tần số lấy mẫu là số lượng mẫu được thu thập trong một giây.",
    "Điều kiện Nyquist yêu cầu tần số lấy mẫu ít nhất bằng hai lần tần số cao nhất.",
    "Độ sâu bit xác định số mức lượng tử hóa trong tín hiệu số.",
    "Bộ lọc thông thấp giữ lại các thành phần tần số thấp và loại bỏ tần số cao.",
]


def create_dialogue(output_filename: str = "speech_dialogue_2person.wav",
                    pause_sec: float = 0.4) -> str:
    """
    Tạo hội thoại 2 người bằng cách:
      1. TTS từng câu với giọng khác nhau (David=nam, Zira=nữ)
      2. Ghép tất cả lại với khoảng lặng pause_sec giây giữa các lượt
      3. Lưu thành 1 file WAV duy nhất

    Đây là âm thanh đối thoại thực sự để phân tích tiếng nói.
    """
    print("\n" + "═"*60)
    print("  TẠO HỘI THOẠI 2 NGƯỜI (David=Nam, Zira=Nữ)")
    print("═"*60)

    segments   = []
    pause_samp = np.zeros(int(pause_sec * FS), dtype=np.float64)
    tmp_files  = []

    for i, (spk, rate, text, lbl) in enumerate(DIALOGUE_SCRIPT):
        tmp_name = f"_tmp_dlg_{i:02d}.wav"
        tmp_path = os.path.join(AUDIO_DIR, tmp_name)
        print(f"  [{lbl}] Người {'Nam' if spk==0 else 'Nữ'}: {text[:55]}...")

        ok = tts_pyttsx3(text, tmp_path, rate=rate, voice_index=spk)
        if not ok:
            print(f"  ✗ Bỏ qua câu {i+1}")
            continue

        # Đọc file WAV tạm
        with wave.open(tmp_path, "r") as wf:
            fs_tmp = wf.getframerate()
            raw    = wf.readframes(wf.getnframes())
        samps = np.frombuffer(raw, dtype=np.int16).astype(np.float64) / 32767.0

        # Resample về FS chung nếu cần
        if fs_tmp != FS:
            g   = gcd(fs_tmp, FS)
            samps = resample_poly(samps, FS//g, fs_tmp//g).astype(np.float64)

        segments.append(samps)
        segments.append(pause_samp.copy())  # Khoảng lặng giữa các lượt
        tmp_files.append(tmp_path)

        time.sleep(0.15)  # Tránh pyttsx3 bị conflict

    if not segments:
        print("  ✗ Không tạo được hội thoại.")
        return None

    # Ghép tất cả segment
    dialogue = np.concatenate(segments)

    # Normalize nhẹ để tránh clipping
    peak = np.max(np.abs(dialogue))
    if peak > 0:
        dialogue = dialogue / peak * 0.85

    out_path = save_wav_float(dialogue, FS, output_filename)

    # Xóa file tạm
    for f in tmp_files:
        if os.path.isfile(f):
            os.remove(f)

    total_dur = len(dialogue) / FS
    print(f"\n  ✓ Hội thoại hoàn thành: {total_dur:.1f}s | {len(DIALOGUE_SCRIPT)} lượt")
    return out_path


def create_single_speech(filename: str = "speech_monologue.wav") -> str:
    """
    Tạo đoạn độc thoại (1 người) – dạng bài giảng ngắn về xử lý âm thanh.
    Dùng giọng nam với tốc độ bình thường.
    """
    monologue = (
        "Welcome to CSE457, Audio and Speech Signal Processing. "
        "In this lab, we analyze digital audio signals. "
        "We will study sampling rate, bit depth, and the Fast Fourier Transform. "
        "The Fourier transform converts a time domain signal to the frequency domain. "
        "This allows us to see which frequencies are present in the audio. "
        "We also apply digital filters to modify the frequency content. "
        "A low pass filter keeps low frequencies and removes high frequencies. "
        "Quantization reduces the number of bits per sample, which adds noise. "
        "Higher bit depth means less quantization noise and better quality. "
        "Thank you for listening."
    )
    return text_to_speech(monologue, filename=filename, voice_index=0, rate=140)


# ════════════════════════════════════════════════════════════
# PHẦN 3 – GIỌNG NÓI TỔNG HỢP THỰC TẾ HƠN
# Dùng mô hình source-filter cải tiến với:
#   - Pitch F0 thay đổi tự nhiên (intonation)
#   - Amplitude modulation (nhịp thở)
#   - Jitter và shimmer (biến động pitch)
#   - Pause ngắn giữa các âm tiết
# ════════════════════════════════════════════════════════════

def make_realistic_speech(fs: int = FS, duration: float = 5.0,
                           gender: str = "male") -> np.ndarray:
    """
    Tạo tín hiệu GIẢ LẬP tiếng nói thực tế hơn dùng mô hình source-filter.

    Cải tiến so với phiên bản đơn giản:
      - F0 (pitch) thay đổi theo thời gian mô phỏng ngữ điệu
      - Jitter (biến động chu kỳ F0 ±2%) để giọng không máy móc
      - Shimmer (biến động biên độ ±5%) tạo cảm giác tự nhiên
      - Khoảng lặng ngắn (8–15% duration) giả lập khoảng dừng hơi
      - Formant F1/F2/F3 thay đổi giữa các âm tiết mô phỏng các nguyên âm

    gender: 'male'  → F0 trung bình 120 Hz, formant thấp hơn
            'female'→ F0 trung bình 200 Hz, formant cao hơn
    """
    N   = int(duration * fs)
    sig = np.zeros(N, dtype=np.float64)
    rng = np.random.default_rng(seed=42 if gender == "male" else 77)

    # ── Thông số theo giới ──
    if gender == "male":
        f0_mean   = 120.0   # Hz – pitch trung bình giọng nam
        f0_range  = 40.0    # Hz – biên độ thay đổi pitch
        formants  = [
            # (F1, F2, F3, BW1, BW2, BW3) – các nguyên âm tiếng nói
            (730,  1090, 2440,  90, 110, 170),   # /a/ – mở
            (270,  2290, 3010,  80, 200, 250),   # /i/ – cao trước
            (520,  1190, 2390,  85, 130, 180),   # /o/ – tròn
            (390,  1990, 2550,  70, 180, 200),   # /e/ – giữa
            (440,  1020, 2240,  80, 120, 160),   # /ɑ/ – mở giữa
        ]
    else:  # female
        f0_mean   = 200.0
        f0_range  = 60.0
        formants  = [
            (850,  1220, 2810,  90, 120, 180),   # /a/ nữ
            (310,  2790, 3310,  90, 250, 280),   # /i/ nữ
            (600,  1170, 2670,  90, 150, 200),   # /o/ nữ
            (460,  2070, 2760,  80, 200, 220),   # /e/ nữ
            (500,  1100, 2500,  85, 130, 170),   # /ɑ/ nữ
        ]

    # ── Tạo F0 contour (đường cong pitch theo thời gian) ──
    # Dùng sinusoid tần số thấp (~0.3Hz) để mô phỏng ngữ điệu lên xuống
    t_full = np.linspace(0, duration, N, endpoint=False)
    f0_contour = f0_mean + f0_range * np.sin(2 * np.pi * 0.3 * t_full)
    # Thêm variation nhanh hơn (~1.5Hz) để giọng tự nhiên hơn
    f0_contour += (f0_range * 0.3) * np.sin(2 * np.pi * 1.5 * t_full + 1.2)
    f0_contour  = np.clip(f0_contour, f0_mean * 0.6, f0_mean * 1.8)

    # ── Chia tín hiệu thành các âm tiết ──
    syl_dur_min = 0.08   # Âm tiết ngắn nhất (ms)
    syl_dur_max = 0.18   # Âm tiết dài nhất
    pause_prob  = 0.12   # Xác suất xuất hiện khoảng lặng

    pos = 0   # Vị trí mẫu hiện tại
    syl_idx = 0

    while pos < N:
        # Thỉnh thoảng chèn khoảng lặng (silence) giữa âm tiết
        if rng.random() < pause_prob and pos > int(0.3 * fs):
            pause_n = int(rng.uniform(0.05, 0.20) * fs)   # 50–200ms
            pos    += min(pause_n, N - pos)
            continue

        # Độ dài âm tiết ngẫu nhiên
        syl_dur = rng.uniform(syl_dur_min, syl_dur_max)
        syl_n   = min(int(syl_dur * fs), N - pos)
        if syl_n <= 0:
            break

        # Chọn ngẫu nhiên 1 bộ formant (nguyên âm)
        f1, f2, f3, bw1, bw2, bw3 = formants[syl_idx % len(formants)]
        syl_idx += 1

        # Voiced (hữu thanh) hay unvoiced (vô thanh, xác suất 15%)
        is_unvoiced = (rng.random() < 0.15)

        # ── Tạo nguồn (source) ──
        if is_unvoiced:
            # Unvoiced: nhiễu trắng (phụ âm /s/, /f/, /sh/)
            source = rng.standard_normal(syl_n) * 0.4
        else:
            # Voiced: chuỗi xung glottal với jitter
            f0_local = float(np.mean(f0_contour[pos:pos+syl_n]))
            source   = np.zeros(syl_n)
            period   = fs / f0_local
            t_syl    = 0.0

            while t_syl < syl_n:
                # Jitter: biến động chu kỳ ±2%
                jitter = rng.uniform(-0.02, 0.02)
                p_curr = period * (1.0 + jitter)
                idx    = int(t_syl)
                if idx < syl_n:
                    # Shimmer: biến động biên độ ±5%
                    shimmer       = rng.uniform(0.95, 1.05)
                    source[idx]  += shimmer
                t_syl += p_curr

        # ── Lọc qua 3 formant (IIR bậc 2) ──
        syl_sig = np.zeros(syl_n)
        for fc, bw in [(f1, bw1), (f2, bw2), (f3, bw3)]:
            if fc >= fs / 2:
                continue
            r     = np.exp(-np.pi * bw / fs)
            theta = 2.0 * np.pi * fc / fs
            b_f   = np.array([1.0 - r])
            a_f   = np.array([1.0, -2.0 * r * np.cos(theta), r * r])
            syl_sig += lfilter(b_f, a_f, source)

        # ── Envelope: fade-in 15ms, fade-out 20ms ──
        fi = min(int(0.015 * fs), syl_n // 4)
        fo = min(int(0.020 * fs), syl_n // 4)
        env = np.ones(syl_n)
        env[:fi] = np.linspace(0, 1, fi)
        env[syl_n-fo:] = np.linspace(1, 0, fo)
        syl_sig *= env

        sig[pos:pos+syl_n] += syl_sig
        pos += syl_n

    # ── Normalize và thêm nhiễu phòng nhỏ (room noise) ──
    peak = np.max(np.abs(sig))
    if peak > 0:
        sig = sig / peak * 0.82
    # Room noise: nhiễu trắng rất nhỏ -50dBFS để giọng nghe tự nhiên
    sig += rng.standard_normal(N) * 0.003

    return np.clip(sig, -1.0, 1.0)


def make_two_speaker_dialogue_synthetic(fs: int = FS,
                                        duration_per_turn: float = 1.2,
                                        n_turns: int = 6,
                                        pause_between: float = 0.35) -> np.ndarray:
    """
    Tạo hội thoại 2 người TỔNG HỢP (không cần TTS):
      - Người A: giọng nam (F0~120Hz)
      - Người B: giọng nữ (F0~200Hz)
      - Xen kẽ nhau, mỗi người nói ~1.2s, nghỉ 0.35s
    Dùng khi TTS không khả dụng hoặc để kiểm tra thuật toán.
    """
    pause_n   = int(pause_between * fs)
    silence   = np.zeros(pause_n, dtype=np.float64)
    segments  = []

    for turn in range(n_turns):
        gender = "male" if turn % 2 == 0 else "female"
        seg    = make_realistic_speech(fs=fs,
                                       duration=duration_per_turn,
                                       gender=gender)
        segments.append(seg)
        segments.append(silence.copy())

    dialogue = np.concatenate(segments)
    peak     = np.max(np.abs(dialogue))
    if peak > 0:
        dialogue = dialogue / peak * 0.85
    return dialogue


# ════════════════════════════════════════════════════════════
# PHẦN 4 – TÍN HIỆU NHẠC TỔNG HỢP
# ════════════════════════════════════════════════════════════

def make_sine_wave(freq: float = 440.0, duration: float = DURATION,
                   fs: int = FS, amp: float = 0.8) -> np.ndarray:
    """Sóng sine đơn tần: x[n] = A·sin(2πf·n/Fs)"""
    t = np.linspace(0, duration, int(duration*fs), endpoint=False)
    return (amp * np.sin(2*np.pi*freq*t)).astype(np.float64)


def make_harmonic_tone(f0: float = 261.63, n_harmonics: int = 8,
                       duration: float = DURATION, fs: int = FS) -> np.ndarray:
    """
    Âm điều hòa: f0 + bội âm, biên độ 1/k (mô phỏng đàn piano).
    Dùng để thực hành phân tích harmonic series.
    """
    t   = np.linspace(0, duration, int(duration*fs), endpoint=False)
    sig = np.zeros_like(t)
    for k in range(1, n_harmonics+1):
        if f0*k < fs/2:
            sig += (1.0/k) * np.sin(2*np.pi*f0*k*t)
    sig = sig / (np.max(np.abs(sig)) + 1e-9) * 0.8
    return sig.astype(np.float64)


def make_chord(fs: int = FS, duration: float = DURATION) -> np.ndarray:
    """
    Hợp âm Do trưởng (C4-E4-G4) + transient percussive mỗi giây.
    Mỗi nốt có harmonic series đầy đủ.
    """
    t   = np.linspace(0, duration, int(duration*fs), endpoint=False)
    sig = np.zeros_like(t)
    for f0 in [261.63, 329.63, 392.00]:
        for k in range(1, 6):
            if f0*k < fs/2:
                sig += (0.4/k) * np.sin(2*np.pi*f0*k*t)
    # Percussive transient mỗi giây
    rng = np.random.default_rng(42)
    for beat in np.arange(0, duration, 1.0):
        i0 = int(beat*fs)
        il = min(int(0.08*fs), len(sig)-i0)
        if il > 0:
            env = np.exp(-50*np.linspace(0, 0.08, il))
            sig[i0:i0+il] += 0.3 * env * rng.standard_normal(il)
    sig = sig / (np.max(np.abs(sig)) + 1e-9) * 0.85
    return sig.astype(np.float64)


# ════════════════════════════════════════════════════════════
# PHẦN 5 – TẠO TẤT CẢ FILE MẪU
# ════════════════════════════════════════════════════════════

def generate_all_synthetic() -> None:
    """
    Tạo đầy đủ các file âm thanh mẫu:
      Nhạc tổng hợp  : music_sine_440hz.wav, music_harmonic_C4.wav, music_chord_Cmaj.wav
      Giọng nói tổng hợp: speech_male.wav, speech_female.wav, speech_dialogue_synthetic.wav
    """
    print("\n" + "═"*60)
    print("  TẠO TÍN HIỆU TỔNG HỢP")
    print("═"*60)

    # ── Nhạc ──
    save_wav_float(make_sine_wave(440.0),         FS, "music_sine_440hz.wav")
    save_wav_float(make_sine_wave(1000.0),         FS, "music_sine_1000hz.wav")
    save_wav_float(make_harmonic_tone(261.63),     FS, "music_harmonic_C4.wav")
    save_wav_float(make_chord(),                   FS, "music_chord_Cmaj.wav")

    # ── Giọng nói tổng hợp ──
    print()
    male_sig   = make_realistic_speech(fs=FS, duration=5.0, gender="male")
    save_wav_float(male_sig,   FS, "speech_male_synth.wav")

    female_sig = make_realistic_speech(fs=FS, duration=5.0, gender="female")
    save_wav_float(female_sig, FS, "speech_female_synth.wav")

    # ── Hội thoại tổng hợp ──
    dlg_sig = make_two_speaker_dialogue_synthetic(
        fs=FS, duration_per_turn=1.2, n_turns=6, pause_between=0.35)
    save_wav_float(dlg_sig, FS, "speech_dialogue_synthetic.wav")

    print(f"\n  ✓ Tín hiệu tổng hợp hoàn thành  →  {AUDIO_DIR}")


def generate_tts_files() -> None:
    """
    Tạo file âm thanh TTS từ các câu mẫu (giọng nam + nữ).
    Cần pyttsx3 hoạt động.
    """
    print("\n" + "═"*60)
    print("  TẠO FILE ÂM THANH BẰNG TTS")
    print("═"*60)

    # Kiểm tra pyttsx3
    try:
        import pyttsx3
        engine = pyttsx3.init()
        voices = engine.getProperty("voices")
        engine.stop()
        n_voices = len(voices) if voices else 0
        print(f"  pyttsx3 OK – {n_voices} giọng có sẵn")
    except Exception as e:
        print(f"  ✗ pyttsx3 lỗi: {e}")
        return

    # Giọng đơn (nam) – độc thoại bài giảng
    create_single_speech(filename="speech_monologue_male.wav")

    # Hội thoại 2 người (David=nam, Zira=nữ)
    create_dialogue(output_filename="speech_dialogue_2person.wav",
                    pause_sec=0.4)

    print(f"\n  ✓ TTS hoàn thành  →  {AUDIO_DIR}")


# ════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════

def main():
    print("\n╔" + "═"*58 + "╗")
    print("║  CSE457 Lab 1 – Sinh viên: Dương Tiến Đạt           ║")
    print("║  MSSV: 2251172268 – Tạo file âm thanh               ║")
    print("╚" + "═"*58 + "╝")
    print("  [1]  Nhập văn bản → Text-to-Speech (WAV)")
    print("  [2]  Tạo hội thoại 2 người (TTS, David + Zira)")
    print("  [3]  Tạo tín hiệu tổng hợp (không cần internet)")
    print("  [4]  Tất cả: TTS + hội thoại + tổng hợp")
    print("  [0]  Thoát")
    print("─"*60)
    choice = input("  Lựa chọn: ").strip()

    if choice == "1":
        interactive_tts()
    elif choice == "2":
        create_dialogue()
    elif choice == "3":
        generate_all_synthetic()
    elif choice == "4":
        generate_all_synthetic()
        generate_tts_files()
    elif choice == "0":
        return
    else:
        generate_all_synthetic()

    # Liệt kê file
    print("\n  ─── File trong audio/ ─────────────────────────────")
    for f in sorted(os.listdir(AUDIO_DIR)):
        fp = os.path.join(AUDIO_DIR, f)
        print(f"  {f:<50}  {os.path.getsize(fp)//1024:>4}KB")


if __name__ == "__main__":
    main()
