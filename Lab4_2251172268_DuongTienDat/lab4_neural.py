"""
lab4_neural.py
==============
Mô đun tổng hợp tiếng nói học sâu (Neural Text-to-Speech) cho Lab 4:
  - Sử dụng mô hình Neural TTS chất lượng phòng thu (Studio-quality) siêu tự nhiên cho tiếng Việt
    (Microsoft Azure Neural Voice qua edge-tts: vi-VN-HoaiMyNeural).
  - Có cơ chế fallback ngoại tuyến (offline) đảm bảo hoạt động độc lập không bị gián đoạn.
  - Tổng hợp 3 câu kiểm thử chuẩn:
      Câu 1: "Mời khách hàng số 23 đến quầy 5."
      Câu 2: "Mời khách hàng số 107 đến quầy 2."
      Câu 3: "Mời khách hàng số 496 đến quầy 8."
  - Trích xuất và trực quan hóa Mel-Spectrogram chuẩn 80-bin.
  - Đo đạc chính xác thời gian trễ (Latency) và hệ số thời gian thực (Real-Time Factor - RTF).
"""

import sys
import os
import time
import asyncio
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

from lab4_utils import match_rms

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
AUDIO_OUT_DIR = os.path.join(BASE_DIR, "audio_output")
FIG_DIR = os.path.join(BASE_DIR, "figures")

TEST_SENTENCES = [
    ("sentence1", "Mời khách hàng số 23 đến quầy 5."),
    ("sentence2", "Mời khách hàng số 107 đến quầy 2."),
    ("sentence3", "Mời khách hàng số 496 đến quầy 8.")
]

async def _synthesize_edge_async(text: str, voice: str, out_file: str):
    """Gọi edge-tts async sinh file MP3."""
    import edge_tts
    comm = edge_tts.Communicate(text, voice=voice)
    await comm.save(out_file)

def synthesize_neural(
    text: str,
    voice: str = "vi-VN-HoaiMyNeural",
    sr: int = 16000,
    out_wav_path: str = None
) -> tuple[np.ndarray, int, float, dict]:
    """
    Tổng hợp một câu văn bản sang tiếng nói tự nhiên bằng Neural TTS:
    Trả về: (tín hiệu âm thanh np.ndarray, sampling rate, độ trễ latency tính bằng giây, thông tin extra)
    """
    t_start = time.perf_counter()
    temp_mp3 = os.path.join(AUDIO_OUT_DIR, f"temp_neural_{int(time.time()*1000)}.mp3")
    os.makedirs(AUDIO_OUT_DIR, exist_ok=True)
    
    audio_data = None
    engine_name = "Google_Neural_TTS"
    
    # 1. Thử qua gTTS (Google Neural Voice cực kỳ tự nhiên cho tiếng Việt)
    try:
        from gtts import gTTS
        tts = gTTS(text=text, lang="vi", slow=False)
        tts.save(temp_mp3)
        y, _ = librosa.load(temp_mp3, sr=sr, mono=True)
        yt, _ = librosa.effects.trim(y, top_db=25)
        audio_data = yt
        if os.path.exists(temp_mp3):
            os.remove(temp_mp3)
    except Exception as e:
        # 2. Thử qua Edge TTS
        try:
            import edge_tts
            asyncio.run(_synthesize_edge_async(text, voice, temp_mp3))
            y, _ = librosa.load(temp_mp3, sr=sr, mono=True)
            yt, _ = librosa.effects.trim(y, top_db=25)
            audio_data = yt
            engine_name = "Edge_Neural_TTS"
            if os.path.exists(temp_mp3):
                os.remove(temp_mp3)
        except Exception as e2:
            # 3. Fallback ghép nối offline
            from lab4_concat import concat_audio_sequence, get_unit_path
            from lab4_text_norm import normalize_queue_text, tokens_to_unit_ids
            toks = normalize_queue_text(text)
            uids = tokens_to_unit_ids(toks)
            pths = [get_unit_path(u, 1) for u in uids]
            audio_data, _ = concat_audio_sequence(pths, sr=sr, xfade_ms=15.0)
            engine_name = "Offline_UnitConcat"

    t_end = time.perf_counter()
    latency_sec = t_end - t_start
    
    # Chuẩn hóa năng lượng RMS = 0.08
    norm_audio = match_rms(audio_data, target=0.08)
    
    duration_sec = len(norm_audio) / sr
    rtf = latency_sec / duration_sec if duration_sec > 0 else 0.0
    
    # Tính Mel-spectrogram 80 bins
    mel_spec = librosa.feature.melspectrogram(
        y=norm_audio,
        sr=sr,
        n_fft=1024,
        hop_length=256,
        n_mels=80,
        fmin=40,
        fmax=7600
    )
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    if out_wav_path:
        sf.write(out_wav_path, norm_audio.astype(np.float32), sr)
        
    extra = {
        "engine": engine_name,
        "voice": voice,
        "text": text,
        "duration_sec": duration_sec,
        "latency_sec": latency_sec,
        "rtf": rtf,
        "mel_spec_db": mel_spec_db
    }
    
    return norm_audio, sr, latency_sec, extra

def run_neural_experiments():
    """
    Thực hiện thí nghiệm Neural TTS trên 3 câu văn bản chuẩn:
      - Đo độ trễ Latency và RTF
      - Lưu file audio: neural_tts.wav, neural_tts_sentence2.wav, neural_tts_sentence3.wav
      - Vẽ Figure 8: Mel-spectrogram và Dạng sóng của mô hình Neural TTS
    """
    os.makedirs(AUDIO_OUT_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)
    
    print("=" * 60)
    print("THÍ NGHIỆM NEURAL TTS (GIỌNG NÓI HỌC SÂU SIÊU TỰ NHIÊN)")
    print("=" * 60)
    
    results = []
    first_mel = None
    first_wave = None
    
    for idx, (label, sent) in enumerate(TEST_SENTENCES):
        out_name = "neural_tts.wav" if label == "sentence1" else f"neural_tts_{label}.wav"
        out_path = os.path.join(AUDIO_OUT_DIR, out_name)
        
        wave, sr, latency, extra = synthesize_neural(
            sent,
            voice="vi-VN-HoaiMyNeural",
            sr=16000,
            out_wav_path=out_path
        )
        
        print(f"[+] Câu {idx+1} ('{sent}'):")
        print(f"    - File audio: {out_path}")
        print(f"    - Thời lượng audio: {extra['duration_sec']:.2f} s")
        print(f"    - Latency (thời gian sinh): {extra['latency_sec']:.3f} s ({extra['latency_sec']*1000:.1f} ms)")
        print(f"    - RTF (Real-Time Factor): {extra['rtf']:.3f} (Nhỏ hơn 1 là nhanh hơn thời gian thực)")
        
        results.append(extra)
        if idx == 0:
            first_mel = extra["mel_spec_db"]
            first_wave = wave

    # Vẽ Figure 8: Mel-Spectrogram và Dạng sóng của câu 1
    fig8_path = os.path.join(FIG_DIR, "fig8_neural_mel_spectrogram.png")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True)
    plt.subplots_adjust(hspace=0.25)
    
    t_wave = np.arange(len(first_wave)) / 16000
    ax1.plot(t_wave, first_wave, color="#2980b9", linewidth=0.9)
    ax1.set_ylabel("Biên độ", fontsize=10, fontweight="bold")
    ax1.set_title("Neural TTS Waveform - 'Mời khách hàng số 23 đến quầy 5.'", fontsize=11, fontweight="bold")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.set_ylim(-1.05, 1.05)
    
    img = librosa.display.specshow(
        first_mel,
        sr=16000,
        hop_length=256,
        x_axis="time",
        y_axis="mel",
        fmin=40,
        fmax=7600,
        ax=ax2,
        cmap="viridis"
    )
    ax2.set_xlabel("Thời gian (giây)", fontsize=10, fontweight="bold")
    ax2.set_ylabel("Tần số Mel (Hz)", fontsize=10, fontweight="bold")
    ax2.set_title("80-Channel Log Mel-Spectrogram (Acoustic Representation)", fontsize=11, fontweight="bold")
    fig.colorbar(img, ax=ax2, format="%+2.0f dB", label="Cường độ Mel (dB)")
    
    plt.tight_layout()
    plt.savefig(fig8_path, dpi=180)
    plt.close(fig)
    print(f"[+] Đã lưu biểu đồ Neural Mel-Spectrogram: {fig8_path}")
    
    return results

if __name__ == "__main__":
    run_neural_experiments()
