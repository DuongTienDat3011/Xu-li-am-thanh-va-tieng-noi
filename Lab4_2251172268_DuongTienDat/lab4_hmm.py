"""
lab4_hmm.py
===========
Mô đun tổng hợp tiếng nói tham số dựa trên HMM (HMM-based Parametric Synthesis - HTS demo) cho Lab 4:
  - Mô phỏng quá trình HMM chạy ở chế độ Generation (sinh tham số thay vì nhận dạng).
  - Ánh xạ chuỗi token -> chuỗi phone/state HMM với phân bố Gaussian của tham số âm học.
  - Sinh quỹ đạo tham số âm học trơn tru (MLPG - Maximum Likelihood Parameter Generation)
    nhờ ràng buộc đặc trưng động (Static + Delta features).
  - Các tham số âm học: Duration theo state, F0 contour (voiced/unvoiced), Formants F1/F2/F3 và Năng lượng.
  - Resynthesis bằng bộ nguồn-lọc (Source-Filter Vocoder): Xung pitch + Nhiễu unvoiced đi qua bộ lọc Formant.
  - Hỗ trợ biến điệu tham số âm học độc lập: Duration scale (0.8x, 1.2x), F0 scale (0.9x, 1.1x).
  - Xuất ra audio_output/hmm_tts.wav và đồ thị quỹ đạo tham số figures/fig7_hmm_trajectories.png.
"""

import sys
import os
import numpy as np
import soundfile as sf
from scipy.signal import lfilter, butter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from lab4_text_norm import normalize_queue_text

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
AUDIO_OUT_DIR = os.path.join(BASE_DIR, "audio_output")
FIG_DIR = os.path.join(BASE_DIR, "figures")

# Bảng tra cứu tham số formant đặc trưng (F1, F2, F3 tính bằng Hz) cho từng từ/nguyên âm
WORD_ACOUSTIC_PROFILES = {
    "mời":   {"f0_base": 165, "f1": 500, "f2": 1100, "f3": 2500, "dur_ms": 350, "voiced": True},
    "khách": {"f0_base": 185, "f1": 700, "f2": 1500, "f3": 2600, "dur_ms": 280, "voiced": True},
    "hàng":  {"f0_base": 155, "f1": 650, "f2": 1300, "f3": 2450, "dur_ms": 340, "voiced": True},
    "số":    {"f0_base": 190, "f1": 450, "f2": 900,  "f3": 2400, "dur_ms": 320, "voiced": True},
    "không": {"f0_base": 160, "f1": 400, "f2": 800,  "f3": 2350, "dur_ms": 300, "voiced": True},
    "một":   {"f0_base": 170, "f1": 480, "f2": 1050, "f3": 2500, "dur_ms": 300, "voiced": True},
    "hai":   {"f0_base": 165, "f1": 750, "f2": 1400, "f3": 2600, "dur_ms": 320, "voiced": True},
    "ba":    {"f0_base": 160, "f1": 720, "f2": 1250, "f3": 2550, "dur_ms": 330, "voiced": True},
    "bốn":   {"f0_base": 185, "f1": 450, "f2": 950,  "f3": 2450, "dur_ms": 320, "voiced": True},
    "năm":   {"f0_base": 165, "f1": 680, "f2": 1350, "f3": 2500, "dur_ms": 340, "voiced": True},
    "sáu":   {"f0_base": 195, "f1": 550, "f2": 1100, "f3": 2550, "dur_ms": 310, "voiced": True},
    "bảy":   {"f0_base": 175, "f1": 650, "f2": 1450, "f3": 2600, "dur_ms": 350, "voiced": True},
    "tám":   {"f0_base": 185, "f1": 700, "f2": 1300, "f3": 2500, "dur_ms": 320, "voiced": True},
    "chín":  {"f0_base": 190, "f1": 350, "f2": 2100, "f3": 2800, "dur_ms": 300, "voiced": True},
    "đến":   {"f0_base": 175, "f1": 500, "f2": 1600, "f3": 2650, "dur_ms": 320, "voiced": True},
    "quầy":  {"f0_base": 155, "f1": 450, "f2": 1200, "f3": 2400, "dur_ms": 340, "voiced": True},
    "_pause_": {"f0_base": 0, "f1": 300, "f2": 800, "f3": 2200, "dur_ms": 120, "voiced": False}
}

def generate_trajectories(
    tokens: list[str],
    sr: int = 16000,
    hop_length: int = 160,  # 10 ms per frame -> 100 fps
    duration_scale: float = 1.0,
    f0_scale: float = 1.0
):
    """
    Tạo chuỗi State và các Quỹ đạo tham số âm học (F0, F1, F2, F3, Energy, State ID).
    Mỗi từ bao gồm 3 state HMM trái-sang-phải (Onset -> Nucleus -> Coda).
    """
    fps = sr / hop_length  # 100 frames/s
    
    state_seq = []
    word_labels = []
    f0_traj = []
    f1_traj = []
    f2_traj = []
    f3_traj = []
    energy_traj = []
    
    global_frame = 0
    
    for word_idx, w in enumerate(tokens):
        prof = WORD_ACOUSTIC_PROFILES.get(w, WORD_ACOUSTIC_PROFILES["_pause_"])
        
        # 3 states cho mỗi từ: tỷ lệ thời lượng [0.25, 0.50, 0.25]
        dur_total_sec = (prof["dur_ms"] / 1000.0) * duration_scale
        total_frames = max(3, int(round(dur_total_sec * fps)))
        f_s1 = int(round(total_frames * 0.25))
        f_s2 = int(round(total_frames * 0.50))
        f_s3 = total_frames - f_s1 - f_s2
        
        state_lens = [f_s1, f_s2, f_s3]
        f0_center = prof["f0_base"] * f0_scale
        
        for s_idx, n_fr in enumerate(state_lens):
            state_id = word_idx * 3 + s_idx
            for fr in range(n_fr):
                rel_pos = fr / max(1, n_fr - 1)  # 0.0 -> 1.0 trong state
                state_seq.append(s_idx + 1)
                word_labels.append(w)
                
                # F0 contour: mô phỏng ngữ điệu tiếng Việt tự nhiên (hơi võng ở giữa hoặc lên/xuống theo thanh điệu)
                if prof["voiced"]:
                    # Biến thiên vi mô (microprosody)
                    mod = np.sin(np.pi * (fr + s_idx * n_fr) / total_frames) * 15.0
                    cur_f0 = f0_center + mod
                else:
                    cur_f0 = 0.0
                f0_traj.append(cur_f0)
                
                # Formants: biến đổi liên tục giữa các state (Dynamic Formant Transitions)
                f1_val = prof["f1"] * (0.85 if s_idx == 0 else (1.05 if s_idx == 1 else 0.90))
                f2_val = prof["f2"] * (0.90 if s_idx == 0 else (1.00 if s_idx == 1 else 0.95))
                f3_val = prof["f3"]
                f1_traj.append(f1_val)
                f2_traj.append(f2_val)
                f3_traj.append(f3_val)
                
                # Energy envelope: hình chuông mượt
                env = np.sin(np.pi * (global_frame % total_frames) / total_frames) ** 0.8
                cur_energy = 0.08 * (env if prof["voiced"] else 0.01)
                energy_traj.append(cur_energy)
                
                global_frame += 1

    # Thêm chèn mượt bằng bộ lọc moving average (Mô phỏng MLPG với delta features)
    from scipy.ndimage import uniform_filter1d
    f0_smooth = uniform_filter1d(np.array(f0_traj, dtype=float), size=5)
    f1_smooth = uniform_filter1d(np.array(f1_traj, dtype=float), size=7)
    f2_smooth = uniform_filter1d(np.array(f2_traj, dtype=float), size=7)
    f3_smooth = uniform_filter1d(np.array(f3_traj, dtype=float), size=7)
    energy_smooth = uniform_filter1d(np.array(energy_traj, dtype=float), size=5)
    
    times = np.arange(len(state_seq)) * (hop_length / sr)
    
    return {
        "times": times,
        "state_seq": np.array(state_seq),
        "word_labels": word_labels,
        "f0": f0_smooth,
        "f1": f1_smooth,
        "f2": f2_smooth,
        "f3": f3_smooth,
        "energy": energy_smooth,
        "fps": fps,
        "hop_length": hop_length,
        "sr": sr
    }

def synthesize_source_filter_vocoder(tracks: dict) -> np.ndarray:
    """
    Tái tạo waveform từ các quỹ đạo tham số HMM bằng Vocoder Nguồn - Lọc (Source-Filter Vocoder):
      - Nguồn kích thích e[n]: Voiced = xung thanh đới (glottal pulse train); Unvoiced = nhiễu Gaussian.
      - Bộ lọc Vocal Tract H(z): Tổ hợp 3 bộ lọc Formant cộng hưởng (Resonator Biquads).
    """
    sr = tracks["sr"]
    hop_length = tracks["hop_length"]
    n_frames = len(tracks["times"])
    total_samples = n_frames * hop_length
    
    # 1. Sinh nguồn kích thích (Excitation Signal)
    excitation = np.zeros(total_samples, dtype=np.float32)
    phase = 0.0
    
    for fr in range(n_frames):
        start = fr * hop_length
        end = min(total_samples, (fr + 1) * hop_length)
        f0_val = tracks["f0"][fr]
        e_val = tracks["energy"][fr]
        
        if f0_val > 50.0:  # Voiced frame
            period = sr / f0_val
            for n in range(start, end):
                phase += 1.0
                if phase >= period:
                    excitation[n] = 1.0  # Glottal pulse
                    phase -= period
                else:
                    excitation[n] = 0.0
            # Thêm chút hơi thở tự nhiên (aspiration noise)
            excitation[start:end] += np.random.normal(0, 0.02, end - start)
        else:  # Unvoiced frame
            excitation[start:end] = np.random.normal(0, 0.15, end - start)
            
        excitation[start:end] *= (e_val / (np.std(excitation[start:end]) + 1e-6))

    # 2. Bộ lọc cộng hưởng Formant (Formant Resonators)
    # Lọc qua formant F1, F2, F3 trung bình
    mean_f1 = float(np.mean(tracks["f1"]))
    mean_f2 = float(np.mean(tracks["f2"]))
    mean_f3 = float(np.mean(tracks["f3"]))
    
    def make_resonator(f_res, bw, fs):
        """Tạo bộ lọc cộng hưởng 2 cực (2-pole resonator biquad)."""
        r = np.exp(-np.pi * bw / fs)
        theta = 2 * np.pi * f_res / fs
        a = [1.0, -2 * r * np.cos(theta), r * r]
        b = [1.0 - r]
        return b, a
        
    b1, a1 = make_resonator(mean_f1, 90, sr)
    b2, a2 = make_resonator(mean_f2, 110, sr)
    b3, a3 = make_resonator(mean_f3, 150, sr)
    
    s1 = lfilter(b1, a1, excitation)
    s2 = lfilter(b2, a2, excitation)
    s3 = lfilter(b3, a3, excitation)
    
    synthesized = 0.5 * s1 + 0.3 * s2 + 0.2 * s3
    
    # Chuẩn hóa năng lượng RMS = 0.08
    rms_cur = np.sqrt(np.mean(synthesized ** 2) + 1e-12)
    synthesized = synthesized * (0.08 / rms_cur)
    synthesized = np.clip(synthesized, -0.99, 0.99)
    
    return synthesized.astype(np.float32)

def run_hmm_experiments(text: str = "Mời khách hàng số 23 đến quầy 5."):
    """
    Thực hiện thí nghiệm tổng hợp tham số HMM cho Lab 4:
      1. Sinh chuỗi tham số và state
      2. Tái tạo audio chuẩn -> audio_output/hmm_tts.wav
      3. Biến điệu tham số: duration scale (0.8x, 1.2x), F0 scale (0.9x, 1.1x)
      4. Vẽ biểu đồ quỹ đạo tham số figures/fig7_hmm_trajectories.png
    """
    os.makedirs(AUDIO_OUT_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)
    
    tokens = normalize_queue_text(text)
    print("=" * 60)
    print("THÍ NGHIỆM HMM-BASED PARAMETRIC SYNTHESIS")
    print("=" * 60)
    print(f"[i] Câu tổng hợp: '{text}' -> Tokens: {tokens}")
    
    # 1. Baseline HMM synthesis
    tracks = generate_trajectories(tokens, duration_scale=1.0, f0_scale=1.0)
    y_hmm = synthesize_source_filter_vocoder(tracks)
    
    hmm_out_path = os.path.join(AUDIO_OUT_DIR, "hmm_tts.wav")
    sf.write(hmm_out_path, y_hmm, tracks["sr"])
    print(f"[+] Đã tạo audio HMM TTS: {hmm_out_path} (độ dài: {len(y_hmm)/tracks['sr']:.2f}s, {len(tracks['times'])} frames)")
    
    # 2. Biến điệu Duration scale (0.8x và 1.2x)
    tracks_dur_slow = generate_trajectories(tokens, duration_scale=1.2, f0_scale=1.0)
    y_dur_slow = synthesize_source_filter_vocoder(tracks_dur_slow)
    sf.write(os.path.join(AUDIO_OUT_DIR, "hmm_duration_1.2x.wav"), y_dur_slow, tracks["sr"])
    
    tracks_dur_fast = generate_trajectories(tokens, duration_scale=0.8, f0_scale=1.0)
    y_dur_fast = synthesize_source_filter_vocoder(tracks_dur_fast)
    sf.write(os.path.join(AUDIO_OUT_DIR, "hmm_duration_0.8x.wav"), y_dur_fast, tracks["sr"])
    
    # 3. Biến điệu F0 scale (0.9x và 1.1x)
    tracks_f0_low = generate_trajectories(tokens, duration_scale=1.0, f0_scale=0.9)
    y_f0_low = synthesize_source_filter_vocoder(tracks_f0_low)
    sf.write(os.path.join(AUDIO_OUT_DIR, "hmm_pitch_0.9x.wav"), y_f0_low, tracks["sr"])
    
    tracks_f0_high = generate_trajectories(tokens, duration_scale=1.0, f0_scale=1.1)
    y_f0_high = synthesize_source_filter_vocoder(tracks_f0_high)
    sf.write(os.path.join(AUDIO_OUT_DIR, "hmm_pitch_1.1x.wav"), y_f0_high, tracks["sr"])
    
    print("[+] Đã tạo các biến thể tham số HMM (duration 0.8x/1.2x và pitch 0.9x/1.1x)")
    
    # 4. Vẽ biểu đồ Figure 7: Quỹ đạo tham số HMM (State, F0, Formants, Energy)
    fig7_path = os.path.join(FIG_DIR, "fig7_hmm_trajectories.png")
    fig, (ax_state, ax_f0, ax_fmt, ax_eng) = plt.subplots(4, 1, figsize=(12, 9), sharex=True)
    plt.subplots_adjust(hspace=0.28)
    
    t = tracks["times"]
    
    # Panel 1: State sequence & Word segmentation
    ax_state.step(t, tracks["state_seq"], where="post", color="#2c3e50", linewidth=1.5)
    ax_state.set_ylabel("HMM State", fontsize=9, fontweight="bold")
    ax_state.set_yticks([1, 2, 3])
    ax_state.set_yticklabels(["S1: Onset", "S2: Nucleus", "S3: Coda"], fontsize=8)
    ax_state.set_title("Chuỗi trạng thái HMM và phân đoạn từ (State Sequence & Segmentation)", fontsize=10, fontweight="bold")
    ax_state.grid(True, linestyle=":", alpha=0.6)
    
    # Đánh dấu ranh giới các từ
    prev_w = None
    for idx, (tm, w) in enumerate(zip(t, tracks["word_labels"])):
        if w != prev_w:
            ax_state.axvline(x=tm, color="#e74c3c", linestyle="--", linewidth=1.0, alpha=0.7)
            ax_state.text(tm + 0.02, 2.5, w, color="#c0392b", fontsize=8.5, fontweight="bold")
            prev_w = w
            
    # Panel 2: F0 contour
    ax_f0.plot(t, tracks["f0"], color="#27ae60", linewidth=2.0, label="F0 Trajectory (Hz)")
    ax_f0.set_ylabel("F0 (Hz)", fontsize=9, fontweight="bold")
    ax_f0.set_title("Quỹ đạo tần số cơ bản sinh từ HMM (F0 Generation Trajectory)", fontsize=10, fontweight="bold")
    ax_f0.set_ylim(50, 260)
    ax_f0.grid(True, linestyle=":", alpha=0.6)
    ax_f0.legend(loc="upper right")
    
    # Panel 3: Formants F1, F2, F3
    ax_fmt.plot(t, tracks["f1"], color="#e67e22", linewidth=1.8, label="F1 (First Formant)")
    ax_fmt.plot(t, tracks["f2"], color="#2980b9", linewidth=1.8, label="F2 (Second Formant)")
    ax_fmt.plot(t, tracks["f3"], color="#8e44ad", linewidth=1.5, linestyle="--", label="F3 (Third Formant)")
    ax_fmt.set_ylabel("Tần số (Hz)", fontsize=9, fontweight="bold")
    ax_fmt.set_title("Quỹ đạo Formant chuyển tiếp trơn tru (MLPG Formant Trajectories)", fontsize=10, fontweight="bold")
    ax_fmt.set_ylim(200, 3200)
    ax_fmt.grid(True, linestyle=":", alpha=0.6)
    ax_fmt.legend(loc="upper right", ncol=3)
    
    # Panel 4: Energy contour
    ax_eng.plot(t, tracks["energy"], color="#d35400", linewidth=1.8, label="Energy Envelope (RMS)")
    ax_eng.set_ylabel("Năng lượng", fontsize=9, fontweight="bold")
    ax_eng.set_xlabel("Thời gian (giây)", fontsize=10, fontweight="bold")
    ax_eng.set_title("Đường bao năng lượng theo frame (Energy Contour)", fontsize=10, fontweight="bold")
    ax_eng.grid(True, linestyle=":", alpha=0.6)
    ax_eng.legend(loc="upper right")
    
    plt.tight_layout()
    plt.savefig(fig7_path, dpi=180)
    plt.close(fig)
    print(f"[+] Đã lưu biểu đồ HMM Trajectories: {fig7_path}")
    
    return {
        "audio_path": hmm_out_path,
        "tracks": tracks,
        "figure_path": fig7_path
    }

if __name__ == "__main__":
    run_hmm_experiments()
