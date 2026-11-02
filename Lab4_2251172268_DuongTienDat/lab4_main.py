"""
lab4_main.py
============
Master Runner thực thi toàn bộ pipeline Lab 4 từ đầu đến cuối:
  - Phần A: Kiểm tra Unit Inventory, vẽ biểu đồ Histogram Duration & RMS (fig1_inventory_stats.png).
  - Phần B & C: Ghép nối Baseline (raw_concat.wav) và Unit Selection DP Viterbi (best_concat.wav),
                khử click bằng Cross-fade (0, 5, 15, 30 ms), xuất cost_analysis.csv.
  - Phần D: Phân tích Prosody & Source-Filter (F0, Time-Stretch 0.8x/1.2x, Pitch-Shift +-2 semitones).
  - Phần E: HMM-based Parametric Synthesis (Trajectoriy generation, Vocoder resynthesis, duration/pitch scaling).
  - Phần F: Neural TTS siêu tự nhiên cho 3 câu kiểm thử (Latency & RTF timing, 80-bin Mel-spectrogram).
  - Phần G: Đánh giá chất lượng nghe chủ quan (Listening Test -> ratings.csv) và so sánh 3 phương pháp.
"""

import sys
import os
import time
import numpy as np
import pandas as pd
import soundfile as sf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from lab4_text_norm import normalize_queue_text, tokens_to_unit_ids
from lab4_utils import load_audio, match_rms
from generate_inventory import create_inventory
from lab4_concat import synthesize_concatenative, get_unit_path
from lab4_prosody import run_prosody_experiments
from lab4_hmm import run_hmm_experiments
from lab4_neural import run_neural_experiments
from lab4_eval import run_evaluation_experiments

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UNITS_DIR = os.path.join(BASE_DIR, "units")
AUDIO_OUT_DIR = os.path.join(BASE_DIR, "audio_output")
FIG_DIR = os.path.join(BASE_DIR, "figures")
METADATA_PATH = os.path.join(BASE_DIR, "metadata.csv")
COST_ANALYSIS_PATH = os.path.join(BASE_DIR, "cost_analysis.csv")

def plot_inventory_statistics(meta_df: pd.DataFrame, save_path: str):
    """Vẽ Histogram phân bố Thời lượng (Duration) và Năng lượng (RMS) của Inventory."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    
    # Histogram Duration
    durations = meta_df["duration_s"].values
    ax1.hist(durations, bins=8, color="#3498db", edgecolor="#2980b9", alpha=0.85)
    mean_dur = float(np.mean(durations))
    ax1.axvline(x=mean_dur, color="#e74c3c", linestyle="--", linewidth=1.5, label=f"Mean = {mean_dur:.3f} s")
    ax1.set_title("Phân bố Thời lượng Unit (Duration Distribution)", fontsize=10.5, fontweight="bold")
    ax1.set_xlabel("Thời lượng (giây)", fontsize=9.5)
    ax1.set_ylabel("Số lượng Unit", fontsize=9.5)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="upper right")
    
    # Histogram RMS
    rms_vals = meta_df["rms"].values
    ax2.hist(rms_vals, bins=8, color="#2ecc71", edgecolor="#27ae60", alpha=0.85)
    ax2.axvline(x=0.08, color="#e74c3c", linestyle="--", linewidth=1.5, label="Mục tiêu Target RMS = 0.080")
    ax2.set_title("Phân bố Năng lượng RMS (Energy Distribution)", fontsize=10.5, fontweight="bold")
    ax2.set_xlabel("RMS", fontsize=9.5)
    ax2.set_ylabel("Số lượng Unit", fontsize=9.5)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="upper right")
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.savefig(save_path, dpi=180)
    plt.close(fig)
    print(f"[+] Đã lưu biểu đồ Thống kê Inventory: {save_path}")

def main():
    print("*" * 75)
    print("CSE457 - XỬ LÝ ÂM THANH VÀ TIẾNG NÓI | LAB 4: TỔNG HỢP TIẾNG NÓI (TTS)")
    print("Sinh viên: Dương Tiến Đạt | MSSV: 2251172268")
    print("*" * 75)
    
    start_total_time = time.time()
    os.makedirs(AUDIO_OUT_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)
    
    # =========================================================================
    # PHẦN A: Khởi tạo dữ liệu và Thống kê Inventory
    # =========================================================================
    print("\n>>> [1/6] PHẦN A - KIỂM TRA INVENTORY & VẼ THỐNG KÊ...")
    if not os.path.exists(METADATA_PATH):
        meta_df = create_inventory(force_regenerate=False)
    else:
        meta_df = pd.read_csv(METADATA_PATH)
        print(f"[i] Đã nạp metadata ({len(meta_df)} units) từ {METADATA_PATH}")
        
    fig1_path = os.path.join(FIG_DIR, "fig1_inventory_stats.png")
    plot_inventory_statistics(meta_df, fig1_path)
    
    # =========================================================================
    # PHẦN B & C: Ghép nối Concatenative & Lựa chọn Unit (Viterbi)
    # =========================================================================
    print("\n>>> [2/6] PHẦN B & C - GHÉP NỐI BASELINE & LỰA CHỌN ĐƠN VỊ TỐI ƯU...")
    test_sentences = [
        ("sentence1", "Mời khách hàng số 23 đến quầy 5."),
        ("sentence2", "Mời khách hàng số 107 đến quầy 2."),
        ("sentence3", "Mời khách hàng số 496 đến quầy 8.")
    ]
    
    cost_records = []
    for prefix, sent_text in test_sentences:
        res = synthesize_concatenative(
            text=sent_text,
            output_prefix=prefix,
            xfade_ms=15.0,
            meta_df=meta_df
        )
        cost_records.append({
            "sentence_id": prefix,
            "text": sent_text,
            "baseline_fixed_cand_cost": round(res["baseline_cost"], 2),
            "viterbi_unit_selection_cost": round(res["optimal_cost"], 2),
            "cost_reduction_pct": round((1.0 - res["optimal_cost"] / res["baseline_cost"]) * 100, 2),
            "chosen_candidates": str(res["best_candidates"])
        })
        
    cost_df = pd.DataFrame(cost_records)
    cost_df.to_csv(COST_ANALYSIS_PATH, index=False, encoding="utf-8")
    print(f"[+] Đã lưu phân tích chi phí Join/Target cost ra: {COST_ANALYSIS_PATH}")
    print(cost_df.to_string(index=False))
    
    # =========================================================================
    # PHẦN D: Phân tích Prosody & Mô hình Nguồn - Lọc
    # =========================================================================
    print("\n>>> [3/6] PHẦN D - PHÂN TÍCH PROSODY & SOURCE-FILTER (F0, SPEED, PITCH)...")
    prosody_res = run_prosody_experiments()
    
    # =========================================================================
    # PHẦN E: Tổng hợp tham số HMM (HMM-based Parametric Synthesis)
    # =========================================================================
    print("\n>>> [4/6] PHẦN E - HMM-BASED PARAMETRIC SYNTHESIS (HTS BACKEND)...")
    hmm_res = run_hmm_experiments(text="Mời khách hàng số 23 đến quầy 5.")
    
    # =========================================================================
    # PHẦN F: Neural TTS học sâu siêu tự nhiên
    # =========================================================================
    print("\n>>> [5/6] PHẦN F - NEURAL TTS HỌC SÂU (PRETRAINED NEURAL VOICES)...")
    neural_res = run_neural_experiments()
    
    # =========================================================================
    # PHẦN G: Đánh giá chất lượng và So sánh 3 phương pháp
    # =========================================================================
    print("\n>>> [6/6] PHẦN G - ĐÁNH GIÁ CHẤT LƯỢNG (LISTENING TEST & REPORT)...")
    summary_eval = run_evaluation_experiments()
    
    total_elapsed = time.time() - start_total_time
    print("\n" + "=" * 75)
    print(f"HOÀN THÀNH TOÀN BỘ PIPELINE LAB 4 TRONG: {total_elapsed:.2f} GIÂY")
    print(f"Tất cả file âm thanh tại: {AUDIO_OUT_DIR}")
    print(f"Tất cả biểu đồ khoa học tại: {FIG_DIR}")
    print("=" * 75)

if __name__ == "__main__":
    main()
