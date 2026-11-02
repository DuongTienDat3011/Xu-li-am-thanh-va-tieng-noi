"""
lab4_concat.py
==============
Mô đun tổng hợp tiếng nói bằng phương pháp Ghép nối (Concatenative Synthesis)
và Lựa chọn đơn vị (Unit Selection) cho Lab 4:
  - Ghép baseline cố định (Candidate 1, xfade = 0 ms) -> raw_concat.wav.
  - Tính Target Cost và Join Cost (Log Spectral Mismatch + Energy Mismatch).
  - Thuật toán Quy hoạch động (Viterbi Search) tìm chuỗi candidate tối ưu.
  - Khử click biên nối bằng kỹ thuật Overlap-Add / Cross-fade (0, 5, 15, 30 ms).
  - Lưu kết quả tổng hợp tối ưu -> best_concat.wav và bảng phân tích chi phí.
"""

import sys
import os
import numpy as np
import soundfile as sf
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from lab4_text_norm import normalize_queue_text, tokens_to_unit_ids
from lab4_utils import load_audio, match_rms, plot_waveform_and_spectrogram, calc_boundary_discontinuity

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UNITS_DIR = os.path.join(BASE_DIR, "units")
AUDIO_OUT_DIR = os.path.join(BASE_DIR, "audio_output")
FIG_DIR = os.path.join(BASE_DIR, "figures")
METADATA_PATH = os.path.join(BASE_DIR, "metadata.csv")

def get_unit_path(unit_ascii: str, candidate: int) -> str:
    """Trả về đường dẫn tuyệt đối đến file WAV của candidate."""
    return os.path.join(UNITS_DIR, f"{unit_ascii}_0{candidate}.wav")

def compute_join_cost(a: np.ndarray, b: np.ndarray, sr: int = 16000, ms: int = 25, nfft: int = 512) -> float:
    """
    Tính chi phí ghép nối (Join Cost) giữa điểm kết thúc của unit a và điểm khởi đầu của unit b.
    Theo công thức mục 3.5 & 5.4 của Lab 4:
      D_spec = (1/K) * sum [ L_left[k] - L_right[k] ]^2
    Kết hợp thành phần chênh lệch RMS biên.
    """
    n = int(sr * ms / 1000)
    n = min(n, len(a), len(b))
    if n <= 0:
        return 0.0
    wa = np.hanning(n)
    wb = np.hanning(n)
    
    fft_a = np.fft.rfft(a[-n:] * wa, n=nfft)
    fft_b = np.fft.rfft(b[:n] * wb, n=nfft)
    
    la = 20 * np.log10(np.maximum(np.abs(fft_a), 1e-8))
    lb = 20 * np.log10(np.maximum(np.abs(fft_b), 1e-8))
    
    spec_dist = float(np.mean((la - lb) ** 2))
    
    rms_a = float(np.sqrt(np.mean(a[-n:]**2) + 1e-12))
    rms_b = float(np.sqrt(np.mean(b[:n]**2) + 1e-12))
    energy_dist = float(abs(rms_a - rms_b) * 100.0)
    
    return spec_dist + energy_dist

def compute_target_cost(unit_ascii: str, candidate: int, meta_df: pd.DataFrame, target_speaker: str = None) -> float:
    """
    Tính chi phí mục tiêu (Target Cost): đo mức độ phù hợp của candidate với ngữ cảnh mong muốn.
    """
    if meta_df is None or meta_df.empty:
        return 0.0
    row = meta_df[(meta_df["ascii"] == unit_ascii) & (meta_df["candidate"] == candidate)]
    if row.empty:
        return 0.0
    dur = row.iloc[0]["duration_s"]
    rms_val = row.iloc[0]["rms"]
    
    # Ưu tiên các candidate có duration ổn định và RMS gần 0.08
    cost_dur = float(((dur - 0.35) / 0.1) ** 2) * 2.0
    cost_rms = float(((rms_val - 0.08) / 0.02) ** 2) * 5.0
    return cost_dur + cost_rms

def crossfade(a: np.ndarray, b: np.ndarray, n_samples: int) -> np.ndarray:
    """
    Nối hai mảng âm thanh với kỹ thuật làm mờ chéo (Cross-fade) độ dài n_samples.
    Công thức: y[n] = (1 - alpha[n])*a[-n:] + alpha[n]*b[:n]
    """
    n_samples = min(n_samples, len(a), len(b))
    if n_samples == 0:
        return np.concatenate([a, b])
    alpha = np.linspace(0.0, 1.0, n_samples, endpoint=True)
    mid = (1.0 - alpha) * a[-n_samples:] + alpha * b[:n_samples]
    return np.concatenate([a[:-n_samples], mid, b[n_samples:]])

def concat_audio_sequence(unit_paths: list[str], sr: int = 16000, xfade_ms: float = 15.0):
    """
    Ghép chuỗi file âm thanh theo danh sách đường dẫn với thời gian cross-fade (ms).
    Trả về: (tín hiệu ghép y, danh sách mốc mẫu biên boundaries)
    """
    signals = [match_rms(load_audio(p, sr=sr)) for p in unit_paths]
    if not signals:
        return np.zeros(0, dtype=np.float32), []
        
    n_fade = int(sr * xfade_ms / 1000.0)
    y = signals[0]
    boundaries = [0]
    
    for s in signals[1:]:
        if n_fade > 0:
            boundaries.append(len(y) - n_fade // 2)
            y = crossfade(y, s, n_fade)
        else:
            boundaries.append(len(y))
            y = np.concatenate([y, s])
            
    return y, boundaries

def run_viterbi_unit_selection(unit_ids: list[str], meta_df: pd.DataFrame, lambda_join: float = 1.0):
    """
    Thuật toán Quy hoạch động Viterbi tìm chuỗi candidate tối ưu:
      theta* = arg min sum(C_target) + lambda * sum(C_join)
    """
    K = len(unit_ids)
    num_cand = 2  # mỗi unit có 2 candidate: _01 và _02
    
    # Bảng chi phí DP V[step, cand_idx]
    V = np.zeros((K, num_cand))
    backpointer = np.zeros((K, num_cand), dtype=int)
    
    # Bộ đệm nạp âm thanh để tính join cost
    cached_waves = {}
    for j, uid in enumerate(unit_ids):
        for c in [1, 2]:
            p = get_unit_path(uid, c)
            cached_waves[(j, c)] = match_rms(load_audio(p))
            
    # Bước khởi tạo (j = 0)
    for c in [1, 2]:
        V[0, c - 1] = compute_target_cost(unit_ids[0], c, meta_df)
        
    # Bước chuyển tiếp quy hoạch động (j = 1 -> K-1)
    for j in range(1, K):
        for c_curr in [1, 2]:
            curr_target = compute_target_cost(unit_ids[j], c_curr, meta_df)
            best_prev_cost = float("inf")
            best_prev_cand = 1
            
            wave_curr = cached_waves[(j, c_curr)]
            for c_prev in [1, 2]:
                wave_prev = cached_waves[(j - 1, c_prev)]
                j_cost = compute_join_cost(wave_prev, wave_curr)
                total_step_cost = V[j - 1, c_prev - 1] + lambda_join * j_cost
                
                if total_step_cost < best_prev_cost:
                    best_prev_cost = total_step_cost
                    best_prev_cand = c_prev
                    
            V[j, c_curr - 1] = curr_target + best_prev_cost
            backpointer[j, c_curr - 1] = best_prev_cand
            
    # Tìm candidate kết thúc có chi phí nhỏ nhất
    best_last_cand = int(np.argmin(V[K - 1, :])) + 1
    best_path = [best_last_cand]
    
    # Truy vết ngược (Backtracking)
    for j in range(K - 1, 0, -1):
        prev_cand = backpointer[j, best_path[-1] - 1]
        best_path.append(prev_cand)
    best_path.reverse()
    
    total_opt_cost = float(np.min(V[K - 1, :]))
    return best_path, total_opt_cost, V

def synthesize_concatenative(
    text: str,
    output_prefix: str = "sentence1",
    xfade_ms: float = 15.0,
    meta_df: pd.DataFrame = None
):
    """
    Tổng hợp trọn vẹn một câu văn bản bằng Concatenative TTS:
      1. Baseline: Candidate 1 cố định, xfade = 0 ms -> raw_concat.wav
      2. Unit Selection: Viterbi tìm chuỗi candidate tối ưu
      3. Smoothing: Nối với crossfade 15 ms -> best_concat.wav
      4. Thí nghiệm các mức cross-fade 0, 5, 15, 30 ms
    """
    tokens = normalize_queue_text(text)
    unit_ids = tokens_to_unit_ids(tokens)
    
    os.makedirs(AUDIO_OUT_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)
    
    # --- 1. Baseline Concatenation ---
    baseline_paths = [get_unit_path(uid, 1) for uid in unit_ids]
    raw_audio, raw_boundaries = concat_audio_sequence(baseline_paths, sr=16000, xfade_ms=0.0)
    raw_path = os.path.join(AUDIO_OUT_DIR, f"raw_concat_{output_prefix}.wav" if output_prefix != "sentence1" else "raw_concat.wav")
    sf.write(raw_path, raw_audio, 16000)
    print(f"[+] Đã tạo Baseline Raw: {raw_path} (độ dài = {len(raw_audio)/16000:.2f}s)")
    
    # Vẽ biểu đồ Baseline có đánh dấu biên
    fig2_path = os.path.join(FIG_DIR, f"fig2_baseline_boundary_{output_prefix}.png" if output_prefix != "sentence1" else "fig2_baseline_boundary.png")
    plot_waveform_and_spectrogram(
        raw_audio,
        sr=16000,
        boundaries_samples=raw_boundaries,
        boundary_labels=tokens,
        title=f"Baseline Concatenation (xfade=0ms) - '{text}'",
        save_path=fig2_path
    )
    
    # --- 2. Unit Selection (Viterbi) ---
    best_cands, opt_cost, V_table = run_viterbi_unit_selection(unit_ids, meta_df, lambda_join=1.0)
    opt_paths = [get_unit_path(uid, c) for uid, c in zip(unit_ids, best_cands)]
    
    # Tính chi phí của chuỗi Candidate 1 thuần túy để so sánh
    base_join_costs = [
        compute_join_cost(match_rms(load_audio(p1)), match_rms(load_audio(p2)))
        for p1, p2 in zip(baseline_paths[:-1], baseline_paths[1:])
    ]
    base_target_costs = [compute_target_cost(uid, 1, meta_df) for uid in unit_ids]
    total_baseline_cost = sum(base_target_costs) + sum(base_join_costs)
    
    print(f"[i] Chi phí Baseline: {total_baseline_cost:.2f} | Chi phí Tối ưu (Unit Selection): {opt_cost:.2f}")
    print(f"[i] Chuỗi candidate được chọn: {best_cands}")
    
    # --- 3. Best Concatenation (Smoothing 15 ms) ---
    best_audio, best_boundaries = concat_audio_sequence(opt_paths, sr=16000, xfade_ms=xfade_ms)
    best_path = os.path.join(AUDIO_OUT_DIR, f"best_concat_{output_prefix}.wav" if output_prefix != "sentence1" else "best_concat.wav")
    sf.write(best_path, best_audio, 16000)
    print(f"[+] Đã tạo Best Concat: {best_path} (độ dài = {len(best_audio)/16000:.2f}s)")
    
    # --- 4. Thí nghiệm Cross-fade 0, 5, 15, 30 ms (Áp dụng cho câu chính sentence1) ---
    xfade_results = {}
    if output_prefix == "sentence1":
        for ms_val in [0, 5, 15, 30]:
            audio_xfade, _ = concat_audio_sequence(opt_paths, sr=16000, xfade_ms=float(ms_val))
            out_f = os.path.join(AUDIO_OUT_DIR, f"xfade_{ms_val}ms.wav")
            sf.write(out_f, audio_xfade, 16000)
            
            # Đo độ chênh phổ trung bình tại các biên
            discontinuities = []
            cur_idx = 0
            for p1, p2 in zip(opt_paths[:-1], opt_paths[1:]):
                w1 = match_rms(load_audio(p1))
                w2 = match_rms(load_audio(p2))
                disc = calc_boundary_discontinuity(w1, w2, ms=25)
                discontinuities.append(disc)
            avg_disc = np.mean(discontinuities)
            xfade_results[ms_val] = avg_disc
            
        # Vẽ biểu đồ so sánh cross-fade
        fig3_path = os.path.join(FIG_DIR, "fig3_crossfade_comparison.png")
        plot_crossfade_comparison(opt_paths, [0, 5, 15, 30], fig3_path)
        
        # Vẽ lưới Unit Selection Lattice
        fig4_path = os.path.join(FIG_DIR, "fig4_unit_selection_lattice.png")
        plot_unit_selection_lattice(unit_ids, tokens, best_cands, V_table, fig4_path)
        
    return {
        "text": text,
        "tokens": tokens,
        "best_candidates": best_cands,
        "baseline_cost": total_baseline_cost,
        "optimal_cost": opt_cost,
        "raw_path": raw_path,
        "best_path": best_path,
        "xfade_results": xfade_results
    }

def plot_crossfade_comparison(opt_paths, xfade_list, save_path):
    """Vẽ dạng sóng cận cảnh tại biên nối đầu tiên (mời -> khách) với 4 mức cross-fade."""
    w1 = match_rms(load_audio(opt_paths[0]))
    w2 = match_rms(load_audio(opt_paths[1]))
    
    fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
    plt.subplots_adjust(hspace=0.35)
    
    for ax, ms_val in zip(axes, xfade_list):
        n_fade = int(16000 * ms_val / 1000)
        merged = crossfade(w1, w2, n_fade)
        # Lấy vùng cửa sổ xung quanh biên: 100 ms trước và 100 ms sau biên
        boundary_idx = len(w1) - n_fade // 2 if n_fade > 0 else len(w1)
        start_crop = max(0, boundary_idx - 1600)
        end_crop = min(len(merged), boundary_idx + 1600)
        crop_signal = merged[start_crop:end_crop]
        t = (np.arange(len(crop_signal)) - (boundary_idx - start_crop)) / 16.0  # ms
        
        ax.plot(t, crop_signal, color="#2ca02c" if ms_val == 15 else "#1f77b4", linewidth=1.1)
        ax.axvline(x=0, color="#d62728", linestyle="--", linewidth=1.2, label="Biên nối" if ms_val == 0 else None)
        ax.set_ylabel("Biên độ", fontsize=9)
        ax.set_title(f"Cross-fade = {ms_val} ms (Vùng overlap = {n_fade} mẫu / {ms_val} ms)", fontsize=10, fontweight="bold")
        ax.grid(True, linestyle=":", alpha=0.6)
        if ms_val == 0:
            ax.legend(loc="upper right")
            
    axes[-1].set_xlabel("Thời gian tương đối quanh điểm nối (ms)", fontsize=10, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=180)
    plt.close(fig)
    print(f"[+] Đã lưu biểu đồ Cross-fade Comparison: {save_path}")

def plot_unit_selection_lattice(unit_ids, tokens, best_cands, V_table, save_path):
    """Vẽ mạng lưới tìm kiếm đường đi tối ưu Viterbi trên lưới candidate."""
    K = len(tokens)
    fig, ax = plt.subplots(figsize=(12, 5))
    
    # Tọa độ các nút
    for j in range(K):
        for c in [1, 2]:
            y_pos = 2 - c  # c=1 -> y=1, c=2 -> y=0
            # Nút được chọn thì tô màu nổi bật
            is_chosen = (best_cands[j] == c)
            node_color = "#e74c3c" if is_chosen else "#bdc3c7"
            ax.scatter(j, y_pos, color=node_color, s=280, zorder=4, edgecolor="#2c3e50", linewidth=1.5)
            cand_label = f"c{c}\n{V_table[j, c-1]:.1f}"
            ax.text(j, y_pos, cand_label, ha="center", va="center", color="white" if is_chosen else "#2c3e50",
                    fontsize=7.5, fontweight="bold", zorder=5)
            
    # Vẽ các cạnh kết nối
    for j in range(K - 1):
        for c1 in [1, 2]:
            y1 = 2 - c1
            for c2 in [1, 2]:
                y2 = 2 - c2
                is_opt_edge = (best_cands[j] == c1 and best_cands[j+1] == c2)
                edge_color = "#e74c3c" if is_opt_edge else "#e0e0e0"
                edge_width = 2.5 if is_opt_edge else 0.8
                edge_alpha = 1.0 if is_opt_edge else 0.4
                ax.plot([j, j+1], [y1, y2], color=edge_color, linewidth=edge_width, alpha=edge_alpha, zorder=2)
                
    ax.set_xticks(range(K))
    ax.set_xticklabels([f"{tok}\n({uid})" for tok, uid in zip(tokens, unit_ids)], fontsize=9, fontweight="bold")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Candidate 2\n(NamMinh / alt)", "Candidate 1\n(HoaiMy / std)"], fontsize=10, fontweight="bold")
    ax.set_title("Mạng lưới Lựa chọn đơn vị (Viterbi Unit Selection Lattice) - Đường đỏ là tối ưu", fontsize=11, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.4, axis="x")
    ax.set_ylim(-0.5, 1.5)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=180)
    plt.close(fig)
    print(f"[+] Đã lưu biểu đồ Unit Selection Lattice: {save_path}")

if __name__ == "__main__":
    meta_df = pd.read_csv(METADATA_PATH) if os.path.exists(METADATA_PATH) else None
    res = synthesize_concatenative("Mời khách hàng số 23 đến quầy 5.", output_prefix="sentence1", meta_df=meta_df)
    print("Hoàn thành thử nghiệm Concatenative Synthesis.")
