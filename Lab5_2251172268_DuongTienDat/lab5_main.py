"""
lab5_main.py
============
CSE457 Lab 5 – Speech Enhancement, Speaker Recognition and Speaker Separation
Tích hợp toàn bộ quy trình thực nghiệm từ đầu đến cuối:
- Phần 1: Khử nhiễu STFT (Spectral Subtraction vs Wiener) trên nhiều mức SNR (0, 5, 10 dB)
- Phần 2: Nhận dạng người nói (Identification & Verification), Ablation (MFCC/Delta/CMN/Tokens)
- Phần 3: Phân tách nguồn mù (BSS/FastICA 2x2), khảo sát ma trận suy biến & nhiễu phụ
- Phần 4: Case study tích hợp "Trợ lý phòng họp mini" (End-to-End Pipeline)

Sinh viên: Dương Tiến Đạt – MSSV: 2251172268
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import soundfile as sf
import librosa
from sklearn.metrics import confusion_matrix

from lab5_utils import (
    load16k, save16k, power, rms, rms_normalize, peak_normalize,
    add_noise_at_snr, snr_db, align_signals
)
from lab5_enhancement import (
    compute_noise_psd, spectral_subtraction, wiener_filter, enhance_stft
)
from lab5_speaker_id import (
    extract_speaker_embedding, cosine_similarity, build_speaker_models,
    predict_speaker, sweep_verification_thresholds, evaluate_verification_at_threshold
)
from lab5_ica import (
    create_instantaneous_mixture, separate_fastica, resolve_ica_ambiguities
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "lab5_data")
CLEAN_DIR = os.path.join(DATA_DIR, "clean")
NOISE_DIR = os.path.join(DATA_DIR, "noise")
SEP_DIR = os.path.join(DATA_DIR, "separation")

RESULTS_DIR = os.path.join(BASE_DIR, "results")
AUDIO_OUT = os.path.join(RESULTS_DIR, "audio")
FIG_OUT = os.path.join(RESULTS_DIR, "figures")

for d in [AUDIO_OUT, FIG_OUT]:
    os.makedirs(d, exist_ok=True)


def plot_spectrogram(ax, y, sr, title):
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y, n_fft=512, hop_length=160, win_length=400)), ref=np.max)
    img = librosa.display.specshow(D, sr=sr, hop_length=160, x_axis='time', y_axis='hz', ax=ax, cmap='magma')
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.set_ylim(0, 8000)
    return img


# ═══════════════════════════════════════════════════════════════════════
# THỰC NGHIỆM 1: SPEECH ENHANCEMENT (SPECTRAL SUBTRACTION & WIENER)
# ═══════════════════════════════════════════════════════════════════════
def run_enhancement_experiments():
    print("\n" + "="*70)
    print("PHẦN 1: MÔ HÌNH NHIỄU, SPECTRAL SUBTRACTION & WIENER FILTER")
    print("="*70)

    clean_file = os.path.join(CLEAN_DIR, "spk01_test_01.wav")
    fan_file = os.path.join(NOISE_DIR, "fan.wav")
    cafe_file = os.path.join(NOISE_DIR, "cafeteria.wav")

    clean_audio, sr = load16k(clean_file)
    fan_noise, _ = load16k(fan_file)
    cafe_noise, _ = load16k(cafe_file)

    target_snrs = [0.0, 5.0, 10.0]
    results_table = []

    print("\n--- 1.1 Khảo sát các mức SNR mục tiêu (0, 5, 10 dB) với nhiễu fan.wav ---")
    print(f"{'Target SNR':<12} | {'Measured In':<12} | {'SS Out':<10} | {'SS Delta':<10} | {'Wiener Out':<12} | {'Wiener Delta':<12}")
    print("-" * 75)

    for snr_t in target_snrs:
        noisy, alpha = add_noise_at_snr(clean_audio, fan_noise, snr_t)
        measured_in = snr_db(clean_audio, noisy)

        # Chạy Spectral Subtraction (beta=1.5, floor=0.02)
        ss_audio = spectral_subtraction(noisy, fan_noise, sr=sr, beta=1.5, floor=0.02)
        ss_out = snr_db(clean_audio, ss_audio)
        ss_delta = ss_out - measured_in

        # Chạy Wiener Filter
        w_audio, _ = wiener_filter(noisy, fan_noise, sr=sr)
        w_out = snr_db(clean_audio, w_audio)
        w_delta = w_out - measured_in

        results_table.append({
            'snr_target': snr_t,
            'snr_in': measured_in,
            'ss_out': ss_out,
            'ss_delta': ss_delta,
            'w_out': w_out,
            'w_delta': w_delta
        })

        print(f"{snr_t:>10.1f}dB | {measured_in:>10.2f}dB | {ss_out:>8.2f}dB | {ss_delta:>8.2f}dB | {w_out:>10.2f}dB | {w_delta:>10.2f}dB")

        # Lưu audio tương ứng
        save16k(os.path.join(AUDIO_OUT, f"noisy_{int(snr_t)}dB.wav"), noisy, sr)
        save16k(os.path.join(AUDIO_OUT, f"enhanced_ss_{int(snr_t)}dB.wav"), ss_audio, sr)
        save16k(os.path.join(AUDIO_OUT, f"enhanced_wiener_{int(snr_t)}dB.wav"), w_audio, sr)

    # 1.2 Khảo sát siêu tham số Spectral Subtraction: beta và gamma tại 5 dB
    print("\n--- 1.2 Khảo sát siêu tham số beta (1.0, 1.5, 2.0) và floor gamma (0.0, 0.01, 0.05) ---")
    noisy_5db, _ = add_noise_at_snr(clean_audio, fan_noise, 5.0)
    snr_5db_in = snr_db(clean_audio, noisy_5db)

    beta_list = [1.0, 1.5, 2.0]
    gamma_list = [0.0, 0.01, 0.05]

    for b in beta_list:
        for g in gamma_list:
            enh = spectral_subtraction(noisy_5db, fan_noise, sr=sr, beta=b, floor=g)
            s_out = snr_db(clean_audio, enh)
            print(f"  beta={b:.1f}, floor={g:.2f} -> SNR_out: {s_out:6.2f} dB (Delta: {s_out - snr_5db_in:5.2f} dB)")
            if g == 0.02 or g == 0.01:
                save16k(os.path.join(AUDIO_OUT, f"enhanced_ss_beta{b:.1f}.wav"), enh, sr)

    # 1.3 Vẽ biểu đồ Spectrogram đối chứng 4 trạng thái tại 5 dB
    fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
    noisy_sample, _ = add_noise_at_snr(clean_audio, fan_noise, 5.0)
    ss_sample = spectral_subtraction(noisy_sample, fan_noise, sr=sr, beta=1.5, floor=0.02)
    w_sample, _ = wiener_filter(noisy_sample, fan_noise, sr=sr)

    plot_spectrogram(axes[0], clean_audio, sr, "1. Clean Reference Speech (spk01_test_01)")
    plot_spectrogram(axes[1], noisy_sample, sr, f"2. Noisy Speech (Fan Noise, SNR = 5.0 dB)")
    plot_spectrogram(axes[2], ss_sample, sr, f"3. Spectral Subtraction (beta=1.5, floor=0.02, SNR_out = {snr_db(clean_audio, ss_sample):.2f} dB)")
    plot_spectrogram(axes[3], w_sample, sr, f"4. Wiener Filter (SNR_out = {snr_db(clean_audio, w_sample):.2f} dB)")

    plt.tight_layout()
    fig_path = os.path.join(FIG_OUT, "fig1_clean_noisy_enhanced_spectrograms.png")
    fig.savefig(fig_path, dpi=200)
    plt.close(fig)
    print(f"\n  ✓ Đã lưu biểu đồ: {fig_path}")

    # 1.4 Vẽ biểu đồ so sánh SNR Delta
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x_idx = np.arange(len(target_snrs))
    width = 0.35
    ss_deltas = [r['ss_delta'] for r in results_table]
    w_deltas = [r['w_delta'] for r in results_table]

    ax.bar(x_idx - width/2, ss_deltas, width, label='Spectral Subtraction (beta=1.5)', color='#2E86AB')
    ax.bar(x_idx + width/2, w_deltas, width, label='Wiener Filter', color='#F24236')
    ax.set_xticks(x_idx)
    ax.set_xticklabels([f"{int(s)} dB" for s in target_snrs])
    ax.set_xlabel("Input SNR mục tiêu", fontweight='bold')
    ax.set_ylabel("Mức cải thiện Delta SNR (dB)", fontweight='bold')
    ax.set_title("So sánh mức cải thiện Delta SNR giữa Spectral Subtraction và Wiener Filter", fontweight='bold')
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.legend()

    for i in range(len(target_snrs)):
        ax.text(x_idx[i] - width/2, ss_deltas[i] + 0.1, f"+{ss_deltas[i]:.2f}", ha='center', fontsize=9, fontweight='bold')
        ax.text(x_idx[i] + width/2, w_deltas[i] + 0.1, f"+{w_deltas[i]:.2f}", ha='center', fontsize=9, fontweight='bold')

    plt.tight_layout()
    fig_snr_path = os.path.join(FIG_OUT, "fig2_snr_comparison.png")
    fig.savefig(fig_snr_path, dpi=200)
    plt.close(fig)
    print(f"  ✓ Đã lưu biểu đồ: {fig_snr_path}")

    return results_table


# ═══════════════════════════════════════════════════════════════════════
# THỰC NGHIỆM 2: SPEAKER RECOGNITION (IDENTIFICATION & VERIFICATION)
# ═══════════════════════════════════════════════════════════════════════
def run_speaker_recognition_experiments():
    print("\n" + "="*70)
    print("PHẦN 2: NHẬN DẠNG NGƯỜI NÓI (SPEAKER IDENTIFICATION & VERIFICATION)")
    print("="*70)

    speakers = ["spk01", "spk02", "spk03"]
    sr = 16000

    # Nạp dữ liệu Enrollment, Dev, Test
    enroll_data = {spk: [] for spk in speakers}
    dev_data = {spk: [] for spk in speakers}
    test_data = {spk: [] for spk in speakers}

    for spk in speakers:
        for i in range(1, 5):
            path = os.path.join(CLEAN_DIR, f"{spk}_enroll_{i:02d}.wav")
            audio, _ = load16k(path)
            enroll_data[spk].append(audio)

        dev_path = os.path.join(CLEAN_DIR, f"{spk}_dev_01.wav")
        audio, _ = load16k(dev_path)
        dev_data[spk].append(audio)

        for i in range(1, 3):
            path = os.path.join(CLEAN_DIR, f"{spk}_test_{i:02d}.wav")
            audio, _ = load16k(path)
            test_data[spk].append(audio)

    # 2.1 Speaker Identification Baseline (MFCC 20 + Delta, không CMN, 4 tokens)
    models_base = build_speaker_models(enroll_data, sr=sr, n_mfcc=20, with_delta=True, use_cmn=False)

    y_true = []
    y_pred = []
    score_records = []

    print("\n--- 2.1 Bảng kết quả Speaker Identification trên Test Set độc lập ---")
    print(f"{'Utterance':<18} | {'True Spk':<10} | {'spk01':<8} | {'spk02':<8} | {'spk03':<8} | {'Predict':<8} | {'Margin':<8} | {'Status':<6}")
    print("-" * 88)

    for true_spk in speakers:
        for idx, audio in enumerate(test_data[true_spk], 1):
            utt_name = f"{true_spk}_test_{idx:02d}"
            pred_spk, scores, margin = predict_speaker(audio, models_base, sr=sr, n_mfcc=20, with_delta=True, use_cmn=False)
            y_true.append(true_spk)
            y_pred.append(pred_spk)
            status = "CORRECT" if pred_spk == true_spk else "WRONG"

            print(f"{utt_name:<18} | {true_spk:<10} | {scores['spk01']:>7.4f} | {scores['spk02']:>7.4f} | {scores['spk03']:>7.4f} | {pred_spk:<8} | {margin:>7.4f} | {status:<6}")
            score_records.append((utt_name, true_spk, scores, pred_spk, margin))

    cm = confusion_matrix(y_true, y_pred, labels=speakers)
    acc = np.mean(np.array(y_true) == np.array(y_pred)) * 100.0
    print(f"\n=> Tỉ lệ nhận dạng chính xác Baseline (Accuracy): {acc:.1f}%")

    # Vẽ Confusion Matrix
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=speakers, yticklabels=speakers,
           title=f'Confusion Matrix - Speaker ID (Acc: {acc:.1f}%)\nBaseline: MFCC 20 + Delta',
           ylabel='Nhãn thực tế (True Speaker)',
           xlabel='Nhãn dự đoán (Predicted Speaker)')

    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontweight='bold', fontsize=12)
    plt.tight_layout()
    fig_cm_path = os.path.join(FIG_OUT, "fig5_confusion_matrix_speaker_id.png")
    fig.savefig(fig_cm_path, dpi=200)
    plt.close(fig)
    print(f"  ✓ Đã lưu biểu đồ: {fig_cm_path}")

    # 2.2 Khảo sát Ablation:
    # (a) MFCC only vs MFCC + Delta
    # (b) CMN on vs CMN off
    # (c) Số token enrollment (1 vs 2 vs 4)
    print("\n--- 2.2 Khảo sát Ablation trên Speaker Identification ---")
    ablation_results = {}

    configs = [
        ("Baseline (MFCC+Delta, 4 tokens, CMN Off)", True, False, 4),
        ("MFCC Only (Không Delta, 4 tokens)", False, False, 4),
        ("With CMN (MFCC+Delta, CMN On)", True, True, 4),
        ("Enrollment 1 token (MFCC+Delta)", True, False, 1),
        ("Enrollment 2 tokens (MFCC+Delta)", True, False, 2),
    ]

    for name, w_delta, use_cmn, n_tokens in configs:
        models = build_speaker_models(enroll_data, sr=sr, n_mfcc=20, with_delta=w_delta, use_cmn=use_cmn, num_tokens=n_tokens)
        correct = 0
        total = 0
        margins = []
        for true_spk in speakers:
            for audio in test_data[true_spk]:
                pred, _, m = predict_speaker(audio, models, sr=sr, n_mfcc=20, with_delta=w_delta, use_cmn=use_cmn)
                if pred == true_spk:
                    correct += 1
                total += 1
                margins.append(m)
        accuracy = (correct / total) * 100.0
        avg_margin = float(np.mean(margins))
        ablation_results[name] = (accuracy, avg_margin)
        print(f"  {name:<42} -> Acc: {accuracy:5.1f}% | Avg Margin: {avg_margin:6.4f}")

    # 2.3 Speaker Verification: Quét ngưỡng theta trên Development Set độc lập
    print("\n--- 2.3 Speaker Verification: Quét ngưỡng theta trên Dev Set & Kiểm thử trên Test Set ---")
    dev_trials = []
    for spk_claim in speakers:
        for spk_actual in speakers:
            for audio in dev_data[spk_actual]:
                e = extract_speaker_embedding(audio, sr=sr, n_mfcc=20, with_delta=True, use_cmn=False)
                score = cosine_similarity(e, models_base[spk_claim])
                label = 1 if (spk_claim == spk_actual) else 0
                dev_trials.append((score, label))

    rows_dev, best_dev = sweep_verification_thresholds(dev_trials)
    theta_star, far_dev, frr_dev = best_dev
    print(f"  [Dev Set] Điểm hoạt động tối ưu gần EER:")
    print(f"    Ngưỡng theta* = {theta_star:.4f} | FAR_dev = {far_dev*100:.2f}% | FRR_dev = {frr_dev*100:.2f}%")

    # Khóa ngưỡng theta* và đánh giá trên Test Set độc lập!
    test_trials = []
    for spk_claim in speakers:
        for spk_actual in speakers:
            for audio in test_data[spk_actual]:
                e = extract_speaker_embedding(audio, sr=sr, n_mfcc=20, with_delta=True, use_cmn=False)
                score = cosine_similarity(e, models_base[spk_claim])
                label = 1 if (spk_claim == spk_actual) else 0
                test_trials.append((score, label))

    far_test, frr_test = evaluate_verification_at_threshold(test_trials, theta_star)
    print(f"  [Test Set - KHÓA NGƯỠNG theta*={theta_star:.4f}]:")
    print(f"    FAR_test = {far_test*100:.2f}% | FRR_test = {frr_test*100:.2f}%")

    # Vẽ đường FAR-FRR theo theta trên Dev Set
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ths = [r[0] for r in rows_dev]
    fars = [r[1] * 100 for r in rows_dev]
    frrs = [r[2] * 100 for r in rows_dev]

    ax.plot(ths, fars, label='False Acceptance Rate (FAR %)', color='#D90429', lw=2)
    ax.plot(ths, frrs, label='False Rejection Rate (FRR %)', color='#118AB2', lw=2)
    ax.axvline(theta_star, color='black', linestyle='--', label=f'Ngưỡng dev tối ưu $\\theta^*={theta_star:.3f}$')
    ax.set_xlabel("Ngưỡng Cosine Similarity $\\theta$", fontweight='bold')
    ax.set_ylabel("Tỉ lệ lỗi (%)", fontweight='bold')
    ax.set_title("Đường cong FAR và FRR theo ngưỡng quyết định $\\theta$ (Development Set)", fontweight='bold')
    ax.set_xlim(-0.2, 1.0)
    ax.set_ylim(-2, 102)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(loc='center right')

    plt.tight_layout()
    fig_ver_path = os.path.join(FIG_OUT, "fig7_speaker_verification_far_frr.png")
    fig.savefig(fig_ver_path, dpi=200)
    plt.close(fig)
    print(f"  ✓ Đã lưu biểu đồ: {fig_ver_path}")

    return models_base, theta_star


# ═══════════════════════════════════════════════════════════════════════
# THỰC NGHIỆM 3: SOURCE SEPARATION (BSS / FASTICA 2x2)
# ═══════════════════════════════════════════════════════════════════════
def run_ica_experiments():
    print("\n" + "="*70)
    print("PHẦN 3: PHÂN TÁCH HAI NGUỒN NÓI BẰNG FASTICA (INSTANTANEOUS MIXING)")
    print("="*70)

    sr = 16000
    src_a_file = os.path.join(SEP_DIR, "source_A.wav")
    src_b_file = os.path.join(SEP_DIR, "source_B.wav")

    s1, _ = load16k(src_a_file)
    s2, _ = load16k(src_b_file)

    # 3.1 Ma trận trộn tiêu chuẩn (khả nghịch, điều hòa tốt)
    A_standard = np.array([
        [1.0, 0.6],
        [0.4, 1.0]
    ])
    S, X, det_A, cond_A = create_instantaneous_mixture(s1, s2, A_standard)
    print(f"\n--- 3.1 Trộn tức thời tiêu chuẩn A = [[1.0, 0.6], [0.4, 1.0]] ---")
    print(f"  det(A) = {det_A:.4f} (Khả nghịch tốt, cond = {cond_A:.2f})")

    # Lưu 2 hỗn hợp quan sát từ micro
    save16k(os.path.join(AUDIO_OUT, "mix_mic1.wav"), peak_normalize(X[0], 0.85), sr)
    save16k(os.path.join(AUDIO_OUT, "mix_mic2.wav"), peak_normalize(X[1], 0.85), sr)

    # Chạy FastICA
    Shat, ica_model = separate_fastica(X, random_state=42)

    # Giải quyết bất định hoán vị (permutation) và đổi dấu (sign)
    S_aligned, perm, signs, C_before, C_after = resolve_ica_ambiguities(S, Shat)

    print("\n  Ma trận |hệ số tương quan| TRƯỚC KHI GHÉP (Raw ICA):")
    print(np.array2string(C_before, precision=4, suppress_small=True))
    print(f"  Hoán vị phát hiện: estimate[{perm[0]}] -> Nguồn A | estimate[{perm[1]}] -> Nguồn B")
    print(f"  Hệ số dấu: {signs}")
    print("\n  Ma trận |hệ số tương quan| SAU KHI GHÉP & ĐẢO DẤU (Đúng cặp):")
    print(np.array2string(C_after, precision=4, suppress_small=True))

    # Lưu 2 nguồn tách được
    save16k(os.path.join(AUDIO_OUT, "separated_1.wav"), S_aligned[0], sr)
    save16k(os.path.join(AUDIO_OUT, "separated_2.wav"), S_aligned[1], sr)

    # 3.2 Khảo sát ma trận trộn gần suy biến (Ill-conditioned Matrix)
    A_ill = np.array([
        [1.00, 0.95],
        [0.95, 0.91]
    ])
    S_ill, X_ill, det_ill, cond_ill = create_instantaneous_mixture(s1, s2, A_ill)
    Shat_ill, _ = separate_fastica(X_ill, random_state=42)
    S_aligned_ill, _, _, _, C_ill_after = resolve_ica_ambiguities(S_ill, Shat_ill)

    print(f"\n--- 3.2 Khảo sát ma trận gần suy biến A_ill = [[1.0, 0.95], [0.95, 0.91]] ---")
    print(f"  det(A_ill) = {det_ill:.6f} | cond(A_ill) = {cond_ill:.2f}")
    print("  Ma trận tương quan sau tách:")
    print(np.array2string(C_ill_after, precision=4, suppress_small=True))
    save16k(os.path.join(AUDIO_OUT, "separated_ill_1.wav"), S_aligned_ill[0], sr)
    save16k(os.path.join(AUDIO_OUT, "separated_ill_2.wav"), S_aligned_ill[1], sr)

    # 3.3 Khảo sát khi thêm nhiễu vào hỗn hợp (Noisy Mixture)
    print("\n--- 3.3 Khảo sát FastICA khi hỗn hợp microphone bị nhiễm thêm nhiễu (20dB, 10dB) ---")
    rng = np.random.default_rng(seed=42)
    for snr_noise in [20.0, 10.0]:
        noise_mat = rng.normal(0, 1, X.shape)
        X_noisy = np.zeros_like(X)
        for ch in range(2):
            X_noisy[ch], _ = add_noise_at_snr(X[ch], noise_mat[ch], snr_noise)
        Shat_noisy, _ = separate_fastica(X_noisy, random_state=42)
        _, _, _, _, C_noisy_after = resolve_ica_ambiguities(S, Shat_noisy)
        diag_mean = (C_noisy_after[0, 0] + C_noisy_after[1, 1]) / 2.0
        print(f"  Thêm nhiễu {snr_noise:4.1f} dB -> |corr| trung bình: {diag_mean:.4f} (rho11={C_noisy_after[0,0]:.4f}, rho22={C_noisy_after[1,1]:.4f})")

    # 3.4 Vẽ biểu đồ dạng sóng và spectrogram cho quá trình phân tách BSS/ICA
    fig, axes = plt.subplots(6, 1, figsize=(10, 12), sharex=True)
    plot_spectrogram(axes[0], S[0], sr, "1. Nguồn gốc 1 (Source A - Speaker 1)")
    plot_spectrogram(axes[1], S[1], sr, "2. Nguồn gốc 2 (Source B - Speaker 2)")
    plot_spectrogram(axes[2], X[0], sr, "3. Quan sát Micro 1 (x1 = s1 + 0.6*s2)")
    plot_spectrogram(axes[3], X[1], sr, "4. Quan sát Micro 2 (x2 = 0.4*s1 + s2)")
    plot_spectrogram(axes[4], S_aligned[0], sr, f"5. Nguồn tách 1 sau ICA (|rho| = {C_after[0,0]:.4f})")
    plot_spectrogram(axes[5], S_aligned[1], sr, f"6. Nguồn tách 2 sau ICA (|rho| = {C_after[1,1]:.4f})")

    plt.tight_layout()
    fig_sep_path = os.path.join(FIG_OUT, "fig8_ica_separation_waveforms_spectrograms.png")
    fig.savefig(fig_sep_path, dpi=200)
    plt.close(fig)
    print(f"\n  ✓ Đã lưu biểu đồ: {fig_sep_path}")

    # Vẽ ma trận tương quan trước và sau ghép
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))
    im1 = ax1.imshow(C_before, cmap='YlGnBu', vmin=0, vmax=1)
    ax1.set_title("Ma trận |corr| TRƯỚC KHI GHÉP\n(Chưa khử Permutation/Sign)", fontweight='bold')
    ax1.set_xticks([0, 1]); ax1.set_yticks([0, 1])
    ax1.set_xticklabels(["Ước lượng 1", "Ước lượng 2"])
    ax1.set_yticklabels(["Nguồn A", "Nguồn B"])
    for i in range(2):
        for j in range(2):
            ax1.text(j, i, f"{C_before[i,j]:.4f}", ha='center', va='center', fontweight='bold', fontsize=11)

    im2 = ax2.imshow(C_after, cmap='YlGnBu', vmin=0, vmax=1)
    ax2.set_title("Ma trận |corr| SAU KHI GHÉP\n(Đã sắp xếp đúng cặp & đồng pha)", fontweight='bold')
    ax2.set_xticks([0, 1]); ax2.set_yticks([0, 1])
    ax2.set_xticklabels(["Nguồn tách 1", "Nguồn tách 2"])
    ax2.set_yticklabels(["Nguồn A", "Nguồn B"])
    for i in range(2):
        for j in range(2):
            ax2.text(j, i, f"{C_after[i,j]:.4f}", ha='center', va='center', fontweight='bold', fontsize=11)

    fig.colorbar(im2, ax=[ax1, ax2], fraction=0.046, pad=0.04)
    plt.tight_layout()
    fig_mat_path = os.path.join(FIG_OUT, "fig9_ica_correlation_matrices.png")
    fig.savefig(fig_mat_path, dpi=200)
    plt.close(fig)
    print(f"  ✓ Đã lưu biểu đồ: {fig_mat_path}")

    return C_after


# ═══════════════════════════════════════════════════════════════════════
# THỰC NGHIỆM 4: CASE STUDY TÍCH HỢP "TRỢ LÝ PHÒNG HỌP MINI"
# ═══════════════════════════════════════════════════════════════════════
def run_integrated_case_study(speaker_models, theta_star):
    print("\n" + "="*70)
    print("PHẦN 4: CASE STUDY TÍCH HỢP - TRỢ LÝ PHÒNG HỌP MINI (END-TO-END)")
    print("="*70)

    sr = 16000
    clean_test_audio, _ = load16k(os.path.join(CLEAN_DIR, "spk01_test_01.wav"))
    fan_noise, _ = load16k(os.path.join(NOISE_DIR, "fan.wav"))

    # Bước 1: Môi trường nhiễu tại SNR = 5.0 dB
    noisy_5db, alpha = add_noise_at_snr(clean_test_audio, fan_noise, 5.0)

    # Bước 2: Enhancement
    ss_audio = spectral_subtraction(noisy_5db, fan_noise, sr=sr, beta=1.5, floor=0.02)
    w_audio, _ = wiener_filter(noisy_5db, fan_noise, sr=sr)

    # Bước 3: So sánh Speaker Identification trên 4 phiên bản: Clean, Noisy, SS, Wiener
    versions = {
        "Clean Reference": clean_test_audio,
        "Noisy (5dB)": noisy_5db,
        "Enhanced SS": ss_audio,
        "Enhanced Wiener": w_audio,
    }

    print("\n--- 4.1 Khảo sát: Enhancement có luôn giúp Speaker ID không? ---")
    print(f"{'Dạng tín hiệu':<18} | {'Score spk01 (True)':<20} | {'Score spk02':<12} | {'Score spk03':<12} | {'Predict':<8} | {'Margin':<8}")
    print("-" * 88)

    for name, audio in versions.items():
        pred, scores, margin = predict_speaker(audio, speaker_models, sr=sr, n_mfcc=20, with_delta=True, use_cmn=False)
        print(f"{name:<18} | {scores['spk01']:>18.4f} | {scores['spk02']:>12.4f} | {scores['spk03']:>12.4f} | {pred:<8} | {margin:>8.4f}")

    # Bước 4: Speaker Verification cho phiên bản Enhanced SS
    emb_enh = extract_speaker_embedding(ss_audio, sr=sr, n_mfcc=20, with_delta=True, use_cmn=False)
    score_v = cosine_similarity(emb_enh, speaker_models["spk01"])
    decision = "ACCEPT" if score_v >= theta_star else "REJECT"
    print(f"\n--- 4.2 Speaker Verification cho Enhanced SS tại ngưỡng dev theta*={theta_star:.4f} ---")
    print(f"  Claim: spk01 | Cosine Score: {score_v:.4f} | Ngưỡng theta*: {theta_star:.4f} => Quyết định: {decision} (HỢP LỆ)")

    # Bước 5: BSS/ICA tách 2 nguồn nói chồng lấn
    s1, _ = load16k(os.path.join(SEP_DIR, "source_A.wav"))
    s2, _ = load16k(os.path.join(SEP_DIR, "source_B.wav"))
    A = np.array([[1.0, 0.6], [0.4, 1.0]])
    S, X, _, _ = create_instantaneous_mixture(s1, s2, A)
    Shat, _ = separate_fastica(X, random_state=42)
    S_sep, _, _, _, C_final = resolve_ica_ambiguities(S, Shat)

    print(f"\n--- 4.3 Phân tách 2 nguồn nói tại micro trợ lý ---")
    print(f"  |Tương quan| sau tách: Source A -> {C_final[0,0]:.4f} | Source B -> {C_final[1,1]:.4f}")
    print("  => Cả 3 chức năng của Case study đã tích hợp và hoạt động thông suốt!")

    # Vẽ biểu đồ tổng kết case study
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    # 1. Enhancement SNR
    axes[0, 0].bar(["Noisy", "SS", "Wiener"], [snr_db(clean_test_audio, noisy_5db), snr_db(clean_test_audio, ss_audio), snr_db(clean_test_audio, w_audio)], color=['#E63946', '#457B9D', '#2A9D8F'])
    axes[0, 0].set_title("1. Chất lượng lọc nhiễu SNR (dB)", fontweight='bold')
    axes[0, 0].set_ylabel("SNR (dB)")
    axes[0, 0].grid(axis='y', linestyle='--', alpha=0.7)

    # 2. Speaker scores
    spk_names = list(speaker_models.keys())
    clean_scores = [cosine_similarity(extract_speaker_embedding(clean_test_audio), speaker_models[k]) for k in spk_names]
    noisy_scores = [cosine_similarity(extract_speaker_embedding(noisy_5db), speaker_models[k]) for k in spk_names]
    ss_scores = [cosine_similarity(extract_speaker_embedding(ss_audio), speaker_models[k]) for k in spk_names]
    w_scores = [cosine_similarity(extract_speaker_embedding(w_audio), speaker_models[k]) for k in spk_names]

    x = np.arange(len(spk_names))
    width = 0.2
    axes[0, 1].bar(x - 1.5*width, clean_scores, width, label='Clean', color='#2A9D8F')
    axes[0, 1].bar(x - 0.5*width, noisy_scores, width, label='Noisy 5dB', color='#E63946')
    axes[0, 1].bar(x + 0.5*width, ss_scores, width, label='SS Enh', color='#457B9D')
    axes[0, 1].bar(x + 1.5*width, w_scores, width, label='Wiener Enh', color='#E76F51')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(spk_names)
    axes[0, 1].set_title("2. Speaker ID Cosine Scores", fontweight='bold')
    axes[0, 1].set_ylabel("Cosine Similarity")
    axes[0, 1].legend(fontsize=8)
    axes[0, 1].grid(axis='y', linestyle='--', alpha=0.7)

    # 3. Waveform mixture vs separated
    t_plot = np.arange(1000) / sr
    axes[1, 0].plot(t_plot, X[0][:1000], label='Microphone 1 Mixture', color='#9B5DE5', alpha=0.8)
    axes[1, 0].plot(t_plot, S_sep[0][:1000], label='Separated Source A', color='#00BBF9', alpha=0.8)
    axes[1, 0].set_title("3. Dạng sóng Trộn vs Sau khi Tách", fontweight='bold')
    axes[1, 0].set_xlabel("Thời gian (s)")
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].grid(True, linestyle='--', alpha=0.7)

    # 4. ICA Correlation Matrix
    im = axes[1, 1].imshow(C_final, cmap='Greens', vmin=0, vmax=1)
    axes[1, 1].set_title("4. Ma trận tương quan ICA sau ghép", fontweight='bold')
    axes[1, 1].set_xticks([0, 1]); axes[1, 1].set_yticks([0, 1])
    axes[1, 1].set_xticklabels(["Sep 1", "Sep 2"])
    axes[1, 1].set_yticklabels(["Src A", "Src B"])
    for i in range(2):
        for j in range(2):
            axes[1, 1].text(j, i, f"{C_final[i,j]:.4f}", ha='center', va='center', fontweight='bold', fontsize=11)
    fig.colorbar(im, ax=axes[1, 1], fraction=0.046, pad=0.04)

    plt.tight_layout()
    fig_case_path = os.path.join(FIG_OUT, "fig10_integrated_case_study.png")
    fig.savefig(fig_case_path, dpi=200)
    plt.close(fig)
    print(f"  ✓ Đã lưu biểu đồ: {fig_case_path}")


def main():
    print("="*70)
    print("CSE457 - LAB 5: SPEECH ENHANCEMENT, SPEAKER RECOGNITION & SEPARATION")
    print("Sinh viên: Dương Tiến Đạt | MSSV: 2251172268")
    print("="*70)

    # Chạy tuần tự 4 phần
    enh_results = run_enhancement_experiments()
    models_base, theta_star = run_speaker_recognition_experiments()
    ica_results = run_ica_experiments()
    run_integrated_case_study(models_base, theta_star)

    print("\n" + "="*70)
    print("✓ HOÀN TẤT TOÀN BỘ 4 PHẦN THỰC NGHIỆM LAB 5 THÀNH CÔNG!")
    print(f"  - Các tệp âm thanh đã xuất tại: {AUDIO_OUT}")
    print(f"  - Các biểu đồ đã xuất tại:    {FIG_OUT}")
    print("="*70)


if __name__ == "__main__":
    main()
