"""
lab5_speaker_id.py
==================
CSE457 Lab 5 – Nhận dạng người nói (Speaker Recognition):
- Trích xuất đặc trưng MFCC (D=20) và Delta MFCC
- Tạo vector embedding thống kê (mean + std)
- Tùy chọn chuẩn hóa kênh CMN (Cepstral Mean Normalization)
- Speaker Identification bằng Cosine Similarity và Confusion Matrix
- Speaker Verification: Genuine/Impostor trials, quét ngưỡng dev theta*, tính FAR, FRR, EER

Sinh viên: Dương Tiến Đạt – MSSV: 2251172268
"""

import numpy as np
import librosa
from numpy.linalg import norm
from sklearn.metrics import confusion_matrix

def extract_speaker_embedding(x, sr=16000, n_mfcc=20, with_delta=True, use_cmn=False):
    """
    Trích xuất vector đại diện người nói (speaker representation):
    - Trích MFCC (mặc định 20 chiều)
    - Nếu use_cmn: trừ trung bình theo thời gian c_t - c_bar
    - Nếu with_delta: tính Delta MFCC
    - Tính kỳ vọng mu và độ lệch chuẩn sigma theo trục thời gian
    - Chuẩn hóa L2 vector kết quả: e / (||e||_2 + 1e-12)
    """
    M = librosa.feature.mfcc(
        y=x,
        sr=sr,
        n_mfcc=n_mfcc,
        n_fft=512,
        win_length=400,
        hop_length=160,
        window='hann'
    )
    
    if use_cmn:
        # Trừ trung bình utterance để giảm ảnh hưởng kênh truyền
        M = M - np.mean(M, axis=1, keepdims=True)
        
    stats = [np.mean(M, axis=1), np.std(M, axis=1)]
    
    if with_delta:
        D = librosa.feature.delta(M)
        stats.extend([np.mean(D, axis=1), np.std(D, axis=1)])
        
    e = np.concatenate(stats)
    return e / (norm(e) + 1e-12)

def cosine_similarity(a, b):
    """
    Tính độ tương đồng Cosine giữa 2 vector:
    cos(a, b) = (a . b) / (||a|| * ||b|| + 1e-12)
    """
    return float(np.dot(a, b) / (norm(a) * norm(b) + 1e-12))

def build_speaker_models(enrollment_dict, sr=16000, n_mfcc=20, with_delta=True, use_cmn=False, num_tokens=None):
    """
    Xây dựng mô hình người nói m_i từ danh sách embedding enrollment:
    m_i = L2_normalize( mean(e_{i, 1..K}) )
    """
    models = {}
    for spk_name, audio_list in enrollment_dict.items():
        used_list = audio_list[:num_tokens] if num_tokens is not None else audio_list
        embs = [extract_speaker_embedding(x, sr=sr, n_mfcc=n_mfcc, with_delta=with_delta, use_cmn=use_cmn)
                for x in used_list]
        avg_emb = np.mean(embs, axis=0)
        models[spk_name] = avg_emb / (norm(avg_emb) + 1e-12)
    return models

def predict_speaker(test_audio, speaker_models, sr=16000, n_mfcc=20, with_delta=True, use_cmn=False):
    """
    Nhận dạng người nói (Identification):
    i_hat = argmax_i cos(e_test, m_i)
    """
    e_test = extract_speaker_embedding(test_audio, sr=sr, n_mfcc=n_mfcc, with_delta=with_delta, use_cmn=use_cmn)
    scores = {name: cosine_similarity(e_test, m) for name, m in speaker_models.items()}
    pred_spk = max(scores, key=scores.get)
    sorted_scores = sorted(scores.values(), reverse=True)
    margin = sorted_scores[0] - (sorted_scores[1] if len(sorted_scores) > 1 else 0.0)
    return pred_spk, scores, margin

def sweep_verification_thresholds(trials, num_thresholds=401):
    """
    Quét ngưỡng theta để xác định điểm hoạt động tối ưu gần EER trên Development set:
    trials: danh sách tuple (score, label), với label=1 (genuine), label=0 (impostor)
    Trả về danh sách (theta, far, frr) và ngưỡng tối ưu best_theta gần EER nhất (|FAR - FRR| nhỏ nhất).
    """
    genuine_scores = np.array([s for s, l in trials if l == 1])
    impostor_scores = np.array([s for s, l in trials if l == 0])
    all_scores = np.concatenate([genuine_scores, impostor_scores])
    
    # Kết hợp lưới thô [-1.0, 1.0] và lưới mịn xung quanh miền phân bố điểm thực tế
    grid_coarse = np.linspace(-1.0, 1.0, num_thresholds)
    grid_fine = np.linspace(np.min(all_scores) - 0.005, np.max(all_scores) + 0.005, 1000)
    thresholds = np.sort(np.unique(np.concatenate([grid_coarse, grid_fine])))
    
    rows = []
    for th in thresholds:
        frr = float(np.mean(genuine_scores < th))
        far = float(np.mean(impostor_scores >= th))
        rows.append((float(th), far, frr))
        
    # Tìm điểm gần Equal Error Rate (|FAR - FRR| nhỏ nhất)
    best_row = min(rows, key=lambda z: abs(z[1] - z[2]))
    return rows, best_row

def evaluate_verification_at_threshold(trials, theta):
    """
    Đánh giá FAR và FRR trên tập kiểm thử độc lập tại ngưỡng theta đã cố định.
    """
    genuine_scores = np.array([s for s, l in trials if l == 1])
    impostor_scores = np.array([s for s, l in trials if l == 0])
    
    frr = float(np.mean(genuine_scores < theta))
    far = float(np.mean(impostor_scores >= theta))
    return far, frr
