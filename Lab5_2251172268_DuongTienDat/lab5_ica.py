"""
lab5_ica.py
===========
CSE457 Lab 5 – Phân tách nguồn mù (Blind Source Separation - BSS) bằng FastICA:
- Mô hình trộn tức thời 2x2: x[n] = A * s[n]
- Kiểm tra điều kiện khả nghịch det(A) != 0
- Thuật toán FastICA (scikit-learn)
- Giải quyết 2 bất định cố hữu của ICA:
    (1) Permutation ambiguity: ghép output theo |correlation| cực đại
    (2) Sign ambiguity: đảo dấu nếu correlation âm
- Đánh giá định lượng qua ma trận tương quan |rho| và SNR phục hồi

Sinh viên: Dương Tiến Đạt – MSSV: 2251172268
"""

import numpy as np
from sklearn.decomposition import FastICA
from lab5_utils import rms_normalize, peak_normalize, snr_db

def create_instantaneous_mixture(s1, s2, A):
    """
    Tạo hỗn hợp quan sát từ 2 microphone:
    S = [rms_normalize(s1); rms_normalize(s2)]
    X = A @ S
    """
    L = min(len(s1), len(s2))
    s1_norm = rms_normalize(s1[:L])
    s2_norm = rms_normalize(s2[:L])
    S = np.vstack([s1_norm, s2_norm])
    
    A = np.asarray(A, dtype=np.float64)
    det_A = float(np.linalg.det(A))
    cond_A = float(np.linalg.cond(A))
    
    X = A @ S  # 2 x L
    return S, X, det_A, cond_A

def separate_fastica(X, random_state=0):
    """
    Áp dụng thuật toán FastICA để tìm ma trận tách W và nguồn xấp xỉ Shat.
    X có kích thước 2 x L -> chuyển vị sang L x 2 cho sklearn.
    """
    ica = FastICA(n_components=2, whiten='unit-variance', random_state=random_state, max_iter=1000)
    Shat = ica.fit_transform(X.T).T  # 2 x L
    return Shat, ica

def compute_correlation_matrix(S, Shat):
    """
    Tính ma trận hệ số tương quan Pearson giữa các nguồn gốc và nguồn ước lượng:
    rho_ij = corr(S[i], Shat[j])
    """
    C_raw = np.zeros((2, 2))
    for i in range(2):
        for j in range(2):
            corr = np.corrcoef(S[i], Shat[j])[0, 1]
            C_raw[i, j] = corr
    return C_raw

def resolve_ica_ambiguities(S, Shat):
    """
    Xử lý hoán vị (permutation) và dấu (sign) của ICA:
    1. Tính C_raw
    2. Xác định phép gán j -> i tối ưu sao cho tổng |corr| cực đại
    3. Nhân -1 nếu corr < 0
    4. Trả về S_aligned đã chuẩn hóa peak, cùng ma trận |rho| sau ghép
    """
    C_raw = compute_correlation_matrix(S, Shat)
    C_abs = np.abs(C_raw)
    
    # Xét 2 trường hợp hoán vị của bài toán 2 nguồn:
    # TH 1: Nguồn 0 gán với ước lượng 0, Nguồn 1 gán với ước lượng 1 (tổng = C_abs[0,0] + C_abs[1,1])
    # TH 2: Nguồn 0 gán với ước lượng 1, Nguồn 1 gán với ước lượng 0 (tổng = C_abs[0,1] + C_abs[1,0])
    score_identity = C_abs[0, 0] + C_abs[1, 1]
    score_permuted = C_abs[0, 1] + C_abs[1, 0]
    
    if score_identity >= score_permuted:
        perm = [0, 1]
    else:
        perm = [1, 0]
        
    S_aligned = np.zeros_like(Shat)
    aligned_signs = []
    
    for i in range(2):
        est_idx = perm[i]
        raw_corr = C_raw[i, est_idx]
        sign_factor = -1.0 if raw_corr < 0 else 1.0
        aligned_signs.append(sign_factor)
        
        # Sửa dấu và gán
        signal = sign_factor * Shat[est_idx]
        # Peak normalize để nghe rõ ràng không bị vỡ tiếng
        S_aligned[i] = peak_normalize(signal, 0.85)
        
    C_aligned = np.zeros((2, 2))
    for i in range(2):
        for j in range(2):
            C_aligned[i, j] = abs(np.corrcoef(S[i], S_aligned[j])[0, 1])
            
    return S_aligned, perm, aligned_signs, C_abs, C_aligned
