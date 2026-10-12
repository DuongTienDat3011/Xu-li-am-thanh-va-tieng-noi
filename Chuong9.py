"""
MÔ HÌNH HÓA ÂM THANH - CHƯƠNG 9 (ACOUSTIC MODELING TOOLKIT)
===================================================================================
Tệp mã nguồn tổng hợp đầy đủ các công thức toán học cốt lõi trong Chương 9 
(Spoken Language Processing) và tích hợp giao diện nhập liệu trực tiếp từ Terminal.

Hướng dẫn: Chạy script này bằng lệnh `python acoustic_modeling.py` trong VS Code.
"""

import numpy as np


# =====================================================================
# 1. BỘ CÔNG THỨC TOÁN HỌC CHƯƠNG 9 (ACOUSTIC MODELING FORMULAS)
# =====================================================================

def bayes_decoding_score(log_acoustic: float, log_language: float, 
                         lm_scale: float = 1.0, word_penalty: float = 0.0) -> float:
    """
    1. Quy tắc Bayes & Đánh giá Chuỗi từ (Bayes Decision Rule):
       Score(W) = log P(X|W) + LSF * log P(W) + WIP * |W|
    """
    return log_acoustic + lm_scale * log_language + word_penalty


def gaussian_multivariate_pdf(x: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> float:
    """
    2. Hàm mật độ xác suất Gaussian đa chiều (Multivariate Gaussian Density):
       N(x; mu, Sigma) = (1 / ((2*pi)^(D/2) * |Sigma|^(1/2))) * exp(-1/2 * (x - mu)^T * Sigma^-1 * (x - mu))
    """
    x = np.asarray(x, dtype=np.float64).flatten()
    mean = np.asarray(mean, dtype=np.float64).flatten()
    cov = np.asarray(cov, dtype=np.float64)

    D = len(x)
    if cov.shape != (D, D):
        raise ValueError(f"Lỗi: Kích thước ma trận hiệp phương sai {cov.shape} không khớp với vector {D} chiều.")

    det_cov = np.linalg.det(cov)
    if det_cov <= 0:
        raise ValueError("Lỗi: Ma trận hiệp phương sai phải xác định dương (det > 0).")

    inv_cov = np.linalg.inv(cov)
    diff = x - mean
    mahalanobis_sq = float(diff.T @ inv_cov @ diff)

    norm_const = 1.0 / (((2.0 * np.pi) ** (D / 2.0)) * np.sqrt(det_cov))
    pdf_val = norm_const * np.exp(-0.5 * mahalanobis_sq)

    return float(pdf_val)


def gmm_output_likelihood(x: np.ndarray, weights: np.ndarray, 
                          means: np.ndarray, covs: np.ndarray) -> float:
    """
    3. Phân bố xác suất đầu ra GMM tại trạng thái j (GMM State Likelihood):
       b_j(x) = sum_{k=1}^M c_{jk} * N(x; mu_{jk}, Sigma_{jk})
    """
    weights = np.asarray(weights, dtype=np.float64)
    means = np.asarray(means, dtype=np.float64)
    covs = np.asarray(covs, dtype=np.float64)

    M = len(weights)
    prob = 0.0

    for k in range(M):
        c_k = weights[k]
        mu_k = means[k]
        cov_k = covs[k]
        N_k = gaussian_multivariate_pdf(x, mu_k, cov_k)
        prob += c_k * N_k

    return prob


def mllr_adaptation(mean: np.ndarray, A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    4. Thích ứng mô hình MLLR (Maximum Likelihood Linear Regression):
       mu' = A * mu + b
    """
    mean = np.asarray(mean, dtype=np.float64)
    A = np.asarray(A, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    return A @ mean + b


def map_adaptation(prior_mean: np.ndarray, obs_data: np.ndarray, 
                   post_probs: np.ndarray, tau: float = 10.0) -> np.ndarray:
    """
    5. Thích ứng mô hình MAP (Maximum A Posteriori):
       hat_{mu}_{ik} = (tau * mu_{ik} + sum_t zeta_t(i,k) * x_t) / (tau + sum_t zeta_t(i,k))
    """
    prior_mean = np.asarray(prior_mean, dtype=np.float64)
    obs_data = np.asarray(obs_data, dtype=np.float64)
    post_probs = np.asarray(post_probs, dtype=np.float64)

    gamma = np.sum(post_probs)
    weighted_sum_x = np.sum(post_probs[:, np.newaxis] * obs_data, axis=0)

    return (tau * prior_mean + weighted_sum_x) / (tau + gamma)


def decision_tree_split_gain(cov_parent: np.ndarray, cov_left: np.ndarray, cov_right: np.ndarray,
                              N_parent: float, N_left: float, N_right: float) -> float:
    """
    6. Mức tăng Log-Likelihood phân tách cây quyết định Senone (Decision Tree Split Gain):
       Delta L = L(Parent) - [ L(Left) + L(Right) ]
       Trong đó L(S) = 0.5 * N_S * (D * (1 + ln(2*pi)) + ln|Sigma_S|)
    """
    D = cov_parent.shape[0]
    c_const = D * (1.0 + np.log(2 * np.pi))
    
    L_parent = 0.5 * N_parent * (c_const + np.log(np.linalg.det(cov_parent)))
    L_left = 0.5 * N_left * (c_const + np.log(np.linalg.det(cov_left)))
    L_right = 0.5 * N_right * (c_const + np.log(np.linalg.det(cov_right)))

    gain = L_parent - (L_left + L_right)
    return float(gain)


# =====================================================================
# 2. HÀM TRỢ GIÚP NHẬP DỮ LIỆU TỪ TERMINAL (INPUT PARSERS)
# =====================================================================

def input_vector(prompt_text: str) -> np.ndarray:
    """Nhập vector từ bàn phím (nhập các số phân cách bằng khoảng trắng hoặc dấu phẩy)"""
    while True:
        try:
            val_str = input(prompt_text).strip()
            if not val_str:
                print("⚠️  Không được để trống. Vui lòng nhập lại!")
                continue
            nums = [float(item) for item in val_str.replace(',', ' ').split()]
            return np.array(nums, dtype=np.float64)
        except ValueError:
            print("❌ Nhập sai định dạng số! Vui lòng nhập các số cách nhau bởi khoảng trắng (Ví dụ: 3.0 hoặc 10.0 5.0).")


def input_matrix(prompt_text: str, rows: int, cols: int) -> np.ndarray:
    """Nhập ma trận kích thước (rows x cols) từ bàn phím"""
    print(f"{prompt_text} (kích thước {rows}x{cols}):")
    matrix = []
    for i in range(rows):
        while True:
            try:
                row_str = input(f"  + Nhập hàng {i + 1}/{rows} ({cols} số): ").strip()
                nums = [float(item) for item in row_str.replace(',', ' ').split()]
                if len(nums) != cols:
                    print(f"❌ Cần nhập đúng {cols} số cho hàng này! Vui lòng nhập lại.")
                    continue
                matrix.append(nums)
                break
            except ValueError:
                print("❌ Nhập sai định dạng số! Vui lòng nhập lại.")
    return np.array(matrix, dtype=np.float64)


# =====================================================================
# 3. GIAO DIỆN TƯƠNG TÁC CONSOLE (INTERACTIVE MENU)
# =====================================================================

def main():
    while True:
        print("\n" + "=" * 65)
        print("    MÔ HÌNH HÓA ÂM THANH - CHƯƠNG 9 (ACOUSTIC MODELING TOOLKIT)")
        print("=" * 65)
        print(" 1. Tính điểm Quy tắc Bayes ASR [Score(W) = P(X|W) + LSF*P(W)]")
        print(" 2. Tính mật độ xác suất Gaussian đa chiều [N(x; mu, Sigma)]")
        print(" 3. Tính mật độ xác suất đầu ra GMM [b_j(x) = sum c_k * N_k]")
        print(" 4. Thích ứng mô hình MLLR [mu' = A*mu + b]")
        print(" 5. Thích ứng mô hình MAP [hat_mu = (tau*mu + sum x) / (tau + gamma)]")
        print(" 6. Tính mức tăng Log-Likelihood cây quyết định Senone [Delta L]")
        print(" 7. Chạy Minh họa Dữ liệu Mẫu (Demo tự động)")
        print(" 0. Thoát chương trình")
        print("-" * 65)

        choice = input("👉 Nhập lựa chọn của bạn (0-7): ").strip()

        if choice == '1':
            print("\n--- 1. TÍNH ĐIỂM QUY TẮC BAYES ASR ---")
            log_a = float(input("Nhập Log Acoustic Likelihood log P(X|W): "))
            log_l = float(input("Nhập Log Language Probability log P(W): "))
            lsf = float(input("Nhập Trọng số Mô hình Ngôn ngữ (LM Scale, mặc định 1.0): ") or "1.0")
            wip = float(input("Nhập Phạt độ dài từ (Word Penalty, mặc định 0.0): ") or "0.0")

            score = bayes_decoding_score(log_a, log_l, lsf, wip)
            print(f"\n✅ KẾT QUẢ: Tổng điểm Bayes Score(W) = {score:.6f}")

        elif choice == '2':
            print("\n--- 2. TÍNH GAUSSIAN ĐA CHIỀU ---")
            x = input_vector("Nhập vector quan sát x (ví dụ: 3.0 hoặc 10.0 5.0): ")
            D = len(x)
            mean = input_vector(f"Nhập vector trung bình mu ({D} số): ")
            cov = input_matrix("Nhập ma trận hiệp phương sai Sigma", D, D)

            try:
                pdf = gaussian_multivariate_pdf(x, mean, cov)
                print(f"\n✅ KẾT QUẢ: Mật độ xác suất N(x; mu, Sigma) = {pdf:.6e}")
            except Exception as e:
                print(f"❌ LỖI: {e}")

        elif choice == '3':
            print("\n--- 3. TÍNH MẬT ĐỘ XÁC SUẤT GMM b_j(x) ---")
            x = input_vector("Nhập vector quan sát x (ví dụ: 3.0 hoặc 10.0 5.0): ")
            D = len(x)
            M = int(input("Nhập số lượng thành phần Gaussian (M): "))
            weights = input_vector(f"Nhập {M} trọng số hỗn hợp c_k (tổng = 1.0): ")

            means = []
            covs = []
            for k in range(M):
                print(f"\n---> Cấu hình cho Thành phần Gaussian {k + 1}/{M}:")
                m_k = input_vector(f"Nhập vector trung bình mu_{k+1} ({D} số): ")
                c_k = input_matrix(f"Nhập ma trận hiệp phương sai Sigma_{k+1}", D, D)
                means.append(m_k)
                covs.append(c_k)

            try:
                prob = gmm_output_likelihood(x, weights, np.array(means), np.array(covs))
                print(f"\n✅ KẾT QUẢ: Xác suất đầu ra GMM b_j(x) = {prob:.6f}")
            except Exception as e:
                print(f"❌ LỖI: {e}")

        elif choice == '4':
            print("\n--- 4. THÍCH ỨNG MÔ HÌNH MLLR ---")
            mean = input_vector("Nhập vector trung bình ban đầu mu (ví dụ: 10.0 5.0): ")
            D = len(mean)
            A = input_matrix("Nhập ma trận biến đổi A", D, D)
            b = input_vector(f"Nhập vector độ lệch b ({D} số): ")

            adapted_mu = mllr_adaptation(mean, A, b)
            print(f"\n✅ KẾT QUẢ: Vector trung bình đã thích ứng MLLR mu' = {adapted_mu}")

        elif choice == '5':
            print("\n--- 5. THÍCH ỨNG MÔ HÌNH MAP ---")
            prior_mean = input_vector("Nhập vector trung bình gốc (Prior mu): ")
            D = len(prior_mean)
            N = int(input("Nhập số lượng quan sát dữ liệu mới (N): "))
            
            obs_list = []
            for t in range(N):
                obs = input_vector(f"Nhập vector quan sát thứ {t+1}/{N} ({D} số): ")
                obs_list.append(obs)
            
            post_probs = input_vector(f"Nhập {N} xác suất hậu phương zeta_t: ")
            tau = float(input("Nhập trọng số cân bằng tau (mặc định 10.0): ") or "10.0")

            adapted_map = map_adaptation(prior_mean, np.array(obs_list), post_probs, tau)
            print(f"\n✅ KẾT QUẢ: Vector trung bình đã thích ứng MAP hat_mu = {adapted_map.round(4)}")

        elif choice == '6':
            print("\n--- 6. MỨC TĂNG LOG-LIKELIHOOD CÂY QUYẾT ĐỊNH SENONE ---")
            D = int(input("Nhập số chiều đặc trưng D: "))
            N_p = float(input("Nhập số mẫu ở Node cha (Parent N): "))
            cov_p = input_matrix("Nhập ma trận hiệp phương sai Node cha", D, D)

            N_l = float(input("Nhập số mẫu ở Node con trái (Left N): "))
            cov_l = input_matrix("Nhập ma trận hiệp phương sai Node con trái", D, D)

            N_r = float(input("Nhập số mẫu ở Node con phải (Right N): "))
            cov_r = input_matrix("Nhập ma trận hiệp phương sai Node con phải", D, D)

            try:
                gain = decision_tree_split_gain(cov_p, cov_l, cov_r, N_p, N_l, N_r)
                print(f"\n✅ KẾT QUẢ: Mức tăng Log-Likelihood Delta L = {gain:.6f}")
            except Exception as e:
                print(f"❌ LỖI: {e}")

        elif choice == '7':
            print("\n--- 7. CHẠY MINH HỌA TỰ ĐỘNG DỮ LIỆU MẪU (DEMO) ---")
            # 1D GMM Demo
            x_1d = np.array([3.0])
            w_1d = np.array([0.6, 0.4])
            m_1d = np.array([[2.0], [5.0]])
            c_1d = np.array([[[1.0]], [[2.0]]])
            res_gmm = gmm_output_likelihood(x_1d, w_1d, m_1d, c_1d)
            print(f"1. GMM PDF (1D, x=3.0) -> b_j(3.0) = {res_gmm:.4f} (Đúng mẫu: 0.1867)")

            # MLLR Demo
            mu_2d = np.array([10.0, 5.0])
            A_2d = np.array([[1.05, 0.0], [0.0, 0.95]])
            b_2d = np.array([0.5, -0.2])
            res_mllr = mllr_adaptation(mu_2d, A_2d, b_2d)
            print(f"2. MLLR (2D) -> mu' = {res_mllr} (Đúng mẫu: [11.0, 4.55])")

        elif choice == '0':
            print("\n Cảm ơn bạn đã sử dụng Acoustic Modeling Toolkit! Tạm biệt.")
            break
        else:
            print("❌ Lựa chọn không hợp lệ. Vui lòng nhập từ 0 đến 7.")


if __name__ == "__main__":
    main()