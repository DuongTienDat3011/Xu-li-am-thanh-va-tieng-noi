import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from hmmlearn import hmm
import warnings

# Tắt cảnh báo để terminal gọn gàng
warnings.filterwarnings("ignore")

# ==========================================
# 1. TẠO TÍN HIỆU ÂM THANH ĐẦU VÀO
# ==========================================
sr = 16000 
t = np.linspace(0, 1, sr)

# Tạo 3 đoạn tín hiệu âm thanh mang đặc tính khác biệt
y1 = 0.5 * np.sin(2 * np.pi * 300 * t)   # Trạng thái 1: Âm trầm (tần số 300Hz)
y2 = 0.5 * np.sin(2 * np.pi * 800 * t)   # Trạng thái 2: Âm bổng (tần số 800Hz)
y3 = 0.1 * np.random.randn(len(t))       # Trạng thái 3: Âm nhiễu vô thanh (Nhiễu trắng)

y = np.concatenate([y1, y2, y3])
duration = len(y) / sr

# ==========================================
# 2. TIỀN XỬ LÝ: TRÍCH XUẤT MFCC
# ==========================================
# Biến đổi sóng âm thô thành chuỗi quan sát O = (o1, o2, ..., oT)
mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
X = mfccs.T  # Đảo ma trận thành kích thước (Số khung thời gian, 13) để đưa vào HMM

# ==========================================
# 3. MÔ HÌNH MARKOV ẨN (HMM)
# ==========================================
# Khởi tạo GMM-HMM với 3 trạng thái ẩn
model = hmm.GaussianHMM(n_components=3, covariance_type="diag", n_iter=100, random_state=42)

# Huấn luyện mô hình (Thuật toán Baum-Welch / EM)
model.fit(X)

# Giải mã chuỗi trạng thái ẩn tối ưu nhất (Thuật toán Viterbi)
hidden_states = model.predict(X)

# ==========================================
# 4. TRỰC QUAN HÓA KẾT QUẢ
# ==========================================
plt.figure(figsize=(12, 10))
plt.rcParams['font.sans-serif'] = ['Arial']

# Biểu đồ 1: Tín hiệu đầu vào
plt.subplot(3, 1, 1)
librosa.display.waveshow(y, sr=sr, color='royalblue', alpha=0.8)
plt.title("Sóng âm thanh thô (3 giây chia làm 3 đoạn đặc tính khác nhau)")
plt.ylabel("Biên độ")

# Biểu đồ 2: Chuỗi quan sát (MFCC)
plt.subplot(3, 1, 2)
img = librosa.display.specshow(mfccs, x_axis='time', sr=sr, cmap='magma')
plt.title("Đặc trưng MFCC (Chuỗi quan sát đưa vào Markov Model)")
plt.ylabel("Hệ số MFCC")
plt.colorbar(img, format="%+2.0f")

# Biểu đồ 3: Quá trình giải mã của Markov (Viterbi Path)
plt.subplot(3, 1, 3)
time_axis = np.linspace(0, duration, len(hidden_states))
plt.plot(time_axis, hidden_states, label='Trạng thái HMM', color='red', drawstyle='steps-mid', linewidth=3)
plt.title("Kết quả phân giải trạng thái ẩn của Markov Model (Thuật toán Viterbi)")
plt.xlabel("Thời gian (giây)")
plt.ylabel("Trạng thái (0, 1, 2)")
plt.yticks([0, 1, 2])
plt.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.show()
