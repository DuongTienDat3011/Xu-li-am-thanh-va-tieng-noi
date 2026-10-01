import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import warnings

# Tắt cảnh báo để terminal gọn gàng
warnings.filterwarnings("ignore")

print("1. Đang tạo sóng âm thanh đầu vào...")
# Tần số lấy mẫu tiêu chuẩn cho giọng nói
sr = 16000 
t = np.linspace(0, 2, sr * 2) # Tạo âm thanh dài 2 giây

# Giả lập tín hiệu: Âm thanh thay đổi tần số từ thấp lên cao (giống như đang luyến giọng)
# kết hợp với một chút nhiễu để tự nhiên hơn.
y = 0.5 * np.sin(2 * np.pi * (300 + 200 * t) * t) + 0.05 * np.random.randn(len(t))

print("2. Đang đi qua Bộ lọc Mel (mô phỏng ốc tai người)...")
# Tính toán phổ Mel (Mel Spectrogram)
mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=40)
# Chuyển sang thang đo dB (Logarit) vì tai người cảm nhận âm lượng theo logarit
mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)

print("3. Đang trích xuất đặc trưng MFCC (Nén dữ liệu)...")
# Áp dụng biến đổi Cosine rời rạc (DCT) lên phổ Mel để lấy 13 hệ số MFCC
mfccs = librosa.feature.mfcc(S=mel_spectrogram_db, n_mfcc=13)

print("4. Đang vẽ hình ảnh trực quan...")
# ==========================================
# VẼ BIỂU ĐỒ (SẼ BẬT LÊN CỬA SỔ HÌNH ẢNH)
# ==========================================
plt.figure(figsize=(12, 10))
plt.rcParams['font.sans-serif'] = ['Arial']

# Biểu đồ 1: Sóng âm thanh nguyên bản (Waveform)
plt.subplot(3, 1, 1)
librosa.display.waveshow(y, sr=sr, color='royalblue', alpha=0.8)
plt.title("1. Sóng âm thanh thô (Waveform) - Đầu vào từ Microphone")
plt.ylabel("Biên độ")
plt.xlabel("Thời gian (s)")

# Biểu đồ 2: Phổ Mel (Mel Spectrogram)
plt.subplot(3, 1, 2)
img_mel = librosa.display.specshow(mel_spectrogram_db, sr=sr, x_axis='time', y_axis='mel', cmap='magma')
plt.title("2. Phổ Mel (Mel-Spectrogram) - Mô phỏng cách tai người cảm nhận tần số")
plt.ylabel("Dải tần Mel")
plt.colorbar(img_mel, format="%+2.0f dB")

# Biểu đồ 3: Đặc trưng MFCC
plt.subplot(3, 1, 3)
# MFCC thường có các giá trị âm dương đan xen, nên dùng thang màu 'coolwarm' sẽ dễ nhìn nhất
img_mfcc = librosa.display.specshow(mfccs, sr=sr, x_axis='time', cmap='coolwarm')
plt.title("3. Đặc trưng MFCC (13 chiều) - Đầu ra nén đưa vào HMM")
plt.ylabel("Hệ số MFCC (1-13)")
plt.xlabel("Thời gian (s)")
plt.colorbar(img_mfcc)

plt.tight_layout()
# Hiển thị cửa sổ hình ảnh
plt.show()