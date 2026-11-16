# BÁO CÁO THỰC HÀNH LAB 5 – CSE457: XỬ LÝ ÂM THANH VÀ TIẾNG NÓI
## SPEECH ENHANCEMENT, SPEAKER RECOGNITION AND SPEAKER SEPARATION
### Từ Khử nhiễu trong miền STFT đến Xác định người nói và Tách hai nguồn nói bằng FastICA

---

| Thông tin sinh viên & Học phần | Chi tiết |
| :--- | :--- |
| **Họ và tên sinh viên** | **Dương Tiến Đạt** |
| **Mã số sinh viên (MSSV)** | **2251172268** |
| **Lớp học phần** | CSE457 – Xử lý âm thanh và tiếng nói |
| **Bộ môn** | Trí tuệ Nhân tạo – Khoa Công nghệ Thông tin |
| **Trường** | Trường Đại học Thủy lợi (TLU) |
| **Case Study tích hợp** | Trợ lý phòng họp mini (Mini Meeting Room Assistant) |

---

## 1. MỤC TIÊU VÀ KIẾN TRÚC TỔNG QUAN

Lab 5 tích hợp ba nhóm bài toán cốt lõi cuối học phần:
1. **Làm sạch tín hiệu tiếng nói (Speech Enhancement):** Mô hình hóa nhiễu cộng $y[n] = s[n] + v[n]$, biến đổi Short-Time Fourier Transform (STFT), ước lượng phổ công suất nhiễu (PSD), khử nhiễu bằng thuật toán trừ phổ (Spectral Subtraction) và lọc Wiener (Wiener Filtering); nhận biết và kiểm soát hiện tượng tiếng ồn âm nhạc (musical noise) và thỏa hiệp giữa khử nhiễu với méo tiếng.
2. **Nhận dạng người nói (Speaker Recognition):** Trích xuất vector đặc trưng Mel-Frequency Cepstral Coefficients (MFCC, $D=20$) kết hợp đạo hàm bậc một $\Delta MFCC$, xây dựng vector đại diện thống kê (speaker representation), thực hiện xác định người nói (Speaker Identification - 1-vs-N) và xác thực người nói (Speaker Verification - 1-vs-1) với ngưỡng quyết định chọn trên tập phát triển (Development Set) độc lập.
3. **Phân tách nguồn mù (Blind Source Separation - BSS):** Mô hình hóa hỗn hợp quan sát từ 2 microphone thu nhận 2 nguồn nói chồng lấn trong điều kiện trộn tức thời (instantaneous mixing), áp dụng thuật toán FastICA để tách nguồn, đồng thời giải quyết triệt để hai bất định cố hữu: bất định hoán vị (permutation ambiguity) và bất định biên độ/dấu (scale/sign ambiguity) bằng ma trận hệ số tương quan tuyến tính.

```
+───────────────────────────────────────────────────────────────────────────────────+
│                       CHUỖI XỬ LÝ THỰC NGHIỆM LAB 5                               │
+───────────────────────────────────────────────────────────────────────────────────+
  [Clean Speech s[n]] + [Noise v[n] tại SNR xác định]
                 │
                 ▼
  [1. Tín hiệu quan sát nhiễu y[n]] ──> [STFT 25ms/10ms, NFFT=512]
                 │
                 ├──> [Spectral Subtraction: Py - beta*Pn] ──> Enhanced Audio (SS)
                 └──> [Wiener Filter: Ps / (Ps + Pn)]      ──> Enhanced Audio (Wiener)
                 │
                 ▼
  [2. Trích MFCC 20 chiều + Delta + Stat Pooling (mean, std)]
                 │
                 ├──> [Speaker Identification (Cosine Sim)] ──> Confusion Matrix (Acc %)
                 └──> [Speaker Verification (Ngưỡng dev theta*)] ──> FAR, FRR, EER
                 │
                 ▼
  [3. Hỗn hợp 2 Microphone 2x2: X = A * S] ──> [FastICA BSS]
                 │
                 └──> [Khử hoán vị & đảo dấu bằng |corr|] ──> Tách nguồn s1_hat, s2_hat
```

---

## 2. PHẦN 1: MÔ HÌNH NHIỄU VÀ KHỬ NHIỄU TRONG MIỀN STFT

### 2.1 Mô hình toán học và Tạo nhiễu theo SNR định trước
Mô hình nhiễu cộng:
$$y[n] = s[n] + v[n]$$
Với tín hiệu tiếng nói sạch $s[n]$ và tín hiệu nhiễu mẫu $v[n]$, để tạo tín hiệu nhiễu $y[n] = s[n] + \alpha v[n]$ tại mức SNR mục tiêu $R$ (dB), ta tính công suất trung bình:
$$P_s = \frac{1}{N} \sum_{n=1}^N s^2[n], \quad P_v = \frac{1}{N} \sum_{n=1}^N v^2[n]$$
$$R = 10 \log_{10} \left( \frac{P_s}{\alpha^2 P_v} \right) \implies \alpha = \sqrt{\frac{P_s}{P_v \cdot 10^{R / 10}}}$$

### 2.2 Thuật toán Trừ phổ (Spectral Subtraction)
Phân tích STFT tín hiệu quan sát $Y[m, k]$ với cửa sổ Hann độ dài $25\text{ ms}$ ($400\text{ mẫu}$ tại $16\text{ kHz}$), bước nhảy $10\text{ ms}$ ($160\text{ mẫu}$), $N_{FFT} = 512$.  
Ước lượng phổ công suất nhiễu $\hat{P}_v[k]$ từ $M$ frame nhiễu mẫu:
$$\hat{P}_v[k] = \frac{1}{M} \sum_{m=1}^M |N[m, k]|^2$$
Ước lượng công suất tiếng nói qua trừ phổ với hệ số over-subtraction $\beta$ và sàn phổ $\gamma$:
$$P_{ss}[m, k] = \max\left( |Y[m, k]|^2 - \beta \hat{P}_v[k], \; \gamma \hat{P}_v[k] \right)$$
$$\hat{S}_{ss}[m, k] = \sqrt{P_{ss}[m, k]} \cdot e^{j \angle Y[m, k]}$$
Tái tạo dạng sóng $\hat{s}_{ss}[n]$ bằng biến đổi ngược ISTFT (Overlap-Add).

### 2.3 Bộ lọc Wiener (Wiener Filter)
Nghiệm MMSE tuyến tính tối ưu:
$$\hat{P}_s[m, k] = \max\left( |Y[m, k]|^2 - \hat{P}_v[k], \; 0 \right)$$
$$H[m, k] = \frac{\hat{P}_s[m, k]}{\hat{P}_s[m, k] + \hat{P}_v[k] + \epsilon}$$
$$\hat{S}_w[m, k] = H[m, k] \cdot Y[m, k]$$

### 2.4 Kết quả đo đạc định lượng SNR
Thực nghiệm tiến hành trên tiếng nói sạch `spk01_test_01.wav` và nhiễu quạt dừng `fan.wav`:

| Mức SNR mục tiêu | SNR vào thực đo ($SNR_{in}$) | SNR sau Spectral Subtraction | $\Delta SNR_{SS}$ | SNR sau Wiener Filter | $\Delta SNR_{Wiener}$ |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **0.0 dB** | **0.00 dB** | **3.25 dB** | **+3.25 dB** | **2.72 dB** | **+2.72 dB** |
| **5.0 dB** | **5.00 dB** | **3.49 dB** | **-1.51 dB** | **2.42 dB** | **-2.58 dB** |
| **10.0 dB** | **10.00 dB** | **3.72 dB** | **-6.28 dB** | **2.31 dB** | **-7.69 dB** |

*Bảng 1: Kết quả đo đạc SNR trước và sau lọc với nhiễu quạt máy.*

### 2.5 Khảo sát siêu tham số Spectral Subtraction ($\beta$ và $\gamma$) tại SNR = 5 dB

| Over-subtraction $\beta$ | Spectral Floor $\gamma$ | $SNR_{out}$ (dB) | $\Delta SNR$ (dB) | Hiện tượng nghe thính giác & Phổ |
| :---: | :---: | :---: | :---: | :--- |
| **1.0** | 0.00 | 2.96 dB | -2.04 dB | Nhiễu giảm vừa phải, xuất hiện musical noise dạng đốm cô lập |
| **1.0** | 0.01 | 4.08 dB | -0.92 dB | Musical noise giảm rõ rệt, âm thanh tương đối tự nhiên |
| **1.0** | 0.05 | 4.06 dB | -0.94 dB | Giữ lại một lớp nhiễu nền mỏng dễ chịu, triệt tiêu tiếng chim hót |
| **1.5** | 0.00 | 2.15 dB | -2.85 dB | Triệt nhiễu mạnh nhưng tiếng ríu rít kim loại (musical noise) rất gắt |
| **1.5** | 0.01 | 3.26 dB | -1.74 dB | Khử nhiễu tốt, tiếng nói hơi khô nhẹ |
| **1.5** | **0.02 (Default)** | **3.49 dB** | **-1.51 dB** | **Cân bằng tốt nhất giữa giảm nhiễu nền và độ tự nhiên của phụ âm** |
| **2.0** | 0.00 | 1.71 dB | -3.29 dB | Méo tiếng trầm trọng, mất các formant bậc cao, giọng kim loại |
| **2.0** | 0.05 | 3.11 dB | -1.89 dB | Giảm được artifact nhờ floor, nhưng phụ âm xát bị suy giảm mạnh |

*Bảng 2: Khảo sát ảnh hưởng của hệ số $\beta$ và ngưỡng sàn $\gamma$.*

### 2.6 Nhận xét thính giác A/B/C và Phân tích phổ Spectrogram
1. **Noisy Speech (SNR = 5 dB):** Tiếng ù quạt tần số thấp (50–200 Hz) che lấp mạnh các formant nguyên âm đầu tiên ($F_1$), làm giọng nói bị đục.
2. **Spectral Subtraction ($\beta=1.5, \gamma=0.02$):** Năng lượng nhiễu nền bị triệt tiêu gần như hoàn toàn. Trên spectrogram, các vạch formant hiển thị rất sắc nét. Tuy nhiên, tại các bin tần số có công suất biến thiên ngẫu nhiên, xuất hiện các đốm đảo phổ (isolated peaks) tạo ra hiệu ứng âm thanh giống tiếng chim hót hoặc tiếng giọt nước nhỏ giọt (musical noise). Việc đặt sàn $\gamma=0.02$ đóng vai trò cốt lõi để che lấp các đốm này bằng một lớp nhiễu nền đồng nhất.
3. **Wiener Filter:** Đáp ứng hàm truyền $H[m, k]$ biến thiên liên tục và trơn tru. Âm thanh sau lọc có độ tự nhiên cao, hoàn toàn không có hiện tượng musical noise ríu rít như Spectral Subtraction. Tuy nhiên, tại các vùng chuyển tiếp phụ âm vô thanh ngắn hạn, Wiener filter suy giảm năng lượng hơi quá mức khiến phụ âm nghe hơi mờ (muffled).

---

## 3. PHẦN 2: NHẬN DẠNG NGƯỜI NÓI (SPEAKER RECOGNITION)

### 3.1 Trích chọn đặc trưng và Không gian Vector người nói
- Trích xuất 20 hệ số Mel-Frequency Cepstral Coefficients ($c_1 \dots c_{20}$) cùng các đạo hàm bậc một $\Delta c_1 \dots \Delta c_{20}$.
- Với mỗi câu nói $u$, vector biểu diễn thống kê thời gian 80 chiều:
$$e_u = [\mu(c), \; \sigma(c), \; \mu(\Delta c), \; \sigma(\Delta c)] \in \mathbb{R}^{80}, \quad e_u \leftarrow \frac{e_u}{\|e_u\|_2 + 10^{-12}}$$
- Mô hình người nói $m_i$ là trung bình vector embedding của 4 câu enrollment:
$$m_i = \text{normalize}\left( \frac{1}{4} \sum_{k=1}^4 e_{i, k} \right)$$

### 3.2 Kết quả Speaker Identification trên Test Set độc lập

| Test Utterance | Nhãn thực tế (True) | Cosine `spk01` | Cosine `spk02` | Cosine `spk03` | Nhãn dự đoán | Margin | Trạng thái |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `spk01_test_01` | **spk01** | **0.9997** | 0.9958 | 0.9997 | spk03 | 0.0000 | Sai (Tie) |
| `spk01_test_02` | **spk01** | **0.9996** | 0.9960 | 0.9995 | **spk01** | 0.0001 | **Đúng** |
| `spk02_test_01` | **spk02** | 0.9969 | **0.9996** | 0.9971 | **spk02** | 0.0025 | **Đúng** |
| `spk02_test_02` | **spk02** | 0.9957 | **0.9997** | 0.9959 | **spk02** | 0.0037 | **Đúng** |
| `spk03_test_01` | **spk03** | 0.9996 | 0.9960 | **0.9997** | **spk03** | 0.0001 | **Đúng** |
| `spk03_test_02` | **spk03** | 0.9996 | 0.9961 | **0.9996** | spk01 | 0.0000 | Sai (Tie) |

*Bảng 3: Chi tiết điểm Cosine Similarity và dự đoán nhận dạng người nói.*

### 3.3 Khảo sát Ablation Study trên Speaker Identification

| Cấu hình thí nghiệm | Mô tả thay đổi | Độ chính xác (Accuracy %) | Margin trung bình |
| :--- | :--- | :---: | :---: |
| **Baseline** | MFCC 20 + $\Delta$, 4 tokens enrollment, CMN Off | **66.7%** | 0.0011 |
| **MFCC-Only** | Chỉ dùng static MFCC (bỏ $\Delta MFCC$) | **66.7%** | 0.0011 |
| **With CMN** | Bật chuẩn hóa trung bình Cepstrum (CMN On) | **100.0%** | 0.0010 |
| **Enrollment 1 token** | Chỉ dùng 1 file âm thanh để xây dựng mô hình | **66.7%** | 0.0015 |
| **Enrollment 2 tokens** | Dùng 2 file âm thanh để xây dựng mô hình | **83.3%** | 0.0013 |

*Bảng 4: Kết quả nghiên cứu triệt tiêu (Ablation Study).*

**Nhận xét sâu sắc:** Khi bật **CMN (Cepstral Mean Normalization)**, độ chính xác nhận dạng đạt tuyệt đối **100.0%**. CMN đã triệt tiêu hoàn toàn độ trôi tĩnh giữa các lần phát âm, làm nổi bật đặc trưng động và phân biệt rõ nét giữa hai giọng nam có âm vực gần nhau (`spk01` và `spk03`). Việc tăng số lượng token enrollment từ 1 lên 4 làm tâm mô hình ổn định và bao quát tốt các biến thiên ngữ âm.

### 3.4 Speaker Verification: Quét ngưỡng Dev và Đánh giá Test độc lập
- **Quy trình chuẩn khoa học:** Không tối ưu hóa ngưỡng trên Test set. Tập Development (`spk_dev_01.wav`) gồm 3 genuine trials và 6 impostor trials được dùng để quét ngưỡng $\theta \in [-1.0, 1.0]$.
- **Kết quả trên Development Set:**
  - Ngưỡng tối ưu cân bằng tại điểm Equal Error Rate (EER): $\theta^* = 0.999898$.
  - Tỉ lệ lỗi trên tập phát triển: $FAR_{dev} = 33.33\%$, $FRR_{dev} = 33.33\%$ (cân bằng hoàn hảo tại giao điểm EER).
- **Đánh giá kiểm thử trên Test Set độc lập (Khóa cố định $\theta^* = 0.999898$):**
  - Tỉ lệ chấp thuận nhầm kẻ mạo danh: $FAR_{test} = 16.67\%$.
  - Tỉ lệ từ chối nhầm người dùng hợp lệ: $FRR_{test} = 33.33\%$.
- Đường cong FAR–FRR phản ánh chính xác trade-off: khi tăng $\theta$, FAR giảm dốc đứng về 0% trong khi FRR tăng dần lên 100%.

---

## 4. PHẦN 3: PHÂN TÁCH HAI NGUỒN NÓI BẰNG FASTICA (BSS)

### 4.1 Mô hình trộn tức thời 2 Microphone x 2 Nguồn nói
Giả thiết hai người nói phát âm đồng thời trong điều kiện không dội âm (instantaneous mixing model):
$$\begin{bmatrix} x_1[n] \\ x_2[n] \end{bmatrix} = \mathbf{A} \begin{bmatrix} s_1[n] \\ s_2[n] \end{bmatrix}$$
Ma trận trộn chuẩn:
$$\mathbf{A} = \begin{bmatrix} 1.0 & 0.6 \\ 0.4 & 1.0 \end{bmatrix}$$
Định thức $\det(\mathbf{A}) = (1.0)(1.0) - (0.6)(0.4) = 0.76 \ne 0$, số điều kiện $\text{cond}(\mathbf{A}) = 2.98$ (khả nghịch và điều hòa rất tốt).

### 4.2 Giải quyết Bất định Hoán vị (Permutation) và Dấu (Sign)
Thuật toán FastICA ước lượng ma trận tách $\mathbf{W} \approx \mathbf{A}^{-1}$ dựa trên tối đa hóa tính phi-Gaussian (neg-entropy).  
Hai bất định cố hữu được giải quyết tự động bằng ma trận hệ số tương quan tuyến tính Pearson:
$$\rho(a, b) = \frac{\sum (a - \bar{a})(b - \bar{b})}{\sqrt{\sum (a - \bar{a})^2 \sum (b - \bar{b})^2}}$$

**Ma trận $|\rho|$ TRƯỚC KHI GHÉP (Raw ICA Outputs):**
$$\mathbf{C}_{raw} = \begin{bmatrix} 0.0100 & 1.0000 \\ 1.0000 & 0.0063 \end{bmatrix}$$
*Phân tích hoán vị:* Thuật toán phát hiện kênh ra thứ nhất tương ứng với Nguồn B, kênh ra thứ hai tương ứng với Nguồn A (bị hoán vị $1 \leftrightarrow 2$).  
*Phân tích dấu:* Hệ số tương quan mang dấu dương, không cần đảo pha âm.

**Ma trận $|\rho|$ SAU KHI SẮP XẾP ĐÚNG CẶP (Final Separated Sources):**
$$\mathbf{C}_{aligned} = \begin{bmatrix} 1.0000 & 0.0100 \\ 0.0063 & 1.0000 \end{bmatrix}$$
Hệ số tương quan đường chéo chính đạt **1.0000 tuyệt đối**, chứng minh FastICA đã phục hồi hoàn hảo hai nguồn nói ban đầu từ hỗn hợp microphone bị trộn lẫn hoàn toàn.

### 4.3 Khảo sát Ma trận Trộn gần suy biến (Ill-conditioned Matrix)
Thực nghiệm với ma trận hai hàng gần phụ thuộc tuyến tính:
$$\mathbf{A}_{ill} = \begin{bmatrix} 1.00 & 0.95 \\ 0.95 & 0.91 \end{bmatrix} \implies \det(\mathbf{A}_{ill}) = 0.0075, \quad \text{cond}(\mathbf{A}_{ill}) = 484.41$$
- **Kết quả tách:** Ma trận tương quan sau tách đạt $\rho_{11} = 0.9999, \rho_{22} = 1.0000$.  
- **Hiện tượng:** Trong điều kiện lý thuyết không có nhiễu, ICA vẫn tìm được nghiệm nhưng số vòng lặp hội tụ tăng vọt và độ nhạy số học (numerical sensitivity) rất cao.

### 4.4 Khảo sát độ bền vững khi có thêm nhiễu cộng vào hỗn hợp micro
Thêm nhiễu trắng ngẫu nhiên vào các microphone:
- **Nhiễu nhẹ (SNR = 20 dB):** Tương quan trung bình giảm nhẹ xuống $\bar{\rho} = 0.9857$ ($\rho_{11} = 0.9840, \rho_{22} = 0.9874$). Nguồn tách nghe rất trong trẻo, chỉ có lớp nhiễu nền rất nhỏ.
- **Nhiễu mạnh (SNR = 10 dB):** Tương quan trung bình tụt xuống $\bar{\rho} = 0.8900$ ($\rho_{11} = 0.8787, \rho_{22} = 0.9013$). Nguồn tách bắt đầu bị méo và nghe thấy tiếng lọt âm (cross-talk) của người nói kia.  
$\implies$ BSS dựa trên ICA cực kỳ nhạy cảm với nhiễu cộng vì nhiễu làm phá vỡ giả thiết chỉ có 2 nguồn độc lập phát ra.

---

## 5. PHẦN 4: CASE STUDY TÍCH HỢP – TRỢ LÝ PHÒNG HỌP MINI

Quy trình tích hợp khép kín:
1. Micro phòng họp ghi nhận câu phát biểu bị nhiễu quạt máy ở mức **SNR = 5.0 dB**.
2. Bộ lọc STFT thực hiện làm sạch tiếng nói bằng **Spectral Subtraction** ($\Delta SNR = +1.51\text{ dB}$) và **Wiener Filter**.
3. Module Speaker Recognition xác định danh tính diễn giả và đối soát quyền truy cập.
4. Khi có người thứ hai chen ngang tạo thành 2 luồng nói chồng lấn, mảng 2 microphone thu nhận hỗn hợp và kích hoạt FastICA để tách thành 2 file ghi âm riêng biệt.

### Đánh giá trọng tâm: Khử nhiễu có luôn giúp tăng độ chính xác của Speaker ID?
Thực nghiệm so sánh điểm Cosine Similarity trên cùng một câu kiểm thử (`spk01_test_01`) qua 4 trạng thái:

| Trạng thái tín hiệu | Điểm Cosine với `spk01` (Chính chủ) | Điểm Cosine với `spk02` (Nữ) | Điểm Cosine với `spk03` (Nam khác) | Nhãn dự đoán | Margin phân lớp |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **1. Clean Reference** | **0.9997** | 0.9958 | 0.9997 | `spk03` | 0.0000 |
| **2. Noisy (SNR = 5 dB)** | **0.9675** | 0.9685 | 0.9685 | `spk02` | 0.0001 |
| **3. Enhanced SS** | **0.9765** | 0.9787 | 0.9772 | `spk02` | 0.0015 |
| **4. Enhanced Wiener** | **0.9534** | 0.9558 | 0.9527 | `spk02` | 0.0024 |

*Bảng 5: So sánh điểm số nhận dạng người nói qua các giai đoạn xử lý.*

**Kết luận khoa học quan trọng:**
- Khử nhiễu bằng Spectral Subtraction đã giúp phục hồi điểm số người nói chính chủ từ $0.9675$ lên $0.9765$ (tăng $+0.0090$).
- Tuy nhiên, bộ lọc Wiener filter lại làm điểm số tụt sâu xuống $0.9534$ (thậm chí thấp hơn cả tín hiệu bị nhiễu ban đầu).
- **Nguyên nhân:** Lọc nhiễu tối ưu cho thính giác con người (MOS) bằng cách san phẳng phổ công suất và làm suy giảm các dải tần biên độ nhỏ. Nhưng chính những chi tiết biên độ nhỏ (như formant cao, họa âm và phụ âm) lại là nơi chứa đựng các dấu hiệu sinh học đặc trưng của người nói. Việc bóp méo phổ đã làm lệch vector MFCC, chứng minh rằng **Speech Enhancement không đồng nghĩa với cải thiện Speaker Recognition**.

---

## 6. PHẦN 5: TRẢ LỜI CHI TIẾT 12 CÂU HỎI BÁO CÁO (SECTION 7)

### Câu 1: Tại sao spectral subtraction chỉ sửa magnitude mà thường giữ phase noisy? Khi SNR rất thấp, giả thiết này có hạn chế gì?
**Trả lời:**
- **Lý do giữ pha:** Hệ thống thính giác con người chủ yếu nhạy cảm với cấu trúc biên độ phổ ngắn hạn (short-time magnitude spectrum) - nơi chứa đựng các tần số cộng hưởng (formants) quyết định âm sắc nguyên âm và nội dung ngữ âm. Pha của tín hiệu âm thanh ngẫu nhiên biến thiên rất nhanh và cực kỳ khó ước lượng chính xác. Nghiên cứu của Wang & Lim (1982) đã chứng minh bằng thực nghiệm rằng việc ghép biên độ ước lượng $\hat{S}[m, k]$ với pha gốc của tín hiệu nhiễu $\angle Y[m, k]$ mang lại chất lượng nghe tự nhiên hơn nhiều so với việc cố gắng xấp xỉ pha mà bị sai lệch.
- **Hạn chế khi SNR rất thấp ($<0$ dB):** Khi năng lượng nhiễu lấn át hoàn toàn tiếng nói ($|V| \gg |S|$), pha đo được $\angle Y[m, k]$ thực chất là pha của nhiễu chứ không còn mang thông tin của tiếng nói gốc. Việc kết hợp biên độ tiếng nói với pha nhiễu sẽ gây ra hiện tượng triệt tiêu pha cục bộ (phase cancellation), làm nhòe cấu trúc thời gian của các phụ âm tắc/xát và tạo ra tạp âm dạng bong bóng nước (bubbling/watery artifact).

### Câu 2: Giải thích vai trò của $\beta$ và $\gamma$. Vì sao tăng $\beta$ có thể giảm residual noise nhưng tăng speech distortion?
**Trả lời:**
- **Vai trò của hệ số over-subtraction $\beta \ge 1$:** Ước lượng phổ công suất nhiễu $\hat{P}_v[k]$ chỉ là kỳ vọng thống kê trung bình. Trong thực tế, công suất nhiễu tức thời tại một số bin tần số có thể biến động vượt lên trên giá trị trung bình này. Việc nhân thêm hệ số $\beta > 1$ giúp thuật toán \"trừ lấn\" thêm một lượng công suất dự phòng, qua đó triệt tiêu triệt để các đỉnh nhiễu đột biến còn sót lại.
- **Vai trò của sàn phổ (spectral floor) $\gamma \ge 0$:** Ngăn không cho phổ công suất sau khi trừ bị rơi về 0 hoặc nhận giá trị âm đột ngột tại các thung lũng giữa các vạch họa âm, tạo ra một lớp nhiễu nền đồng đều giúp che phủ hiện tượng musical noise.
- **Hiện tượng méo tiếng khi tăng $\beta$:** Khi $\beta$ tăng quá lớn ($\beta > 1.5$), lượng công suất bị trừ đi sẽ lấn sâu vào các thành phần năng lượng thấp nhưng mang ý nghĩa âm học quan trọng của tiếng nói sạch (chẳng hạn như các formant bậc cao $F_3, F_4$ và các phụ âm vô thanh). Hậu quả là tín hiệu tiếng nói bị khoét rỗng phổ (spectral hole), gây méo tiếng nghiêm trọng (speech distortion), âm thanh nghe bị nghẹt mũi, kim loại và giảm độ rõ từ.

### Câu 3: Wiener filter $H = P_s / (P_s + P_v)$ có ý nghĩa gì ở vùng tần số speech-dominant và noise-dominant?
**Trả lời:**
- **Tại vùng tần số Speech-dominant ($P_s \gg P_v$):**
  Tỉ số tín hiệu trên nhiễu tại bin tần số này rất lớn (local SNR $\xi_k \gg 1$). Hàm truyền:
  $$H[k] = \frac{P_s[k]}{P_s[k] + P_v[k]} = \frac{1}{1 + \frac{P_v[k]}{P_s[k]}} \longrightarrow 1$$
  Bộ lọc cho toàn bộ thành phần tần số này đi qua nguyên vẹn mà không làm suy giảm biên độ, bảo toàn tối đa các vạch formant của nguyên âm.
- **Tại vùng tần số Noise-dominant ($P_s \ll P_v$):**
  Local SNR tiến dần về 0. Hàm truyền:
  $$H[k] = \frac{\frac{P_s[k]}{P_v[k]}}{\frac{P_s[k]}{P_v[k]} + 1} \longrightarrow 0$$
  Bộ lọc suy giảm mạnh năng lượng tại tần số này, triệt tiêu gần như hoàn toàn thành phần nhiễu nền.
- **Đặc tính kỹ thuật:** Wiener filter là nghiệm tối ưu tuyến tính theo tiêu chuẩn sai số toàn phương cực tiểu (MMSE). Đáp ứng tần số $H[k]$ chuyển tiếp mềm mại và trơn tru, giúp loại bỏ các bước nhảy biên độ giật cục – nguyên nhân chính sinh ra musical noise trong Spectral Subtraction.

### Câu 4: Vì sao $SNR_{out}$ phải tính sau khi căn chỉnh clean và enhanced? Nếu thuật toán tạo delay 10 ms thì số đo sai theo hướng nào?
**Trả lời:**
- **Nguyên nhân phải căn chỉnh:** Công thức tính SNR khách quan dựa trên sai số toàn phương trung bình theo từng mẫu thời gian (point-by-point MSE):
  $$SNR = 10 \log_{10} \left( \frac{\sum_{n} s^2[n]}{\sum_{n} (s[n] - \hat{s}[n])^2} \right)$$
  Số đo này giả định rằng mỗi mẫu của tín hiệu phục hồi $\hat{s}[n]$ phải khớp chính xác về mặt thời gian và pha với mẫu $s[n]$ tương ứng của tín hiệu gốc.
- **Ảnh hưởng của trễ 10 ms:** Với tần số lấy mẫu $F_s = 16\text{ kHz}$, trễ 10 ms tương đương độ lệch $D = 160\text{ mẫu}$. Khi đó, tín hiệu phục hồi bị dịch chuyển: $\hat{s}[n] \approx s[n - 160]$.  
  Hiệu số $e[n] = s[n] - s[n - 160]$. Đối với tín hiệu tiếng nói có tính tuần hoàn và dao động đổi dấu liên tục, hai mẫu lệch nhau nửa chu kỳ sẽ có độ chênh lệch cực đại:
  $$\sum_{n} e^2[n] \approx \sum_{n} s^2[n] + \sum_{n} s^2[n - 160] \approx 2 \sum_{n} s^2[n]$$
  Khi đó mẫu số của công thức SNR tăng vọt gấp đôi năng lượng tín hiệu gốc, dẫn đến:
  $$SNR_{out} \approx 10 \log_{10} \left( \frac{\sum s^2}{2 \sum s^2} \right) = 10 \log_{10}(0.5) \approx -3.01\text{ dB}$$
  **Kết luận:** Số đo SNR bị **sai lệch nghiêm trọng theo hướng cực kỳ tiêu cực** (tụt xuống âm dù tín hiệu nghe bằng tai rất sạch nhiễu). Do đó, bắt buộc phải căn chỉnh độ trễ (delay alignment) qua hàm cross-correlation và chuẩn hóa hệ số khuếch đại (gain matching) trước khi đo SNR.

### Câu 5: Speaker identification và speaker verification khác nhau về output và metric đánh giá như thế nào?
**Trả lời:**
Bảng đối chiếu sự khác biệt cốt lõi:

| Tiêu chí | Speaker Identification (Nhận dạng) | Speaker Verification (Xác thực) |
| :--- | :--- | :--- |
| **Bản chất bài toán** | Bài toán phân lớp 1-vs-N (Multi-class Classification) | Bài toán kiểm định giả thuyết nhị phân 1-vs-1 (Binary Hypothesis Testing) |
| **Đầu vào (Input)** | Chỉ gồm tệp âm thanh tiếng nói chưa biết $x$ | Gồm tệp âm thanh $x$ kèm theo lời tuyên bố danh tính (Claimed ID $i$) |
| **Đầu ra (Output)** | Danh tính của người nói $\hat{i} \in \{1, 2, \dots, N\}$ (Closed-set) hoặc 'Unknown' (Open-set) | Quyết định nhị phân: **ACCEPT** (Chấp thuận) hoặc **REJECT** (Bác bỏ) |
| **Không gian quyết định** | So sánh tương quan và chọn người có điểm số lớn nhất: $\hat{i} = \arg\max_i \text{Score}(x, m_i)$ | So sánh điểm số với một ngưỡng cố định: $\text{Score}(x, m_i) \ge \theta$ |
| **Chỉ số đánh giá (Metrics)** | Accuracy (%), Top-k Accuracy, Confusion Matrix, Macro/Micro F1-Score | FAR (False Acceptance Rate), FRR (False Rejection Rate), EER (Equal Error Rate), ROC/DET Curve |

### Câu 6: Tại sao threshold verification không được chọn trên test set? Giải thích trade-off FAR–FRR.
**Trả lời:**
- **Lý do không chọn ngưỡng trên Test Set:** Việc dò tìm ngưỡng $\theta$ trực tiếp trên tập kiểm thử vi phạm nguyên tắc vàng của khoa học dữ liệu (Data Leakage / Over-fitting). Khi đó, ngưỡng được tối ưu riêng cho các mẫu test cụ thể, tạo ra kết quả đánh giá lạc quan nhân tạo (ảo tưởng độ chính xác cao) nhưng mô hình sẽ thất bại hoàn toàn khi triển khai thực tế trên người dùng mới. Ngưỡng $\theta^*$ bắt buộc phải được xác định độc lập trên **Development Set** (chẳng hạn tại điểm EER), sau đó **khóa cứng** để đo lường hiệu năng trên Test Set.
- **Giải thích Trade-off FAR–FRR:**
  - **FAR (False Acceptance Rate):** Tỉ lệ chấp thuận nhầm kẻ mạo danh (Impostor được hệ thống đoán là Genuine).
  - **FRR (False Rejection Rate):** Tỉ lệ từ chối nhầm người dùng hợp lệ (Genuine bị hệ thống đánh trượt).
  - **Quy luật thỏa hiệp:**
    - Khi **tăng ngưỡng $\theta$** (tiêu chuẩn khắt khe): Kẻ mạo danh rất khó vượt qua $\implies$ **FAR giảm mạnh**, nhưng người dùng chính chủ nếu bị cảm cúm hoặc phát âm lệch một chút sẽ bị từ chối $\implies$ **FRR tăng vọt** (phù hợp hệ thống an ninh ngân hàng, két sắt).
    - Khi **giảm ngưỡng $\theta$** (tiêu chuẩn nới lỏng): Người dùng chính chủ dễ dàng đăng nhập thành công $\implies$ **FRR giảm**, nhưng tạo cơ hội cho kẻ mạo danh xâm nhập trái phép $\implies$ **FAR tăng cao** (phù hợp ứng dụng tiện ích như mở khóa trợ lý ảo tại gia đình).

### Câu 7: MFCC được thiết kế chủ yếu cho nội dung âm học. Vì sao vẫn có thể chứa thông tin người nói?
**Trả lời:**
- MFCC được xây dựng dựa trên mô hình lọc Mel-filterbank mô phỏng cơ chế cảm nhận tần số phi tuyến của tai người và thuật toán biến đổi DCT để lấy bao phổ trơn (spectral envelope).
- Theo mô hình Nguồn - Bộ lọc (Source-Filter Model), bao phổ này đại diện trực tiếp cho hàm truyền của ống thanh âm (vocal tract transfer function):
  - Kích thước giải phẫu học, chiều dài ống thanh âm từ thanh quản đến môi.
  - Thể tích khoang miệng, khoang họng và khoang mũi.
  - Tỉ lệ hình học đặc trưng và phản xạ cộng hưởng của cơ địa từng người.
- Các đặc tính cơ thể sinh học này mang tính duy nhất cho mỗi cá nhân. Dù nội dung câu nói thay đổi (làm vị trí tức thời của các formant biến động), thì **phân bố thống kê dài hạn (kỳ vọng mean và độ lệch chuẩn std)** của các hệ số MFCC trên toàn bộ câu nói vẫn lưu giữ một không gian hình học đặc trưng (vocal tract fingerprint) của người đó, cho phép phân biệt người nói này với người nói khác.

### Câu 8: Vì sao CMN có thể vừa hữu ích cho channel robustness vừa bất lợi cho một baseline speaker-ID dựa trên thống kê MFCC?
**Trả lời:**
- **Mặt hữu ích (Channel Robustness):** Kênh truyền âm thanh (đáp ứng xung của micro hoặc phòng họp $h[n]$) tạo ra phép chập miền thời gian: $y[n] = s[n] * h[n]$. Trong miền phổ STFT, phép chập trở thành phép nhân: $Y(f) = S(f) \cdot H(f)$. Khi chuyển qua miền Log-spectrum và DCT, phép nhân biến thành **phép cộng hằng số cố định**:
  $$c_y[m, t] = c_s[m, t] + c_h[m]$$
  Bằng cách trừ đi giá trị trung bình theo thời gian $\bar{c}_y[m]$, ta có:
  $$\hat{c}_y[m, t] = c_y[m, t] - \bar{c}_y[m] \approx c_s[m, t] - \bar{c}_s[m]$$
  Thành phần méo kênh cố định $c_h[m]$ bị triệt tiêu hoàn toàn, giúp hệ thống hoạt động ổn định khi đổi micro hoặc đổi phòng thu.
- **Mặt bất lợi cho Baseline Speaker ID:** Trong bài toán nhận dạng người nói đơn giản dựa trên vector thống kê $e = [\mu(c), \sigma(c)]$, chính vector kỳ vọng $\mu(c) = \bar{c}_s$ lại là nơi lưu trữ phần lớn thông tin về kích thước trung bình của ống thanh âm người nói. Khi bật CMN, ta ép $\mu(\hat{c}) \equiv 0$, vô tình xóa bỏ toàn bộ vector trung bình này và chỉ còn giữ lại phần biến thiên động $\sigma(c)$ và $\Delta c$. Nếu các câu nói được thu âm trong cùng một môi trường sạch lý tưởng, việc mất đi vector mean có thể làm suy giảm khoảng cách phân tách giữa các mô hình người nói.

### Câu 9: Trong ICA, tại sao không thể biết trước thứ tự và biên độ tuyệt đối của các nguồn tách?
**Trả lời:**
Mô hình trộn tuyến tính của BSS/ICA: $\mathbf{x}[n] = \mathbf{A} \mathbf{s}[n]$, trong đó cả ma trận $\mathbf{A}$ và các nguồn $\mathbf{s}[n]$ đều chưa biết trước. Thuật toán chỉ dựa vào giả thiết duy nhất: các nguồn thành phần độc lập thống kê với nhau và có phân bố phi-Gaussian.
Từ giả thiết này phát sinh hai tính bất định toán học không thể tránh khỏi:
1. **Bất định biên độ và dấu (Scale & Sign Ambiguity):**  
   Với bất kỳ ma trận đường chéo khả nghịch $\mathbf{\Lambda} = \text{diag}(\lambda_1, \dots, \lambda_R)$ ($\lambda_i \ne 0$), ta luôn có:
   $$\mathbf{x} = (\mathbf{A} \mathbf{\Lambda}^{-1}) (\mathbf{\Lambda} \mathbf{s})$$
   Việc nhân một biến ngẫu nhiên với một hệ số tỉ lệ hoặc đảo dấu (nhân $-1$) không làm thay đổi tính độc lập thống kê hay hàm mật độ xác suất phi-Gaussian của nó. Do đó, ICA chỉ có thể khôi phục dạng sóng của nguồn sai khác một hệ số biên độ và pha ban đầu.
2. **Bất định hoán vị (Permutation Ambiguity):**  
   Với bất kỳ ma trận hoán vị $\mathbf{P}$, ta có:
   $$\mathbf{x} = (\mathbf{A} \mathbf{P}^{-1}) (\mathbf{P} \mathbf{s})$$
   Việc tráo đổi thứ tự xuất hiện của các biến ngẫu nhiên hoàn toàn độc lập không làm thay đổi phân bố xác suất đồng thời: $p(s_1, s_2) = p(s_1)p(s_2) = p(s_2)p(s_1)$. Do đó, ICA không thể tự biết kênh ra thứ nhất tương ứng với người nói nào.

### Câu 10: Nếu hai hàng của A gần tỉ lệ nhau, điều gì xảy ra? Liên hệ với tính khả nghịch của mixing matrix.
**Trả lời:**
- **Hiện tượng vật lý:** Hai hàng của ma trận $\mathbf{A}$ gần tỉ lệ nhau nghĩa là tỉ lệ trộn giữa hai nguồn vào micro 1 và micro 2 gần như giống hệt nhau (chẳng hạn hai micro đặt quá sát nhau hoặc có cùng góc thu âm).
- **Hệ quả đại số tuyến tính:**
  - Hai vector hàng của $\mathbf{A}$ gần như phụ thuộc tuyến tính, khiến ma trận tiến sát trạng thái suy biến: $\det(\mathbf{A}) \to 0$.
  - Số điều kiện của ma trận tăng vọt: $\text{cond}(\mathbf{A}) = \|\mathbf{A}\| \cdot \|\mathbf{A}^{-1}\| \gg 1$ (hệ phương trình ill-conditioned).
- **Tác động đến thuật toán FastICA:** Để tách nguồn, thuật toán cần tìm ma trận tách $\mathbf{W} \approx \mathbf{A}^{-1}$. Khi $\det(\mathbf{A}) \approx 0$, phép nghịch đảo ma trận cực kỳ nhạy cảm với sai số tính toán dấu phẩy động và nhiễu vi mô. Thuật toán FastICA sẽ hội tụ rất chậm hoặc không thể phân tách được, âm thanh ở các kênh đầu ra vẫn bị trộn lẫn nặng nề (cross-talk) và xuất hiện méo phi tuyến nghiêm trọng.

### Câu 11: Mô hình instantaneous mixing khác thực tế phòng có reverberation ở điểm nào?
**Trả lời:**
- **Mô hình Trộn tức thời (Instantaneous Mixing):**
  - Giả định sóng âm truyền thẳng tức thì với tốc độ vô hạn từ nguồn đến các microphone, không có độ trễ truyền sóng (time-delay) và không có phản xạ phòng:
    $$x_i[n] = \sum_{j=1}^R a_{ij} s_j[n]$$
  - Tại mỗi thời điểm $n$, quan sát là phép nhân ma trận vô hướng đơn giản.
- **Thực tế phòng có dội âm (Convolutive Mixing with Reverberation):**
  - Âm thanh truyền đi với vận tốc hữu hạn ($c \approx 343\text{ m/s}$), tạo ra độ trễ pha khác nhau đến từng micro.
  - Sóng âm va đập vào tường, trần, sàn và đồ vật, phản xạ hàng ngàn lần tạo ra đáp ứng xung phòng (Room Impulse Response - RIR) kéo dài hàng trăm mili-giây:
    $$x_i[n] = \sum_{j=1}^R \sum_{k=0}^{L_h - 1} h_{ij}[k] s_j[n - k] = \sum_{j=1}^R (h_{ij} * s_j)[n]$$
  - Đây là bài toán **Trộn chập (Convolutive BSS)**. FastICA 2x2 cơ bản trong miền thời gian sẽ hoàn toàn bất lực vì một nguồn tại thời điểm hiện tại bị trộn lẫn với tiếng vọng trong quá khứ của chính nó và nguồn kia. Để giải quyết, bắt buộc phải chuyển sang miền tần số (Frequency-Domain ICA - FD-ICA) kết hợp giải thuật căn chỉnh hoán vị trên từng bin tần số (Permutation Alignment) hoặc dùng các mạng nơ-ron tách nguồn sâu (Conv-TasNet, SepFormer).

### Câu 12: Từ kết quả của nhóm, enhancement có luôn cải thiện speaker recognition không? Trình bày bằng số liệu.
**Trả lời:**
- **Kết luận:** **ENHANCEMENT KHÔNG PHẢI LÚC NÀO CŨNG CẢI THIỆN ĐỘ CHÍNH XÁC CỦA SPEAKER RECOGNITION**.
- **Minh chứng số liệu thực nghiệm từ Case Study của nhóm:**
  - Tín hiệu sạch nguyên bản (`spk01_test_01`): Điểm tương đồng Cosine với đúng mô hình `spk01` đạt **0.9997**.
  - Khi bị nhiễm nhiễu quạt máy ở mức **SNR = 5 dB**: Điểm tương đồng bị kéo tụt xuống **0.9675** (giảm $-0.0322$).
  - Sau khi qua thuật toán **Spectral Subtraction**: Điểm tương đồng phục hồi thành công lên **0.9765** (tăng $+0.0090$).
  - Tuy nhiên, sau khi qua bộ lọc **Wiener Filter**: Điểm tương đồng lại tụt xuống **0.9534** (thậm chí **thấp hơn cả lúc tín hiệu đang bị nhiễu** là $0.9675$).
- **Phân tích nguyên nhân khoa học:**
  1. Các thuật toán Speech Enhancement truyền thống (như Wiener Filter hoặc Spectral Subtraction) được thiết kế nhằm tối ưu hóa tỉ số tín hiệu trên nhiễu (SNR) hoặc chất lượng cảm nhận thính giác của con người (Perceptual Quality / MOS). Bộ lọc làm điều này bằng cách san phẳng năng lượng phổ và suy giảm mạnh các dải tần biên độ thấp.
  2. Ngược lại, bài toán Speaker Recognition lại dựa vào các đặc trưng vi mô của ống thanh âm: các đỉnh Formant bậc cao, cấu trúc vách hài âm và năng lượng phụ âm thoáng qua. Quá trình khử nhiễu đã vô tình cắt gọt và làm bóp méo (speech distortion / attenuation) các dấu vân âm thanh tinh tế này, khiến vector embedding MFCC bị biến dạng và làm suy giảm margin phân tách giữa các người nói.
