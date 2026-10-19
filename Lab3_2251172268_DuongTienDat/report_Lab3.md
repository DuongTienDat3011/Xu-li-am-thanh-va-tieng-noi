# Báo cáo Lab 3 – CSE457: Xử lý âm thanh và tiếng nói
## Nhận dạng tiếng nói nâng cao với HMM

| Thông tin | Nội dung |
|-----------|----------|
| **Sinh viên** | Dương Tiến Đạt |
| **MSSV** | 2251172268 |
| **Học phần** | CSE457 – Xử lý âm thanh và tiếng nói |
| **Bài** | Lab 3 – Gaussian HMM, Baum-Welch, Viterbi, LM re-ranking |

---

## 1. Bài toán và dữ liệu

### 1.1 Vocabulary (5 lệnh tiếng Việt)

| ASCII | Tiếng Việt | Đặc điểm âm học |
|-------|-----------|-----------------|
| `mo`    | Mở    | Nasal /m/ + rounded vowel /ɔ/ |
| `dong`  | Đóng  | Stop /d/ + vowel /o/ + nasal /ŋ/ |
| `len`   | Lên   | Lateral /l/ + vowel /e/ + nasal /n/ |
| `xuong` | Xuống | Fricative /s/ + vowel /ɨ/ + nasal /ŋ/ |
| `dung`  | Dừng  | Fricative /z/ + vowel /ɨ/ + nasal /ŋ/ |

### 1.2 Dataset

| Chia | File/lệnh | Tổng | Ghi chú |
|------|----------|------|---------|
| Train | 6 | 30 | _01 – _06, Baum-Welch |
| Test | 2 | 10 | _07 – _08, đánh giá độc lập |

- Format: WAV 16-bit mono, Fs = 16,000 Hz
- Dữ liệu tổng hợp bằng source-filter model với biến thể tự nhiên (pitch ±15%, tốc độ ±20%)
- Không data leakage: tập train và test hoàn toàn tách biệt

---

## 2. Cấu hình tham số

| Tham số | Giá trị | Lý do |
|---------|---------|-------|
| Fs | 16,000 Hz | Chuẩn ASR |
| Frame | 25 ms = 400 mẫu | Giả thiết gần dừng |
| Hop | 10 ms = 160 mẫu | 100 frame/s |
| Window | Hamming | Giảm spectral leakage |
| Pre-emphasis α | 0.97 | y[n]=x[n]−0.97x[n−1] |
| NFFT | 512 | FFT mỗi frame |
| Mel filters | 26 | Tốt hơn 24 (Lab 2) |
| MFCC | 13 | Hệ số cepstral |
| **Total feat** | **39** | 13 MFCC + 13 Δ + 13 Δ² |
| CMVN | Utterance | Giảm channel mismatch |
| HMM states | 5 | Whole-word left-to-right |
| Covariance | Diagonal | Ít tham số, ổn định |
| Baum-Welch iter | 80 | Tối đa |
| Self-loop prob | 0.7 | a_{ii}=0.7, a_{i,i+1}=0.3 |

---

## 3. Kết quả từng phần

### Phần A – Trích đặc trưng 39D

**Tại sao cần 39 chiều:**
- 13 MFCC tĩnh: mô tả hình dạng phổ tại frame t
- 13 Δ-MFCC: xu hướng thay đổi → phân biệt phụ âm đầu/cuối
- 13 Δ²-MFCC: gia tốc → phát hiện burst/transient nhanh

**HMM chuẩn giả định output independence** (mỗi frame độc lập với nhau). Delta features đưa thông tin thời gian VÀO frame mà không vi phạm giả thiết này.

| Lệnh | T (frames) | D (features) |
|------|-----------|-------------|
| mo | 40 | 39 |
| dong | 44 | 39 |
| len | 45 | 39 |
| xuong | 44 | 39 |
| dung | 47 | 39 |

### Phần B – Khởi tạo HMM trái-sang-phải

**Left-to-right topology:**
- π = [1, 0, 0, 0, 0] — luôn bắt đầu ở state 0
- a_{ii} = 0.7 (self-loop), a_{i,i+1} = 0.3 (tiến)
- a_{4,4} = 1.0 (state cuối chỉ self-loop)
- Cấm chuyển ngược và bỏ qua state

**Lý do diagonal covariance:**
- Full covariance: D²/2 = 39²/2 = 760 tham số/state → 3,800/HMM
- Diagonal covariance: D = 39 tham số/state → 195/HMM
- Với chỉ 6 file train/lớp (~240 frame), diagonal tránh overfitting

### Phần C – Huấn luyện Baum-Welch

| Lệnh | Vòng | LL final | Hội tụ |
|------|------|---------|--------|
| mo | 16 | −3,032 | ✓ |
| dong | 19 | −11,213 | ✓ |
| len | 6 | −11,878 | ✓ |
| xuong | 28 | −8,173 | ✓ |
| dung | 21 | −10,177 | ✓ |

Tất cả 5 model hội tụ. Log-likelihood tăng đơn điệu theo vòng lặp EM.

### Phần D – Nhận dạng Maximum Likelihood + Viterbi

**Bảng acoustic scores (log P(X|λ_w)):**

| File | True | Pred | mo | dong | len | xuong | dung |
|------|------|------|-----|------|-----|-------|------|
| mo_07 | Mở | Mở✓ | **−485** | −2999 | −2603 | −3012 | −2581 |
| dong_07 | Đóng | Đóng✓ | −4828 | **−1848** | −3111 | −3327 | −2605 |
| len_07 | Lên | Lên✓ | −4685 | −3700 | **−1981** | −3270 | −2713 |
| xuong_07 | Xuống | Xuống✓ | −5272 | −3355 | −2935 | **−1518** | −2075 |
| dung_07 | Dừng | Dừng✓ | −5142 | −3643 | −2288 | −2404 | **−1679** |

Mỗi file được nhận dạng đúng. Model đúng có score cao nhất, margin lớn so với top-2.

**Viterbi path phân tích:**
- Path không giảm (left-to-right) ✓
- Self-loop thể hiện ở phần nguyên âm kéo dài (nhiều frame liên tiếp giữ state)
- Số frame/state phản ánh thời lượng phoneme

### Phần E – So sánh HMM vs DTW

| Chỉ số | HMM Gaussian | DTW Nearest-Template |
|--------|-------------|---------------------|
| **Accuracy** | **100.0%** | **100.0%** |
| Test time/trial | **11.4 ms** | 403.3 ms |
| Training | Baum-Welch EM | Lưu template |
| Score type | log P(X\|λ) (↑ tốt) | DTW cost (↓ tốt) |
| Train files/lớp | 6 | 6 |
| Mở rộng vocab | Subword+Lexicon+LM | Khó mở rộng |

**Nhận xét:**
- HMM nhanh hơn DTW **35×** ở test time (log-prob vs DP full matrix)
- Accuracy bằng nhau với dataset đơn giản này
- HMM ưu thế khi: nhiều dữ liệu, từ vựng lớn, cần kết hợp LM
- DTW ưu thế khi: ít dữ liệu, cần giải thích đơn giản

### Phần F – Lexicon mini

| Từ | Phoneme sequence | IPA | Phone HMM chain |
|----|-----------------|-----|----------------|
| mo | m – oo | /mɔː/ | λ_m → λ_oo |
| dong | d – oo – ng_f | /doŋ/ | λ_d → λ_oo → λ_ng_f |
| len | l – ee – n_f | /leːn/ | λ_l → λ_ee → λ_n_f |
| xuong | x – uu – ng_f | /suŋ/ | λ_x → λ_uu → λ_ng_f |
| dung | z – uu – ng_f | /zuŋ/ | λ_z → λ_uu → λ_ng_f |

**Khi nào cần lexicon:**
- Whole-word HMM: **không cần** (1 model = 1 từ)
- Phone/subword HMM: **bắt buộc** để nối phone HMM theo từ

### Phần G – Bigram LM + MAP Re-ranking

$$\text{Score}(w) = \frac{\log P_A(X|\lambda_w)}{T} + \lambda_{LM} \cdot \log P_L(w|w_{prev})$$

**Thí nghiệm re-ranking** (utterance "xuống", prev_word="mở"):

| λ_LM | Pred | Top-1 score | Top-2 score |
|------|------|------------|------------|
| 0.0 | Xuống | −36.137 | −49.409 |
| 0.1 | Xuống | −36.432 | −49.703 |
| 0.3 | Xuống | −37.021 | −50.292 |
| 1.0 | Xuống | −39.082 | −52.353 |

Với utterance có margin acoustic lớn, LM không thay đổi quyết định. LM sẽ có hiệu quả khi margin acoustic nhỏ (top-1 và top-2 gần nhau).

### Phần H – Đánh giá & Thí nghiệm

**Kết quả baseline:** Accuracy = **100.0%** (10/10 test file)

**Thí nghiệm E1 – n_states:**

| n_states | Accuracy | Ghi chú |
|---------|---------|---------|
| 3 | 100.0% | Underfit nhưng vẫn đủ cho 5 lệnh |
| 5 | 100.0% | Baseline tốt nhất |
| 7 | 100.0% | Cần nhiều data hơn để tránh degenerate |

**Thí nghiệm E5 – Noise mismatch:**

| SNR | Accuracy | Nhận xét |
|-----|---------|---------|
| 40 dB | 100.0% | Gần như sạch |
| 20 dB | 60.0% | Giảm đáng kể |
| 10 dB | 20.0% | Gần như random |

Noise là nguồn mismatch lớn nhất. Cần: noise-robust training, noise suppression, hoặc multi-condition training.

---

## 4. Câu hỏi báo cáo

### 1. Forward và Viterbi khác nhau ở phép toán nào?

| | Forward | Viterbi |
|--|---------|---------|
| Phép toán | **Cộng** tất cả đường | **Max** một đường tốt nhất |
| Câu trả lời | P(X\|λ) = Σ qua mọi chuỗi q | q* = argmax P(X,q\|λ) |
| Dùng cho | Training (Baum-Welch), likelihood | Alignment, nhận dạng |

$$\text{Forward: } \alpha_t(j) = b_j(x_t)\sum_i \alpha_{t-1}(i)a_{ij}$$
$$\text{Viterbi: } \delta_t(j) = b_j(x_t)\max_i [\delta_{t-1}(i)a_{ij}]$$

### 2. γₜ(i) và ξₜ(i,j) trong EM

$$\gamma_t(i) = P(q_t=i|O,\lambda) = \frac{\alpha_t(i)\beta_t(i)}{P(O|\lambda)}$$

**Ý nghĩa:** γₜ(i) = "trách nhiệm" của state i cho frame t. Nếu γₜ(i) = 0.8 → state i chịu 80% trách nhiệm giải thích frame t.

$$\xi_t(i,j) = \frac{\alpha_t(i)a_{ij}b_j(x_{t+1})\beta_{t+1}(j)}{P(O|\lambda)}$$

**Ý nghĩa:** ξₜ(i,j) = "trách nhiệm" của transition i→j tại thời điểm t.

### 3. Baum-Welch hội tụ xấu khi nào?

1. **Khởi tạo xấu:** μ và σ² khởi tạo ở vùng sai → EM mắc kẹt ở cực đại cục bộ tệ
2. **Quá ít dữ liệu:** Gaussian collapse → variance → 0, model suy biến
3. **n_states quá lớn:** Nhiều state hơn "phoneme" thực → state rỗng không có dữ liệu

### 4. Diagonal covariance khi dữ liệu ít

- Full covariance: cần D²/2 tham số/state → D=39: 760 params/state
- Diagonal: chỉ D = 39 params/state → đơn giản hơn ~20×
- Với 6 file train × ~40 frame = 240 frame/lớp ÷ 5 state ≈ 48 frame/state
- 48 frame cho 760 tham số → **underdetermined** → dễ singular
- Diagonal: 48 frame cho 39 tham số → ổn định

### 5. Delta features hữu ích dù HMM đã mô hình hóa chuỗi

HMM **giả định output independence**: P(x₁,...,xₜ|q₁,...,qₜ) = ∏ₜ P(xₜ|qₜ).

Điều này có nghĩa: frame t không biết frame t-1 trực tiếp. Delta đưa thông tin về **biến đổi thời gian cục bộ** vào từng vector frame mà không vi phạm giả thiết này.

### 6. Whole-word vs Subword HMM và lexicon

| Trường hợp | Cần lexicon? |
|-----------|-------------|
| Whole-word HMM (5 lệnh) | **Không** – mỗi model = 1 từ |
| Phone HMM (triphone) | **Bắt buộc** – map word → phone sequence |
| Mở rộng từ mới | **Bắt buộc** – lắp ghép phone HMM |

### 7. Self-loop giúp gì khi nói chậm?

Khi nói chậm, utterance dài hơn → nhiều frame hơn cần đi qua cùng số state. Self-loop a_{ii} cho phép state ở lại nhiều frame → HMM tự nhiên **stretch** theo thời gian. DTW cũng làm điều này bằng warping path ngang/dọc.

### 8. Chia log-likelihood cho T?

**Không cần** khi so sánh cùng một utterance (T như nhau cho mọi model). **Cần** khi:
- So sánh utterances có độ dài khác nhau
- Đặt reject threshold: nếu log P/T < θ → trả về "unknown"

### 9. Warping path DTW vs Viterbi state path

| | DTW warping path | Viterbi state path |
|--|-----------------|-------------------|
| Dạng | Chuỗi cặp (i,j): frame X ↔ frame Y | Chuỗi state q*: qₜ ∈ {0,...,S-1} |
| Mục đích | Căn chỉnh 2 chuỗi | Giải mã trạng thái ẩn |
| Cost | Σ d(xᵢ, yⱼ) trên path | max P(X,q\|λ) |
| Hướng | Monotone trên 2D grid | Monotone trên 1D time |

### 10. OOV → argmax vẫn chọn 1 lớp

argmax luôn chọn lớp có likelihood cao nhất trong vocabulary. Để nhận "unknown":

$$\text{Reject nếu: } \frac{\log P(X|\lambda_{\hat{w}})}{T} < \theta$$

Chọn θ từ phân bố score của đúng lớp (ví dụ: percentile 5th của correct class).

---

## 5. So sánh DTW và HMM

| Khía cạnh | DTW (Lab 2) | HMM (Lab 3) |
|-----------|------------|------------|
| Biểu diễn | Template chuỗi vector | Mô hình xác suất (A,B,π) |
| Biến thiên tốc độ | Warping path trực tiếp | Self-loop/transition ngầm |
| Huấn luyện | Lưu template | Baum-Welch EM |
| Score | Distance ↓ | Log-likelihood ↑ |
| Dữ liệu | Tốt với ít mẫu | Cần đủ dữ liệu |
| Mở rộng | Từ vựng nhỏ | Subword + Lexicon + LM |
| Tốc độ test | 403 ms/trial | 11 ms/trial |
| Alignment | Warping path | Viterbi state path |

---

## 6. Tài liệu tham khảo

[1] Huang, X., Acero, A., & Hon, H-W. *Spoken Language Processing*. Prentice-Hall, 2001.
- Ch.8: HMM (Forward, Viterbi, Baum-Welch)
- Ch.9: Acoustic Modeling (MFCC+Δ+Δ², Lexical Baseforms)
- Ch.11: Language Modeling (Bigram, Perplexity)

[2] Rabiner, L.R., & Schafer, R.W. *Theory and Applications of Digital Speech Processing*. Pearson, 2011.
- Ch.14 Mục 14.2–14.7: ASR, Acoustic+LM, Search/Decoding

[3] Jurafsky, D., & Martin, J.H. *Speech and Language Processing*. Prentice-Hall, 2008.
- Ch.3: N-gram LM | Ch.26: ASR Feature Extraction

[4] Đề cương chi tiết CSE457 – TLU, 2023.
