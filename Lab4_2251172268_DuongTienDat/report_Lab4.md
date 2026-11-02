# Báo cáo Thực hành Lab 4 – CSE457: Xử lý âm thanh và tiếng nói
## TỔNG HỢP TIẾNG NÓI (TEXT-TO-SPEECH)
### Từ Kiến trúc TTS, Ghép nối và Unit Selection đến Tham số HMM và TTS Học sâu

---

| Thông tin | Chi tiết |
| :--- | :--- |
| **Họ và tên sinh viên** | **Dương Tiến Đạt** |
| **Mã số sinh viên (MSSV)** | **2251172268** |
| **Lớp học phần** | CSE457 – Xử lý âm thanh và tiếng nói |
| **Bộ môn** | Trí tuệ nhân tạo – Khoa Công nghệ thông tin |
| **Trường** | Trường Đại học Thủy lợi |
| **Case Study** | Hệ thống thông báo xếp hàng miền hẹp (Queue Calling System) |
| **Yêu cầu giọng nói** | Chuyển đổi từ văn bản qua giọng nói **siêu tự nhiên, truyền cảm, mượt mà** |

---

## 1. Mục tiêu và Kiến trúc tổng quan hệ thống TTS

### 1.1 Mục tiêu thực hành
- Nối kết kiến thức xử lý tín hiệu số, trích chọn đặc trưng âm học ($F_0$, Formant, Spectrogram) và mô hình học máy với bài toán phát sinh dạng sóng tiếng nói (Waveform generation).
- Xây dựng và thực nghiệm trực tiếp 3 trường phái tổng hợp tiếng nói lớn:
  1. **Concatenative Synthesis & Unit Selection:** Ghép nối các đơn vị âm thanh tự nhiên có sẵn trong kho ngữ liệu.
  2. **HMM-based Parametric Synthesis:** Mô hình tham số thống kê chạy ở chế độ Generation kết hợp bộ lọc nguồn - lọc (Source-Filter Vocoder).
  3. **Neural TTS (Deep Learning):** Mô hình học sâu kết hợp Neural Vocoder mang lại chất lượng giọng nói siêu tự nhiên như người thật.
- Đánh giá chất lượng âm thanh toàn diện qua các chỉ số: **Độ rõ từ (Intelligibility %)**, **Độ tự nhiên (MOS Naturalness)**, **Độ mượt mà biên nối (MOS Continuity)**, **Độ lệch phổ biên (Boundary LSD)** và **Độ trễ tính toán (Latency / RTF)**.

### 1.2 Kiến trúc xử lý chuỗi (Pipeline) của hệ thống Text-to-Speech
Một hệ thống TTS hoàn chỉnh gồm 4 tầng xử lý nối tiếp:
```
[ Văn bản thô (Raw Text) ]
           │
           ▼
[ 1. Phân tích văn bản (Text Analysis & Normalization) ]
  • Lowercase, loại bỏ dấu câu thừa
  • Mở rộng số, viết tắt, ký hiệu thành từ đọc được (W = w1, w2, ..., wK)
           │
           ▼
[ 2. Phân tích ngữ âm & Từ điển (Phonetic Analysis / Lexicon) ]
  • Chuyển từ thành chuỗi đơn vị phát âm / âm vị (G2P: W -> P)
  • Xác định ranh giới từ và cấu trúc âm tiết
           │
           ▼
[ 3. Phân tích & Dự đoán ngữ điệu (Prosodic Analysis) ]
  • Dự đoán trường độ (Duration d), đường cong cao độ (F0 contour)
  • Dự đoán trọng âm và đường bao năng lượng (Energy A(t))
           │
           ▼
[ 4. Tầng tổng hợp âm thanh (Synthesis Backend / Vocoder) ]
  • Concatenation: Chọn candidate tối ưu qua Dynamic Programming + Cross-fade
  • Parametric HMM: Sinh quỹ đạo formant qua MLPG + Vocoder nguồn-lọc
  • Neural TTS: Sinh Mel-spectrogram -> Neural Vocoder (HiFi-GAN/WaveNet)
           │
           ▼
[ Dạng sóng âm thanh (Waveform Audio 16 kHz Mono) ]
```

---

## 2. Chuẩn bị dữ liệu và Text Normalization (Phần A)

### 2.1 Chuẩn hóa văn bản trong miền hẹp (Text Normalization)
Trong bài toán thông báo số thứ tự xếp hàng tại quầy giao dịch, văn bản có cấu trúc chuẩn hóa xác định. Hàm `normalize_queue_text()` xử lý:
- Chuyển toàn bộ ký tự về chữ thường, loại bỏ các ký tự dấu câu (`.`, `,`, `!`, `?`, `:`).
- Tách từng token từ chuỗi. Với các chuỗi số (digits như `"23"`, `"107"`, `"496"`), thực hiện phân rã và mở rộng thành chuỗi từ đếm tiếng Việt:
  - `"23"` $\rightarrow$ `['hai', 'ba']`
  - `"107"` $\rightarrow$ `['một', 'không', 'bảy']`
  - `"5"` $\rightarrow$ `['năm']`
- Câu kiểm thử chính: `"Mời khách hàng số 23 đến quầy 5."`  
  $\rightarrow$ `['mời', 'khách', 'hàng', 'số', 'hai', 'ba', 'đến', 'quầy', 'năm']`

### 2.2 Thiết kế và xây dựng Unit Inventory
Kho dữ liệu âm thanh (Inventory) gồm 16 từ vựng cơ sở:
`mời`, `khách`, `hàng`, `số`, `không`, `một`, `hai`, `ba`, `bốn`, `năm`, `sáu`, `bảy`, `tám`, `chín`, `đến`, `quầy`.

Để phục vụ giải thuật **Lựa chọn đơn vị (Unit Selection)**, mỗi từ vựng được xây dựng với **2 candidates** (`_01` và `_02`):
- **Candidate 1 (`_01.wav`):** Giọng nữ chuẩn (`vi-VN-HoaiMyNeural`), phát âm tốc độ trung bình, cao độ ổn định.
- **Candidate 2 (`_02.wav`):** Giọng biến thiên (`vi-VN-NamMinhNeural` hoặc ngữ điệu luyến thanh điệu nhẹ), tốc độ nhanh hơn, tạo sự đa dạng về đặc trưng âm học ở ranh giới.
- **Tiền xử lý âm thanh:**
  - Định dạng chuẩn: WAV mono, $F_s = 16,000$ Hz, 16-bit PCM.
  - Cắt lọc khoảng lặng (Silence trimming) với ngưỡng năng lượng $-25$ dB.
  - Chuẩn hóa năng lượng biên độ theo RMS mục tiêu: $target\_rms = 0.08$.

### 2.3 Phân tích thống kê Inventory (Metadata & Histogram)
Dữ liệu 32 đơn vị âm thanh được lưu trữ chi tiết trong `metadata.csv`. Thống kê đặc trưng:
- **Thời lượng (Duration):** Dao động từ $0.192$ s (từ ngắn như `khách`) đến $0.416$ s (từ dài có thanh trắc/ngang như `bảy`, `mời`), trung bình đạt $0.345$ s.
- **Năng lượng (RMS):** Toàn bộ 32 units đều đạt chính xác $RMS = 0.0800$ sau chuẩn hóa, triệt tiêu hoàn toàn sự chênh lệch âm lượng ngẫu nhiên giữa các file thu âm.

> Biểu đồ phân bố được trực quan hóa tại file [`figures/fig1_inventory_stats.png`](file:///d:/Xu-li-am-thanh-va-tieng-noi/Lab4_2251172268_DuongTienDat/figures/fig1_inventory_stats.png).

---

## 3. Ghép nối Baseline và Lựa chọn đơn vị tối ưu (Phần B & C)

### 3.1 Baseline Concatenation (Ghép thẳng không cross-fade)
- **Phương pháp:** Với mỗi từ trong câu, hệ thống chọn cố định Candidate 1 (`_01.wav`), sau đó nối tuần tự các mảng mẫu lại với nhau ($xfade = 0$ ms).
- **Kết quả:** File âm thanh [`audio_output/raw_concat.wav`](file:///d:/Xu-li-am-thanh-va-tieng-noi/Lab4_2251172268_DuongTienDat/audio_output/raw_concat.wav) (thời lượng $2.91$ s).
- **Phân tích lỗi âm học:**
  - Xuất hiện tiếng **click / pop** rất rõ tại các ranh giới nối từ (đặc biệt giữa `mời` $\rightarrow$ `khách` và `hàng` $\rightarrow$ `số`).
  - Nguyên nhân: Sự lệch pha đột ngột giữa mẫu cuối của từ trước và mẫu đầu của từ sau, gây ra bước nhảy biên độ gián đoạn (Discontinuity step) dẫn đến thành phần tần số cao ngoài ý muốn trên phổ.
  - Vị trí ranh giới được đánh dấu chi tiết trên dạng sóng và phổ tại file [`figures/fig2_baseline_boundary.png`](file:///d:/Xu-li-am-thanh-va-tieng-noi/Lab4_2251172268_DuongTienDat/figures/fig2_baseline_boundary.png).

### 3.2 Hàm chi phí trong Unit Selection
Bài toán chọn chuỗi candidate tối ưu $\theta^* = (\theta_1^*, \theta_2^*, \dots, \theta_K^*)$ được mô hình hóa bằng bài toán cực tiểu hóa tổng chi phí:
$$\theta^* = \arg\min_{\theta_1 \dots \theta_K} \left[ \sum_{j=1}^K C_{target}(\theta_j, t_j) + \lambda \sum_{j=1}^{K-1} C_{join}(\theta_j, \theta_{j+1}) \right]$$

1. **Target Cost ($C_{target}$):**
   $$C_{target}(\theta_j) = w_{dur} \left( \frac{dur(\theta_j) - 0.35}{0.10} \right)^2 + w_{rms} \left( \frac{rms(\theta_j) - 0.08}{0.02} \right)^2$$
   Ưu tiên các candidate có thời lượng tự nhiên ổn định và năng lượng bám sát $0.08$.

2. **Join Cost ($C_{join}$):**
   Đo mức độ ăn khớp âm học giữa $25$ ms cuối của unit trước ($\theta_j$) và $25$ ms đầu của unit sau ($\theta_{j+1}$):
   $$D_{spec} = \frac{1}{M} \sum_{k=1}^M \left[ 20 \log_{10} |A[k]| - 20 \log_{10} |B[k]| \right]^2$$
   $$C_{join} = D_{spec} + 100.0 \cdot |RMS_{left} - RMS_{right}|$$

### 3.3 Thuật toán Quy hoạch động Viterbi trên lưới Candidate
- Lưới gồm $K = 9$ tầng (stages tương ứng 9 từ), mỗi tầng có 2 trạng thái candidate.
- Bảng quy hoạch động: $V[j, c] = C_{target}(j, c) + \min_{p \in \{1, 2\}} [V[j-1, p] + \lambda C_{join}(p, c)]$.
- Truy vết ngược (Backtracking) tìm ra chuỗi candidate có tổng chi phí nhỏ nhất.
- **Kết quả so sánh chi phí:**
  - Baseline cố định Candidate 1: Tổng chi phí = **$6,291.90$**
  - Chuỗi tối ưu qua Viterbi: Tổng chi phí = **$4,451.75$** (**Giảm 29.25% chi phí!**)
  - Chuỗi candidate được chọn: `[c2, c1, c2, c1, c2, c2, c2, c1, c2]`
  - Biểu đồ đường đi tối ưu trên mạng lưới được vẽ tại [`figures/fig4_unit_selection_lattice.png`](file:///d:/Xu-li-am-thanh-va-tieng-noi/Lab4_2251172268_DuongTienDat/figures/fig4_unit_selection_lattice.png).

### 3.4 Khử click bằng Overlap-Add Cross-fade
Áp dụng cửa sổ trọng số tuyến tính $\alpha[n] = \frac{n}{N}$ trên vùng $N$ mẫu ($0 \le n < N$):
$$y[n] = (1 - \alpha[n]) x_1[n] + \alpha[n] x_2[n]$$

Thực nghiệm so sánh 4 mức cross-fade: $0$ ms, $5$ ms, $15$ ms và $30$ ms:
- **$0$ ms:** Bất liên tục biên lớn ($LSD = 48.5$ dB), click thính giác rất rõ.
- **$5$ ms:** Giảm bớt click nhưng vẫn còn cảm giác khấc nhẹ giữa nguyên âm.
- **$15$ ms:** **Tối ưu nhất** ($LSD = 12.3$ dB). Click biến mất hoàn toàn, dạng sóng chuyển tiếp mượt mà, không làm mờ phụ âm đầu.
- **$30$ ms:** Quá dài, gây hiện tượng chồng phổ (spectral smearing) và làm nhòe phụ âm tắc vô thanh.
- File âm thanh tối ưu: [`audio_output/best_concat.wav`](file:///d:/Xu-li-am-thanh-va-tieng-noi/Lab4_2251172268_DuongTienDat/audio_output/best_concat.wav).
- Biểu đồ dạng sóng so sánh 4 mức cross-fade tại [`figures/fig3_crossfade_comparison.png`](file:///d:/Xu-li-am-thanh-va-tieng-noi/Lab4_2251172268_DuongTienDat/figures/fig3_crossfade_comparison.png).

---

## 4. Ngữ điệu (Prosody) và Mô hình Nguồn - Lọc (Phần D)

### 4.1 Phân tích cao độ F0 trên đoạn hữu thanh
- Trên đoạn nguyên âm hữu thanh của từ `mời` ($50$ ms = 800 mẫu), tính hàm tự tương quan chuẩn hóa:
  $$R_{xx}[k] = \frac{\sum_{n} x[n] x[n+k]}{\sqrt{\sum_n x^2[n] \sum_n x^2[n+k]}}$$
- Vị trí đỉnh cực đại thứ nhất tìm được tại độ trễ $N_p = 78$ mẫu.
- **Kiểm chứng công thức tần số cơ bản:**
  $$F_0 = \frac{F_s}{N_p} = \frac{16,000}{78} \approx 205.13 \text{ Hz}$$
- Theo dõi toàn câu (Pitch Tracking bằng thuật toán pYIN): $F_0$ biến thiên trong khoảng $115.7$ – $338.8$ Hz, giá trị trung bình toàn câu đạt $168.56$ Hz, phù hợp hoàn hảo với cao độ thanh âm tiếng Việt.
- Biểu đồ F0 và Autocorrelation tại [`figures/fig5_prosody_pitch_f0.png`](file:///d:/Xu-li-am-thanh-va-tieng-noi/Lab4_2251172268_DuongTienDat/figures/fig5_prosody_pitch_f0.png).

### 4.2 Thí nghiệm Time-Stretch và Pitch-Shift
Áp dụng lý thuyết tách biệt Nguồn - Lọc (Source-Filter Model):
$$X(z) = E(z) \cdot H(z)$$

1. **Biến đổi tốc độ (Time-Stretch):**
   - Tốc độ $0.8\times$ (chậm hơn): Thời lượng tăng lên $3.49$ s ([`audio_output/speed_0.8x.wav`](file:///d:/Xu-li-am-thanh-va-tieng-noi/Lab4_2251172268_DuongTienDat/audio_output/speed_0.8x.wav)).
   - Tốc độ $1.2\times$ (nhanh hơn): Thời lượng rút ngắn còn $2.33$ s ([`audio_output/speed_1.2x.wav`](file:///d:/Xu-li-am-thanh-va-tieng-noi/Lab4_2251172268_DuongTienDat/audio_output/speed_1.2x.wav)).
   - Nhờ Phase Vocoder, cao độ $F_0$ và bao phổ Formant được giữ nguyên vẹn.
2. **Biến đổi cao độ (Pitch-Shift $\pm 2$ Semitones):**
   - Tỷ số tần số: $r = 2^{\Delta s / 12}$.
   - Tăng cao độ $+2$ semitones: $r = 2^{2/12} \approx 1.1225$, giọng nghe thanh thoát, trẻ trung hơn ([`audio_output/pitch_plus2.wav`](file:///d:/Xu-li-am-thanh-va-tieng-noi/Lab4_2251172268_DuongTienDat/audio_output/pitch_plus2.wav)).
   - Giảm cao độ $-2$ semitones: $r = 2^{-2/12} \approx 0.8909$, giọng trầm ấm hơn ([`audio_output/pitch_minus2.wav`](file:///d:/Xu-li-am-thanh-va-tieng-noi/Lab4_2251172268_DuongTienDat/audio_output/pitch_minus2.wav)).
   - Thời lượng của file giữ nguyên chính xác $2.79$ s.
- Biểu đồ so sánh 5 phổ Spectrogram tại [`figures/fig6_prosody_speed_pitch_spectrogram.png`](file:///d:/Xu-li-am-thanh-va-tieng-noi/Lab4_2251172268_DuongTienDat/figures/fig6_prosody_speed_pitch_spectrogram.png).

---

## 5. Tổng hợp tham số dựa trên HMM (Phần E)

### 5.1 Nguyên lý HMM Generation và MLPG
Trong hướng tiếp cận HMM-based synthesis (HTS):
- HMM không dùng để giải mã nhận dạng mà dùng để **sinh chuỗi quan sát tham số âm học**:
  $$\hat{O}^* = \arg\max_O P(O | \lambda, Q)$$
- Mỗi âm vị/từ được biểu diễn bởi chuỗi 3 trạng thái HMM trái-sang-phải (Left-to-Right: Onset $\rightarrow$ Nucleus $\rightarrow$ Coda).
- Vector quan sát bao gồm đặc trưng tĩnh và động: $O_t = [c_t, \Delta c_t, \Delta^2 c_t]^T$. Ràng buộc đạo hàm bậc 1 ($\Delta$) và bậc 2 ($\Delta^2$) triệt tiêu hiện tượng bậc thang, tạo nên các quỹ đạo tham số (F0, Formants $F_1, F_2, F_3$) trơn tru liên tục qua thời gian.

### 5.2 Tái tạo Waveform bằng Source-Filter Vocoder
- Nguồn kích thích $e[n]$:
  - Khung hữu thanh (Voiced): Chuỗi xung tuần hoàn với chu kỳ $T(t) = F_s / F_0(t)$.
  - Khung vô thanh (Unvoiced): Nhiễu trắng Gaussian $\mathcal{N}(0, \sigma^2)$.
- Bộ lọc ống thanh âm $H(z)$: Tổ hợp 3 bộ lọc Formant cộng hưởng (2-pole resonator biquads) theo các quỹ đạo $F_1, F_2, F_3$.
- File âm thanh: [`audio_output/hmm_tts.wav`](file:///d:/Xu-li-am-thanh-va-tieng-noi/Lab4_2251172268_DuongTienDat/audio_output/hmm_tts.wav) ($2.94$ s, 294 frames).
- Thực nghiệm biến điệu tham số độc lập:
  - Duration scale $0.8\times$ và $1.2\times$ ([`audio_output/hmm_duration_0.8x.wav`](file:///d:/Xu-li-am-thanh-va-tieng-noi/Lab4_2251172268_DuongTienDat/audio_output/hmm_duration_0.8x.wav), [`audio_output/hmm_duration_1.2x.wav`](file:///d:/Xu-li-am-thanh-va-tieng-noi/Lab4_2251172268_DuongTienDat/audio_output/hmm_duration_1.2x.wav)).
  - Pitch scale $0.9\times$ và $1.1\times$ ([`audio_output/hmm_pitch_0.9x.wav`](file:///d:/Xu-li-am-thanh-va-tieng-noi/Lab4_2251172268_DuongTienDat/audio_output/hmm_pitch_0.9x.wav), [`audio_output/hmm_pitch_1.1x.wav`](file:///d:/Xu-li-am-thanh-va-tieng-noi/Lab4_2251172268_DuongTienDat/audio_output/hmm_pitch_1.1x.wav)).
- Biểu đồ 4 tầng tham số HMM (State, F0, Formants, Energy) tại [`figures/fig7_hmm_trajectories.png`](file:///d:/Xu-li-am-thanh-va-tieng-noi/Lab4_2251172268_DuongTienDat/figures/fig7_hmm_trajectories.png).

---

## 6. Tổng hợp tiếng nói học sâu siêu tự nhiên (Phần F)

### 6.1 Mô hình Neural TTS cho tiếng Việt
- Áp dụng mô hình Neural TTS tiên tiến nhất hiện nay (kiến trúc Transformer / FastSpeech2 kết hợp HiFi-GAN Vocoder).
- Giọng đọc mẫu phát âm tiếng Việt chuẩn: truyền cảm, ngân vang tự nhiên, chuyển thanh điệu (huyền, sắc, hỏi, ngã, nặng) mượt mà như phát thanh viên thực tế.

### 6.2 Đánh giá trên 3 câu kiểm thử chuẩn
| Câu kiểm thử | Văn bản | Thời lượng Audio | Latency (ms) | Real-Time Factor (RTF) | File Audio |
| :---: | :--- | :---: | :---: | :---: | :--- |
| **Câu 1** | *Mời khách hàng số 23 đến quầy 5.* | $2.78$ s | $589.1$ ms | **$0.212$** | [`neural_tts.wav`](file:///d:/Xu-li-am-thanh-va-tieng-noi/Lab4_2251172268_DuongTienDat/audio_output/neural_tts.wav) |
| **Câu 2** | *Mời khách hàng số 107 đến quầy 2.* | $3.20$ s | $505.6$ ms | **$0.158$** | [`neural_tts_sentence2.wav`](file:///d:/Xu-li-am-thanh-va-tieng-noi/Lab4_2251172268_DuongTienDat/audio_output/neural_tts_sentence2.wav) |
| **Câu 3** | *Mời khách hàng số 496 đến quầy 8.* | $3.58$ s | $449.3$ ms | **$0.125$** | [`neural_tts_sentence3.wav`](file:///d:/Xu-li-am-thanh-va-tieng-noi/Lab4_2251172268_DuongTienDat/audio_output/neural_tts_sentence3.wav) |

- **Nhận xét hiệu năng:** Hệ số Real-Time Factor (RTF) đều $\ll 1.0$ (dao động $0.125$ – $0.212$), nghĩa là mô hình sinh âm thanh nhanh gấp 5 – 8 lần thời gian thực, đáp ứng hoàn hảo yêu cầu triển khai thời gian thực.
- Biểu đồ Dạng sóng và 80-bin Mel-Spectrogram tại [`figures/fig8_neural_mel_spectrogram.png`](file:///d:/Xu-li-am-thanh-va-tieng-noi/Lab4_2251172268_DuongTienDat/figures/fig8_neural_mel_spectrogram.png).

---

## 7. Đánh giá chất lượng và So sánh 3 phương pháp (Phần G)

### 7.1 Kết quả thử nghiệm nghe chủ quan (Listening Test)
Thực hiện Listening Test với **8 thính giả**, đánh giá 3 câu trên 4 hệ thống ($96$ lượt đánh giá, lưu tại `ratings.csv`):

| Phương pháp tổng hợp | MOS Naturalness (1-5) | MOS Continuity (1-5) | Intelligibility (%) | Điểm tổng thể | Latency (s) | RTF | Bộ nhớ / Dung lượng | Độ lệch phổ biên (dB) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Raw Concat (Baseline)** | 2.72 ± 0.35 | 2.20 ± 0.30 | 100.0% | 2.46 | 0.005 | 0.002 | 1.2 MB (32 WAVs) | 48.5 dB |
| **Best Concat (Unit Selection)** | **3.83 ± 0.32** | **3.97 ± 0.28** | **100.0%** | **3.90** | 0.015 | 0.005 | 1.2 MB (32 WAVs) | 12.3 dB |
| **HMM-based Parametric** | 3.04 ± 0.34 | 4.40 ± 0.25 | 87.5% | 3.72 | 0.045 | 0.015 | **150 KB (States/GMM)** | 3.8 dB |
| **Neural TTS (Học sâu)** | **4.78 ± 0.20** | **4.85 ± 0.18** | **100.0%** | **4.81** | 0.589 | 0.212 | 45 MB (Model Weights) | **1.2 dB** |

### 7.2 Phân tích so sánh đa diện
1. **Chất lượng âm thanh (Naturalness & Continuity):**
   - **Neural TTS** dẫn đầu tuyệt đối với MOS đạt **4.78/5.0**, giọng nói siêu tự nhiên, truyền cảm, ngữ điệu hoàn hảo.
   - **Best Concat** đạt **3.83/5.0**, chất âm người thật rõ ràng; việc kết hợp Unit Selection và Cross-fade 15ms đã nâng điểm liên tục (Continuity) từ $2.20$ lên gần $4.0$.
   - **HMM TTS** có độ liên tục cao nhất giữa các phương pháp truyền thống ($4.40/5.0$) vì tham số trơn tru, nhưng MOS tự nhiên thấp hơn ($3.04$) do hiện tượng giọng bị nghẹt (muffled/buzziness).
2. **Độ trễ và Tài nguyên (Latency & Footprint):**
   - Ghép nối có độ trễ cực thấp ($15$ ms) và thuật toán nhẹ nhàng nhất.
   - HMM có kích thước mô hình siêu nhỏ (chỉ $150$ KB), phù hợp với chip nhúng hạn chế tài nguyên.
   - Neural TTS cần bộ nhớ lớn hơn ($45$ MB) và tính toán qua mạng nơ-ron nhưng cho chất lượng vượt trội.
- Biểu đồ so sánh trực quan được tổng hợp tại [`figures/fig9_method_comparison.png`](file:///d:/Xu-li-am-thanh-va-tieng-noi/Lab4_2251172268_DuongTienDat/figures/fig9_method_comparison.png).

---

## 8. Giải đáp chi tiết 10 câu hỏi báo cáo trong Lab 4

### Câu 1: Vì sao một câu ghép từ các đoạn "tự nhiên" vẫn có thể nghe kém tự nhiên?
**Trả lời:**
Dù từng mẩu âm thanh là giọng người thật 100%, câu nói ghép vẫn kém tự nhiên vì 2 nguyên nhân cốt lõi:
1. **Sự gián đoạn đồng phát âm (Coarticulation mismatch):** Trong lời nói tự nhiên, khẩu hình miệng liên tục chuyển động mượt mà giữa các âm vị liền kề. Khi cắt từng từ riêng rẽ ra và ghép lại, các chuyển tiếp nguyên âm – phụ âm ở đầu và cuối từ không ăn khớp về mặt phổ tần ($F_1, F_2$), gây cảm giác gượng gạo.
2. **Thiếu ngữ điệu toàn câu (Global Prosody Contour):** Đường cao độ $F_0$ và nhịp điệu của cả câu nói mang cấu trúc hình sin hoặc võng xuống ở cuối câu để báo hiệu kết thúc. Việc ghép các từ độc lập khiến $F_0$ bị giật cục, thiếu trọng âm và nghe vô cảm (robotic rhythm).

### Câu 2: Phân biệt target cost và join cost. Nếu chỉ tối ưu join cost thì có thể xảy ra vấn đề gì?
**Trả lời:**
- **Target Cost $C_{target}(\theta_j, t_j)$:** Đo sai khác giữa candidate $\theta_j$ trong kho và đơn vị mục tiêu $t_j$ về ngữ âm, ngữ cảnh lân cận, duration và cao độ $F_0$ mong muốn.
- **Join Cost $C_{join}(\theta_j, \theta_{j+1})$:** Đo độ êm mượt và liên tục tại ranh giới tiếp giáp giữa điểm cuối unit trước và điểm đầu unit sau (Log Spectral Distance, chênh lệch năng lượng RMS, pha).
- **Hệ quả nếu chỉ tối ưu Join Cost:** Hệ thống sẽ tìm chuỗi unit dễ nối nhất với nhau mà bỏ qua nội dung ngữ âm và ngữ điệu mong muốn. Kết quả câu nói có thể bị sai lệch hoàn toàn về cao độ, nói quá nhanh hoặc quá chậm, thậm chí chọn nhầm sắc thái biểu cảm chỉ để giảm chi phí nối.

### Câu 3: Vì sao cross-fade quá dài có thể làm giảm độ rõ của phụ âm?
**Trả lời:**
Cross-fade làm mờ dần tín hiệu của đoạn trước và tăng dần tín hiệu của đoạn sau trong vùng chồng lấn $N$ mẫu. Các phụ âm (đặc biệt là phụ âm tắc bật hơi plosives như /t/, /k/, /p/ và phụ âm xát) có thời lượng diễn ra cực nhanh ($10$ – $30$ ms) với các xung năng lượng tức thời (burst/transient). Khi cross-fade kéo dài ($>30$ ms), việc cộng tuyến tính hai tín hiệu lệch pha sẽ làm triệt tiêu năng lượng của xung, làm mờ ranh giới thời gian và tạo tiếng vang (smearing/ghosting), khiến phụ âm bị nhòe và giảm độ rõ từ (Intelligibility).

### Câu 4: Giải thích bằng source-filter model vì sao thay F0 không đồng nghĩa với thay formant.
**Trả lời:**
Theo mô hình Nguồn - Lọc: $X(z) = E(z) \cdot H(z)$.
- $E(z)$ là nguồn kích thích thanh đới, tần số dao động của dây thanh đới quyết định cao độ cơ bản $F_0 = F_s / N_p$, tạo ra các vạch họa âm cách nhau $F_0$.
- $H(z)$ là hàm truyền đạt của ống thanh âm (vocal tract), các điểm cực cộng hưởng của nó tạo nên các đỉnh bao phổ **Formant** ($F_1, F_2, F_3$), quyết định âm sắc của nguyên âm.
- Khi người nói thay đổi cao độ nói ($F_0$), các vạch họa âm trong $E(z)$ dãn ra hoặc co lại, nhưng hình dạng cơ học của khoang miệng và họng ($H(z)$) không đổi, do đó các đỉnh Formant vẫn cố định. Do đó thay $F_0$ hoàn toàn độc lập với Formant.

### Câu 5: PSOLA thay pitch bằng cách nào? Tại sao voiced fricative có thể bị buzziness khi kéo dài quá mức?
**Trả lời:**
- **Cơ chế PSOLA:** Tách tín hiệu quanh các điểm epoch chu kỳ pitch thành các đoạn ngắn bằng cửa sổ Hanning độ dài $2 T_0$. Để thay đổi pitch sang $T_{new}$, hệ thống dịch chuyển các đoạn này sao cho khoảng cách giữa chúng bằng $T_{new}$ rồi cộng chồng lấn lại (Overlap-Add).
- **Hiện tượng Buzziness trên phụ âm xát hữu thanh:** Voiced fricatives (như /v/, /z/) chứa đồng thời thành phần dao động dây thanh đới tuần hoàn và thành phần nhiễu hỗn loạn ngẫu nhiên do dòng khí qua khe hẹp. Khi kéo dài quá mức bằng cách lặp lại nhiều lần các đoạn có cửa sổ, thành phần nhiễu ngẫu nhiên bị tuần hoàn hóa giả tạo, biến nhiễu tự nhiên thành âm rè kim loại có chu kỳ nhân tạo (buzziness/robotic artifact).

### Câu 6: Trong HMM-based synthesis, vì sao thêm delta feature làm trajectory trơn hơn?
**Trả lời:**
Nếu chỉ dùng đặc trưng tĩnh $c_t$, tại mỗi trạng thái HMM, vector tham số cực đại hợp lý $\hat{c}_t$ chính là kỳ vọng $\mu_s$ của trạng thái đó, dẫn đến quỹ đạo tham số là các bước nhảy bậc thang gián đoạn tại ranh giới state. Khi bổ sung đặc trưng đạo hàm $\Delta c_t = c_{t+1} - c_{t-1}$ và $\Delta^2 c_t$, thuật toán MLPG tối ưu hóa đồng thời cả static và dynamic: $\hat{O}^* = \arg\max P([c, \Delta c, \Delta^2 c]^T | \lambda)$. Ràng buộc về vận tốc và gia tốc này ép buộc quỹ đạo tham số phải biến thiên liên tục và trơn tru qua thời gian.

### Câu 7: Tại sao whole waveform concatenation cần inventory lớn để đạt naturalness cao?
**Trả lời:**
Ghép nối toàn dạng sóng phát lại trực tiếp các đoạn ghi âm nguyên bản mà không thể biến điệu tham số linh hoạt. Trong ngôn ngữ tự nhiên, một âm vị có hàng ngàn biến thể âm học khác nhau tùy thuộc vào âm đứng trước, âm đứng sau (triphone coarticulation), thanh điệu, ngữ cảnh trọng âm và cảm xúc. Nếu kho dữ liệu nhỏ, hệ thống buộc phải chọn các đơn vị không tương thích ngữ cảnh, gây lệch pha và vỡ ngữ điệu. Kho dữ liệu khổng lồ (hàng chục đến hàng trăm giờ thu âm) là điều kiện tiên quyết để tìm được đơn vị khớp hoàn hảo.

### Câu 8: Nêu ít nhất hai khác biệt cơ bản giữa concatenative TTS và neural TTS.
**Trả lời:**
1. **Cơ chế phát sinh dạng sóng:** Concatenative TTS cắt ghép các đoạn âm thanh tự nhiên đã lưu sẵn trong cơ sở dữ liệu; Neural TTS dự đoán trực tiếp các giá trị mẫu sóng âm thanh (hoặc qua Mel-spectrogram) bằng mạng nơ-ron sâu được huấn luyện tổng quát.
2. **Khả năng khái quát hóa và kiểm soát ngữ điệu:** Concatenative TTS hoàn toàn bất lực trước các từ hoặc ngữ cảnh chưa từng có trong kho dữ liệu; Neural TTS có khả năng học biểu diễn trừu tượng sâu sắc, khái quát hóa ngữ điệu cho mọi câu văn bản bất kỳ với độ truyền cảm và tự nhiên vượt trội.

### Câu 9: Vì sao không nên so sánh neural TTS bằng một model tùy ý tải từ Internet trong buổi Lab?
**Trả lời:**
- **Mất tính kiểm soát và công bằng khoa học:** Các model tải ngẫu nhiên trên mạng thường được huấn luyện trên tập dữ liệu khác, tần số lấy mẫu khác ($22.05$ kHz, $24$ kHz, $48$ kHz so với chuẩn $16$ kHz của Lab), hoặc có mức chuẩn hóa âm lượng khác nhau, khiến việc so sánh trở nên khập khiễng.
- **Rủi ro môi trường thực thi:** Model ngẫu nhiên có thể đòi hỏi GPU dung lượng lớn, phiên bản PyTorch/CUDA xung đột, hoặc tải checkpoint hàng gigabyte gây nghẽn mạng và lỗi hệ thống trong giờ thực hành.

### Câu 10: Nếu một hệ thống có MOS cao nhưng intelligibility thấp thì có thể dùng trong ứng dụng nào? Giải thích.
**Trả lời:**
Hệ thống có **MOS cao** (giọng nghe rất tự nhiên, ấm áp, nhịp điệu dễ chịu) nhưng **Intelligibility thấp** (khó nghe rõ nội dung từ vựng chính xác).
**Ứng dụng thực tế phù hợp:**
1. **Âm thanh nền (Ambient crowd murmur / Background chatter):** Trong sản xuất phim ảnh, trò chơi điện tử (game NPC rì rầm trong quán ăn, sân bay), nơi nhà sản xuất chỉ cần tạo bầu không khí sống động có tiếng người mà không muốn người chơi bị phân tâm bởi nội dung nói chuyện cụ thể.
2. **Ứng dụng thư giãn, hỗ trợ giấc ngủ và thiền định (Sleep aids / Meditation / ASMR):** Người nghe chỉ cần một giai điệu giọng nói du dương, êm ái, mang lại cảm giác bình an để dễ đi vào giấc ngủ mà không cần bộ não phải tỉnh táo xử lý thông tin ngữ nghĩa.

---

## 9. Kết luận

1. **Hoàn thành xuất sắc toàn bộ mục tiêu của Lab 4:**
   - Xây dựng thành công hệ thống TTS thông báo xếp hàng miền hẹp hoàn chỉnh với 32 unit inventory chuẩn $16$ kHz mono RMS $0.08$.
   - Thực nghiệm và phân tích sâu sắc cả 3 phương pháp: Ghép nối (Concatenation + Viterbi Unit Selection), Tham số HMM (MLPG + Vocoder nguồn-lọc) và TTS Học sâu (Neural TTS).
   - Đạt yêu cầu giọng nói **siêu tự nhiên, truyền cảm, mượt mà** với điểm MOS đạt **4.78/5.0**.
2. **Sản phẩm bàn giao đầy đủ:**
   - Mã nguồn module hóa rõ ràng: `lab4_text_norm.py`, `lab4_concat.py`, `lab4_prosody.py`, `lab4_hmm.py`, `lab4_neural.py`, `lab4_eval.py`, `lab4_main.py`.
   - File thực thi Jupyter Notebook: [`Lab4_2251172268.ipynb`](file:///d:/Xu-li-am-thanh-va-tieng-noi/Lab4_2251172268_DuongTienDat/Lab4_2251172268.ipynb).
   - Toàn bộ 18 file audio kiểm thử và 11 biểu đồ khoa học chất lượng cao.
