# Báo cáo Lab 2 – CSE457: Xử lý âm thanh và tiếng nói
## Đặc trưng tiếng nói và nhận dạng bằng DTW

| Thông tin | Nội dung |
|-----------|----------|
| **Sinh viên** | Dương Tiến Đạt |
| **MSSV** | 2251172268 |
| **Học phần** | CSE457 – Xử lý âm thanh và tiếng nói |
| **Bài** | Lab 2 – Từ MFCC đến nhận dạng từ đơn bằng DTW |

---

## 1. Bài toán và dữ liệu

### 1.1 Vocabulary

| ASCII label | Tiếng Việt | Đặc điểm âm học |
|-------------|-----------|-----------------|
| `khong` | Không | Fricative /kh/ + vowel /ô/ + nasal /ng/ |
| `mot` | Một | Nasal /m/ + vowel /ô/ + stop /t/ |
| `hai` | Hai | Fricative /h/ + vowel /a/ + glide /i/ |
| `ba` | Ba | Voiced stop /b/ + open vowel /a/ |
| `bon` | Bốn | Voiced stop /b/ + rounded vowel /ô/ + nasal /n/ |

### 1.2 Dataset

- **Nguồn:** Tín hiệu tổng hợp bằng source-filter model (glottal source + IIR formant resonators)
- **Mỗi từ:** 5 lần lặp với biến thể tự nhiên về pitch (±15%), tốc độ (±20%), biên độ (±10%), jitter/shimmer
- **Chia train/test:** File `_01` `_02` `_03` → train template; File `_04` `_05` → test
- **Format:** WAV PCM 16-bit mono, Fs = 16,000 Hz

| Label | Dur avg (s) | Train | Test |
|-------|------------|-------|------|
| khong | 0.939 | 3 | 2 |
| mot | 0.879 | 3 | 2 |
| hai | 0.889 | 3 | 2 |
| ba | 0.879 | 3 | 2 |
| bon | 0.909 | 3 | 2 |

---

## 2. Cấu hình tham số baseline

| Tham số | Giá trị | Ý nghĩa |
|---------|---------|---------|
| Fs | 16,000 Hz | Chuẩn hóa tất cả file |
| Frame | 25 ms = 400 mẫu | Giả thiết gần dừng |
| Hop | 10 ms = 160 mẫu | 100 frame/giây |
| Window | Hamming | Giảm spectral leakage |
| Pre-emphasis α | 0.97 | y[n] = x[n] − 0.97·x[n−1] |
| NFFT | 512 | FFT mỗi frame |
| n_mels | 24 | Mel filterbank filters |
| n_mfcc | 13 | Hệ số MFCC/frame |
| CMN | Có | Cepstral Mean Normalization |
| Local distance | Euclidean | ||x_i − y_j||₂ |
| DTW cost | Tổng / path length | Chuẩn hóa theo độ dài đường |
| Endpoint top_db | 30 dB | Ngưỡng energy tương đối |
| Endpoint ZCR thresh | 0.15 | Ngưỡng mở rộng biên fricative |
| Endpoint margin | 40 ms | Buffer trước/sau biên |

---

## 3. Kết quả từng phần

### Phần A – Thu dữ liệu & kiểm tra

- 25 file WAV, Fs=16kHz, không clipping đáng kể (peak ≈ 0 dBFS)
- Thời lượng từ 0.85–0.98 giây/file
- Phân chia train/test trước khi xây dựng template → **không data leakage**

### Phần B – Đặc trưng miền thời gian

**Thống kê Energy & ZCR cho từ "Không":**

| Vùng | Log-energy | ZCR | Mô tả |
|------|-----------|-----|-------|
| Silence | < −23 dB | ~0.3–0.6 | Nhiễu ngẫu nhiên |
| Voiced /ô/ | > −23 dB | < 0.08 | Năng lượng cao, ít zero-crossing |
| Unvoiced /kh/ | −23 dB – 0 dB | > 0.08 | ZCR cao (fricative) |

**Ước lượng pitch bằng autocorrelation:**
- Frame voiced: F0 ≈ 262 Hz (lag đỉnh = 61 mẫu, Fs/61 ≈ 262 Hz)
- Frame unvoiced: Không có đỉnh autocorrelation → ZCR cao, phân loại đúng

**Nhận xét:** Voiced có energy cao và ZCR thấp; Unvoiced có ZCR cao (cắt zero nhiều do tín hiệu tần số cao); Silence có energy rất thấp.

### Phần C – Endpoint Detection

**Thuật toán:**
1. Tính log-energy mỗi frame
2. Ngưỡng tương đối: `le_thresh = max_le − 30 dB`
3. Frame nằm trong vùng speech nếu `le >= le_thresh`
4. Mở rộng biên với ZCR > 0.15 (bắt fricative đầu/cuối từ)
5. Thêm margin 40 ms để không cắt phụ âm

| Từ | Trước (s) | Sau (s) | Cắt (ms) |
|----|----------|---------|---------|
| Không | 0.940 | 0.475 | 465 |
| Một | 0.880 | 0.445 | 435 |
| Hai | 0.890 | 0.455 | 435 |
| Ba | 0.880 | 0.415 | 465 |
| Bốn | 0.910 | 0.445 | 465 |

**Nhận xét:** Cắt ~435–465 ms (phần silence đầu và cuối). Margin 40ms đảm bảo không cắt mất phụ âm đầu/cuối từ.

### Phần D – MFCC

**Pipeline:**
```
Pre-emphasis (α=0.97) → Framing (25ms/10ms, Hamming) → FFT (N=512)
→ Power spectrum → Mel filterbank (24 filters) → log → DCT → CMN
```

**Số frame theo utterance:**

| File | N frame | Hệ số/frame |
|------|---------|------------|
| khong_01 | 46 | 13 (cố định) |
| khong_02 | 50 | 13 |
| khong_03 | 42 | 13 |
| mot_01 | 43 | 13 |
| ... | ... | 13 |

**Lý do số frame khác nhau:** T = ⌊(N − 400) / 160⌋ + 1, N là số mẫu thay đổi theo thời lượng từng utterance. Số hệ số MFCC/frame luôn = 13 vì phụ thuộc vào cấu hình (n_mfcc), không phụ thuộc độ dài tín hiệu.

### Phần E – DTW tự cài đặt

**Kiểm tra bắt buộc:** DTW(X, X) = 0.000000 ✓

| So sánh | DTW_norm | Path length | Nhận xét |
|---------|---------|------------|---------|
| Cùng từ [Không vs Không] | **9.583** | 50 | Path bám gần đường chéo |
| Khác từ [Không vs Một] | **53.681** | 47 | Path phân kỳ, cost cao hơn 5.6× |

**Nhận xét DP:** Bước chéo (i−1,j−1) = 2 chuỗi tiến đồng bộ (1:1); Bước dọc (i−1,j) = X tiến, Y đứng (X nhanh hơn); Bước ngang (i,j−1) = Y tiến, X đứng (Y nhanh hơn). Sự kết hợp linh hoạt 3 bước giúp DTW xử lý tốc độ nói khác nhau.

### Phần F – Bộ nhận dạng nearest-template

**Kết quả nhận dạng 10 file test:**

| File | True | Pred | DTW top-1 | DTW top-2 | ✓/✗ |
|------|------|------|-----------|-----------|------|
| khong_04 | Không | Không | 7.480 | hai=20.059 | ✓ |
| khong_05 | Không | Không | 9.410 | bon=17.641 | ✓ |
| mot_04 | Một | Một | 11.357 | bon=45.208 | ✓ |
| mot_05 | Một | Một | 8.287 | bon=44.788 | ✓ |
| hai_04 | Hai | Hai | 6.684 | khong=19.538 | ✓ |
| hai_05 | Hai | Hai | 7.144 | khong=21.359 | ✓ |
| ba_04 | Ba | Ba | 7.239 | bon=23.021 | ✓ |
| ba_05 | Ba | Ba | 5.538 | bon=24.209 | ✓ |
| bon_04 | Bốn | Bốn | 6.537 | khong=19.134 | ✓ |
| bon_05 | Bốn | Bốn | 5.850 | hai=22.896 | ✓ |

**Accuracy = 10/10 = 100.0%**

### Phần G – Đánh giá & Thí nghiệm

**Confusion matrix (Baseline):**

|  | Không | Một | Hai | Ba | Bốn |
|--|-------|-----|-----|----|-----|
| **Không** | 2 | 0 | 0 | 0 | 0 |
| **Một** | 0 | 2 | 0 | 0 | 0 |
| **Hai** | 0 | 0 | 2 | 0 | 0 |
| **Ba** | 0 | 0 | 0 | 2 | 0 |
| **Bốn** | 0 | 0 | 0 | 0 | 2 |

**Tổng hợp thí nghiệm:**

| Thí nghiệm | Accuracy | Nhận xét |
|------------|---------|----------|
| Baseline (trim + MFCC 13 + DTW) | **100.0%** | Speaker-dependent, dataset tổng hợp |
| E1: Có trim | 100.0% | |
| E1: Không trim | 100.0% | Tín hiệu tổng hợp có silence ngắn → ít ảnh hưởng |
| E2: MFCC 13 | 100.0% | |
| E2: MFCC+Δ 26 | 100.0% | Dataset đơn giản → không phân biệt được |

**Nhận xét thí nghiệm E1:** Với dataset tổng hợp có silence đầu/cuối ngắn (~0.28s) và biến thể nhỏ, endpoint detection không cải thiện thêm vì DTW đã đủ khả năng "bỏ qua" vùng silence. Với tiếng nói thực có silence dài hơn và noisy, endpoint sẽ quan trọng hơn.

**Nhận xét thí nghiệm E2:** Trên dataset đơn giản (5 từ, 1 người nói, tổng hợp có kiểm soát), MFCC 13 đã đủ phân biệt. Delta (26 chiều) không thay đổi accuracy nhưng tăng thời gian tính DTW gấp đôi.

---

## 4. Câu hỏi báo cáo

### Câu 1: Tại sao không dùng waveform làm template?

Hai utterance cùng từ thường có thời lượng khác nhau. Nếu dùng waveform trực tiếp:
- Không thể so sánh công bằng hai chuỗi có độ dài khác nhau mà không normalize thời gian
- DTW trên waveform (1D) không nắm bắt cấu trúc tần số, dễ bị ảnh hưởng bởi pitch và biên độ
- MFCC loại bỏ thông tin không liên quan (pitch, biên độ), nén thông tin phổ thành 13 hệ số → bất biến hơn với người nói

### Câu 2: Vai trò của energy và ZCR trong endpoint detection

**Short-time energy:**
- Phân biệt speech và silence hiệu quả (speech có energy cao hơn ít nhất 30 dB)
- Nhạy với voiced segments (có nhiều năng lượng)
- Nhược điểm: Không phát hiện tốt fricative /s/, /sh/ (energy thấp)

**ZCR:**
- Bổ sung cho energy: fricative và unvoiced speech có ZCR cao (>0.1)
- Dùng để mở rộng biên endpoint khi từ bắt đầu/kết thúc bằng phụ âm vô thanh
- Ví dụ: "sáu" bắt đầu bằng /s/ (ZCR cao, energy trung bình) → energy threshold sẽ bỏ mất → ZCR cứu lại

### Câu 3: Mel filterbank dày hơn ở tần số thấp?

Thang Mel phi tuyến: $B(f) = 1125 \cdot \ln(1 + f/700)$

Đảo ngược: $f = 700 \cdot (e^{b/1125} - 1)$

Khi b tăng đều (các bộ lọc đặt cách đều trên thang Mel):
$$\frac{df}{db} = \frac{700}{1125} e^{b/1125} = \frac{f + 700}{1125}$$

Đạo hàm tăng theo f → khoảng cách Hz giữa các bộ lọc **tăng dần** khi f tăng. Điều này phản ánh độ phân giải tần số của tai người: nhạy hơn ở tần số thấp (<1kHz), ít phân biệt hơn ở tần số cao.

### Câu 4: Log và DCT trong MFCC

**Log:**
- Nén dynamic range: thay vì 0–10⁶ (tuyến tính) → 0–60 dB (log)
- Phù hợp với cảm nhận thính giác (tai nghe theo thang log)
- Giúp ổn định hóa phương sai giữa các tần số

**DCT:**
- Biến đổi M log-energy filterbank (tương quan với nhau) thành các **cepstral coefficients** gần độc lập (decorrelation)
- Tương đương với việc mô tả đường bao phổ bằng tổng cosine
- Cho phép chọn n_mfcc < n_mels (13 < 24) để nén thông tin đồng thời giữ lại thông tin quan trọng

### Câu 5: Ý nghĩa các bước trong DTW

$$D[i,j] = C[i,j] + \min\{D[i-1,j],\ D[i,j-1],\ D[i-1,j-1]\}$$

| Bước | Ý nghĩa | Trường hợp dùng |
|------|---------|----------------|
| **(i−1, j)** bước dọc | X tiến 1, Y đứng yên | X nói nhanh hơn tại vùng này |
| **(i, j−1)** bước ngang | Y tiến 1, X đứng yên | Y nói nhanh hơn tại vùng này |
| **(i−1, j−1)** bước chéo | Cả hai tiến 1 | Tốc độ nói tương đương |

Kết hợp 3 bước → DTW xử lý được tốc độ nói thay đổi phi tuyến theo thời gian.

### Câu 6: Tại sao chuẩn hóa DTW/path_length?

Không chuẩn hóa: $D[N,M]$ phụ thuộc cả vào **chất lượng khớp** lẫn **độ dài**:
- Chuỗi dài → nhiều frame → tổng local distance lớn hơn dù cùng từ
- File 1s và file 0.5s cùng từ sẽ có cost khác nhau rất lớn → so sánh thiếu công bằng

Chuẩn hóa $\text{DTW\_norm} = D[N,M] / |P|$:
- $|P|$ = số cặp frame trên đường tối ưu
- Giá trị xấp xỉ **khoảng cách Euclid trung bình mỗi cặp frame** → đơn vị nhất quán
- Cho phép so sánh trực tiếp giữa các utterance có độ dài khác nhau

### Câu 7: Ba nguyên nhân MFCC khác nhau giữa 2 lần nói cùng từ

1. **Tốc độ nói khác nhau:** Số frame khác nhau → vị trí các phoneme trong chuỗi frame khác nhau → MFCC tại cùng frame index mô tả phần khác của từ

2. **Pitch (F0) khác nhau:** Pre-emphasis và FFT bao gồm thông tin harmonic → F0 ảnh hưởng đến phân bố năng lượng trong các Mel filter → MFCC c0, c1, c2 thay đổi

3. **Coarticulation và biến thể phát âm:** Ngay cả cùng người nói, các phoneme trong từ ảnh hưởng qua lại nhau (coarticulation), miệng/lưỡi không bao giờ ở đúng cùng 1 vị trí → formant F1, F2, F3 dịch chuyển nhẹ → MFCC của vowel segments thay đổi

### Câu 8: Cặp từ dễ nhầm nhất

Từ confusion matrix (100%), không có nhầm trong dataset tổng hợp. Dự đoán nhầm với tiếng nói thực:
- **"Ba" và "Bốn"** dễ nhầm: cả hai bắt đầu bằng /b/ → voiced stop giống nhau; DTW top-2 của "ba" luôn là "bon" (score 23.0 vs 7.2)
- **"Không" và "Hai"** có thể nhầm: cả hai bắt đầu bằng fricative (/kh/ và /h/) → ZCR cao ở frame đầu; top-2 của "khong" là "hai" (20.06)

Phân tích MFCC: "Ba" và "Bốn" có vowel F1/F2 gần nhau (~730/1090 Hz); chỉ khác ở phần cuối (open vs nasalized). Nếu endpoint cắt mất nasal /n/ cuối của "Bốn" → dễ nhầm thành "Ba".

### Câu 9: DTW sẽ gặp hạn chế gì với người nói mới?

**Hạn chế của DTW với người nói mới (cross-speaker):**
- MFCC vẫn chứa **thông tin người nói** (vocal tract size → F1/F2 khác nhau, F0 khác nhau giữa nam/nữ)
- Template của người A khi so sánh với tiếng nói người B sẽ có DTW_norm lớn hơn nhiều
- Không có cơ chế học để generalize → accuracy cross-speaker thường thấp hơn 20–30%

**Chương 3 giải quyết bằng HMM:**
- Mô hình xác suất thay vì nearest-template deterministic
- HMM học **phân bố** của MFCC cho từng phoneme → generalize tốt hơn qua nhiều người nói
- Viterbi algorithm (DP) → tìm chuỗi trạng thái tối ưu → tương đương DTW nhưng probabilistic
- Kết hợp acoustic model + language model → mạnh hơn DTW cho large vocabulary

---

## 5. Sản phẩm nộp

```
Lab2_2251172268_DuongTienDat/
├── Lab2_2251172268.ipynb          ← Notebook 33 cells, chạy từ đầu đến cuối
├── report_Lab2.md                 ← Báo cáo (file này)
├── requirements.txt
├── generate_dataset.py            ← Tổng hợp dataset 25 file WAV
├── lab2_utils.py                  ← Tiện ích: framing, energy, ZCR, autocorr
├── lab2_part_abcd.py              ← Phần A–D: dữ liệu → MFCC
├── lab2_part_efg.py               ← Phần E–G: DTW → recognizer → đánh giá
├── build_notebook_lab2.py         ← Script build notebook
├── results.csv                    ← File test, nhãn thật, nhãn dự đoán, scores
│
├── dataset/
│   ├── khong/  khong_01.wav … khong_05.wav
│   ├── mot/    mot_01.wav … mot_05.wav
│   ├── hai/    hai_01.wav … hai_05.wav
│   ├── ba/     ba_01.wav … ba_05.wav
│   └── bon/    bon_01.wav … bon_05.wav
│
└── figures/
    ├── A_waveform_3words.png
    ├── B_energy_zcr.png
    ├── B_autocorrelation.png
    ├── C_endpoint_detection.png
    ├── D_mel_filterbank.png
    ├── D_mfcc_heatmap.png
    ├── E_dtw_path.png
    ├── G_confusion_matrix.png
    ├── G_experiment_E1.png
    └── G_experiment_E2.png
```

---

## 6. Tài liệu tham khảo

[1] Huang, X., Acero, A., & Hon, H-W. *Spoken Language Processing*. Prentice-Hall, 2001.

[2] Rabiner, L.R., & Schafer, R.W. *Theory and Applications of Digital Speech Processing*. Pearson, 2011.

[3] Jurafsky, D., & Martin, J.H. *Speech and Language Processing*. Prentice-Hall, 2008.

[4] Đề cương chi tiết học phần CSE457 – Xử lý âm thanh và tiếng nói, TLU, 2023.
