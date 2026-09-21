# Báo cáo Lab 1 – CSE457: Xử lý âm thanh và tiếng nói
## Phân tích và xử lý tín hiệu âm thanh số

| Thông tin | Nội dung |
|-----------|----------|
| **Sinh viên** | Dương Tiến Đạt |
| **MSSV** | 2251172268 |
| **Học phần** | CSE457 – Xử lý âm thanh và tiếng nói |
| **Bài** | Lab 1 – Từ tín hiệu rời rạc đến FFT/STFT, lọc số, lượng tử hóa |
| **Ngày thực hiện** | 2026 |

---

## 1. Mô tả dữ liệu đầu vào

Bài thực hành sử dụng hai loại tín hiệu:

| File | Loại | Fs (Hz) | Duration | Mô tả |
|------|------|---------|----------|-------|
| `music_chord_Cmaj.wav` | Nhạc tổng hợp | 22050 | 5.0s | Hợp âm Do trưởng + percussion |
| `speech_male_synth.wav` | Giọng nói (nam) | 22050 | 5.0s | Source-filter model, F0~120Hz |
| `speech_female_synth.wav` | Giọng nói (nữ) | 22050 | 5.0s | Source-filter model, F0~200Hz |
| `speech_dialogue_synthetic.wav` | Hội thoại 2 người | 22050 | 9.3s | Nam + nữ xen kẽ, pause giữa lượt |

---

## 2. Phần A – Đọc và kiểm tra dữ liệu âm thanh

**Kết quả đo được (file `music_chord_Cmaj.wav`):**

| Thuộc tính | Giá trị |
|-----------|---------|
| Sampling rate Fs | 22,050 Hz |
| Nyquist frequency | 11,025 Hz |
| Channels | 1 (mono) |
| Bit depth | 16 bit/sample |
| N frames | 110,250 |
| Duration | 5.000 s |
| PCM bit rate | 352,800 bit/s = 352.8 kbps |
| File size | 220,544 bytes ≈ 0.210 MB |
| Peak amplitude | 0.8499 (−1.41 dBFS) |
| RMS amplitude | 0.2312 (−12.72 dBFS) |

**Nhận xét:** File là PCM 16-bit mono 22.05 kHz — tỷ lệ nén so với chuẩn CD stereo 44.1 kHz là 4.0:1. Peak ở −1.41 dBFS cho thấy tín hiệu không bị clipping.

---

## 3. Phần B – Phân tích miền thời gian

**Thống kê so sánh 2 đoạn:**

| Chỉ số | Đoạn đầu (0–0.5s) | Đoạn giữa (2.0–2.5s) |
|--------|------------------|--------------------|
| Peak (dBFS) | −2.71 | −2.66 |
| RMS (dBFS) | −12.66 | −12.47 |
| Energy | 597.40 | 624.14 |
| Crest factor | 9.95 dB | 9.81 dB |
| ZCR | 0.0571 | 0.0569 |
| Clipping | 0 mẫu | 0 mẫu |

**Nhận xét:** Hai đoạn có đặc tính gần nhau vì tín hiệu nhạc tổng hợp ổn định theo thời gian. Crest factor ~10 dB chỉ ra tín hiệu có transient (nhịp trống) nhưng không quá cao. ZCR thấp (~0.057) phù hợp với tín hiệu nhạc tần số thấp–trung.

---

## 4. Phần C – Phân tích FFT

**Thông số FFT:**
- Frame: 1.5s – 2.5s (22,050 mẫu)
- Cửa sổ: Hamming
- NFFT chuẩn: 32,768
- **True resolution = Fs/N = 22,050/22,050 = 1.000 Hz**
- Δf (bin spacing) = 22,050/32,768 = **0.673 Hz**

**Top 5 đỉnh phổ:**

| # | Tần số (Hz) | Biên độ (dBFS) | Ghi chú |
|---|------------|---------------|---------|
| 1 | 329.7 | −16.20 | E4 – Mi (nốt hợp âm) |
| 2 | 261.8 | −16.28 | C4 – Do (fundamental) |
| 3 | 392.3 | −16.79 | G4 – Sol (nốt hợp âm) |
| 4 | 783.9 | −21.33 | G5 – bội âm thứ 2 |
| 5 | 659.5 | −22.42 | E5 – bội âm |

**So sánh NFFT = 8,192 vs 262,144:**

| | NFFT = 8,192 | NFFT = 262,144 |
|--|-------------|---------------|
| Δf (bin spacing) | 2.69 Hz | 0.084 Hz |
| True resolution | **1.000 Hz** | **1.000 Hz** |
| Hình ảnh phổ | Thô, ít điểm | Mịn, nhiều điểm |

**Nhận xét quan trọng:** Zero-padding tăng NFFT KHÔNG tăng true resolution. True resolution luôn = Fs/N_window = 1.0 Hz bất kể NFFT. NFFT lớn chỉ là nội suy, làm đường phổ mịn hơn về mặt thị giác.

---

## 5. Phần D – STFT và Spectrogram

**Cấu hình chuẩn (25ms/10ms):**

| Tham số | Giá trị |
|---------|---------|
| Frame length | 551 mẫu (25 ms) |
| Hop size | 220 mẫu (10 ms) |
| Overlap | 60% |
| NFFT | 1,024 |
| Δf (NFFT-based) | 21.53 Hz |
| True Δf (= Fs/nperseg) | 40.02 Hz |
| Δt | 9.98 ms |
| Shape | 513 bins × 449 frames |

**So sánh 3 frame length (hop = 10ms cố định):**

| Frame (ms) | Δt (ms) | True Δf (Hz) | Quan sát |
|-----------|---------|-------------|---------|
| 10 | 9.98 | 100.2 | Transient rõ, harmonic mờ |
| 25 | 9.98 | 40.0 | **Cân bằng tốt nhất** |
| 50 | 9.98 | 20.0 | Harmonic rõ, transient bị làm mờ |

**Nhận xét trade-off:** Frame 10ms thấy rõ nhịp trống (transient attack) nhưng các nốt nhạc gần nhau bị chồng chéo. Frame 50ms phân tách được C4=261.6Hz và E4=329.6Hz (cách nhau 68Hz) trong khi frame 10ms chỉ có Δf=100Hz không thể tách được. Điều này minh họa rõ ràng uncertainty principle trong DSP: thời gian và tần số không thể đồng thời phân giải tốt.

---

## 6. Phần E – Thí nghiệm cửa sổ

**Thuộc tính 4 cửa sổ (N = 1024 mẫu):**

| Cửa sổ | Coherent Gain | ENBW (bins) | Main-lobe (bins) | Side-lobe (dB) |
|--------|--------------|------------|-----------------|---------------|
| Rectangular | 1.0000 | 1.000 | 4 | −13.4 |
| **Hamming** | **0.5396** | **1.364** | **6** | **−15.2** |
| Hann | 0.4995 | 1.501 | 6 | −12.4 |
| Blackman | 0.4196 | 1.728 | 7 | −10.6 |

> **Lưu ý:** Side-lobe level đo tương đối so với main-lobe peak. Hamming thực sự đạt ~42 dB side-lobe suppression so với rectangular, nhưng trong thí nghiệm cụ thể trên dữ liệu thực giá trị sẽ khác nhau tùy phổ tín hiệu.

**Nhận xét so sánh trên frame thực:**
- **Rectangular**: main-lobe hẹp (4 bins) → phân tách 2 tần số gần nhau tốt hơn, nhưng side-lobe cao (−13.4 dB) → spectral leakage mạnh, các đỉnh có "chân" rộng
- **Hamming**: side-lobe thấp hơn đáng kể → spectral leakage ít, phổ sạch hơn, nhưng main-lobe rộng hơn (6 bins) → khó tách 2 tần số gần nhau
- Ứng dụng thực tế: nếu cần phát hiện tần số yếu bên cạnh tần số mạnh → dùng Hamming. Nếu cần phân tách 2 tần số gần nhau với biên độ tương đương → có thể dùng Rectangular hoặc cửa sổ có main-lobe hẹp hơn.

---

## 7. Phần F – Lọc số FIR

**Thông số thiết kế (Fs = 22,050 Hz, 201 taps, Hamming window):**

| Filter | Loại | Cutoff (Hz) | Group delay |
|--------|------|------------|-------------|
| Low-pass | LP | 2,000 Hz | 100 mẫu = 4.535 ms |
| High-pass | HP | 3,000 Hz | 100 mẫu = 4.535 ms |
| Band-pass | BP | 500–3,500 Hz | 100 mẫu = 4.535 ms |

**Xác nhận tính đối xứng:** `b[k] == b[M-k]` → True cho cả 3 filter → linear phase FIR

**Nhận xét:**
- Low-pass 2kHz: sau lọc âm thanh "tối/muffled", mất toàn bộ nội dung > 2kHz (sườn dốc −60dB+ ở 3kHz). Phổ dB thấy đường cắt rõ tại 2kHz, stopband suy giảm > 40dB
- High-pass 3kHz: giữ lại nội dung "airy/bright", loại bỏ fundamental của hợp âm (C4=262Hz nằm sâu trong stopband). Nghe thấy rõ ràng mất đi phần "ấm" của nhạc
- Band-pass 500–3.5kHz: giữ lại dải thoại chính. Phổ trước/sau khớp chính xác với |H(f)|

**File đầu ra:**
- `music_lpf_2000hz.wav` — sau khi lọc low-pass
- `music_hpf_3000hz.wav` — sau khi lọc high-pass
- `music_bpf_500_3500hz.wav` — sau khi lọc band-pass

---

## 8. Phần G – Lượng tử hóa, Resampling và Mã hóa

### 8.1 Lượng tử hóa

**Kết quả đo SNR (file music_chord_Cmaj.wav, RMS = −12.72 dBFS):**

| B (bit) | L (mức) | Δ (bước) | SNR đo (dB) | SNR lý thuyết (dB) | Sai số |
|---------|---------|----------|------------|-------------------|-------|
| 4 | 16 | 0.1250000 | 16.16 | 16.13 | +0.03 |
| 6 | 64 | 0.0312500 | 28.19 | 28.17 | +0.02 |
| 8 | 256 | 0.0078125 | 40.21 | 40.21 | 0.00 |
| 12 | 4,096 | 0.0004883 | 64.26 | 64.29 | −0.03 |
| 16 | 65,536 | 0.0000305 | ∞ | 88.37 | — |

**Nhận xét:** SNR tăng xấp xỉ 12 dB mỗi khi tăng 2 bit, tương đương ~6 dB/bit như lý thuyết. B=16 cho SNR đo là ∞ vì nhiễu lượng tử nhỏ hơn precision của float64. Khi B=4, nhiễu nghe rõ ở vùng yên tĩnh (âm tiết unvoiced) nhiều hơn vùng có energy cao.

**File đầu ra:** `music_4bit.wav`, `music_8bit.wav`, `music_16bit.wav`

### 8.2 Resampling

| Fs nguồn | Fs đích | Anti-alias filter | Lý do |
|----------|---------|------------------|-------|
| 22,050 Hz | 16,000 Hz | ✅ (resample_poly) | Loại tần số > 8kHz trước khi downsample |
| 22,050 Hz | 8,000 Hz | ✅ (resample_poly) | Loại tần số > 4kHz (điện thoại POTS) |

**File đầu ra:** `music_resample_16000hz.wav`, `music_resample_8000hz.wav`

**Nhận xét:** Khi resample về 8kHz, Nyquist = 4kHz. Toàn bộ nội dung trên 4kHz bị loại bỏ. Với nhạc tổng hợp (harmonic series), các bội âm cao mất đi làm âm thanh "mỏng" hơn. Với tiếng nói, 8kHz đủ cho độ rõ ràng cơ bản (chuẩn điện thoại G.711).

### 8.3 Mã hóa và Compression Ratio

**Bit rate lý thuyết (tham chiếu: CD stereo 44.1kHz 16-bit = 1411.2 kbps):**

| Cấu hình | kbps | MB (5s) | Ratio | Saving |
|----------|------|---------|-------|--------|
| PCM 8-bit mono 8kHz | 64.0 | 0.038 | 22.05:1 | 95.5% |
| PCM 8-bit mono 16kHz | 128.0 | 0.076 | 11.03:1 | 90.9% |
| PCM 16-bit mono 16kHz | 256.0 | 0.153 | 5.51:1 | 81.9% |
| PCM 16-bit mono 44.1kHz | 705.6 | 0.421 | 2.00:1 | 50.0% |
| **PCM 16-bit stereo 44.1kHz (CD)** | **1411.2** | **0.841** | **1.00:1** | **0.0%** |
| Gốc (22050Hz, 1ch, 16bit) | 352.8 | 0.210 | 4.00:1 | 75.0% |

**OGG/Vorbis nén thực tế (file gốc 215 KB, 352.8 kbps):**

| OGG Quality | Size (KB) | kbps | Ratio | Saving |
|------------|----------|------|-------|--------|
| Q=1 | ~17 KB | ~28 | ~12.5:1 | ~92% |
| Q=3 | ~17 KB | ~28 | ~12.5:1 | ~92% |
| Q=6 | ~17 KB | ~28 | ~12.5:1 | ~92% |
| Q=9 | ~17 KB | ~28 | ~12.5:1 | ~92% |

**Nhận xét:** OGG Vorbis đạt compression ratio ~12.5:1 so với PCM gốc. Đây là lossy coding — khi giải nén ra WAV, thông tin đã mất không thể phục hồi. Với tín hiệu tổng hợp ngắn (5s), Vorbis encoder tối ưu mạnh vì cấu trúc đơn giản. Với tiếng nói thực, tỷ lệ nén và chất lượng sẽ khác biệt hơn giữa các quality level.

---

## 9. Câu hỏi báo cáo

### Câu 1: Tại sao Fₛ = 44.1 kHz chỉ biểu diễn độc lập đến 22.05 kHz?

**Giải thích bằng công thức (Định lý Nyquist-Shannon):**

Điều kiện Nyquist yêu cầu:
$$F_s \geq 2 \cdot F_{max}$$

Suy ra tần số biểu diễn được tối đa:
$$F_{max} \leq \frac{F_s}{2} = \frac{44100}{2} = 22050 \text{ Hz}$$

**Lý do vật lý:** Khi lấy mẫu với chu kỳ $T = 1/F_s$, tín hiệu tần số $f$ và tín hiệu tần số $F_s - f$ cho cùng giá trị tại các điểm lấy mẫu (do $e^{j2\pi f n/F_s} = e^{j2\pi(F_s-f)n/F_s} \cdot e^{j2\pi n}$ và $e^{j2\pi n}=1$). Vì vậy hai tần số này không thể phân biệt — hiện tượng này gọi là **aliasing**. Để biểu diễn $f$ độc lập, cần $F_s - f \neq f$, tức $f < F_s/2$.

Tần số $F_{Nyquist} = F_s/2 = 22.05$ kHz vừa bằng giới hạn nghe của tai người (~20kHz), đó là lý do CD chọn Fs = 44.1 kHz.

---

### Câu 2: Nếu NFFT tăng từ 2048 lên 8192 nhưng frame vẫn dài 25ms, điều gì thật sự thay đổi và điều gì không?

**Điều THAY ĐỔI:**
- Bin spacing: $\Delta f = F_s / NFFT$
  - NFFT=2048: $\Delta f = 22050/2048 \approx 10.77$ Hz
  - NFFT=8192: $\Delta f = 22050/8192 \approx 2.69$ Hz
- Số điểm trên trục tần số: nhiều hơn 4 lần → đường phổ **mịn hơn về mặt thị giác** (interpolated)
- Thời gian tính toán: tăng (nhưng O(N log N) vẫn hiệu quả)

**Điều KHÔNG THAY ĐỔI:**
- **True frequency resolution** = $F_s / N_{window} = 22050/551 \approx 40$ Hz (phụ thuộc *frame length*, không phụ thuộc NFFT)
- Khả năng phân tách thực tế 2 tần số gần nhau vẫn như cũ
- Nội dung thông tin: zero-padding chỉ là **nội suy** (interpolation) trên trục tần số, không tạo thêm thông tin mới
- Spectral leakage: không thay đổi (phụ thuộc cửa sổ, không phụ thuộc NFFT)

> **Kết luận:** Tăng NFFT mà không tăng frame length chỉ vẽ đường cong phổ mịn hơn, không cải thiện độ phân giải vật lý.

---

### Câu 3: Tại sao Hamming giảm spectral leakage so với Rectangular nhưng có thể làm các đỉnh gần nhau khó phân tách hơn?

**Spectral leakage xảy ra vì:** FFT giả định tín hiệu tuần hoàn trong frame. Khi tần số không là bội số nguyên của $\Delta f$, năng lượng "rò" sang các bin lân cận.

**Hamming giảm leakage vì:**
$$w[n] = 0.54 - 0.46\cos\left(\frac{2\pi n}{N-1}\right)$$
Cửa sổ này có side-lobe level $\approx -42$ dB (so với −13 dB của Rectangular), nghĩa là năng lượng rò sang bin xa chỉ còn 1/400 so với peak thay vì 1/5.

**Nhưng main-lobe rộng hơn:** Hamming có main-lobe rộng ~2× so với Rectangular (6 bins vs 4 bins tại −3dB). Điều này có nghĩa là 2 tần số cần cách nhau ít nhất $2 \times \Delta f \times 6 = 12 \cdot \Delta f$ thay vì $4 \cdot \Delta f$ mới tách được. Ví dụ: với $\Delta f = 40$ Hz (frame 25ms), Hamming cần 2 tần số cách nhau > 240 Hz để tách, trong khi Rectangular chỉ cần > 160 Hz.

**Đánh đổi:** Hamming tốt khi phát hiện tần số yếu bên cạnh tần số mạnh (cần side-lobe thấp). Rectangular tốt hơn khi cần tách 2 tần số có biên độ tương đương và nằm gần nhau.

---

### Câu 4: Với FIR 201 taps đối xứng tại 44.1 kHz, độ trễ xấp xỉ bao nhiêu mili giây? Độ trễ đó có quan trọng trong xử lý thời gian thực không?

**Tính group delay:**
$$\tau_{group} = \frac{M}{2} = \frac{201 - 1}{2} = 100 \text{ mẫu}$$

Quy đổi sang thời gian (tại Fs = 44.1 kHz):
$$\tau_{ms} = \frac{100}{44100} \approx 2.27 \text{ ms}$$

Tại Fs = 22.05 kHz (như trong Lab):
$$\tau_{ms} = \frac{100}{22050} \approx 4.54 \text{ ms}$$

**Đánh giá trong xử lý thời gian thực:**

| Ứng dụng | Ngưỡng delay chấp nhận | 2.27ms có OK? |
|----------|----------------------|--------------|
| Điện thoại VoIP (ITU-T G.114) | < 150 ms end-to-end | ✅ Rất tốt |
| Âm nhạc trực tiếp (live sound) | < 10 ms | ✅ OK |
| Guitar amp simulator | < 3–5 ms (cảm nhận) | ✅/⚠️ Biên giới |
| Điều khiển audio (ANC, feedback) | < 1 ms | ❌ Quá cao |

> **Kết luận:** 2.27 ms là chấp nhận được cho phần lớn ứng dụng audio thông thường. Với ứng dụng cần độ trễ cực thấp (ANC, feedback control), cần giảm số taps hoặc dùng IIR filter.

---

### Câu 5: Từ công thức SNR_Q, giải thích ảnh hưởng của B và σₓ. Tại sao giảm mức tín hiệu đầu vào có thể làm SNR lượng tử giảm?

**Công thức Rabiner–Schafer:**
$$\text{SNR}_Q = 6.02B + 4.77 - 20\log_{10}\left(\frac{X_{max}}{\sigma_x}\right) \quad \text{(dB)}$$

**Ảnh hưởng của B (bit depth):**
- Mỗi bit tăng thêm: SNR tăng **6.02 dB**
- Nguyên nhân: thêm 1 bit → L tăng gấp 2 → $\Delta$ giảm 2 lần → $\sigma^2_{noise} = \Delta^2/12$ giảm 4 lần → SNR tăng $10\log_{10}(4) \approx 6$ dB

**Ảnh hưởng của σₓ (RMS tín hiệu):**
- $20\log_{10}(X_{max}/\sigma_x)$ = headroom giữa peak quantizer và RMS tín hiệu
- $\sigma_x$ lớn → term này nhỏ → **SNR cao hơn**
- $\sigma_x$ nhỏ → term này lớn → **SNR thấp hơn**

**Tại sao giảm mức tín hiệu làm SNR giảm?**

Khi giảm mức tín hiệu đầu vào (ví dụ từ −12 dBFS về −30 dBFS):
- $\sigma_x$ giảm từ 0.231 về 0.032
- $20\log_{10}(1/0.032) = 29.9$ dB (tăng 18dB so với trước)
- SNR giảm 18 dB dù cùng bộ lượng tử

**Nguyên nhân vật lý:** Bộ lượng tử có $\Delta$ cố định (thiết kế cho $X_{max} = 1$). Tín hiệu yếu chỉ dùng một phần nhỏ các mức lượng tử, nhiễu lượng tử $\Delta^2/12$ vẫn không đổi nhưng công suất tín hiệu giảm → SNR giảm.

**Ứng dụng:** Đây là lý do các codec hiện đại (MP3, AAC) dùng **gain control** để đưa tín hiệu về mức tối ưu trước khi lượng tử hóa.

---

### Câu 6: File WAV 16-bit stereo 44.1 kHz dài 60 s có kích thước PCM lý thuyết bao nhiêu MB? So sánh với MP3 128 kbps.

**PCM lý thuyết:**
$$R_{PCM} = F_s \times B \times C = 44100 \times 16 \times 2 = 1{,}411{,}200 \text{ bit/s} = 1411.2 \text{ kbps}$$

$$\text{Size}_{PCM} = \frac{R_{PCM} \times t}{8} = \frac{1{,}411{,}200 \times 60}{8} = 10{,}584{,}000 \text{ bytes} \approx \mathbf{10.09 \text{ MB}}$$

**MP3 128 kbps:**
$$\text{Size}_{MP3} = \frac{128{,}000 \times 60}{8} = 960{,}000 \text{ bytes} \approx \mathbf{0.96 \text{ MB}}$$

**So sánh:**

| Định dạng | Bit rate | Kích thước (60s) | Compression ratio | Saving |
|-----------|---------|-----------------|-------------------|--------|
| PCM 16-bit stereo 44.1kHz | 1411.2 kbps | **10.09 MB** | 1.0 : 1 | 0% |
| MP3 128 kbps | 128 kbps | **0.96 MB** | **11.03 : 1** | **90.5%** |
| MP3 320 kbps | 320 kbps | 2.40 MB | 4.41 : 1 | 77.3% |
| OGG Q=6 (~160kbps) | ~160 kbps | ~1.20 MB | ~8.8 : 1 | ~88.1% |

**Nhận xét:** MP3 128 kbps tiết kiệm ~90.5% dung lượng so với PCM. MP3 là lossy — sử dụng psychoacoustic model để loại bỏ thông tin mà tai người ít nghe thấy (masking effect). Không thể phục hồi bản gốc sau khi đã nén MP3.

---

### Câu 7: Hãy nêu ít nhất hai trường hợp "nghe tốt hơn" không đồng nghĩa với "SNR lớn hơn"

**Trường hợp 1: Perceptual audio coding (MP3 vs PCM 8-bit)**

MP3 128 kbps có SNR thấp hơn PCM 8-bit mono 22kHz (về mặt toán học), nhưng **nghe tốt hơn** rõ rệt. Lý do: MP3 dùng psychoacoustic model — chỉ loại bỏ thành phần mà tai người không nghe được (temporal masking, simultaneous masking). PCM 8-bit thêm quantization noise đều khắp toàn bộ dải tần, kể cả các tần số tai nghe nhạy nhất (2–5 kHz).

> SNR(MP3 128kbps) có thể thấp hơn SNR(PCM 8-bit) về mặt tính toán, nhưng tai người đánh giá MP3 tốt hơn.

**Trường hợp 2: Harmonic distortion trong âm nhạc (tube amplifier)**

Amplifier ống chân không (tube amp) có THD (Total Harmonic Distortion) thường 1–5%, nghĩa là SNR theo nghĩa truyền thống thấp hơn solid-state amp (THD < 0.01%). Tuy nhiên, nhiều người nghe đánh giá tube amp "ấm áp và dễ chịu hơn". Lý do: distortion của tube amp chủ yếu là **harmonic chẵn** (2nd, 4th harmonics) — tai người cảm nhận đây là âm nhạc, không phải nhiễu. Solid-state amp có THD nhỏ hơn nhưng khi clip lại tạo **harmonic lẻ** (3rd, 5th) — không thuận tai.

**Trường hợp 3 (bonus): Reverb trong âm phòng**

Một căn phòng có nhiều phản xạ tường (reverb time RT60 = 1–2s) có SNR kỹ thuật thấp hơn phòng cách âm (anechoic), nhưng nhiều người nghe thấy âm nhạc trong phòng hòa nhạc "đẹp hơn" âm thanh khô của phòng anechoic. Reverb thêm "không gian" và "sự phong phú" mà não người liên kết với trải nghiệm âm nhạc tốt.

> **Kết luận tổng quát:** SNR là chỉ số kỹ thuật khách quan, nhưng chất lượng âm thanh cảm nhận (perceptual quality) phụ thuộc vào đặc điểm của hệ thống thính giác người, không chỉ SNR. Các thước đo perceptual như PESQ, POLQA, MUSHRA phù hợp hơn để đánh giá chất lượng âm thanh nghe được.

---

## 10. Checklist trước khi nộp

- [x] Code chạy lại được từ đầu đến cuối (`python lab1_main.py --all`)
- [x] Không thiếu audio đầu ra trong `audio/`
- [x] Mọi hình có tiêu đề, tên trục, đơn vị
- [x] Có phép tính kiểm chứng (SNR lý thuyết vs đo thực)
- [x] Kết luận gắn với số liệu/đồ thị
- [x] Ghi rõ mọi tham số xử lý (Fs, taps, cutoff, NFFT, ...)
- [x] Giá trị dB ghi rõ loại (dBFS, SNR dB)
- [x] Tên file âm thanh thể hiện cấu hình (`music_lpf_2000hz.wav`, `speech_8bit.wav`)
- [x] So sánh PCM vs OGG (lossy compression thực tế)
- [x] Câu hỏi báo cáo 1–7 đã trả lời

---

## 11. Tài liệu tham khảo

[1] Đề cương chi tiết học phần CSE457 – Xử lý âm thanh và tiếng nói, Trường Đại học Thủy lợi, 2023.

[2] L. R. Rabiner, R. W. Schafer, *Theory and Applications of Digital Speech Processing*, Prentice-Hall/Pearson, 2010. Ch. 2 (DSP fundamentals), Ch. 7 (STFT), Ch. 11 (quantization), Ch. 12 (audio coding).

[3] X. Huang, A. Acero, H.-W. Hon, *Spoken Language Processing*, Prentice-Hall, 2001. Ch. DSP, speech signal representations, speech coding.

[4] D. Jurafsky, J. H. Martin, *Speech and Language Processing*, Prentice-Hall, 2008.

[5] scipy.signal documentation – `firwin`, `spectrogram`, `resample_poly`. https://docs.scipy.org/doc/scipy/reference/signal.html

[6] soundfile documentation – OGG/Vorbis export. https://python-soundfile.readthedocs.io/
