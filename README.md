# CSE457 – Xử lý âm thanh và tiếng nói
## Lab 1: Phân tích và xử lý tín hiệu âm thanh số

| Thông tin | Nội dung |
|-----------|----------|
| **Sinh viên** | Dương Tiến Đạt |
| **MSSV** | 2251172268 |
| **Học phần** | CSE457 – Xử lý âm thanh và tiếng nói |
| **Trường** | Đại học Thủy lợi (TLU) |

---

## Cấu trúc thư mục

```
Lab01_2251172268_DuongTienDat/
├── Lab01_2251172268.ipynb          ← Notebook chính (chạy từ đầu đến cuối)
├── report_Lab01.md                 ← Báo cáo đầy đủ (7 câu hỏi + kết quả)
├── requirements.txt                ← Thư viện cần cài
│
├── generate_audio.py               ← Tạo audio: TTS + giọng nói tổng hợp + hội thoại
├── lab1_utils.py                   ← Tiện ích dùng chung
├── lab1_part_a.py                  ← Phần A: Đọc và kiểm tra dữ liệu
├── lab1_part_b.py                  ← Phần B: Phân tích miền thời gian
├── lab1_part_c.py                  ← Phần C: FFT
├── lab1_part_d.py                  ← Phần D: STFT / Spectrogram
├── lab1_part_e.py                  ← Phần E: Thí nghiệm cửa sổ
├── lab1_part_f.py                  ← Phần F: Lọc số FIR
├── lab1_part_g.py                  ← Phần G: Lượng tử hóa, Resampling, Mã hóa
├── lab1_main.py                    ← Script chạy toàn bộ pipeline A → G
│
├── audio/
│   ├── input_music.wav             ← Tín hiệu nhạc đầu vào (hợp âm C trưởng)
│   ├── input_speech_male.wav       ← Giọng nói nam tổng hợp (F0~120Hz)
│   ├── input_speech_female.wav     ← Giọng nói nữ tổng hợp (F0~200Hz)
│   ├── input_dialogue.wav          ← Hội thoại 2 người (nam + nữ xen kẽ)
│   ├── filtered_music_lpf_2000hz.wav   ← Sau FIR low-pass 2kHz
│   ├── filtered_music_hpf_3000hz.wav   ← Sau FIR high-pass 3kHz
│   ├── filtered_music_bpf_500_3500hz.wav ← Sau FIR band-pass 500–3500Hz
│   ├── quantized_music_4bit.wav    ← Lượng tử hóa 4-bit (SNR≈16dB)
│   ├── quantized_music_8bit.wav    ← Lượng tử hóa 8-bit (SNR≈40dB)
│   ├── quantized_music_16bit.wav   ← Lượng tử hóa 16-bit (SNR≈91dB)
│   ├── resampled_music_16000hz.wav ← Downsample 22050→16000Hz
│   ├── resampled_music_8000hz.wav  ← Downsample 22050→8000Hz
│   ├── compressed_music_ogg_q1.ogg ← OGG Vorbis Q=1 (nén mạnh, ~7.7:1)
│   └── compressed_music_ogg_q9.ogg ← OGG Vorbis Q=9 (chất lượng cao)
│
└── figures/
    ├── waveform.png                ← Waveform toàn tệp + Short-Time Energy
    ├── fft.png                     ← Phổ FFT (tuyến tính + dB + đỉnh)
    ├── spectrogram.png             ← Spectrogram 25ms/10ms
    ├── filter_response.png         ← Đáp ứng tần số FIR (mag + phase + group delay)
    ├── A_*.png                     ← Kết quả Phần A
    ├── B_*.png                     ← Kết quả Phần B
    ├── C_*.png                     ← Kết quả Phần C
    ├── D_*.png                     ← Kết quả Phần D
    ├── E_*.png                     ← Kết quả Phần E
    ├── F_*.png                     ← Kết quả Phần F
    └── G_*.png                     ← Kết quả Phần G
```

---

## Cách chạy

### Cài thư viện
```bash
pip install -r requirements.txt
```

### Tạo file âm thanh mẫu
```bash
python generate_audio.py
# Chọn [3] để tạo tín hiệu tổng hợp (không cần internet)
# Chọn [2] để tạo hội thoại TTS (cần pyttsx3)
```

### Chạy toàn bộ pipeline A → G
```bash
python lab1_main.py --all
# hoặc chạy từng phần:
python lab1_main.py --part C
```

### Chạy Notebook
Mở `Lab01_2251172268.ipynb` trong VS Code hoặc Jupyter, đổi `CHOICE` để chọn file âm thanh, chạy **Run All**.

---

## Nội dung Lab

| Phần | Chủ đề | Kết quả chính |
|------|--------|--------------|
| **A** | Đọc metadata, chuẩn hóa | Fs, Nyquist, PCM bitrate, Peak/RMS dBFS |
| **B** | Time-domain: Peak, RMS, Energy, ZCR, STE | Waveform + Short-Time Energy, so sánh 2 đoạn |
| **C** | FFT, phổ biên độ/dB, so sánh NFFT | True resolution = Fs/N_window ≠ Δf = Fs/NFFT |
| **D** | STFT/Spectrogram, trade-off time/freq | 3 frame length 10/25/50ms + transient detection |
| **E** | Cửa sổ Rectangular vs Hamming | Main-lobe, side-lobe, spectral leakage |
| **F** | FIR LP/HP/BP, group delay compensation | H(f) + phổ trước/sau + WAV |
| **G** | Quantization SNR, Resampling, OGG coding | SNR 6dB/bit, anti-alias resample, ratio ~7.7:1 |

---

## Câu hỏi báo cáo (tóm tắt)

1. **Nyquist**: $F_s = 44.1$ kHz → biểu diễn đến $F_s/2 = 22.05$ kHz
2. **NFFT tăng, frame cố định**: Δf nhỏ hơn (phổ mịn) nhưng true resolution không đổi
3. **Hamming vs Rectangular**: Side-lobe −42dB thấp hơn nhưng main-lobe rộng hơn ~2×
4. **FIR 201 taps tại 44.1kHz**: delay = 100 mẫu ≈ **2.27 ms**
5. **SNR_Q giảm khi σ_x giảm**: $\text{SNR}_Q = 6.02B + 4.77 - 20\log_{10}(X_{max}/\sigma_x)$
6. **WAV 60s CD quality**: $(44100×16×2×60)/8 =$ **10.09 MB** vs MP3 128kbps = 0.96 MB
7. **Nghe tốt ≠ SNR lớn**: (1) MP3 128kbps vs PCM 8-bit; (2) Tube amp THD chẵn vs solid-state

---

## Tài liệu tham khảo

- Rabiner & Schafer, *Theory and Applications of Digital Speech Processing*, Pearson, 2010
- Huang, Acero & Hon, *Spoken Language Processing*, Prentice-Hall, 2001
- Đề cương CSE457, Trường Đại học Thủy lợi, 2023
