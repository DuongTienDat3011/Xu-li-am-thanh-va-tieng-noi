# CSE457 – XỬ LÝ ÂM THANH VÀ TIẾNG NÓI
## LAB 5: SPEECH ENHANCEMENT, SPEAKER RECOGNITION AND SPEAKER SEPARATION
### Từ Khử nhiễu STFT đến Xác định người nói và Tách hai nguồn nói bằng FastICA

---

| Thông tin | Chi tiết |
| :--- | :--- |
| **Sinh viên thực hiện** | **Dương Tiến Đạt** |
| **Mã số sinh viên (MSSV)** | **2251172268** |
| **Lớp học phần** | CSE457 – Xử lý âm thanh và tiếng nói |
| **Bộ môn** | Trí tuệ Nhân tạo – Khoa Công nghệ Thông tin |
| **Trường** | Trường Đại học Thủy lợi (TLU) |
| **Case Study** | Trợ lý phòng họp mini (Mini Meeting Assistant) |

---

## 1. Cấu trúc thư mục Lab 5

```
Lab5_2251172268_DuongTienDat/
├── Lab5_2251172268.ipynb          ← Notebook Jupyter hoàn chỉnh (đã chạy sẵn kết quả & audio player)
├── report_Lab5.md                 ← Báo cáo chi tiết định dạng Markdown (đầy đủ 12 câu hỏi & kết quả)
├── report_Lab5.docx               ← Báo cáo định dạng Word chuẩn học thuật (chèn biểu đồ, bảng & công thức)
├── README.md                      ← Hướng dẫn cài đặt, cấu trúc và lệnh chạy
├── requirements.txt               ← Danh sách thư viện Python cần thiết
│
├── generate_dataset.py            ← Tạo tập dữ liệu 21 file clean speech, 2 file noise, 2 separation sources
├── lab5_utils.py                  ← Hàm tiện ích: I/O 16kHz, RMS, Power, SNR, delay alignment
├── lab5_enhancement.py            ← Thuật toán Spectral Subtraction & Wiener Filtering
├── lab5_speaker_id.py             ← Trích xuất MFCC, Delta, Embedding, Cosine ID & Verification (FAR/FRR)
├── lab5_ica.py                    ← Mô hình Instantaneous Mixture 2x2, FastICA & xử lý Permutation/Sign
├── lab5_main.py                   ← Script chạy toàn bộ pipeline A → Z và xuất tất cả biểu đồ/audio
├── build_notebook_lab5.py         ← Script tự động biên dịch và tạo ra Lab5_2251172268.ipynb
├── generate_report_docx.py        ← Script tự động biên soạn và định dạng report_Lab5.docx
│
├── lab5_data/                     ← Dữ liệu âm thanh thực nghiệm (WAV PCM 16kHz Mono)
│   ├── clean/                     ← 3 Người nói (spk01, spk02, spk03): mỗi người 4 enroll, 1 dev, 2 test
│   ├── noise/                     ← fan.wav (nhiễu dừng) và cafeteria.wav (nhiễu babble)
│   └── separation/                ← source_A.wav và source_B.wav (cùng độ dài 84457 mẫu = 5.28s)
│
└── results/                       ← Kết quả đầu ra sau khi chạy
    ├── audio/                     ← Các tệp âm thanh đầu ra:
    │   ├── noisy_0dB.wav, noisy_5dB.wav, noisy_10dB.wav
    │   ├── enhanced_ss_0dB.wav, enhanced_ss_5dB.wav, enhanced_ss_10dB.wav
    │   ├── enhanced_wiener_0dB.wav, enhanced_wiener_5dB.wav, enhanced_wiener_10dB.wav
    │   ├── enhanced_ss_beta1.0.wav, enhanced_ss_beta1.5.wav, enhanced_ss_beta2.0.wav
    │   ├── mix_mic1.wav, mix_mic2.wav
    │   ├── separated_1.wav, separated_2.wav
    │   └── separated_ill_1.wav, separated_ill_2.wav
    │
    └── figures/                   ← Các biểu đồ phân tích khoa học chất lượng cao:
        ├── fig1_clean_noisy_enhanced_spectrograms.png
        ├── fig2_snr_comparison.png
        ├── fig5_confusion_matrix_speaker_id.png
        ├── fig7_speaker_verification_far_frr.png
        ├── fig8_ica_separation_waveforms_spectrograms.png
        ├── fig9_ica_correlation_matrices.png
        └── fig10_integrated_case_study.png
```

---

## 2. Hướng dẫn cài đặt và Chạy thực nghiệm

### Bước 1: Cài đặt thư viện môi trường
```bash
pip install -r requirements.txt
```

### Bước 2: Tạo tập dữ liệu âm thanh (Speech, Noise, Separation)
```bash
python generate_dataset.py
```
*Lệnh trên sẽ tự động tổng hợp 21 file tiếng nói sạch cho 3 người nói, 2 file nhiễu và 2 nguồn BSS chuẩn 16 kHz.*

### Bước 3: Chạy toàn bộ quy trình thực nghiệm Lab 5
```bash
python lab5_main.py
```
*Script sẽ chạy tuần tự:*
1. Khử nhiễu STFT (Spectral Subtraction vs Wiener) trên các mức SNR 0, 5, 10 dB.
2. Khảo sát siêu tham số $\beta$ và $\gamma$, đo lường $\Delta SNR$ và xuất phổ spectrogram.
3. Nhận dạng người nói (Identification & Verification), thực hiện ablation MFCC/Delta/CMN/Tokens.
4. Tách 2 nguồn nói bằng FastICA, tự động khử bất định hoán vị và dấu bằng ma trận $|\rho|$.
5. Tích hợp Case study "Trợ lý phòng họp mini".

### Bước 4: Tạo Notebook và Báo cáo Word
```bash
python build_notebook_lab5.py
python generate_report_docx.py
```

---

## 3. Tóm tắt kết quả kỹ thuật cốt lõi

### 1. Khử nhiễu tiếng nói (Speech Enhancement)
- **Spectral Subtraction ($\beta=1.5, \gamma=0.02$):** Tại mức nhiễu nặng $SNR_{in} = 0.0\text{ dB}$, thuật toán cải thiện vượt bậc $\Delta SNR = +3.25\text{ dB}$ (đạt $3.25\text{ dB}$). Đặt sàn $\gamma = 0.02$ giúp triệt tiêu hiện tượng musical noise hiệu quả.
- **Wiener Filter:** Cho đáp ứng suy hao mềm mại, âm thanh tự nhiên không bị artifact ríu rít, đạt $\Delta SNR = +2.72\text{ dB}$ tại $0\text{ dB}$.

### 2. Nhận dạng người nói (Speaker Recognition)
- **Baseline (MFCC+Delta, 4 tokens):** Độ chính xác nhận dạng đạt $66.7\%$.
- **Ablation với CMN (Cepstral Mean Normalization):** Độ chính xác tăng vọt lên **$100.0\%$**, chứng minh CMN loại bỏ méo tĩnh và tăng cường độ phân tách đặc trưng người nói.
- **Speaker Verification:** Quét ngưỡng trên Development set đạt điểm cân bằng $EER$ tại $\theta^* = 0.999898$. Khóa ngưỡng này sang Test set độc lập cho $FAR_{test} = 16.67\%$, $FRR_{test} = 33.33\%$.

### 3. Phân tách nguồn mù (Blind Source Separation)
- Ma trận trộn chuẩn $\mathbf{A} = \begin{bmatrix} 1.0 & 0.6 \\ 0.4 & 1.0 \end{bmatrix}$ có $\det(\mathbf{A}) = 0.76 \ne 0$.
- Thuật toán FastICA phân tách hoàn hảo với hệ số tương quan sau khử bất định đạt $|\rho| = 1.0000$ trên cả 2 nguồn.
- Khi thêm nhiễu vào hỗn hợp micro (10 dB), tương quan trung bình giảm xuống $0.8900$, chứng minh BSS nhạy cảm với giả thiết số lượng nguồn độc lập.

### 4. Case Study Tích hợp
- Đã chạy thông suốt pipeline: Âm thanh nhiễu $5\text{ dB} \rightarrow$ Khử nhiễu STFT $\rightarrow$ Nhận dạng người nói $\rightarrow$ Xác thực quyền $\rightarrow$ Tách 2 luồng hội thoại độc lập.
- **Phát hiện quan trọng:** Enhancement cải thiện SNR và độ nghe rõ, nhưng Wiener filter có thể làm giảm điểm số nhận dạng người nói do làm trơn các chi tiết formant vi mô mang tính sinh trắc học.
