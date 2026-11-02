"""
build_notebook_lab4.py
======================
Tạo file Jupyter Notebook hoàn chỉnh và tự động thực thi: Lab4_2251172268.ipynb.
Sinh viên: Dương Tiến Đạt - MSSV: 2251172268
"""

import json
import os
import sys

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
NB_PATH = os.path.join(BASE_DIR, "Lab4_2251172268.ipynb")

def md(*lines):
    return {"cell_type": "markdown", "metadata": {}, "source": list(lines)}

def code(*lines, outputs=None):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": outputs or [], "source": list(lines)}

def build():
    cells = []
    
    # ─── HEADER & THÔNG TIN SINH VIÊN ──────────────────────────────────────
    cells.append(md(
        "# CSE457 – XỬ LÝ ÂM THANH VÀ TIẾNG NÓI\n",
        "## BÁO CÁO THỰC HÀNH LAB 4: TỔNG HỢP TIẾNG NÓI (TEXT-TO-SPEECH)\n",
        "### Từ Ghép nối (Concatenative) & Unit Selection đến Tham số HMM và TTS Học sâu (Neural TTS)\n",
        "---\n",
        "| Thông tin | Chi tiết |\n",
        "| :--- | :--- |\n",
        "| **Họ và tên sinh viên** | **Dương Tiến Đạt** |\n",
        "| **Mã số sinh viên (MSSV)** | **2251172268** |\n",
        "| **Lớp học phần** | CSE457 – Xử lý âm thanh và tiếng nói |\n",
        "| **Case Study** | Hệ thống thông báo xếp hàng miền hẹp (Queue Calling TTS) |\n",
        "| **Yêu cầu giọng nói** | Chuyển đổi văn bản sang tiếng nói **siêu tự nhiên**, mượt mà, đúng thanh điệu |\n",
        "\n",
        "**Kiến trúc tổng thể của hệ thống Text-to-Speech:**\n",
        "```\n",
        "Text Input ──> Text Normalization ──> Phonetic / Unit Selection ──> Prosody / Acoustic Backend ──> Waveform Vocoder\n",
        "```\n",
        "> Notebook này được thiết kế để có thể **chạy lại 100% tự động từ đầu đến cuối**, tích hợp sẵn Audio Player và biểu đồ khoa học trực quan.\n"
    ))
    
    # ─── PHẦN 0: CÀI ĐẶT VÀ IMPORT ──────────────────────────────────────────
    cells.append(md("## 0. Cài đặt môi trường và Thư viện cần thiết"))
    cells.append(code(
        "import os, sys, time, math, glob\n",
        "import numpy as np\n",
        "import pandas as pd\n",
        "import soundfile as sf\n",
        "import librosa\n",
        "import librosa.display\n",
        "import matplotlib.pyplot as plt\n",
        "from IPython.display import Audio, display, Markdown\n",
        "\n",
        "from lab4_text_norm import normalize_queue_text, tokens_to_unit_ids\n",
        "from lab4_utils import load_audio, match_rms, calc_rms, compute_log_spec, plot_waveform_and_spectrogram\n",
        "from lab4_concat import synthesize_concatenative, get_unit_path\n",
        "from lab4_prosody import estimate_f0_autocorr, track_f0_contour\n",
        "from lab4_hmm import generate_trajectories, synthesize_source_filter_vocoder\n",
        "from lab4_neural import synthesize_neural\n",
        "\n",
        "print('✓ Tất cả các thư viện đã được nạp thành công!')\n"
    ))
    
    # ─── PHẦN 1: INVENTORY & TEXT NORMALIZATION ─────────────────────────────
    cells.append(md(
        "## 1. Chuẩn hóa văn bản và Khởi tạo Unit Inventory (Phần A)\n",
        "\n",
        "### 1.1 Nguyên lý Text Normalization trong miền hẹp\n",
        "Văn bản đầu vào của hệ thống xếp hàng gồm chữ số và từ ngữ viết hoa/thường:  \n",
        "- `\"Mời khách hàng số 23 đến quầy 5.\"` -> `['mời', 'khách', 'hàng', 'số', 'hai', 'ba', 'đến', 'quầy', 'năm']`\n",
        "- Các chữ số được tách và đọc tuần tự theo số đếm chuẩn tiếng Việt.\n",
        "\n",
        "### 1.2 Thiết kế Unit Inventory\n",
        "- Tổng cộng 16 từ vựng cơ sở: `mời`, `khách`, `hàng`, `số`, `không` đến `chín`, `đến`, `quầy`.\n",
        "- Mỗi từ có **2 candidates** (`_01` và `_02`) với sắc thái và tốc độ khác nhau để kiểm thử Unit Selection.\n",
        "- Đều được chuẩn hóa ở $F_s = 16,000$ Hz, mono, 16-bit PCM và $RMS = 0.08$.\n"
    ))
    cells.append(code(
        "meta_df = pd.read_csv('metadata.csv')\n",
        "print(f'Tổng số unit trong inventory: {len(meta_df)} files WAV')\n",
        "display(meta_df.head(10))\n",
        "\n",
        "# Hiển thị biểu đồ phân bố thống kê của Inventory\n",
        "if os.path.exists('figures/fig1_inventory_stats.png'):\n",
        "    from IPython.display import Image\n",
        "    display(Image('figures/fig1_inventory_stats.png'))\n"
    ))
    
    # ─── PHẦN 2: BASELINE CONCATENATION & UNIT SELECTION ────────────────────
    cells.append(md(
        "## 2. Ghép nối Baseline & Lựa chọn đơn vị tối ưu (Phần B & C)\n",
        "\n",
        "### 2.1 Baseline Concatenation (Ghép thẳng không cross-fade)\n",
        "Ghép trực tiếp candidate cố định `_01` cho từng token mà không xử lý biên nối ($xfade\\_ms = 0$).  \n",
        "Quan sát hiện tượng click, chênh lệch năng lượng và đứt gãy pha tại ranh giới các từ.\n",
        "\n",
        "### 2.2 Unit Selection bằng Quy hoạch động (Viterbi DP)\n",
        "Hàm mục tiêu tối ưu hóa toàn cục:  \n",
        "$$\\theta^* = \\arg\\min_{\\theta_1, \\dots, \\theta_K} \\left[ \\sum_{j=1}^K C_{target}(\\theta_j, t_j) + \\lambda \\sum_{j=1}^{K-1} C_{join}(\\theta_j, \\theta_{j+1}) \\right]$$\n",
        "\n",
        "Trong đó $C_{join}$ đo độ lệch phổ Log Spectral Distance (LSD) ở 25 ms tiếp giáp:  \n",
        "$$D_{spec} = \\frac{1}{K} \\sum_{k} \\left( L_{left}[k] - L_{right}[k] \\right)^2 + \\beta |RMS_{left} - RMS_{right}|$$\n",
        "\n",
        "### 2.3 Khử click biên bằng Overlap-Add / Cross-fade\n",
        "$$y[n] = (1 - \\alpha[n]) x_1[n] + \\alpha[n] x_2[n], \\quad 0 \\le n < N$$\n"
    ))
    cells.append(code(
        "# Đọc bảng phân tích chi phí Join/Target cost\n",
        "cost_df = pd.read_csv('cost_analysis.csv')\n",
        "display(cost_df)\n",
        "\n",
        "print('--- NGHE THỬ AUDIO SO SÁNH GIỮA BASELINE VÀ TỐI ƯU UNIT SELECTION ---')\n",
        "print('1. Baseline Concatenation (Ghép thẳng thô - xfade = 0ms):')\n",
        "display(Audio('audio_output/raw_concat.wav'))\n",
        "\n",
        "print('2. Best Concatenation (Unit Selection Viterbi + Cross-fade 15ms):')\n",
        "display(Audio('audio_output/best_concat.wav'))\n"
    ))
    cells.append(code(
        "# Hiển thị hình ảnh Waveform + Spectrogram đánh dấu biên và so sánh Cross-fade\n",
        "from IPython.display import Image\n",
        "if os.path.exists('figures/fig2_baseline_boundary.png'):\n",
        "    display(Markdown('### Biểu đồ Waveform và Spectrogram đánh dấu biên ghép nối:'))\n",
        "    display(Image('figures/fig2_baseline_boundary.png'))\n",
        "\n",
        "if os.path.exists('figures/fig3_crossfade_comparison.png'):\n",
        "    display(Markdown('### Cận cảnh sóng âm tại biên ghép nối với các mức Cross-fade (0, 5, 15, 30 ms):'))\n",
        "    display(Image('figures/fig3_crossfade_comparison.png'))\n",
        "\n",
        "if os.path.exists('figures/fig4_unit_selection_lattice.png'):\n",
        "    display(Markdown('### Mạng lưới tìm đường tối ưu Viterbi (Unit Selection Lattice):'))\n",
        "    display(Image('figures/fig4_unit_selection_lattice.png'))\n"
    ))
    
    # ─── PHẦN 3: PROSODY & SOURCE-FILTER ───────────────────────────────────
    cells.append(md(
        "## 3. Ngữ điệu (Prosody) & Mô hình Nguồn - Lọc Source-Filter (Phần D)\n",
        "\n",
        "### 3.1 Đo tần số cơ bản F0 và chu kỳ pitch $N_p$\n",
        "- Công thức lý thuyết: $F_0 = \\frac{F_s}{N_p}$\n",
        "- Với $F_s = 16,000$ Hz, chu kỳ pitch tìm được $N_p = 78$ mẫu $\\implies F_0 = 16,000 / 78 = 205.13$ Hz.\n",
        "\n",
        "### 3.2 Time-stretch và Pitch Shift độc lập\n",
        "- Biến đổi tốc độ (Speed $0.8\\times, 1.2\\times$) không làm thay đổi cao độ $F_0$.\n",
        "- Biến đổi cao độ (Pitch Shift $\\pm 2$ semitones với $r = 2^{\\Delta s / 12}$) giữ nguyên độ dài thời gian và hình dạng bao phổ formant $H(z)$.\n"
    ))
    cells.append(code(
        "print('--- NGHE THỬ CÁC BIẾN THỂ BIẾN ĐIỆU NGỮ ĐIỆU (PROSODY) ---')\n",
        "print('1. Tốc độ chậm (Speed 0.8x):')\n",
        "display(Audio('audio_output/speed_0.8x.wav'))\n",
        "\n",
        "print('2. Tốc độ nhanh (Speed 1.2x):')\n",
        "display(Audio('audio_output/speed_1.2x.wav'))\n",
        "\n",
        "print('3. Tăng cao độ (+2 Semitones):')\n",
        "display(Audio('audio_output/pitch_plus2.wav'))\n",
        "\n",
        "print('4. Giảm cao độ (-2 Semitones):')\n",
        "display(Audio('audio_output/pitch_minus2.wav'))\n",
        "\n",
        "if os.path.exists('figures/fig5_prosody_pitch_f0.png'):\n",
        "    display(Image('figures/fig5_prosody_pitch_f0.png'))\n",
        "if os.path.exists('figures/fig6_prosody_speed_pitch_spectrogram.png'):\n",
        "    display(Image('figures/fig6_prosody_speed_pitch_spectrogram.png'))\n"
    ))
    
    # ─── PHẦN 4: HMM-BASED PARAMETRIC SYNTHESIS ─────────────────────────────
    cells.append(md(
        "## 4. Tổng hợp tham số dựa trên HMM (HMM-based Synthesis - Phần E)\n",
        "\n",
        "### 4.1 Cơ chế Generation trong HMM\n",
        "- Không dùng để nhận dạng mà sinh chuỗi quan sát $\\hat{O}^* = \\arg\\max_O P(O | \\lambda, \\text{state sequence})$.\n",
        "- Nhờ bổ sung đặc trưng động $\\Delta$ và $\\Delta^2$, nghiệm tham số trở nên trơn tru liên tục (MLPG).\n",
        "- Tái tạo âm thanh qua Vocoder nguồn-lọc (xung glottal + formant filters).\n"
    ))
    cells.append(code(
        "print('--- NGHE THỬ HMM PARAMETRIC SYNTHESIS ---')\n",
        "print('1. HMM TTS Gốc:')\n",
        "display(Audio('audio_output/hmm_tts.wav'))\n",
        "\n",
        "print('2. HMM Biến điệu F0 (Pitch 1.1x):')\n",
        "display(Audio('audio_output/hmm_pitch_1.1x.wav'))\n",
        "\n",
        "if os.path.exists('figures/fig7_hmm_trajectories.png'):\n",
        "    display(Image('figures/fig7_hmm_trajectories.png'))\n"
    ))
    
    # ─── PHẦN 5: NEURAL TTS HỌC SÂU ─────────────────────────────────────────
    cells.append(md(
        "## 5. Tổng hợp tiếng nói học sâu siêu tự nhiên (Neural TTS - Phần F)\n",
        "\n",
        "### 5.1 Kiến trúc Text -> Acoustic Model -> Neural Vocoder\n",
        "- Sử dụng mô hình Neural TTS chất lượng cao cho tiếng Việt với ngữ điệu tự nhiên như người thật.\n",
        "- Phân tích biểu diễn phổ Mel 80 bins và đo đạc độ trễ Latency/RTF.\n"
    ))
    cells.append(code(
        "print('--- NGHE THỬ 3 CÂU KIỂM THỬ BẰNG NEURAL TTS ---')\n",
        "print('Câu 1: Mời khách hàng số 23 đến quầy 5.')\n",
        "display(Audio('audio_output/neural_tts.wav'))\n",
        "\n",
        "print('Câu 2: Mời khách hàng số 107 đến quầy 2.')\n",
        "display(Audio('audio_output/neural_tts_sentence2.wav'))\n",
        "\n",
        "print('Câu 3: Mời khách hàng số 496 đến quầy 8.')\n",
        "display(Audio('audio_output/neural_tts_sentence3.wav'))\n",
        "\n",
        "if os.path.exists('figures/fig8_neural_mel_spectrogram.png'):\n",
        "    display(Image('figures/fig8_neural_mel_spectrogram.png'))\n"
    ))
    
    # ─── PHẦN 6: ĐÁNH GIÁ & SO SÁNH 3 PHƯƠNG PHÁP ───────────────────────────
    cells.append(md(
        "## 6. Đánh giá chất lượng và So sánh 3 phương pháp (Phần G)\n",
        "\n",
        "### 6.1 Bảng so sánh toàn diện 4 hệ thống\n",
        "- **Concatenation Baseline (Raw)**\n",
        "- **Concatenation Tối ưu (Unit Selection + 15ms xfade)**\n",
        "- **HMM-based Parametric Synthesis**\n",
        "- **Neural TTS**\n"
    ))
    cells.append(code(
        "ratings_df = pd.read_csv('ratings.csv')\n",
        "print(f'Dữ liệu Listening Test ({len(ratings_df)} mẫu đánh giá từ 8 thính giả):')\n",
        "summary = ratings_df.groupby('method').agg(\n",
        "    MOS_Naturalness=('naturalness_mos', 'mean'),\n",
        "    MOS_Continuity=('continuity_score', 'mean'),\n",
        "    Intelligibility_Pct=('intelligibility', lambda x: float(np.mean(x) * 100)),\n",
        "    Overall_Score=('overall_score', 'mean')\n",
        ").reset_index()\n",
        "display(summary)\n",
        "\n",
        "if os.path.exists('figures/fig9_method_comparison.png'):\n",
        "    display(Image('figures/fig9_method_comparison.png'))\n"
    ))
    
    # ─── PHẦN 7: TRẢ LỜI 10 CÂU HỎI BÁO CÁO ─────────────────────────────────
    cells.append(md(
        "## 7. Giải đáp chi tiết 10 câu hỏi báo cáo trong Lab 4\n",
        "\n",
        "### Câu 1: Vì sao một câu ghép từ các đoạn \"tự nhiên\" vẫn có thể nghe kém tự nhiên?\n",
        "**Trả lời:** Mặc dù từng unit là âm thanh tự nhiên của người thật, khi ghép lại câu nói vẫn kém tự nhiên vì:\n",
        "1. **Thiếu ngữ điệu toàn câu (Prosody Coarticulation):** Con người khi nói cả câu luôn có đường bao cao độ $F_0$, độ ngân duration và trọng âm biến thiên liên tục theo ngữ nghĩa. Từng từ ghi riêng lẻ mang ngữ điệu rời rạc, làm câu nói bị giật và vô hồn.\n",
        "2. **Bất liên tục âm học ở ranh giới biên (Boundary Discontinuity):** Giữa các điểm nối có sự chênh lệch đột ngột về pha, biên độ RMS và phổ tần số ($F_1, F_2$), gây ra tiếng click/pop hoặc cảm giác chắp vá.\n",
        "\n",
        "### Câu 2: Phân biệt target cost và join cost. Nếu chỉ tối ưu join cost thì có thể xảy ra vấn đề gì?\n",
        "**Trả lời:**\n",
        "- **Target Cost $C_{target}(\\theta_j, t_j)$:** Đo mức độ phù hợp giữa candidate trong kho và unit mục tiêu cần phát âm (về ngữ âm, ngữ cảnh lân cận, duration, F0 mong muốn).\n",
        "- **Join Cost $C_{join}(\\theta_j, \\theta_{j+1})$:** Đo độ êm mượt và tương thích tại ranh giới chuyển tiếp giữa điểm kết thúc của unit trước và điểm khởi đầu của unit sau (Log Spectral Distance, chênh lệch RMS, F0).\n",
        "- **Hậu quả nếu chỉ tối ưu Join Cost:** Hệ thống có thể chọn một chuỗi các unit nối rất êm với nhau (do phổ tiếp giáp gần nhau) nhưng lại hoàn toàn sai lệch về cao độ, trường độ hoặc sắc thái biểu cảm của câu nói mục tiêu, tạo nên giọng nói bị biến dạng ngữ điệu kỳ dị.\n",
        "\n",
        "### Câu 3: Vì sao cross-fade quá dài có thể làm giảm độ rõ của phụ âm?\n",
        "**Trả lời:** Cross-fade làm mờ dần tín hiệu của đoạn trước và tăng dần tín hiệu của đoạn sau trong vùng chồng lấn. Các phụ âm (đặc biệt là phụ âm tắc plosives như /t/, /k/, /p/ và phụ âm xát fricatives) diễn ra cực kỳ nhanh (10 – 30 ms) với các xung năng lượng đột ngột (burst/transient). Nếu cross-fade quá dài (>30 ms), quá trình cộng tuyến tính sẽ làm suy giảm đỉnh xung đột ngột, triệt tiêu pha hoặc tạo tiếng vang (smearing/ghosting), làm âm nghe bị nhòe và mất đi sự sắc nét của phụ âm.\n",
        "\n",
        "### Câu 4: Giải thích bằng source-filter model vì sao thay F0 không đồng nghĩa với thay formant.\n",
        "**Trả lời:**\n",
        "- Theo mô hình nguồn-lọc $X(z) = E(z) \\cdot H(z)$:\n",
        "  - **Nguồn kích thích $E(z)$:** Được quyết định bởi sự rung của dây thanh đới. Tần số rung này chính là $F_0 = F_s / N_p$, tạo ra chuỗi các họa âm cách nhau một khoảng $F_0$.\n",
        "  - **Bộ lọc ống thanh âm $H(z)$:** Được quyết định bởi hình dạng khoang miệng, họng, lưỡi và môi. Các tần số cộng hưởng của ống thanh âm chính là các **Formant** ($F_1, F_2, F_3$) quyết định âm sắc của nguyên âm.\n",
        "- Khi một người thay đổi cao độ nói ($F_0$), các vạch họa âm trong nguồn $E(z)$ dãn ra hoặc co lại, nhưng hình dạng khoang miệng $H(z)$ vẫn giữ nguyên, do đó vị trí các đỉnh Formant không thay đổi.\n",
        "\n",
        "### Câu 5: PSOLA thay pitch bằng cách nào? Tại sao voiced fricative có thể bị buzziness khi kéo dài quá mức?\n",
        "**Trả lời:**\n",
        "- **Cơ chế PSOLA (Pitch-Synchronous Overlap and Add):** Phân tích tín hiệu thành các đoạn ngắn bao quanh từng chu kỳ pitch (epoch) bằng cửa sổ Hanning độ dài $2 T_0$. Để thay đổi pitch sang chu kỳ mới $T_{new}$, các đoạn này được dịch chuyển sao cho khoảng cách giữa chúng bằng $T_{new}$ rồi cộng chồng lấn lại (Overlap-Add).\n",
        "- **Hiện tượng Buzziness trên phụ âm xát hữu thanh (Voiced Fricative):** Voiced fricative chứa đồng thời hai thành phần: dao động tuần hoàn của dây thanh đới và nhiễu hỗn loạn ngẫu nhiên do dòng khí qua khe hẹp. Khi kéo dài quá mức bằng cách lặp lại nhiều lần các đoạn có cửa sổ, thành phần nhiễu ngẫu nhiên bị nhân bản tuần hoàn giả tạo, làm mất tính ngẫu nhiên tự nhiên và biến thành âm thanh rè kim loại (buzzy/robotic artifact).\n",
        "\n",
        "### Câu 6: Trong HMM-based synthesis, vì sao thêm delta feature làm trajectory trơn hơn?\n",
        "**Trả lời:** Nếu chỉ sử dụng đặc trưng tĩnh (static features $c_t$), tại mỗi trạng thái HMM, vector cực đại hợp lý $\\hat{c}_t$ sẽ là giá trị trung bình $\\mu$ của trạng thái đó, dẫn đến quỹ đạo tham số là các hàm bậc thang giật cục khi nhảy từ state này sang state khác. Khi bổ sung thêm đặc trưng biến thiên động $\\Delta c_t = c_{t+1} - c_{t-1}$ và $\\Delta^2 c_t$, thuật toán MLPG tối ưu hóa đồng thời xác suất của cả static và dynamic: $\\hat{O}^* = \\arg\\max_O P([c, \\Delta c, \\Delta^2 c]^T | \\lambda)$. Ràng buộc vận tốc và gia tốc này bắt buộc quỹ đạo các điểm tham số phải trơn tru, triệt tiêu hoàn toàn sự gián đoạn giữa các state.\n",
        "\n",
        "### Câu 7: Tại sao whole waveform concatenation cần inventory lớn để đạt naturalness cao?\n",
        "**Trả lời:** Whole waveform concatenation sử dụng nguyên vẹn các đoạn sóng âm tự nhiên mà không qua bộ lọc tham số. Do đó, để phát âm tự nhiên trong mọi ngữ cảnh, kho dữ liệu phải chứa đầy đủ mọi biến thể âm học: sự biến đổi theo âm đứng trước và đứng sau (triphone/diphone coarticulation), mọi thanh điệu, trọng âm, tốc độ và ngữ điệu câu. Nếu inventory nhỏ, hệ thống buộc phải chọn các unit không khớp ngữ cảnh, dẫn đến chênh lệch phổ và cao độ nghiêm trọng tại biên nối.\n",
        "\n",
        "### Câu 8: Nêu ít nhất hai khác biệt cơ bản giữa concatenative TTS và neural TTS.\n",
        "**Trả lời:**\n",
        "1. **Bản chất tạo dạng sóng:** Concatenative TTS lựa chọn và ghép các đoạn âm thanh tự nhiên có sẵn trong kho lưu trữ, trong khi Neural TTS sinh trực tiếp dạng sóng mẫu từng mẫu (hoặc qua Mel-spectrogram rồi Neural Vocoder như HiFi-GAN/WaveNet) bằng các mạng nơ-ron sâu được học từ dữ liệu lớn.\n",
        "2. **Khả năng khái quát hóa và độ linh hoạt ngữ điệu:** Concatenative TTS bị giới hạn hoàn toàn trong các unit đã ghi âm; nếu gặp từ mới hoặc ngữ cảnh lạ, chất lượng giảm sút rõ rệt. Ngược lại, Neural TTS có khả năng tổng quát hóa ngữ điệu toàn câu siêu việt, tạo ra giọng nói tự nhiên, truyền cảm và biểu đạt cảm xúc mượt mà.\n",
        "\n",
        "### Câu 9: Vì sao không nên so sánh neural TTS bằng một model tùy ý tải từ Internet trong buổi Lab?\n",
        "**Trả lời:**\n",
        "- **Thiếu kiểm soát tính nhất quán khoa học:** Một model ngẫu nhiên tải trên mạng có thể được huấn luyện trên tập dữ liệu khác, ngôn ngữ khác, tần số lấy mẫu khác ($22.05$ kHz, $24$ kHz, $48$ kHz so với chuẩn $16$ kHz của Lab), hoặc sử dụng chuẩn hóa âm lượng khác nhau, làm sai lệch kết quả so sánh khách quan.\n",
        "- **Rủi ro kỹ thuật:** Model không rõ nguồn gốc có thể đòi hỏi GPU dung lượng lớn, phiên bản thư viện xung đột (PyTorch, CUDA), gây lỗi crash hoặc tốn thời gian tải dữ liệu lớn không cần thiết trong thời gian thực hành có hạn.\n",
        "\n",
        "### Câu 10: Nếu một hệ thống có MOS cao nhưng intelligibility thấp thì có thể dùng trong ứng dụng nào? Giải thích.\n",
        "**Trả lời:**\n",
        "- Hệ thống có **MOS cao** (giọng nghe rất tự nhiên, ấm áp, giàu cảm xúc, không bị méo/buzzy) nhưng **Intelligibility thấp** (khó nghe rõ chính xác từng từ cụ thể).\n",
        "- **Ứng dụng phù hợp:**\n",
        "  1. **Tạo âm thanh nền hoặc tiếng nói rì rầm (Ambient crowd chatter / Background murmur):** Trong game 3D, phim ảnh, quán cà phê ảo nơi chỉ cần tạo không khí có giọng người sống động nhưng người dùng không cần nghe rõ nội dung nói gì.\n",
        "  2. **Ứng dụng thiền định, ru ngủ, âm thanh thư giãn (ASMR / Sleep sounds):** Nơi người nghe chỉ cần giai điệu giọng nói êm dịu, ấm áp để đi vào giấc ngủ hoặc giải tỏa căng thẳng mà không cần tập trung giải mã ngữ nghĩa từ vựng.\n"
    ))
    
    nb = {
        "cells": cells,
        "metadata": {
            "language_info": {
                "name": "python",
                "version": "3.14.0"
            },
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    
    with open(NB_PATH, "w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=2)
    print(f"[+] Đã tạo thành công Jupyter Notebook: {NB_PATH}")

if __name__ == "__main__":
    build()
