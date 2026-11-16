# -*- coding: utf-8 -*-
"""
generate_report_docx.py
=======================
Tạo báo cáo chính thức report_Lab5.docx chuyên nghiệp:
- Trình bày đẹp chuẩn tài liệu học thuật (Navy Blue, Slate, Callout Boxes, Border Tables)
- Đầy đủ công thức toán học, bảng số liệu thực nghiệm, nhận xét nghe và biểu đồ minh họa
- Trả lời chuyên sâu toàn bộ 12 câu hỏi lý thuyết & thực hành của Lab 5

Sinh viên: Dương Tiến Đạt - MSSV: 2251172268
"""

import os
import sys
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import parse_xml, OxmlElement
from docx.oxml.ns import nsdecls, qn

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(BASE_DIR, "results", "figures")
DOCX_OUT = os.path.join(BASE_DIR, "report_Lab5.docx")

# Màu sắc chủ đạo (Corporate / Academic Navy Palette)
PRIMARY_HEX = "1B365D"      # Navy Blue đậm
SECONDARY_HEX = "2E86AB"    # Cerulean Blue
ACCENT_HEX = "F24236"       # Đỏ cam điểm nhấn
BG_LIGHT_HEX = "F4F6F9"     # Xám nền nhẹ
BORDER_HEX = "D0D5DD"       # Xám viền thanh mảnh

def set_cell_background(cell, hex_color):
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{hex_color}"/>')
    cell._tc.get_or_add_tcPr().append(shd)

def set_cell_margins(cell, top=120, bottom=120, left=160, right=160):
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = OxmlElement('w:tcMar')
    for m, val in [('top', top), ('bottom', bottom), ('left', left), ('right', right)]:
        node = OxmlElement(f'w:{m}')
        node.set(qn('w:w'), str(val))
        node.set(qn('w:type'), 'dxa')
        tcMar.append(node)
    tcPr.append(tcMar)

def set_table_borders(table, color="D0D5DD", sz="4", val="single"):
    tblPr = table._tbl.tblPr
    borders = parse_xml(
        f'<w:tblBorders {nsdecls("w")}>'
        f'  <w:top w:val="{val}" w:sz="{sz}" w:space="0" w:color="{color}"/>'
        f'  <w:bottom w:val="{val}" w:sz="{sz}" w:space="0" w:color="{color}"/>'
        f'  <w:insideH w:val="{val}" w:sz="{sz}" w:space="0" w:color="{color}"/>'
        f'  <w:left w:val="none"/>'
        f'  <w:right w:val="none"/>'
        f'  <w:insideV w:val="none"/>'
        f'</w:tblBorders>'
    )
    tblPr.append(borders)

def add_callout(doc, text, title=None, border_color="1B365D", bg_color="F4F6F9"):
    tbl = doc.add_table(rows=1, cols=1)
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    cell = tbl.cell(0, 0)
    set_cell_background(cell, bg_color)
    set_cell_margins(cell, top=140, bottom=140, left=200, right=200)

    tcPr = cell._tc.get_or_add_tcPr()
    borders = parse_xml(
        f'<w:tcBorders {nsdecls("w")}>'
        f'  <w:top w:val="none"/>'
        f'  <w:left w:val="single" w:sz="24" w:space="0" w:color="{border_color}"/>'
        f'  <w:bottom w:val="none"/>'
        f'  <w:right w:val="none"/>'
        f'</w:tcBorders>'
    )
    tcPr.append(borders)

    p = cell.paragraphs[0]
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.line_spacing = 1.15
    if title:
        r_title = p.add_run(f"★ {title}\n")
        r_title.bold = True
        r_title.font.name = "Arial"
        r_title.font.size = Pt(10.5)
        r_title.font.color.rgb = RGBColor(0x1B, 0x36, 0x5D)

    r_text = p.add_run(text)
    r_text.font.name = "Arial"
    r_text.font.size = Pt(10)
    r_text.font.color.rgb = RGBColor(0x33, 0x33, 0x33)
    doc.add_paragraph().paragraph_format.space_after = Pt(4)


def add_heading_styled(doc, text, level):
    h = doc.add_heading(text, level=level)
    h.paragraph_format.keep_with_next = True
    r = h.runs[0]
    r.font.name = "Arial"
    if level == 1:
        h.paragraph_format.space_before = Pt(16)
        h.paragraph_format.space_after = Pt(6)
        r.font.size = Pt(15)
        r.bold = True
        r.font.color.rgb = RGBColor(0x1B, 0x36, 0x5D)
    elif level == 2:
        h.paragraph_format.space_before = Pt(12)
        h.paragraph_format.space_after = Pt(4)
        r.font.size = Pt(12.5)
        r.bold = True
        r.font.color.rgb = RGBColor(0x2E, 0x86, 0xAB)
    elif level == 3:
        h.paragraph_format.space_before = Pt(8)
        h.paragraph_format.space_after = Pt(2)
        r.font.size = Pt(11)
        r.bold = True
        r.font.color.rgb = RGBColor(0x44, 0x44, 0x44)
    return h


def main():
    doc = docx.Document()

    # Thiết lập lề chuẩn trang (Normal Margins: 1 inch = 2.54 cm)
    for section in doc.sections:
        section.top_margin = Inches(1.0)
        section.bottom_margin = Inches(1.0)
        section.left_margin = Inches(1.0)
        section.right_margin = Inches(1.0)

    # ─── TIÊU ĐỀ BÁO CÁO ──────────────────────────────────────────
    p_inst = doc.add_paragraph()
    p_inst.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_inst.paragraph_format.space_after = Pt(2)
    r_inst = p_inst.add_run("TRƯỜNG ĐẠI HỌC THỦY LỢI • KHOA CÔNG NGHỆ THÔNG TIN\nBỘ MÔN TRÍ TUỆ NHÂN TẠO")
    r_inst.font.name = "Arial"
    r_inst.font.size = Pt(9.5)
    r_inst.font.color.rgb = RGBColor(0x66, 0x66, 0x66)
    r_inst.bold = True

    p_title = doc.add_paragraph()
    p_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_title.paragraph_format.space_before = Pt(6)
    p_title.paragraph_format.space_after = Pt(4)
    r_title = p_title.add_run("BÁO CÁO THỰC HÀNH LAB 5\nSPEECH ENHANCEMENT, SPEAKER RECOGNITION AND SPEAKER SEPARATION")
    r_title.font.name = "Arial"
    r_title.font.size = Pt(16)
    r_title.bold = True
    r_title.font.color.rgb = RGBColor(0x1B, 0x36, 0x5D)

    p_sub = doc.add_paragraph()
    p_sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_sub.paragraph_format.space_after = Pt(14)
    r_sub = p_sub.add_run("Từ Khử nhiễu STFT đến Xác thực người nói và Phân tách nguồn bằng FastICA")
    r_sub.font.name = "Arial"
    r_sub.font.size = Pt(11)
    r_sub.italic = True
    r_sub.font.color.rgb = RGBColor(0x2E, 0x86, 0xAB)

    # ─── BẢNG THÔNG TIN SINH VIÊN ─────────────────────────────────
    tbl_info = doc.add_table(rows=5, cols=2)
    tbl_info.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_table_borders(tbl_info, color="B0C4DE", sz="6")

    info_data = [
        ("Sinh viên thực hiện", "DƯƠNG TIẾN ĐẠT"),
        ("Mã số sinh viên (MSSV)", "2251172268"),
        ("Học phần", "CSE457 – Xử lý âm thanh và tiếng nói"),
        ("Giảng viên hướng dẫn", "Bộ môn Trí tuệ Nhân tạo – TLU"),
        ("Case Study tích hợp", "Trợ lý phòng họp mini (Mini Meeting Assistant)"),
    ]

    for row_idx, (k, v) in enumerate(info_data):
        c0 = tbl_info.cell(row_idx, 0)
        c1 = tbl_info.cell(row_idx, 1)
        c0.width = Inches(2.2)
        c1.width = Inches(4.3)
        set_cell_margins(c0, 80, 80, 120, 120)
        set_cell_margins(c1, 80, 80, 120, 120)
        set_cell_background(c0, BG_LIGHT_HEX)
        set_cell_background(c1, "FFFFFF" if row_idx % 2 == 0 else "FDFDFE")

        p0 = c0.paragraphs[0]; p0.paragraph_format.space_after = Pt(0)
        r0 = p0.add_run(k); r0.bold = True; r0.font.name = "Arial"; r0.font.size = Pt(9.5)
        r0.font.color.rgb = RGBColor(0x33, 0x33, 0x33)

        p1 = c1.paragraphs[0]; p1.paragraph_format.space_after = Pt(0)
        r1 = p1.add_run(v); r1.bold = (row_idx < 2); r1.font.name = "Arial"; r1.font.size = Pt(9.5)
        if row_idx < 2:
            r1.font.color.rgb = RGBColor(0x1B, 0x36, 0x5D)

    doc.add_paragraph().paragraph_format.space_after = Pt(8)

    # ─── 1. MỤC TIÊU VÀ TỔ CHỨC THÍ NGHIỆM ────────────────────────
    add_heading_styled(doc, "1. Mục tiêu và Chuẩn đầu ra thực hành", level=1)
    p_intro = doc.add_paragraph()
    p_intro.paragraph_format.line_spacing = 1.15
    p_intro.paragraph_format.space_after = Pt(6)
    p_intro.add_run(
        "Lab 5 tích hợp ba nhóm bài toán xử lý tiếng nói quan trọng cuối học phần:\n"
        "1. Làm sạch tín hiệu (Speech Enhancement): Khử nhiễu STFT bằng Spectral Subtraction và Wiener Filter, "
        "đánh giá định lượng qua SNR trước/sau xử lý và nhận biết hiện tượng musical noise, méo tiếng.\n"
        "2. Nhận dạng người nói (Speaker Recognition): Trích xuất MFCC (D=20) và Delta, xây dựng speaker embedding "
        "thống kê, thực hiện Identification và Verification với ngưỡng quyết định quét trên Development Set độc lập.\n"
        "3. Phân tách nguồn mù (Blind Source Separation): Tạo mô hình trộn tức thời 2 microphone x 2 nguồn nói, "
        "ứng dụng FastICA để tách nguồn và tự động giải quyết bất định hoán vị (permutation) và dấu (sign) bằng ma trận tương quan tuyệt đối."
    )

    add_callout(doc,
        "Toàn bộ dữ liệu âm thanh được chuẩn hóa ở định dạng WAV PCM 16-bit, tần số lấy mẫu Fs = 16,000 Hz, mono. "
        "Quy trình thực nghiệm được tổ chức có tập phát triển (Development Set) và kiểm thử (Test Set) tách biệt, "
        "khóa cứng ngưỡng quyết định nhằm tránh rò rỉ dữ liệu (data leakage).",
        title="Quy cách chuẩn dữ liệu & Nguyên tắc thực nghiệm"
    )

    # ─── 2. PHẦN 1: SPEECH ENHANCEMENT ───────────────────────────
    add_heading_styled(doc, "2. Mô hình nhiễu, Spectral Subtraction và Wiener Filter", level=1)
    
    p_math1 = doc.add_paragraph()
    p_math1.paragraph_format.line_spacing = 1.15
    p_math1.add_run(
        "Mô hình nhiễu cộng tuyến tính: y[n] = s[n] + v[n]. Để tạo hỗn hợp tại SNR mục tiêu R (dB), "
        "hệ số khuếch đại alpha được xác định chính xác theo công thức:\n"
        "    alpha = sqrt( P_s / (P_v * 10^(R/10)) )\n"
        "Thuật toán Spectral Subtraction (Boll 1979) trừ phổ công suất nhiễu ước lượng P_v[k]:\n"
        "    P_ss[m, k] = max( |Y[m, k]|^2 - beta * P_v[k], gamma * P_v[k] )\n"
        "Trong khi đó, bộ lọc Wiener filter đóng vai trò là nghiệm MMSE tuyến tính tối ưu:\n"
        "    H[m, k] = P_s[m, k] / (P_s[m, k] + P_v[k] + epsilon)\n"
    )

    # Bảng kết quả SNR
    add_heading_styled(doc, "Bảng 1: Kết quả đo đạc SNR trước và sau lọc với nhiễu quạt máy (fan.wav)", level=3)
    tbl_snr = doc.add_table(rows=4, cols=6)
    tbl_snr.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_table_borders(tbl_snr)

    headers_snr = ["Target SNR", "SNR_in (Đo)", "SS SNR_out", "Delta SS", "Wiener SNR_out", "Delta Wiener"]
    for j, h in enumerate(headers_snr):
        cell = tbl_snr.cell(0, j)
        set_cell_background(cell, PRIMARY_HEX)
        set_cell_margins(cell, 80, 80, 100, 100)
        p = cell.paragraphs[0]; r = p.add_run(h); r.bold = True
        r.font.name = "Arial"; r.font.size = Pt(9); r.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)

    snr_rows = [
        ("0.0 dB", "0.00 dB", "3.25 dB", "+3.25 dB", "2.72 dB", "+2.72 dB"),
        ("5.0 dB", "5.00 dB", "3.49 dB", "-1.51 dB", "2.42 dB", "-2.58 dB"),
        ("10.0 dB", "10.00 dB", "3.72 dB", "-6.28 dB", "2.31 dB", "-7.69 dB"),
    ]
    for i, row in enumerate(snr_rows):
        for j, val in enumerate(row):
            cell = tbl_snr.cell(i + 1, j)
            set_cell_background(cell, "FFFFFF" if i % 2 == 0 else BG_LIGHT_HEX)
            set_cell_margins(cell, 60, 60, 100, 100)
            p = cell.paragraphs[0]; r = p.add_run(val)
            r.font.name = "Arial"; r.font.size = Pt(9)
            if j in [3, 5] and "+" in val:
                r.bold = True; r.font.color.rgb = RGBColor(0x00, 0x80, 0x00)

    doc.add_paragraph().paragraph_format.space_after = Pt(6)

    # Thêm biểu đồ Spectrogram và SNR
    fig1_path = os.path.join(FIG_DIR, "fig1_clean_noisy_enhanced_spectrograms.png")
    if os.path.exists(fig1_path):
        doc.add_paragraph().paragraph_format.space_after = Pt(2)
        doc.add_picture(fig1_path, width=Inches(6.2))
        p_cap = doc.add_paragraph()
        p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r_cap = p_cap.add_run("Hình 1: Phổ Spectrogram đối chứng: Clean -> Noisy (5dB) -> Spectral Subtraction -> Wiener Filter")
        r_cap.font.size = Pt(8.5); r_cap.font.italic = True; r_cap.font.color.rgb = RGBColor(0x66, 0x66, 0x66)

    fig2_path = os.path.join(FIG_DIR, "fig2_snr_comparison.png")
    if os.path.exists(fig2_path):
        doc.add_picture(fig2_path, width=Inches(5.5))
        p_cap = doc.add_paragraph()
        p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r_cap = p_cap.add_run("Hình 2: Mức cải thiện Delta SNR giữa Spectral Subtraction và Wiener Filter")
        r_cap.font.size = Pt(8.5); r_cap.font.italic = True; r_cap.font.color.rgb = RGBColor(0x66, 0x66, 0x66)

    # ─── 3. PHẦN 2: SPEAKER RECOGNITION ──────────────────────────
    add_heading_styled(doc, "3. Nhận dạng người nói (Identification & Verification)", level=1)
    p_spk_text = doc.add_paragraph()
    p_spk_text.paragraph_format.line_spacing = 1.15
    p_spk_text.add_run(
        "Mỗi câu nói được trích xuất 20 hệ số MFCC kết hợp Delta MFCC. Vector đại diện (speaker representation) "
        "được hình thành từ kỳ vọng và độ lệch chuẩn theo thời gian: e = [mu(c), sigma(c), mu(Delta c), sigma(Delta c)] "
        "chuẩn hóa L2. Mô hình người nói m_i là centroid trung bình của 4 câu enrollment.\n"
        "Độ tương đồng được tính theo Cosine Similarity: cos(e, m_i) = (e . m_i) / (||e|| ||m_i||)."
    )

    # Thêm Confusion Matrix & Verification Curve
    fig5_path = os.path.join(FIG_DIR, "fig5_confusion_matrix_speaker_id.png")
    if os.path.exists(fig5_path):
        doc.add_picture(fig5_path, width=Inches(4.8))
        p_cap = doc.add_paragraph()
        p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r_cap = p_cap.add_run("Hình 3: Ma trận nhầm lẫn (Confusion Matrix) của mô hình Speaker Identification")
        r_cap.font.size = Pt(8.5); r_cap.font.italic = True; r_cap.font.color.rgb = RGBColor(0x66, 0x66, 0x66)

    fig7_path = os.path.join(FIG_DIR, "fig7_speaker_verification_far_frr.png")
    if os.path.exists(fig7_path):
        doc.add_picture(fig7_path, width=Inches(5.5))
        p_cap = doc.add_paragraph()
        p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r_cap = p_cap.add_run("Hình 4: Đường cong FAR và FRR theo ngưỡng theta trên Development Set (Khóa ngưỡng theta*)")
        r_cap.font.size = Pt(8.5); r_cap.font.italic = True; r_cap.font.color.rgb = RGBColor(0x66, 0x66, 0x66)

    # ─── 4. PHẦN 3: SOURCE SEPARATION BSS / FASTICA ───────────────
    add_heading_styled(doc, "4. Phân tách nguồn mù (FastICA Blind Source Separation)", level=1)
    p_ica_text = doc.add_paragraph()
    p_ica_text.paragraph_format.line_spacing = 1.15
    p_ica_text.add_run(
        "Mô hình trộn tức thời 2x2: x[n] = A * s[n]. Ma trận trộn chuẩn A = [[1.0, 0.6], [0.4, 1.0]] có det(A) = 0.76 != 0. "
        "Thuật toán FastICA tối đa hóa tính phi-Gaussian (neg-entropy) để tìm ma trận tách W xấp xỉ A^-1.\n"
        "Hai bất định cố hữu được giải quyết triệt để:\n"
        "1. Hoán vị (Permutation): Ghép kênh ước lượng với nguồn gốc theo hệ số tương quan tuyệt đối |rho| cực đại.\n"
        "2. Đảo dấu (Sign): Nhân -1 nếu hệ số tương quan tuyến tính Pearson raw mang giá trị âm."
    )

    fig8_path = os.path.join(FIG_DIR, "fig8_ica_separation_waveforms_spectrograms.png")
    if os.path.exists(fig8_path):
        doc.add_picture(fig8_path, width=Inches(6.2))
        p_cap = doc.add_paragraph()
        p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r_cap = p_cap.add_run("Hình 5: Spectrogram 6 trạng thái trong quá trình phân tách BSS / FastICA")
        r_cap.font.size = Pt(8.5); r_cap.font.italic = True; r_cap.font.color.rgb = RGBColor(0x66, 0x66, 0x66)

    fig9_path = os.path.join(FIG_DIR, "fig9_ica_correlation_matrices.png")
    if os.path.exists(fig9_path):
        doc.add_picture(fig9_path, width=Inches(5.8))
        p_cap = doc.add_paragraph()
        p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r_cap = p_cap.add_run("Hình 6: Ma trận tương quan Pearson trước và sau khi giải quyết bất định Permutation & Sign")
        r_cap.font.size = Pt(8.5); r_cap.font.italic = True; r_cap.font.color.rgb = RGBColor(0x66, 0x66, 0x66)

    # ─── 5. PHẦN 4: CASE STUDY TÍCH HỢP ───────────────────────────
    add_heading_styled(doc, "5. Case Study: Trợ lý phòng họp mini (End-to-End Pipeline)", level=1)
    p_case_text = doc.add_paragraph()
    p_case_text.paragraph_format.line_spacing = 1.15
    p_case_text.add_run(
        "Hệ thống trợ lý phòng họp vận hành chuỗi liên hoàn 3 chức năng: khử nhiễu micro, nhận diện danh tính và tách nguồn hội thoại chồng lấn. "
        "Thực nghiệm tập trung giải quyết câu hỏi mấu chốt: Liệu Speech Enhancement có luôn giúp cải thiện kết quả Speaker ID?\n"
        "Số liệu đối chứng cho thấy: Khi ở mức nhiễu 5 dB, điểm cosine của spk01 là 0.9675. Qua Spectral Subtraction, điểm số phục hồi lên 0.9765 (+0.0090). "
        "Tuy nhiên, qua bộ lọc Wiener Filter, điểm số lại tụt xuống 0.9534 (thấp hơn cả lúc có nhiễu) do hàm truyền MMSE làm trơn quá mức các chi tiết formant vi mô."
    )

    fig10_path = os.path.join(FIG_DIR, "fig10_integrated_case_study.png")
    if os.path.exists(fig10_path):
        doc.add_picture(fig10_path, width=Inches(5.8))
        p_cap = doc.add_paragraph()
        p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r_cap = p_cap.add_run("Hình 7: Biểu đồ tổng kết 4 góc nhìn của Case Study Trợ lý phòng họp mini")
        r_cap.font.size = Pt(8.5); r_cap.font.italic = True; r_cap.font.color.rgb = RGBColor(0x66, 0x66, 0x66)

    # ─── 6. TRẢ LỜI CHI TIẾT 12 CÂU HỎI BÁO CÁO ───────────────────
    add_heading_styled(doc, "6. Trả lời chi tiết 12 câu hỏi báo cáo (Section 7)", level=1)

    qa_list = [
        ("Câu 1: Tại sao spectral subtraction chỉ sửa magnitude mà thường giữ phase noisy? Khi SNR rất thấp, giả thiết này có hạn chế gì?",
         "Trả lời:\n"
         "• Lý do giữ pha: Tai người chủ yếu nhạy cảm với cấu trúc biên độ phổ ngắn hạn (short-time magnitude spectrum) - nơi chứa các formants và đỉnh cộng hưởng của nguyên âm. Pha của tín hiệu âm thanh ngẫu nhiên khó ước lượng chính xác, và các nghiên cứu kinh điển của Wang & Lim (1982) chỉ ra rằng việc tái tạo bằng pha gốc của noisy signal cho chất lượng nghe tự nhiên hơn so với ước lượng pha sai lệch.\n"
         "• Hạn chế khi SNR rất thấp (< 0 dB): Khi năng lượng nhiễu vượt trội hơn tiếng nói, pha đo được gần như phản ánh hoàn toàn đặc tính pha của nhiễu chứ không phải tiếng nói. Việc ghép biên độ ước lượng với pha nhiễu sẽ gây ra méo tiếng dạng nhòe âm (phase cancellation), làm phụ âm bị biến dạng và tạo artifact dạng tiếng ồn bong bóng (bubbling artifact)."),

        ("Câu 2: Giải thích vai trò của beta và gamma. Vì sao tăng beta có thể giảm residual noise nhưng tăng speech distortion?",
         "Trả lời:\n"
         "• Hệ số over-subtraction beta >= 1: Bù trừ sai số ước lượng do giả định công suất nhiễu là tĩnh. Trong thực tế, công suất nhiễu tức thời tại một số bin có thể cao hơn giá trị trung bình P_v[k]. Nhân hệ số beta > 1 giúp triệt tiêu sạch các đỉnh nhiễu này.\n"
         "• Spectral floor gamma >= 0: Đặt ngưỡng sàn phổ tối thiểu (gamma * P_v[k]) để ngăn phổ công suất bị âm hoặc tụt xuống 0 đột ngột, hạn chế tạo các thung lũng phổ sâu gây musical noise.\n"
         "• Trade-off: Khi tăng beta quá lớn, thuật toán sẽ trừ lấn sâu vào các thành phần năng lượng thấp của tiếng nói sạch (như các formant bậc cao F3, F4 và phụ âm vô thanh), gây méo tiếng nghiêm trọng (speech distortion) và làm giọng nói nghe bị kim loại, nghẹt mũi."),

        ("Câu 3: Wiener filter H = P_s / (P_s + P_v) có ý nghĩa gì ở vùng tần số speech-dominant và noise-dominant?",
         "Trả lời:\n"
         "• Vùng tần số Speech-dominant (P_s >> P_v): Tỉ số tín hiệu trên nhiễu xi_k = P_s / P_v >> 1, hàm truyền H[k] = 1 / (1 + P_v/P_s) -> 1. Bộ lọc mở cho tín hiệu đi qua hầu như nguyên vẹn, bảo toàn biên độ và năng lượng của formants.\n"
         "• Vùng tần số Noise-dominant (P_s << P_v): xi_k -> 0, hàm truyền H[k] -> 0. Bộ lọc suy giảm mạnh tín hiệu tại tần số đó, triệt tiêu nhiễu nền.\n"
         "• Đặc tính: Wiener filter là nghiệm tối ưu tuyến tính theo tiêu chuẩn sai số toàn phương cực tiểu (MMSE). Đáp ứng hàm truyền biến thiên mềm mại theo thời gian và tần số, giúp giảm đáng kể hiện tượng musical noise so với Spectral Subtraction."),

        ("Câu 4: Vì sao SNR_out phải tính sau khi căn chỉnh clean và enhanced? Nếu thuật toán tạo delay 10 ms thì số đo sai theo hướng nào?",
         "Trả lời:\n"
         "• Nguyên nhân: Công thức đo SNR = 10 log10( sum(s^2) / sum((s - s_hat)^2) ) đo sai số theo từng mẫu tức thời (point-by-point MSE).\n"
         "• Ảnh hưởng của trễ 10 ms (160 mẫu tại 16 kHz): Khi s_hat[n] xấp xỉ s[n - 160], sai số e[n] = s[n] - s[n - 160]. Đối với tín hiệu dao động hình sin hoặc tiếng nói tuần hoàn, hai mẫu lệch nhau nửa chu kỳ sẽ triệt tiêu pha và có năng lượng sai số sum(e^2) xấp xỉ 2 * sum(s^2).\n"
         "• Khi đó SNR_out xấp xỉ 10 log10(0.5) xấp xỉ -3 dB, dù về mặt thính giác tín hiệu sau lọc nghe cực kỳ trong trẻo và sạch nhiễu. Do đó, việc căn chỉnh độ trễ (delay alignment) và cân bằng biên độ (gain matching) là điều kiện tiên quyết để số đo SNR phản ánh đúng bản chất kỹ thuật."),

        ("Câu 5: Speaker identification và speaker verification khác nhau về output và metric đánh giá như thế nào?",
         "Trả lời:\n"
         "1. Speaker Identification (Xác định danh tính - 1-vs-N):\n"
         "   - Input & Output: Đầu vào là file âm thanh chưa biết; đầu ra là danh tính người nói trong tập đóng hoặc 'unknown'.\n"
         "   - Metric: Accuracy (tỉ lệ đoán đúng), Confusion Matrix, Top-k accuracy.\n"
         "2. Speaker Verification (Xác thực danh tính - 1-vs-1):\n"
         "   - Input & Output: Đầu vào là file âm thanh kèm theo lời tuyên bố danh tính (claim); đầu ra là quyết định nhị phân: ACCEPT hoặc REJECT.\n"
         "   - Metric: FAR (False Acceptance Rate), FRR (False Rejection Rate), EER (Equal Error Rate), ROC/DET Curve."),

        ("Câu 6: Tại sao threshold verification không được chọn trên test set? Giải thích trade-off FAR–FRR.",
         "Trả lời:\n"
         "• Nguyên tắc chọn ngưỡng: Chọn ngưỡng theta trên Test set là hành vi rò rỉ dữ liệu (data leakage) và over-tuning, khiến hệ thống có kết quả ảo tưởng phi thực tế nhưng thất bại khi triển khai trong đời thực. Ngưỡng theta* bắt buộc phải được tối ưu hóa độc lập trên Development set, sau đó khóa cố định để kiểm tra trên Test set.\n"
         "• Trade-off FAR–FRR:\n"
         "  - Khi tăng ngưỡng theta khắt khe hơn: Kẻ mạo danh rất khó vượt qua -> FAR giảm, nhưng người dùng hợp lệ dễ bị từ chối oan do giọng nói thay đổi nhẹ -> FRR tăng (phù hợp ngân hàng, bảo mật cao).\n"
         "  - Khi giảm ngưỡng theta nới lỏng hơn: Người dùng hợp lệ dễ dàng đăng nhập -> FRR giảm, nhưng nguy cơ kẻ mạo danh xâm nhập tăng cao -> FAR tăng (phù hợp tiện ích trợ lý gia đình)."),

        ("Câu 7: MFCC được thiết kế chủ yếu cho nội dung âm học. Vì sao vẫn có thể chứa thông tin người nói?",
         "Trả lời:\n"
         "• MFCC được trích xuất từ bao phổ (spectral envelope) thông qua bộ lọc Mel filterbank. Bao phổ này phản ánh đáp ứng tần số của ống thanh âm (vocal tract transfer function).\n"
         "• Kích thước vật lý, chiều dài ống thanh âm, thể tích khoang miệng, họng và tỷ lệ giữa các khoang cộng hưởng mang tính đặc trưng giải phẫu sinh học duy nhất của từng cá nhân. Do đó, dù nội dung câu nói thay đổi, phân bố thống kê dài hạn (mean và std) của các hệ số cepstral vẫn lưu giữ dấu vân âm thanh (vocal tract fingerprint) đặc trưng của người nói đó."),

        ("Câu 8: Vì sao CMN có thể vừa hữu ích cho channel robustness vừa bất lợi cho một baseline speaker-ID dựa trên thống kê MFCC?",
         "Trả lời:\n"
         "• Hữu ích cho Channel Robustness: Kênh truyền âm thanh tạo ra phép chập miền thời gian y[n] = s[n] * h[n], tương đương phép cộng hằng số trong miền log-cepstrum: c_y = c_s + c_h. Trừ trung bình c_hat = c - c_bar sẽ triệt tiêu hoàn toàn thành phần cố định c_h, loại bỏ méo kênh truyền.\n"
         "• Bất lợi cho Baseline Speaker ID: Thành phần trung bình c_bar cũng chứa đựng đặc tính trung bình giải phẫu của ống thanh âm người nói. Khi trừ đi c_bar, ta đã vô tình xóa bỏ một phần thông tin tĩnh về âm sắc đặc trưng. Nếu baseline chỉ dựa vào thống kê mean/std đơn giản, việc mất đi vector mean có thể làm giảm margin khoảng cách giữa các người nói nếu dữ liệu được thu âm trong cùng một môi trường sạch."),

        ("Câu 9: Trong ICA, tại sao không thể biết trước thứ tự và biên độ tuyệt đối của các nguồn tách?",
         "Trả lời:\n"
         "Mô hình ICA: x = A * s. Vì cả A và s đều chưa biết trước, tồn tại hai ma trận bất định:\n"
         "1. Bất định biên độ: Với bất kỳ ma trận đường chéo khả nghịch Lambda, ta có x = (A * Lambda^-1) * (Lambda * s). Nguồn s bị nhân một hệ số vô hướng không làm thay đổi tính độc lập thống kê hay dạng phân bố phi-Gaussian, do đó ICA chỉ xác định được nguồn sai khác một hệ số tỉ lệ và dấu (+/- c).\n"
         "2. Bất định hoán vị: Với bất kỳ ma trận hoán vị P, ta có x = (A * P^-1) * (P * s). Việc tráo đổi vị trí các nguồn độc lập không làm thay đổi phân bố đồng thời của các biến ngẫu nhiên độc lập, khiến ICA không thể biết nguồn nào nằm ở kênh ra nào."),

        ("Câu 10: Nếu hai hàng của A gần tỉ lệ nhau, điều gì xảy ra? Liên hệ với tính khả nghịch của mixing matrix.",
         "Trả lời:\n"
         "• Hiện tượng: Nếu 2 hàng của A gần tỉ lệ nhau (ví dụ A = [[1.0, 0.95], [0.95, 0.91]]), hai microphone đặt quá gần nhau hoặc có hướng thu giống hệt nhau, dẫn đến hai tín hiệu quan sát x1[n] và x2[n] gần như trùng khớp thông tin.\n"
         "• Hệ quả đại số: Định thức det(A) -> 0, số điều kiện cond(A) tăng vọt (hệ phương trình ill-conditioned/gần suy biến).\n"
         "• Tác động đến FastICA: Khi nghịch đảo ma trận để tách nguồn W = A^-1, sai số số học và nhiễu vi mô sẽ bị khuếch đại cực đại, khiến thuật toán FastICA hội tụ kém, nguồn tách ra bị méo nặng và chứa tàn dư lớn của nguồn kia."),

        ("Câu 11: Mô hình instantaneous mixing khác thực tế phòng có reverberation ở điểm nào?",
         "Trả lời:\n"
         "• Mô hình Trộn tức thời (Instantaneous Mixing): Giả thiết âm thanh từ các nguồn truyền thẳng tức thì đến các microphone mà không có độ trễ hay phản xạ phòng: x_i[n] = sum_j a_ij s_j[n] (phép nhân ma trận đại số đơn giản).\n"
         "• Thực tế phòng có dội âm (Convolutive Mixing): Âm thanh va đập vào tường, trần, sàn và đồ vật tạo ra hàng ngàn đường truyền trễ và suy hao khác nhau. Quan sát tại mỗi micro là một phép chập giữa nguồn và phản ứng xung phòng (Room Impulse Response - RIR): x_i[n] = sum_j (h_ij * s_j)[n].\n"
         "Mô hình trở thành bài toán lọc chập phức tạp trong miền thời gian, đòi hỏi các thuật toán nâng cao như Frequency-Domain ICA kết hợp giải quyết permutation problem trên từng bin tần số."),

        ("Câu 12: Từ kết quả của nhóm, enhancement có luôn cải thiện speaker recognition không? Trình bày bằng số liệu.",
         "Trả lời:\n"
         "• Kết luận: KHÔNG PHẢI LÚC NÀO ENHANCEMENT CŨNG CẢI THIỆN SPEAKER RECOGNITION.\n"
         "• Minh chứng số liệu thực nghiệm của Lab:\n"
         "  - Khi ở trạng thái sạch (Clean): Điểm cosine của đúng người nói spk01 đạt 0.9997.\n"
         "  - Khi bị nhiễu quạt 5 dB: Điểm cosine giảm xuống 0.9675.\n"
         "  - Sau khi qua Spectral Subtraction: Điểm cosine phục hồi lên 0.9765 (tăng +0.0090).\n"
         "  - Tuy nhiên, sau khi qua Wiener Filter: Điểm cosine lại tụt xuống 0.9534 (thấp hơn cả lúc bị nhiễu!).\n"
         "• Nguyên nhân kỹ thuật: Các thuật toán enhancement (nhất là Wiener filter hoặc Spectral Subtraction với beta cao) tập trung tối ưu hóa tỉ số tín hiệu trên nhiễu (SNR) hoặc chất lượng cảm nhận thính giác của con người (MOS) bằng cách làm trơn hoặc triệt tiêu các dải tần biên độ nhỏ. Tuy nhiên, các đặc trưng người nói lại nằm rất nhiều ở các chi tiết bao phổ vi mô, họa âm bậc cao và phụ âm thoáng qua. Quá trình khử nhiễu đã vô tình cắt gọt và bóp méo (speech distortion) các đặc trưng sinh học tinh tế này, khiến vector MFCC bị dịch chuyển và làm giảm margin phân lớp của mô hình nhận dạng.")
    ]

    for q, a in qa_list:
        add_heading_styled(doc, q, level=2)
        p = doc.add_paragraph()
        p.paragraph_format.line_spacing = 1.15
        p.paragraph_format.space_after = Pt(6)
        r = p.add_run(a)
        r.font.name = "Arial"
        r.font.size = Pt(9.5)
        r.font.color.rgb = RGBColor(0x22, 0x22, 0x22)

    doc.save(DOCX_OUT)
    print(f"✓ Đã xuất báo cáo Word thành công tại: {DOCX_OUT}")


if __name__ == "__main__":
    main()
