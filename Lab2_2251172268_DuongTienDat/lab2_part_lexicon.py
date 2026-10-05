"""
lab2_part_lexicon.py
====================
CSE457 Lab 2 – Mục 3.5: Từ điển phát âm (Pronunciation Lexicon)
Sinh viên: Dương Tiến Đạt – MSSV: 2251172268

Nội dung theo Chương 3:
  3.5 Pronunciation Lexicon: Lexical Baseforms, phoneme mapping
    - Bảng ký hiệu âm vị học IPA / ARPABET cho tiếng Việt
    - Từ điển phát âm 5 từ (khong/mot/hai/ba/bon) + mở rộng
    - Letter-to-Sound (G2P) rules đơn giản
    - Tích hợp với pipeline MFCC: phoneme alignment theo Viterbi

Tài liệu:
  - Huang et al., Mục 9.4.4: Lexical Baseforms
  - Jurafsky & Martin, Ch.25 Mục 25.1, 25.5: Phonetic Transcription
"""

import os, math, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from lab2_utils import (
    FS, HOP_LEN, LABELS, LABEL_VI, FIGURES_DIR,
    read_wav, savefig, style_ax, get_train_test_files
)
from lab2_part_abcd import compute_mfcc, endpoint_detection

# ════════════════════════════════════════════════════════════
# 3.5 TỪ ĐIỂN PHÁT ÂM TIẾNG VIỆT
#
# Ký hiệu âm vị học đơn giản hóa cho 5 từ (ASCII-friendly):
#   Consonants: b, d, g, h, k, kh, m, n, ng, t, v, z
#   Vowels    : a, aa, e, ee, i, o, oo, u, uu
#   Finals    : k, m, n, ng, nh, p, t (âm cuối)
#   Tones     : (bỏ qua trong baseline; ghi chú riêng)
#
# Format: WORD → [phoneme_list]
# Mỗi phoneme tương ứng với 1 trạng thái HMM (left-to-right)
# ════════════════════════════════════════════════════════════

# Từ điển phát âm tiếng Việt (đơn giản hóa cho Lab)
# Format: word_ascii: (vietnamese, [phoneme_sequence], ipa_approx)
PRONUNCIATION_DICT = {
    # ── 5 từ chính của Lab ──
    "khong": ("Không", ["kh", "o",  "ng"],       "/kʰoŋ/"),
    "mot":   ("Một",   ["m",  "o",  "k_final"],   "/moːk̚/"),
    "hai":   ("Hai",   ["h",  "a",  "i_final"],   "/haːj/"),
    "ba":    ("Ba",    ["b",  "a"],               "/baː/"),
    "bon":   ("Bốn",   ["b",  "o",  "n_final"],   "/boːn/"),

    # ── Mở rộng (số đếm 6–10) ──
    "nam":   ("Năm",   ["n",  "aa", "m_final"],   "/naːm/"),
    "sau":   ("Sáu",   ["s",  "aa", "u_final"],   "/saːw/"),
    "bay":   ("Bảy",   ["b",  "aa", "y_final"],   "/baːj/"),
    "tam":   ("Tám",   ["t",  "aa", "m_final"],   "/taːm/"),
    "chin":  ("Chín",  ["ch", "i",  "n_final"],   "/tɕiːn/"),
    "muoi":  ("Mười",  ["m",  "uu", "i_final"],   "/mɨːj/"),

    # ── Từ lệnh ──
    "mo":    ("Mở",    ["m",  "oo"],             "/mɘː/"),
    "dong":  ("Đóng",  ["d",  "oo", "ng_final"], "/doŋ/"),
    "tren":  ("Trên",  ["tr", "e",  "n_final"],  "/tɾeːn/"),
    "duoi":  ("Dưới",  ["d",  "uu", "i_final"],  "/zɨːj/"),
    "vao":   ("Vào",   ["v",  "aa", "o_final"],  "/vaːw/"),
    "ra":    ("Ra",    ["r",  "a"],              "/ɾaː/"),
    "len":   ("Lên",   ["l",  "e",  "n_final"],  "/leːn/"),
    "xuong": ("Xuống", ["s",  "uu", "ng_final"], "/suŋ/"),
}

# Tập hợp tất cả phoneme
ALL_PHONEMES = sorted(set(
    ph for entry in PRONUNCIATION_DICT.values()
    for ph in entry[1]
))

# Mô tả các phoneme tiếng Việt
PHONEME_DESCRIPTIONS = {
    "b"       : "Phụ âm bật hơi hữu thanh /b/",
    "d"       : "Phụ âm tắc hữu thanh /d/ (tương đương /z/ ở miền Nam)",
    "g"       : "Phụ âm tắc hữu thanh /g/",
    "h"       : "Phụ âm xát vô thanh /h/",
    "k"       : "Phụ âm tắc vô thanh /k/",
    "kh"      : "Phụ âm xát vô thanh /kʰ/ (fricative)",
    "ch"      : "Phụ âm tắc-xát /tɕ/",
    "tr"      : "Phụ âm tắc-xát hồi âm /tɾ/",
    "m"       : "Phụ âm mũi /m/",
    "n"       : "Phụ âm mũi /n/",
    "ng"      : "Phụ âm mũi /ŋ/ (đầu từ)",
    "ng_final": "Phụ âm mũi /ŋ/ (cuối từ)",
    "l"       : "Phụ âm bên /l/",
    "r"       : "Phụ âm rung /ɾ/",
    "s"       : "Phụ âm xát vô thanh /s/",
    "t"       : "Phụ âm tắc vô thanh /t/",
    "v"       : "Phụ âm xát /v/",
    "a"       : "Nguyên âm mở /aː/",
    "aa"      : "Nguyên âm mở ngắn /a/",
    "e"       : "Nguyên âm trung /eː/",
    "ee"      : "Nguyên âm trung ngắn /e/",
    "i"       : "Nguyên âm cao trước /iː/",
    "o"       : "Nguyên âm tròn /oː/",
    "oo"      : "Nguyên âm trung tròn /ɔ/",
    "u"       : "Nguyên âm tròn cao /uː/",
    "uu"      : "Nguyên âm không tròn cao /ɨː/",
    "k_final" : "Phụ âm tắc vô thanh cuối /k̚/ (không phát ra)",
    "n_final" : "Phụ âm mũi cuối /n/",
    "m_final" : "Phụ âm mũi cuối /m/",
    "i_final" : "Glide /j/ cuối",
    "u_final" : "Glide /w/ cuối",
    "y_final" : "Glide /j/ cuối",
    "o_final" : "Glide /w/ cuối",
    "nh"      : "Phụ âm mũi ngạc /ɲ/",
}


# ════════════════════════════════════════════════════════════
# G2P – Grapheme-to-Phoneme Rules (Letter-to-Sound)
# Quy tắc chuyển đổi chữ → âm cho tiếng Việt
# (Huang et al. Mục 14.8; Jurafsky Ch.25 Mục 25.5)
# ════════════════════════════════════════════════════════════

G2P_RULES = [
    # (grapheme_pattern, phoneme)
    # Phụ âm đặc biệt (ưu tiên khớp dài trước)
    ("kh",  "kh"),
    ("ng",  "ng"),
    ("nh",  "nh"),
    ("ch",  "ch"),
    ("ph",  "f"),
    ("th",  "t_asp"),
    ("tr",  "tr"),
    ("gi",  "z"),
    # Nguyên âm đặc biệt
    ("oo",  "oo"),
    ("uu",  "uu"),
    ("aa",  "aa"),
    ("ee",  "ee"),
    # Đơn
    ("b",   "b"), ("c",  "k"), ("d",  "d"), ("đ", "d"),
    ("g",   "g"), ("h",  "h"), ("i",  "i"), ("k", "k"),
    ("l",   "l"), ("m",  "m"), ("n",  "n"), ("o", "o"),
    ("p",   "p"), ("q",  "kw"),("r",  "r"), ("s", "s"),
    ("t",   "t"), ("u",  "u"), ("v",  "v"), ("x", "s"),
    ("y",   "i"), ("a",  "a"), ("e",  "e"),
]

def grapheme_to_phoneme(word: str) -> list:
    """
    Chuyển đổi từ tiếng Việt (ASCII) → chuỗi phoneme.
    Dùng rule-based matching (longest match first).

    word: chuỗi ASCII, ví dụ "khong" → ["kh","o","ng"]
    """
    # Ưu tiên tra từ điển trước
    if word in PRONUNCIATION_DICT:
        return PRONUNCIATION_DICT[word][1]

    # Fallback: rule-based G2P
    phonemes = []
    i = 0
    while i < len(word):
        matched = False
        # Thử khớp dài nhất (greedy)
        for pattern, ph in G2P_RULES:
            if word[i:i+len(pattern)].lower() == pattern:
                phonemes.append(ph)
                i += len(pattern)
                matched = True
                break
        if not matched:
            phonemes.append(word[i])  # giữ nguyên ký tự
            i += 1
    return phonemes


# ════════════════════════════════════════════════════════════
# PHONEME ALIGNMENT
# Ánh xạ frame MFCC → phoneme dùng Viterbi của HMM đã học
# (Sau khi có HMM models, ta có thể dùng Viterbi để align phoneme)
# ════════════════════════════════════════════════════════════

def phoneme_duration_heuristic(mfcc: np.ndarray,
                                phonemes: list) -> list:
    """
    Ước lượng thô thời lượng mỗi phoneme bằng phân chia đều.
    (Thực tế dùng Viterbi alignment từ HMM)

    Returns: list of (phoneme, start_frame, end_frame, start_ms, end_ms)
    """
    T   = len(mfcc)
    n_ph = len(phonemes)
    if n_ph == 0:
        return []

    frames_per_ph = T / n_ph
    segments = []
    for k, ph in enumerate(phonemes):
        s_fr = int(k * frames_per_ph)
        e_fr = int((k+1) * frames_per_ph)
        e_fr = min(e_fr, T)
        s_ms = s_fr * HOP_LEN / FS * 1000
        e_ms = e_fr * HOP_LEN / FS * 1000
        segments.append((ph, s_fr, e_fr, s_ms, e_ms))
    return segments


# ════════════════════════════════════════════════════════════
# VẼ HÌNH & DEMO
# ════════════════════════════════════════════════════════════

def plot_lexicon_overview() -> None:
    """
    Vẽ tổng quan từ điển phát âm:
      (a) Bảng từ → phoneme sequence
      (b) Phân bố phoneme trong corpus
    """
    fig = plt.figure(figsize=(14, 9))
    gs  = plt.GridSpec(2, 2, figure=fig, hspace=0.5, wspace=0.4)

    # ── Panel 1: Bảng từ điển ──
    ax1 = fig.add_subplot(gs[0, :])
    ax1.axis("off")
    words_to_show = list(PRONUNCIATION_DICT.keys())[:12]
    table_data = []
    for w in words_to_show:
        vi, phs, ipa = PRONUNCIATION_DICT[w]
        table_data.append([
            w, vi, " – ".join(phs), ipa
        ])
    table = ax1.table(
        cellText  = table_data,
        colLabels = ["ASCII", "Tiếng Việt", "Phoneme sequence", "IPA"],
        cellLoc   = "center",
        loc       = "center",
        bbox      = [0, 0, 1, 1]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#3498db")
            cell.set_text_props(color="white", fontweight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#f0f0f0")
        cell.set_edgecolor("gray")
    ax1.set_title("3.5 Từ điển phát âm tiếng Việt (Pronunciation Lexicon)",
                  fontsize=11, fontweight="bold", pad=15)

    # ── Panel 2: Phân bố phoneme trong từ điển ──
    ax2 = fig.add_subplot(gs[1, 0])
    all_ph_counts = {}
    for entry in PRONUNCIATION_DICT.values():
        for ph in entry[1]:
            all_ph_counts[ph] = all_ph_counts.get(ph, 0) + 1

    ph_sorted = sorted(all_ph_counts.items(), key=lambda kv: -kv[1])
    ph_names  = [p[0] for p in ph_sorted]
    ph_counts = [p[1] for p in ph_sorted]
    bars = ax2.bar(range(len(ph_names)), ph_counts,
                   color=plt.cm.tab20(np.linspace(0, 1, len(ph_names))),
                   edgecolor="black", linewidth=0.4)
    ax2.set_xticks(range(len(ph_names)))
    ax2.set_xticklabels(ph_names, rotation=45, ha="right", fontsize=7)
    style_ax(ax2, ylabel="Số từ chứa phoneme",
             title="Phân bố phoneme trong từ điển")

    # ── Panel 3: Phân loại phoneme theo loại ──
    ax3 = fig.add_subplot(gs[1, 1])
    ph_types = {
        "Phụ âm đầu": [p for p in ALL_PHONEMES if not p.endswith("_final") and p not in
                        ["a","aa","e","ee","i","o","oo","u","uu"]],
        "Nguyên âm":  ["a","aa","e","ee","i","o","oo","u","uu"],
        "Phụ âm cuối":[p for p in ALL_PHONEMES if p.endswith("_final")],
    }
    sizes  = [len(v) for v in ph_types.values()]
    labels = [f"{k}\n({n})" for k, n in zip(ph_types.keys(), sizes)]
    ax3.pie(sizes, labels=labels, autopct="%1.0f%%",
            colors=["steelblue", "darkorange", "seagreen"],
            startangle=90, textprops={"fontsize": 9})
    ax3.set_title("Phân loại phoneme", fontsize=10, fontweight="bold")

    fig.suptitle("3.5 Mục – Từ điển phát âm (Pronunciation Lexicon)\n"
                 "Lexical Baseforms: mỗi từ ánh xạ tới chuỗi phoneme",
                 fontsize=11, fontweight="bold")
    savefig(fig, "X1_lexicon_overview.png")


def plot_phoneme_alignment(train_files: dict) -> None:
    """
    Vẽ MFCC + phoneme alignment cho 3 từ.
    """
    words = ["khong", "mot", "hai"]
    fig, axes = plt.subplots(len(words), 1, figsize=(13, 9))

    for ax, word in zip(axes, words):
        fpath = train_files[word][0]
        x     = read_wav(fpath)
        xt, _, _ = endpoint_detection(x)
        mfcc  = compute_mfcc(xt)
        phonemes = PRONUNCIATION_DICT[word][1]
        segments = phoneme_duration_heuristic(mfcc, phonemes)

        # Vẽ MFCC
        t_ms  = np.arange(len(mfcc)) * HOP_LEN / FS * 1000
        im = ax.imshow(mfcc.T, aspect="auto", origin="lower",
                       cmap="RdYlBu_r",
                       extent=[0, t_ms[-1], 0.5, 13.5],
                       alpha=0.85)

        # Vẽ biên phoneme và nhãn
        colors_ph = ["#e74c3c", "#2ecc71", "#3498db", "#f39c12", "#9b59b6"]
        for k, (ph, sf, ef, s_ms, e_ms) in enumerate(segments):
            c = colors_ph[k % len(colors_ph)]
            ax.axvline(s_ms, color=c, lw=1.5, ls="--", alpha=0.9)
            mid_ms = (s_ms + e_ms) / 2
            ax.text(mid_ms, 13.8, f"/{ph}/",
                    ha="center", va="bottom", fontsize=9,
                    color=c, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.1",
                              fc="white", alpha=0.7, ec=c))

        ipa = PRONUNCIATION_DICT[word][2]
        ax.set_title(
            f"[{LABEL_VI[word]}] IPA: {ipa} | Phonemes: {' → '.join(phonemes)}",
            fontsize=10, fontweight="bold"
        )
        ax.set_ylabel("MFCC coefficient")
        ax.set_xlabel("Thời gian (ms)")
        ax.set_ylim([0.5, 14.5])

    fig.suptitle("3.5 – Phoneme Alignment: MFCC + Từ điển phát âm\n"
                 "(Chia đều thời lượng – Thực tế dùng Viterbi HMM)",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    savefig(fig, "X2_phoneme_alignment.png")


def print_g2p_demo() -> None:
    """Demo G2P conversion cho một số từ."""
    test_words = list(PRONUNCIATION_DICT.keys()) + ["xin", "chao", "dep"]
    print("\n  Demo G2P (Grapheme-to-Phoneme):")
    print(f"  {'Từ ASCII':<12} {'Tiếng Việt':<12} {'Phoneme sequence':<30} {'IPA'}")
    print("  " + "─"*70)
    for w in test_words:
        if w in PRONUNCIATION_DICT:
            vi, phs, ipa = PRONUNCIATION_DICT[w]
        else:
            vi = w; phs = grapheme_to_phoneme(w); ipa = "?"
        print(f"  {w:<12} {vi:<12} {' – '.join(phs):<30} {ipa}")


def run_part_lexicon() -> dict:
    """Chạy toàn bộ mục 3.5 Pronunciation Lexicon."""
    print("\n" + "═"*60)
    print("  MỤC 3.5 – TỪ ĐIỂN PHÁT ÂM (PRONUNCIATION LEXICON)")
    print("═"*60)

    print(f"\n  Từ điển: {len(PRONUNCIATION_DICT)} từ")
    print(f"  Tập phoneme: {len(ALL_PHONEMES)} phoneme: {ALL_PHONEMES}")

    # Demo G2P
    print_g2p_demo()

    # Vẽ hình
    print("\n  Vẽ tổng quan từ điển...")
    plot_lexicon_overview()

    train_files, _ = get_train_test_files()
    print("\n  Vẽ phoneme alignment...")
    plot_phoneme_alignment(train_files)

    print("\n  Nhận xét:")
    print("    - Từ điển phát âm ánh xạ word → chuỗi phoneme (lexical baseforms)")
    print("    - G2P rules cho phép xử lý từ ngoài từ điển (OOV)")
    print("    - Tiếng Việt có 6 thanh điệu → cần mã hóa thêm trong ứng dụng thực")
    print("    - Phoneme alignment (Viterbi) cho phép huấn luyện HMM mức phoneme")
    print("    - CMUDict/PRONLEX là chuẩn cho tiếng Anh; tiếng Việt cần xây dựng riêng")
    print("\n  ✓ Mục 3.5 hoàn thành.")

    return {"lexicon": PRONUNCIATION_DICT, "phonemes": ALL_PHONEMES}


if __name__ == "__main__":
    run_part_lexicon()
