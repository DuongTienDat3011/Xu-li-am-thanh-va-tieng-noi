"""
lab4_text_norm.py
=================
Chuẩn hóa văn bản tiếng Việt cho hệ thống Text-to-Speech miền hẹp (Xếp hàng thông báo).
Thực hiện:
  - Lowercase, loại bỏ dấu câu (chấm, phẩy, chấm than, hỏi, hai chấm...).
  - Tách token.
  - Phân tích và mở rộng chuỗi số (digits expansion) sang từ vựng âm học chuẩn.
  - Ánh xạ token tiếng Việt sang mã định danh unit tương ứng trong kho âm thanh (Inventory).
"""

import sys
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# Bảng quy đổi chữ số sang từ tiếng Việt trong miền hẹp xếp hàng
DIGIT_MAP = {
    "0": "không",
    "1": "một",
    "2": "hai",
    "3": "ba",
    "4": "bốn",
    "5": "năm",
    "6": "sáu",
    "7": "bảy",
    "8": "tám",
    "9": "chín",
}

# Ánh xạ từ tiếng Việt sang mã nhãn ascii để tra cứu file trong inventory
WORD_TO_ASCII = {
    "mời": "moi",
    "khách": "khach",
    "hàng": "hang",
    "số": "so",
    "không": "khong",
    "một": "mot",
    "hai": "hai",
    "ba": "ba",
    "bốn": "bon",
    "năm": "nam",
    "sáu": "sau",
    "bảy": "bay",
    "tám": "tam",
    "chín": "chin",
    "đến": "den",
    "quầy": "quay",
}

def normalize_queue_text(text: str) -> list[str]:
    """
    Chuẩn hóa văn bản thông báo xếp hàng:
    Ví dụ: 'Mời khách hàng số 23 đến quầy 5.'
    -> ['mời', 'khách', 'hàng', 'số', 'hai', 'ba', 'đến', 'quầy', 'năm']
    """
    # 1. Lowercase và xóa các dấu câu thông dụng
    clean_text = text.lower()
    for ch in [".", ",", "!", "?", ":", ";", "(", ")", '"', "'", "-"]:
        clean_text = clean_text.replace(ch, " ")
    
    # 2. Tách từ và duyệt từng token
    tokens = clean_text.split()
    normalized_tokens = []
    
    for tok in tokens:
        # Nếu token là chuỗi số (ví dụ: '23', '5', '107')
        if tok.isdigit():
            normalized_tokens.extend(DIGIT_MAP[d] for d in tok if d in DIGIT_MAP)
        else:
            normalized_tokens.append(tok)
            
    return normalized_tokens

def tokens_to_unit_ids(tokens: list[str]) -> list[str]:
    """
    Chuyển danh sách token tiếng Việt sang mã unit ASCII cơ sở:
    Ví dụ: ['mời', 'khách', 'hai'] -> ['moi', 'khach', 'hai']
    """
    unit_ids = []
    for tok in tokens:
        if tok in WORD_TO_ASCII:
            unit_ids.append(WORD_TO_ASCII[tok])
        else:
            raise ValueError(f"Từ vựng '{tok}' nằm ngoài từ điển miền hẹp (OOV - Out of Vocabulary)!")
    return unit_ids

if __name__ == "__main__":
    test_cases = [
        "Mời khách hàng số 23 đến quầy 5.",
        "Mời khách hàng số 107 đến quầy 2.",
        "Mời khách hàng số 496 đến quầy 8!"
    ]
    print("--- KIỂM TRA CHUẨN HÓA VĂN BẢN (TEXT NORMALIZATION) ---")
    for tc in test_cases:
        toks = normalize_queue_text(tc)
        uids = tokens_to_unit_ids(toks)
        print(f"Gốc: '{tc}'")
        print(f"Tokens: {toks}")
        print(f"Unit IDs: {uids}")
        print("-" * 50)
