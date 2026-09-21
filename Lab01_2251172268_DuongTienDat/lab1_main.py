"""
lab1_main.py
============
CSE457 Lab 1 – Script điều phối: chạy toàn bộ pipeline từ A → G
-----------------------------------------------------------------
Cách dùng trong VS Code terminal:
  python lab1_main.py              # Menu tương tác
  python lab1_main.py --tts        # Tạo file âm thanh TTS trước
  python lab1_main.py --all        # Tự động chạy tất cả A–G
  python lab1_main.py --part C     # Chạy riêng phần C
  python lab1_main.py --generate   # Chỉ tạo file tổng hợp
  python lab1_main.py --all --file audio/speech_like.wav  # Chỉ định file
"""

import os
import sys
import time
import argparse

# Thêm thư mục chứa lab1_main.py vào sys.path để import các module khác
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from lab1_utils import (
    read_wav, choose_audio_file, write_wav,
    AUDIO_DIR, FIGURES_DIR
)


# ════════════════════════════════════════════════════════════
# TIỆN ÍCH IN GIAO DIỆN TERMINAL
# ════════════════════════════════════════════════════════════

def banner():
    """In banner tiêu đề khi khởi động."""
    print("\n" + "╔" + "═" * 60 + "╗")
    print("║  CSE457 – XỬ LÝ ÂM THANH VÀ TIẾNG NÓI                  ║")
    print("║  LAB 1: Phân tích và xử lý tín hiệu âm thanh số         ║")
    print("╚" + "═" * 60 + "╝")


def section_header(title: str):
    """In header đóng khung cho mỗi phần."""
    width = 60
    bar   = "═" * width
    print(f"\n╔{bar}╗")
    print(f"║  {title:<{width-2}}║")
    print(f"╚{bar}╝")


def run_step(label: str, func, *args, **kwargs):
    """
    Gọi một hàm (phần lab), đo thời gian thực thi.
    Bắt exception để pipeline không dừng khi 1 phần lỗi.

    Returns:
        Kết quả của func(*args, **kwargs), hoặc None nếu có lỗi.
    """
    section_header(label)
    t0 = time.perf_counter()
    try:
        result  = func(*args, **kwargs)
        elapsed = time.perf_counter() - t0
        print(f"\n  ✓ {label} – hoàn thành ({elapsed:.1f}s)")
        return result
    except Exception as e:
        elapsed = time.perf_counter() - t0
        print(f"\n  ✗ {label} – LỖI ({elapsed:.1f}s): {e}")
        import traceback
        traceback.print_exc()
        return None


# ════════════════════════════════════════════════════════════
# ĐẢM BẢO CÓ FILE ÂM THANH ĐỂ PHÂN TÍCH
# Kiểm tra thư mục audio/. Nếu trống → tự tạo file tổng hợp mặc định.
# ════════════════════════════════════════════════════════════

def ensure_audio_exists() -> bool:
    """
    Trả về True nếu có ít nhất 1 file WAV trong audio/.
    Nếu không có → gọi generate_audio.py tạo file mẫu tự động.
    """
    wavs = [f for f in os.listdir(AUDIO_DIR) if f.lower().endswith(".wav")]
    if wavs:
        return True

    print("\n  ⚠ Chưa có file âm thanh nào trong audio/")
    print("  Đang tạo tín hiệu tổng hợp mặc định...")
    try:
        from generate_audio import generate_all_synthetic
        generate_all_synthetic()
        return True
    except Exception as e:
        print(f"  ✗ Không tạo được file mẫu: {e}")
        return False


# ════════════════════════════════════════════════════════════
# MENU TẠO FILE ÂM THANH (TTS + SYNTHETIC)
# ════════════════════════════════════════════════════════════

def menu_tts():
    """
    Gọi hàm main() của generate_audio.py để hiện menu tạo file âm thanh.
    Người dùng có thể nhập văn bản để TTS hoặc tạo tín hiệu tổng hợp.
    """
    section_header("TẠO FILE ÂM THANH (Text-to-Speech + Synthetic)")
    from generate_audio import main as gen_main
    gen_main()


# ════════════════════════════════════════════════════════════
# BẢNG ĐỊNH NGHĨA CÁC PHẦN LAB
# PARTS: dict ánh xạ key (A-G) → (mô tả, tên module)
# ════════════════════════════════════════════════════════════

PARTS = {
    "A": ("Đọc và kiểm tra dữ liệu âm thanh",       "lab1_part_a"),
    "B": ("Phân tích miền thời gian",                "lab1_part_b"),
    "C": ("Phân tích FFT",                           "lab1_part_c"),
    "D": ("STFT và Spectrogram",                     "lab1_part_d"),
    "E": ("Thí nghiệm cửa sổ",                      "lab1_part_e"),
    "F": ("Lọc số FIR",                              "lab1_part_f"),
    "G": ("Lượng tử hóa, Resampling và Mã hóa",     "lab1_part_g"),
}


def import_part(module_name: str):
    """
    Import động module lab1_part_X theo tên.
    Dùng importlib để tránh phải import tất cả ở đầu file.
    """
    import importlib
    return importlib.import_module(module_name)


def run_all_parts(info: dict):
    """
    Chạy tuần tự tất cả phần A → G.
    Truyền dict info (audio data) xuyên suốt để tránh đọc lại file nhiều lần.
    """
    t_total = time.perf_counter()
    failed  = []

    for key, (desc, mod_name) in PARTS.items():
        mod    = import_part(mod_name)
        result = run_step(f"Phần {key}: {desc}", mod.run, info)
        # Ghi lại phần bị lỗi (F và G trả về dict nên không bị None)
        if result is None and key not in ("F", "G"):
            failed.append(key)

    elapsed = time.perf_counter() - t_total
    print("\n" + "═" * 62)
    if failed:
        print(f"  ⚠ Các phần bị lỗi: {', '.join(failed)}")
    else:
        print(f"  ✓ Tất cả phần A–G hoàn thành trong {elapsed:.1f}s")
    print("═" * 62)
    print_summary()   # In danh sách file đầu ra


def run_single_part(part_key: str, info: dict):
    """Chạy riêng 1 phần theo key (A/B/C/D/E/F/G)."""
    key = part_key.upper()
    if key not in PARTS:
        print(f"  ✗ Phần '{key}' không hợp lệ. Chọn trong: {list(PARTS.keys())}")
        return

    desc, mod_name = PARTS[key]
    mod = import_part(mod_name)
    run_step(f"Phần {key}: {desc}", mod.run, info)
    print_summary()


# ════════════════════════════════════════════════════════════
# IN TÓM TẮT FILE ĐẦU RA
# Liệt kê tất cả file WAV và PNG đã tạo kèm kích thước
# ════════════════════════════════════════════════════════════

def print_summary():
    """In danh sách tất cả file âm thanh và hình ảnh đã tạo ra."""
    print("\n" + "─" * 62)
    print("  KẾT QUẢ ĐẦU RA")
    print("─" * 62)

    # Liệt kê file WAV trong audio/
    wavs = sorted(f for f in os.listdir(AUDIO_DIR)
                  if f.lower().endswith(".wav"))
    if wavs:
        print(f"\n  📁 audio/  ({len(wavs)} file WAV)")
        for f in wavs:
            sz = os.path.getsize(os.path.join(AUDIO_DIR, f))
            print(f"     {f:<45} {sz//1024:>5} KB")

    # Liệt kê hình PNG trong figures/
    pngs = sorted(f for f in os.listdir(FIGURES_DIR)
                  if f.lower().endswith(".png"))
    if pngs:
        print(f"\n  📁 figures/  ({len(pngs)} hình PNG)")
        for f in pngs:
            sz = os.path.getsize(os.path.join(FIGURES_DIR, f))
            print(f"     {f:<45} {sz//1024:>5} KB")

    print()


# ════════════════════════════════════════════════════════════
# MENU TƯƠNG TÁC CHÍNH
# Hiển thị danh sách tùy chọn và xử lý input người dùng
# ════════════════════════════════════════════════════════════

def interactive_menu():
    """
    Vòng lặp menu cho đến khi người dùng nhấn Q.
    Các lựa chọn:
      [0] Tạo/thêm file âm thanh
      [1] Chạy tất cả A–G
      [A-G] Chạy riêng phần đó
      [S] Xem danh sách file đầu ra
      [Q] Thoát
    """
    banner()

    # Đảm bảo có ít nhất 1 file WAV trước khi tiếp tục
    if not ensure_audio_exists():
        print("  Không thể tiếp tục vì không có file âm thanh.")
        return

    print("\n  MENU CHÍNH")
    print("  " + "─" * 50)
    print("  [0]  Tạo / thêm file âm thanh (TTS + Synthetic)")
    print("  [1]  Chọn file âm thanh và chạy TẤT CẢ A–G")
    for key, (desc, _) in PARTS.items():
        print(f"  [{key}]  Chạy Phần {key}: {desc}")
    print("  [S]  Xem tóm tắt file đầu ra")
    print("  [Q]  Thoát")
    print("  " + "─" * 50)

    while True:
        choice = input("\n  Lựa chọn: ").strip().upper()

        if choice == "Q":
            print("  Thoát.")
            break

        elif choice == "0":
            # Mở menu tạo file âm thanh
            menu_tts()

        elif choice in ("1", "ALL"):
            # Chọn file → đọc → chạy tất cả phần
            filepath = choose_audio_file()
            print(f"\n  File đã chọn: {filepath}")
            info = read_wav(filepath)
            run_all_parts(info)

        elif choice in PARTS:
            # Chọn file → chạy riêng phần được yêu cầu
            filepath = choose_audio_file()
            info     = read_wav(filepath)
            run_single_part(choice, info)

        elif choice == "S":
            print_summary()

        else:
            print("  ✗ Lựa chọn không hợp lệ.")


# ════════════════════════════════════════════════════════════
# COMMAND-LINE INTERFACE (CLI) bằng argparse
# Cho phép chạy không cần menu (non-interactive)
# Ví dụ: python lab1_main.py --all --file audio/chord_C_major.wav
# ════════════════════════════════════════════════════════════

def parse_args():
    """Định nghĩa và parse các argument dòng lệnh."""
    p = argparse.ArgumentParser(
        description="CSE457 Lab 1 – Pipeline xử lý âm thanh",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument("--tts",  action="store_true",
                   help="Chạy TTS để tạo file âm thanh trước")
    p.add_argument("--all",  action="store_true",
                   help="Tự động chạy tất cả phần A–G")
    p.add_argument("--part", type=str, metavar="X",
                   help="Chạy riêng phần X (A/B/C/D/E/F/G)")
    p.add_argument("--file", type=str, metavar="PATH",
                   help="Đường dẫn file âm thanh (bỏ qua menu chọn file)")
    p.add_argument("--generate", action="store_true",
                   help="Chỉ tạo file âm thanh tổng hợp và thoát")
    return p.parse_args()


def main():
    """
    Hàm main: phân nhánh theo argument dòng lệnh.
    Nếu không có argument → chạy interactive_menu().
    """
    args = parse_args()

    # ── Chỉ tạo file tổng hợp ──
    if args.generate:
        from generate_audio import generate_all_synthetic
        generate_all_synthetic()
        return

    # ── Tạo file TTS trước ──
    if args.tts:
        menu_tts()

    # ── Chế độ non-interactive (--all hoặc --part) ──
    if args.all or args.part:
        ensure_audio_exists()

        # Xác định file âm thanh sẽ dùng
        if args.file:
            if not os.path.isfile(args.file):
                print(f"  ✗ File không tồn tại: {args.file}")
                sys.exit(1)
            filepath = args.file
        else:
            # Lấy file WAV đầu tiên (theo thứ tự alphabet)
            wavs = sorted(f for f in os.listdir(AUDIO_DIR)
                          if f.lower().endswith(".wav"))
            if not wavs:
                print("  ✗ Không có file WAV trong audio/")
                sys.exit(1)
            filepath = os.path.join(AUDIO_DIR, wavs[0])
            print(f"  Auto-select file: {filepath}")

        # Đọc file một lần, dùng cho tất cả phần
        info = read_wav(filepath)

        if args.all:
            banner()
            run_all_parts(info)
        elif args.part:
            run_single_part(args.part, info)
        return

    # ── Chế độ interactive (mặc định khi không có argument) ──
    interactive_menu()


if __name__ == "__main__":
    main()
