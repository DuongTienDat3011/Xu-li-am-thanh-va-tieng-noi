"""
lab3_main.py
============
CSE457 Lab 3 – Script chính
Sinh viên: Dương Tiến Đạt – MSSV: 2251172268

Cách dùng:
  python lab3_main.py              # Menu tương tác
  python lab3_main.py --all        # Chạy tất cả A–H
  python lab3_main.py --part A     # Chạy riêng phần A
"""

import os, sys, time, argparse
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from lab3_utils  import get_files, FIGURES_DIR, LABELS


def run_all():
    from lab3_features import run_part_a
    from lab3_hmm      import run_part_bcd
    from lab3_eval     import run_all_efgh

    t0   = time.perf_counter()
    data = run_part_a()
    data = run_part_bcd(data)
    run_all_efgh(data)
    elapsed = time.perf_counter()-t0

    print("\n"+"═"*62)
    print(f"  ✓ Toàn bộ Lab 3 hoàn thành trong {elapsed:.1f}s")
    print("═"*62)
    _print_summary()


def _print_summary():
    import pathlib
    print("\n  📁 figures/")
    for f in sorted(pathlib.Path(FIGURES_DIR).glob("*.png")):
        sz = f.stat().st_size//1024
        print(f"     {f.name:<50} {sz:>4}KB")


def main():
    p = argparse.ArgumentParser(description="CSE457 Lab 3 Pipeline")
    p.add_argument("--all",  action="store_true")
    p.add_argument("--part", type=str, metavar="X")
    args = p.parse_args()

    if args.all:
        run_all()
        return

    if args.part:
        from lab3_utils import get_files
        train_files, test_files = get_files()
        data = {"train_files": train_files, "test_files": test_files}
        key = args.part.upper()
        if key == "A":
            from lab3_features import run_part_a; run_part_a()
        elif key in ("B","C","D","BCD"):
            from lab3_features import run_part_a
            from lab3_hmm import run_part_bcd
            run_part_bcd(run_part_a())
        elif key == "E":
            from lab3_features import run_part_a
            from lab3_hmm import run_part_bcd
            from lab3_eval import run_part_e
            d = run_part_bcd(run_part_a())
            run_part_e(d["models"],d["train_files"],d["test_files"])
        elif key == "F":
            from lab3_eval import run_part_f; run_part_f()
        elif key == "G":
            from lab3_features import run_part_a
            from lab3_hmm import run_part_bcd
            from lab3_eval import run_part_g
            d = run_part_bcd(run_part_a())
            run_part_g(d["models"])
        elif key == "H":
            from lab3_features import run_part_a
            from lab3_hmm import run_part_bcd
            from lab3_eval import run_part_h
            d = run_part_bcd(run_part_a())
            run_part_h(d["models"],d["train_files"],d["test_files"])
        else:
            print(f"Phần '{key}' không hợp lệ. Chọn: A B C D E F G H --all")
        return

    # Interactive menu
    print("\n╔"+"═"*58+"╗")
    print("║  CSE457 Lab 3 – Nhận dạng tiếng nói nâng cao với HMM  ║")
    print("║  Sinh viên: Dương Tiến Đạt – MSSV: 2251172268         ║")
    print("╚"+"═"*58+"╝")
    print("  [1]  Chạy tất cả A–H (khuyến nghị)")
    print("  [A]  Phần A: Trích đặc trưng 39D")
    print("  [BCD]Phần B/C/D: HMM + Baum-Welch + Viterbi")
    print("  [E]  Phần E: So sánh HMM vs DTW")
    print("  [F]  Phần F: Lexicon mini")
    print("  [G]  Phần G: Bigram LM re-ranking")
    print("  [H]  Phần H: Đánh giá & thí nghiệm E1–E5")
    print("  [0]  Thoát")
    print("─"*60)
    choice = input("  Lựa chọn: ").strip().upper()
    if choice in ("1","ALL"):
        run_all()
    elif choice == "0":
        pass
    else:
        sys.argv.append("--part"); sys.argv.append(choice)
        main()


if __name__ == "__main__":
    main()
