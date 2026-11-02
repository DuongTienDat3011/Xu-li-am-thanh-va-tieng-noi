"""
interactive_tts.py
==================
Chương trình Chuyển đổi Văn bản thành Giọng nói Tương tác (Interactive Text-to-Speech)
Cho phép người dùng:
  - Tự nhập câu văn bản bất kỳ hoặc nhập số thứ tự & số quầy.
  - Chọn câu mẫu có sẵn hoặc tự động sinh ngẫu nhiên.
  - Lựa chọn phương pháp tổng hợp (Neural TTS siêu tự nhiên, Unit Selection, HMM).
  - Tự động phát âm thanh ra loa máy tính (Playback audio).
  - Đo đạc chi tiết thời gian xử lý, độ trễ Latency và Real-Time Factor (RTF).
"""

import sys
import os
import time
import random
import soundfile as sf
import numpy as np

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# Nạp các module chức năng từ Lab 4
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from lab4_text_norm import normalize_queue_text, tokens_to_unit_ids
from lab4_neural import synthesize_neural
from lab4_concat import synthesize_concatenative
from lab4_hmm import generate_trajectories, synthesize_source_filter_vocoder

AUDIO_OUT_DIR = os.path.join(BASE_DIR, "audio_output")
os.makedirs(AUDIO_OUT_DIR, exist_ok=True)

PRESET_SENTENCES = [
    "Mời khách hàng số 23 đến quầy 5.",
    "Mời khách hàng số 107 đến quầy 2.",
    "Mời khách hàng số 496 đến quầy 8.",
    "Mời khách hàng số 15 đến quầy 1.",
    "Mời khách hàng số 86 đến quầy 4.",
    "Mời khách hàng số 359 đến quầy 6."
]

def play_audio(file_path: str):
    """Phát âm thanh trực tiếp ra loa máy tính Windows."""
    try:
        import winsound
        print(f"[▶] Đang phát âm thanh: {os.path.basename(file_path)}...")
        winsound.PlaySound(file_path, winsound.SND_FILENAME)
        print("[✓] Đã phát xong!")
    except Exception as e:
        # Fallback mở bằng trình phát mặc định
        try:
            os.system(f'start "" "{file_path}"')
        except Exception:
            print(f"[!] Không thể tự động phát qua winsound: {e}")

def run_tts(text: str, method: str = "neural", voice: str = "vi-VN-HoaiMyNeural", auto_play: bool = True):
    """
    Xử lý tổng hợp văn bản sang tiếng nói theo phương pháp được chọn.
    """
    print("\n" + "-" * 65)
    print(f"[*] VĂN BẢN ĐẦU VÀO: \"{text}\"")
    print("-" * 65)
    
    # 1. Chuẩn hóa văn bản
    tokens = normalize_queue_text(text)
    print(f"[1] Chuẩn hóa văn bản (Text Normalization):")
    print(f"    - Danh sách token phát âm: {tokens}")
    print(f"    - Tổng số token: {len(tokens)} từ")
    
    out_wav = os.path.join(AUDIO_OUT_DIR, "custom_output.wav")
    t0 = time.perf_counter()
    
    if method == "neural":
        print(f"[2] Phương pháp tổng hợp: NEURAL TTS (Giọng nói học sâu siêu tự nhiên)")
        wave, sr, latency, extra = synthesize_neural(text, voice=voice, sr=16000, out_wav_path=out_wav)
        dur = len(wave) / sr
        rtf = latency / dur if dur > 0 else 0.0
        print(f"[3] Kết quả tổng hợp:")
        print(f"    - File lưu tại: {out_wav}")
        print(f"    - Thời lượng audio: {dur:.2f} giây")
        print(f"    - Thời gian sinh (Latency): {latency*1000:.1f} ms")
        print(f"    - Hệ số Real-Time Factor (RTF): {rtf:.3f} (Nhanh hơn thời gian thực {1/rtf:.1f}x)")
        
    elif method == "unit_selection":
        print(f"[2] Phương pháp tổng hợp: CONCATENATIVE TTS (Lựa chọn đơn vị Viterbi DP + Cross-fade 15ms)")
        try:
            res = synthesize_concatenative(text, output_prefix="custom", xfade_ms=15.0)
            out_wav = res["best_path"]
            latency = time.perf_counter() - t0
            wave, sr = sf.read(out_wav)
            dur = len(wave) / sr
            rtf = latency / dur if dur > 0 else 0.0
            print(f"[3] Kết quả tổng hợp:")
            print(f"    - File lưu tại: {out_wav}")
            print(f"    - Candidate chuỗi được chọn: {res['best_candidates']}")
            print(f"    - Tổng chi phí tối ưu: {res['optimal_cost']:.2f}")
            print(f"    - Thời lượng audio: {dur:.2f} giây")
            print(f"    - Thời gian sinh (Latency): {latency*1000:.1f} ms")
            print(f"    - RTF: {rtf:.3f}")
        except Exception as e:
            print(f"[!] Lỗi do chứa từ ngoài từ điển miền hẹp: {e}")
            print("[→] Tự động chuyển sang Neural TTS để đọc câu này...")
            return run_tts(text, method="neural", voice=voice, auto_play=auto_play)
            
    elif method == "hmm":
        print(f"[2] Phương pháp tổng hợp: HMM-BASED PARAMETRIC SYNTHESIS (Mô hình nguồn - lọc)")
        try:
            tracks = generate_trajectories(tokens)
            wave = synthesize_source_filter_vocoder(tracks)
            sf.write(out_wav, wave, tracks["sr"])
            latency = time.perf_counter() - t0
            dur = len(wave) / tracks["sr"]
            rtf = latency / dur if dur > 0 else 0.0
            print(f"[3] Kết quả tổng hợp:")
            print(f"    - File lưu tại: {out_wav}")
            print(f"    - Số frames tham số: {len(tracks['times'])}")
            print(f"    - Thời lượng audio: {dur:.2f} giây")
            print(f"    - Thời gian sinh (Latency): {latency*1000:.1f} ms")
            print(f"    - RTF: {rtf:.3f}")
        except Exception as e:
            print(f"[!] Lỗi tham số HMM: {e}")
            print("[→] Chuyển sang Neural TTS...")
            return run_tts(text, method="neural", voice=voice, auto_play=auto_play)
            
    print("-" * 65)
    
    if auto_play:
        play_audio(out_wav)
        
    return out_wav

def interactive_menu():
    """Giao diện dòng lệnh tương tác trực quan."""
    while True:
        print("\n" + "=" * 70)
        print("    CSE457 • LAB 4: HỆ THỐNG TỔNG HỢP TIẾNG NÓI TƯƠNG TÁC (TTS)")
        print("    Sinh viên: Dương Tiến Đạt  |  MSSV: 2251172268")
        print("=" * 70)
        print("  [1] Nhập số thứ tự và số quầy (VD: Khách 52, Quầy 3)")
        print("  [2] Nhập câu văn bản tiếng Việt bất kỳ tùy ý")
        print("  [3] Chọn câu mẫu có sẵn trong hệ thống (Presets)")
        print("  [4] Tự động sinh câu ngẫu nhiên (Random Ticket & Counter)")
        print("  [5] So sánh chất lượng cả 3 phương pháp (Neural vs Unit Concat vs HMM)")
        print("  [0] Thoát chương trình")
        print("=" * 70)
        
        try:
            choice = input("👉 Mời bạn chọn chức năng [0 - 5]: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nĐã thoát chương trình.")
            break
            
        if choice == "0":
            print("Cảm ơn bạn đã sử dụng chương trình!")
            break
            
        elif choice == "1":
            try:
                ticket_str = input("Nhập số vé / số khách hàng (VD: 85): ").strip()
                counter_str = input("Nhập số quầy phục vụ (VD: 3): ").strip()
                if not ticket_str: ticket_str = "23"
                if not counter_str: counter_str = "5"
                text = f"Mời khách hàng số {ticket_str} đến quầy {counter_str}."
            except Exception:
                text = "Mời khách hàng số 23 đến quầy 5."
            run_tts(text, method="neural")
            
        elif choice == "2":
            text = input("Nhập câu văn bản tiếng Việt của bạn: ").strip()
            if not text:
                text = "Xin chào, mời khách hàng số 99 đến quầy 1."
            run_tts(text, method="neural")
            
        elif choice == "3":
            print("\nDanh sách câu mẫu có sẵn:")
            for i, p in enumerate(PRESET_SENTENCES, 1):
                print(f"  {i}. {p}")
            p_idx = input(f"Chọn câu mẫu [1 - {len(PRESET_SENTENCES)}]: ").strip()
            try:
                idx = int(p_idx) - 1
                text = PRESET_SENTENCES[max(0, min(idx, len(PRESET_SENTENCES) - 1))]
            except ValueError:
                text = PRESET_SENTENCES[0]
            run_tts(text, method="neural")
            
        elif choice == "4":
            ticket = random.randint(1, 999)
            counter = random.randint(1, 9)
            text = f"Mời khách hàng số {ticket} đến quầy {counter}."
            print(f"[🎲] Câu ngẫu nhiên được tạo: \"{text}\"")
            run_tts(text, method="neural")
            
        elif choice == "5":
            sample_text = "Mời khách hàng số 23 đến quầy 5."
            print(f"\n>>> SO SÁNH 3 PHƯƠNG PHÁP TRÊN CÙNG CÂU: \"{sample_text}\"")
            print("\n1. Phương pháp Neural TTS (Học sâu siêu tự nhiên):")
            run_tts(sample_text, method="neural", auto_play=True)
            time.sleep(1.0)
            
            print("\n2. Phương pháp Unit Selection (Ghép nối tối ưu Viterbi):")
            run_tts(sample_text, method="unit_selection", auto_play=True)
            time.sleep(1.0)
            
            print("\n3. Phương pháp HMM Parametric Synthesis (Nguồn - Lọc):")
            run_tts(sample_text, method="hmm", auto_play=True)

if __name__ == "__main__":
    # Nếu chạy có cờ --demo hoặc --text thì chạy không cần prompt
    if len(sys.argv) > 1:
        if "--text" in sys.argv:
            idx = sys.argv.index("--text")
            input_text = sys.argv[idx + 1] if idx + 1 < len(sys.argv) else "Mời khách hàng số 23 đến quầy 5."
            run_tts(input_text, method="neural", auto_play=False)
        elif "--auto" in sys.argv or "--demo" in sys.argv:
            print("Chạy chế độ Demo tự động sinh câu mẫu:")
            demo_text = "Mời khách hàng số 78 đến quầy 4."
            run_tts(demo_text, method="neural", auto_play=False)
    else:
        interactive_menu()
