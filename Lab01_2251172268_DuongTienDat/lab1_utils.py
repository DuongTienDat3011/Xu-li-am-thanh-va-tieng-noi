"""
lab1_utils.py
=============
CSE457 Lab 1 – Các hàm tiện ích dùng chung cho mọi phần (A–G).
Không chạy độc lập. Được import bởi lab1_part_*.py và lab1_main.py.
"""

import os
import wave
import struct
import numpy as np
import matplotlib
matplotlib.use("Agg")           # Chế độ không GUI: vẽ và lưu ra file PNG
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ── Thiết lập đường dẫn thư mục chuẩn ──────────────────────
# BASE_DIR    : thư mục gốc chứa tất cả các file Lab 1
# AUDIO_DIR   : thư mục chứa file âm thanh WAV đầu vào/đầu ra
# FIGURES_DIR : thư mục chứa hình đồ thị PNG
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
AUDIO_DIR   = os.path.join(BASE_DIR, "audio")
FIGURES_DIR = os.path.join(BASE_DIR, "figures")
os.makedirs(AUDIO_DIR,   exist_ok=True)   # Tự tạo nếu chưa có
os.makedirs(FIGURES_DIR, exist_ok=True)


# ════════════════════════════════════════════════════════════
# ĐỌC / GHI FILE WAV
# Hỗ trợ 8-bit, 16-bit, 24-bit, 32-bit PCM
# ════════════════════════════════════════════════════════════

def read_wav(filepath: str) -> dict:
    """
    Đọc file WAV và trả về dict chứa toàn bộ thông tin cần dùng.

    Xử lý:
      1. Mở file bằng thư viện wave (đọc header + raw bytes)
      2. Giải mã bytes thành mảng numpy đúng kiểu (int8/16/32)
      3. Chuẩn hóa về [-1.0, 1.0] (full-scale normalization)
      4. Tạo bản mono (trung bình kênh nếu stereo)

    Returns
    -------
    dict với các key:
      x          : ndarray float64 – tín hiệu đã chuẩn hóa, shape (N,) hoặc (N,C)
      x_norm     : ndarray float64 [-1,1] – alias của x
      x_mono     : ndarray float64 [-1,1] – kênh mono
      fs         : int   – sampling rate (Hz)
      channels   : int   – số kênh (1=mono, 2=stereo)
      bit_depth  : int   – bit/sample (8/16/24/32)
      n_frames   : int   – tổng số frame
      duration   : float – thời lượng (giây)
      sample_width: int  – bytes/sample
      file_size  : int   – kích thước file (bytes)
      filename   : str   – tên file không có đường dẫn
      filepath   : str   – đường dẫn đầy đủ
    """
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"Không tìm thấy file: {filepath}")

    file_size = os.path.getsize(filepath)

    # ── Đọc header WAV ──
    with wave.open(filepath, "r") as wf:
        fs           = wf.getframerate()    # Sampling rate (Hz)
        channels     = wf.getnchannels()    # Số kênh
        sample_width = wf.getsampwidth()    # Bytes/sample
        bit_depth    = sample_width * 8     # Bit/sample
        n_frames     = wf.getnframes()      # Tổng số frame
        raw_bytes    = wf.readframes(n_frames)  # Đọc toàn bộ dữ liệu PCM

    # ── Chọn kiểu dữ liệu numpy theo bit depth ──
    if bit_depth == 8:
        dtype = np.uint8    # 8-bit: unsigned [0, 255]
    elif bit_depth == 16:
        dtype = np.int16    # 16-bit: signed [-32768, 32767]
    elif bit_depth == 24:
        # 24-bit PCM: numpy không có kiểu int24, phải đọc thủ công
        raw_arr    = np.frombuffer(raw_bytes, dtype=np.uint8)
        n_samps    = len(raw_bytes) // 3
        samples_32 = np.zeros(n_samps, dtype=np.int32)

        # Ghép 3 byte liên tiếp thành int32 (little-endian)
        for i in range(n_samps):
            b0, b1, b2 = raw_arr[i*3], raw_arr[i*3+1], raw_arr[i*3+2]
            val = (b2 << 16) | (b1 << 8) | b0
            if val >= (1 << 23):    # Xử lý số âm (two's complement)
                val -= (1 << 24)
            samples_32[i] = val

        # Reshape và chuẩn hóa
        samples = samples_32.reshape(-1, channels) if channels > 1 else samples_32
        max_val = float(2 ** 23)
        x       = samples.astype(np.float64) / max_val
        x_norm  = np.clip(x, -1.0, 1.0)
        x_mono  = np.mean(x_norm, axis=1) if channels > 1 else x_norm.flatten()
        return dict(x=x, x_norm=x_norm, x_mono=x_mono,
                    fs=fs, channels=channels, bit_depth=bit_depth,
                    n_frames=n_frames, duration=n_frames/fs,
                    sample_width=sample_width, file_size=file_size,
                    filename=os.path.basename(filepath), filepath=filepath)
    elif bit_depth == 32:
        dtype = np.int32    # 32-bit: signed
    else:
        raise ValueError(f"Bit-depth {bit_depth} không hỗ trợ")

    # ── Giải mã bytes → mảng số nguyên ──
    samples = np.frombuffer(raw_bytes, dtype=dtype)
    if channels > 1:
        samples = samples.reshape(-1, channels)   # Shape (N, C) nếu multi-channel

    # ── Chuẩn hóa về [-1.0, 1.0] ──
    # 8-bit unsigned [0,255] → [-1,1]: trừ 128 rồi chia 128
    # 16/32-bit signed: chia cho 2^(B-1)
    if bit_depth == 8:
        x_norm = (samples.astype(np.float64) - 128.0) / 128.0
    else:
        max_val = float(2 ** (bit_depth - 1))
        x_norm  = samples.astype(np.float64) / max_val

    x_norm = np.clip(x_norm, -1.0, 1.0)    # Giới hạn tránh tràn số

    # ── Tạo bản mono ──
    # Stereo: trung bình kênh trái và phải → giảm số mẫu từ (N,2) → (N,)
    x_mono = np.mean(x_norm, axis=1) if channels > 1 else x_norm.flatten()

    return dict(
        x           = x_norm,
        x_norm      = x_norm,
        x_mono      = x_mono,
        fs          = fs,
        channels    = channels,
        bit_depth   = bit_depth,
        n_frames    = n_frames,
        duration    = n_frames / fs,
        sample_width= sample_width,
        file_size   = file_size,
        filename    = os.path.basename(filepath),
        filepath    = filepath,
    )


def write_wav(signal: np.ndarray, fs: int, filename: str,
              bit_depth: int = 16) -> str:
    """
    Lưu mảng float64 [-1,1] thành file WAV PCM mono.
    Hỗ trợ 8-bit, 16-bit, 32-bit.

    Parameters:
        signal    : tín hiệu 1D float64 trong [-1, 1]
        fs        : sampling rate (Hz)
        filename  : tên file (chỉ tên, không cần đường dẫn)
        bit_depth : độ sâu bit (8, 16, hoặc 32)

    Returns:
        Đường dẫn tuyệt đối của file WAV đã lưu
    """
    path = os.path.join(AUDIO_DIR, filename)
    sig  = np.clip(signal, -1.0, 1.0)   # Clip để tránh clipping khi ghi

    # Chuyển float → int theo bit depth
    if bit_depth == 16:
        s    = (sig * 32767).astype(np.int16)
        sw   = 2
        fmt  = f"<{len(s)}h"
        data = struct.pack(fmt, *s)
    elif bit_depth == 8:
        # 8-bit: ánh xạ [-1,1] → [0,255]
        s    = ((sig + 1.0) * 127.5).astype(np.uint8)
        sw   = 1
        data = s.tobytes()
    elif bit_depth == 32:
        s    = (sig * 2147483647).astype(np.int32)
        sw   = 4
        fmt  = f"<{len(s)}i"
        data = struct.pack(fmt, *s)
    else:
        raise ValueError(f"bit_depth {bit_depth} không hỗ trợ")

    # Ghi file WAV
    with wave.open(path, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(sw)
        wf.setframerate(fs)
        wf.writeframes(data)

    return path


# ════════════════════════════════════════════════════════════
# CHỌN FILE ÂM THANH (menu terminal tương tác)
# Hiện danh sách file trong audio/ cho người dùng lựa chọn
# ════════════════════════════════════════════════════════════

def choose_audio_file(default_idx: int = 0) -> str:
    """
    Liệt kê tất cả file audio trong thư mục audio/ và cho người dùng chọn.
    Hỗ trợ định dạng: .wav, .mp3, .flac, .ogg.
    Nếu chưa có file nào → tự động tạo file sine mẫu.

    Parameters:
        default_idx : chỉ số mặc định (0-based) nếu người dùng nhấn Enter

    Returns:
        Đường dẫn đầy đủ đến file âm thanh đã chọn
    """
    exts  = (".wav", ".mp3", ".flac", ".ogg")
    files = sorted([f for f in os.listdir(AUDIO_DIR)
                    if f.lower().endswith(exts)])

    # Nếu chưa có file nào → tạo file sine mẫu để tiếp tục
    if not files:
        print("  ⚠ Chưa có file âm thanh. Đang tạo file mẫu...")
        _create_default_audio()
        files = sorted([f for f in os.listdir(AUDIO_DIR)
                        if f.lower().endswith(exts)])

    # In danh sách file kèm thông tin
    print("\n" + "─" * 60)
    print("  CHỌN FILE ÂM THANH ĐỂ PHÂN TÍCH")
    print("─" * 60)
    for i, f in enumerate(files, 1):
        p  = os.path.join(AUDIO_DIR, f)
        sz = os.path.getsize(p)
        try:
            with wave.open(p, "r") as wf:
                dur = wf.getnframes() / wf.getframerate()
                fs  = wf.getframerate()
            meta = f"  Fs={fs}Hz  {dur:.2f}s"
        except Exception:
            meta = ""
        print(f"  [{i}] {f:<40}{sz//1024:>5}KB{meta}")

    print(f"  [{len(files)+1}] Nhập đường dẫn file khác")
    print("─" * 60)

    # Vòng lặp xử lý đầu vào người dùng
    while True:
        raw = input(f"  Chọn [1–{len(files)+1}], Enter={default_idx+1}: ").strip()
        if raw == "":
            idx = default_idx            # Dùng mặc định khi Enter
        else:
            try:
                idx = int(raw) - 1
            except ValueError:
                print("  Nhập số hợp lệ.")
                continue

        if 0 <= idx < len(files):
            return os.path.join(AUDIO_DIR, files[idx])
        elif idx == len(files):
            # Cho phép nhập đường dẫn tùy ý
            custom = input("  Nhập đường dẫn: ").strip().strip('"')
            if os.path.isfile(custom):
                return custom
            print("  ✗ File không tồn tại.")
        else:
            print("  ✗ Lựa chọn không hợp lệ.")


def _create_default_audio():
    """Tạo nhanh 1 file sine 440Hz để không bị lỗi khi audio/ trống."""
    from generate_audio import make_sine_wave, save_wav_float
    save_wav_float(make_sine_wave(440), 22050, "sine_440hz.wav")


# ════════════════════════════════════════════════════════════
# TIỆN ÍCH ĐỒ THỊ (Matplotlib helpers)
# Dùng chung cho tất cả phần để đồng nhất giao diện hình vẽ
# ════════════════════════════════════════════════════════════

def savefig(fig: plt.Figure, name: str) -> None:
    """
    Lưu figure vào thư mục figures/ với tên cho trước.
    Tự động đóng figure sau khi lưu để giải phóng bộ nhớ.
    In đường dẫn file vừa lưu ra console.
    """
    path = os.path.join(FIGURES_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")  # 150 DPI đủ chất lượng in
    plt.close(fig)
    print(f"  → Hình: {path}")


def style_ax(ax: plt.Axes, xlabel: str = "", ylabel: str = "",
             title: str = "", grid: bool = True) -> None:
    """
    Áp dụng style thống nhất cho một Axes:
      - Font size 9 cho nhãn trục
      - Font size 10 bold cho tiêu đề
      - Grid mờ (alpha=0.3) nếu grid=True
    """
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9)
    if title:
        ax.set_title(title, fontsize=10, fontweight="bold")
    if grid:
        ax.grid(True, alpha=0.3, linewidth=0.5)
    ax.tick_params(labelsize=8)


def db(x: np.ndarray, ref: float = 1.0, floor: float = -120.0) -> np.ndarray:
    """
    Chuyển biên độ tuyến tính sang decibel:
      dB = 20 · log10(|x| / ref)
    Sử dụng floor để tránh log(0) → -inf.
    Ví dụ: db(0.5) ≈ -6.02 dBFS
    """
    return np.maximum(20 * np.log10(np.abs(x) / ref + 1e-15), floor)


def power_db(x: np.ndarray, floor: float = -120.0) -> np.ndarray:
    """
    Chuyển công suất (power) sang decibel:
      power_dB = 10 · log10(x)
    Dùng cho power spectrum (x = |X|²).
    """
    return np.maximum(10 * np.log10(x + 1e-15), floor)
