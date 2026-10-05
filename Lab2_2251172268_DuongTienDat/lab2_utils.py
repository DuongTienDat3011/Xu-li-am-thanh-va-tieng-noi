"""
lab2_utils.py
=============
CSE457 Lab 2 – Tiện ích dùng chung cho toàn bộ pipeline
Sinh viên: Dương Tiến Đạt – MSSV: 2251172268

Bao gồm:
  - Đọc WAV, resample về 16 kHz
  - Framing + Hamming window
  - Short-time energy, magnitude, RMS
  - Zero-Crossing Rate (ZCR)
  - Short-time autocorrelation + ước lượng F0
  - Tiện ích vẽ hình (matplotlib Agg)
"""

import os, wave, struct
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.signal import resample_poly
from math import gcd

# ── Đường dẫn chuẩn ────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
FIGURES_DIR = os.path.join(BASE_DIR, "figures")
TRIM_DIR    = os.path.join(BASE_DIR, "audio_trimmed")
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(TRIM_DIR,    exist_ok=True)

# ── Hằng số tham số baseline ────────────────────────────────
FS         = 16000    # Sampling rate (Hz)
FRAME_MS   = 25       # Frame length (ms)
HOP_MS     = 10       # Hop size (ms)
WIN_LEN    = int(FS * FRAME_MS / 1000)   # 400 mẫu
HOP_LEN    = int(FS * HOP_MS   / 1000)   # 160 mẫu
PRE_EMPH   = 0.97     # Pre-emphasis coefficient
N_FFT      = 512      # FFT points
N_MELS     = 24       # Mel filterbank filters
N_MFCC     = 13       # MFCC coefficients

LABELS     = ["khong", "mot", "hai", "ba", "bon"]
LABEL_VI   = {"khong": "Không", "mot": "Một",
              "hai": "Hai", "ba": "Ba", "bon": "Bốn"}

# ════════════════════════════════════════════════════════════
# 1. ĐỌC WAV
# ════════════════════════════════════════════════════════════

def read_wav(path: str, target_fs: int = FS) -> np.ndarray:
    """
    Đọc WAV PCM → float64 [-1,1], resample về target_fs nếu cần.
    Trả về mảng 1D (mono).
    """
    with wave.open(path, "r") as wf:
        fs   = wf.getframerate()
        ch   = wf.getnchannels()
        sw   = wf.getsampwidth()
        bits = sw * 8
        nfr  = wf.getnframes()
        raw  = wf.readframes(nfr)

    dtype = {8: np.uint8, 16: np.int16, 32: np.int32}.get(bits, np.int16)
    samps = np.frombuffer(raw, dtype=dtype)
    if ch > 1:
        samps = samps.reshape(-1, ch)

    mv  = 128.0 if bits == 8 else float(2 ** (bits - 1))
    x   = (samps.astype(np.float64) - (128 if bits == 8 else 0)) / mv
    x   = np.clip(x, -1.0, 1.0)
    if ch > 1:
        x = np.mean(x, axis=1)
    else:
        x = x.flatten()

    # Resample nếu cần
    if fs != target_fs:
        g = gcd(fs, target_fs)
        x = resample_poly(x, target_fs // g, fs // g).astype(np.float64)
        x = np.clip(x, -1.0, 1.0)

    # Normalize full-scale
    peak = np.max(np.abs(x))
    if peak > 0:
        x = x / peak

    return x


def write_wav(signal: np.ndarray, fs: int, path: str) -> None:
    """Lưu float64 [-1,1] → WAV 16-bit mono."""
    s16 = (np.clip(signal, -1.0, 1.0) * 32767).astype(np.int16)
    with wave.open(path, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(fs)
        wf.writeframes(struct.pack(f"<{len(s16)}h", *s16))


# ════════════════════════════════════════════════════════════
# 2. PRE-EMPHASIS
# ════════════════════════════════════════════════════════════

def pre_emphasis(x: np.ndarray, alpha: float = PRE_EMPH) -> np.ndarray:
    """
    Bộ lọc tiền nhấn: y[n] = x[n] - α·x[n-1]
    Tăng tương đối thành phần tần số cao trước phân tích phổ.
    alpha ≈ 0.97 là giá trị chuẩn theo Rabiner-Schafer.
    """
    y       = np.empty_like(x)
    y[0]    = x[0]
    y[1:]   = x[1:] - alpha * x[:-1]
    return y


# ════════════════════════════════════════════════════════════
# 3. FRAMING + HAMMING WINDOW
# ════════════════════════════════════════════════════════════

def framing(x: np.ndarray, win_len: int = WIN_LEN,
            hop_len: int = HOP_LEN) -> np.ndarray:
    """
    Chia tín hiệu x thành các frame chồng lấn.
    Áp cửa sổ Hamming lên từng frame.

    Parameters
    ----------
    x       : tín hiệu 1D float64
    win_len : số mẫu/frame (L)
    hop_len : số mẫu bước nhảy (R)

    Returns
    -------
    frames  : ndarray shape (n_frames, win_len)
    t_frames: thời gian tâm frame (giây)
    """
    N        = len(x)
    n_frames = max(1, 1 + (N - win_len) // hop_len)
    hamming  = np.hamming(win_len)

    frames   = np.zeros((n_frames, win_len), dtype=np.float64)
    t_frames = np.zeros(n_frames)

    for i in range(n_frames):
        start       = i * hop_len
        end         = start + win_len
        seg         = x[start:min(end, N)]
        if len(seg) < win_len:
            seg = np.pad(seg, (0, win_len - len(seg)))
        frames[i]   = seg * hamming
        t_frames[i] = (start + win_len / 2) / FS

    return frames, t_frames


# ════════════════════════════════════════════════════════════
# 4. SHORT-TIME ENERGY, MAGNITUDE, RMS
# ════════════════════════════════════════════════════════════

def short_time_energy(frames: np.ndarray) -> np.ndarray:
    """
    Short-time energy: E_r = Σ x_r[n]²
    Nhấn mạnh các mẫu biên độ lớn (bình phương).
    """
    return np.sum(frames ** 2, axis=1)


def short_time_magnitude(frames: np.ndarray) -> np.ndarray:
    """Short-time magnitude: M_r = Σ |x_r[n]|"""
    return np.sum(np.abs(frames), axis=1)


def short_time_rms(frames: np.ndarray) -> np.ndarray:
    """RMS: RMS_r = √(1/L · Σ x_r[n]²)"""
    return np.sqrt(np.mean(frames ** 2, axis=1))


def log_energy(frames: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """Log-energy: E_r(dB) = 10·log10(E_r + ε)"""
    return 10.0 * np.log10(short_time_energy(frames) + eps)


# ════════════════════════════════════════════════════════════
# 5. ZERO-CROSSING RATE (ZCR)
# ════════════════════════════════════════════════════════════

def zero_crossing_rate(frames: np.ndarray) -> np.ndarray:
    """
    ZCR: Z_r = 1/(2L) · Σ |sgn(x[m]) - sgn(x[m-1])|

    Khi tín hiệu đổi dấu: |sgn(x[m]) - sgn(x[m-1])| = 2
    → Chia 2L → số lần cắt zero trung bình trên mẫu.

    Voiced: ZCR thấp (~0.02–0.05 với F0=130Hz, Fs=16kHz)
    Unvoiced/fricative: ZCR cao (~0.1–0.4)
    Silence: ZCR gần 0 hoặc ngẫu nhiên thấp
    """
    L   = frames.shape[1]
    sgn = np.sign(frames)
    # Xử lý mẫu = 0 → gán +1 (theo Rabiner-Schafer)
    sgn[sgn == 0] = 1.0
    diffs = np.abs(np.diff(sgn, axis=1))   # shape (n_frames, L-1)
    zcr   = np.sum(diffs, axis=1) / (2.0 * L)
    return zcr


# ════════════════════════════════════════════════════════════
# 6. SHORT-TIME AUTOCORRELATION + PITCH
# ════════════════════════════════════════════════════════════

def short_time_autocorr(frame: np.ndarray,
                         max_lag: int = None) -> np.ndarray:
    """
    Tự tương quan ngắn hạn của 1 frame:
      R[k] = Σ_n x_r[n] · x_r[n+k]

    Dùng để phát hiện tính tuần hoàn (voiced/unvoiced)
    và ước lượng pitch F0.
    """
    L = len(frame)
    if max_lag is None:
        max_lag = L // 2

    r = np.array([
        np.sum(frame[:L-k] * frame[k:L]) if k < L else 0.0
        for k in range(max_lag + 1)
    ])
    # Normalize by R[0]
    if r[0] > 0:
        r = r / r[0]
    return r


def estimate_pitch(frame: np.ndarray, fs: int = FS,
                   f0_min: float = 60.0,
                   f0_max: float = 400.0) -> float:
    """
    Ước lượng F0 bằng autocorrelation:
      F0 = Fs / N0  (N0 = lag của đỉnh autocorr đầu tiên)

    Returns: F0 (Hz) hoặc 0 nếu không tìm thấy đỉnh
    """
    lag_min = int(fs / f0_max)
    lag_max = int(fs / f0_min)
    r       = short_time_autocorr(frame, max_lag=lag_max)

    if len(r) <= lag_min:
        return 0.0

    search  = r[lag_min:lag_max + 1]
    if len(search) == 0:
        return 0.0

    peak_idx = int(np.argmax(search))
    N0       = peak_idx + lag_min
    if N0 == 0 or r[N0] < 0.3:   # threshold voiced/unvoiced
        return 0.0

    return float(fs) / float(N0)


# ════════════════════════════════════════════════════════════
# 7. TIỆN ÍCH VẼ HÌNH
# ════════════════════════════════════════════════════════════

def savefig(fig: plt.Figure, name: str) -> str:
    """Lưu figure vào figures/ với tên cho trước. Trả về đường dẫn."""
    path = os.path.join(FIGURES_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {path}")
    return path


def style_ax(ax: plt.Axes, xlabel: str = "", ylabel: str = "",
             title: str = "", grid: bool = True) -> None:
    """Style thống nhất."""
    if xlabel: ax.set_xlabel(xlabel, fontsize=9)
    if ylabel: ax.set_ylabel(ylabel, fontsize=9)
    if title : ax.set_title(title,  fontsize=10, fontweight="bold")
    if grid  : ax.grid(True, alpha=0.3, linewidth=0.5)
    ax.tick_params(labelsize=8)


def db(v: float, floor: float = -120.0) -> float:
    """20·log10(|v|) với floor."""
    return float(np.maximum(20 * np.log10(abs(v) + 1e-15), floor))


# ════════════════════════════════════════════════════════════
# 8. TIỆN ÍCH DATASET
# ════════════════════════════════════════════════════════════

def get_train_test_files(root: str = DATASET_DIR,
                          labels: list = LABELS,
                          n_train: int = 3):
    """
    Chia file theo quy ước: file _01.._n_train → train, còn lại → test.
    Trả về:
      train_files: dict {label: [path, ...]}
      test_files : dict {label: [path, ...]}
    """
    import pathlib
    train_files = {lab: [] for lab in labels}
    test_files  = {lab: [] for lab in labels}

    for lab in labels:
        folder = pathlib.Path(root) / lab
        files  = sorted(folder.glob("*.wav"))
        for i, f in enumerate(files):
            if i < n_train:
                train_files[lab].append(str(f))
            else:
                test_files[lab].append(str(f))

    return train_files, test_files


def dataset_info(root: str = DATASET_DIR, labels: list = LABELS) -> None:
    """In thông tin dataset."""
    import pathlib
    print("\n  Dataset: {} words × {} reps".format(
        len(labels), 5))
    print("  " + "─" * 55)
    print(f"  {'Label':<8} {'N files':<10} {'Fs':<8} {'Dur avg (s)'}")
    print("  " + "─" * 55)
    for lab in labels:
        folder = pathlib.Path(root) / lab
        files  = sorted(folder.glob("*.wav"))
        durs   = []
        for f in files:
            x = read_wav(str(f))
            durs.append(len(x) / FS)
        print(f"  {lab:<8} {len(files):<10} {FS:<8} {np.mean(durs):.3f} s")
    print("  " + "─" * 55)
