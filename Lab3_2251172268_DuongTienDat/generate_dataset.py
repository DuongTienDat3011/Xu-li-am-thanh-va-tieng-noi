"""
generate_dataset.py
===================
CSE457 Lab 3 – Tạo dataset 5 lệnh tiếng Việt tách rời
Vocabulary: mo (mở), dong (đóng), len (lên), xuong (xuống), dung (dừng)

Cấu trúc:
  dataset/train/<label>/ : 6 file/từ  (_01.._06) → huấn luyện HMM
  dataset/test/<label>/  : 2 file/từ  (_07.._08) → đánh giá

Mỗi lần lặp có biến thể tự nhiên về:
  - Pitch F0 (±15%), tốc độ nói (±20%), biên độ (±10%)
  - Jitter (±2%), shimmer (±5%), noise nền nhỏ

Sinh viên: Dương Tiến Đạt – MSSV: 2251172268
"""

import os, wave, struct
import numpy as np
from scipy.signal import lfilter, resample_poly
from math import gcd

FS       = 16000
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET  = os.path.join(BASE_DIR, "dataset")

# ════════════════════════════════════════════════════════════
# Định nghĩa âm học 5 lệnh tiếng Việt
# Format: [(type, f0, dur_s, [(f1,bw1),(f2,bw2),(f3,bw3)], pre_sil, post_sil)]
# ════════════════════════════════════════════════════════════
WORDS = {
    # "mở" = /m/ + /ɘː/ (nguyên âm mở giữa)
    "mo": [
        ("nasal",    130, 0.10, [(250,60),(2200,200),(3200,250)], 0.25, 0.0),
        ("voiced",   130, 0.20, [(500,90),(1000,120),(2500,200)], 0.0,  0.28),
    ],
    # "đóng" = /d/ + /o/ + /ng/ (nasal cuối)
    "dong": [
        ("stop_voiced", 120, 0.07, [(200,200),(1000,300),(2500,400)], 0.25, 0.0),
        ("voiced",      120, 0.16, [(450,80),(700,90),(2400,180)],   0.0,  0.0),
        ("nasal",       120, 0.12, [(250,60),(2200,200),(3200,250)], 0.0,  0.28),
    ],
    # "lên" = /l/ + /eː/ + /n/ (nasal cuối)
    "len": [
        ("voiced",   135, 0.07, [(350,70),(2000,200),(3000,250)], 0.25, 0.0),
        ("voiced",   135, 0.18, [(390,70),(1990,180),(2550,200)], 0.0,  0.0),
        ("nasal",    135, 0.10, [(280,65),(2400,180),(3100,240)], 0.0,  0.28),
    ],
    # "xuống" = /s/ + /uː/ + /ŋ/ (fricative đầu)
    "xuong": [
        ("unvoiced",   0, 0.07, [(2000,600),(4500,700),(7000,800)], 0.25, 0.0),
        ("voiced",   115, 0.16, [(270,80),(2290,200),(3010,250)],  0.0,  0.0),
        ("nasal",    115, 0.12, [(250,60),(2200,200),(3200,250)],  0.0,  0.28),
    ],
    # "dừng" = /z/ + /ɨː/ + /ng/ (fricative hữu thanh)
    "dung": [
        ("voiced",   125, 0.07, [(350,100),(2200,300),(3500,400)], 0.25, 0.0),
        ("voiced",   125, 0.18, [(270,80),(2290,200),(3010,250)],  0.0,  0.0),
        ("nasal",    125, 0.12, [(250,60),(2200,200),(3200,250)],  0.0,  0.28),
    ],
}

# 8 biến thể (6 train + 2 test)
VARIATIONS = [
    (1.00, 1.00, 1.00, 38, 101),
    (0.88, 1.10, 0.95, 35, 202),
    (1.14, 0.92, 1.06, 40, 303),
    (0.93, 1.06, 0.97, 33, 404),
    (1.09, 0.95, 1.03, 42, 505),
    (0.97, 1.03, 0.99, 37, 606),
    # test:
    (1.05, 0.98, 1.01, 36, 707),
    (0.91, 1.07, 0.96, 41, 808),
]

def _glottal(f0, n, fs, rng):
    src = np.zeros(n); period = fs/f0; t = 0.0
    while t < n:
        idx = int(t)
        if idx < n: src[idx] += rng.uniform(0.95,1.05)
        t += period*(1+rng.uniform(-0.02,0.02))
    return src

def _formant(src, formants, fs):
    out = np.zeros(len(src))
    for (fc,bw) in formants:
        if fc >= fs/2: continue
        r = np.exp(-np.pi*bw/fs); th = 2*np.pi*fc/fs
        out += lfilter([1-r],[1,-2*r*np.cos(th),r*r], src)
    return out

def _fade(sig, fs, fi=10, fo=15):
    fi_n = min(int(fi*1e-3*fs),len(sig)//4)
    fo_n = min(int(fo*1e-3*fs),len(sig)//4)
    env = np.ones(len(sig))
    env[:fi_n] = np.linspace(0,1,fi_n)
    env[len(sig)-fo_n:] = np.linspace(1,0,fo_n)
    return sig*env

def synth_word(phonemes, fs, speed=1.0, f0_scale=1.0, amp=1.0, snr_db=38, rng=None):
    if rng is None: rng = np.random.default_rng(42)
    segs = []
    for (ptype,f0,dur,fmts,pre,post) in phonemes:
        if pre > 0: segs.append(np.zeros(int(pre*fs)))
        n = max(32, int(dur/speed*fs))
        f0c = f0*f0_scale
        if ptype in ("voiced","nasal","stop_voiced"):
            if ptype == "stop_voiced":
                bn = int(0.02*fs)
                burst = _formant(rng.standard_normal(bn)*0.3,[(1200,800),(2800,1000)],fs)
                vs = _glottal(f0c, n-bn, fs, rng)
                seg = np.concatenate([_fade(burst,fs,5,5), _formant(vs,fmts,fs)])
            else:
                src = _glottal(f0c, n, fs, rng)
                seg = _formant(src, fmts, fs)
                if ptype=="nasal": seg *= 0.6
        else:
            seg = _formant(rng.standard_normal(n)*0.5, fmts, fs)
        seg = _fade(seg, fs)
        segs.append(seg)
        if post > 0: segs.append(np.zeros(int(post*fs)))
    sig = np.concatenate(segs).astype(np.float64)
    sp = np.mean(sig**2)+1e-10
    sig += rng.standard_normal(len(sig))*np.sqrt(sp/10**(snr_db/10))
    pk = np.max(np.abs(sig))
    return np.clip(sig/pk*amp*0.85,-1,1) if pk>0 else sig

def save_wav(sig, fs, path):
    s16 = (np.clip(sig,-1,1)*32767).astype(np.int16)
    with wave.open(path,"w") as wf:
        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(fs)
        wf.writeframes(struct.pack(f"<{len(s16)}h",*s16))

def generate_all():
    print("="*60)
    print("  CSE457 Lab 3 – Tạo dataset 5 lệnh × 8 lần")
    print("  Train: _01.._06 | Test: _07.._08")
    print("="*60)
    total = 0
    for word, phonemes in WORDS.items():
        print(f"\n  [{word.upper()}]")
        for rep,(speed,f0s,amps,snr,seed) in enumerate(VARIATIONS, 1):
            rng  = np.random.default_rng(seed)
            sig  = synth_word(phonemes, FS, speed=speed, f0_scale=f0s,
                               amp=amps, snr_db=snr, rng=rng)
            split = "train" if rep <= 6 else "test"
            folder = os.path.join(DATASET, split, word)
            os.makedirs(folder, exist_ok=True)
            fname  = f"{word}_{rep:02d}.wav"
            fpath  = os.path.join(folder, fname)
            save_wav(sig, FS, fpath)
            dur  = len(sig)/FS
            size = os.path.getsize(fpath)
            print(f"    [{split}] {fname}  {dur:.3f}s  {size//1024}KB")
            total += 1
    print(f"\n  ✓ Tổng: {total} file WAV")
    print(f"    Train: {total//8*6} | Test: {total//8*2}")

if __name__ == "__main__":
    generate_all()
