"""
lab3_utils.py
=============
CSE457 Lab 3 – Tiện ích dùng chung
Sinh viên: Dương Tiến Đạt – MSSV: 2251172268
"""

import os, wave, struct, pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import resample_poly
from math import gcd

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
FIGURES_DIR = os.path.join(BASE_DIR, "figures")
MODELS_DIR  = os.path.join(BASE_DIR, "models")
for d in [FIGURES_DIR, MODELS_DIR]:
    os.makedirs(d, exist_ok=True)

FS        = 16000
LABELS    = ["mo", "dong", "len", "xuong", "dung"]
LABEL_VI  = {"mo":"Mở","dong":"Đóng","len":"Lên","xuong":"Xuống","dung":"Dừng"}

# ── Đọc WAV ──────────────────────────────────────────────
def read_wav(path, target_fs=FS):
    with wave.open(path,"r") as wf:
        fs=wf.getframerate(); ch=wf.getnchannels()
        sw=wf.getsampwidth(); bits=sw*8
        raw=wf.readframes(wf.getnframes())
    dt={8:np.uint8,16:np.int16,32:np.int32}.get(bits,np.int16)
    s=np.frombuffer(raw,dtype=dt)
    if ch>1: s=s.reshape(-1,ch)
    mv=128.0 if bits==8 else float(2**(bits-1))
    x=(s.astype(np.float64)-(128 if bits==8 else 0))/mv
    x=np.clip(x,-1,1)
    if ch>1: x=np.mean(x,axis=1)
    else:    x=x.flatten()
    if fs!=target_fs:
        g=gcd(fs,target_fs); x=resample_poly(x,target_fs//g,fs//g).astype(np.float64)
        x=np.clip(x,-1,1)
    pk=np.max(np.abs(x)); return x/pk if pk>0 else x

def write_wav(sig, fs, path):
    s16=(np.clip(sig,-1,1)*32767).astype(np.int16)
    with wave.open(path,"w") as wf:
        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(fs)
        wf.writeframes(struct.pack(f"<{len(s16)}h",*s16))

# ── Lấy danh sách file ──────────────────────────────────
def get_files():
    train,test={},{}
    for lab in LABELS:
        tp=sorted(pathlib.Path(DATASET_DIR,"train",lab).glob("*.wav"))
        ts=sorted(pathlib.Path(DATASET_DIR,"test", lab).glob("*.wav"))
        train[lab]=[str(p) for p in tp]
        test[lab] =[str(p) for p in ts]
    return train, test

# ── Tiện ích vẽ ─────────────────────────────────────────
def savefig(fig, name):
    path=os.path.join(FIGURES_DIR,name)
    fig.savefig(path,dpi=150,bbox_inches="tight")
    plt.close(fig); print(f"  → {path}")

def style_ax(ax, xlabel="", ylabel="", title="", grid=True):
    if xlabel: ax.set_xlabel(xlabel,fontsize=9)
    if ylabel: ax.set_ylabel(ylabel,fontsize=9)
    if title : ax.set_title(title,fontsize=10,fontweight="bold")
    if grid  : ax.grid(True,alpha=0.3)
    ax.tick_params(labelsize=8)
