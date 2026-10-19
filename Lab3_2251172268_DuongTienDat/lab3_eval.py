"""
lab3_eval.py
============
CSE457 Lab 3 – Phần E, F, G, H: So sánh DTW, Lexicon, LM re-ranking, Đánh giá
Sinh viên: Dương Tiến Đạt – MSSV: 2251172268
"""

import os, math, time, csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score

from lab3_utils import (
    FS, LABELS, LABEL_VI, FIGURES_DIR,
    savefig, style_ax, get_files
)
from lab3_features import extract_feat, HOP_LEN
from lab3_hmm import GaussianHMM_LR, recognize


# ═══════════════════════════════════════════════════════════
# E. SO SÁNH HMM VỚI DTW BASELINE
# ═══════════════════════════════════════════════════════════

def dtw_distance(X, Y):
    """DTW chuẩn (Euclidean local distance, 3-step DP)."""
    N,M = len(X),len(Y)
    C = np.array([[np.sqrt(np.sum((X[i]-Y[j])**2)) for j in range(M)]
                  for i in range(N)])
    D = np.full((N+1,M+1),np.inf); D[0,0]=0.
    back = np.zeros((N+1,M+1,2),dtype=np.int32)
    for i in range(1,N+1):
        for j in range(1,M+1):
            loc=C[i-1,j-1]
            cands=[(D[i-1,j],i-1,j),(D[i,j-1],i,j-1),(D[i-1,j-1],i-1,j-1)]
            bv,pi,pj=min(cands,key=lambda z:z[0])
            D[i,j]=loc+bv; back[i,j]=(pi,pj)
    path=[]; ci,cj=N,M
    while ci>0 or cj>0:
        path.append((ci-1,cj-1)); ci,cj=back[ci,cj]
    path.reverse()
    return D[N,M]/max(len(path),1)


def evaluate_system(models_or_templates, test_files, method="hmm",
                    apply_trim=True) -> tuple:
    """Đánh giá accuracy và tính thời gian test/trial."""
    y_true, y_pred = [], []
    t0 = time.perf_counter()
    for lab in LABELS:
        for fp in test_files[lab]:
            feat = extract_feat(fp, apply_trim=apply_trim)
            if method == "hmm":
                pred, _ = recognize(feat, models_or_templates)
            else:  # dtw
                scores = {}
                for l, refs in models_or_templates.items():
                    scores[l] = min(dtw_distance(feat, R) for R in refs)
                pred = min(scores, key=scores.get)
            y_true.append(lab); y_pred.append(pred)
    elapsed = time.perf_counter() - t0
    acc = accuracy_score(y_true, y_pred)
    t_per = elapsed / len(y_true) * 1000
    return y_true, y_pred, acc, t_per


def plot_comparison(hmm_res, dtw_res, tag="") -> None:
    """Vẽ confusion matrix và bảng so sánh HMM vs DTW."""
    fig, axes = plt.subplots(1,2, figsize=(14,7))
    for ax, (yt,yp,acc,tms,title) in zip(axes, [
        (*hmm_res[:2], hmm_res[2], hmm_res[3],
         f"HMM Gaussian\nAccuracy={hmm_res[2]*100:.1f}%  {hmm_res[3]:.1f}ms/trial"),
        (*dtw_res[:2], dtw_res[2], dtw_res[3],
         f"DTW Nearest-Template\nAccuracy={dtw_res[2]*100:.1f}%  {dtw_res[3]:.1f}ms/trial"),
    ]):
        cm = confusion_matrix(yt,yp,labels=LABELS)
        im = ax.imshow(cm, cmap="Blues", vmin=0)
        plt.colorbar(im,ax=ax,fraction=0.046)
        for i in range(len(LABELS)):
            for j in range(len(LABELS)):
                clr="white" if cm[i,j]>cm.max()*.6 else "black"
                ax.text(j,i,str(cm[i,j]),ha="center",va="center",
                        fontsize=12,fontweight="bold",color=clr)
        vil=[LABEL_VI[l] for l in LABELS]
        ax.set_xticks(range(len(LABELS))); ax.set_xticklabels(vil,fontsize=9)
        ax.set_yticks(range(len(LABELS))); ax.set_yticklabels(vil,fontsize=9)
        ax.set_xlabel("Dự đoán"); ax.set_ylabel("Nhãn thật")
        ax.set_title(title,fontsize=10,fontweight="bold")
    fig.suptitle(f"Phần E – So sánh HMM vs DTW{tag}",fontsize=12,fontweight="bold")
    plt.tight_layout()
    savefig(fig, f"E_hmm_vs_dtw{tag}.png")


def run_part_e(models, train_files, test_files) -> None:
    print("\n"+"═"*60)
    print("  PHẦN E – SO SÁNH HMM VỚI DTW BASELINE")
    print("═"*60)

    # Build DTW templates (3 file đầu = train)
    dtw_templates = {}
    for lab in LABELS:
        dtw_templates[lab] = [extract_feat(fp) for fp in train_files[lab]]

    # Đánh giá HMM
    hmm_res = evaluate_system(models, test_files, method="hmm")
    dtw_res = evaluate_system(dtw_templates, test_files, method="dtw")

    print(f"\n  ┌{'─'*24}┬{'─'*12}┬{'─'*14}┐")
    print(f"  │ {'Chỉ số':<22} │ {'HMM':>10} │ {'DTW':>12} │")
    print(f"  ├{'─'*24}┼{'─'*12}┼{'─'*14}┤")
    print(f"  │ {'Accuracy':<22} │ {hmm_res[2]*100:>9.1f}% │ {dtw_res[2]*100:>11.1f}% │")
    print(f"  │ {'Test time/trial (ms)':<22} │ {hmm_res[3]:>10.1f} │ {dtw_res[3]:>12.1f} │")
    print(f"  │ {'Train files/class':<22} │ {'6':>10} │ {'6 (template)':>12} │")
    print(f"  │ {'Score type':<22} │ {'log P(X|λ)':>10} │ {'DTW dist':>12} │")
    print(f"  └{'─'*24}┴{'─'*12}┴{'─'*14}┘")
    print(f"\n  Nhận xét:")
    print(f"    - HMM: mô hình xác suất → score là log P(X|λ) (cao hơn = tốt hơn)")
    print(f"    - DTW: khoảng cách → score là cost (thấp hơn = tốt hơn)")
    print(f"    - HMM cần nhiều dữ liệu hơn để tham số ổn định")
    print(f"    - DTW hoạt động tốt với ít mẫu (3 template/lớp)")

    plot_comparison(hmm_res, dtw_res)
    return {"hmm_res": hmm_res, "dtw_res": dtw_res}


# ═══════════════════════════════════════════════════════════
# F. LEXICON MINI
# ═══════════════════════════════════════════════════════════

LEXICON_CONTENT = """# Từ điển phát âm tiếng Việt (simplified phoneme notation)
# Format: WORD  PHONE_1  PHONE_2  ...  PHONE_N
# Phone symbols dùng ASCII-friendly notation
#
# Phụ âm đầu: b, d, g, h, k, kh, l, m, n, ng, s, t, tr, x, z
# Nguyên âm  : a, aa, e, ee, i, o, oo, u, uu
# Phụ âm cuối: k_f, m_f, n_f, ng_f, p_f, t_f
# (hậu tố _f = final position)

mo      m       oo
dong    d       oo      ng_f
len     l       ee      n_f
xuong   x       uu      ng_f
dung    z       uu      ng_f
"""

PHONEME_DESCRIPTIONS = {
    "m"   : "Nasal bilabial /m/",
    "d"   : "Voiced alveolar stop /d/ (~ /z/ in Southern)",
    "l"   : "Lateral /l/",
    "x"   : "Voiceless fricative /s/ (xu)",
    "z"   : "Voiced fricative /z/ (dừng)",
    "oo"  : "Mid-back rounded vowel /ɔ/",
    "ee"  : "Mid-front unrounded vowel /e/",
    "uu"  : "High back unrounded vowel /ɨ/",
    "ng_f": "Velar nasal final /ŋ/",
    "n_f" : "Alveolar nasal final /n/",
}

def parse_lexicon(content: str) -> dict:
    """
    Parser từ điển phát âm.
    Format: WORD PHONE_1 PHONE_2 ... → dict {word: [phone_list]}
    """
    lex = {}
    for line in content.strip().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) >= 2:
            lex[parts[0]] = parts[1:]
    return lex

def run_part_f() -> dict:
    print("\n"+"═"*60)
    print("  PHẦN F – LEXICON MINI (TỪ ĐIỂN PHÁT ÂM)")
    print("═"*60)

    # Lưu lexicon.txt
    lex_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lexicon.txt")
    with open(lex_path,"w",encoding="utf-8") as f:
        f.write(LEXICON_CONTENT)
    print(f"  Đã lưu: {lex_path}")

    # Parse
    lex = parse_lexicon(LEXICON_CONTENT)
    print(f"\n  Từ điển ({len(lex)} từ):")
    print(f"  {'Word':<10} {'Phoneme sequence':<25} {'IPA approx'}")
    print("  "+"─"*55)
    ipa_map = {"mo":"/mɔː/","dong":"/doŋ/","len":"/leːn/",
               "xuong":"/suŋ/","dung":"/zuŋ/"}
    for word, phones in lex.items():
        print(f"  {word:<10} {' – '.join(phones):<25} {ipa_map.get(word,'?')}")

    print("\n  Giải thích kiến trúc:")
    print("    Whole-word HMM: không cần lexicon (1 HMM = 1 từ)")
    print("    Subword/Phone HMM: cần lexicon (word → phone sequence)")
    print("    Word HMM = nối phone HMM theo lexicon:")
    for word, phones in lex.items():
        chain = " → ".join(f"λ_{p}" for p in phones)
        print(f"      λ_{word} = {chain}")

    # Vẽ lexicon
    fig, ax = plt.subplots(figsize=(12,5))
    ax.axis("off")
    rows = [[w, " – ".join(phones), ipa_map.get(w,"?"),
             " + ".join(f"HMM({p})" for p in phones)]
            for w,phones in lex.items()]
    tbl = ax.table(
        cellText  = rows,
        colLabels = ["Từ", "Phoneme sequence", "IPA", "Phone HMM chain"],
        cellLoc   = "center", loc="center", bbox=[0,0,1,1]
    )
    tbl.auto_set_font_size(False); tbl.set_fontsize(10)
    for (r,c),cell in tbl.get_celld().items():
        if r==0:
            cell.set_facecolor("#2980b9"); cell.set_text_props(color="white",fontweight="bold")
        elif r%2==0: cell.set_facecolor("#f0f0f0")
        cell.set_edgecolor("gray")
    ax.set_title("Phần F – Lexicon: Word → Phone HMM chain\n"
                 "Whole-word HMM không cần lexicon; Phone HMM cần lexicon để concatenate",
                 fontsize=11, fontweight="bold", pad=20)
    savefig(fig, "F_lexicon.png")

    return {"lexicon": lex}


# ═══════════════════════════════════════════════════════════
# G. BIGRAM + MAP RE-RANKING
# ═══════════════════════════════════════════════════════════

CORPUS_SENTENCES = [
    ["<s>","mo","dong","<e>"],   ["<s>","len","xuong","<e>"],
    ["<s>","dung","mo","<e>"],   ["<s>","dong","len","<e>"],
    ["<s>","xuong","dung","<e>"],["<s>","mo","len","<e>"],
    ["<s>","dong","xuong","<e>"],["<s>","len","mo","<e>"],
    ["<s>","dung","dong","<e>"], ["<s>","xuong","mo","<e>"],
    ["<s>","mo","mo","<e>"],     ["<s>","dong","dung","<e>"],
    ["<s>","len","dung","<e>"],  ["<s>","xuong","len","<e>"],
    ["<s>","dung","xuong","<e>"],
]

def build_ngrams(corpus):
    c1, c2 = {}, {}
    for sent in corpus:
        for i in range(len(sent)-1):
            c1[sent[i]] = c1.get(sent[i],0)+1
            pair = (sent[i],sent[i+1])
            c2[pair] = c2.get(pair,0)+1
    return c1, c2

def bigram_prob(prev, cur, c1, c2, V, k=0.5):
    """P(cur|prev) với add-k smoothing."""
    return (c2.get((prev,cur),0)+k) / (c1.get(prev,0)+k*V)

def map_rerank(feat, prev_word, models, c1, c2, lm_weight=0.3):
    """
    MAP re-ranking:
      Score(w) = log P_A(X|λ_w)/T + λ_LM·log P_L(w|prev)

    λ_LM: cân bằng acoustic và language model score.
    """
    T = len(feat); V = len(LABELS)
    scores = {}
    for lab in LABELS:
        a_score = models[lab].score(feat) / T   # normalized acoustic
        lm      = math.log(bigram_prob(prev_word, lab, c1, c2, V)+1e-10)
        scores[lab] = a_score + lm_weight*lm
    pred = max(scores, key=scores.get)
    return pred, scores

def run_part_g(models) -> dict:
    print("\n"+"═"*60)
    print("  PHẦN G – BIGRAM LM + MAP RE-RANKING")
    print("═"*60)

    c1, c2 = build_ngrams(CORPUS_SENTENCES)
    V = len(LABELS)

    print(f"  Corpus: {len(CORPUS_SENTENCES)} câu lệnh")
    print(f"  Bigram counts (add-0.5 smoothing):")
    print(f"  {'prev/cur':<10}", end="")
    for lab in LABELS: print(f"  {LABEL_VI[lab]:>8}", end="")
    print()
    print("  "+"─"*55)
    for prev in LABELS:
        print(f"  {LABEL_VI[prev]:<10}", end="")
        for cur in LABELS:
            p = bigram_prob(prev, cur, c1, c2, V)
            print(f"  {p:>8.3f}", end="")
        print()

    # Thử nghiệm re-ranking với λLM = 0, 0.1, 0.3, 1.0
    print(f"\n  Thí nghiệm re-ranking (λLM = 0 / 0.1 / 0.3 / 1.0):")
    print(f"  Giả sử utterance test là 'xuong', prev_word='mo'")

    train_files, test_files = get_files()
    feat_test = extract_feat(test_files["xuong"][0])
    prev_w = "mo"

    results_lm = {}
    print(f"\n  {'λ_LM':<8} {'Pred':<8} {'Score top-3'}")
    print("  "+"─"*55)
    for lm_w in [0.0, 0.1, 0.3, 1.0]:
        pred, sc = map_rerank(feat_test, prev_w, models, c1, c2, lm_weight=lm_w)
        top3 = list(sorted(sc.items(), key=lambda x:-x[1]))[:3]
        top3_str = "  ".join(f"{LABEL_VI[l]}={v:.3f}" for l,v in top3)
        print(f"  {lm_w:<8.1f} {LABEL_VI[pred]:<8} {top3_str}")
        results_lm[lm_w] = (pred, sc)

    # Vẽ heatmap bigram
    P_mat = np.array([[bigram_prob(prev,cur,c1,c2,V)
                        for cur in LABELS] for prev in LABELS])
    fig, ax = plt.subplots(figsize=(8,6))
    im = ax.imshow(P_mat, cmap="Blues", vmin=0)
    plt.colorbar(im, ax=ax, label="P(w_cur | w_prev)")
    for i in range(len(LABELS)):
        for j in range(len(LABELS)):
            ax.text(j,i,f"{P_mat[i,j]:.2f}",ha="center",va="center",fontsize=10)
    vil=[LABEL_VI[l] for l in LABELS]
    ax.set_xticks(range(len(LABELS))); ax.set_xticklabels(vil,rotation=20,fontsize=9)
    ax.set_yticks(range(len(LABELS))); ax.set_yticklabels(vil,fontsize=9)
    ax.set_xlabel("Từ hiện tại (cur)"); ax.set_ylabel("Từ trước (prev)")
    ax.set_title("Phần G – Bigram P(w_cur|w_prev) với Add-0.5 smoothing\n"
                 "Score(W) = log P_A(X|W)/T + λ_LM·log P_L(W)",
                 fontsize=10, fontweight="bold")
    plt.tight_layout()
    savefig(fig, "G_bigram_heatmap.png")

    print("\n  Nhận xét:")
    print("    - λ_LM=0: thuần acoustic, không dùng LM")
    print("    - λ_LM=0.1–0.3: LM hỗ trợ khi acoustic scores gần nhau")
    print("    - λ_LM=1.0: LM quá mạnh → có thể lấn át acoustic evidence")
    print("    - Full decoder cần tìm đồng thời word boundaries; re-ranking")
    print("      này chỉ minh họa nguyên lý MAP của công thức Bayes")

    return {"c1": c1, "c2": c2}


# ═══════════════════════════════════════════════════════════
# H. ĐÁNH GIÁ ĐẦY ĐỦ + THÍ NGHIỆM E1–E6
# ═══════════════════════════════════════════════════════════

def full_evaluation(models, test_files, tag="") -> tuple:
    """Đánh giá accuracy + confusion matrix đầy đủ."""
    y_true, y_pred = [], []
    score_correct, score_wrong = [], []

    for lab in LABELS:
        for fp in test_files[lab]:
            feat = extract_feat(fp)
            pred, scores = recognize(feat, models)
            y_true.append(lab); y_pred.append(pred)
            # Top-1 score (normalized)
            best_score = scores[list(scores.keys())[0]] / len(feat)
            if pred == lab:
                score_correct.append(best_score)
            else:
                score_wrong.append(best_score)

    acc = accuracy_score(y_true, y_pred)
    return y_true, y_pred, acc, score_correct, score_wrong


def plot_confusion(y_true, y_pred, title, fname):
    cm  = confusion_matrix(y_true, y_pred, labels=LABELS)
    acc = accuracy_score(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8,7))
    im = ax.imshow(cm, cmap="Blues", vmin=0)
    plt.colorbar(im, ax=ax, fraction=0.046)
    for i in range(len(LABELS)):
        for j in range(len(LABELS)):
            clr = "white" if cm[i,j]>cm.max()*.6 else "black"
            ax.text(j,i,str(cm[i,j]),ha="center",va="center",
                    fontsize=13,fontweight="bold",color=clr)
    vil=[LABEL_VI[l] for l in LABELS]
    ax.set_xticks(range(len(LABELS))); ax.set_xticklabels(vil,fontsize=10)
    ax.set_yticks(range(len(LABELS))); ax.set_yticklabels(vil,fontsize=10)
    ax.set_xlabel("Dự đoán"); ax.set_ylabel("Nhãn thật")
    ax.set_title(f"{title}\nAccuracy = {acc*100:.1f}%",
                 fontsize=11, fontweight="bold")
    plt.tight_layout(); savefig(fig, fname)
    return acc


def experiment_e1_n_states(train_files, test_files) -> None:
    """E1: n_states ∈ {3,5,7}."""
    from lab3_hmm import train_models
    print("\n  ── E1: n_states = {3, 5, 7} ───────────────────────")
    results = {}
    for ns in [3, 5, 7]:
        models_ns = train_models(train_files, n_states=ns, n_iter=80)
        yt,yp,acc,_,_ = full_evaluation(models_ns, test_files)
        results[ns] = acc
        cvg = sum(1 for m in models_ns.values() if m.converged_)
        print(f"    n_states={ns}: Acc={acc*100:.1f}%  ({cvg}/{len(LABELS)} hội tụ)")

    fig, ax = plt.subplots(figsize=(7,4))
    ax.bar([str(k) for k in results.keys()], [v*100 for v in results.values()],
           color=["steelblue","darkorange","seagreen"], edgecolor="black")
    for k,v in results.items():
        ax.text(str(k), v*100+0.5, f"{v*100:.1f}%", ha="center", fontsize=10)
    style_ax(ax, xlabel="n_states", ylabel="Accuracy (%)",
             title="E1 – Accuracy theo n_states\n(Ít state: underfit | Nhiều state: có thể không hội tụ)")
    ax.set_ylim([0,115])
    plt.tight_layout(); savefig(fig, "H_E1_n_states.png")


def experiment_e2_feat(train_files, test_files) -> None:
    """E2: MFCC 13 vs MFCC+Δ+Δ² 39."""
    from lab3_hmm import GaussianHMM_LR
    print("\n  ── E2: MFCC 13D vs MFCC+Δ+Δ² 39D ─────────────────")

    from lab3_features import (compute_mfcc_static, compute_delta,
                                 cmvn, endpoint_trim, read_wav)
    from lab3_utils import read_wav as rw

    def feat_13(fp):
        x = read_wav(fp); x = endpoint_trim(x)
        c = compute_mfcc_static(x)
        return cmvn(c)

    def feat_39(fp):
        return extract_feat(fp)

    accs = {}
    for dim_tag, feat_fn, n_dim in [("MFCC 13D",feat_13,13),
                                     ("MFCC+Δ+Δ² 39D",feat_39,39)]:
        # Build templates DTW style (dùng DTW cho nhanh)
        templates = {lab: [feat_fn(fp) for fp in train_files[lab]]
                     for lab in LABELS}
        yt,yp=[],[]
        for lab in LABELS:
            for fp in test_files[lab]:
                f = feat_fn(fp)
                sc={l:min(dtw_distance(f,R) for R in templates[l]) for l in LABELS}
                yt.append(lab); yp.append(min(sc,key=sc.get))
        acc=accuracy_score(yt,yp); accs[dim_tag]=acc
        print(f"    {dim_tag:<20}: Acc={acc*100:.1f}%")

    print(f"    Nhận xét: Δ,Δ² đưa thông tin temporal vào frame → "
          f"giúp phân biệt phụ âm đầu/cuối tốt hơn")


def experiment_e5_noise(models, test_files) -> None:
    """E5: Mismatch – thêm noise vào test utterance."""
    print("\n  ── E5: Speaker/Environment Mismatch (Noise) ───────")
    from lab3_features import endpoint_trim, compute_mfcc_static, compute_delta, cmvn
    rng = np.random.default_rng(99)

    for noise_snr in [40, 20, 10]:
        yt,yp=[],[]
        for lab in LABELS:
            for fp in test_files[lab]:
                x = read_wav_local(fp)
                # Thêm Gaussian noise
                sp = np.mean(x**2)+1e-10
                x_noisy = x + rng.standard_normal(len(x))*np.sqrt(sp/10**(noise_snr/10))
                x_noisy = np.clip(x_noisy,-1,1)
                x_noisy = endpoint_trim(x_noisy)
                c=compute_mfcc_static(x_noisy)
                d1=compute_delta(c); d2=compute_delta(d1)
                feat=cmvn(np.concatenate([c,d1,d2],axis=1))
                pred,_=recognize(feat,models)
                yt.append(lab); yp.append(pred)
        acc=accuracy_score(yt,yp)
        print(f"    SNR={noise_snr:>3}dB: Acc={acc*100:.1f}%")

    print("    Nhận xét: Noise làm giảm accuracy → cần feature normalization / noise robust training")

def read_wav_local(path):
    """Đọc WAV về float64 (local helper)."""
    from lab3_utils import read_wav; return read_wav(path)


def run_part_h(models, train_files, test_files) -> None:
    print("\n"+"═"*60)
    print("  PHẦN H – ĐÁNH GIÁ ĐẦY ĐỦ + THÍ NGHIỆM E1–E5")
    print("═"*60)

    # Đánh giá chính
    yt,yp,acc,sc_c,sc_w = full_evaluation(models, test_files)
    print(f"\n  [Kết quả baseline] Accuracy = {acc*100:.1f}%")

    plot_confusion(yt, yp,
                   "Phần H – Confusion Matrix HMM (n_states=5, 39D)",
                   "H_confusion_matrix.png")

    # Score distribution
    fig, ax = plt.subplots(figsize=(9,5))
    if sc_c: ax.hist(sc_c, bins=10, alpha=0.7, color="steelblue",
                     label=f"Đúng lớp (n={len(sc_c)})")
    if sc_w: ax.hist(sc_w, bins=10, alpha=0.7, color="tomato",
                     label=f"Sai lớp (n={len(sc_w)})")
    style_ax(ax, xlabel="Normalized log-likelihood (score/T)",
             ylabel="Count",
             title="Phần H – Phân bố Score: đúng lớp vs sai lớp\n"
                   "Đề xuất reject threshold: dưới mức overlap")
    ax.legend(fontsize=9)
    plt.tight_layout(); savefig(fig, "H_score_distribution.png")

    # Thí nghiệm
    experiment_e1_n_states(train_files, test_files)
    experiment_e2_feat(train_files, test_files)
    experiment_e5_noise(models, test_files)

    print("\n  ✓ Phần H hoàn thành.")


# ═══════════════════════════════════════════════════════════
# CHẠY TOÀN BỘ PHẦN E → H
# ═══════════════════════════════════════════════════════════

def run_all_efgh(data: dict = None) -> None:
    if data is None:
        from lab3_features import run_part_a
        from lab3_hmm import run_part_bcd
        data = run_part_bcd(run_part_a())

    models      = data["models"]
    train_files = data["train_files"]
    test_files  = data["test_files"]

    run_part_e(models, train_files, test_files)
    run_part_f()
    run_part_g(models)
    run_part_h(models, train_files, test_files)
    print("\n  ✓ Phần E–H hoàn thành.")


if __name__ == "__main__":
    run_all_efgh()
