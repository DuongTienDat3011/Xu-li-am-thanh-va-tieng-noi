"""
lab2_part_efg.py
================
CSE457 Lab 2 – Phần E, F, G
Sinh viên: Dương Tiến Đạt – MSSV: 2251172268

E. DTW tự cài đặt (Euclidean distance + dynamic programming + backtrack)
F. Bộ nhận dạng nearest-template
G. Đánh giá: accuracy, confusion matrix, thí nghiệm E1 + E2
"""

import os, csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.metrics import confusion_matrix, accuracy_score

import lab2_utils as U
from lab2_utils import (
    FS, WIN_LEN, HOP_LEN, N_MFCC, LABELS, LABEL_VI,
    DATASET_DIR, FIGURES_DIR,
    read_wav, savefig, style_ax
)
from lab2_part_abcd import (
    compute_mfcc, endpoint_detection,
    run_part_a, run_part_c, run_part_d
)


# ════════════════════════════════════════════════════════════
# PHẦN E – DTW TỰ CÀI ĐẶT
# ════════════════════════════════════════════════════════════

def euclidean_distance(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Khoảng cách Euclidean giữa hai vector MFCC:
      d(x_i, y_j) = ||x_i - y_j||_2 = sqrt(Σ_q (x_i[q] - y_j[q])²)
    """
    return float(np.sqrt(np.sum((v1 - v2) ** 2)))


def dtw_distance(X: np.ndarray, Y: np.ndarray) -> tuple:
    """
    Dynamic Time Warping – tự cài đặt theo đề bài.

    X: (N, D) – chuỗi test (N frames, D=13 MFCC)
    Y: (M, D) – chuỗi template (M frames)

    Công thức DP (3 bước cục bộ):
      D[i,j] = C[i,j] + min{ D[i-1,j], D[i,j-1], D[i-1,j-1] }

    Điều kiện biên: D[0,0] = 0, bắt đầu tại (1,1), kết thúc tại (N,M).

    Returns
    -------
    dtw_norm : float – DTW cost chuẩn hóa theo path length
    path     : list of (i,j) – optimal warping path
    D_cost   : ndarray (N,M) – cumulative cost matrix (không kể padding)
    C_local  : ndarray (N,M) – local distance matrix
    """
    N, D = X.shape
    M    = Y.shape[0]

    # ── Ma trận local distance (N × M) ──
    C = np.zeros((N, M))
    for i in range(N):
        for j in range(M):
            C[i, j] = euclidean_distance(X[i], Y[j])

    # ── Ma trận cumulative cost (N+1) × (M+1), biên = inf ──
    D = np.full((N + 1, M + 1), np.inf)
    D[0, 0] = 0.0

    # ── Lưu con trỏ backtrack ──
    # back[i,j] = (pi, pj) là ô trước trong đường tối ưu
    back = np.zeros((N + 1, M + 1, 2), dtype=np.int32)

    # ── Lấp ma trận DP ──
    for i in range(1, N + 1):
        for j in range(1, M + 1):
            local = C[i - 1, j - 1]   # local distance của (i,j)

            # 3 bước cục bộ được phép:
            #   (i-1, j)   → bước dọc   (X tiến, Y đứng yên)
            #   (i, j-1)   → bước ngang (Y tiến, X đứng yên)
            #   (i-1, j-1) → bước chéo  (cả hai tiến)
            candidates = [
                (D[i - 1, j    ], i - 1, j    ),   # từ trên
                (D[i,     j - 1], i,     j - 1),   # từ trái
                (D[i - 1, j - 1], i - 1, j - 1),   # từ chéo
            ]
            best_cost, pi, pj = min(candidates, key=lambda z: z[0])
            D[i, j]   = local + best_cost
            back[i, j] = (pi, pj)

    # ── Backtrack từ (N, M) về (0, 0) ──
    path = []
    ci, cj = N, M
    while ci > 0 or cj > 0:
        path.append((ci - 1, cj - 1))   # chuyển về index 0-based
        pi, pj = back[ci, cj]
        ci, cj = pi, pj
    path.reverse()

    # ── Chuẩn hóa theo độ dài đường đi ──
    path_len = max(len(path), 1)
    dtw_norm = D[N, M] / path_len

    return dtw_norm, path, D[1:, 1:], C


def run_part_e(all_mfcc: dict) -> None:
    """
    Phần E: Vẽ local-distance matrix và optimal path.
      (i)  Hai utterance cùng từ  → DTW_norm nhỏ, path gần đường chéo
      (ii) Hai từ khác nhau       → DTW_norm lớn, path xa đường chéo
    """
    print("\n" + "═" * 60)
    print("  PHẦN E – DTW TỰ CÀI ĐẶT")
    print("═" * 60)

    # ── Kiểm tra bắt buộc: DTW(X, X) ≈ 0 ──
    X_test = all_mfcc["khong"][0]
    dtw_self, _, _, _ = dtw_distance(X_test, X_test)
    print(f"  Kiểm tra DTW(X,X): {dtw_self:.6f}  (phải ≈ 0)")
    assert dtw_self < 1e-6, f"DTW(X,X) = {dtw_self:.6f} ≠ 0 – có lỗi!"

    # ── Case 1: cùng từ "khong" (utterance 1 vs 2) ──
    X1 = all_mfcc["khong"][0]
    X2 = all_mfcc["khong"][1]
    dtw_same, path_same, D_same, C_same = dtw_distance(X1, X2)
    print(f"\n  [Cùng từ 'Không'] DTW_norm = {dtw_same:.4f}  "
          f"| Path length = {len(path_same)}")

    # ── Case 2: hai từ khác nhau ("khong" vs "mot") ──
    Y1 = all_mfcc["mot"][0]
    dtw_diff, path_diff, D_diff, C_diff = dtw_distance(X1, Y1)
    print(f"  [Khác từ 'Không'/'Một'] DTW_norm = {dtw_diff:.4f}  "
          f"| Path length = {len(path_diff)}")
    print(f"  → DTW_norm cùng từ / khác từ = {dtw_same/dtw_diff:.3f} "
          f"({'nhỏ hơn' if dtw_same < dtw_diff else 'lớn hơn'} → "
          f"{'đúng' if dtw_same < dtw_diff else 'KO đúng'})")

    # ── Vẽ 4 subplot: 2 trường hợp × (local dist + path) ──
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    for row, (label, X_p, Y_p, dtw_n, path_p, D_p, C_p, ttl_l, ttl_r) in enumerate([
        ("cung_tu",  X1, X2, dtw_same, path_same, D_same, C_same,
         "Local Distance – Cùng từ [Không vs Không]",
         "Cumulative Cost + Optimal Path – Cùng từ"),
        ("khac_tu",  X1, Y1, dtw_diff, path_diff, D_diff, C_diff,
         "Local Distance – Khác từ [Không vs Một]",
         "Cumulative Cost + Optimal Path – Khác từ"),
    ]):
        N_, M_ = X_p.shape[0], Y_p.shape[0]

        # Local distance
        im0 = axes[row, 0].imshow(C_p, aspect="auto", origin="lower",
                                    cmap="Blues",
                                    extent=[0, M_, 0, N_])
        plt.colorbar(im0, ax=axes[row, 0], label="Euclid distance")
        if path_p:
            px = [p[1] + 0.5 for p in path_p]
            py = [p[0] + 0.5 for p in path_p]
            axes[row, 0].plot(px, py, "r-", lw=1.5, label="Optimal path")
            axes[row, 0].legend(fontsize=8)
        style_ax(axes[row, 0],
                 xlabel="Frame j (template)", ylabel="Frame i (test)",
                 title=f"{ttl_l}")

        # Cumulative cost
        im1 = axes[row, 1].imshow(D_p, aspect="auto", origin="lower",
                                    cmap="YlOrRd",
                                    extent=[0, M_, 0, N_])
        plt.colorbar(im1, ax=axes[row, 1], label="Cumulative cost")
        if path_p:
            axes[row, 1].plot(px, py, "b-", lw=1.5, label="Optimal path")
            axes[row, 1].legend(fontsize=8)
        # Đường chéo tham chiếu
        diag_len = min(N_, M_)
        axes[row, 1].plot([0, diag_len], [0, diag_len],
                           "g--", lw=0.8, alpha=0.6, label="Đường chéo")
        style_ax(axes[row, 1],
                 xlabel="Frame j (template)", ylabel="Frame i (test)",
                 title=f"{ttl_r}\nDTW_norm = {dtw_n:.4f}")
        axes[row, 1].legend(fontsize=8)

    fig.suptitle("Phần E – DTW: Local Distance & Optimal Path\n"
                 "Cùng từ: path gần đường chéo, cost thấp | "
                 "Khác từ: path xa, cost cao",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    savefig(fig, "E_dtw_path.png")

    print("\n  Nhận xét:")
    print("    - Cùng từ: path bám tương đối gần đường chéo (tốc độ nói gần nhau)")
    print("    - Khác từ: path phân kỳ, vùng đỏ tập trung ở corner → cost cao")
    print("    - Phải chuẩn hóa DTW / path_length để so sánh công bằng")
    print("\n  ✓ Phần E hoàn thành.")


# ════════════════════════════════════════════════════════════
# PHẦN F – BỘ NHẬN DẠNG NEAREST-TEMPLATE
# ════════════════════════════════════════════════════════════

def build_templates(train_files: dict,
                     use_trim: bool = True) -> dict:
    """
    Xây dựng tập template từ training files.
    Mỗi từ có 3 template (3 file đầu).

    use_trim : Nếu True → áp endpoint detection trước khi trích MFCC
    """
    templates = {lab: [] for lab in LABELS}
    for lab in LABELS:
        for fpath in train_files[lab]:
            x = read_wav(fpath)
            if use_trim:
                x, _, _ = endpoint_detection(x)
            mfcc = compute_mfcc(x)
            templates[lab].append(mfcc)
    return templates


def recognize_one(x_test: np.ndarray, templates: dict,
                   use_trim: bool = True) -> tuple:
    """
    Nhận dạng 1 utterance dùng nearest-template DTW.

    Returns
    -------
    pred    : nhãn dự đoán
    scores  : dict {label: DTW_norm} sắp xếp tăng dần
    """
    if use_trim:
        x_test, _, _ = endpoint_detection(x_test)
    X = compute_mfcc(x_test)

    scores = {}
    for lab, refs in templates.items():
        # Lấy khoảng cách nhỏ nhất trong số 3 templates
        dists = [dtw_distance(X, R)[0] for R in refs]
        scores[lab] = float(min(dists))

    # Sắp xếp theo khoảng cách tăng dần
    scores_sorted = dict(sorted(scores.items(), key=lambda kv: kv[1]))
    pred = next(iter(scores_sorted))   # Nhãn có DTW_norm nhỏ nhất
    return pred, scores_sorted


def run_part_f(data: dict) -> dict:
    """
    Phần F: Nhận dạng tất cả test file, in top-3, tạo results.csv.
    """
    print("\n" + "═" * 60)
    print("  PHẦN F – BỘ NHẬN DẠNG NEAREST-TEMPLATE")
    print("═" * 60)

    train_files = data["train_files"]
    test_files  = data["test_files"]

    templates = build_templates(train_files, use_trim=True)
    print(f"  Templates: {sum(len(v) for v in templates.values())} file "
          f"({len(LABELS)} từ × {len(templates[LABELS[0]])} template/từ)")

    results = []
    y_true, y_pred = [], []

    print(f"\n  {'File':<22} {'True':<8} {'Pred':<8} {'Top-3 scores'}")
    print("  " + "─" * 72)

    for lab in LABELS:
        for fpath in test_files[lab]:
            x    = read_wav(fpath)
            pred, scores = recognize_one(x, templates, use_trim=True)
            y_true.append(lab); y_pred.append(pred)
            fname = os.path.basename(fpath)
            top3  = list(scores.items())[:3]
            top3_str = "  ".join(f"{k}={v:.4f}" for k, v in top3)
            ok_str = "✓" if pred == lab else "✗"
            print(f"  {fname:<22} {lab:<8} {pred:<8} {ok_str}  {top3_str}")
            results.append({
                "file": fname, "true": lab, "pred": pred,
                "top1_label": top3[0][0], "top1_score": round(top3[0][1], 5),
                "top2_label": top3[1][0] if len(top3) > 1 else "",
                "top2_score": round(top3[1][1], 5) if len(top3) > 1 else "",
            })

    # Lưu results.csv
    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "results.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    print(f"\n  → Kết quả lưu: {csv_path}")

    acc = accuracy_score(y_true, y_pred)
    print(f"\n  Accuracy = {len([1 for t,p in zip(y_true,y_pred) if t==p])}"
          f"/{len(y_true)} = {acc*100:.1f}%")

    print("\n  ✓ Phần F hoàn thành.")
    return {"templates": templates, "y_true": y_true, "y_pred": y_pred,
            "results": results}


# ════════════════════════════════════════════════════════════
# PHẦN G – ĐÁNH GIÁ & THÍ NGHIỆM
# ════════════════════════════════════════════════════════════

def plot_confusion_matrix(y_true: list, y_pred: list,
                           labels: list, title: str,
                           save_name: str) -> float:
    """Vẽ confusion matrix và trả về accuracy."""
    cm  = confusion_matrix(y_true, y_pred, labels=labels)
    acc = accuracy_score(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues",
                   vmin=0, vmax=cm.max() + 1)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Hiển thị số liệu trong từng ô
    for i in range(len(labels)):
        for j in range(len(labels)):
            val  = cm[i, j]
            clr  = "white" if val > cm.max() * 0.6 else "black"
            ax.text(j, i, str(val), ha="center", va="center",
                    fontsize=12, fontweight="bold", color=clr)

    vi_labels = [LABEL_VI.get(l, l) for l in labels]
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(vi_labels, fontsize=10)
    ax.set_yticklabels(vi_labels, fontsize=10)
    ax.set_xlabel("Dự đoán", fontsize=11)
    ax.set_ylabel("Nhãn thật", fontsize=11)
    ax.set_title(f"{title}\nAccuracy = {acc*100:.1f}%",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    savefig(fig, save_name)
    return acc


def experiment_e1(train_files: dict, test_files: dict) -> dict:
    """
    Thí nghiệm E1: Có endpoint detection vs. không endpoint detection.
    Giữ cố định: MFCC 13 hệ số, DTW, templates.
    """
    print("\n  ── Thí nghiệm E1: có trim vs. không trim ──────────────")
    results_e1 = {}

    for use_trim, tag in [(True, "Có trim"), (False, "Không trim")]:
        templates = build_templates(train_files, use_trim=use_trim)
        y_true, y_pred = [], []
        for lab in LABELS:
            for fpath in test_files[lab]:
                x    = read_wav(fpath)
                pred, _ = recognize_one(x, templates, use_trim=use_trim)
                y_true.append(lab); y_pred.append(pred)

        acc = accuracy_score(y_true, y_pred)
        results_e1[tag] = {"y_true": y_true, "y_pred": y_pred, "acc": acc}
        print(f"    {tag:<12}: Accuracy = {acc*100:.1f}%")

    # So sánh
    acc_trim   = results_e1["Có trim"]["acc"]
    acc_notrim = results_e1["Không trim"]["acc"]
    delta = (acc_trim - acc_notrim) * 100
    print(f"    → Trim cải thiện accuracy: {delta:+.1f}%")
    print(f"    Nhận xét: Endpoint detection {'giúp cải thiện' if delta > 0 else 'không giúp cải thiện'} "
          f"vì {'loại bỏ silence thừa giúp DTW căn chỉnh đúng phần speech' if delta > 0 else 'tín hiệu tổng hợp đã có silence ngắn'}")
    return results_e1


def experiment_e2(train_files: dict, test_files: dict) -> dict:
    """
    Thí nghiệm E2: MFCC 13 hệ số vs. MFCC + Delta (26 hệ số).
    Delta = đạo hàm bậc 1 của MFCC theo thời gian (dynamic feature).
    Giữ cố định: train/test split, endpoint detection.
    """
    print("\n  ── Thí nghiệm E2: MFCC vs. MFCC+Δ ─────────────────────")

    def compute_delta(mfcc: np.ndarray, win: int = 2) -> np.ndarray:
        """
        Tính delta (đạo hàm bậc 1 xấp xỉ) của MFCC:
          Δ[t] = Σ_{k=1}^{win} k·(c[t+k] - c[t-k]) / (2·Σ k²)
        """
        T, D     = mfcc.shape
        delta    = np.zeros_like(mfcc)
        denom    = 2.0 * sum(k ** 2 for k in range(1, win + 1))
        for t in range(T):
            for k in range(1, win + 1):
                t_fwd = min(t + k, T - 1)
                t_bwd = max(t - k, 0)
                delta[t] += k * (mfcc[t_fwd] - mfcc[t_bwd])
        delta /= (denom + 1e-10)
        return delta

    def mfcc_with_delta(x: np.ndarray) -> np.ndarray:
        """Trích MFCC + Δ, concatenate → 26 chiều."""
        x_t, _, _ = endpoint_detection(x)
        mfcc      = compute_mfcc(x_t)
        delta     = compute_delta(mfcc)
        return np.concatenate([mfcc, delta], axis=1)   # (T, 26)

    results_e2 = {}

    for use_delta, tag in [(False, "MFCC 13"), (True, "MFCC+Δ 26")]:
        feat_fn  = mfcc_with_delta if use_delta else (
            lambda x: compute_mfcc(endpoint_detection(x)[0]))

        # Build templates
        tpls = {lab: [] for lab in LABELS}
        for lab in LABELS:
            for fpath in train_files[lab]:
                x = read_wav(fpath)
                tpls[lab].append(feat_fn(x))

        # Evaluate
        y_true, y_pred = [], []
        for lab in LABELS:
            for fpath in test_files[lab]:
                x    = read_wav(fpath)
                X    = feat_fn(x)
                scores = {}
                for l, refs in tpls.items():
                    scores[l] = min(dtw_distance(X, R)[0] for R in refs)
                pred = min(scores, key=scores.get)
                y_true.append(lab); y_pred.append(pred)

        acc = accuracy_score(y_true, y_pred)
        results_e2[tag] = {"y_true": y_true, "y_pred": y_pred, "acc": acc}
        print(f"    {tag:<12}: Accuracy = {acc*100:.1f}%  (dim={13 if not use_delta else 26})")

    delta_acc = (results_e2["MFCC+Δ 26"]["acc"] -
                 results_e2["MFCC 13"]["acc"]) * 100
    print(f"    → Delta feature thay đổi accuracy: {delta_acc:+.1f}%")
    print(f"    Nhận xét: Delta {'cải thiện' if delta_acc >= 0 else 'giảm'} accuracy vì "
          f"{'nắm bắt được thay đổi động theo thời gian' if delta_acc >= 0 else 'thêm chiều gây nhiễu với dataset nhỏ'}")
    return results_e2


def run_part_g(data: dict) -> None:
    """
    Phần G: Đánh giá tổng hợp và 2 thí nghiệm bắt buộc E1, E2.
    """
    print("\n" + "═" * 60)
    print("  PHẦN G – ĐÁNH GIÁ & THÍ NGHIỆM")
    print("═" * 60)

    y_true      = data["y_true"]
    y_pred      = data["y_pred"]
    train_files = data["train_files"]
    test_files  = data["test_files"]

    # ── Confusion matrix chính ──
    acc = accuracy_score(y_true, y_pred)
    print(f"\n  [Kết quả chính] Accuracy = {acc*100:.1f}%")

    from sklearn.metrics import confusion_matrix as cm_fn
    cm = cm_fn(y_true, y_pred, labels=LABELS)
    print("\n  Confusion matrix:")
    print("  " + " ".join(f"{l:>6}" for l in LABELS))
    for i, lab in enumerate(LABELS):
        print(f"  {lab:<6}" + " ".join(f"{cm[i,j]:>6}" for j in range(len(LABELS))))

    plot_confusion_matrix(y_true, y_pred, LABELS,
                           "Phần G – Confusion Matrix (Baseline)\n"
                           "Train: 3 file/từ | Test: 2 file/từ | MFCC+trim+DTW",
                           "G_confusion_matrix.png")

    # ── Phân tích cặp từ dễ nhầm ──
    for i in range(len(LABELS)):
        for j in range(len(LABELS)):
            if i != j and cm[i, j] > 0:
                print(f"  ⚠ [{LABEL_VI[LABELS[i]]}] bị nhầm thành [{LABEL_VI[LABELS[j]]}]: "
                      f"{cm[i,j]} lần → phân tích MFCC để tìm nguyên nhân")

    # ── Thí nghiệm E1 ──
    res_e1 = experiment_e1(train_files, test_files)

    # Vẽ confusion matrix E1
    fig_e1, axes_e1 = plt.subplots(1, 2, figsize=(15, 7))
    for ax, (tag, res) in zip(axes_e1, res_e1.items()):
        cm_e = cm_fn(res["y_true"], res["y_pred"], labels=LABELS)
        im   = ax.imshow(cm_e, cmap="Blues", vmin=0)
        plt.colorbar(im, ax=ax)
        for ii in range(len(LABELS)):
            for jj in range(len(LABELS)):
                clr = "white" if cm_e[ii,jj] > cm_e.max()*0.6 else "black"
                ax.text(jj, ii, str(cm_e[ii,jj]), ha="center", va="center",
                        fontsize=11, fontweight="bold", color=clr)
        vi_lbl = [LABEL_VI[l] for l in LABELS]
        ax.set_xticks(range(len(LABELS))); ax.set_xticklabels(vi_lbl, fontsize=9)
        ax.set_yticks(range(len(LABELS))); ax.set_yticklabels(vi_lbl, fontsize=9)
        ax.set_xlabel("Dự đoán"); ax.set_ylabel("Nhãn thật")
        ax.set_title(f"{tag}\nAccuracy = {res['acc']*100:.1f}%",
                     fontweight="bold", fontsize=10)
    fig_e1.suptitle("Thí nghiệm E1 – Có trim vs. Không trim",
                    fontsize=12, fontweight="bold")
    plt.tight_layout()
    savefig(fig_e1, "G_experiment_E1.png")

    # ── Thí nghiệm E2 ──
    res_e2 = experiment_e2(train_files, test_files)

    fig_e2, axes_e2 = plt.subplots(1, 2, figsize=(15, 7))
    for ax, (tag, res) in zip(axes_e2, res_e2.items()):
        cm_e = cm_fn(res["y_true"], res["y_pred"], labels=LABELS)
        im   = ax.imshow(cm_e, cmap="Blues", vmin=0)
        plt.colorbar(im, ax=ax)
        for ii in range(len(LABELS)):
            for jj in range(len(LABELS)):
                clr = "white" if cm_e[ii,jj] > cm_e.max()*0.6 else "black"
                ax.text(jj, ii, str(cm_e[ii,jj]), ha="center", va="center",
                        fontsize=11, fontweight="bold", color=clr)
        vi_lbl = [LABEL_VI[l] for l in LABELS]
        ax.set_xticks(range(len(LABELS))); ax.set_xticklabels(vi_lbl, fontsize=9)
        ax.set_yticks(range(len(LABELS))); ax.set_yticklabels(vi_lbl, fontsize=9)
        ax.set_xlabel("Dự đoán"); ax.set_ylabel("Nhãn thật")
        ax.set_title(f"{tag}\nAccuracy = {res['acc']*100:.1f}%",
                     fontweight="bold", fontsize=10)
    fig_e2.suptitle("Thí nghiệm E2 – MFCC 13 vs. MFCC+Δ (26 hệ số)",
                    fontsize=12, fontweight="bold")
    plt.tight_layout()
    savefig(fig_e2, "G_experiment_E2.png")

    # ── Tóm tắt ──
    print("\n  ┌──────────────────────────┬─────────────┐")
    print("  │ Thí nghiệm               │ Accuracy    │")
    print("  ├──────────────────────────┼─────────────┤")
    print(f"  │ Baseline (trim + MFCC13) │ {acc*100:>8.1f}%  │")
    for tag, res in res_e1.items():
        print(f"  │ E1: {tag:<21}│ {res['acc']*100:>8.1f}%  │")
    for tag, res in res_e2.items():
        print(f"  │ E2: {tag:<21}│ {res['acc']*100:>8.1f}%  │")
    print("  └──────────────────────────┴─────────────┘")

    print("\n  ✓ Phần G hoàn thành.")


# ════════════════════════════════════════════════════════════
# CHẠY ĐẦY ĐỦ PHẦN E → G
# ════════════════════════════════════════════════════════════

def run_all_efg(data_abcd: dict = None):
    if data_abcd is None:
        from lab2_part_abcd import run_all_abcd
        data_abcd = run_all_abcd()

    run_part_e(data_abcd["all_mfcc"])
    data_f = run_part_f(data_abcd)
    run_part_g({**data_abcd, **data_f})
    return data_f


if __name__ == "__main__":
    from lab2_part_abcd import run_all_abcd
    data = run_all_abcd()
    run_all_efg(data)
    print("\n  ✓ Phần E–G hoàn thành.\n")
