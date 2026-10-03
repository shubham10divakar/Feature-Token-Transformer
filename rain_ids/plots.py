"""Paper figures. Every figure is saved as 300-dpi PNG and vector PDF."""
from pathlib import Path

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn import metrics as M  # noqa: E402

# fixed categorical order (never cycled); >8 series -> small multiples instead
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
SEQ = "Blues"

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 300, "font.size": 9, "axes.titlesize": 10,
    "axes.labelsize": 9, "legend.fontsize": 8, "lines.linewidth": 2,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "legend.frameon": False,
    "pdf.fonttype": 42,
})


def _save(fig, out: Path, name):
    out.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out / f"{name}.png", bbox_inches="tight")
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def _grid(n, w=2.6, h=2.4):
    cols = min(n, 5)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(w * cols, h * rows), squeeze=False)
    for ax in axes.flat[n:]:
        ax.set_visible(False)
    return fig, axes.flat


# ------------------------------------------------------------ training curves
def training_curves(history, out, best_epoch=None):
    ep = [h["epoch"] for h in history]
    panels = [
        ("Loss", [("train", "train_loss"), ("val", "val_loss")]),
        ("Accuracy", [("train", "train_acc"), ("val", "val_acc")]),
        ("Macro-F1", [("train", "train_macro_f1"), ("val", "val_macro_f1")]),
        ("Binary F1 (val)", [("val", "val_bin_f1")]),
        ("Learning rate", [("lr", "lr")]),
        ("Epoch time (s)", [("time", "epoch_time")]),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(11, 5.6))
    for ax, (title, series) in zip(axes.flat, panels):
        for i, (lab, key) in enumerate(series):
            ys = [h.get(key, np.nan) for h in history]
            ax.plot(ep, ys, color=SERIES[i], label=lab)
        if best_epoch is not None:
            ax.axvline(best_epoch, color=INK2, lw=1, ls="--")
        ax.set_title(title)
        ax.set_xlabel("epoch")
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
        if len(series) > 1:
            ax.legend()
    if best_epoch is not None:
        fig.suptitle(f"dashed line = best epoch ({best_epoch})", fontsize=8, color=INK2)
    _save(fig, out, "training_curves")


# ----------------------------------------------------------- confusion matrix
def confusion_matrices(cm, class_names, out, prefix="multiclass"):
    n = len(class_names)
    size = max(4.5, 0.55 * n + 2)
    for norm in (False, True):
        mat = cm / np.clip(cm.sum(1, keepdims=True), 1, None) if norm else cm
        fig, ax = plt.subplots(figsize=(size, size * 0.85))
        im = ax.imshow(mat, cmap=SEQ, vmin=0, vmax=1 if norm else None)
        ax.grid(False)
        ax.set_xticks(range(n), class_names, rotation=45, ha="right")
        ax.set_yticks(range(n), class_names)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        thr = (mat.max() if mat.size else 1) / 2
        for i in range(n):
            for j in range(n):
                v = mat[i, j]
                txt = f"{v:.2f}" if norm else f"{int(v):,}"
                ax.text(j, i, txt, ha="center", va="center", fontsize=6.5 if n > 8 else 7.5,
                        color="white" if v > thr else INK)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title("Confusion matrix" + (" (row-normalised = recall)" if norm else " (counts)"))
        _save(fig, out, f"{prefix}_confusion_{'normalized' if norm else 'counts'}")


# ------------------------------------------------------- ROC / PR (multi-class)
def multiclass_roc_pr(y, prob, class_names, out):
    C = len(class_names)
    present = [i for i in range(C) if (y == i).any() and (y != i).any()]
    Y = np.eye(C)[y]
    for kind in ("roc", "pr"):
        fig, axes = _grid(len(present))
        for ax, i in zip(axes, present):
            if kind == "roc":
                fpr, tpr, _ = M.roc_curve(Y[:, i], prob[:, i])
                ax.plot(fpr, tpr, color=SERIES[0])
                ax.plot([0, 1], [0, 1], color=GRID, lw=1)
                ax.set_title(f"{class_names[i]}  AUC={M.auc(fpr, tpr):.4f}", fontsize=8.5)
                ax.set_xlabel("FPR")
                ax.set_ylabel("TPR")
            else:
                p, r, _ = M.precision_recall_curve(Y[:, i], prob[:, i])
                ap = M.average_precision_score(Y[:, i], prob[:, i])
                ax.plot(r, p, color=SERIES[0])
                ax.axhline(Y[:, i].mean(), color=GRID, lw=1)
                ax.set_title(f"{class_names[i]}  AP={ap:.4f}", fontsize=8.5)
                ax.set_xlabel("Recall")
                ax.set_ylabel("Precision")
            ax.set_xlim(-0.01, 1.01)
            ax.set_ylim(-0.01, 1.01)
        _save(fig, out, f"multiclass_{kind}_per_class")

    # micro / macro averages
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    Yp, Pp = Y[:, present], prob[:, present]
    fpr_mi, tpr_mi, _ = M.roc_curve(Yp.ravel(), Pp.ravel())
    grid = np.linspace(0, 1, 1001)
    tprs = [np.interp(grid, *M.roc_curve(Yp[:, j], Pp[:, j])[:2]) for j in range(len(present))]
    ax = axes[0]
    ax.plot(fpr_mi, tpr_mi, color=SERIES[0], label=f"micro-avg AUC={M.auc(fpr_mi, tpr_mi):.4f}")
    ax.plot(grid, np.mean(tprs, 0), color=SERIES[1], label=f"macro-avg AUC={M.auc(grid, np.mean(tprs, 0)):.4f}")
    ax.plot([0, 1], [0, 1], color=GRID, lw=1)
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title("ROC (one-vs-rest)")
    ax.legend(loc="lower right")
    ax = axes[1]
    p, r, _ = M.precision_recall_curve(Yp.ravel(), Pp.ravel())
    ax.plot(r, p, color=SERIES[0], label=f"micro-avg AP={M.average_precision_score(Yp, Pp, average='micro'):.4f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall")
    ax.legend(loc="lower left")
    _save(fig, out, "multiclass_roc_pr_average")


def per_class_bars(per_class, out):
    df = per_class[per_class["support"] > 0]
    n = len(df)
    x = np.arange(n)
    w = 0.27
    fig, ax = plt.subplots(figsize=(max(6, 0.75 * n + 2), 3.6))
    for i, m in enumerate(["precision", "recall", "f1"]):
        ax.bar(x + (i - 1) * w, df[m], w * 0.92, color=SERIES[i], label=m.capitalize())
    ax.axhline(0.5, color=INK2, lw=1, ls=":")
    ax.set_xticks(x, df["class"], rotation=30, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_title("Per-class precision / recall / F1 (dotted = 0.5 recall floor)")
    ax.legend(ncol=3, loc="lower left")
    ax.grid(axis="x", visible=False)
    _save(fig, out, "per_class_metrics")


# ------------------------------------------------------------------- binary
def binary_curves(y, score, out, name="binary"):
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.6))
    fpr, tpr, _ = M.roc_curve(y, score)
    ax = axes[0]
    ax.plot(fpr, tpr, color=SERIES[0], label=f"AUC={M.auc(fpr, tpr):.4f}")
    ax.plot([0, 1], [0, 1], color=GRID, lw=1)
    ax.set(xlabel="False alarm rate (FPR)", ylabel="Detection rate (TPR)", title="ROC")
    ax.legend(loc="lower right")
    ax = axes[1]
    ax.plot(fpr, tpr, color=SERIES[0])
    ax.set_xscale("log")
    ax.set_xlim(1e-5, 1)
    ax.set(xlabel="FPR (log)", ylabel="TPR", title="ROC, low-FPR region")
    ax = axes[2]
    p, r, th = M.precision_recall_curve(y, score)
    ax.plot(r, p, color=SERIES[0], label=f"AP={M.average_precision_score(y, score):.4f}")
    ax.axhline(y.mean(), color=GRID, lw=1)
    ax.set(xlabel="Recall", ylabel="Precision", title="Precision-Recall")
    ax.legend(loc="lower left")
    ax = axes[3]
    f1 = 2 * p * r / np.clip(p + r, 1e-12, None)
    ax.plot(th, p[:-1], color=SERIES[0], label="Precision")
    ax.plot(th, r[:-1], color=SERIES[1], label="Recall")
    ax.plot(th, f1[:-1], color=SERIES[2], label="F1")
    ax.set(xlabel="Threshold", title="Metrics vs threshold")
    ax.legend(loc="lower left")
    _save(fig, out, f"{name}_roc_pr_threshold")

    fig, ax = plt.subplots(figsize=(5, 3.4))
    bins = np.linspace(0, 1, 51)
    ax.hist(score[y == 0], bins, color=SERIES[0], alpha=0.75, label="Normal")
    ax.hist(score[y == 1], bins, color=SERIES[1], alpha=0.75, label="Attack")
    ax.set_yscale("log")
    ax.set(xlabel="P(attack)", ylabel="count (log)", title="Score distribution")
    ax.legend()
    _save(fig, out, f"{name}_score_distribution")

    tn, fp, fn, tp = M.confusion_matrix(y, (score >= 0.5).astype(int), labels=[0, 1]).ravel()
    confusion_matrices(np.array([[tn, fp], [fn, tp]]), ["Normal", "Attack"], out, prefix=name)


def calibration(y_bin, score, conf, correct, out, n_bins=15):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    for ax, (yy, ss, title) in zip(axes, [(y_bin, score, "Binary head: P(attack)"),
                                          (correct, conf, "Multi-class: top-label confidence")]):
        bins = np.linspace(0, 1, n_bins + 1)
        idx = np.clip(np.digitize(ss, bins) - 1, 0, n_bins - 1)
        xs, ys, ns = [], [], []
        for b in range(n_bins):
            m = idx == b
            if m.any():
                xs.append(ss[m].mean())
                ys.append(yy[m].mean())
                ns.append(m.sum())
        ax.plot([0, 1], [0, 1], color=GRID, lw=1)
        ax.plot(xs, ys, color=SERIES[0], marker="o", ms=4)
        ax.set(xlabel="Predicted probability", ylabel="Observed frequency", title=title,
               xlim=(0, 1), ylim=(0, 1.02))
    _save(fig, out, "calibration_reliability")


# ---------------------------------------------------------- interpretability
def iteration_attention(iter_w, gates, class_names, out):
    """iter_w, gates: (C, K) per-class mean."""
    K = iter_w.shape[1]
    fig, axes = plt.subplots(1, 2, figsize=(10, max(3, 0.32 * len(class_names) + 1.5)))
    for ax, mat, title in [(axes[0], iter_w, "Iteration attention weight"),
                           (axes[1], gates, "Mean GRU gate opening z_k")]:
        im = ax.imshow(mat, cmap=SEQ, aspect="auto")
        ax.grid(False)
        ax.set_xticks(range(K), [f"k={k + 1}" for k in range(K)])
        ax.set_yticks(range(len(class_names)), class_names)
        for i in range(mat.shape[0]):
            for j in range(K):
                ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if mat[i, j] > (mat.max() + mat.min()) / 2 else INK)
        ax.set_title(title)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    _save(fig, out, "iteration_attention_and_gates")


def feature_attention(cls_attn_class, feature_names, class_names, out, top=20):
    """cls_attn_class: (C, K, F) per-class mean CLS->feature attention."""
    glob = cls_attn_class.mean((0, 1))
    order = np.argsort(glob)[::-1][:top]
    fig, ax = plt.subplots(figsize=(6, 0.26 * len(order) + 1.2))
    ax.barh(range(len(order))[::-1], glob[order], color=SERIES[0], height=0.75)
    ax.set_yticks(range(len(order))[::-1], [feature_names[i] for i in order])
    ax.set(xlabel="mean CLS attention", title=f"Top-{len(order)} features by CLS attention")
    ax.grid(axis="y", visible=False)
    _save(fig, out, "feature_attention_top")

    mat = cls_attn_class.mean(1)[:, order]
    mat = mat / mat.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(0.38 * len(order) + 3, 0.34 * len(class_names) + 1.6))
    im = ax.imshow(mat, cmap=SEQ, aspect="auto")
    ax.grid(False)
    ax.set_xticks(range(len(order)), [feature_names[i] for i in order], rotation=60, ha="right")
    ax.set_yticks(range(len(class_names)), class_names)
    ax.set_title("Per-class CLS attention over top features (row-normalised)")
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    _save(fig, out, "feature_attention_per_class")

    K = cls_attn_class.shape[1]
    if K > 1:
        mat = cls_attn_class.mean(0)[:, order]
        fig, ax = plt.subplots(figsize=(0.38 * len(order) + 3, 0.34 * K + 1.6))
        im = ax.imshow(mat, cmap=SEQ, aspect="auto")
        ax.grid(False)
        ax.set_xticks(range(len(order)), [feature_names[i] for i in order], rotation=60, ha="right")
        ax.set_yticks(range(K), [f"k={k + 1}" for k in range(K)])
        ax.set_title("CLS attention over features at each recursion step")
        fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
        _save(fig, out, "feature_attention_per_iteration")


def embedding_tsne(emb, y, class_names, out, seed=0):
    from sklearn.manifold import TSNE
    xy = TSNE(n_components=2, init="pca", perplexity=30, random_state=seed).fit_transform(emb)
    classes = [c for c in range(len(class_names)) if (y == c).any()]
    if len(classes) <= len(SERIES):
        fig, ax = plt.subplots(figsize=(6, 5))
        for i, c in enumerate(classes):
            m = y == c
            ax.scatter(xy[m, 0], xy[m, 1], s=4, color=SERIES[i], label=class_names[c], alpha=0.7)
        ax.legend(markerscale=3, loc="best", fontsize=7)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title("t-SNE of final representation z")
    else:  # too many classes for distinct hues -> small multiples, one class highlighted per panel
        fig, axes = _grid(len(classes), 2.4, 2.3)
        for ax, c in zip(axes, classes):
            ax.scatter(xy[:, 0], xy[:, 1], s=2, color=GRID)
            m = y == c
            ax.scatter(xy[m, 0], xy[m, 1], s=3, color=SERIES[0])
            ax.set_title(class_names[c], fontsize=8.5)
            ax.set_xticks([])
            ax.set_yticks([])
    _save(fig, out, "embedding_tsne")
