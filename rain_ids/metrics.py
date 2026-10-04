"""All metrics needed for the paper: multi-class, binary and per-class."""
import numpy as np
import pandas as pd
from sklearn import metrics as M


def _safe(fn, *a, **k):
    try:
        return float(fn(*a, **k))
    except ValueError:
        return float("nan")


def quick_scores(y, prob):
    """Cheap per-epoch validation scores."""
    pred = prob.argmax(1)
    return {"acc": M.accuracy_score(y, pred),
            "macro_f1": M.f1_score(y, pred, average="macro", zero_division=0)}


def multiclass_metrics(y, prob, class_names):
    C = len(class_names)
    labels = np.arange(C)
    pred = prob.argmax(1)
    cm = M.confusion_matrix(y, pred, labels=labels)
    present = np.unique(y)
    y_onehot = np.eye(C)[y]

    out = {
        "accuracy": M.accuracy_score(y, pred),
        "balanced_accuracy": M.balanced_accuracy_score(y, pred),
        "mcc": M.matthews_corrcoef(y, pred),
        "cohen_kappa": M.cohen_kappa_score(y, pred),
        "log_loss": _safe(M.log_loss, y, np.clip(prob, 1e-12, 1), labels=labels),
        "top2_accuracy": _safe(M.top_k_accuracy_score, y, prob, k=2, labels=labels) if C > 2 else float("nan"),
        "brier_multi": float(np.mean(np.sum((prob - y_onehot) ** 2, 1))),
    }
    for avg in ("macro", "weighted", "micro"):
        p, r, f, _ = M.precision_recall_fscore_support(y, pred, labels=labels, average=avg, zero_division=0)
        out[f"{avg}_precision"], out[f"{avg}_recall"], out[f"{avg}_f1"] = p, r, f
    # AUCs over classes present in y (OvR)
    pr = prob[:, present]
    pr = pr / pr.sum(1, keepdims=True)
    if len(present) > 2:
        out["roc_auc_ovr_macro"] = _safe(M.roc_auc_score, y, pr, multi_class="ovr", average="macro", labels=present)
        out["roc_auc_ovr_weighted"] = _safe(M.roc_auc_score, y, pr, multi_class="ovr", average="weighted", labels=present)
        out["roc_auc_ovo_macro"] = _safe(M.roc_auc_score, y, pr, multi_class="ovo", average="macro", labels=present)
    elif len(present) == 2:              # binary task: OvR / OvO AUC is the ordinary AUC
        auc = _safe(M.roc_auc_score, y == present[1], pr[:, 1])
        out["roc_auc_ovr_macro"] = out["roc_auc_ovr_weighted"] = out["roc_auc_ovo_macro"] = auc
    out["pr_auc_macro"] = _safe(M.average_precision_score, y_onehot[:, present], prob[:, present], average="macro")
    out["pr_auc_micro"] = _safe(M.average_precision_score, y_onehot[:, present], prob[:, present], average="micro")

    # per-class table
    p, r, f, s = M.precision_recall_fscore_support(y, pred, labels=labels, zero_division=0)
    tp = np.diag(cm)
    fp = cm.sum(0) - tp
    fn = cm.sum(1) - tp
    tn = cm.sum() - tp - fp - fn
    rows = []
    for i, c in enumerate(class_names):
        auc = _safe(M.roc_auc_score, y_onehot[:, i], prob[:, i]) if 0 < s[i] < len(y) else float("nan")
        ap = _safe(M.average_precision_score, y_onehot[:, i], prob[:, i]) if s[i] > 0 else float("nan")
        rows.append({
            "class": c, "support": int(s[i]), "precision": p[i], "recall": r[i], "f1": f[i],
            "specificity": tn[i] / max(tn[i] + fp[i], 1), "fpr": fp[i] / max(fp[i] + tn[i], 1),
            "fnr": fn[i] / max(fn[i] + tp[i], 1), "roc_auc": auc, "pr_auc": ap,
            "tp": int(tp[i]), "fp": int(fp[i]), "fn": int(fn[i]), "tn": int(tn[i]),
        })
    per_class = pd.DataFrame(rows)
    sup = per_class["support"] > 0
    out["min_class_recall"] = float(per_class.loc[sup, "recall"].min())
    out["classes_below_0.5_recall"] = per_class.loc[sup & (per_class["recall"] < 0.5), "class"].tolist()
    out["macro_fpr"] = float(per_class.loc[sup, "fpr"].mean())
    return {k: (float(v) if isinstance(v, (np.floating, float, int, np.integer)) else v)
            for k, v in out.items()}, per_class, cm


def binary_metrics(y, score, threshold=0.5):
    """y in {0,1} (1 = attack), score = P(attack)."""
    pred = (score >= threshold).astype(int)
    tn, fp, fn, tp = M.confusion_matrix(y, pred, labels=[0, 1]).ravel()
    out = {
        "threshold": threshold,
        "accuracy": M.accuracy_score(y, pred),
        "balanced_accuracy": M.balanced_accuracy_score(y, pred),
        "precision": M.precision_score(y, pred, zero_division=0),
        "recall_detection_rate": M.recall_score(y, pred, zero_division=0),
        "f1": M.f1_score(y, pred, zero_division=0),
        "f2": M.fbeta_score(y, pred, beta=2, zero_division=0),
        "specificity": tn / max(tn + fp, 1),
        "false_alarm_rate_fpr": fp / max(fp + tn, 1),
        "fnr_miss_rate": fn / max(fn + tp, 1),
        "npv": tn / max(tn + fn, 1),
        "mcc": M.matthews_corrcoef(y, pred),
        "cohen_kappa": M.cohen_kappa_score(y, pred),
        "roc_auc": _safe(M.roc_auc_score, y, score),
        "pr_auc_ap": _safe(M.average_precision_score, y, score),
        "log_loss": _safe(M.log_loss, y, np.clip(score, 1e-12, 1 - 1e-12), labels=[0, 1]),
        "brier": M.brier_score_loss(y, score),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    }
    # threshold that maximises F1 (report only; choose thresholds on val, not test)
    pr, rc, th = M.precision_recall_curve(y, score)
    f1s = 2 * pr * rc / np.clip(pr + rc, 1e-12, None)
    if len(th):
        i = int(np.nanargmax(f1s[:-1]))
        out["best_f1_threshold"], out["best_f1"] = float(th[i]), float(f1s[i])
    fpr, tpr, _ = M.roc_curve(y, score)
    for target in (0.001, 0.01, 0.05):
        out[f"tpr_at_fpr_{target}"] = float(np.interp(target, fpr, tpr))
    out["ece"] = expected_calibration_error(y, score)
    return {k: (float(v) if not isinstance(v, (list, str)) else v) for k, v in out.items()}


def expected_calibration_error(y, conf_or_score, n_bins=15, correct=None):
    """Binary ECE if `correct` is None, else top-label ECE with confidences."""
    if correct is None:
        bins = np.linspace(0, 1, n_bins + 1)
        idx = np.clip(np.digitize(conf_or_score, bins) - 1, 0, n_bins - 1)
        ece = 0.0
        for b in range(n_bins):
            m = idx == b
            if m.any():
                ece += m.mean() * abs(y[m].mean() - conf_or_score[m].mean())
        return float(ece)
    conf = conf_or_score
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(conf, bins) - 1, 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(ece)
