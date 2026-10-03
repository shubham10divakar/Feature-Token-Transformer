"""Inference and the end-of-run report (metrics tables + all curves)."""
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from . import metrics as Mx
from . import plots
from .utils import write_json

log = logging.getLogger("rain")


@torch.no_grad()
def predict(model, x_num, x_cat, device, batch_size=4096, need_weights=False, amp_dtype=None):
    """Returns probs (N, C), P(attack) from the binary head (N,), embeddings, and
    (if need_weights) attention / gate tensors."""
    model.eval()
    probs, pbin, emb, extras = [], [], [], {"iter_attn": [], "cls_attn": [], "gates": []}
    for i in range(0, len(x_num), batch_size):
        xn = x_num[i:i + batch_size].to(device, non_blocking=True)
        xc = x_cat[i:i + batch_size].to(device, non_blocking=True)
        with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            o = model(xn, xc, need_weights=need_weights)
        probs.append(o["logits"].float().softmax(-1).cpu())
        pbin.append(torch.sigmoid(o["logit_bin"].float()).cpu())
        emb.append(o["embedding"].float().cpu())
        if need_weights:
            for k in extras:
                if k in o:
                    extras[k].append(o[k].float().cpu())
    out = {"prob": torch.cat(probs).numpy(), "p_bin": torch.cat(pbin).numpy(),
           "embedding": torch.cat(emb).numpy()}
    for k, v in extras.items():
        if v:
            out[k] = torch.cat(v).numpy()
    return out


@torch.no_grad()
def measure_efficiency(model, x_num, x_cat, device, batch_size=4096, reps=5):
    n_params = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / 2 ** 20
    model.eval()
    xn, xc = x_num[:batch_size].to(device), x_cat[:batch_size].to(device)
    for _ in range(2):
        model(xn, xc)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(reps):
        model(xn, xc)
    if device.type == "cuda":
        torch.cuda.synchronize()
    dt = (time.perf_counter() - t) / reps
    x1n, x1c = x_num[:1].to(device), x_cat[:1].to(device)
    t = time.perf_counter()
    for _ in range(50):
        model(x1n, x1c)
    if device.type == "cuda":
        torch.cuda.synchronize()
    lat = (time.perf_counter() - t) / 50
    return {"params": n_params, "trainable_params": n_train, "model_size_mb": size_mb,
            "throughput_samples_per_s": len(xn) / dt, "latency_single_sample_ms": lat * 1000,
            "batch_inference_ms_per_sample": dt / len(xn) * 1000}


def _per_class_mean(arr, y, C):
    return np.stack([arr[y == c].mean(0) if (y == c).any() else np.zeros(arr.shape[1:]) for c in range(C)])


def full_report(model, arrays, meta, device, out_dir: Path, split="test", full=True,
                tsne=False, amp_dtype=None, train_info=None):
    """Compute every metric and figure for `split` and write them to out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = out_dir / "figures"
    names = meta["class_names"]
    C, normal = len(names), meta["normal_idx"]
    x_num = torch.from_numpy(arrays[f"{split}_num"])
    x_cat = torch.from_numpy(arrays[f"{split}_cat"])
    y = arrays[f"{split}_y"]
    y_bin = (y != normal).astype(int)

    pr = predict(model, x_num, x_cat, device, need_weights=full, amp_dtype=amp_dtype)
    prob, p_bin = pr["prob"], pr["p_bin"]
    p_attack_from_multi = 1.0 - prob[:, normal]

    mc, per_class, cm = Mx.multiclass_metrics(y, prob, names)
    pred = prob.argmax(1)
    mc["ece_top_label"] = Mx.expected_calibration_error(None, prob.max(1), correct=(pred == y).astype(float))
    bh = Mx.binary_metrics(y_bin, p_bin)
    bm = Mx.binary_metrics(y_bin, p_attack_from_multi)
    report = {"split": split, "n_samples": int(len(y)), "multiclass": mc,
              "binary_head": bh, "binary_from_multiclass_head": bm}
    if full:
        report["efficiency"] = measure_efficiency(model, x_num, x_cat, device)
    if train_info:
        report["training"] = train_info

    write_json(out_dir / "metrics.json", report)
    per_class.to_csv(out_dir / "per_class_metrics.csv", index=False)
    pd.DataFrame(cm, index=names, columns=names).to_csv(out_dir / "confusion_matrix.csv")
    _latex_tables(report, per_class, out_dir)
    np.savez_compressed(out_dir / "predictions.npz", y=y, prob=prob, p_bin=p_bin)
    _print_summary(report, per_class)

    if not full:
        return report
    log.info("rendering figures ...")
    plots.confusion_matrices(cm, names, fig_dir)
    plots.multiclass_roc_pr(y, prob, names, fig_dir)
    plots.per_class_bars(per_class, fig_dir)
    plots.binary_curves(y_bin, p_bin, fig_dir, name="binary_head")
    plots.binary_curves(y_bin, p_attack_from_multi, fig_dir, name="binary_from_multiclass")
    plots.calibration(y_bin, p_bin, prob.max(1), (pred == y).astype(float), fig_dir)
    if "cls_attn" in pr:
        cls_c = _per_class_mean(pr["cls_attn"], y, C)
        plots.feature_attention(cls_c, meta["feature_names"], names, fig_dir)
        pd.DataFrame(pr["cls_attn"].mean((0, 1)), index=meta["feature_names"],
                     columns=["mean_cls_attention"]).sort_values("mean_cls_attention", ascending=False) \
            .to_csv(out_dir / "feature_attention.csv")
    if "iter_attn" in pr:
        iw, gt = _per_class_mean(pr["iter_attn"], y, C), _per_class_mean(pr["gates"], y, C)
        plots.iteration_attention(iw, gt, names, fig_dir)
        K = iw.shape[1]
        pd.DataFrame(iw, index=names, columns=[f"k{k + 1}" for k in range(K)]).to_csv(out_dir / "iteration_attention.csv")
    if tsne:
        rng = np.random.default_rng(0)
        idx = np.concatenate([rng.choice(np.flatnonzero(y == c), min(600, (y == c).sum()), replace=False)
                              for c in range(C) if (y == c).any()])
        log.info(f"t-SNE on {len(idx):,} points ...")
        plots.embedding_tsne(pr["embedding"][idx], y[idx], names, fig_dir)
    log.info(f"report written to {out_dir}")
    return report


def _latex_tables(report, per_class, out_dir):
    pc = per_class[["class", "support", "precision", "recall", "f1", "fpr", "roc_auc"]].copy()
    pc.columns = ["Class", "Support", "Precision", "Recall", "F1", "FPR", "ROC-AUC"]
    (out_dir / "per_class_table.tex").write_text(
        pc.to_latex(index=False, float_format="%.4f", caption="Per-class results", label="tab:per_class"))
    mc, bh = report["multiclass"], report["binary_head"]
    rows = [("Accuracy", mc["accuracy"], bh["accuracy"]),
            ("Balanced accuracy", mc["balanced_accuracy"], bh["balanced_accuracy"]),
            ("Precision (macro / binary)", mc["macro_precision"], bh["precision"]),
            ("Recall (macro / DR)", mc["macro_recall"], bh["recall_detection_rate"]),
            ("F1 (macro / binary)", mc["macro_f1"], bh["f1"]),
            ("Weighted F1", mc["weighted_f1"], float("nan")),
            ("MCC", mc["mcc"], bh["mcc"]),
            ("Cohen's kappa", mc["cohen_kappa"], bh["cohen_kappa"]),
            ("ROC-AUC", mc.get("roc_auc_ovr_macro", float("nan")), bh["roc_auc"]),
            ("PR-AUC", mc["pr_auc_macro"], bh["pr_auc_ap"]),
            ("FPR / false alarm rate", mc["macro_fpr"], bh["false_alarm_rate_fpr"])]
    df = pd.DataFrame(rows, columns=["Metric", "Multi-class", "Binary"])
    (out_dir / "summary_table.tex").write_text(
        df.to_latex(index=False, float_format="%.4f", na_rep="--", caption="Overall results", label="tab:overall"))
    df.to_csv(out_dir / "summary_table.csv", index=False)


def _print_summary(r, per_class):
    mc, bh = r["multiclass"], r["binary_head"]
    log.info("=" * 70)
    log.info(f"[{r['split']}] n={r['n_samples']:,}")
    log.info(f"multi-class : acc={mc['accuracy']:.4f}  macro-F1={mc['macro_f1']:.4f}  "
             f"weighted-F1={mc['weighted_f1']:.4f}  MCC={mc['mcc']:.4f}  "
             f"AUC={mc.get('roc_auc_ovr_macro', float('nan')):.4f}  min-recall={mc['min_class_recall']:.3f}")
    log.info(f"binary head : acc={bh['accuracy']:.4f}  F1={bh['f1']:.4f}  DR={bh['recall_detection_rate']:.4f}  "
             f"FAR={bh['false_alarm_rate_fpr']:.4f}  AUC={bh['roc_auc']:.4f}")
    if mc["classes_below_0.5_recall"]:
        log.info(f"classes below 0.5 recall: {mc['classes_below_0.5_recall']}")
    log.info("\n" + per_class[["class", "support", "precision", "recall", "f1", "fpr"]]
             .to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    log.info("=" * 70)
