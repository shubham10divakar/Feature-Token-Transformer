"""Run every model on every dataset with identical data, then tabulate test (and val) results.

  python benchmark.py                                            # all datasets, all models, seed 0
  python benchmark.py --datasets unsw_official nslkdd --models xgboost mlp
  python benchmark.py --tag dedup -- --dedup_test                # extra args after "--" go to every run
                                                                 # (XGBoost-only options go to XGBoost only)
  python benchmark.py --summary_only                             # just rebuild the table

Models: ceiling (data_check.py oracle), xgboost, xgboost_bal, mlp, ft_transformer, rain_k1, rain.
Finished runs (report/metrics.json present) are skipped; unfinished neural runs resume.
Runs live in <out_dir>/<dataset>[_<tag>]_<model>_s<seed>; the table is <out_dir>/summary[_<tag>].csv.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

DATASETS = ["nslkdd", "unsw_official", "unsw_full", "cicids2017"]
MODELS = {   # name -> (script, extra args, directory suffix written by the script)
    "ceiling": ("data_check.py", [], None),
    "xgboost": ("baselines.py", [], "xgboost"),
    "xgboost_bal": ("baselines.py", ["--xgb_balance", "1"], "xgboost_bal"),
    "mlp": ("train.py", ["--model", "mlp"], "mlp"),
    "ft_transformer": ("train.py", ["--model", "ft_transformer", "--K", "3"], "ft_transformer"),
    "rain_k1": ("train.py", ["--model", "rain", "--K", "1"], "rain_k1"),
    "rain": ("train.py", ["--model", "rain", "--K", "4"], "rain"),
}


def run_dir(out_dir, prefix, model, seed):
    return Path(out_dir) / f"{prefix}_{MODELS[model][2]}_s{seed}"


XGB_ONLY = {"--n_estimators", "--max_depth", "--xgb_lr", "--xgb_balance"}   # each takes one value


def _drop_xgb_args(rest):
    out, it = [], iter(rest)
    for a in it:
        if a in XGB_ONLY:
            next(it, None)
        else:
            out.append(a)
    return out


def launch(model, dataset, prefix, seed, out_dir, rest):
    script, margs, suffix = MODELS[model]
    if script != "baselines.py":
        rest = _drop_xgb_args(rest)
    cmd = [sys.executable, script, "--dataset", dataset, "--seed", str(seed), "--out_dir", out_dir] + margs
    if script == "train.py":
        cmd += ["--run_name", f"{prefix}_{suffix}_s{seed}", "--resume"]
    elif script == "baselines.py":
        cmd += ["--run_name", f"{prefix}_rain"]          # baselines.py names its dir <prefix>_xgboost[_bal]_s<seed>
    else:
        cmd += ["--run_name", f"{prefix}_rain"]          # data_check.py writes <out_dir>/data_check/<prefix>.json
    cmd += rest
    print(">>", " ".join(cmd), flush=True)
    return subprocess.run(cmd).returncode == 0


def collect(out_dir, prefix, dataset, models, seeds):
    rows = []
    for m in models:
        if m == "ceiling":
            f = Path(out_dir) / "data_check" / f"{prefix}.json"
            if f.exists():
                r = json.loads(f.read_text())
                rows.append({"dataset": dataset, "model": "ceiling (oracle)", "seed": "-",
                             "test_acc": r["oracle_ceiling_accuracy"], "test_macro_f1": r["oracle_ceiling_macro_f1"]})
            continue
        for s in seeds:
            f = run_dir(out_dir, prefix, m, s) / "report" / "metrics.json"
            if not f.exists():
                continue
            r = json.loads(f.read_text())
            mc, bh = r["multiclass"], r["binary_head"]
            row = {"dataset": dataset, "model": m, "seed": s,
                   "test_acc": mc["accuracy"], "test_macro_f1": mc["macro_f1"],
                   "test_weighted_f1": mc["weighted_f1"], "test_mcc": mc["mcc"],
                   "test_min_recall": mc["min_class_recall"],
                   "test_bin_f1": bh["f1"], "test_dr": bh["recall_detection_rate"],
                   "test_far": bh["false_alarm_rate_fpr"], "test_bin_auc": bh["roc_auc"]}
            v = f.parent / "val" / "metrics.json"
            if v.exists():
                vr = json.loads(v.read_text())
                row.update(val_acc=vr["multiclass"]["accuracy"], val_macro_f1=vr["multiclass"]["macro_f1"])
            rows.append(row)
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", default=DATASETS)
    p.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--out_dir", default="runs/bench")
    p.add_argument("--tag", default="", help="names this protocol variant, e.g. dedup")
    p.add_argument("--force", action="store_true", help="rerun even if a finished report exists")
    p.add_argument("--summary_only", action="store_true")
    args, rest = p.parse_known_args()
    rest = [a for a in rest if a != "--"]
    tag = f"_{args.tag}" if args.tag else ""

    failed, rows = [], []
    for ds in args.datasets:
        prefix = f"{ds}{tag}"
        if not args.summary_only:
            for m in args.models:
                seeds = [0] if m == "ceiling" else args.seeds
                for s in seeds:
                    done = (Path(args.out_dir) / "data_check" / f"{prefix}.json" if m == "ceiling"
                            else run_dir(args.out_dir, prefix, m, s) / "report" / "metrics.json")
                    if done.exists() and not args.force:
                        print(f"-- skip {ds} {m} s{s} (finished)", flush=True)
                        continue
                    if not launch(m, ds, prefix, s, args.out_dir, rest):
                        failed.append(f"{ds}/{m}/s{s}")
        rows += collect(args.out_dir, prefix, ds, args.models, args.seeds)

    if not rows:
        print("no finished runs yet")
        return
    df = pd.DataFrame(rows)
    out = Path(args.out_dir) / f"summary{tag}.csv"
    df.to_csv(out, index=False)
    metric_cols = [c for c in df.columns if c.startswith(("test_", "val_"))]
    agg = df.groupby(["dataset", "model"], sort=False)[metric_cols].agg(["mean", "std"])
    agg.to_csv(out.with_name(out.stem + "_mean_std.csv"))
    pd.set_option("display.width", 250)
    show = ["test_acc", "test_macro_f1", "test_weighted_f1", "test_min_recall", "test_bin_f1",
            "test_far", "val_macro_f1"]
    print(df.groupby(["dataset", "model"], sort=False)[[c for c in show if c in df]].mean().round(4).to_string())
    print(f"\nsaved {out}")
    if failed:
        print("FAILED:", ", ".join(failed))


if __name__ == "__main__":
    main()
