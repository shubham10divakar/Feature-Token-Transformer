"""XGBoost baseline on exactly the same processed data (same cache, same split).

  python baselines.py --dataset unsw_official
  python baselines.py --dataset unsw_official --xgb_balance 1    # balanced sample weights
Any data argument of train.py (e.g. --drop_leaky, --cic_label, --dedup_test) is accepted.
XGBoost options: --n_estimators, --max_depth, --xgb_lr, --xgb_balance (0/1).
MLP baseline: python train.py --model mlp --dataset ...
FT-Transformer baseline: python train.py --model ft_transformer --dataset ...
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from rain_ids import metrics as Mx
from rain_ids import plots
from rain_ids.data import prepare_data
from rain_ids.evaluate import _latex_tables, _print_summary
from rain_ids.utils import setup_logging, write_json
from train import get_args


def main():
    extra = {"--n_estimators": 2000, "--max_depth": 8, "--xgb_lr": 0.05, "--xgb_balance": 0}
    argv, xgb_cfg = [], dict(extra)
    it = iter(sys.argv[1:])
    for a in it:
        if a in extra:
            xgb_cfg[a] = type(extra[a])(next(it))
        else:
            argv.append(a)
    args = get_args(argv)
    tag = "_bal" if xgb_cfg["--xgb_balance"] else ""
    run_dir = Path(args.out_dir) / f"{args.run_name.split('_' + args.model)[0]}_xgboost{tag}_s{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    log = setup_logging(run_dir)
    arrays, meta, _, _ = prepare_data(args)
    names, normal = meta["class_names"], meta["normal_idx"]

    def X(s):
        return np.concatenate([arrays[f"{s}_num"], arrays[f"{s}_cat"].astype(np.float32)], 1)

    clf = xgb.XGBClassifier(n_estimators=xgb_cfg["--n_estimators"], max_depth=xgb_cfg["--max_depth"],
                            learning_rate=xgb_cfg["--xgb_lr"], subsample=0.8, colsample_bytree=0.8,
                            tree_method="hist", device="cuda" if args.device.startswith("cuda") else "cpu",
                            early_stopping_rounds=50, eval_metric="mlogloss", random_state=args.seed)
    sw = None
    if xgb_cfg["--xgb_balance"]:
        cnt = np.bincount(arrays["train_y"], minlength=len(names)).astype(float)
        sw = (cnt.sum() / np.clip(cnt, 1, None) / len(names))[arrays["train_y"]]
    t = time.time()
    clf.fit(X("train"), arrays["train_y"], sample_weight=sw,
            eval_set=[(X("val"), arrays["val_y"])], verbose=100)
    train_time = time.time() - t
    t = time.time()
    prob = clf.predict_proba(X("test"))
    infer = time.time() - t
    y = arrays["test_y"]
    y_bin = (y != normal).astype(int)
    mc, per_class, cm = Mx.multiclass_metrics(y, prob, names)
    bm = Mx.binary_metrics(y_bin, 1 - prob[:, normal])
    report = {"split": "test", "n_samples": int(len(y)), "multiclass": mc,
              "binary_head": bm, "binary_from_multiclass_head": bm,
              "efficiency": {"train_time_s": train_time, "throughput_samples_per_s": len(y) / infer,
                             "best_iteration": int(clf.best_iteration)}}
    out = run_dir / "report"
    out.mkdir(exist_ok=True)
    write_json(out / "metrics.json", report)
    per_class.to_csv(out / "per_class_metrics.csv", index=False)
    pd.DataFrame(cm, index=names, columns=names).to_csv(out / "confusion_matrix.csv")
    np.savez_compressed(out / "predictions.npz", y=y, prob=prob, p_bin=1 - prob[:, normal])
    write_json(run_dir / "args.json", {**vars(args), **{k.lstrip("-"): v for k, v in xgb_cfg.items()}})
    _latex_tables(report, per_class, out)
    _print_summary(report, per_class)
    fig = out / "figures"
    plots.confusion_matrices(cm, names, fig)
    plots.multiclass_roc_pr(y, prob, names, fig)
    plots.per_class_bars(per_class, fig)
    plots.binary_curves(y_bin, 1 - prob[:, normal], fig, name="binary_from_multiclass")
    pv = clf.predict_proba(X("val"))
    vmc, vpc, _ = Mx.multiclass_metrics(arrays["val_y"], pv, names)
    vbm = Mx.binary_metrics((arrays["val_y"] != normal).astype(int), 1 - pv[:, normal])
    (out / "val").mkdir(exist_ok=True)
    write_json(out / "val" / "metrics.json", {"split": "val", "n_samples": int(len(pv)), "multiclass": vmc,
                                              "binary_head": vbm, "binary_from_multiclass_head": vbm})
    vpc.to_csv(out / "val" / "per_class_metrics.csv", index=False)
    imp = pd.Series(clf.feature_importances_, index=meta["feature_names"]).sort_values(ascending=False)
    imp.to_csv(out / "xgb_feature_importance.csv")
    log.info(f"done: {run_dir.resolve()}")


if __name__ == "__main__":
    main()
