"""Recursion-depth ablation: train RAIN-IDS for each K and tabulate test results.

  python ablation.py --dataset unsw_official --Ks 1 2 4 6 8 --seeds 0 1 2
Every run resumes automatically, so the sweep can be interrupted and restarted.
Extra arguments after "--" are passed to train.py, e.g.  -- --embedding periodic
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--Ks", type=int, nargs="+", default=[1, 2, 4, 6, 8])
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--out_dir", default="runs")
    args, rest = p.parse_known_args()
    rest = [a for a in rest if a != "--"]

    rows = []
    for K in args.Ks:
        for s in args.seeds:
            name = f"ablation_{args.dataset}_K{K}_s{s}"
            cmd = [sys.executable, "train.py", "--dataset", args.dataset, "--K", str(K), "--seed", str(s),
                   "--run_name", name, "--out_dir", args.out_dir, "--resume", "--report", "basic"]
            if K >= 6:
                cmd.append("--grad_checkpoint")
            print(">>", " ".join(cmd + rest), flush=True)
            subprocess.run(cmd + rest, check=True)
            r = json.loads((Path(args.out_dir) / name / "report" / "metrics.json").read_text())
            rows.append({"K": K, "seed": s, "acc": r["multiclass"]["accuracy"],
                         "macro_f1": r["multiclass"]["macro_f1"], "weighted_f1": r["multiclass"]["weighted_f1"],
                         "min_recall": r["multiclass"]["min_class_recall"],
                         "bin_acc": r["binary_head"]["accuracy"], "bin_f1": r["binary_head"]["f1"]})
    df = pd.DataFrame(rows)
    agg = df.groupby("K").agg(["mean", "std"]).drop(columns="seed")
    out = Path(args.out_dir) / f"ablation_{args.dataset}_K_summary.csv"
    df.to_csv(out.with_name(out.stem + "_all.csv"), index=False)
    agg.to_csv(out)
    print(agg.round(4).to_string())
    print(f"saved {out}")


if __name__ == "__main__":
    main()
