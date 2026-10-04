"""Label-noise ceiling: how well can ANY model do on this split?

  python data_check.py --dataset unsw_official
  python data_check.py --dataset unsw_official --dedup_test
Any data argument of train.py is accepted. Uses the same load/clean/dedup/split as prepare_data,
on the raw feature values (before scaling and rebalancing).

Reports, for the test split:
  - rows whose feature vector also occurs in train, and how many of those have a different label
    from the train majority label for that vector
  - an oracle ceiling: every test row predicted with the majority label of its own feature vector
    in test. No deterministic model can beat this accuracy / macro-F1.
Results go to runs/data_check/<dataset...>.json
"""
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from rain_ids.data import _dedup, _load_raw
from rain_ids.utils import setup_logging
from train import get_args


def _majority(df):
    """most frequent label per feature-vector key (Series indexed by key)"""
    n = df.groupby(["__k", "__y"]).size().reset_index(name="n")
    return n.sort_values("n", kind="stable").drop_duplicates("__k", keep="last").set_index("__k")["__y"]


def _keys(df, feats):
    return pd.util.hash_pandas_object(df[feats].astype(str), index=False).to_numpy()


def main():
    args = get_args()
    name = args.run_name.split("_" + args.model)[0]
    out_dir = Path(args.out_dir) / "data_check"
    out_dir.mkdir(parents=True, exist_ok=True)
    log = setup_logging(out_dir)
    spec, tr, te = _load_raw(args)
    feats = [c for c in tr.columns if c != "__y"]
    if te is None:
        df = _dedup(tr, feats + ["__y"], "full set")
        tr, te = train_test_split(df, test_size=args.test_size, stratify=df["__y"], random_state=args.split_seed)
    else:
        tr = _dedup(tr, feats + ["__y"], "train")
        if args.dedup_test:
            te = _dedup(te, feats + ["__y"], "test")

    tr = tr.assign(__k=_keys(tr, feats))
    te = te.assign(__k=_keys(te, feats))
    y = te["__y"].to_numpy()

    # train majority label per feature vector
    tr_major = _majority(tr)
    in_train = te["__k"].isin(tr_major.index)
    conflict = in_train & (te["__k"].map(tr_major) != te["__y"])
    # oracle: majority label of each vector within test
    te_major = _majority(te)
    oracle = te["__k"].map(te_major).to_numpy()
    ambiguous = te.groupby("__k")["__y"].transform("nunique") > 1

    labels = sorted(set(y))
    per_class = pd.DataFrame({
        "support": te["__y"].value_counts(),
        "in_train": te.loc[in_train, "__y"].value_counts(),
        "train_label_differs": te.loc[conflict, "__y"].value_counts(),
        "ambiguous_in_test": te.loc[ambiguous, "__y"].value_counts(),
        "oracle_f1": pd.Series(f1_score(y, oracle, labels=labels, average=None, zero_division=0), index=labels),
    }).fillna(0).astype({"support": int, "in_train": int, "train_label_differs": int, "ambiguous_in_test": int})

    res = {
        "dataset": name, "n_train": len(tr), "n_test": len(te),
        "test_rows_seen_in_train": int(in_train.sum()),
        "test_rows_seen_in_train_with_other_label": int(conflict.sum()),
        "test_rows_ambiguous_within_test": int(ambiguous.sum()),
        "oracle_ceiling_accuracy": float(accuracy_score(y, oracle)),
        "oracle_ceiling_macro_f1": float(f1_score(y, oracle, average="macro")),
        "per_class": per_class.round(4).to_dict(orient="index"),
    }
    n = len(te)
    log.info(f"[{name}] train {len(tr):,}  test {n:,}")
    log.info(f"  test rows whose features occur in train : {res['test_rows_seen_in_train']:,} "
             f"({res['test_rows_seen_in_train'] / n:.1%})")
    log.info(f"    ... with a different label than train : {res['test_rows_seen_in_train_with_other_label']:,} "
             f"({res['test_rows_seen_in_train_with_other_label'] / n:.1%})")
    log.info(f"  test rows with ambiguous features       : {res['test_rows_ambiguous_within_test']:,} "
             f"({res['test_rows_ambiguous_within_test'] / n:.1%})")
    log.info(f"  ORACLE CEILING  acc={res['oracle_ceiling_accuracy']:.4f}  "
             f"macro-F1={res['oracle_ceiling_macro_f1']:.4f}")
    log.info("\n" + per_class.round(4).to_string())
    (out_dir / f"{name}.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
