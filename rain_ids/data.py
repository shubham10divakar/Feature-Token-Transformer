"""Load, deduplicate, split, transform and cache a dataset (design doc steps 1-8).

prepare_data() returns the processed arrays and caches them (plus preproc.pkl)
under cache/<dataset>_<hash>/ so a resumed or repeated run skips the work.
"""
import hashlib
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import DATASETS, NSL_TEST21
from .preprocess import Preprocessor, rebalance

log = logging.getLogger("rain")

# arguments that change the processed data (anything else can vary across runs)
DATA_KEYS = ["dataset", "data_root", "cic_label", "nsl_test", "drop_leaky", "dedup_test",
             "test_size", "val_size", "split_seed", "undersample", "smote_target",
             "no_rebalance", "rare_min", "max_train_rows"]


def parse_undersample(s, default):
    if s is None:
        return dict(default)
    if s.strip().lower() in ("", "none", "off"):
        return {}
    out = {}
    for part in s.split(","):
        k, v = part.split("=")
        out[k.strip()] = int(v)
    return out


def _load(root: Path, rel: str) -> pd.DataFrame:
    path = root / rel
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - check --data_root")
    log.info(f"loading {path}")
    return pd.read_parquet(path)


def _dedup(df, cols, what):
    before = df[cols[-1]].value_counts()
    df = df.drop_duplicates(subset=cols, ignore_index=True)
    removed = (before - df[cols[-1]].value_counts().reindex(before.index, fill_value=0))
    log.info(f"dedup {what}: removed {int(removed.sum()):,} rows "
             f"{ {k: int(v) for k, v in removed.items() if v} }")
    return df


def _load_raw(args):
    spec = DATASETS[args.dataset]
    root = Path(args.data_root)
    label = spec.label_col
    if args.dataset == "cicids2017":
        label = spec.alt_label_cols[args.cic_label]

    if spec.split == "official":
        tr = _load(root, spec.train_file)
        te_file = NSL_TEST21 if (args.dataset == "nslkdd" and args.nsl_test == "21") else spec.test_file
        te = _load(root, te_file)
        if args.dataset.startswith("unsw") and len(tr) < len(te):
            # the published UNSW-NB15 "training"/"testing" files are swapped (175k train / 82k test)
            log.warning(f"swapped-file check: train has {len(tr):,} rows < test {len(te):,}; swapping")
            tr, te = te, tr
    else:
        tr, te = _load(root, spec.file), None

    def clean(df):
        df = df.copy()
        y = df[label].astype(str).str.strip()
        if args.dataset.startswith("unsw"):
            y = y.replace({"Backdoors": "Backdoor", "": "Normal", "nan": "Normal"})
        drop = [c for c in spec.drop_cols + [label] if c in df.columns]
        if args.drop_leaky:
            drop += [c for c in spec.leak_cols if c in df.columns]
        X = df.drop(columns=drop)
        X["__y"] = y.values
        return X

    tr = clean(tr)
    te = clean(te) if te is not None else None
    return spec, tr, te


def prepare_data(args, cache_root="cache"):
    spec = DATASETS[args.dataset]
    cfg = {k: getattr(args, k, None) for k in DATA_KEYS}
    cfg["data_root"] = str(Path(cfg["data_root"]).resolve())
    h = hashlib.md5(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:10]
    cdir = Path(cache_root) / f"{args.dataset}_{h}"
    if (cdir / "arrays.npz").exists() and not getattr(args, "rebuild_cache", False):
        log.info(f"using cached processed data: {cdir}")
        arr = dict(np.load(cdir / "arrays.npz"))
        meta = json.loads((cdir / "meta.json").read_text())
        return arr, meta, Preprocessor.load(cdir / "preproc.pkl"), cdir

    cdir.mkdir(parents=True, exist_ok=True)
    spec, tr, te = _load_raw(args)
    feats = [c for c in tr.columns if c != "__y"]

    # 3. deduplicate on X + y before splitting
    if spec.split == "random":
        df = _dedup(tr, feats + ["__y"], "full set")
        tr, te = train_test_split(df, test_size=args.test_size, stratify=df["__y"],
                                  random_state=args.split_seed)
        tr, te = tr.reset_index(drop=True), te.reset_index(drop=True)
    else:
        tr = _dedup(tr, feats + ["__y"], "train")
        if args.dedup_test:
            te = _dedup(te, feats + ["__y"], "test")
    # train/test overlap check on X
    key_tr = pd.util.hash_pandas_object(tr[feats].astype(str), index=False)
    key_te = pd.util.hash_pandas_object(te[feats].astype(str), index=False)
    overlap = int(key_te.isin(set(key_tr.values)).sum())
    log.info(f"train/test overlap (identical feature vectors): {overlap:,} of {len(te):,} test rows")

    if args.max_train_rows and len(tr) > args.max_train_rows:   # quick debugging runs
        tr, _ = train_test_split(tr, train_size=args.max_train_rows, stratify=tr["__y"],
                                 random_state=args.split_seed)

    # 4. validation = 10% of train, stratified
    tr, va = train_test_split(tr, test_size=args.val_size, stratify=tr["__y"],
                              random_state=args.split_seed)

    class_names = sorted(set(tr["__y"]) | set(va["__y"]) | set(te["__y"]))
    cidx = {c: i for i, c in enumerate(class_names)}
    normal_idx = cidx[spec.normal_label]

    # 5-6. fit transforms on train only
    pp = Preprocessor(spec.cat_cols, spec.port_cols, rare_min=args.rare_min, seed=args.split_seed)
    pp.fit(tr[feats])
    arrays = {}
    for name, d in (("train", tr), ("val", va), ("test", te)):
        xn, xc = pp.transform(d[feats])
        arrays[f"{name}_num"], arrays[f"{name}_cat"] = xn, xc
        arrays[f"{name}_y"] = d["__y"].map(cidx).to_numpy().astype(np.int64)

    counts_before = np.bincount(arrays["train_y"], minlength=len(class_names))
    # 7. rebalance train only
    if not args.no_rebalance:
        us = parse_undersample(args.undersample, spec.undersample)
        xn, xc, y = rebalance(arrays["train_num"], arrays["train_cat"], arrays["train_y"],
                              class_names, us, args.smote_target, pp.discrete_num_idx,
                              seed=args.split_seed)
        perm = np.random.default_rng(args.split_seed).permutation(len(y))
        arrays["train_num"], arrays["train_cat"], arrays["train_y"] = xn[perm], xc[perm], y[perm]

    meta = {
        "dataset": args.dataset, "class_names": class_names, "normal_idx": normal_idx,
        "feature_names": pp.feature_names, "num_feature_names": pp.num_feature_names,
        "cat_feature_names": pp.cat_tokens, "cat_cardinalities": pp.cat_cardinalities,
        "n_num": len(pp.num_feature_names), "train_test_overlap": overlap,
        "counts": {s: np.bincount(arrays[f"{s}_y"], minlength=len(class_names)).tolist()
                   for s in ("train", "val", "test")},
        "train_counts_before_rebalance": counts_before.tolist(),
        "config": cfg,
    }
    # 8. persist
    pp.save(cdir / "preproc.pkl")
    np.savez(cdir / "arrays.npz", **arrays)
    (cdir / "meta.json").write_text(json.dumps(meta, indent=2))
    log.info(f"cached processed data to {cdir}")
    return arrays, meta, pp, cdir
