"""Per-column-type transforms (design doc steps 5-8).

The fitted Preprocessor is pickled as preproc.pkl so validation, test and
other datasets go through the identical object.
"""
import logging
import pickle

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import QuantileTransformer

log = logging.getLogger("rain")

UNK, RARE = "<UNK>", "<RARE>"   # indices 0 and 1 in every vocab


def port_bucket(p: pd.Series) -> pd.Series:
    p = pd.to_numeric(p, errors="coerce")
    out = np.full(len(p), "nan", dtype=object)
    v = p.to_numpy()
    out[(v >= 0) & (v <= 1023)] = "well_known"
    out[(v >= 1024) & (v <= 49151)] = "registered"
    out[v >= 49152] = "dynamic"
    return pd.Series(out, index=p.index)


def signed_log1p(x):
    return np.sign(x) * np.log1p(np.abs(x))


class Preprocessor:
    def __init__(self, cat_cols, port_cols, rare_min=20, n_quantiles=1000,
                 skew_threshold=2.0, missing_indicator_frac=0.01, seed=0):
        self.cat_cols_in = list(cat_cols)
        self.port_cols_in = list(port_cols)
        self.rare_min = rare_min
        self.n_quantiles = n_quantiles
        self.skew_threshold = skew_threshold
        self.missing_indicator_frac = missing_indicator_frac
        self.seed = seed

    # ------------------------------------------------------------------ fit
    def fit(self, X: pd.DataFrame):
        X = X.copy()
        self.port_cols = [c for c in self.port_cols_in if c in X.columns]
        self.cat_cols = [c for c in self.cat_cols_in if c in X.columns]
        for p in self.port_cols:
            X[p + "_bucket"] = port_bucket(X[p])
        self.cat_tokens = self.cat_cols + [p + "_bucket" for p in self.port_cols]
        self.num_cols = [c for c in X.columns if c not in self.cat_tokens]

        num = X[self.num_cols].apply(pd.to_numeric, errors="coerce").astype("float64")
        for p in self.port_cols:          # negative port (-1 = unknown) -> NaN
            num.loc[num[p] < 0, p] = np.nan
        num = num.replace([np.inf, -np.inf], np.nan)

        miss = num.isna().mean()
        self.medians = num.median().fillna(0.0)
        self.indicator_cols = [c for c in self.num_cols if miss[c] > self.missing_indicator_frac]
        num = num.fillna(self.medians)

        vals = num.to_numpy()
        self.binary_cols = [c for i, c in enumerate(self.num_cols)
                            if np.isin(np.unique(vals[:, i]), [0.0, 1.0]).all()]
        cont = [c for c in self.num_cols if c not in self.binary_cols]
        skew = num[cont].skew().abs() if cont else pd.Series(dtype=float)
        self.log_cols = sorted(set(self.port_cols) |
                               {c for c in cont if skew.get(c, 0) > self.skew_threshold})
        self.quantile_cols = cont
        if cont:
            arr = self._log(num[cont]).to_numpy()
            self.qt = QuantileTransformer(n_quantiles=min(self.n_quantiles, len(arr)),
                                          output_distribution="normal",
                                          subsample=1_000_000, random_state=self.seed)
            self.qt.fit(arr)

        self.vocabs = {}
        for c in self.cat_tokens:
            counts = X[c].astype(str).value_counts()
            frequent = sorted(counts[counts >= self.rare_min].index)
            self.vocabs[c] = {v: i + 2 for i, v in enumerate(frequent)}
            n_rare = int(counts[counts < self.rare_min].sum())
            if n_rare:
                log.info(f"  {c}: {len(frequent)} values kept, {n_rare} train rows -> {RARE}")
        self.num_feature_names = self.num_cols + [c + "_missing" for c in self.indicator_cols]
        self.feature_names = self.num_feature_names + self.cat_tokens
        # columns that must stay exactly 0/1 after SMOTE
        self.discrete_num_idx = [self.num_feature_names.index(c) for c in self.binary_cols] + \
            list(range(len(self.num_cols), len(self.num_feature_names)))
        self._seen = {c: set(self.vocabs[c]) | set(X[c].astype(str).unique()) for c in self.cat_tokens}
        log.info(f"  tokens: {len(self.num_feature_names)} numeric "
                 f"({len(self.binary_cols)} binary, {len(self.log_cols)} log1p, "
                 f"{len(self.indicator_cols)} missing-indicators) + {len(self.cat_tokens)} categorical")
        return self

    def _log(self, df):
        df = df.copy()
        for c in self.log_cols:
            if c in df.columns:
                df[c] = signed_log1p(df[c])
        return df

    @property
    def cat_cardinalities(self):
        return [len(self.vocabs[c]) + 2 for c in self.cat_tokens]

    # ------------------------------------------------------------ transform
    def transform(self, X: pd.DataFrame):
        X = X.copy()
        for p in self.port_cols:
            X[p + "_bucket"] = port_bucket(X[p])
        num = X[self.num_cols].apply(pd.to_numeric, errors="coerce").astype("float64")
        for p in self.port_cols:
            num.loc[num[p] < 0, p] = np.nan
        num = num.replace([np.inf, -np.inf], np.nan)
        ind = num[self.indicator_cols].isna().astype("float64").to_numpy()
        num = num.fillna(self.medians)
        if self.quantile_cols:
            num[self.quantile_cols] = self.qt.transform(self._log(num[self.quantile_cols]).to_numpy())
        x_num = np.concatenate([num.to_numpy(), ind], 1).astype(np.float32)

        x_cat = np.zeros((len(X), len(self.cat_tokens)), dtype=np.int64)
        for j, c in enumerate(self.cat_tokens):
            # frequent -> own index, seen-but-rare -> <RARE> (1), unseen -> <UNK> (0)
            full = {v: self.vocabs[c].get(v, 1) for v in self._seen[c]}
            x_cat[:, j] = X[c].astype(str).map(full).fillna(0).astype(np.int64).to_numpy()
        return x_num, x_cat

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        with open(path, "rb") as f:
            return pickle.load(f)


# ---------------------------------------------------------------- rebalance
def rebalance(x_num, x_cat, y, class_names, undersample: dict, smote_target: int,
              discrete_idx, seed=0):
    """Undersample big classes, BorderlineSMOTE small ones (train split only).

    SMOTE runs on the transformed numeric space; categorical tokens of a
    synthetic row are copied from its nearest real neighbour of the same class.
    """
    from imblearn.over_sampling import SMOTE, BorderlineSMOTE

    rng = np.random.default_rng(seed)
    keep = []
    for c in np.unique(y):
        idx = np.flatnonzero(y == c)
        cap = undersample.get(class_names[c])
        if cap and len(idx) > cap:
            log.info(f"  undersample {class_names[c]}: {len(idx):,} -> {cap:,}")
            idx = rng.choice(idx, cap, replace=False)
        keep.append(idx)
    keep = np.sort(np.concatenate(keep))
    x_num, x_cat, y = x_num[keep], x_cat[keep], y[keep]

    if smote_target <= 0:
        return x_num, x_cat, y
    new_num, new_cat, new_y = [], [], []
    for c in np.unique(y):
        n_c = int((y == c).sum())
        if n_c >= smote_target or n_c < 2:
            continue
        k = min(5, n_c - 1)
        strategy = {int(c): smote_target}
        syn = None
        try:
            sm = BorderlineSMOTE(sampling_strategy=strategy, k_neighbors=k, m_neighbors=10,
                                 random_state=seed)
            xr, _ = sm.fit_resample(x_num, y)
            syn = xr[len(x_num):]
        except Exception as e:  # noqa: BLE001
            log.warning(f"  BorderlineSMOTE failed for {class_names[c]} ({e}); falling back to SMOTE")
        if syn is None or len(syn) < smote_target - n_c:
            # no / too few borderline samples -> plain SMOTE for the remainder
            got = 0 if syn is None else len(syn)
            need = smote_target - n_c - got
            sm = SMOTE(sampling_strategy={int(c): n_c + need}, k_neighbors=k, random_state=seed)
            xr, _ = sm.fit_resample(x_num, y)
            extra = xr[len(x_num):]
            syn = extra if syn is None else np.concatenate([syn, extra])
        real = np.flatnonzero(y == c)
        nn = NearestNeighbors(n_neighbors=1).fit(x_num[real])
        nearest = real[nn.kneighbors(syn, return_distance=False)[:, 0]]
        syn = syn.astype(np.float32)
        if discrete_idx:
            syn[:, discrete_idx] = np.round(np.clip(syn[:, discrete_idx], 0, 1))
        new_num.append(syn)
        new_cat.append(x_cat[nearest])
        new_y.append(np.full(len(syn), c, dtype=y.dtype))
        log.info(f"  SMOTE {class_names[c]}: {n_c:,} -> {n_c + len(syn):,}")
    if new_y:
        x_num = np.concatenate([x_num] + new_num)
        x_cat = np.concatenate([x_cat] + new_cat)
        y = np.concatenate([y] + new_y)
    return x_num, x_cat, y
