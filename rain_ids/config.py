"""Dataset registry and default paths.

Every dataset goes through the same code path; only the spec below differs.
Point RAIN_DATA_ROOT (env var) or --data_root at the datasets folder.
"""
import os
from dataclasses import dataclass, field

DEFAULT_DATA_ROOT = os.environ.get(
    "RAIN_DATA_ROOT",
    r"D:\D\my docs\my docs\ideas\attention based works\Intrusion detection works\datasets",
)


@dataclass
class DatasetSpec:
    name: str
    description: str
    label_col: str
    normal_label: str
    split: str                      # "official" (train/test files) or "random" (stratified split)
    train_file: str = ""
    test_file: str = ""
    file: str = ""                  # single file for "random" split
    drop_cols: list = field(default_factory=list)   # never used as features
    cat_cols: list = field(default_factory=list)
    port_cols: list = field(default_factory=list)
    leak_cols: list = field(default_factory=list)   # optionally dropped with --drop_leaky
    undersample: dict = field(default_factory=dict) # class name -> cap (train only)
    alt_label_cols: dict = field(default_factory=dict)


DATASETS = {
    "unsw_official": DatasetSpec(
        name="unsw_official",
        description="UNSW-NB15 official train/test split (175k / 82k, 10 classes)",
        train_file="unsw-nb15/clean/unsw_nb15_training_set.parquet",
        test_file="unsw-nb15/clean/unsw_nb15_testing_set.parquet",
        split="official",
        label_col="attack_cat", normal_label="Normal",
        drop_cols=["id", "label", "srcip", "dstip", "stime", "ltime"],
        cat_cols=["proto", "service", "state"],
        port_cols=["sport", "dsport"],
        leak_cols=["sttl", "dttl", "ct_state_ttl"],
        undersample={"Normal": 80_000, "Generic": 30_000},
    ),
    "unsw_full": DatasetSpec(
        name="unsw_full",
        description="UNSW-NB15 full deduplicated set (2.06M rows), stratified 80/20 split",
        file="unsw-nb15/clean/unsw_nb15_full_dedup.parquet",
        split="random",
        label_col="attack_cat", normal_label="Normal",
        drop_cols=["label", "srcip", "dstip", "stime", "ltime"],
        cat_cols=["proto", "service", "state"],
        port_cols=["sport", "dsport"],
        leak_cols=["sttl", "dttl", "ct_state_ttl"],
        undersample={"Normal": 80_000, "Generic": 30_000},
    ),
    "cicids2017": DatasetSpec(
        name="cicids2017",
        description="CIC-IDS-2017 cleaned (2.23M rows), attack families (8 classes), stratified 80/20 split",
        file="cicids2017/clean/cicids2017_clean.parquet",
        split="random",
        label_col="Attack Family", normal_label="Benign",
        alt_label_cols={"family": "Attack Family", "label": "Label"},
        drop_cols=["Label", "Attack Family", "Label Binary", "Day"],
        cat_cols=["Protocol"],
        undersample={"Benign": 150_000, "DoS": 60_000, "DDoS": 60_000},
    ),
    "nslkdd": DatasetSpec(
        name="nslkdd",
        description="NSL-KDD KDDTrain+ / KDDTest+ (5 classes)",
        train_file="nslkdd/clean/nslkdd_train.parquet",
        test_file="nslkdd/clean/nslkdd_test.parquet",
        split="official",
        label_col="attack_cat", normal_label="normal",
        drop_cols=["attack", "difficulty", "label", "num_outbound_cmds"],
        cat_cols=["protocol_type", "service", "flag"],
        undersample={},
    ),
}

# NSL-KDD alternative test file (the harder subset)
NSL_TEST21 = "nslkdd/clean/nslkdd_test_21.parquet"
