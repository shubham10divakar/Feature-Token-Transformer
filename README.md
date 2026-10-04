# RAIN-IDS v7: Feature-Token Transformer with a recursive encoder

This repo implements the design in `RAIN-IDS Feature-Token Transformer — Design Doc.md`. Each flow feature becomes a token. One weight-tied pre-LN Transformer block (ReGLU FFN) runs K times, with a GRU gate and an untied LayerNorm after every pass. Attention over the K CLS summaries then feeds a multi-class head and a binary head.

```
rain_ids/
  config.py      dataset registry (paths, label/categorical/port columns, undersampling caps)
  data.py        load -> drop leakage cols -> dedup -> split -> transform -> rebalance -> cache
  preprocess.py  Preprocessor (RARE/UNK vocabs, port buckets, log1p + QuantileTransformer,
                 median impute + missing indicators) and BorderlineSMOTE rebalancing
  model.py       FeatureTokenizer (linear | periodic), RecursiveEncoder, IterationAttention,
                 RAINIDS, and the FTTransformer baseline
  metrics.py     every paper metric (multi-class, binary, per-class)
  plots.py       all figures (PNG 300 dpi + PDF)
  evaluate.py    inference + end-of-run report
train.py         train / resume / evaluate
baselines.py     XGBoost on the identical processed data
ablation.py      K sweep (K = 1 vs K > 1) across seeds
data_check.py    label-noise ceiling: best accuracy / macro-F1 any model can reach on the split
benchmark.py     every model x every dataset on identical data, one summary table
```

## Setup

```powershell
pip install -r requirements.txt
```

The datasets path defaults to `D:\D\my docs\...\Intrusion detection works\datasets`. You can override it with `--data_root` or the `RAIN_DATA_ROOT` environment variable.

## Choosing a dataset

| `--dataset` | Data | Split |
|---|---|---|
| `unsw_official` | UNSW-NB15 official train/test (10 classes) | official (the swapped files are detected and fixed automatically) |
| `unsw_full` | UNSW-NB15 full dedup (2.06M rows) | stratified 80/20 (`--test_size`) |
| `cicids2017` | CIC-IDS-2017 cleaned; `--cic_label family` (8) or `label` (15) | stratified 80/20 |
| `nslkdd` | NSL-KDD; `--nsl_test plus` (KDDTest+) or `21` (KDDTest-21) | official |

If you run `python train.py` without `--dataset`, it shows an interactive menu.

Useful data options:
- `--drop_leaky`: drops the UNSW TTL leakage features (`sttl`, `dttl`, `ct_state_ttl`).
- `--dedup_test`: deduplicates the official test file as well.
- `--train_file_only`: ignores the official test file (UNSW official, NSL-KDD). The train file is split at random into train/val/test (`--test_size`, `--val_size`), and every report uses that held-out test part. The other datasets are single files and already split this way.
- `--undersample "Normal=80000,Generic=30000"` or `none`: sets the class caps.
- `--smote_target 2000` (0 turns SMOTE off) and `--no_rebalance`.

The processed data is cached under `cache/<dataset>_<hash>/`, together with `preproc.pkl` (the fitted transformers, vocabularies and column order). Any run with the same data settings reuses it. `--rebuild_cache` forces a rebuild.

## Baselines and benchmark

```powershell
python benchmark.py                                  # ceiling, XGBoost (plain + balanced), MLP, FT-Transformer,
                                                     #   RAIN K=1, RAIN K=4 on all four datasets
python benchmark.py --datasets unsw_official --models ceiling xgboost mlp
python benchmark.py --tag dedup -- --dedup_test      # a second protocol; args after -- go to every run
python benchmark.py --tag trainonly -- --train_file_only   # random split of the train file only
python benchmark.py --tag binary --models xgboost mlp rain -- --task binary   # normal vs attack only
python benchmark.py --summary_only                   # rebuild runs/bench/summary.csv
python train.py --dataset nslkdd --model mlp         # MLP alone (--mlp_hidden, --mlp_layers, --mlp_dropout)
python data_check.py --dataset unsw_official         # ceiling alone
```

## Training, checkpoints, resuming, early stopping

```powershell
python train.py --dataset unsw_official                         # defaults: d=64, K=4, heads=4, batch 2048
python train.py --dataset unsw_official --resume                # continue from runs/<run>/checkpoints/last.pt
python train.py --dataset unsw_official --resume --epochs 300   # extend a run (also works after early stopping
                                                                #   if you raise --patience)
python train.py --dataset unsw_official --resume_from runs/<run>/checkpoints/epoch_020.pt
python train.py --dataset unsw_official --eval_only             # regenerate the report from best.pt
```

- **Saved every epoch:** `checkpoints/epoch_XXX.pt`, plus `last.pt` and `best.pt` when the monitored metric improves. Each holds the model, optimizer, LR scheduler, AMP scaler, early-stopping state, history and every RNG state, so a resumed run continues exactly where it stopped. `--keep_epochs N` keeps only the newest N epoch files, and `--no_save_every_epoch` keeps only `last.pt` and `best.pt`. Writes are atomic, so a crash never corrupts the previous checkpoint. Pressing Ctrl-C also leaves `last.pt` resumable.
- **Resume:** the run directory is `runs/<run_name>`. By default the name is built from the dataset, model, d, K, embedding and seed, so rerunning the same command with `--resume` finds the run. Data and model settings always come from the checkpoint, and the training settings (epochs, patience) come from the command line.
- **Early stopping:** `--patience 15` (0 turns it off), `--min_delta`, and `--monitor val_macro_f1 | val_loss | val_bin_f1 | val_acc`.
- **Other training options:** `--task both|multi|binary` (with `--lambda_bin`; with `binary` the report's multi-class section and per-class tables cover the two classes normal / Attack, and XGBoost trains a binary model), `--embedding periodic`, `--class_weight`, `--label_smoothing`, `--grad_checkpoint` (for K ≥ 6), `--no_amp`, and `--deterministic`.
- **LR schedule:** warmup followed by cosine decay over `--epochs`. If you change `--epochs` on resume, the rest of the schedule is stretched to the new length.

## End-of-run report (`--report full`, the default)

`runs/<run>/report/` contains the following:
- `metrics.json` has everything listed below.
  - **Multi-class:** accuracy, balanced accuracy, macro/weighted/micro P/R/F1, MCC, Cohen's κ, ROC-AUC (OvR macro/weighted, OvO), PR-AUC, log-loss, Brier score, top-2 accuracy, ECE, minimum per-class recall, classes below 0.5 recall, and macro FPR.
  - **Binary**, from both the binary head and the multi-class head: accuracy, precision, detection rate, F1, F2, specificity, false alarm rate, miss rate, NPV, MCC, κ, ROC-AUC, PR-AUC, TPR at FPR of 0.1%, 1% and 5%, the best-F1 threshold, Brier score, and ECE.
  - **Efficiency:** parameter count, model size, throughput, single-sample latency, and training time.
- `per_class_metrics.csv` has P/R/F1, specificity, FPR, FNR, AUC and AP for each class. `confusion_matrix.csv`, `predictions.npz`, and the LaTeX tables `summary_table.tex` and `per_class_table.tex` are also written.
- `figures/` holds every figure as PNG and PDF:
  - training curves (loss, accuracy, macro-F1, binary F1, LR, epoch time)
  - confusion matrices (counts and normalized; multi-class and binary)
  - per-class ROC and PR curves, plus micro/macro averages
  - per-class P/R/F1 bars
  - binary ROC, log-FPR ROC, PR, metrics-vs-threshold and score-distribution plots
  - calibration reliability diagrams
  - interpretability: iteration-attention and gate heatmaps by class, top features by CLS attention, per-class and per-iteration feature attention
  - with `--tsne`: a t-SNE of the learned representation
- `report/val/` holds the same metrics on the validation split.

`--report basic` writes only the metric tables, which is faster for sweeps.

## Baselines and ablation

```powershell
python train.py   --dataset unsw_official --model ft_transformer --K 3      # FT-Transformer
python baselines.py --dataset unsw_official                                # XGBoost (GPU), same cache
python ablation.py --dataset unsw_official --Ks 1 2 4 6 8 --seeds 0 1 2    # recursion ablation
python train.py   --dataset unsw_official --embedding periodic             # periodic-embedding ablation
```
