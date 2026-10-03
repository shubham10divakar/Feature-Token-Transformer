# RAIN-IDS Feature-Token Transformer — Design Doc

Oct 3, 2026 · @Subham Divakar

## Overview and goals

RAIN-IDS v7 replaces the current per-feature embedding → GRU recursive-attention stack with a **feature-token Transformer** whose encoder block is applied recursively (weight-tied), with a GRU gate and LayerNorm on every iteration. Each flow feature becomes a token. Self-attention learns which feature interactions matter, and the recursion refines that view over K passes.

The current v5/v6 recursive attention is bilinear attention driven by a GRU controller over feature embeddings. v7 keeps that recursive, iteratively normalized identity, but puts a full multi-head self-attention encoder inside the loop. This makes the model directly comparable to FT-Transformer, the strongest tabular Transformer baseline, while keeping RAIN's novelty claim.

**Success criteria (UNSW-NB15, official split):**

| Task | Accuracy | Macro-F1 | Also required |
| --- | --- | --- | --- |
| Multi-class (10 classes) | ≥ 95% | ≥ 93% (97% = strong) | Per-class recall reported; no class below 0.5 recall |
| Binary | ≥ 98% | ≥ 97% | — |

Beyond the numbers, v7 must:

- beat a tuned FT-Transformer and XGBoost on macro-F1;
- show, through ablation, that the recursion itself adds performance (K = 1 vs K > 1);
- reproduce on CIC-IDS2017 and NSL-KDD with the same code path.

## Data pipeline

Reuse the v5 cleaning steps unchanged. The new work is the split, the per-column-type transforms, and the token layout.

1. **Load and clean (v5, as is):** 49 named columns, Latin-1 encoding with the BOM stripped, `attack_cat` NaN → Normal, Backdoors → Backdoor, ports cast to numeric, and the swapped-file check.
2. **Drop:** `srcip`, `dstip`, `stime`, `ltime` (identity and time leakage), plus `label` and `attack_cat` from X.
3. **Deduplicate before splitting.** UNSW-NB15 has many exact duplicate rows. Drop duplicates on X + y, and log the count per class. Also check for overlap between train and test after the split.
4. **Split:** fit everything on the train split only. Hold out 10% of train, stratified, as validation. The test split is touched once, at the end.
5. **Column typing:**
   - Categorical: `proto`, `service`, `state`. Map values seen fewer than 20 times in train to `<RARE>`; unseen values at test time map to `<UNK>`.
   - Ports (`sport`, `dsport`): add a bucket feature (well-known 0–1023, registered 1024–49151, dynamic 49152+, NaN) as a categorical token, and also keep a log1p numeric token.
   - Binary flags (`is_ftp_login`, `is_sm_ips_ports`): numeric 0/1.
   - All other numeric columns: `log1p` on heavy-tailed columns (bytes, packets, load, duration, jitter), then a QuantileTransformer (normal output, 1,000 quantiles) fit on train.
6. **Missing values:** median-impute numerics from train, and add a missing-indicator token only for columns more than 1% missing.
7. **Rebalancing (train only, after the split):** undersample Normal to about 80k and Generic to about 30k, and run BorderlineSMOTE for classes under 2,000 samples. SMOTE runs on the *transformed* numeric space; categorical columns of synthetic rows take the nearest neighbour's value.
8. **Persist:** save the fitted transformers, vocabularies and column order as one `preproc.pkl`, so validation, test and the other datasets go through the identical object.

Output per sample: `x_num` (float tensor, n\_num), `x_cat` (int tensor, n\_cat), `y_multi`, `y_bin`. With about 39 numeric and 5 categorical inputs, the model sees about 44 feature tokens plus CLS.

## Tokenization

Every feature becomes one d-dimensional token. A learned CLS token is prepended, so the sequence is `[CLS, t_1 … t_F]`, with shape (B, F+1, d) and d = 64 by default.

**Numeric feature i**, with its own weight and bias vectors (FT-Transformer style):

```latex
t_i = x_i \cdot w_i + b_i, \quad w_i, b_i \in \mathbb{R}^d
```

**Option to ablate:** periodic embeddings (Gorishniy et al. 2022). These map x\_i to sin/cos features at k = 16 learned frequencies, then through a linear layer to d. They often add 0.5–1.5 F1 points on heavy-tailed features.

**Categorical feature j:** `t_j = E_j[x_j] + b_j`, using one `nn.Embedding` per column (vocab size from train, plus `<RARE>`, `<UNK>`).

**Feature identity:** no positional encoding is needed. Each token's own w\_i / E\_j already identifies the feature. Order is fixed by the column list in `preproc.pkl`.

```python
class FeatureTokenizer(nn.Module):
    def __init__(self, n_num, cat_cards, d):
        super().__init__()
        self.w = nn.Parameter(torch.empty(n_num, d)); nn.init.kaiming_uniform_(self.w)
        self.b = nn.Parameter(torch.zeros(n_num + len(cat_cards), d))
        self.cat = nn.ModuleList(nn.Embedding(c, d) for c in cat_cards)
        self.cls = nn.Parameter(torch.randn(1, 1, d) * 0.02)

    def forward(self, x_num, x_cat):
        t_num = x_num.unsqueeze(-1) * self.w                     # (B, n_num, d)
        t_cat = torch.stack([e(x_cat[:, j]) for j, e in enumerate(self.cat)], 1)
        t = torch.cat([t_num, t_cat], 1) + self.b                # (B, F, d)
        return torch.cat([self.cls.expand(len(t), -1, -1), t], 1)  # (B, F+1, d)
```

The v6 input BatchNorm is dropped: the quantile transform already standardises inputs, and BatchNorm interacts badly with the rebalanced batches.

## Model architecture

The model runs one shared Transformer block K times over the feature tokens. A GRU-style gate decides how much of each pass to keep, and a separate LayerNorm follows every pass. Attention over the K CLS summaries then feeds two classification heads.

&#91;embedded content: feature-token Transformer · tokenizer, recursive block, iteration attention, two heads\]

Read it top to bottom. Each flow's \~44 features become tokens, and the CLS token joins them. The shaded block runs K times with the same weights; each pass, every token attends to every other one, and the gate decides how much of the update to keep. The K CLS summaries are then weighed into one vector that feeds both heads.

**Recursive block (weights shared across iterations):**

1. Add an iteration embedding: `H' = H_{k-1} + e_k`. This tells the shared block which pass it is on.
2. Run a pre-LN Transformer layer:
   - multi-head self-attention, 4 heads, d = 64
   - ReGLU feed-forward with hidden size 2d
   - attention dropout 0.1, FFN dropout 0.1, residual dropout 0.0

   This gives the candidate `H~`.
3. Apply the per-token GRU gate and the per-iteration LayerNorm (*iterative normalization*; LN\_k is **not** shared):

```latex
z_k = \sigma(W_z [\tilde{H}_k ; H_{k-1}] + b_z), \qquad H_k = \mathrm{LN}_k\big((1 - z_k) \odot H_{k-1} + z_k \odot \tilde{H}_k\big)
```

4. Keep the summary `s_k = H_k[CLS]`.

**Iteration attention:** a learned query attends over `[s_1 … s_K]` with 4 heads, giving `z`. This keeps the v6 iteration-level attention, and its weights show which depth each class relies on.

**Heads:** `LN → GELU → Linear(d, 10)` for multi-class, and `LN → GELU → Linear(d, 1)` for binary.

**Defaults:** d = 64, K = 4, heads = 4. That comes to roughly 60k parameters, so try d = 128 if the model underfits. v6 used 7 iterations; sweep K ∈ {1, 2, 4, 6, 8}.

```python
class RecursiveEncoder(nn.Module):
    def __init__(self, d=64, heads=4, K=4, p=0.1):
        super().__init__()
        self.K = K
        self.block = nn.TransformerEncoderLayer(d, heads, 2 * d, p, activation='gelu',
                                                batch_first=True, norm_first=True)  # swap FFN for ReGLU later
        self.iter_emb = nn.Parameter(torch.zeros(K, 1, 1, d))
        self.gate = nn.Linear(2 * d, d)
        self.norms = nn.ModuleList(nn.LayerNorm(d) for _ in range(K))      # untied per iteration

    def forward(self, h):
        summaries = []
        for k in range(self.K):
            cand = self.block(h + self.iter_emb[k])
            z = torch.sigmoid(self.gate(torch.cat([cand, h], -1)))
            h = self.norms[k]((1 - z) * h + z * cand)
            summaries.append(h[:, 0])
        return torch.stack(summaries, 1)                               # (B, K, d)
```

The gate bias b\_z is initialised to −2, so early training stays close to the identity and does not blow up. Turn on gradient checkpointing per iteration if K ≥ 6 at batch 2,048 runs out of memory on the T4.

## Training setup

Carry the v6 loss and schedule over unchanged, so any change in results comes from the architecture. Only the batch size and the regularization are retuned for the Transformer.

**Loss:**

```latex
\mathcal{L} = \mathcal{L}_{\text{focal}}(\hat{y}_{multi}, y_{multi}) + \lambda \, \mathcal{L}_{\text{BCE}}(\hat{y}_{bin}, y_{bin}), \quad \lambda = 0.5
```

- Focal loss with γ = 1.5, label smoothing 0.1, and α = sqrt-scaled inverse class frequency (computed after rebalancing).
- Sweep λ ∈ {0.3, 0.5, 1.0}. If binary F1 is already high, a lower λ protects the multi-class head.

| Setting | Value | Notes |
| --- | --- | --- |
| Optimizer | AdamW, lr 1e-4, weight decay 1e-4 | No decay on embeddings, biases, LayerNorm |
| Schedule | 5-epoch linear warmup → cosine, restart at epoch 100 | As v6 |
| Batch size | 1,024 (try 2,048) | Use AMP (fp16) on the T4 |
| Epochs | 150 max, early stop on val macro-F1, patience 20 | Save best by macro-F1, not by loss |
| Dropout | attention 0.1, FFN 0.1 | Raise to 0.2 if train/val gap > 5 F1 points |
| Gradient clipping | max norm 1.0 | Recursion can spike gradients |
| Seeds | 5 seeds per final config | Report mean ± std |

**Sampling:** use a class-balanced sampler *or* focal α, never both at full strength, because stacking them over-corrects toward rare classes. Start with α only.

**Logging:** print `VERSION = "v7.x"` first, then log per-epoch val macro-F1, per-class recall, and the mean gate value z per iteration. A mean z near 0 means a pass is doing nothing.

## Evaluation and baselines

The headline metric is macro-F1 on the untouched test split. Report accuracy too, but never on its own.

**Report for every run:**

- accuracy, macro-F1, weighted-F1
- per-class precision and recall
- a confusion matrix
- binary accuracy and F1, plus FAR (false alarm rate) and DR (detection rate)
- for the final model: params, training time per epoch, and inference latency per 1k flows on the T4

**Baselines** (same preprocessing, same split, tuned with equal effort):

| Model | Why it's in the table |
| --- | --- |
| XGBoost / LightGBM | Strongest classical baseline on tabular data |
| Random Forest | Standard reference in IDS papers |
| MLP | Simplest deep baseline |
| 1D-CNN, CNN-BiLSTM-Attention | Most common published IDS deep models |
| FT-Transformer (3 layers, no recursion) | The direct non-recursive counterpart |
| RAIN-IDS v6 | Shows what v7 adds over your own previous model |

**Ablations** (each removes one idea):

1. K = 1 vs 2, 4, 6, 8, with params held equal: does recursion help?
2. Shared weights vs 4 untied layers (a standard 4-layer Transformer).
3. GRU gate removed (plain residual).
4. Shared LayerNorm vs per-iteration LN\_k.
5. Iteration attention vs using only s\_K.
6. Linear vs periodic numeric embeddings.
7. With vs without Normal/Generic undersampling.

**Interpretability (for the paper):**

- average CLS-to-feature attention per class;
- iteration-attention weights per class (which pass depth each attack relies on);
- a SHAP comparison against XGBoost for the top features.

## Implementation plan

Build in six steps, each with a check that must pass before moving on. If you stay in one Kaggle notebook, keep these as clearly separated cells.

```
rain_ids_v7/
  config.py          # VERSION, dataset paths, all hyperparameters
  data/
    unsw.py          # v5 cleaning + split + dedup
    preprocess.py    # Preprocessor: fit/transform, vocabs, quantile, save/load preproc.pkl
    rebalance.py     # undersample + BorderlineSMOTE (train only)
    dataset.py       # TensorDataset of (x_num, x_cat, y_multi, y_bin)
  model/
    tokenizer.py     # FeatureTokenizer
    encoder.py       # RecursiveEncoder (shared block, gate, LN_k)
    heads.py         # IterationAttention + two heads
    rain.py          # RAINTransformer = tokenizer → encoder → heads
  train.py           # loop, AMP, focal loss, early stop on macro-F1
  evaluate.py        # metrics, confusion matrix, attention dumps
  baselines/         # xgb.py, ft_transformer.py, cnn_bilstm.py
```

- [ ] **1. Data:** build `preprocess.py` and `dataset.py`. Check: the shapes print as expected, the train/test duplicate overlap is 0, and the class counts are logged before and after rebalancing.
- [ ] **2. Model, K = 1:** build the tokenizer, encoder and heads. Check: the model overfits 512 samples to about 100% train accuracy within 200 steps.
- [ ] **3. First full run, K = 1:** this is effectively an FT-Transformer. Check: macro-F1 is within about 2 points of XGBoost, which confirms the pipeline is sound.
- [ ] **4. Recursion on, K = 4:** check that the mean gate z is above 0.1 on every pass and that val macro-F1 is at least as good as with K = 1.
- [ ] **5. Sweeps:** K, d, λ and dropout, with 1 seed each, then 5 seeds for the best two configs.
- [ ] **6. Baselines and ablations:** fill in the evaluation tables, then repeat the best config on CIC-IDS2017 and NSL-KDD.

## Risks and open questions

| Risk | Signal | Mitigation |
| --- | --- | --- |
| Transformer doesn't beat XGBoost | K = 1 run trails XGB by more than 2 macro-F1 | Try periodic embeddings and d = 128 first. If it still trails, frame the contribution as interpretability plus matched accuracy |
| Recursion adds nothing | Mean gate z ≈ 0 on later passes; K = 4 ≈ K = 1 | Lower the gate bias init to −1, or add a small auxiliary loss on each s\_k |
| Rare classes still collapse (Worms, Analysis, Backdoor) | Per-class recall < 0.5 | Tune per-class thresholds on val; try class-balanced sampling instead of focal α |
| Analysis / Backdoor / DoS are near-identical in feature space | Confusion stays between these three in every model | Known UNSW-NB15 limitation: report it, and show it holds for the baselines too |
| Feature leakage via `sttl`, `ct_state_ttl` | One feature dominates attention and SHAP | Add an ablation without TTL features; reviewers increasingly ask for it |
| Overfitting from SMOTE samples | Big train/val gap on SMOTE-boosted classes | Cap synthetic samples, or drop SMOTE and rely on focal loss alone |

**Open questions:**

- [ ] Which split is canonical: the official 175k/82k partition, or a stratified split of the full 2.54M records? The official one is better for comparing with published work.
- [ ] Treat ports as numeric plus bucket (as planned), or as hashed categorical tokens?
- [ ] Keep v6's 7 iterations as the default K for continuity, or default to 4 and let the sweep decide?
- [ ] Should the binary label come from the multi-class head (Normal vs the rest) instead of a separate head? This would be a simpler model and an easy ablation.
