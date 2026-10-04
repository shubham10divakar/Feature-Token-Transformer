"""RAIN-IDS v7: feature-token Transformer with a recursive, weight-tied encoder block.

Also contains the FT-Transformer baseline (same tokenizer, K untied stacked
blocks, CLS head) so both models share one training/evaluation path.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


# ------------------------------------------------------------------ tokenizer
class PeriodicEmbedding(nn.Module):
    """x_i -> [sin, cos](2*pi*c_i*x_i) with k learned frequencies -> linear to d (per feature)."""

    def __init__(self, n, d, k=16, sigma=1.0):
        super().__init__()
        self.c = nn.Parameter(torch.randn(n, k) * sigma)
        self.w = nn.Parameter(torch.empty(n, 2 * k, d))
        nn.init.kaiming_uniform_(self.w, a=math.sqrt(5))

    def forward(self, x):                                   # (B, n)
        v = 2 * math.pi * x.unsqueeze(-1) * self.c          # (B, n, k)
        v = torch.cat([torch.sin(v), torch.cos(v)], -1)     # (B, n, 2k)
        return torch.einsum("bnk,nkd->bnd", v, self.w)


class FeatureTokenizer(nn.Module):
    def __init__(self, n_num, cat_cards, d, embedding="linear", n_freq=16):
        super().__init__()
        self.embedding = embedding
        if embedding == "periodic":
            self.periodic = PeriodicEmbedding(n_num, d, n_freq)
        else:
            self.w = nn.Parameter(torch.empty(n_num, d))
            nn.init.kaiming_uniform_(self.w, a=math.sqrt(5))
        self.b = nn.Parameter(torch.zeros(n_num + len(cat_cards), d))
        self.cat = nn.ModuleList(nn.Embedding(c, d) for c in cat_cards)
        for e in self.cat:
            nn.init.kaiming_uniform_(e.weight, a=math.sqrt(5))
        self.cls = nn.Parameter(torch.randn(1, 1, d) * 0.02)

    def forward(self, x_num, x_cat):
        if self.embedding == "periodic":
            t_num = self.periodic(x_num)
        else:
            t_num = x_num.unsqueeze(-1) * self.w                          # (B, n_num, d)
        parts = [t_num]
        if len(self.cat):
            parts.append(torch.stack([e(x_cat[:, j]) for j, e in enumerate(self.cat)], 1))
        t = torch.cat(parts, 1) + self.b                                  # (B, F, d)
        return torch.cat([self.cls.expand(len(t), -1, -1), t], 1)         # (B, F+1, d)


# -------------------------------------------------------------- encoder block
class MHSA(nn.Module):
    def __init__(self, d, heads, attn_dropout):
        super().__init__()
        assert d % heads == 0
        self.h, self.p = heads, attn_dropout
        self.qkv = nn.Linear(d, 3 * d)
        self.out = nn.Linear(d, d)

    def forward(self, x, need_weights=False):
        B, T, d = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.h, d // self.h).permute(2, 0, 3, 1, 4)
        p = self.p if self.training else 0.0
        if need_weights:
            a = (q @ k.transpose(-2, -1)) / math.sqrt(d // self.h)
            w = a.softmax(-1)
            o = F.dropout(w, p, self.training) @ v
        else:
            w = None
            o = F.scaled_dot_product_attention(q, k, v, dropout_p=p)
        return self.out(o.transpose(1, 2).reshape(B, T, d)), w


class ReGLU(nn.Module):
    def __init__(self, d, hidden, dropout):
        super().__init__()
        self.lin1 = nn.Linear(d, 2 * hidden)
        self.drop = nn.Dropout(dropout)
        self.lin2 = nn.Linear(hidden, d)

    def forward(self, x):
        a, b = self.lin1(x).chunk(2, -1)
        return self.lin2(self.drop(a * F.relu(b)))


class GELUFFN(nn.Module):
    def __init__(self, d, hidden, dropout):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, hidden), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(hidden, d))

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    """Pre-LN: x + drop(MHSA(LN(x))); x + drop(FFN(LN(x)))."""

    def __init__(self, d, heads, ffn_mult=2.0, attn_dropout=0.1, ffn_dropout=0.1,
                 residual_dropout=0.0, ffn="reglu"):
        super().__init__()
        hidden = int(ffn_mult * d)
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = MHSA(d, heads, attn_dropout)
        self.ffn = (ReGLU if ffn == "reglu" else GELUFFN)(d, hidden, ffn_dropout)
        self.drop = nn.Dropout(residual_dropout)

    def forward(self, x, need_weights=False):
        a, w = self.attn(self.ln1(x), need_weights)
        x = x + self.drop(a)
        x = x + self.drop(self.ffn(self.ln2(x)))
        return x, w


# ---------------------------------------------------------- recursive encoder
class RecursiveEncoder(nn.Module):
    """One shared block applied K times; GRU-style gate + untied LayerNorm per pass."""

    def __init__(self, d=64, heads=4, K=4, gate_bias=-2.0, grad_checkpoint=False, **blk):
        super().__init__()
        self.K = K
        self.grad_checkpoint = grad_checkpoint
        self.block = TransformerBlock(d, heads, **blk)
        self.iter_emb = nn.Parameter(torch.zeros(K, 1, 1, d))
        nn.init.normal_(self.iter_emb, std=0.02)
        self.gate = nn.Linear(2 * d, d)
        nn.init.constant_(self.gate.bias, gate_bias)
        self.norms = nn.ModuleList(nn.LayerNorm(d) for _ in range(K))

    def _step(self, h, k, need_weights=False):
        cand, w = self.block(h + self.iter_emb[k], need_weights)
        z = torch.sigmoid(self.gate(torch.cat([cand, h], -1)))
        h = self.norms[k]((1 - z) * h + z * cand)
        return h, z, w

    def forward(self, h, need_weights=False):
        summaries, attns, gates = [], [], []
        for k in range(self.K):
            if self.grad_checkpoint and self.training and not need_weights:
                h, z, w = checkpoint(self._step, h, k, use_reentrant=False)
            else:
                h, z, w = self._step(h, k, need_weights)
            summaries.append(h[:, 0])
            if need_weights:
                attns.append(w[:, :, 0, 1:].mean(1))      # CLS -> feature attention (B, F), head-avg
                gates.append(z.mean((1, 2)))             # mean gate opening (B,)
        out = {"summaries": torch.stack(summaries, 1)}  # (B, K, d)
        if need_weights:
            out["cls_attn"] = torch.stack(attns, 1)      # (B, K, F)
            out["gates"] = torch.stack(gates, 1)         # (B, K)
        return out


class IterationAttention(nn.Module):
    """Learned query attends over the K CLS summaries."""

    def __init__(self, d, heads=4):
        super().__init__()
        self.q = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.mha = nn.MultiheadAttention(d, heads, batch_first=True)

    def forward(self, s):
        z, w = self.mha(self.q.expand(len(s), -1, -1), s, s, need_weights=True)
        return z.squeeze(1), w.squeeze(1)                 # (B, d), (B, K)


def _head(d, out):
    return nn.Sequential(nn.LayerNorm(d), nn.GELU(), nn.Linear(d, out))


class RAINIDS(nn.Module):
    def __init__(self, n_num, cat_cards, n_classes, d=64, K=4, heads=4, embedding="linear",
                 ffn="reglu", ffn_mult=2.0, attn_dropout=0.1, ffn_dropout=0.1,
                 residual_dropout=0.0, gate_bias=-2.0, grad_checkpoint=False):
        super().__init__()
        self.tok = FeatureTokenizer(n_num, cat_cards, d, embedding)
        self.enc = RecursiveEncoder(d, heads, K, gate_bias, grad_checkpoint, ffn_mult=ffn_mult,
                                    attn_dropout=attn_dropout, ffn_dropout=ffn_dropout,
                                    residual_dropout=residual_dropout, ffn=ffn)
        self.iter_attn = IterationAttention(d, heads)
        self.head_multi = _head(d, n_classes)
        self.head_bin = _head(d, 1)

    def forward(self, x_num, x_cat, need_weights=False):
        e = self.enc(self.tok(x_num, x_cat), need_weights)
        z, iw = self.iter_attn(e["summaries"])
        out = {"logits": self.head_multi(z), "logit_bin": self.head_bin(z).squeeze(-1), "embedding": z}
        if need_weights:
            out.update(iter_attn=iw, cls_attn=e["cls_attn"], gates=e["gates"])
        return out


class FTTransformer(nn.Module):
    """Baseline: Gorishniy et al. 2021 - K untied pre-LN blocks, CLS -> head."""

    def __init__(self, n_num, cat_cards, n_classes, d=64, K=3, heads=4, embedding="linear",
                 ffn="reglu", ffn_mult=2.0, attn_dropout=0.1, ffn_dropout=0.1,
                 residual_dropout=0.0, **_):
        super().__init__()
        self.tok = FeatureTokenizer(n_num, cat_cards, d, embedding)
        self.blocks = nn.ModuleList(
            TransformerBlock(d, heads, ffn_mult, attn_dropout, ffn_dropout, residual_dropout, ffn)
            for _ in range(K))
        self.head_multi = _head(d, n_classes)
        self.head_bin = _head(d, 1)

    def forward(self, x_num, x_cat, need_weights=False):
        h = self.tok(x_num, x_cat)
        attns = []
        for blk in self.blocks:
            h, w = blk(h, need_weights)
            if need_weights:
                attns.append(w[:, :, 0, 1:].mean(1))
        z = h[:, 0]
        out = {"logits": self.head_multi(z), "logit_bin": self.head_bin(z).squeeze(-1), "embedding": z}
        if need_weights:
            out["cls_attn"] = torch.stack(attns, 1)
        return out


class MLP(nn.Module):
    """Baseline: numeric features + small categorical embeddings -> L x (Linear, BN, ReLU, Dropout)."""

    def __init__(self, n_num, cat_cards, n_classes, hidden=256, layers=3, dropout=0.1, **_):
        super().__init__()
        self.cat = nn.ModuleList(nn.Embedding(c, min(16, (c + 1) // 2 + 1)) for c in cat_cards)
        width = n_num + sum(e.embedding_dim for e in self.cat)
        blocks = []
        for _ in range(layers):
            blocks += [nn.Linear(width, hidden), nn.BatchNorm1d(hidden), nn.ReLU(), nn.Dropout(dropout)]
            width = hidden
        self.body = nn.Sequential(*blocks)
        self.head_multi = nn.Linear(hidden, n_classes)
        self.head_bin = nn.Linear(hidden, 1)

    def forward(self, x_num, x_cat, need_weights=False):
        parts = [x_num] + [e(x_cat[:, j]) for j, e in enumerate(self.cat)]
        z = self.body(torch.cat(parts, 1))
        return {"logits": self.head_multi(z), "logit_bin": self.head_bin(z).squeeze(-1), "embedding": z}


MODEL_KEYS = ["model", "d", "K", "heads", "embedding", "ffn", "ffn_mult", "attn_dropout",
              "ffn_dropout", "residual_dropout", "gate_bias", "mlp_hidden", "mlp_layers", "mlp_dropout"]


def build_model(args, meta):
    if args.model == "mlp":
        return MLP(meta["n_num"], meta["cat_cardinalities"], len(meta["class_names"]),
                   hidden=args.mlp_hidden, layers=args.mlp_layers, dropout=args.mlp_dropout)
    cls = {"rain": RAINIDS, "ft_transformer": FTTransformer}[args.model]
    return cls(meta["n_num"], meta["cat_cardinalities"], len(meta["class_names"]),
               d=args.d, K=args.K, heads=args.heads, embedding=args.embedding, ffn=args.ffn,
               ffn_mult=args.ffn_mult, attn_dropout=args.attn_dropout, ffn_dropout=args.ffn_dropout,
               residual_dropout=args.residual_dropout, gate_bias=args.gate_bias,
               grad_checkpoint=args.grad_checkpoint)
