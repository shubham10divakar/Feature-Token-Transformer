"""Train / resume / evaluate RAIN-IDS v7 (or the FT-Transformer baseline).

Examples
  python train.py --dataset unsw_official
  python train.py --dataset unsw_official --resume                 # continue from last.pt
  python train.py --dataset unsw_official --resume_from runs/<run>/checkpoints/epoch_012.pt
  python train.py --dataset nslkdd --model ft_transformer --K 3
  python train.py --dataset nslkdd --model mlp --mlp_hidden 256 --mlp_layers 3
  python train.py --dataset cicids2017 --eval_only                 # re-make report from best.pt
  python train.py                                                  # interactive dataset menu
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from rain_ids.config import DATASETS, DEFAULT_DATA_ROOT
from rain_ids.data import DATA_KEYS, prepare_data
from rain_ids.evaluate import full_report, predict
from rain_ids.metrics import quick_scores
from rain_ids.model import MODEL_KEYS, build_model
from rain_ids.plots import training_curves
from rain_ids.utils import (EarlyStopping, load_checkpoint, rng_state, save_checkpoint,
                            seed_everything, set_rng_state, setup_logging, write_json)


def get_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_argument_group("data")
    g.add_argument("--dataset", choices=list(DATASETS), help="omit for an interactive menu")
    g.add_argument("--data_root", default=DEFAULT_DATA_ROOT)
    g.add_argument("--cic_label", choices=["family", "label"], default="family",
                   help="CIC-IDS2017: 8 attack families or 15 fine-grained labels")
    g.add_argument("--nsl_test", choices=["plus", "21"], default="plus", help="NSL-KDD: KDDTest+ or KDDTest-21")
    g.add_argument("--drop_leaky", action="store_true", help="UNSW: drop sttl, dttl, ct_state_ttl")
    g.add_argument("--dedup_test", action="store_true", help="official splits: also deduplicate the test file")
    g.add_argument("--train_file_only", action="store_true",
                   help="official splits: ignore the test file; split the train file into train/val/test "
                        "(stratified, --test_size / --val_size)")
    g.add_argument("--test_size", type=float, default=0.2, help="random-split datasets and --train_file_only")
    g.add_argument("--val_size", type=float, default=0.1)
    g.add_argument("--split_seed", type=int, default=42)
    g.add_argument("--undersample", default=None,
                   help='class caps for train, e.g. "Normal=80000,Generic=30000"; "none" disables; default per dataset')
    g.add_argument("--smote_target", type=int, default=2000, help="BorderlineSMOTE classes below this up to it; 0 = off")
    g.add_argument("--no_rebalance", action="store_true")
    g.add_argument("--rare_min", type=int, default=20)
    g.add_argument("--max_train_rows", type=int, default=0, help="subsample train (debugging)")
    g.add_argument("--rebuild_cache", action="store_true")

    g = p.add_argument_group("model")
    g.add_argument("--model", choices=["rain", "ft_transformer", "mlp"], default="rain")
    g.add_argument("--d", type=int, default=64)
    g.add_argument("--K", type=int, default=4, help="recursion steps (rain) / layers (ft_transformer)")
    g.add_argument("--heads", type=int, default=4)
    g.add_argument("--embedding", choices=["linear", "periodic"], default="linear")
    g.add_argument("--ffn", choices=["reglu", "gelu"], default="reglu")
    g.add_argument("--ffn_mult", type=float, default=2.0)
    g.add_argument("--attn_dropout", type=float, default=0.1)
    g.add_argument("--ffn_dropout", type=float, default=0.1)
    g.add_argument("--residual_dropout", type=float, default=0.0)
    g.add_argument("--gate_bias", type=float, default=-2.0)
    g.add_argument("--grad_checkpoint", action="store_true", help="per-iteration checkpointing (large K)")
    g.add_argument("--mlp_hidden", type=int, default=256, help="mlp: hidden width")
    g.add_argument("--mlp_layers", type=int, default=3, help="mlp: hidden layers")
    g.add_argument("--mlp_dropout", type=float, default=0.1, help="mlp: dropout")

    g = p.add_argument_group("training")
    g.add_argument("--task", choices=["both", "multi", "binary"], default="both",
                   help="which heads get a loss; both heads are always evaluated")
    g.add_argument("--lambda_bin", type=float, default=0.5)
    g.add_argument("--epochs", type=int, default=200)
    g.add_argument("--batch_size", type=int, default=2048)
    g.add_argument("--lr", type=float, default=1e-3)
    g.add_argument("--weight_decay", type=float, default=1e-5)
    g.add_argument("--warmup_epochs", type=float, default=3)
    g.add_argument("--min_lr_ratio", type=float, default=0.01)
    g.add_argument("--label_smoothing", type=float, default=0.0)
    g.add_argument("--class_weight", choices=["none", "balanced", "sqrt"], default="none")
    g.add_argument("--grad_clip", type=float, default=1.0)
    g.add_argument("--no_amp", action="store_true")
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--deterministic", action="store_true")
    g.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    g = p.add_argument_group("early stopping")
    g.add_argument("--patience", type=int, default=15, help="0 disables early stopping")
    g.add_argument("--min_delta", type=float, default=1e-4)
    g.add_argument("--monitor", choices=["val_macro_f1", "val_loss", "val_bin_f1", "val_acc"], default=None,
                   help="default: val_macro_f1 (val_bin_f1 when --task binary)")

    g = p.add_argument_group("checkpoints / output")
    g.add_argument("--out_dir", default="runs")
    g.add_argument("--run_name", default=None, help="default: <dataset>_<model>_d<d>_K<K>_s<seed>")
    g.add_argument("--resume", action="store_true", help="resume from <run>/checkpoints/last.pt if it exists")
    g.add_argument("--resume_from", default=None, help="resume from a specific checkpoint file")
    g.add_argument("--no_save_every_epoch", action="store_true", help="only keep last.pt and best.pt")
    g.add_argument("--keep_epochs", type=int, default=0, help="keep only the N newest epoch_*.pt (0 = all)")

    g = p.add_argument_group("report")
    g.add_argument("--report", choices=["full", "basic"], default="full",
                   help="full = every paper metric + all curves/figures at the end; basic = metric tables only")
    g.add_argument("--tsne", action="store_true", help="also plot a t-SNE of the learned representation")
    g.add_argument("--eval_only", action="store_true", help="skip training, report from best.pt (or --resume_from)")
    args = p.parse_args(argv)

    if args.dataset is None:
        args.dataset = choose_dataset()
    if args.monitor is None:
        args.monitor = "val_bin_f1" if args.task == "binary" else "val_macro_f1"
    if args.run_name is None:
        extra = "_leaky-dropped" if args.drop_leaky else ""
        extra += f"_{args.cic_label}" if args.dataset == "cicids2017" else ""
        extra += "_test21" if args.dataset == "nslkdd" and args.nsl_test == "21" else ""
        extra += "_dedup-test" if args.dedup_test else ""
        extra += "_trainonly" if args.train_file_only else ""
        if args.model == "mlp":
            arch = f"h{args.mlp_hidden}_L{args.mlp_layers}"
        else:
            arch = f"d{args.d}_K{args.K}_{args.embedding}"
        args.run_name = f"{args.dataset}{extra}_{args.model}_{arch}_s{args.seed}"
    return args


def choose_dataset():
    names = list(DATASETS)
    if not sys.stdin.isatty():
        sys.exit("--dataset is required when not running interactively. Choices: " + ", ".join(names))
    print("\nAvailable datasets:")
    for i, n in enumerate(names, 1):
        print(f"  {i}. {n:15s} {DATASETS[n].description}")
    while True:
        s = input(f"Choose a dataset [1-{len(names)}]: ").strip()
        if s.isdigit() and 1 <= int(s) <= len(names):
            return names[int(s) - 1]
        if s in names:
            return s


def lr_lambda_factory(args, steps_per_epoch):
    warm = max(1, int(args.warmup_epochs * steps_per_epoch))
    total = max(warm + 1, args.epochs * steps_per_epoch)

    def f(step):
        if step < warm:
            return (step + 1) / warm
        t = min(1.0, (step - warm) / (total - warm))
        return args.min_lr_ratio + (1 - args.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * t))
    return f


def compute_loss(out, y, yb, args, class_w):
    loss = 0.0
    if args.task in ("both", "multi"):
        loss = loss + F.cross_entropy(out["logits"].float(), y, weight=class_w, label_smoothing=args.label_smoothing)
    if args.task in ("both", "binary"):
        lam = args.lambda_bin if args.task == "both" else 1.0
        loss = loss + lam * F.binary_cross_entropy_with_logits(out["logit_bin"].float(), yb)
    return loss


def main(argv=None):
    args = get_args(argv)
    run_dir = Path(args.out_dir) / args.run_name
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log = setup_logging(run_dir)

    # ------------------------------------------------ locate checkpoint to resume
    ckpt = None
    if args.resume_from:
        ckpt = load_checkpoint(args.resume_from)
        log.info(f"resuming from {args.resume_from} (epoch {ckpt['epoch']})")
    elif args.eval_only:
        best = ckpt_dir / "best.pt"
        if not best.exists():
            sys.exit(f"--eval_only: {best} does not exist")
        ckpt = load_checkpoint(best)
    elif args.resume:
        last = ckpt_dir / "last.pt"
        if last.exists():
            ckpt = load_checkpoint(last)
            log.info(f"resuming from {last} (epoch {ckpt['epoch']} completed)")
        else:
            log.info(f"--resume: no {last} yet, starting a new run")
    elif (ckpt_dir / "last.pt").exists():
        log.warning(f"{ckpt_dir / 'last.pt'} exists and will be overwritten (use --resume to continue it)")

    if ckpt is not None:   # data/model settings must match the checkpoint
        saved = ckpt["args"]
        for k in DATA_KEYS + MODEL_KEYS:
            if k in saved and getattr(args, k, None) != saved[k]:
                log.warning(f"  using checkpoint value {k}={saved[k]!r} (command line had {getattr(args, k, None)!r})")
                setattr(args, k, saved[k])

    seed_everything(args.seed, args.deterministic)
    device = torch.device(args.device)
    write_json(run_dir / "args.json", vars(args))
    log.info("args: " + json.dumps(vars(args)))

    # ------------------------------------------------------------------- data
    arrays, meta, _, cache_dir = prepare_data(args)
    names = meta["class_names"]
    log.info(f"classes: {names}")
    for s in ("train", "val", "test"):
        log.info(f"  {s:5s}: {len(arrays[f'{s}_y']):>9,}  " +
                 " ".join(f"{n}={c:,}" for n, c in zip(names, meta["counts"][s])))
    (run_dir / "data_meta.json").write_text(json.dumps({**meta, "cache_dir": str(cache_dir)}, indent=2))

    model = build_model(args, meta).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    log.info(f"model: {args.model}  params={n_params:,}")

    use_amp = device.type == "cuda" and not args.no_amp
    amp_dtype = (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16) if use_amp else None
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and amp_dtype == torch.float16)

    # ------------------------------------------------------- optimizer & sched
    decay, no_decay = [], []
    for n_, p_ in model.named_parameters():
        (decay if p_.ndim >= 2 and "tok." not in n_ and "iter_emb" not in n_ else no_decay).append(p_)
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": args.weight_decay},
                             {"params": no_decay, "weight_decay": 0.0}], lr=args.lr)
    n_train = len(arrays["train_y"])
    steps_per_epoch = math.ceil(n_train / args.batch_size)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda_factory(args, steps_per_epoch))
    es = EarlyStopping(args.patience, args.min_delta, "min" if args.monitor == "val_loss" else "max")
    history, start_epoch = [], 1

    if ckpt is not None:
        model.load_state_dict(ckpt["model"])
        if not args.eval_only:
            opt.load_state_dict(ckpt["optimizer"])
            sched.load_state_dict(ckpt["scheduler"])
            if ckpt.get("scaler"):
                scaler.load_state_dict(ckpt["scaler"])
            es.load_state_dict(ckpt["early_stopping"])
            es.patience = args.patience          # allow extending patience on resume
            history = ckpt["history"]
            start_epoch = ckpt["epoch"] + 1
            set_rng_state(ckpt["rng"])

    # ------------------------------------------------------------- training
    train_time = sum(h.get("epoch_time", 0) for h in history)
    if not args.eval_only:
        xn_tr = torch.from_numpy(arrays["train_num"]).to(device)
        xc_tr = torch.from_numpy(arrays["train_cat"]).to(device)
        y_tr = torch.from_numpy(arrays["train_y"]).to(device)
        yb_tr = (y_tr != meta["normal_idx"]).float()
        xn_va, xc_va = torch.from_numpy(arrays["val_num"]), torch.from_numpy(arrays["val_cat"])
        y_va = arrays["val_y"]
        yb_va = (y_va != meta["normal_idx"]).astype(np.float32)

        class_w = None
        if args.class_weight != "none":
            cnt = np.bincount(arrays["train_y"], minlength=len(names)).astype(float)
            w = cnt.sum() / np.clip(cnt, 1, None) / len(names)
            w = np.sqrt(w) if args.class_weight == "sqrt" else w
            class_w = torch.tensor(w / w.mean(), dtype=torch.float32, device=device)

        if start_epoch > args.epochs or es.should_stop:
            log.info(f"training already finished (epoch {start_epoch - 1}, best epoch {es.best_epoch}); "
                     "going straight to the report")
        try:
            for epoch in range(start_epoch, args.epochs + 1):
                if es.should_stop:
                    break
                t0 = time.time()
                model.train()
                perm = torch.randperm(n_train, device=device)
                tot_loss, preds = 0.0, []
                for i in range(0, n_train, args.batch_size):
                    idx = perm[i:i + args.batch_size]
                    if len(idx) < 2:
                        continue
                    with torch.autocast(device.type, dtype=amp_dtype, enabled=use_amp):
                        out = model(xn_tr[idx], xc_tr[idx])
                    loss = compute_loss(out, y_tr[idx], yb_tr[idx], args, class_w)
                    opt.zero_grad(set_to_none=True)
                    scaler.scale(loss).backward()
                    if args.grad_clip > 0:
                        scaler.unscale_(opt)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                    scaler.step(opt)
                    scaler.update()
                    sched.step()
                    tot_loss += loss.item() * len(idx)
                    preds.append(out["logits"].argmax(1).detach())
                if not math.isfinite(tot_loss):
                    log.error("non-finite training loss; stopping (last good checkpoint is intact)")
                    break
                pred_tr = torch.cat(preds).cpu().numpy()
                y_seen = arrays["train_y"][perm.cpu().numpy()][:len(pred_tr)]
                tr_q = quick_scores(y_seen, np.eye(len(names), dtype=np.float32)[pred_tr])

                pv = predict(model, xn_va, xc_va, device, amp_dtype=amp_dtype)
                p_true = np.clip(pv["prob"][np.arange(len(y_va)), y_va], 1e-12, 1)
                pb = np.clip(pv["p_bin"], 1e-7, 1 - 1e-7)
                bce = -np.mean(yb_va * np.log(pb) + (1 - yb_va) * np.log(1 - pb))
                ce = -np.mean(np.log(p_true))
                val_loss = {"both": ce + args.lambda_bin * bce, "multi": ce, "binary": bce}[args.task]
                va_q = quick_scores(y_va, pv["prob"])
                from sklearn.metrics import f1_score
                val_bin_f1 = f1_score(yb_va, (pv["p_bin"] >= 0.5).astype(int), zero_division=0)

                rec = {"epoch": epoch, "lr": opt.param_groups[0]["lr"],
                       "train_loss": tot_loss / n_train, "train_acc": tr_q["acc"], "train_macro_f1": tr_q["macro_f1"],
                       "val_loss": float(val_loss), "val_acc": va_q["acc"], "val_macro_f1": va_q["macro_f1"],
                       "val_bin_f1": float(val_bin_f1), "epoch_time": time.time() - t0}
                improved = es.step(rec[args.monitor], epoch)
                rec["best_so_far"] = es.best
                history.append(rec)
                train_time += rec["epoch_time"]
                log.info(f"ep {epoch:3d} | loss {rec['train_loss']:.4f}/{rec['val_loss']:.4f} | "
                         f"acc {rec['train_acc']:.4f}/{rec['val_acc']:.4f} | mF1 {rec['train_macro_f1']:.4f}/"
                         f"{rec['val_macro_f1']:.4f} | binF1 {rec['val_bin_f1']:.4f} | lr {rec['lr']:.2e} | "
                         f"{rec['epoch_time']:.1f}s" + ("  *best*" if improved else f"  ({es.bad_epochs}/{args.patience})"))

                state = {"epoch": epoch, "model": model.state_dict(), "optimizer": opt.state_dict(),
                         "scheduler": sched.state_dict(), "scaler": scaler.state_dict(),
                         "early_stopping": es.state_dict(), "history": history, "args": vars(args),
                         "meta": {k: meta[k] for k in ("class_names", "normal_idx", "feature_names",
                                                       "n_num", "cat_cardinalities")},
                         "rng": rng_state(), "monitor": args.monitor, "finished": es.should_stop}
                if not args.no_save_every_epoch:
                    save_checkpoint(ckpt_dir / f"epoch_{epoch:03d}.pt", state)
                    if args.keep_epochs > 0:
                        for old in sorted(ckpt_dir.glob("epoch_*.pt"))[:-args.keep_epochs]:
                            old.unlink()
                if improved:
                    save_checkpoint(ckpt_dir / "best.pt", state)
                save_checkpoint(ckpt_dir / "last.pt", state)
                pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
                if es.should_stop:
                    log.info(f"early stopping: no {args.monitor} improvement for {args.patience} epochs "
                             f"(best {es.best:.4f} at epoch {es.best_epoch})")
        except KeyboardInterrupt:
            log.warning("interrupted - last completed epoch is in last.pt; rerun with --resume to continue")
            return

        if history:
            training_curves(history, run_dir / "report" / "figures", es.best_epoch)
        best = load_checkpoint(ckpt_dir / "best.pt", device)
        model.load_state_dict(best["model"])
        log.info(f"loaded best checkpoint (epoch {best['epoch']}, {args.monitor}={es.best:.4f})")
    elif ckpt.get("history"):
        training_curves(ckpt["history"], run_dir / "report" / "figures",
                        ckpt["early_stopping"]["best_epoch"])

    # --------------------------------------------------------------- report
    hist = history or (ckpt or {}).get("history", [])
    info = {"best_epoch": int(es.best_epoch or (ckpt or {}).get("epoch", 0)), "epochs_run": len(hist),
            "total_train_time_s": train_time,
            "mean_epoch_time_s": float(np.mean([h["epoch_time"] for h in hist])) if hist else None,
            "monitor": args.monitor}
    full = args.report == "full"
    full_report(model, arrays, meta, device, run_dir / "report", "test", full=full, tsne=args.tsne,
                amp_dtype=None, train_info=info)
    full_report(model, arrays, meta, device, run_dir / "report" / "val", "val", full=False, amp_dtype=None)
    log.info(f"done. run directory: {run_dir.resolve()}")


if __name__ == "__main__":
    main()
