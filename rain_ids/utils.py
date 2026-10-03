import json
import logging
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch


def setup_logging(run_dir: Path):
    log = logging.getLogger("rain")
    log.setLevel(logging.INFO)
    log.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(message)s", "%H:%M:%S")
    for h in (logging.StreamHandler(sys.stdout), logging.FileHandler(run_dir / "train.log", encoding="utf-8")):
        h.setFormatter(fmt)
        log.addHandler(h)
    return log


def seed_everything(seed, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False


class EarlyStopping:
    def __init__(self, patience=15, min_delta=1e-4, mode="max"):
        self.patience, self.min_delta, self.mode = patience, min_delta, mode
        self.best = -np.inf if mode == "max" else np.inf
        self.best_epoch = 0
        self.bad_epochs = 0

    def step(self, value, epoch):
        better = (value > self.best + self.min_delta) if self.mode == "max" else (value < self.best - self.min_delta)
        if better:
            self.best, self.best_epoch, self.bad_epochs = value, epoch, 0
        else:
            self.bad_epochs += 1
        return better

    @property
    def should_stop(self):
        return self.patience > 0 and self.bad_epochs >= self.patience

    def state_dict(self):
        return dict(best=self.best, best_epoch=self.best_epoch, bad_epochs=self.bad_epochs)

    def load_state_dict(self, s):
        self.best, self.best_epoch, self.bad_epochs = s["best"], s["best_epoch"], s["bad_epochs"]


def rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def set_rng_state(s):
    random.setstate(s["python"])
    np.random.set_state(s["numpy"])
    torch.set_rng_state(s["torch"])
    if s.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(s["cuda"])


def save_checkpoint(path: Path, state: dict):
    """Atomic save: a crash mid-write never corrupts the previous checkpoint."""
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)


def load_checkpoint(path, device="cpu"):
    return torch.load(path, map_location=device, weights_only=False)


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
