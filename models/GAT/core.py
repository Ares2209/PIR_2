"""Training plumbing for the GAT pipeline.

Consolidates what used to live in five separate modules:
  - model.py        → GAT nn.Module
  - trainer.py      → Trainer class + per_class_report
  - checkpoint.py   → unwrap_module, run_dirname, save_checkpoint, find_best_checkpoint
  - distributed.py  → DistEnv + init/teardown + rank-aware logger
  - log_utils.py    → colored tqdm-aware logger
"""

from __future__ import annotations

import contextlib
import logging
import os
import re
import shutil
import sys
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from sklearn.metrics import (
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
    recall_score,
)
from torch.nn.parallel import DistributedDataParallel as DDP
from torch_geometric.nn import GATConv
from tqdm import tqdm


# ═════════════════════════════════════════════════════════════════════════════
# Logging
# ═════════════════════════════════════════════════════════════════════════════
class _TqdmHandler(logging.StreamHandler):
    """Route log records through tqdm.write so they don't shred active bars."""

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg, file=sys.stderr)
            self.flush()
        except Exception:
            self.handleError(record)


class _ColorFormatter(logging.Formatter):
    GREY = "\x1b[38;5;245m"
    GREEN = "\x1b[32;1m"
    YELLOW = "\x1b[33;1m"
    RED = "\x1b[31;1m"
    BOLD_RED = "\x1b[41;1m"
    CYAN = "\x1b[36;1m"
    BLUE = "\x1b[34;1m"
    RESET = "\x1b[0m"

    LEVEL_COLORS = {
        logging.DEBUG: GREY,
        logging.INFO: CYAN,
        logging.WARNING: YELLOW,
        logging.ERROR: RED,
        logging.CRITICAL: BOLD_RED,
    }

    def __init__(self, use_color: bool = True):
        super().__init__()
        self.use_color = use_color

    def format(self, record):
        ts = self.formatTime(record, "%H:%M:%S")
        level = record.levelname
        msg = record.getMessage()
        if not self.use_color:
            return f"[{ts}] {level:<8} {msg}"
        color = self.LEVEL_COLORS.get(record.levelno, "")
        return (f"{self.GREY}[{ts}]{self.RESET} "
                f"{color}{level:<8}{self.RESET} {msg}")


def get_logger(name: str = "gat", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(level)
    handler = _TqdmHandler()
    handler.setFormatter(_ColorFormatter(use_color=sys.stderr.isatty()))
    logger.addHandler(handler)
    logger.propagate = False
    return logger


SUCCESS = 25
logging.addLevelName(SUCCESS, "SUCCESS")
_ColorFormatter.LEVEL_COLORS[SUCCESS] = _ColorFormatter.GREEN


def _success(self, msg, *args, **kwargs):
    if self.isEnabledFor(SUCCESS):
        self._log(SUCCESS, msg, args, **kwargs)


logging.Logger.success = _success


# ═════════════════════════════════════════════════════════════════════════════
# Distributed (DDP)
# ═════════════════════════════════════════════════════════════════════════════
class DistEnv:
    """Frozen container with rank/world-size + device after init."""

    def __init__(self, rank: int, world_size: int, local_rank: int,
                 device: torch.device, is_distributed: bool):
        self.rank = rank
        self.world_size = world_size
        self.local_rank = local_rank
        self.device = device
        self.is_distributed = is_distributed

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def init_distributed() -> DistEnv:
    env_world = int(os.environ.get("WORLD_SIZE", "1"))
    is_distributed = env_world > 1 and torch.cuda.is_available()
    if is_distributed:
        local_rank = int(os.environ["LOCAL_RANK"])
        rank       = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        local_rank = rank = 0
        world_size = 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return DistEnv(rank, world_size, local_rank, device, is_distributed)


def make_local_env() -> DistEnv:
    """Single-process DistEnv for inference / analysis scripts. Picks CUDA if
    available; never calls `init_process_group`."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return DistEnv(rank=0, world_size=1, local_rank=0,
                   device=device, is_distributed=False)


def teardown_distributed(env: DistEnv) -> None:
    if env.is_distributed and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def broadcast_decision(value: float, flag: bool, env: DistEnv) -> tuple[float, bool]:
    """Broadcast (scalar, bool) from rank 0 to every rank — used to keep the
    LR scheduler step and the early-stopping flag consistent across replicas."""
    if not env.is_distributed:
        return value, flag
    t = torch.tensor([float(value), 1.0 if flag else 0.0],
                     device=env.device, dtype=torch.float64)
    dist.broadcast(t, src=0)
    return float(t[0].item()), bool(t[1].item() > 0.5)


class _NoLog:
    """Silent stand-in for the logger on non-main ranks (no-op for any method)."""

    def __getattr__(self, name):
        return lambda *a, **k: None


def rank_aware_logger(real_logger, env: DistEnv):
    return real_logger if env.is_main else _NoLog()


# ═════════════════════════════════════════════════════════════════════════════
# Checkpoint I/O
# ═════════════════════════════════════════════════════════════════════════════
def unwrap_module(m):
    """Strip torch.compile and DDP wrappers to return the raw module."""
    for _ in range(4):
        if hasattr(m, "_orig_mod"):
            m = m._orig_mod
        elif isinstance(m, DDP):
            m = m.module
        else:
            break
    return m


def run_dirname(args, class_weights) -> str:
    w_str = "-".join(f"{w:.2f}" for w in class_weights)
    return (f"lr{args.lr:.0e}_wd{args.weight_decay:.0e}_bs{args.batch_size}"
            f"__cw{w_str}")


def save_checkpoint(ckpt_dir: Path, old_dir: Path, val_score: float, val_acc: float,
                    epoch: int, state: dict, prev_score: float, min_delta: float,
                    keep_old: int, log) -> bool:
    """Move previous bests to `old/`, save the new best, prune old/ to `keep_old`.
    `val_score` is the combined tracking metric (see `--track_metric`).
    Returns True if a new checkpoint was written."""
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    old_dir.mkdir(parents=True, exist_ok=True)
    if val_score <= prev_score + min_delta:
        return False
    for p in ckpt_dir.glob("gat_*.pt"):
        shutil.move(str(p), str(old_dir / p.name))
    name = f"gat_epoch{epoch:03d}_score{val_score:.4f}_acc{val_acc:.4f}.pt"
    torch.save(state, ckpt_dir / name)
    if keep_old >= 0:
        olds = sorted(old_dir.glob("gat_*.pt"), key=lambda p: p.stat().st_mtime)
        to_delete = olds[:-keep_old] if keep_old else olds
        for p in to_delete:
            p.unlink(missing_ok=True)
    log.success(f"saved new best → {name} (prev score={prev_score:.4f})")
    return True


def find_best_checkpoint(ckpt_root: Path) -> Path | None:
    """Best `gat_*.pt` under `ckpt_root`. Prefers the highest `score<float>` in
    the filename (current naming), then `bal<float>` (older), then `mcc<float>`
    (legacy), then most recent by mtime. Skips `old/`."""
    candidates = [p for p in ckpt_root.rglob("gat_*.pt") if "old" not in p.parts]
    if not candidates:
        return None
    for tag in ("score", "bal", "mcc"):
        tagged = []
        for p in candidates:
            m = re.search(rf"{tag}([-+]?\d*\.\d+)", p.name)
            if m:
                tagged.append((float(m.group(1)), p))
        if tagged:
            return max(tagged, key=lambda t: t[0])[1]
    return max(candidates, key=lambda p: p.stat().st_mtime)


# ═════════════════════════════════════════════════════════════════════════════
# Model
# ═════════════════════════════════════════════════════════════════════════════
class GAT(torch.nn.Module):
    """Multi-layer GATConv with additive drone-signature fusion. Each layer
    averages `num_heads` attention heads (concat=False) to keep
    `hidden_channels` constant, with residual connections + LayerNorm.

    `grad_checkpoint=True` wraps each GATConv block in activation
    checkpointing, trading ~25-30% compute for ~num_layers× lower activation
    memory. Only active during training (`self.training=True`); inference is
    unaffected.
    """

    def __init__(self, num_node_features, num_drone_features, hidden_channels,
                 out_channels, dropout, attn_dropout, num_layers, num_heads,
                 grad_checkpoint: bool = True):
        super().__init__()
        self.node_proj  = torch.nn.Linear(num_node_features,  hidden_channels)
        self.drone_proj = torch.nn.Linear(num_drone_features, hidden_channels)
        self.convs = torch.nn.ModuleList([
            GATConv(hidden_channels, hidden_channels,
                    heads=num_heads, concat=False, dropout=attn_dropout)
            for _ in range(num_layers)
        ])
        self.norms = torch.nn.ModuleList(
            [torch.nn.LayerNorm(hidden_channels) for _ in range(num_layers)]
        )
        self.output_proj = torch.nn.Linear(hidden_channels, out_channels)
        self.dropout = dropout
        self.grad_checkpoint = grad_checkpoint

    def _layer_block(self, conv, norm, h, edge_index):
        return F.dropout(F.relu(norm(conv(h, edge_index))),
                         p=self.dropout, training=self.training)

    def forward(self, x, edge_index, drone_feat, batch):
        h = self.node_proj(x) + self.drone_proj(drone_feat)[batch]
        use_ckpt = self.grad_checkpoint and self.training
        for conv, norm in zip(self.convs, self.norms):
            if use_ckpt:
                # use_reentrant=False is required to play nicely with DDP
                # (and with torch.compile in recent PyTorch versions).
                h = h + torch.utils.checkpoint.checkpoint(
                    self._layer_block, conv, norm, h, edge_index,
                    use_reentrant=False,
                )
            else:
                h = h + self._layer_block(conv, norm, h, edge_index)
        return self.output_proj(h)


# ═════════════════════════════════════════════════════════════════════════════
# Trainer + per-class report
# ═════════════════════════════════════════════════════════════════════════════
class Trainer:
    def __init__(self, model, ddp_model, optimizer, class_weights, env,
                 args, logger, *, show_pbar: bool):
        self.model = model
        self.ddp_model = ddp_model
        self.optimizer = optimizer
        self.class_weights = class_weights
        self.env = env
        self.args = args
        self.log = logger
        self.show_pbar = show_pbar
        self.use_bf16 = (env.device.type == "cuda"
                         and torch.cuda.is_bf16_supported())
        self._amp_ctx = (
            (lambda: torch.autocast(device_type="cuda", dtype=torch.bfloat16))
            if self.use_bf16 else contextlib.nullcontext
        )

    def loss_fn(self, logits, target):
        return F.cross_entropy(logits, target,
                               weight=self.class_weights, ignore_index=-1)

    def train_step(self, loader, sampler, epoch: int) -> float:
        self.model.train()
        if self.env.is_distributed and sampler is not None:
            sampler.set_epoch(epoch)
        total_loss = total_nodes = 0
        accum = max(1, self.args.grad_accum_steps)

        pbar = tqdm(loader, desc=f"train ep{epoch:03d}", leave=False, unit="batch",
                    ncols=0, ascii=True, disable=not self.show_pbar)
        self.optimizer.zero_grad(set_to_none=True)
        n_batches = len(loader)
        for step, batch in enumerate(pbar):
            batch = batch.to(self.env.device, non_blocking=True)
            is_step = (step + 1) % accum == 0 or (step + 1) == n_batches
            # Skip DDP gradient sync on accumulation steps to save NCCL bandwidth.
            sync_ctx = (self.ddp_model.no_sync()
                        if (self.env.is_distributed and not is_step)
                        else contextlib.nullcontext())
            with sync_ctx:
                with self._amp_ctx():
                    out  = self.model(batch.x, batch.edge_index,
                                      batch.drone_feat, batch.batch)
                    loss = self.loss_fn(out, batch.y)
                (loss / accum).backward()
            total_loss  += float(loss.detach()) * batch.num_nodes
            total_nodes += batch.num_nodes
            if is_step:
                if self.args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.ddp_model.parameters(), self.args.grad_clip)
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
            pbar.set_postfix(loss=f"{total_loss / max(total_nodes, 1):.4f}")
        return total_loss / max(total_nodes, 1)

    @torch.no_grad()
    def evaluate(self, loader, num_classes: int, desc: str = "eval"):
        self.model.eval()
        eval_module = unwrap_module(self.model)
        total_loss = total_nodes = 0
        preds_all, targets_all = [], []
        for batch in tqdm(loader, desc=desc, leave=False, unit="batch",
                          ncols=0, ascii=True, disable=not self.show_pbar):
            batch = batch.to(self.env.device, non_blocking=True)
            with self._amp_ctx():
                out  = eval_module(batch.x, batch.edge_index,
                                   batch.drone_feat, batch.batch)
                loss = self.loss_fn(out, batch.y)
            total_loss  += float(loss.detach()) * batch.num_nodes
            total_nodes += batch.num_nodes
            preds_all.append(out.argmax(dim=-1).cpu())
            targets_all.append(batch.y.cpu())
        preds   = torch.cat(preds_all).numpy()
        targets = torch.cat(targets_all).numpy()
        valid   = targets >= 0
        preds, targets = preds[valid], targets[valid]
        acc      = float((preds == targets).mean())
        mcc      = float(matthews_corrcoef(targets, preds))
        bal_acc  = float(balanced_accuracy_score(targets, preds))
        macro_f1 = float(f1_score(targets, preds, average="macro", zero_division=0))
        kappa    = float(cohen_kappa_score(targets, preds))
        rare_labels = [c for c in (3, 4) if c < num_classes]
        rare_rec = float(recall_score(
            targets, preds, labels=rare_labels, average="macro",
            zero_division=0)) if rare_labels else 0.0
        return (total_loss / total_nodes, acc, mcc, bal_acc, macro_f1, kappa,
                rare_rec, preds, targets)


def per_class_report(name: str, preds, targets, num_classes: int, log) -> None:
    if preds is None or len(preds) == 0:
        log.warning(f"[{name}] no predictions recorded.")
        return
    labels = list(range(num_classes))
    class_names = [f"class {c}" for c in labels]
    prec, rec, f1, sup = precision_recall_fscore_support(
        targets, preds, labels=labels, zero_division=0
    )
    acc = float((preds == targets).mean())
    log.info(f"[{name}] per-class metrics (accuracy={acc:.4f})")
    log.info(f"  {'class':<10}{'precision':>11}{'recall':>11}{'f1':>11}{'support':>11}")
    for c, p, r, f, s in zip(class_names, prec, rec, f1, sup):
        log.info(f"  {c:<10}{p:>11.4f}{r:>11.4f}{f:>11.4f}{int(s):>11d}")
    macro    = precision_recall_fscore_support(
        targets, preds, labels=labels, average="macro",    zero_division=0)
    weighted = precision_recall_fscore_support(
        targets, preds, labels=labels, average="weighted", zero_division=0)
    log.info(f"  {'macro avg':<10}{macro[0]:>11.4f}{macro[1]:>11.4f}{macro[2]:>11.4f}"
             f"{int(sup.sum()):>11d}")
    log.info(f"  {'weighted':<10}{weighted[0]:>11.4f}{weighted[1]:>11.4f}{weighted[2]:>11.4f}"
             f"{int(sup.sum()):>11d}")
    cm = confusion_matrix(targets, preds, labels=labels)
    log.info("  confusion matrix (rows=true, cols=pred):")
    log.info("       " + "".join(f"{c:>8}" for c in labels))
    for i, row in enumerate(cm):
        log.info(f"  {i:>4} " + "".join(f"{v:>8d}" for v in row))
