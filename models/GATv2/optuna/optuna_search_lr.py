"""Optuna search for the best learning rate — multi-GPU edition.

Mirrors `optuna_search.py` but the only sampled hyperparameter is the
optimizer's learning rate (log-uniform). Class weights are inherited from
`config.CLASS_WEIGHTS` (or auto-computed when None). Weight decay can be
co-sampled with `--also-wd`.

Each trial:
  - samples LR (and optionally WD) for the run
  - trains a fresh GAT with DDP + AMP bf16 + torch.compile
  - reports the combined --metric (default 'f1m,mcc') every --eval-every
  - MedianPruner kills below-median runs early

Usage examples (auto-relaunches under torchrun if N_GPU > 1):
    # 30 trials over log-uniform LR [1e-5, 1e-2]
    python models/GAT/optuna_search_lr.py --n-trials 30

    # Also co-search weight_decay
    python models/GAT/optuna_search_lr.py --n-trials 50 --also-wd

    # Restrict GPUs / resume an existing study
    CUDA_VISIBLE_DEVICES=0,1 python models/GAT/optuna_search_lr.py --n-trials 30
    python models/GAT/optuna_search_lr.py --n-trials 30 --study-name gat-lr-v2
"""

from __future__ import annotations

# ─────────────────────────────────────────────────────────────────────────────
# Auto-launch under torchrun when multiple GPUs are visible.
# (Mirror of optuna_search.py — must run before heavy imports.)
# ─────────────────────────────────────────────────────────────────────────────
import os
import signal
import sys

try:
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
except (AttributeError, ValueError):
    pass

os.environ.setdefault("TORCHELASTIC_SIGNALS_TO_HANDLE",
                      "SIGTERM,SIGINT,SIGQUIT")

try:
    os.setsid()
except (PermissionError, OSError):
    pass

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
# Flaky NCCL P2P on the 2080 Ti rig corrupts VRAM during DDP → illegal memory
# access. Route collectives through host RAM. Must precede the torchrun re-exec.
os.environ.setdefault("NCCL_P2P_DISABLE", "1")
# Dormammu: GPU 5 is hardware-faulty (corrupts VRAM under NCCL load → illegal
# memory access). Exclude by default; override via explicit CUDA_VISIBLE_DEVICES.
# Must precede the device_count() probe and torchrun re-exec.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0,1,2,3,4,6,7")

if "WORLD_SIZE" not in os.environ:
    try:
        import torch as _torch_probe
        _n_gpus = (_torch_probe.cuda.device_count()
                   if _torch_probe.cuda.is_available() else 0)
    except Exception:
        _n_gpus = 0
    if _n_gpus > 1:
        print(f"[optuna_search_lr] Detected {_n_gpus} CUDA devices — re-launching "
              f"with torch.distributed.run (--nproc_per_node={_n_gpus})",
              flush=True)
        _cmd = [sys.executable, "-m", "torch.distributed.run",
                "--standalone", f"--nproc_per_node={_n_gpus}",
                "--signals-to-handle=SIGTERM,SIGINT,SIGQUIT",
                os.path.abspath(__file__)] + sys.argv[1:]
        os.execvp(_cmd[0], _cmd)


import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

try:
    import optuna
    from optuna.pruners import MedianPruner
    from optuna.samplers import TPESampler
except ImportError:
    print("[optuna_search_lr] optuna n'est pas installé. "
          "Installe-le avec : pip install optuna", file=sys.stderr)
    raise

# Make the repo root importable so the `models.GATv2.*` package resolves
# regardless of how this script is launched (incl. torchrun re-exec by path).
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import models.GATv2.config as config
from models.GATv2.core import (
    GATv2 as GAT,
    Trainer,
    broadcast_decision,
    get_logger,
    init_distributed,
    rank_aware_logger,
    teardown_distributed,
)
from models.GATv2.dataio import build_loaders, compute_class_weights, load_dataset, map_level_split

OPTUNA_DIR = Path(__file__).resolve().parent / "optuna_runs"

_CACHE: dict = {}


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    # Optuna
    p.add_argument("--n-trials", type=int, default=30)
    p.add_argument("--study-name", type=str, default="gatv2-lr")
    p.add_argument("--storage", type=str, default=None,
                   help="Optuna storage URL. Default = sqlite under optuna_runs/.")
    p.add_argument("--seed", type=int, default=config.SEED)
    p.add_argument("--metric", type=str, default="f1m,mcc",
                   help="Comma-separated metrics, maximised as arithmetic mean. "
                        "Available: bal, acc, mcc, f1m, kap, rare. "
                        "Examples: 'bal' | 'f1m' | 'f1m,mcc'.")
    # LR search space
    p.add_argument("--lr-min", type=float, default=1e-6)
    p.add_argument("--lr-max", type=float, default=1e-3)
    # Optionally co-search weight decay
    p.add_argument("--also-wd", action="store_true",
                   help="Also sample weight_decay log-uniformly.")
    p.add_argument("--wd-min", type=float, default=1e-6)
    p.add_argument("--wd-max", type=float, default=1e-2)
    # Training budget per trial (smaller than the real run on purpose)
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--patience", type=int, default=25)
    p.add_argument("--eval-every", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=max(1, config.BATCH_SIZE // 2),
                   help="Defaults to half the classic-training batch size to "
                        "leave VRAM headroom — tighten further if GPUs are "
                        "shared with other jobs.")
    p.add_argument("--grad-accum-steps", type=int, default=config.GRAD_ACCUM_STEPS)
    # Pruner
    p.add_argument("--pruner-startup-trials", type=int, default=5)
    p.add_argument("--pruner-warmup-epochs", type=int, default=20)
    return p


# ─────────────────────────────────────────────────────────────────────────────
# Metric parsing (mirrors optuna_search.py)
# ─────────────────────────────────────────────────────────────────────────────
_METRIC_MAP = {"acc": 1, "mcc": 2, "bal": 3, "f1m": 4, "kap": 5, "rare": 6}


def _parse_metrics(spec: str) -> list[tuple[str, int]]:
    """Parse 'f1m,mcc' → [('f1m', 4), ('mcc', 2)]. Raises on unknown names."""
    names = [p.strip() for p in spec.split(",") if p.strip()]
    if not names:
        raise ValueError(f"empty --metric spec: {spec!r}")
    unknown = [n for n in names if n not in _METRIC_MAP]
    if unknown:
        raise ValueError(
            f"unknown metric(s) {unknown}; available: {sorted(_METRIC_MAP)}"
        )
    return [(n, _METRIC_MAP[n]) for n in names]


# ─────────────────────────────────────────────────────────────────────────────
# Dataset / args helpers
# ─────────────────────────────────────────────────────────────────────────────
def _load_once(opts) -> dict:
    """Load dataset + split + class weights once. Reused across trials."""
    if _CACHE:
        return _CACHE
    graphs, metas = load_dataset()
    split = map_level_split(metas, config.VAL_RATIO, config.SPLIT_SEED)
    train_counts, class_weights_np, is_manual = compute_class_weights(
        metas, split["train"][1], config.NUM_CLASSES)
    _CACHE.update(
        graphs        = graphs,
        metas         = metas,
        split         = split,
        counts        = train_counts,
        class_weights = class_weights_np,
        weights_manual = is_manual,
    )
    return _CACHE


def _make_args(opts, lr: float, weight_decay: float) -> SimpleNamespace:
    return SimpleNamespace(
        hidden_channels   = config.HIDDEN_CHANNELS,
        num_layers        = config.NUM_LAYERS,
        num_heads         = config.NUM_HEADS,
        dropout           = config.DROPOUT,
        attn_dropout      = config.ATTN_DROPOUT,
        lr                = lr,
        weight_decay      = weight_decay,
        epochs            = opts.epochs,
        batch_size        = opts.batch_size,
        grad_clip         = config.GRAD_CLIP,
        grad_accum_steps  = opts.grad_accum_steps,
        val_ratio         = config.VAL_RATIO,
        split_seed        = config.SPLIT_SEED,
        seed              = opts.seed,
        patience          = opts.patience,
        min_delta         = config.MIN_DELTA,
        eval_every        = opts.eval_every,
        lr_factor         = config.LR_FACTOR,
        lr_patience       = config.LR_PATIENCE,
        lr_min            = config.LR_MIN,
        keep_old_ckpts    = 0,
        grad_checkpoint   = True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Cross-rank synchronization helpers
# ─────────────────────────────────────────────────────────────────────────────
def _bcast_object(obj, env):
    """Broadcast an arbitrary picklable object from rank 0. No-op when single-process."""
    if not env.is_distributed:
        return obj
    lst = [obj]
    dist.broadcast_object_list(lst, src=0)
    return lst[0]


def _bcast_prune(prune_local: bool, env) -> bool:
    """Broadcast the prune decision (rank 0 → all)."""
    if not env.is_distributed:
        return prune_local
    t = torch.tensor([1.0 if prune_local else 0.0],
                     device=env.device, dtype=torch.float64)
    dist.broadcast(t, src=0)
    return bool(t.item() > 0.5)


# ─────────────────────────────────────────────────────────────────────────────
# Per-trial DDP training + eval (called collectively by ALL ranks)
# ─────────────────────────────────────────────────────────────────────────────
def _train_eval_replica(lr: float, weight_decay: float, opts, log, env,
                        trial=None) -> float:
    """Run one full DDP training+eval at the given LR/WD. Returns best val
    score on rank 0 (other ranks return 0.0). Raises optuna.TrialPruned on
    rank 0 if the pruner kicks in."""
    cache = _load_once(opts)
    graphs, split = cache["graphs"], cache["split"]
    class_weights_np = cache["class_weights"]
    args = _make_args(opts, lr=lr, weight_decay=weight_decay)

    torch.manual_seed(opts.seed)
    np.random.seed(opts.seed)

    train_loader, val_loader, _, train_sampler = build_loaders(
        graphs, split, args.batch_size, env, args.seed)

    base_model = GAT(
        num_node_features  = config.NUM_FEATURES,
        num_drone_features = config.DRONE_FEAT_DIM,
        hidden_channels    = args.hidden_channels,
        out_channels       = config.NUM_CLASSES,
        dropout            = args.dropout,
        attn_dropout       = args.attn_dropout,
        num_layers         = args.num_layers,
        num_heads          = args.num_heads,
        grad_checkpoint    = args.grad_checkpoint,
    ).to(env.device)

    if env.is_distributed:
        ddp_model = DDP(base_model, device_ids=[env.local_rank],
                        output_device=env.local_rank)
    else:
        ddp_model = base_model

    try:
        compiled = torch.compile(ddp_model)
    except Exception as e:
        log.warning(f"torch.compile disabled ({e})")
        compiled = ddp_model

    optimizer = torch.optim.Adam(ddp_model.parameters(),
                                 lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=args.lr_factor,
        patience=args.lr_patience, min_lr=args.lr_min,
    )

    cw = torch.tensor(class_weights_np, device=env.device, dtype=torch.float32)
    trainer = Trainer(compiled, ddp_model, optimizer, cw, env, args, log,
                      show_pbar=False)
    metrics = _parse_metrics(opts.metric)
    metric_label = "+".join(n for n, _ in metrics)

    best = -1.0
    best_epoch = 0
    pruned = False
    try:
        for epoch in range(1, args.epochs + 1):
            t0 = time.time()
            tr_loss = trainer.train_step(train_loader, train_sampler, epoch)
            dt = time.time() - t0

            do_eval = (epoch % args.eval_every == 0
                       or epoch == 1 or epoch == args.epochs)
            if not do_eval:
                if env.is_main:
                    log.info(f"  ep{epoch:03d} | loss {tr_loss:.4f} | "
                             f"{dt:.1f}s | (no eval)")
                continue

            # ── Evaluate on rank 0 only ──────────────────────────────────────
            score_local = 0.0
            should_stop_local = False
            should_prune_local = False
            if env.is_main:
                eval_out = trainer.evaluate(val_loader, config.NUM_CLASSES,
                                            desc=f"trial-eval ep{epoch:03d}")
                per_metric = [float(eval_out[i]) for _, i in metrics]
                score_local = float(np.mean(per_metric))
                if score_local > best + args.min_delta:
                    best, best_epoch = score_local, epoch
                if trial is not None:
                    trial.report(score_local, step=epoch)
                    should_prune_local = trial.should_prune()
                should_stop_local = (epoch - best_epoch >= args.patience)
                breakdown = " ".join(f"{n}={v:.4f}"
                                     for (n, _), v in zip(metrics, per_metric))
                log.info(f"  ep{epoch:03d} | loss {tr_loss:.4f} | "
                         f"{metric_label}={score_local:.4f} "
                         f"[{breakdown}] | "
                         f"best={best:.4f} (ep{best_epoch}) | {dt:.1f}s")

            # ── Sync score (for scheduler), early-stop, prune flags ───────────
            score_sync, should_stop = broadcast_decision(
                score_local, should_stop_local, env)
            scheduler.step(score_sync)
            should_prune = _bcast_prune(should_prune_local, env)

            if should_prune:
                if env.is_main:
                    log.info(f"  → pruned at epoch {epoch}")
                pruned = True
                break
            if should_stop:
                if env.is_main:
                    log.info(f"  early-stop at epoch {epoch} (best ep{best_epoch})")
                break
    finally:
        del trainer, compiled, ddp_model, base_model
        del optimizer, scheduler, train_loader, val_loader
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if pruned and env.is_main:
        raise optuna.TrialPruned()
    return best if env.is_main else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Optuna objective (rank 0 only)
# ─────────────────────────────────────────────────────────────────────────────
def make_objective(opts, log, env):
    def objective(trial: "optuna.trial.Trial") -> float:
        lr = trial.suggest_float("lr", opts.lr_min, opts.lr_max, log=True)
        if opts.also_wd:
            wd = trial.suggest_float("weight_decay", opts.wd_min, opts.wd_max,
                                     log=True)
        else:
            wd = config.WEIGHT_DECAY

        # Tell the workers to start a new trial with these hyperparams.
        _bcast_object(("trial", lr, wd), env)

        log.info("─" * 60)
        if opts.also_wd:
            log.info(f"Trial {trial.number} | lr={lr:.3e} | wd={wd:.3e}")
        else:
            log.info(f"Trial {trial.number} | lr={lr:.3e}")
        score = _train_eval_replica(lr, wd, opts, log, env, trial=trial)
        log.info(f"Trial {trial.number} | final {opts.metric} = {score:.4f}")
        return score
    return objective


# ─────────────────────────────────────────────────────────────────────────────
# Worker loop (non-main ranks)
# ─────────────────────────────────────────────────────────────────────────────
def _worker_loop(opts, env, log) -> None:
    """Non-main ranks live here. Receive (lr, wd | stop) from rank 0,
    join the DDP training, then loop back. Exit when rank 0 sends 'stop'."""
    while True:
        msg = _bcast_object(None, env)
        if not isinstance(msg, tuple) or msg[0] == "stop":
            return
        _, lr, wd = msg
        try:
            _train_eval_replica(lr, wd, opts, log, env, trial=None)
        except Exception:
            # Rank 0 may raise TrialPruned (it returns to study.optimize).
            # Workers just resume waiting for the next broadcast.
            pass


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    opts = build_parser().parse_args()

    env = init_distributed()
    base_log = get_logger("gat.optuna_lr")
    log = rank_aware_logger(base_log, env)

    log.info(f"rank {env.rank}/{env.world_size} | distributed={env.is_distributed} "
             f"| device={env.device}")

    # Warm dataset cache on every rank (loaders are built per-trial).
    _load_once(opts)
    log.info(f"Train class counts : {_CACHE['counts'].tolist()}")
    log.info(
        f"Class weights      : [{', '.join(f'{w:.3f}' for w in _CACHE['class_weights'])}]"
        f" ({'manual' if _CACHE['weights_manual'] else 'auto'})"
    )

    if env.is_main:
        OPTUNA_DIR.mkdir(parents=True, exist_ok=True)
        storage = (opts.storage
                   or f"sqlite:///{OPTUNA_DIR / (opts.study_name + '.db')}")

        sampler = TPESampler(seed=opts.seed, multivariate=True, group=True)
        pruner = MedianPruner(
            n_startup_trials=opts.pruner_startup_trials,
            n_warmup_steps=opts.pruner_warmup_epochs,
        )
        study = optuna.create_study(
            study_name=opts.study_name, storage=storage,
            direction="maximize", sampler=sampler, pruner=pruner,
            load_if_exists=True,
        )
        metrics_preview = _parse_metrics(opts.metric)
        log.info(f"Study      : {opts.study_name}  ({storage})")
        log.info(
            f"Metric     : {'+'.join(n for n, _ in metrics_preview)}  "
            f"(maximize{' — mean of ' + str(len(metrics_preview)) if len(metrics_preview) > 1 else ''})"
        )
        log.info(
            f"Search     : lr log-uniform [{opts.lr_min:.1e}, {opts.lr_max:.1e}]"
            + (f" | wd log-uniform [{opts.wd_min:.1e}, {opts.wd_max:.1e}]"
               if opts.also_wd else "")
        )
        log.info(f"Per trial  : up to {opts.epochs} epochs, "
                 f"patience {opts.patience}, eval every {opts.eval_every}")
        log.info(f"Trials     : {opts.n_trials}")

        try:
            study.optimize(make_objective(opts, log, env),
                           n_trials=opts.n_trials, gc_after_trial=True)
        finally:
            # Always release the workers so the script can exit cleanly,
            # even on Ctrl-C / unexpected exception.
            _bcast_object(("stop",), env)

        # ── Report best ──────────────────────────────────────────────────────
        best = study.best_trial

        log.success("=" * 70)
        log.success(
            f"  BEST TRIAL #{best.number} — "
            f"{'+'.join(n for n, _ in metrics_preview)} = {best.value:.4f}"
        )
        log.success("=" * 70)
        log.info(f"  best lr             : {best.params['lr']:.3e}")
        if opts.also_wd:
            log.info(f"  best weight_decay   : {best.params['weight_decay']:.3e}")
        log.info("  → paste into config.py:")
        log.info(f"  LR           = {best.params['lr']:.3e}")
        if opts.also_wd:
            log.info(f"  WEIGHT_DECAY = {best.params['weight_decay']:.3e}")

        out_json = OPTUNA_DIR / f"{opts.study_name}__best.json"
        out_json.write_text(json.dumps({
            "metric":            opts.metric,
            "metric_names":      [n for n, _ in metrics_preview],
            "value":             best.value,
            "trial":             best.number,
            "best_lr":           float(best.params["lr"]),
            "best_weight_decay": (float(best.params["weight_decay"])
                                  if opts.also_wd else None),
            "params":            best.params,
            "n_completed":       len([t for t in study.trials
                                      if t.state == optuna.trial.TrialState.COMPLETE]),
            "n_pruned":          len([t for t in study.trials
                                      if t.state == optuna.trial.TrialState.PRUNED]),
        }, indent=2))
        log.success(f"  best dumped to {out_json}")
    else:
        _worker_loop(opts, env, log)

    teardown_distributed(env)


if __name__ == "__main__":
    main()
