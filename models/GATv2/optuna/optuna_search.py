"""Optuna search for the best CLASS_WEIGHTS matrix — multi-GPU edition.

Mirrors the classic GAT training pipeline (`GAT.py`) inside each Optuna trial:
DDP across all visible GPUs, torch.compile, DistributedSampler, AMP bf16,
gradient accumulation with no_sync, eval on rank 0 + broadcast of the score
and the early-stop / prune flags.

Each trial samples a 7-dim class-weight vector, trains a fresh GAT for a
reduced epoch budget, and reports the best value of the chosen validation
metric (default: balanced accuracy).

Pruning : the trial reports the metric at every `eval_every` epoch and
Optuna's MedianPruner kills runs that look worse than the running median.

Speed : the dataset, the map-level split and the training class counts are
loaded once at startup and reused across trials.

Usage examples (the script auto-relaunches under torchrun if N_GPU > 1):
    # Recherche standard, 50 essais
    python models/GAT/optuna_search.py --n-trials 50

    # Restreindre les GPUs utilisés
    CUDA_VISIBLE_DEVICES=0,1 python models/GAT/optuna_search.py --n-trials 50

    # Reprendre une étude existante
    python models/GAT/optuna_search.py --n-trials 30 --study-name gat-cw
"""

from __future__ import annotations

# ─────────────────────────────────────────────────────────────────────────────
# Auto-launch under torchrun when multiple GPUs are visible.
# (Mirror of GAT.py — must run before heavy imports so the parent exits fast.)
# ─────────────────────────────────────────────────────────────────────────────
import os
import signal
import sys

# Survive terminal close in every process. Three layers, because each one
# alone is insufficient under torchrun:
#   1. Ignore SIGHUP in the current process (covers the launcher itself).
#   2. Tell torchrun NOT to install its own SIGHUP handler — by default it
#      registers one in torch/distributed/elastic/multiprocessing/api.py that
#      raises SignalException and tears down all workers, overriding our
#      SIG_IGN above. The env var is read by torchrun at start_processes().
#   3. Detach from the controlling terminal via setsid() so SIGHUP is never
#      delivered in the first place when the SSH session closes (also helps
#      when systemd-logind has KillUserProcesses=no but the tty hangs up).
try:
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
except (AttributeError, ValueError):
    pass  # non-POSIX or non-main thread

os.environ.setdefault("TORCHELASTIC_SIGNALS_TO_HANDLE",
                      "SIGTERM,SIGINT,SIGQUIT")

try:
    os.setsid()
except (PermissionError, OSError):
    pass  # already a session leader (e.g. relaunched under torchrun)

# Reduces VRAM fragmentation between trials. Helps when GPUs are shared with
# other jobs. Must be set BEFORE the first torch import.
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
        print(f"[optuna_search] Detected {_n_gpus} CUDA devices — re-launching "
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
    print("[optuna_search] optuna n'est pas installé. "
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
from models.GATv2.dataio import build_loaders, load_dataset, map_level_split

OPTUNA_DIR = Path(__file__).resolve().parent / "optuna_runs"

_CACHE: dict = {}


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    # Optuna
    p.add_argument("--n-trials", type=int, default=50)
    p.add_argument("--study-name", type=str, default="gatv2-class-weights")
    p.add_argument("--storage", type=str, default=None,
                   help="Optuna storage URL. Default = sqlite under optuna_runs/.")
    p.add_argument("--seed", type=int, default=config.SEED)
    p.add_argument("--metric", type=str, default="bal",
                   help="Comma-separated list of metrics to optimize "
                        "(maximised together via arithmetic mean). "
                        "Available: bal, acc, mcc, f1m, kap, rare. "
                        "Examples: 'bal' | 'f1m' | 'bal,f1m'.")
    # Search space
    p.add_argument("--w-min", type=float, default=0.1)
    p.add_argument("--w-max", type=float, default=30.0)
    p.add_argument("--normalize-weights", action=argparse.BooleanOptionalAction,
                   default=True)
    # Training budget per trial (smaller than the real run on purpose)
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--patience", type=int, default=25)
    p.add_argument("--eval-every", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=max(1, config.BATCH_SIZE // 2),
                   help="Defaults to half the classic-training batch size to "
                        "leave VRAM headroom — tighten further if GPUs are "
                        "shared with other jobs.")
    p.add_argument("--grad-accum-steps", type=int, default=config.GRAD_ACCUM_STEPS)
    p.add_argument("--lr", type=float, default=config.LR)
    p.add_argument("--weight-decay", type=float, default=config.WEIGHT_DECAY)
    # Pruner
    p.add_argument("--pruner-startup-trials", type=int, default=5)
    p.add_argument("--pruner-warmup-epochs", type=int, default=20)
    return p


# ─────────────────────────────────────────────────────────────────────────────
# Dataset / args helpers
# ─────────────────────────────────────────────────────────────────────────────
def _load_once(opts) -> dict:
    if _CACHE:
        return _CACHE
    graphs, metas = load_dataset()
    split = map_level_split(metas, config.VAL_RATIO, config.SPLIT_SEED)
    counts = np.zeros(config.NUM_CLASSES, dtype=np.int64)
    for i in split["train"][1]:
        counts += np.asarray(metas[i]["class_counts"], dtype=np.int64)
    _CACHE.update(graphs=graphs, metas=metas, split=split, counts=counts)
    return _CACHE


def _make_args(opts) -> SimpleNamespace:
    return SimpleNamespace(
        hidden_channels   = config.HIDDEN_CHANNELS,
        num_layers        = config.NUM_LAYERS,
        num_heads         = config.NUM_HEADS,
        dropout           = config.DROPOUT,
        attn_dropout      = config.ATTN_DROPOUT,
        lr                = opts.lr,
        weight_decay      = opts.weight_decay,
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


_METRIC_MAP = {"acc": 1, "mcc": 2, "bal": 3, "f1m": 4, "kap": 5, "rare": 6}


def _parse_metrics(spec: str) -> list[tuple[str, int]]:
    """Parse 'bal,f1m' → [('bal', 3), ('f1m', 4)]. Raises on unknown names."""
    names = [p.strip() for p in spec.split(",") if p.strip()]
    if not names:
        raise ValueError(f"empty --metric spec: {spec!r}")
    unknown = [n for n in names if n not in _METRIC_MAP]
    if unknown:
        raise ValueError(
            f"unknown metric(s) {unknown}; "
            f"available: {sorted(_METRIC_MAP)}"
        )
    return [(n, _METRIC_MAP[n]) for n in names]


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
def _train_eval_replica(class_weights_np: np.ndarray, opts, log, env,
                        trial=None) -> float:
    """Run one full DDP training+eval on the given class weights.
    All ranks must enter together. Returns best val score (rank 0 only;
    other ranks return 0.0). Raises optuna.TrialPruned on rank 0 if the
    pruner kicks in (other ranks just break out of the loop)."""
    cache = _load_once(opts)
    graphs, split = cache["graphs"], cache["split"]
    args = _make_args(opts)

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
        # Free GPU memory between trials.
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
        weights = np.array(
            [trial.suggest_float(f"w{c}", opts.w_min, opts.w_max, log=True)
             for c in range(config.NUM_CLASSES)],
            dtype=np.float32,
        )
        if opts.normalize_weights:
            weights = (weights / weights.mean()).astype(np.float32)
            for c, w in enumerate(weights):
                trial.set_user_attr(f"w{c}_norm", float(w))

        # Tell the workers to start a new trial with these weights.
        _bcast_object(("trial", weights), env)

        log.info("─" * 60)
        log.info(f"Trial {trial.number} | weights = "
                 f"[{', '.join(f'{w:.3f}' for w in weights)}]")
        score = _train_eval_replica(weights, opts, log, env, trial=trial)
        log.info(f"Trial {trial.number} | final {opts.metric} = {score:.4f}")
        return score
    return objective


# ─────────────────────────────────────────────────────────────────────────────
# Worker loop (non-main ranks)
# ─────────────────────────────────────────────────────────────────────────────
def _worker_loop(opts, env, log) -> None:
    """Non-main ranks live here. Receive (weights | stop) from rank 0,
    join the DDP training, then loop back. Exit when rank 0 sends 'stop'."""
    while True:
        msg = _bcast_object(None, env)
        if not isinstance(msg, tuple) or msg[0] == "stop":
            return
        _, weights = msg
        try:
            _train_eval_replica(weights, opts, log, env, trial=None)
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
    base_log = get_logger("gat.optuna")
    log = rank_aware_logger(base_log, env)

    log.info(f"rank {env.rank}/{env.world_size} | distributed={env.is_distributed} "
             f"| device={env.device}")

    # Warm dataset cache on every rank (loaders are built per-trial).
    _load_once(opts)
    log.info(f"Train class counts: {_CACHE['counts'].tolist()}")

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
        log.info(f"Search     : log-uniform [{opts.w_min}, {opts.w_max}] "
                 f"per class, normalize={opts.normalize_weights}")
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
        raw = np.array([best.params[f"w{c}"] for c in range(config.NUM_CLASSES)],
                       dtype=np.float32)
        final = (raw / raw.mean()) if opts.normalize_weights else raw

        log.success("=" * 70)
        log.success(
            f"  BEST TRIAL #{best.number} — "
            f"{'+'.join(n for n, _ in metrics_preview)} = {best.value:.4f}"
        )
        log.success("=" * 70)
        log.info(f"  raw weights         : [{', '.join(f'{w:.4f}' for w in raw)}]")
        log.info(f"  normalised weights  : [{', '.join(f'{w:.4f}' for w in final)}]")
        log.info("  → paste this into config.py:")
        log.info(f"  CLASS_WEIGHTS = [{', '.join(f'{w:.3f}' for w in final)}]")

        out_json = OPTUNA_DIR / f"{opts.study_name}__best.json"
        out_json.write_text(json.dumps({
            "metric": opts.metric,
            "metric_names": [n for n, _ in metrics_preview],
            "value":  best.value,
            "trial":  best.number,
            "raw_weights":        raw.tolist(),
            "normalized_weights": final.tolist(),
            "params":             best.params,
            "n_completed":        len([t for t in study.trials
                                       if t.state == optuna.trial.TrialState.COMPLETE]),
            "n_pruned":           len([t for t in study.trials
                                       if t.state == optuna.trial.TrialState.PRUNED]),
        }, indent=2))
        log.success(f"  best dumped to {out_json}")
    else:
        _worker_loop(opts, env, log)

    teardown_distributed(env)


if __name__ == "__main__":
    main()
