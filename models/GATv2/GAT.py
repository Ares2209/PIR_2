import argparse
import os
import sys

# Ensure the repo root (three levels up: models/GATv2/GAT.py -> repo root) is
# importable so that `import models.GATv2....` works whether launched as a
# module, as a script, or re-launched via torch.distributed.run.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# GPU physiques défectueux sur Dormammu à ne jamais utiliser : le GPU 5
# (2080 Ti HS) corrompt les calculs et fait planter les runs multi-GPU
# (device-side assert « index out of bounds » sur le rank correspondant).
# Réintégration testée le 2026-07-10 → replante immédiatement sur rank 5
# (remove_self_loops / edge_index[:, mask]). Confirmé HS, on le ré-exclut.
_BAD_GPU_IDS = {5}

if "WORLD_SIZE" not in os.environ:
    try:
        import torch as _torch_probe
        _n_gpus_total = (_torch_probe.cuda.device_count()
                         if _torch_probe.cuda.is_available() else 0)
    except Exception:
        _n_gpus_total = 0

    # Si l'utilisateur n'a pas déjà fixé CUDA_VISIBLE_DEVICES, on exclut
    # automatiquement les GPU défectueux avant de compter les devices.
    if "CUDA_VISIBLE_DEVICES" not in os.environ and _n_gpus_total > 0:
        _visible = [str(i) for i in range(_n_gpus_total)
                    if i not in _BAD_GPU_IDS]
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(_visible)
        _excluded = sorted(_BAD_GPU_IDS & set(range(_n_gpus_total)))
        if _excluded:
            print(f"[GAT] Excluding faulty GPU(s) {_excluded} — "
                  f"CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}",
                  flush=True)

    # Nombre de GPU réellement visibles après exclusion.
    _visible_env = os.environ.get("CUDA_VISIBLE_DEVICES")
    if _visible_env is not None:
        _n_gpus_detected = len([d for d in _visible_env.split(",") if d != ""])
    else:
        _n_gpus_detected = _n_gpus_total

    if _n_gpus_detected > 1:
        print(f"[GAT] Detected {_n_gpus_detected} usable CUDA devices — "
              f"re-launching with torch.distributed.run "
              f"(--nproc_per_node={_n_gpus_detected})", flush=True)
        # GPU 5 HS => NCCL P2P instable sur ce nœud, on le désactive.
        os.environ.setdefault("NCCL_P2P_DISABLE", "1")
        # 2080 Ti ~11 GiB : marge mémoire très fine. On active les segments
        # extensibles pour récupérer la mémoire réservée-mais-non-allouée
        # (fragmentation) et éviter les OOM au backward sur les gros graphes.
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        _cmd = [sys.executable, "-m", "torch.distributed.run",
                "--standalone", f"--nproc_per_node={_n_gpus_detected}",
                os.path.abspath(__file__)] + sys.argv[1:]
        os.environ["PYTHONPATH"] = (
            _REPO_ROOT + os.pathsep + os.environ.get("PYTHONPATH", "")
        ).rstrip(os.pathsep)
        os.execvp(_cmd[0], _cmd)

import random
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

import models.GATv2.config as config
from models.GATv2.config import ATTN_DROPOUT, DRONE_FEAT_DIM, NUM_CLASSES, NUM_FEATURES, SHARD_DIR
from models.GATv2.core import (
    GATv2 as GAT,
    Trainer,
    broadcast_decision,
    get_logger,
    init_distributed,
    per_class_report,
    rank_aware_logger,
    run_dirname,
    save_checkpoint,
    teardown_distributed,
    unwrap_module,
)
from models.GATv2.dataio import (
    build_loaders,
    compute_class_weights,
    finalize_run_artifacts,
    load_dataset,
    make_run_dir,
    map_level_split,
    save_history_plots,
    save_training_config,
)


CKPT_ROOT = Path(__file__).resolve().parent / "checkpoints"
PLOTS_DIR = Path(__file__).resolve().parent / "plots"

_STOP_LABEL = {
    "completed":         "Training completed (all epochs)",
    "max_epochs":        "Training completed (all epochs)",
    "early_stopping":    "Training stopped early (patience exhausted)",
    "manual_interrupt":  "Training stopped manually (KeyboardInterrupt)",
}

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--hidden_channels", type=int, default=config.HIDDEN_CHANNELS)
    p.add_argument("--num_layers", type=int, default=config.NUM_LAYERS)
    p.add_argument("--num_heads", type=int, default=config.NUM_HEADS)
    p.add_argument("--dropout", type=float, default=config.DROPOUT)
    p.add_argument("--lr", type=float, default=config.LR)
    p.add_argument("--weight_decay", type=float, default=config.WEIGHT_DECAY)
    p.add_argument("--epochs", type=int, default=config.EPOCHS)
    p.add_argument("--batch_size", type=int, default=config.BATCH_SIZE)
    p.add_argument("--eval_batch_size", type=int, default=config.EVAL_BATCH_SIZE)
    p.add_argument("--grad_clip", type=float, default=config.GRAD_CLIP)
    p.add_argument("--grad_accum_steps", type=int, default=config.GRAD_ACCUM_STEPS)
    p.add_argument("--val_ratio", type=float, default=config.VAL_RATIO)
    p.add_argument("--split_seed", type=int, default=config.SPLIT_SEED)
    p.add_argument("--seed", type=int, default=config.SEED)
    p.add_argument("--patience", type=int, default=config.PATIENCE)
    p.add_argument("--min_delta", type=float, default=config.MIN_DELTA)
    p.add_argument("--eval_every", type=int, default=config.EVAL_EVERY)
    p.add_argument("--lr_factor", type=float, default=config.LR_FACTOR)
    p.add_argument("--lr_patience", type=int, default=config.LR_PATIENCE)
    p.add_argument("--lr_min", type=float, default=config.LR_MIN)
    p.add_argument("--keep_old_ckpts", type=int, default=config.KEEP_OLD_CHECKPOINTS)
    p.add_argument(
        "--grad_checkpoint", action=argparse.BooleanOptionalAction, default=True,
        help="Activation checkpointing per GATConv layer. Default ON: cuts "
             "activation memory ~num_layers× (lets you push hidden_channels) at "
             "~25-30%% compute cost. Disable with --no-grad_checkpoint if you "
             "have plenty of VRAM headroom and want max throughput.",
    )
    p.add_argument(
        "--track_metric", type=str, default="f1m,mcc",
        help="Comma-separated list of validation metrics that drive best-ckpt "
             "selection, early-stopping and the LR scheduler (arithmetic mean). "
             "Available: bal, acc, mcc, f1m, kap, rare. Default: 'f1m,mcc'.",
    )
    return p


# eval_out tuple from Trainer.evaluate:
#   (loss, acc, mcc, bal, f1m, kap, rare, preds, targets)
_EVAL_METRIC_INDICES = {"acc": 1, "mcc": 2, "bal": 3, "f1m": 4, "kap": 5, "rare": 6}


def _parse_track_metric(spec: str) -> list[tuple[str, int]]:
    """Parse 'f1m,mcc' → [('f1m', 4), ('mcc', 2)]."""
    names = [p.strip() for p in spec.split(",") if p.strip()]
    if not names:
        raise ValueError(f"empty --track_metric spec: {spec!r}")
    unknown = [n for n in names if n not in _EVAL_METRIC_INDICES]
    if unknown:
        raise ValueError(
            f"unknown metric(s) {unknown}; "
            f"available: {sorted(_EVAL_METRIC_INDICES)}"
        )
    return [(n, _EVAL_METRIC_INDICES[n]) for n in names]

def _seed_everything(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _fmt_time(seconds: float) -> str:
    seconds = int(seconds)
    h, r = divmod(seconds, 3600)
    m, s = divmod(r, 60)
    return f"{h:d}h{m:02d}m{s:02d}s" if h else f"{m:d}m{s:02d}s"


def _build_model(args, env, log):
    base_model = GAT(
        num_node_features  = NUM_FEATURES,
        num_drone_features = DRONE_FEAT_DIM,
        hidden_channels    = args.hidden_channels,
        out_channels       = NUM_CLASSES,
        dropout            = args.dropout,
        attn_dropout       = ATTN_DROPOUT,
        num_layers         = args.num_layers,
        num_heads          = args.num_heads,
        grad_checkpoint    = args.grad_checkpoint,
    ).to(env.device)
    log.info(f"Activation checkpointing: {args.grad_checkpoint}")

    if env.is_distributed:
        ddp_model = DDP(base_model, device_ids=[env.local_rank],
                        output_device=env.local_rank)
    else:
        ddp_model = base_model

    try:
        model = torch.compile(ddp_model)
        log.info("torch.compile: enabled")
    except Exception as e:
        log.warning(f"torch.compile: disabled ({e})")
        model = ddp_model
    return model, ddp_model


def _build_state_for_save(model, args, epoch, val_score, val_acc, val_metrics,
                          track_metric_spec, class_weights_np, train_counts,
                          test_maps, val_maps):
    return {
        "model_state":         unwrap_module(model).state_dict(),
        "args":                vars(args),
        "epoch":               epoch,
        "val_score":           val_score,
        "val_acc":             val_acc,
        "val_metrics":         val_metrics,
        "track_metric":        track_metric_spec,
        # kept for backwards-compat with existing tooling
        "val_bal_acc":         val_metrics.get("bal", 0.0),
        "num_node_features":   NUM_FEATURES,
        "num_drone_features":  DRONE_FEAT_DIM,
        "num_classes":         NUM_CLASSES,
        "class_weights":       class_weights_np.tolist(),
        "train_class_counts":  train_counts.tolist(),
        "test_maps":           sorted(test_maps),
        "val_maps":            sorted(val_maps),
    }


def main() -> None:
    args = build_parser().parse_args()

    env = init_distributed()
    is_tty = sys.stdout.isatty()
    show_pbar = env.is_main and is_tty
    log = rank_aware_logger(get_logger("gat.train"), env)

    _seed_everything(args.seed)
    log.info(f"Seed: {args.seed} | rank {env.rank}/{env.world_size} | "
             f"distributed={env.is_distributed}")
    log.info(f"Using device: {env.device}")

    # ── Dataset / split / class weights ───────────────────────────────────────
    log.info(f"Loading sharded dataset from {SHARD_DIR}")
    graphs, metas = load_dataset()
    log.info(f"Loaded {len(graphs)} graphs | node_feat={NUM_FEATURES} "
             f"drone_feat={DRONE_FEAT_DIM} | classes={NUM_CLASSES}")

    split = map_level_split(metas, args.val_ratio, args.split_seed)
    train_maps, train_idx = split["train"]
    val_maps,   val_idx   = split["val"]
    test_maps,  test_idx  = split["test"]
    log.info(f"Map-level split | train={len(train_maps)} maps ({len(train_idx)} graphs) "
             f"| val={len(val_maps)} ({len(val_idx)}) | test={len(test_maps)} ({len(test_idx)})")

    train_counts, class_weights_np, is_manual = compute_class_weights(
        metas, train_idx, NUM_CLASSES)
    log.info(f"Train class counts : {train_counts.tolist()}")
    log.info(f"Class weights      : [{', '.join(f'{w:.3f}' for w in class_weights_np)}]"
             f" ({'manual' if is_manual else 'auto'})")

    train_loader, val_loader, test_loader, train_sampler = build_loaders(
        graphs, split, args.batch_size, env, args.seed,
        eval_batch_size=args.eval_batch_size)

    ckpt_dir = CKPT_ROOT / run_dirname(args, class_weights_np.tolist())
    old_dir  = ckpt_dir / "old"

    # ── Run directory (timestamped) — holds config dump + final plots ─────────
    if env.is_main:
        run_dir, _ = make_run_dir(PLOTS_DIR)
        save_training_config(
            run_dir, args, class_weights_np, train_counts,
            train_maps, val_maps, test_maps,
            args.track_metric, env, log,
        )
        log.info(f"Run dir            : {run_dir}")
    else:
        run_dir = None

    # ── Model / optimizer / trainer ───────────────────────────────────────────
    model, ddp_model = _build_model(args, env, log)
    optimizer = torch.optim.Adam(ddp_model.parameters(), lr=args.lr,
                                 weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=args.lr_factor,
        patience=args.lr_patience, min_lr=args.lr_min,
    )

    class_weights = torch.tensor(class_weights_np, device=env.device)
    trainer = Trainer(model, ddp_model, optimizer, class_weights,
                      env, args, log, show_pbar=show_pbar)
    log.info(f"AMP bfloat16: {trainer.use_bf16}")
    log.info(f"Batch size: {args.batch_size} × grad_accum {args.grad_accum_steps} "
             f"× world_size {env.world_size} = effective "
             f"{args.batch_size * args.grad_accum_steps * env.world_size}")

    # ── Training loop ─────────────────────────────────────────────────────────
    tracked_metrics = _parse_track_metric(args.track_metric)
    track_label     = "+".join(n for n, _ in tracked_metrics)
    log.info(f"Tracking metric: {track_label} "
             f"(mean of {len(tracked_metrics)} — drives best-ckpt, "
             f"early-stop, LR scheduler)")

    best_val_score, best_val_acc = -1.0, 0.0
    best_val_preds = best_val_targets = None
    best_epoch = 0
    times: list[float] = []
    last_epoch = 0
    stop_reason = "completed"

    history = {
        "epoch": [], "train_loss": [], "val_loss": [],
        "val_acc": [], "val_bal": [], "val_score": [],
        "val_f1m": [], "val_mcc": [],
        "val_prec": [], "val_rec": [], "val_f1": [], "lr": [],
    }
    labels_list = list(range(NUM_CLASSES))

    def per_cls(arr):
        return "  ".join(f"c{c}:{arr[c]:.3f}" for c in labels_list)

    epoch_bar = tqdm(range(1, args.epochs + 1), desc="epochs", unit="ep",
                     ncols=0, ascii=True, disable=not show_pbar)

    try:
        for epoch in epoch_bar:
            last_epoch = epoch
            t0 = time.time()
            tr_loss = trainer.train_step(train_loader, train_sampler, epoch)

            do_eval = (epoch % args.eval_every == 0
                       or epoch == 1 or epoch == args.epochs)
            if not do_eval:
                times.append(time.time() - t0)
                epoch_bar.set_postfix(loss=f"{tr_loss:.4f}", eval="skip")
                log.debug(f"Ep {epoch:03d} | loss {tr_loss:.4f} (eval skipped)")
                continue

            # Eval / saving happen on rank 0 only — others wait at the broadcast.
            val_score_local = 0.0
            should_stop_local = False
            if env.is_main:
                # Return the training step's cached blocks to the allocator so
                # the large per-edge attention tensor built during eval has room
                # (training memory stays resident on rank 0 throughout eval).
                if env.device.type == "cuda":
                    torch.cuda.empty_cache()
                (val_loss, val_acc, val_mcc, val_bal, val_f1m, val_kap, val_rare,
                 val_preds, val_targets) = trainer.evaluate(
                    val_loader, NUM_CLASSES, desc=f"val ep{epoch:03d}")
                eval_tuple = (val_loss, val_acc, val_mcc, val_bal, val_f1m,
                              val_kap, val_rare)
                per_metric = [float(eval_tuple[i]) for _, i in tracked_metrics]
                val_score = float(np.mean(per_metric))
                val_metrics_dict = {
                    "acc": val_acc, "mcc": val_mcc, "bal": val_bal,
                    "f1m": val_f1m, "kap": val_kap, "rare": val_rare,
                }
                current_lr = optimizer.param_groups[0]["lr"]

                if val_score > best_val_score + args.min_delta:
                    prev_best_score = best_val_score
                    best_val_score, best_val_acc = val_score, val_acc
                    best_val_preds, best_val_targets = val_preds, val_targets
                    best_epoch = epoch
                    save_checkpoint(
                        ckpt_dir, old_dir, val_score, val_acc, epoch,
                        _build_state_for_save(model, args, epoch, val_score,
                                              val_acc, val_metrics_dict,
                                              args.track_metric,
                                              class_weights_np, train_counts,
                                              test_maps, val_maps),
                        prev_best_score, args.min_delta, args.keep_old_ckpts, log)

                epochs_no_improve = epoch - best_epoch
                v_p, v_r, v_f, _ = precision_recall_fscore_support(
                    val_targets, val_preds, labels=labels_list, zero_division=0)
                history["epoch"].append(epoch)
                history["train_loss"].append(tr_loss)
                history["val_loss"].append(val_loss)
                history["val_acc"].append(val_acc)
                history["val_bal"].append(val_bal)
                history["val_score"].append(val_score)
                history["val_f1m"].append(val_f1m)
                history["val_mcc"].append(val_mcc)
                history["val_prec"].append(v_p)
                history["val_rec"].append(v_r)
                history["val_f1"].append(v_f)
                history["lr"].append(current_lr)
                times.append(time.time() - t0)

                breakdown = " ".join(f"{n}={v:.3f}"
                                     for (n, _), v in zip(tracked_metrics, per_metric))
                epoch_bar.set_postfix(loss=f"{tr_loss:.4f}",
                                      score=f"{val_score:.3f}",
                                      best=f"{best_val_score:.3f}",
                                      lr=f"{current_lr:.1e}",
                                      pat=f"{epochs_no_improve}/{args.patience}")
                log.info(f"── Epoch {epoch:03d} ──  (loss {tr_loss:.4f} | lr {current_lr:.2e} | "
                         f"pat {epochs_no_improve}/{args.patience})")
                log.info(f"  VAL    | acc {val_acc:.3f} | bal {val_bal:.3f} | "
                         f"mcc {val_mcc:.3f} | f1m {val_f1m:.3f} | "
                         f"kap {val_kap:.3f} | rareRec {val_rare:.3f}")
                log.info(f"  TRACK  | {track_label}={val_score:.4f} [{breakdown}] | "
                         f"best={best_val_score:.4f} (ep{best_epoch})")
                log.info(f"    ├─ precision  {per_cls(v_p)}")
                log.info(f"    ├─ recall     {per_cls(v_r)}")
                log.info(f"    └─ f1         {per_cls(v_f)}")

                should_stop_local = epochs_no_improve >= args.patience
                val_score_local = val_score
            else:
                times.append(time.time() - t0)

            # Sync the LR-scheduler signal and the early-stopping flag across
            # ranks so optimizer state stays identical and all ranks break together.
            val_score_sync, should_stop = broadcast_decision(
                val_score_local, should_stop_local, env)
            scheduler.step(val_score_sync)

            if should_stop:
                stop_reason = "early_stopping"
                log.warning(f"Early stopping at epoch {epoch} "
                            f"(no improvement for {args.patience} epochs, "
                            f"best epoch {best_epoch})")
                break
        else:
            stop_reason = "max_epochs"
    except KeyboardInterrupt:
        stop_reason = "manual_interrupt"
        log.warning("Interrupted by user.")
    finally:
        epoch_bar.close()

    # ── Summary ───────────────────────────────────────────────────────────────
    total_time  = sum(times)
    median_time = float(torch.tensor(times).median()) if times else 0.0
    mean_time   = float(torch.tensor(times).mean())   if times else 0.0

    log.success("=" * 70)
    log.success(f"  TRAINING SUMMARY — {_STOP_LABEL[stop_reason]}")
    log.success("=" * 70)
    log.info(f"  Epochs run         : {last_epoch}/{args.epochs}")
    log.info(f"  Best epoch         : {best_epoch}")
    log.info(f"  Tracked metric     : {track_label}")
    log.info(f"  Best val score/Acc : {best_val_score:.4f} / {best_val_acc:.4f}")
    log.info(f"  Total time         : {_fmt_time(total_time)}")
    log.info(f"  Mean/median epoch  : {mean_time:.2f}s / {median_time:.2f}s")
    log.info(f"  Checkpoint dir     : {ckpt_dir}")
    log.success("=" * 70)

    # ── Final test on rank 0 ──────────────────────────────────────────────────
    if env.is_main:
        per_class_report("VAL  @ best epoch", best_val_preds, best_val_targets,
                         NUM_CLASSES, log)

        ckpt_files = sorted(ckpt_dir.glob("gat_*.pt"))
        if ckpt_files:
            best_ckpt = torch.load(ckpt_files[-1], weights_only=False)
            unwrap_module(model).load_state_dict(best_ckpt["model_state"])
            log.info(f"Loaded best checkpoint for final test: {ckpt_files[-1].name}")
        else:
            log.warning("No checkpoint found — evaluating test with last model weights")
        (ts_loss, ts_acc, ts_mcc, ts_bal, ts_f1m, ts_kap, ts_rare,
         ts_preds, ts_targets) = trainer.evaluate(
            test_loader, NUM_CLASSES, desc="final test")
        log.success(f"  FINAL TEST | acc {ts_acc:.3f} | mcc {ts_mcc:.3f} | "
                    f"bal {ts_bal:.3f} | f1m {ts_f1m:.3f} | "
                    f"kap {ts_kap:.3f} | rareRec {ts_rare:.3f}")
        per_class_report("TEST (final, best ckpt)", ts_preds, ts_targets,
                         NUM_CLASSES, log)

        save_history_plots(history, best_epoch, run_dir, log)
        finalize_run_artifacts(run_dir, ckpt_dir, log)

    teardown_distributed(env)


if __name__ == "__main__":
    main()
