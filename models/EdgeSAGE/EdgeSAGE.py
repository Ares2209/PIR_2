"""Node-level 7-class EdgeSAGE trained on the NoiseMap graphs built by
`dataset/data/generated/build_graphs.py`. Each graph = one noise map,
nodes = mesh faces, features = face geometry + drone acoustic signature
(LP_REF spectrum + slope + META), labels = color class (0:violet, 1:blue,
2:yellow, 3:orange, 4:red, 5:dark_red, 6:occluded).

EdgeSAGEConv = SAGE-style aggregation with edge-aware messages:
  message(j→i) = MLP([h_j, h_j - h_i])   (edge-conditioned, like EdgeConv)
  aggregate    = mean over neighbours    (GraphSAGE mean aggregator)
  update       = Linear([h_i, aggr])     (SAGE self/neighbour concat)
"""
import argparse
import contextlib
import os
import random
import shutil
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
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
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MessagePassing
from tqdm import tqdm

import config
from log_utils import get_logger

log = get_logger("edgesage.train")

ROOT = Path(__file__).resolve().parents[2]
DATA_FILE = ROOT / "dataset" / "data" / "generated" / "processed" / "graphs.pt"
CKPT_ROOT = Path(__file__).resolve().parent / "checkpoints"
PLOTS_DIR = Path(__file__).resolve().parent / "plots"

parser = argparse.ArgumentParser()
parser.add_argument("--hidden_channels", type=int, default=config.HIDDEN_CHANNELS)
parser.add_argument("--num_layers", type=int, default=config.NUM_LAYERS)
parser.add_argument("--dropout", type=float, default=config.DROPOUT)
parser.add_argument("--lr", type=float, default=config.LR)
parser.add_argument("--weight_decay", type=float, default=config.WEIGHT_DECAY)
parser.add_argument("--epochs", type=int, default=config.EPOCHS)
parser.add_argument("--batch_size", type=int, default=config.BATCH_SIZE)
parser.add_argument("--grad_clip", type=float, default=config.GRAD_CLIP)
parser.add_argument("--grad_accum_steps", type=int, default=config.GRAD_ACCUM_STEPS,
                    help="Accumulate gradients over N mini-batches before "
                         "optimizer.step() — effective batch = batch_size * N")
parser.add_argument("--val_ratio", type=float, default=config.VAL_RATIO)
parser.add_argument("--test_ratio", type=float, default=config.TEST_RATIO)
parser.add_argument("--split_seed", type=int, default=config.SPLIT_SEED)
parser.add_argument("--seed", type=int, default=config.SEED,
                    help="Seed for weight init, dropout, and DataLoader shuffling")
parser.add_argument("--patience", type=int, default=config.PATIENCE,
                    help="Early stopping: stop after N epochs without val MCC improvement")
parser.add_argument("--min_delta", type=float, default=config.MIN_DELTA,
                    help="Minimum val MCC gain to count as improvement")
parser.add_argument("--eval_every", type=int, default=config.EVAL_EVERY,
                    help="Run val/test every N epochs (epoch 1 and last always evaluated)")
parser.add_argument("--lr_factor", type=float, default=config.LR_FACTOR)
parser.add_argument("--lr_patience", type=int, default=config.LR_PATIENCE)
parser.add_argument("--lr_min", type=float, default=config.LR_MIN)
parser.add_argument("--keep_old_ckpts", type=int, default=config.KEEP_OLD_CHECKPOINTS)
args = parser.parse_args()


def seed_everything(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


seed_everything(args.seed)
log.info(f"Seed: {args.seed}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log.info(f"Using device: {device}")

# ─────────────────────────────────────────────────────────────────────────────
# Load dataset  (features are already normalised by gen_graphs.py — no
# second standardisation here to avoid double-normalisation of the same data).
# ─────────────────────────────────────────────────────────────────────────────
blob = torch.load(DATA_FILE, weights_only=False)
graphs = blob["graphs"]
num_classes = blob["num_classes"]
num_node_features = blob["num_node_features"]
num_drone_features = blob["num_drone_features"]
metas = blob["meta"]
log.info(f"Loaded {len(graphs)} graphs | node_feat={num_node_features} "
         f"drone_feat={num_drone_features} | classes={num_classes}")

# ─────────────────────────────────────────────────────────────────────────────
# Group split BY MAP NAME — same city must never appear in two splits, even
# under different drones, otherwise the model can memorise geometry.
# ─────────────────────────────────────────────────────────────────────────────
map_names = [m["map"] for m in metas]
unique_maps = sorted(set(map_names))
map_rng = torch.Generator().manual_seed(args.split_seed)
map_perm = torch.randperm(len(unique_maps), generator=map_rng).tolist()

n_test_maps = max(1, int(round(len(unique_maps) * args.test_ratio)))
n_val_maps = max(1, int(round(len(unique_maps) * args.val_ratio)))
test_maps = {unique_maps[i] for i in map_perm[:n_test_maps]}
val_maps = {unique_maps[i] for i in map_perm[n_test_maps:n_test_maps + n_val_maps]}
train_maps = {unique_maps[i] for i in map_perm[n_test_maps + n_val_maps:]}

test_idx = [i for i, m in enumerate(map_names) if m in test_maps]
val_idx = [i for i, m in enumerate(map_names) if m in val_maps]
train_idx = [i for i, m in enumerate(map_names) if m in train_maps]

log.info(f"Map-level split | train={len(train_maps)} maps ({len(train_idx)} graphs) "
         f"| val={len(val_maps)} ({len(val_idx)}) | test={len(test_maps)} ({len(test_idx)})")

# ─────────────────────────────────────────────────────────────────────────────
# Class weights for weighted cross-entropy, computed from the TRAIN split
# class distribution (inverse-sqrt frequency, mean-normalised so the weights
# average to 1.0 — keeps the loss scale stable regardless of batch composition).
# ─────────────────────────────────────────────────────────────────────────────
train_counts = np.zeros(num_classes, dtype=np.int64)
for i in train_idx:
    train_counts += np.asarray(metas[i]["class_counts"], dtype=np.int64)
_safe_counts = np.maximum(train_counts, 1)
_raw_weights = 1.0 / np.sqrt(_safe_counts.astype(np.float64))
class_weights_np = (_raw_weights / _raw_weights.mean()).astype(np.float32)
log.info(f"Train class counts : {train_counts.tolist()}")
log.info(f"Class weights      : [{', '.join(f'{w:.3f}' for w in class_weights_np)}]")


def _run_dirname(args, class_weights):
    w_str = "-".join(f"{w:.2f}" for w in class_weights)
    return (f"lr{args.lr:.0e}_wd{args.weight_decay:.0e}_bs{args.batch_size}"
            f"__cw{w_str}")


CKPT_DIR = CKPT_ROOT / _run_dirname(args, class_weights_np.tolist())
OLD_DIR = CKPT_DIR / "old"

test_set = [graphs[i] for i in test_idx]
val_set = [graphs[i] for i in val_idx]
train_set = [graphs[i] for i in train_idx]

loader_generator = torch.Generator().manual_seed(args.seed)
_pin = device.type == "cuda"
train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                          generator=loader_generator, pin_memory=_pin)
val_loader = DataLoader(val_set, batch_size=args.batch_size, pin_memory=_pin)
test_loader = DataLoader(test_set, batch_size=args.batch_size, pin_memory=_pin)


class EdgeSAGEConv(MessagePassing):
    """SAGE-style convolution with edge-aware messages.

    Combines EdgeConv (edge-conditioned message via [h_j, h_j - h_i]) with
    GraphSAGE's self/neighbour concat update — keeps the edge geometry signal
    while retaining the node's own representation across layers.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__(aggr="mean")
        self.edge_mlp = torch.nn.Sequential(
            torch.nn.Linear(2 * in_channels, out_channels),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(out_channels, out_channels),
        )
        self.lin_self = torch.nn.Linear(in_channels, out_channels)
        self.lin_out = torch.nn.Linear(2 * out_channels, out_channels)

    def forward(self, x, edge_index):
        aggr = self.propagate(edge_index, x=x)
        return self.lin_out(torch.cat([self.lin_self(x), aggr], dim=-1))

    def message(self, x_i, x_j):
        return self.edge_mlp(torch.cat([x_j, x_j - x_i], dim=-1))


class EdgeSAGE(torch.nn.Module):
    """EdgeSAGE with additive drone-signature fusion.

    The drone signature is projected once per graph (B × hidden) and broadcast
    to nodes via `batch`, rather than concatenated to every node feature vector
    (which duplicated the 51-d vector across N nodes and blew up memory).
    """

    def __init__(self, num_node_features, num_drone_features, hidden_channels,
                 out_channels, dropout, num_layers):
        super().__init__()
        self.node_proj = torch.nn.Linear(num_node_features, hidden_channels)
        self.drone_proj = torch.nn.Linear(num_drone_features, hidden_channels)
        self.convs = torch.nn.ModuleList(
            [EdgeSAGEConv(hidden_channels, hidden_channels) for _ in range(num_layers)]
        )
        self.norms = torch.nn.ModuleList(
            [torch.nn.LayerNorm(hidden_channels) for _ in range(num_layers)]
        )
        self.output_proj = torch.nn.Linear(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x, edge_index, drone_feat, batch):
        h = self.node_proj(x) + self.drone_proj(drone_feat)[batch]
        for conv, norm in zip(self.convs, self.norms):
            h = h + F.dropout(F.relu(norm(conv(h, edge_index))),
                              p=self.dropout, training=self.training)
        return self.output_proj(h)


model = EdgeSAGE(num_node_features, num_drone_features, args.hidden_channels,
                 num_classes, args.dropout, args.num_layers).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="max", factor=args.lr_factor,
    patience=args.lr_patience, min_lr=args.lr_min,
)

# bf16 autocast when supported: speedup without the numerical pitfalls of fp16
# (no GradScaler required). Falls back to fp32 on CPU / older GPUs.
_use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
amp_ctx = (
    (lambda: torch.autocast(device_type="cuda", dtype=torch.bfloat16))
    if _use_bf16 else contextlib.nullcontext
)
log.info(f"AMP bfloat16: {_use_bf16}")
log.info(f"Batch size: {args.batch_size} × grad_accum {args.grad_accum_steps} "
         f"= effective {args.batch_size * args.grad_accum_steps}")

try:
    model = torch.compile(model)
    log.info("torch.compile: enabled")
except Exception as e:
    log.warning(f"torch.compile: disabled ({e})")

class_weights = torch.tensor(class_weights_np, device=device)


def weighted_ce_loss(logits, target):
    return F.cross_entropy(logits, target, weight=class_weights)


def train_step(epoch: int | None = None):
    model.train()
    total_loss = total_nodes = 0
    accum_steps = max(1, args.grad_accum_steps)
    desc = f"train ep{epoch:03d}" if epoch is not None else "train"
    pbar = tqdm(train_loader, desc=desc, leave=False, unit="batch",
                dynamic_ncols=True)
    optimizer.zero_grad(set_to_none=True)
    n_batches = len(train_loader)
    for step, batch in enumerate(pbar):
        batch = batch.to(device, non_blocking=True)
        with amp_ctx():
            out = model(batch.x, batch.edge_index, batch.drone_feat, batch.batch)
            loss = weighted_ce_loss(out, batch.y)
        (loss / accum_steps).backward()
        total_loss += float(loss) * batch.num_nodes
        total_nodes += batch.num_nodes
        # Step on every accum_steps-th batch, plus a final flush so leftover
        # gradients at the end of the epoch are not silently dropped.
        is_step = (step + 1) % accum_steps == 0 or (step + 1) == n_batches
        if is_step:
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        pbar.set_postfix(loss=f"{total_loss / max(total_nodes, 1):.4f}")
    return total_loss / total_nodes


@torch.no_grad()
def evaluate(loader, desc: str = "eval"):
    model.eval()
    total_loss = total_nodes = 0
    preds_all, targets_all = [], []
    for batch in tqdm(loader, desc=desc, leave=False, unit="batch",
                      dynamic_ncols=True):
        batch = batch.to(device, non_blocking=True)
        with amp_ctx():
            out = model(batch.x, batch.edge_index, batch.drone_feat, batch.batch)
            loss = weighted_ce_loss(out, batch.y)
        total_loss += float(loss) * batch.num_nodes
        total_nodes += batch.num_nodes
        preds_all.append(out.argmax(dim=-1).cpu())
        targets_all.append(batch.y.cpu())
    preds = torch.cat(preds_all).numpy()
    targets = torch.cat(targets_all).numpy()
    acc = float((preds == targets).mean())
    mcc = float(matthews_corrcoef(targets, preds))
    bal_acc = float(balanced_accuracy_score(targets, preds))
    macro_f1 = float(f1_score(targets, preds, average="macro", zero_division=0))
    kappa = float(cohen_kappa_score(targets, preds))
    rare_labels = [c for c in (3, 4) if c < num_classes]
    rare_rec = float(recall_score(
        targets, preds, labels=rare_labels, average="macro", zero_division=0)) \
        if rare_labels else 0.0
    return (total_loss / total_nodes, acc, mcc, bal_acc, macro_f1, kappa,
            rare_rec, preds, targets)


def save_if_better(val_mcc, val_acc, test_mcc, test_acc, epoch, state, prev_mcc):
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    OLD_DIR.mkdir(parents=True, exist_ok=True)
    if val_mcc <= prev_mcc + args.min_delta:
        return False
    for p in CKPT_DIR.glob("edgesage_*.pt"):
        shutil.move(str(p), str(OLD_DIR / p.name))
    name = (f"edgesage_epoch{epoch:03d}_mcc{val_mcc:.4f}_acc{val_acc:.4f}"
            f"_testmcc{test_mcc:.4f}_testacc{test_acc:.4f}.pt")
    torch.save(state, CKPT_DIR / name)
    # Rotate old checkpoints: keep only the most recent N in ./old/
    if args.keep_old_ckpts >= 0:
        olds = sorted(OLD_DIR.glob("edgesage_*.pt"), key=lambda p: p.stat().st_mtime)
        for p in olds[:-args.keep_old_ckpts] if args.keep_old_ckpts else olds:
            p.unlink(missing_ok=True)
    log.success(f"saved new best → {name} (prev MCC={prev_mcc:.4f})")
    return True


best_val_mcc = -2.0
best_val_acc = 0.0
test_mcc_at_best = 0.0
test_acc_at_best = 0.0
best_val_preds = best_val_targets = None
best_ts_preds = best_ts_targets = None
epochs_no_improve = 0
best_epoch = 0
times = []
last_epoch = 0
stop_reason = "completed"

history = {
    "epoch": [],
    "train_loss": [], "val_loss": [], "test_loss": [],
    "val_acc": [], "test_acc": [],
    "val_mcc": [], "test_mcc": [],
    "val_prec": [], "val_rec": [], "val_f1": [],
    "test_prec": [], "test_rec": [], "test_f1": [],
    "lr": [],
}
labels_list = list(range(num_classes))


def _per_cls(arr):
    return "  ".join(f"c{c}:{arr[c]:.3f}" for c in labels_list)


epoch_bar = tqdm(range(1, args.epochs + 1), desc="epochs", unit="ep",
                 dynamic_ncols=True)
try:
    for epoch in epoch_bar:
        last_epoch = epoch
        t0 = time.time()
        tr_loss = train_step(epoch)

        do_eval = (epoch % args.eval_every == 0
                   or epoch == 1
                   or epoch == args.epochs)
        if not do_eval:
            times.append(time.time() - t0)
            epoch_bar.set_postfix(loss=f"{tr_loss:.4f}", eval="skip")
            log.debug(f"Ep {epoch:03d} | loss {tr_loss:.4f} (eval skipped)")
            continue

        (val_loss, val_acc, val_mcc, val_bal, val_f1m, val_kap, val_rare,
         val_preds, val_targets) = evaluate(val_loader, desc=f"val ep{epoch:03d}")
        (ts_loss, ts_acc, ts_mcc, ts_bal, ts_f1m, ts_kap, ts_rare,
         ts_preds, ts_targets) = evaluate(test_loader, desc=f"test ep{epoch:03d}")

        scheduler.step(val_mcc)
        current_lr = optimizer.param_groups[0]["lr"]

        improved = val_mcc > best_val_mcc + args.min_delta
        if improved:
            prev_best_mcc = best_val_mcc
            best_val_mcc, best_val_acc = val_mcc, val_acc
            test_mcc_at_best, test_acc_at_best = ts_mcc, ts_acc
            best_val_preds, best_val_targets = val_preds, val_targets
            best_ts_preds, best_ts_targets = ts_preds, ts_targets
            best_epoch = epoch
            epochs_no_improve = 0
            state_src = getattr(model, "_orig_mod", model)
            save_if_better(val_mcc, val_acc, ts_mcc, ts_acc, epoch, {
                "model_state": state_src.state_dict(),
                "args": vars(args),
                "epoch": epoch,
                "val_mcc": val_mcc, "val_acc": val_acc,
                "test_mcc": ts_mcc, "test_acc": ts_acc,
                "num_node_features": num_node_features,
                "num_drone_features": num_drone_features,
                "num_classes": num_classes,
                "class_weights": class_weights_np.tolist(),
                "train_class_counts": train_counts.tolist(),
                "test_maps": sorted(test_maps),
                "val_maps": sorted(val_maps),
            }, prev_best_mcc)
        else:
            epochs_no_improve += args.eval_every

        v_p, v_r, v_f, _ = precision_recall_fscore_support(
            val_targets, val_preds, labels=labels_list, zero_division=0)
        t_p, t_r, t_f, _ = precision_recall_fscore_support(
            ts_targets, ts_preds, labels=labels_list, zero_division=0)
        history["epoch"].append(epoch)
        history["train_loss"].append(tr_loss)
        history["val_loss"].append(val_loss)
        history["test_loss"].append(ts_loss)
        history["val_acc"].append(val_acc)
        history["test_acc"].append(ts_acc)
        history["val_mcc"].append(val_mcc)
        history["test_mcc"].append(ts_mcc)
        history["val_prec"].append(v_p)
        history["val_rec"].append(v_r)
        history["val_f1"].append(v_f)
        history["test_prec"].append(t_p)
        history["test_rec"].append(t_r)
        history["test_f1"].append(t_f)
        history["lr"].append(current_lr)

        times.append(time.time() - t0)

        epoch_bar.set_postfix(loss=f"{tr_loss:.4f}", val_mcc=f"{val_mcc:.3f}",
                              best=f"{best_val_mcc:.3f}",
                              lr=f"{current_lr:.1e}",
                              pat=f"{epochs_no_improve}/{args.patience}")
        log.info(f"── Epoch {epoch:03d} ──  (loss {tr_loss:.4f} | lr {current_lr:.2e} | "
                 f"pat {epochs_no_improve}/{args.patience})")
        log.info(f"  VAL    | acc {val_acc:.3f} | mcc {val_mcc:.3f} | "
                 f"bal {val_bal:.3f} | f1m {val_f1m:.3f} | "
                 f"kap {val_kap:.3f} | rareRec {val_rare:.3f}")
        log.info(f"    ├─ precision  {_per_cls(v_p)}")
        log.info(f"    ├─ recall     {_per_cls(v_r)}")
        log.info(f"    └─ f1         {_per_cls(v_f)}")
        log.info(f"  TEST   | acc {ts_acc:.3f} | mcc {ts_mcc:.3f} | "
                 f"bal {ts_bal:.3f} | f1m {ts_f1m:.3f} | "
                 f"kap {ts_kap:.3f} | rareRec {ts_rare:.3f}")
        log.info(f"    ├─ precision  {_per_cls(t_p)}")
        log.info(f"    ├─ recall     {_per_cls(t_r)}")
        log.info(f"    └─ f1         {_per_cls(t_f)}")
        if epochs_no_improve >= args.patience:
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


def _fmt_time(seconds: float) -> str:
    seconds = int(seconds)
    h, r = divmod(seconds, 3600)
    m, s = divmod(r, 60)
    return f"{h:d}h{m:02d}m{s:02d}s" if h else f"{m:d}m{s:02d}s"


reason_label = {
    "completed": "Training completed (all epochs)",
    "max_epochs": "Training completed (all epochs)",
    "early_stopping": "Training stopped early (patience exhausted)",
    "manual_interrupt": "Training stopped manually (KeyboardInterrupt)",
}[stop_reason]

total_time = sum(times)
median_time = float(torch.tensor(times).median()) if times else 0.0
mean_time = float(torch.tensor(times).mean()) if times else 0.0

log.success("=" * 70)
log.success(f"  TRAINING SUMMARY — {reason_label}")
log.success("=" * 70)
log.info(f"  Epochs run         : {last_epoch}/{args.epochs}")
log.info(f"  Best epoch         : {best_epoch}")
log.info(f"  Best val   MCC/Acc : {best_val_mcc:.4f} / {best_val_acc:.4f}")
log.info(f"  Test @ best MCC/Acc: {test_mcc_at_best:.4f} / {test_acc_at_best:.4f}")
log.info(f"  Total time         : {_fmt_time(total_time)}")
log.info(f"  Mean/median epoch  : {mean_time:.2f}s / {median_time:.2f}s")
log.info(f"  Checkpoint dir     : {CKPT_DIR}")
log.success("=" * 70)


def _per_class_report(name, preds, targets):
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
    macro = precision_recall_fscore_support(
        targets, preds, labels=labels, average="macro", zero_division=0
    )
    weighted = precision_recall_fscore_support(
        targets, preds, labels=labels, average="weighted", zero_division=0
    )
    log.info(f"  {'macro avg':<10}{macro[0]:>11.4f}{macro[1]:>11.4f}{macro[2]:>11.4f}"
             f"{int(sup.sum()):>11d}")
    log.info(f"  {'weighted':<10}{weighted[0]:>11.4f}{weighted[1]:>11.4f}{weighted[2]:>11.4f}"
             f"{int(sup.sum()):>11d}")
    cm = confusion_matrix(targets, preds, labels=labels)
    log.info("  confusion matrix (rows=true, cols=pred):")
    header = "       " + "".join(f"{c:>8}" for c in labels)
    log.info(header)
    for i, row in enumerate(cm):
        log.info(f"  {i:>4} " + "".join(f"{v:>8d}" for v in row))


_per_class_report("VAL  @ best epoch", best_val_preds, best_val_targets)
_per_class_report("TEST @ best epoch", best_ts_preds, best_ts_targets)


def _save_history_plots(hist, best_epoch, out_dir):
    if not hist["epoch"]:
        log.warning("No epoch history recorded, skipping plots.")
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    ep = np.array(hist["epoch"])

    fig, axes = plt.subplots(3, 3, figsize=(18, 15))

    # --- Row 0: global metrics (loss / accuracy / MCC) ---
    for ax, key, title in zip(
        axes[0], ["loss", "acc", "mcc"], ["Loss", "Accuracy", "MCC"]
    ):
        if key == "loss":
            ax.plot(ep, hist["train_loss"], label="train")
        ax.plot(ep, hist[f"val_{key}"], label="val")
        ax.plot(ep, hist[f"test_{key}"], label="test")
        if best_epoch:
            ax.axvline(best_epoch, color="red", linestyle="--",
                       alpha=0.5, label=f"best ep {best_epoch}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(title)
        ax.set_title(f"{title} per epoch")
        ax.grid(alpha=0.3)
        ax.legend()

    # --- Rows 1-2: per-class precision / recall / F1 (val then test) ---
    for row, split in enumerate(("val", "test"), start=1):
        prec = np.array(hist[f"{split}_prec"])
        rec = np.array(hist[f"{split}_rec"])
        f1 = np.array(hist[f"{split}_f1"])
        for ax, arr, title in zip(
            axes[row], [prec, rec, f1], ["Precision", "Recall", "F1"]
        ):
            for c in range(arr.shape[1]):
                ax.plot(ep, arr[:, c], label=f"class {c}")
            if best_epoch:
                ax.axvline(best_epoch, color="red", linestyle="--", alpha=0.5)
            ax.set_xlabel("Epoch")
            ax.set_ylabel(title)
            ax.set_title(f"{split.upper()} — {title} per class")
            ax.set_ylim(-0.02, 1.02)
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8)

    fig.tight_layout()
    p = out_dir / f"metrics_{stamp}.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    log.success(f"saved {p}")

    # --- Learning-rate curve (separate figure, log scale) ---
    if hist.get("lr"):
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(ep, hist["lr"])
        if best_epoch:
            ax.axvline(best_epoch, color="red", linestyle="--",
                       alpha=0.5, label=f"best ep {best_epoch}")
            ax.legend()
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Learning rate")
        ax.set_yscale("log")
        ax.set_title("Learning-rate schedule")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        p = out_dir / f"lr_{stamp}.png"
        fig.savefig(p, dpi=120)
        plt.close(fig)
        log.success(f"saved {p}")


_save_history_plots(history, best_epoch, PLOTS_DIR)
