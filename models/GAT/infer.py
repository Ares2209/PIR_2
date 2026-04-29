"""Run a trained 7-class GAT on a base mesh to produce a face-colored
noise-map PLY for a given drone and source position.

Usage:
    python models/GAT/infer.py --drone M2 --x 1.2 --y -3.4 --z 0.5
    python models/GAT/infer.py --drone F-4 --x 4 --y 4 --z 4 --ckpt /home/ldena/Bureau/PIR/models/GAT/checkpoints/lr5e-04_wd5e-05_bs4__cw0.20-0.17-0.33-0.61-1.46-4.09-0.14/gat_epoch045_mcc0.5626_acc0.6847_testmcc0.5509_testacc0.6788.pt
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv

ROOT    = Path(__file__).resolve().parents[2]
GAT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(GAT_DIR))

# Tout vient de config.py — gen_graphs.py n'exporte pas de fonctions publiques
from config import (  # noqa: E402
    DRONES, NUM_FEATURES, DRONE_FEAT_DIM, RGB_TO_CLASS,
    _parse_ply, _face_adjacency, _normalize_drone_vector,
    _load_base_mesh, _OCC_CHUNK,
)
from log_utils import get_logger  # noqa: E402

# Import de la construction des features (définie dans gen_graphs)
sys.path.insert(0, str(ROOT / "dataset" / "data" / "generated"))
from gen_graphs import _build_node_features, _compute_occlusion  # noqa: E402

log = get_logger("gat.infer")

VILLE_PLY    = ROOT / "dataset" / "blender" / "ville.ply"
CKPT_DIR     = GAT_DIR / "checkpoints"
OUT_DIR      = GAT_DIR / "predictions"
NOISEMAP_DIR = ROOT / "dataset" / "data" / "NoiseMap-RT-main"
NOISEMAP_BIN = NOISEMAP_DIR / "build" / "NoiseMap"

CLASS_TO_RGB = np.array([
    [128,   0, 200],  # 0 violet
    [  0,  80, 255],  # 1 blue
    [255, 230,   0],  # 2 yellow
    [255, 140,   0],  # 3 orange
    [255,   0,   0],  # 4 red
    [100,   0,   0],  # 5 dark_red
    [ 30,  30,  30],  # 6 occluded
], dtype=np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
# Model — must mirror GAT.py exactly
# ─────────────────────────────────────────────────────────────────────────────
class GAT(torch.nn.Module):
    def __init__(self, num_node_features, num_drone_features, hidden_channels,
                 out_channels, dropout, attn_dropout, num_layers, num_heads):
        super().__init__()
        self.node_proj  = torch.nn.Linear(num_node_features,  hidden_channels)
        self.drone_proj = torch.nn.Linear(num_drone_features, hidden_channels)
        self.convs = torch.nn.ModuleList([
            GATConv(hidden_channels, hidden_channels,
                    heads=num_heads, concat=False, dropout=attn_dropout)
            for _ in range(num_layers)
        ])
        self.norms = torch.nn.ModuleList([
            torch.nn.LayerNorm(hidden_channels) for _ in range(num_layers)
        ])
        self.output_proj = torch.nn.Linear(hidden_channels, out_channels)
        self.dropout     = dropout

    def forward(self, x, edge_index, drone_feat, batch):
        h = self.node_proj(x) + self.drone_proj(drone_feat)[batch]
        for conv, norm in zip(self.convs, self.norms):
            h = h + F.dropout(F.relu(norm(conv(h, edge_index))),
                              p=self.dropout, training=self.training)
        return self.output_proj(h)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def rgb_to_class_exact(rgb: np.ndarray) -> np.ndarray:
    y = np.full(rgb.shape[0], -1, dtype=np.int64)
    for (r, g, b), c in RGB_TO_CLASS.items():
        y[(rgb[:, 0] == r) & (rgb[:, 1] == g) & (rgb[:, 2] == b)] = c
    return y


def pick_best_ckpt():
    best, best_mcc = None, -2.0
    for p in CKPT_DIR.rglob("gat_*.pt"):
        if "old" in p.parts:
            continue
        m = re.search(r"mcc([-+]?\d*\.\d+)", p.name)
        if m and float(m.group(1)) > best_mcc:
            best_mcc = float(m.group(1))
            best = p
    return best


def write_ply_face_colors(path, verts, faces, face_rgb):
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(verts)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write(f"element face {len(faces)}\n")
        f.write("property list uchar int vertex_indices\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for x, y, z in verts:
            f.write(f"{x:.6f} {y:.6f} {z:.6f}\n")
        for (a, b, c), (r, g, bl) in zip(faces, face_rgb):
            f.write(f"3 {a} {b} {c} {int(r)} {int(g)} {int(bl)}\n")


def run_noisemap(ville_path: Path, drone, x, y, z) -> Path:
    if not NOISEMAP_BIN.exists():
        raise FileNotFoundError(f"NoiseMap binary not found: {NOISEMAP_BIN}")
    ville_abs = ville_path.resolve()
    if not ville_abs.is_file():
        raise FileNotFoundError(f"Ville mesh not found: {ville_abs}")
    cmd = [str(NOISEMAP_BIN), str(ville_abs),
           str(x), str(y), str(z), "--drone", drone]
    log.info(f"Running ground-truth NoiseMap: {' '.join(cmd)}")
    subprocess.run(cmd, cwd=NOISEMAP_BIN.parent, check=True,
                   stdout=subprocess.DEVNULL)
    return ville_abs.with_name(ville_abs.stem + "_noisemap.ply")


def _load_and_normalize_stats(ckpt: dict) -> tuple[np.ndarray, np.ndarray] | None:
    """Charge mean/std depuis le checkpoint si disponible."""
    mean = ckpt.get("node_mean")
    std  = ckpt.get("node_std")
    if mean is not None and std is not None:
        return np.array(mean, dtype=np.float32), np.array(std, dtype=np.float32)
    # Fallback : node_stats.json à côté des shards
    stats_path = ROOT / "dataset" / "data" / "generated" / "processed" / "node_stats.json"
    if stats_path.exists():
        import json
        with stats_path.open() as f:
            stats = json.load(f)
        from config import FEAT_KEYS
        mean = np.array([stats[k][0] for k in FEAT_KEYS], dtype=np.float32)
        std  = np.array([stats[k][1] for k in FEAT_KEYS], dtype=np.float32)
        log.info(f"Loaded normalisation stats from {stats_path}")
        return mean, std
    log.warning("No normalisation stats found — features will NOT be normalised.")
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drone",      required=True, choices=DRONES)
    ap.add_argument("--x",          type=float, required=True)
    ap.add_argument("--y",          type=float, required=True)
    ap.add_argument("--z",          type=float, required=True)
    ap.add_argument("--ckpt",       type=str, default=None)
    ap.add_argument("--ville",      type=str, default=str(VILLE_PLY))
    ap.add_argument("--out",        type=str, default=None)
    ap.add_argument("--no-compare", action="store_true",
                    help="Skip running NoiseMap ground truth comparison")
    args = ap.parse_args()

    # ── Checkpoint ────────────────────────────────────────────────────────────
    ckpt_path = Path(args.ckpt) if args.ckpt else pick_best_ckpt()
    if ckpt_path is None or not ckpt_path.exists():
        raise FileNotFoundError(f"No checkpoint found in {CKPT_DIR}")
    log.info(f"Loading checkpoint: {ckpt_path.name}")
    ckpt = torch.load(ckpt_path, weights_only=False, map_location="cpu")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Using device: {device}")

    # ── Modèle ────────────────────────────────────────────────────────────────
    model = GAT(
        num_node_features  = ckpt["num_node_features"],
        num_drone_features = ckpt["num_drone_features"],
        hidden_channels    = ckpt["args"]["hidden_channels"],
        out_channels       = ckpt["num_classes"],
        dropout            = ckpt["args"].get("dropout",      0.0),
        attn_dropout       = ckpt["args"].get("attn_dropout", 0.0),
        num_layers         = ckpt["args"].get("num_layers",   5),
        num_heads          = ckpt["args"].get("num_heads",    4),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # ── Mesh ──────────────────────────────────────────────────────────────────
    ville_path = Path(args.ville)
    log.info(f"Parsing mesh: {ville_path}")
    verts, faces, _ = _parse_ply(ville_path)
    map_name        = ville_path.stem          # e.g. "ville"
    src             = np.array([args.x, args.y, args.z], dtype=np.float32)

    # ── Features (même pipeline que gen_graphs.py) ────────────────────────────
    feats = _build_node_features(verts, faces, src, map_name)  # (M, NUM_FEATURES)
    assert feats.shape[1] == NUM_FEATURES, \
        f"Feature dim mismatch: got {feats.shape[1]}, expected {NUM_FEATURES}"

    # Normalisation
    norm_stats = _load_and_normalize_stats(ckpt)
    if norm_stats is not None:
        mean, std = norm_stats
        feats = (feats - mean) / std

    drone_feat_np = _normalize_drone_vector(args.drone)        # (DRONE_FEAT_DIM,)
    ei            = _face_adjacency(faces)                     # (2, E)

    # Centroïdes pour les plots
    v0 = verts[faces[:, 0]]; v1 = verts[faces[:, 1]]; v2 = verts[faces[:, 2]]
    centroids = (v0 + v1 + v2) / 3.0

    x_t       = torch.from_numpy(feats).to(device)
    df_t      = torch.from_numpy(drone_feat_np).unsqueeze(0).to(device)  # (1, DRONE_FEAT_DIM)
    ei_t      = torch.from_numpy(ei).long().to(device)
    batch_vec = torch.zeros(x_t.shape[0], dtype=torch.long, device=device)

    # ── Inférence ─────────────────────────────────────────────────────────────
    with torch.no_grad():
        logits = model(x_t, ei_t, df_t, batch_vec)
        cls    = logits.argmax(dim=-1).cpu().numpy()

    rgb_out = CLASS_TO_RGB[cls]
    counts  = np.bincount(cls, minlength=ckpt["num_classes"])
    log.info(f"Class distribution (pred): {counts.tolist()}")

    # ── Écriture PLY prédiction ───────────────────────────────────────────────
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_name = args.out or (
        f"NoiseMap_{args.x:.4f}_{args.y:.4f}_{args.z:.4f}_{args.drone}_pred.ply"
    )
    out_path = OUT_DIR / out_name
    write_ply_face_colors(out_path, verts, faces, rgb_out)
    log.success(f"Written: {out_path}")

    if args.no_compare:
        return

    # ── Ground truth via NoiseMap ─────────────────────────────────────────────
    try:
        noisemap_out = run_noisemap(ville_path, args.drone, args.x, args.y, args.z)
        gt_verts, gt_faces, gt_rgb = _parse_ply(noisemap_out)
    except (FileNotFoundError, subprocess.CalledProcessError) as e:
        log.warning(f"Ground-truth comparison skipped: {e}")
        return
    if gt_rgb is None or not gt_rgb.any():
        log.warning("Ground-truth comparison skipped: no face RGB in NoiseMap output.")
        return

    if gt_faces.shape[0] != faces.shape[0]:
        log.warning(f"Face count mismatch: pred={faces.shape[0]} gt={gt_faces.shape[0]} — "
                    "matching by nearest centroid.")
        gt_c = ((gt_verts[gt_faces[:, 0]] + gt_verts[gt_faces[:, 1]]
                 + gt_verts[gt_faces[:, 2]]) / 3.0)
        try:
            from scipy.spatial import cKDTree
            _, nn = cKDTree(gt_c).query(centroids, k=1)
        except ImportError:
            diffs = centroids[:, None, :] - gt_c[None, :, :]
            nn    = np.argmin(np.linalg.norm(diffs, axis=-1), axis=-1)
        gt_rgb = gt_rgb[nn]

    gt_cls = rgb_to_class_exact(gt_rgb)
    valid  = gt_cls >= 0
    if not valid.any():
        log.warning("Ground-truth comparison skipped: no classifiable GT faces.")
        return

    gt_counts = np.bincount(gt_cls[valid], minlength=ckpt["num_classes"])
    acc = float((cls[valid] == gt_cls[valid]).mean())
    try:
        from sklearn.metrics import confusion_matrix, matthews_corrcoef
        mcc = matthews_corrcoef(gt_cls[valid], cls[valid])
        cm  = confusion_matrix(gt_cls[valid], cls[valid],
                               labels=list(range(ckpt["num_classes"])))
    except Exception:
        mcc, cm = float("nan"), None

    log.info(f"Class distribution (gt):   {gt_counts.tolist()}")
    log.success(f"Accuracy vs NoiseMap: {acc:.4f} | MCC: {mcc:.4f}")

    per_cls = (np.diag(cm) / np.maximum(gt_counts, 1)
               if cm is not None else None)
    if per_cls is not None:
        log.info("Per-class recall:")
        for i, r in enumerate(per_cls):
            log.info(f"  class {i}: gt={gt_counts[i]:6d} recall={r:.3f}")
        log.info("Confusion matrix (rows=gt, cols=pred):")
        for row in cm:
            log.info("  " + " ".join(f"{v:6d}" for v in row))

    gt_rgb_out = CLASS_TO_RGB[np.where(valid, gt_cls, 0)]
    gt_path    = OUT_DIR / out_name.replace("_pred.ply", "_gt.ply")
    write_ply_face_colors(gt_path, verts, faces, gt_rgb_out)
    log.success(f"Written GT: {gt_path}")

    plot_path = OUT_DIR / out_name.replace("_pred.ply", "_plots.png")
    make_plots(centroids, cls, gt_cls, counts, gt_counts, cm, per_cls,
               acc, mcc, args, plot_path)
    log.success(f"Written plots: {plot_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────────────────────
def make_plots(centroids, pred_cls, gt_cls, pred_counts, gt_counts, cm, per_cls,
               acc, mcc, args, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap

    K       = len(CLASS_TO_RGB)
    palette = (CLASS_TO_RGB / 255.0).clip(0, 1)
    cmap    = ListedColormap(palette)
    norm    = BoundaryNorm(np.arange(K + 1) - 0.5, K)

    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(f"Drone {args.drone} — source ({args.x}, {args.y}, {args.z}) "
                 f"| acc={acc:.3f}  MCC={mcc:.3f}", fontsize=13)

    ax1 = fig.add_subplot(2, 3, 1)
    ax1.scatter(centroids[:, 0], centroids[:, 1], c=gt_cls, cmap=cmap, norm=norm,
                s=2, marker=".")
    ax1.scatter([args.x], [args.y], c="red", marker="*", s=120,
                edgecolors="black", linewidths=0.7, label="source")
    ax1.set_title("Ground truth (NoiseMap)")
    ax1.set_xlabel("x"); ax1.set_ylabel("y"); ax1.set_aspect("equal")
    ax1.legend(loc="upper right", fontsize=8)

    ax2 = fig.add_subplot(2, 3, 2)
    ax2.scatter(centroids[:, 0], centroids[:, 1], c=pred_cls, cmap=cmap, norm=norm,
                s=2, marker=".")
    ax2.scatter([args.x], [args.y], c="red", marker="*", s=120,
                edgecolors="black", linewidths=0.7)
    ax2.set_title("GAT prediction")
    ax2.set_xlabel("x"); ax2.set_ylabel("y"); ax2.set_aspect("equal")

    ax3 = fig.add_subplot(2, 3, 3)
    err = (pred_cls != gt_cls).astype(np.float32)
    ax3.scatter(centroids[err == 0, 0], centroids[err == 0, 1], c="lightgrey",
                s=2, marker=".", label="correct")
    ax3.scatter(centroids[err == 1, 0], centroids[err == 1, 1], c="red",
                s=4, marker=".", label="error")
    ax3.scatter([args.x], [args.y], c="blue", marker="*", s=120,
                edgecolors="black", linewidths=0.7)
    ax3.set_title(f"Errors ({int(err.sum())}/{len(err)})")
    ax3.set_xlabel("x"); ax3.set_ylabel("y"); ax3.set_aspect("equal")
    ax3.legend(loc="upper right", fontsize=8)

    ax4 = fig.add_subplot(2, 3, 4)
    idx = np.arange(K); w = 0.4
    ax4.bar(idx - w / 2, gt_counts,   w, label="GT",
            color=palette, edgecolor="black")
    ax4.bar(idx + w / 2, pred_counts, w, label="Pred",
            color=palette, edgecolor="black", hatch="//")
    ax4.set_xticks(idx); ax4.set_xlabel("class"); ax4.set_ylabel("# faces")
    ax4.set_title("Class distribution"); ax4.legend()

    ax5 = fig.add_subplot(2, 3, 5)
    if cm is not None:
        cm_norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
        im = ax5.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
        for i in range(K):
            for j in range(K):
                ax5.text(j, i, f"{cm[i, j]}", ha="center", va="center",
                         color="white" if cm_norm[i, j] > 0.5 else "black",
                         fontsize=8)
        ax5.set_xticks(idx); ax5.set_yticks(idx)
        ax5.set_xlabel("predicted"); ax5.set_ylabel("ground truth")
        ax5.set_title("Confusion matrix (row-normalised)")
        fig.colorbar(im, ax=ax5, fraction=0.046, pad=0.04)

    ax6 = fig.add_subplot(2, 3, 6)
    if per_cls is not None:
        ax6.bar(idx, per_cls, color=palette, edgecolor="black")
        ax6.set_ylim(0, 1.05); ax6.set_xticks(idx)
        ax6.set_xlabel("class"); ax6.set_ylabel("recall")
        ax6.set_title("Per-class recall")
        for i, r in enumerate(per_cls):
            ax6.text(i, r + 0.02, f"{r:.2f}", ha="center", fontsize=8)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    main()
