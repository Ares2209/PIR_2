"""Hyperparameters for the GAT model. Edit values here to tune training."""

from __future__ import annotations

import json
import re
import warnings
from pathlib import Path
from typing import Optional

import numpy as np

# Model architecture
HIDDEN_CHANNELS = 64
NUM_LAYERS      = 3
NUM_HEADS       = 4          
ATTN_DROPOUT    = 0.2        
DROPOUT         = 0.2       

LR               = 5e-4
WEIGHT_DECAY     = 5e-5
EPOCHS           = 300
BATCH_SIZE       = 4
GRAD_CLIP        = 1.0
GRAD_ACCUM_STEPS = 4

VAL_RATIO  = 0.15
TEST_RATIO = 0.15
SPLIT_SEED = 0
SEED       = 42

EVAL_EVERY  = 5
PATIENCE    = 50
MIN_DELTA   = 1e-4

LR_FACTOR   = 0.5
LR_PATIENCE = 10
LR_MIN      = 1e-6

KEEP_OLD_CHECKPOINTS = 3

# ─────────────────────────────────────────────────────────────────────────────
# Chemins
# ─────────────────────────────────────────────────────────────────────────────
SCRIPT_DIR  = Path(__file__).resolve().parent
REPO_ROOT   = SCRIPT_DIR.parents[1]
GENERATED   = REPO_ROOT / "dataset" / "data" / "generated"
OUT_DIR     = GENERATED / "processed"
SHARD_DIR   = OUT_DIR / "shards"
STATS_FILE  = OUT_DIR / "node_stats.json"
BLENDER_DIR = REPO_ROOT / "dataset" / "blender"


# ─────────────────────────────────────────────────────────────────────────────
# Drones — signatures acoustiques
# ─────────────────────────────────────────────────────────────────────────────
DRONES = ["F-4", "I2", "M2", "S-9"]

META: dict[str, tuple[int, float, float]] = {
    "M2":  (4,  907.0, 6540.0),
    "I2":  (4, 3400.0, 4560.0),
    "F-4": (4,  800.0, 6420.0),
    "S-9": (6, 3300.0, 6900.0),
}

LP_REF: dict[str, list[float]] = {
    "M2": [
        40, 42, 43, 45, 55, 60, 58, 62, 65, 68,
        70, 68, 65, 63, 60, 58, 55, 52,
        48, 44, 40, 35, 28, 20,
    ],
    "I2": [
        45, 48, 50, 52, 60, 65, 63, 66, 68, 70,
        72, 70, 68, 65, 63, 60, 57, 54,
        50, 46, 42, 37, 30, 22,
    ],
    "F-4": [
        38, 40, 42, 44, 52, 57, 55, 58, 60, 63,
        65, 63, 60, 58, 55, 52, 50, 47,
        43, 39, 35, 30, 24, 16,
    ],
    "S-9": [
        48, 50, 52, 55, 63, 68, 66, 70, 72, 74,
        75, 73, 70, 68, 65, 62, 59, 56,
        52, 48, 44, 39, 32, 24,
    ],
}

SLOPE: dict[str, list[float]] = {
    "M2": [
        1.5e-3,  2.9e-3, -1.3e-4, 4.4e-3, 3.8e-3, 4.6e-3, 4.4e-3, 4.3e-3,
        4.0e-3,  3.6e-3,  3.7e-3, 2.7e-3, 3.8e-3, 3.8e-3, 3.6e-3, 3.5e-3,
        3.5e-3,  3.3e-3,  3.3e-3, 3.3e-3, 2.3e-3, 7.5e-4, 1.8e-4, 1.8e-4,
    ],
    "I2": [
        6.6e-3, 5.4e-3, 9.8e-3, 1.1e-2, 8.7e-3, 9.3e-3, 1.1e-2, 8.1e-3,
        8.5e-3, 8.8e-3, 7.7e-3, 7.8e-3, 8.0e-3, 8.0e-3, 8.0e-3, 8.1e-3,
        8.4e-3, 8.4e-3, 9.0e-3, 8.3e-3, 7.9e-3, 6.2e-3, 4.9e-3, 4.9e-3,
    ],
    "F-4": [
        2.3e-4, 3.6e-2, 0.0,    0.0,    3.3e-2, 1.7e-2, 1.2e-2, 4.1e-2,
        1.8e-2, 4.5e-2, 2.3e-2, 2.9e-2, 2.6e-2, 2.3e-2, 1.8e-2, 2.2e-2,
        1.8e-2, 2.0e-2, 2.0e-2, 1.4e-2, 1.7e-2, 1.9e-2, 1.9e-2, 1.9e-2,
    ],
    "S-9": [
        5.8e-3, 1.4e-3, 4.1e-3, 7.4e-3, 6.3e-3, 1.0e-2, 7.0e-3, 8.2e-3,
        6.0e-3, 1.4e-3, 4.1e-3, 3.0e-3, 5.2e-4, 2.4e-3, 3.8e-3, 3.6e-3,
        4.3e-3, 4.0e-3, 3.8e-3, 4.3e-3, 2.8e-3, 7.4e-4, 2.9e-3, 2.9e-3,
    ],
}

N_BANDS        = 24
NUM_FEATURES   = 18
DRONE_FEAT_DIM = 3 + N_BANDS + N_BANDS  # = 51


# ─────────────────────────────────────────────────────────────────────────────
# Features — ordre canonique des colonnes (doit matcher gen_graphs.py)
# ─────────────────────────────────────────────────────────────────────────────
FEAT_KEYS = [
    "log_dist", "cos_ns", "rel_x", "rel_y", "rel_z",
    "log_height", "log_area", "normal_z", "log_horiz_dist", "occluded",
    "cos_angles", "grazing_angle", "elevation_angle",
    "obstacle_proximity", "slope_discontinuity",
    "normal_x", "normal_y", "cos_horiz",
]
assert len(FEAT_KEYS) == NUM_FEATURES, "FEAT_KEYS doit avoir NUM_FEATURES entrées"


# ─────────────────────────────────────────────────────────────────────────────
# Stats de normalisation (z-score). Valeurs par défaut écrasées par
# `node_stats.json` à l'import si le fichier existe.
# ─────────────────────────────────────────────────────────────────────────────
NODE_STATS: dict[str, tuple[float, float]] = {
    "log_dist":            (1.2334,  0.2244),
    "cos_ns":              (-0.0256, 0.4906),
    "rel_x":               (0.0077,  0.6940),
    "rel_y":               (0.0087,  0.6962),
    "rel_z":               (-0.0385, 0.1788),
    "log_height":          (0.2712,  0.2985),
    "log_area":            (0.0280,  0.0197),
    "normal_z":            (0.4053,  0.6046),
    "log_horiz_dist":      (1.2245,  0.2351),
    "occluded":            (0.3065,  0.4610),
    "cos_angles":          (0.4793,  0.5988),
    "grazing_angle":       (0.1860,  0.3684),
    "elevation_angle":     (0.9263,  0.0335),
    "obstacle_proximity":  (0.2643,  0.2055),
    "slope_discontinuity": (0.1996,  0.2058),
    "normal_x":            (0.0,     0.5),
    "normal_y":            (0.0,     0.5),
    "cos_horiz":           (0.0,     0.5),
}

DRONE_NORM = {
    "n_blades": (4.0, 6.0),
    "rpm":      (800.0, 7000.0),
    "lp_ref":   (16.0, 75.0),
    "slope":    (-1.3e-4, 4.5e-2),
}


def _load_stats_from_json() -> None:
    """Si STATS_FILE existe, écrase NODE_STATS avec les vraies valeurs calculées
    par gen_graphs.py. Permet à infer.py d'utiliser automatiquement les bonnes
    stats sans modification de code."""
    if not STATS_FILE.exists():
        return
    try:
        with STATS_FILE.open() as f:
            loaded = json.load(f)
        for k, v in loaded.items():
            NODE_STATS[k] = (float(v[0]), float(v[1]))
    except Exception as exc:
        warnings.warn(f"impossible de charger {STATS_FILE}: {exc}")


_load_stats_from_json()


# ─────────────────────────────────────────────────────────────────────────────
# Mapping classes ↔ couleurs RGB
# ─────────────────────────────────────────────────────────────────────────────
RGB_TO_CLASS: dict[tuple[int, int, int], int] = {
    (128,   0, 200): 0,   # violet    SPL < 0 dB
    (  0,  80, 255): 1,   # blue      SPL < 15 dB
    (255, 230,   0): 2,   # yellow    SPL < 25 dB
    (255, 140,   0): 3,   # orange    SPL < 35 dB
    (255,   0,   0): 4,   # red       SPL < 45 dB
    (100,   0,   0): 5,   # dark_red  SPL >= 45 dB
    ( 30,  30,  30): 6,   # occluded  non visible from drone
}
NUM_CLASSES   = 7
RGB_TOLERANCE = 8   # ±8 par canal pour absorber les arrondis de rendu


# ─────────────────────────────────────────────────────────────────────────────
# Filename pattern
# ─────────────────────────────────────────────────────────────────────────────
FNAME_RE = re.compile(
    r"^NoiseMap_"
    r"(?P<map>[^_]+(?:_[^_]+)*?)_"   # map_name : tout jusqu'aux 3 floats
    r"(?P<x>[+-]?\d+(?:\.\d+)?)_"
    r"(?P<y>[+-]?\d+(?:\.\d+)?)_"
    r"(?P<z>[+-]?\d+(?:\.\d+)?)_"
    r"(?P<drone>.+)"                  # drone_id : tout le reste (peut contenir _)
    r"\.ply$"
)


# ─────────────────────────────────────────────────────────────────────────────
# Drone feature normalization (vecteur acoustique 51-dim)
# ─────────────────────────────────────────────────────────────────────────────
def _normalize_drone_vector(drone: str) -> np.ndarray:
    """Retourne le vecteur acoustique 51-dim normalisé en [0, 1] (float32)."""
    n, rmin, rmax = META[drone]

    n_norm    = (n    - DRONE_NORM["n_blades"][0]) / (
                 DRONE_NORM["n_blades"][1] - DRONE_NORM["n_blades"][0])
    rmin_norm = (rmin - DRONE_NORM["rpm"][0]) / (
                 DRONE_NORM["rpm"][1] - DRONE_NORM["rpm"][0])
    rmax_norm = (rmax - DRONE_NORM["rpm"][0]) / (
                 DRONE_NORM["rpm"][1] - DRONE_NORM["rpm"][0])

    lp      = np.array(LP_REF[drone], dtype=np.float32)
    lp_norm = (lp - DRONE_NORM["lp_ref"][0]) / (
               DRONE_NORM["lp_ref"][1] - DRONE_NORM["lp_ref"][0])

    sl      = np.array(SLOPE[drone], dtype=np.float32)
    sl_norm = (sl - DRONE_NORM["slope"][0]) / (
               DRONE_NORM["slope"][1] - DRONE_NORM["slope"][0])

    return np.concatenate(
        [[n_norm, rmin_norm, rmax_norm], lp_norm, sl_norm]
    ).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Node feature normalization (z-score avec NODE_STATS)
# ─────────────────────────────────────────────────────────────────────────────
def _normalize_node_features(feats: np.ndarray) -> np.ndarray:
    """Z-score des features nœuds avec les stats globales (NODE_STATS)."""
    means = np.array([NODE_STATS[k][0] for k in FEAT_KEYS], dtype=np.float32)
    stds  = np.array([NODE_STATS[k][1] for k in FEAT_KEYS], dtype=np.float32)
    stds  = np.where(stds > 1e-8, stds, 1.0)   # garde-fou
    return ((feats - means) / stds).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# PLY parser (ASCII uniquement)
# ─────────────────────────────────────────────────────────────────────────────
def _parse_header(f) -> list[tuple[str, int, list[tuple[str, bool]]]]:
    """Parse le header PLY ASCII; retourne [(elem_name, count, props), …]."""
    first = f.readline().strip()
    if first != "ply":
        raise ValueError("not a PLY file")
    fmt = f.readline().strip()
    if "ascii" not in fmt:
        raise ValueError(f"only ASCII PLY supported, got: {fmt}")

    elements: list[tuple[str, int, list]] = []
    current: Optional[tuple] = None

    while True:
        line = f.readline()
        if not line:
            raise ValueError("unexpected EOF in PLY header")
        line = line.strip()
        if line == "end_header":
            break
        if line.startswith(("comment", "format", "obj_info")):
            continue

        toks = line.split()
        if toks[0] == "element":
            if current is not None:
                elements.append(current)
            current = (toks[1], int(toks[2]), [])
        elif toks[0] == "property" and current is not None:
            is_list = toks[1] == "list"
            current[2].append((toks[-1], is_list))

    if current is not None:
        elements.append(current)
    return elements


def _parse_ply(path: Path):
    """Parse un PLY ASCII. Retourne (verts, faces, face_rgb).

    verts    : (N, 3) float32
    faces    : (M, 3) int64   — indices validés
    face_rgb : (M, 3) uint8
    """
    verts = faces = face_rgb = None

    with path.open("r") as f:
        elements = _parse_header(f)

        for elem_name, n, props in elements:
            names = [p[0] for p in props]

            if elem_name == "vertex":
                ix = names.index("x")
                iy = names.index("y")
                iz = names.index("z")
                arr = np.empty((n, 3), dtype=np.float32)
                for i in range(n):
                    toks = f.readline().split()
                    arr[i, 0] = float(toks[ix])
                    arr[i, 1] = float(toks[iy])
                    arr[i, 2] = float(toks[iz])
                verts = arr

            elif elem_name == "face":
                has_rgb = {"red", "green", "blue"}.issubset(set(names))
                fa  = np.empty((n, 3), dtype=np.int64)
                rgb = np.zeros((n, 3), dtype=np.uint8)

                for i in range(n):
                    toks = f.readline().split()
                    pos  = 0
                    vals: dict[str, object] = {}

                    for name, is_list in props:
                        if is_list:
                            k = int(toks[pos]); pos += 1
                            vals[name] = toks[pos: pos + k]; pos += k
                        else:
                            vals[name] = toks[pos]; pos += 1

                    vi = vals["vertex_indices"]
                    fa[i, 0] = int(vi[0])
                    fa[i, 1] = int(vi[1])
                    fa[i, 2] = int(vi[2])
                    if has_rgb:
                        rgb[i, 0] = int(vals["red"])
                        rgb[i, 1] = int(vals["green"])
                        rgb[i, 2] = int(vals["blue"])

                faces    = fa
                face_rgb = rgb

            else:
                for _ in range(n):
                    f.readline()

    if verts is None or faces is None or face_rgb is None:
        raise ValueError(f"{path}: missing vertex or face element")

    # Validation des indices
    n_verts = verts.shape[0]
    valid = (
        (faces[:, 0] >= 0) & (faces[:, 0] < n_verts) &
        (faces[:, 1] >= 0) & (faces[:, 1] < n_verts) &
        (faces[:, 2] >= 0) & (faces[:, 2] < n_verts)
    )
    if not valid.all():
        warnings.warn(f"{path.name}: dropped {int((~valid).sum())} invalid faces")
        faces, face_rgb = faces[valid], face_rgb[valid]

    # Faces dégénérées
    degen = (
        (faces[:, 0] == faces[:, 1]) |
        (faces[:, 1] == faces[:, 2]) |
        (faces[:, 0] == faces[:, 2])
    )
    if degen.any():
        warnings.warn(f"{path.name}: dropped {int(degen.sum())} degenerate faces")
        faces, face_rgb = faces[~degen], face_rgb[~degen]

    return verts, faces, face_rgb


# ─────────────────────────────────────────────────────────────────────────────
# Face adjacency (edges entre faces partageant une arête)
# ─────────────────────────────────────────────────────────────────────────────
def _face_adjacency(faces: np.ndarray) -> np.ndarray:
    """Construit edge_index (2, E) symétrique et dédupliqué entre faces."""
    n = faces.shape[0]
    if n == 0:
        return np.empty((2, 0), dtype=np.int64)

    e = np.concatenate([
        np.sort(faces[:, [0, 1]], axis=1),
        np.sort(faces[:, [1, 2]], axis=1),
        np.sort(faces[:, [0, 2]], axis=1),
    ], axis=0)

    fid    = np.tile(np.arange(n, dtype=np.int64), 3)
    order  = np.lexsort((e[:, 1], e[:, 0]))
    e_s    = e[order]
    fid_s  = fid[order]

    eq  = (e_s[1:, 0] == e_s[:-1, 0]) & (e_s[1:, 1] == e_s[:-1, 1])
    src = fid_s[:-1][eq]
    dst = fid_s[1:][eq]

    if src.size == 0:
        return np.empty((2, 0), dtype=np.int64)

    s = np.concatenate([src, dst])
    d = np.concatenate([dst, src])
    no_self = s != d
    s, d = s[no_self], d[no_self]

    pairs = np.unique(np.stack([s, d], axis=1), axis=0)
    return pairs.T.astype(np.int64)


# ─────────────────────────────────────────────────────────────────────────────
# Classification RGB → classe
# ─────────────────────────────────────────────────────────────────────────────
def _classify_with_tolerance(rgb: np.ndarray) -> np.ndarray:
    """Mappe (N, 3) uint8 RGB → classes; inconnu → -1.

    Tolérance ±RGB_TOLERANCE par canal pour absorber les arrondis de rendu.
    """
    y     = np.full(rgb.shape[0], -1, dtype=np.int64)
    rgb_f = rgb.astype(np.int16)

    for (r, g, b), cls in RGB_TO_CLASS.items():
        mask = (
            (np.abs(rgb_f[:, 0] - r) <= RGB_TOLERANCE) &
            (np.abs(rgb_f[:, 1] - g) <= RGB_TOLERANCE) &
            (np.abs(rgb_f[:, 2] - b) <= RGB_TOLERANCE)
        )
        y[mask] = cls

    return y


# ─────────────────────────────────────────────────────────────────────────────
# Cache des meshes de base (ray-cast occlusion via trimesh BVH)
# ─────────────────────────────────────────────────────────────────────────────
_BASE_MESH_CACHE: dict[str, "object"] = {}
_OCC_CHUNK = 4096   # taille de batch de rayons (RAM bounded)


def _load_base_mesh(map_name: str):
    """Charge avec cache le mesh de base depuis BLENDER_DIR/<map_name>.ply."""
    if map_name in _BASE_MESH_CACHE:
        return _BASE_MESH_CACHE[map_name]
    import trimesh   # import local : évite la dépendance dure
    ply = BLENDER_DIR / f"{map_name}.ply"
    if not ply.exists():
        warnings.warn(
            f"base mesh for '{map_name}' not found at {ply} — "
            f"occlusion feature will be all zeros"
        )
        _BASE_MESH_CACHE[map_name] = None
        return None
    mesh = trimesh.load(str(ply), process=False, force="mesh")
    # Purge des triangles quasi-dégénérés (causent NaN dans points_to_barycentric)
    keep = mesh.area_faces > 1e-8
    if not keep.all():
        mesh.update_faces(keep)
        mesh.remove_unreferenced_vertices()
    # Force la construction du BVH
    _ = mesh.ray.intersects_any(
        ray_origins=np.array([[0.0, 0.0, 1e6]]),
        ray_directions=np.array([[0.0, 0.0, -1.0]]),
    )
    _BASE_MESH_CACHE[map_name] = mesh
    return mesh


# ─────────────────────────────────────────────────────────────────────────────
# Loader du dataset shardé (utilisé par EdgeSAGE.py et infer.py)
# ─────────────────────────────────────────────────────────────────────────────
def load_sharded_dataset(shard_dir: Path = SHARD_DIR):
    """Charge tous les shards en mémoire et retourne (graphs, metas).

    Pic mémoire ≈ taille totale du dataset (pas de torch.save serialisation),
    sensiblement plus léger que l'ancien `torch.load(graphs.pt)` qui devait
    désérialiser un blob unique de plusieurs GB.

    metas[i] = {map, drone, class_counts, n_nodes} pour le graphe i.
    """
    import torch
    shard_paths = sorted(shard_dir.glob("shard_*.pt"))
    if not shard_paths:
        raise FileNotFoundError(f"Aucun shard trouvé dans {shard_dir}")

    graphs: list = []
    metas:  list[dict] = []

    for p in shard_paths:
        chunk = torch.load(p, weights_only=False)
        for g in chunk:
            y_np   = g.y.numpy()
            valid  = y_np >= 0
            counts = (np.bincount(y_np[valid], minlength=NUM_CLASSES).tolist()
                      if valid.any() else [0] * NUM_CLASSES)
            metas.append({
                "map":          getattr(g, "map_name", "unknown"),
                "drone":        getattr(g, "drone_id", "unknown"),
                "class_counts": counts,
                "n_nodes":      int(g.num_nodes),
            })
            graphs.append(g)

    return graphs, metas
