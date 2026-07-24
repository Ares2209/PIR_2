"""Hyperparameters and canonical constants for the GAT pipeline.

This file is import-side-effect-light: it defines constants only. PLY parsing,
feature normalisation, dataset loading and other helpers live in `dataio.py`;
training plumbing lives in `core.py`.
"""

from __future__ import annotations

import json
import re
import warnings
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# Model architecture
# ─────────────────────────────────────────────────────────────────────────────
HIDDEN_CHANNELS = 128
NUM_LAYERS      = 6
NUM_HEADS       = 5
ATTN_DROPOUT    = 0.2858666834614854
DROPOUT         = 0.2

# ─────────────────────────────────────────────────────────────────────────────
# Optimization
# ─────────────────────────────────────────────────────────────────────────────
LR               = 0.0005407835674001513
WEIGHT_DECAY     = 5.024783335894379e-06
EPOCHS           = 600
BATCH_SIZE       = 4
GRAD_CLIP        = 1.0
GRAD_ACCUM_STEPS = 4

VAL_RATIO  = 0.15
TEST_RATIO = 0.15
SPLIT_SEED = 0
SEED       = 42

# ── Evaluation split ─────────────────────────────────────────────────────────
# Split strategies, tried in this order when SPLIT_FROM_CITY_FILE is True:
#
#   1. City-roles file (city_split.json, copied by gen_graphs.py from the
#      city_roles.json that generate_dataset.py writes): the eval cities — the
#      "normal condition" cities generated with uniform-z and a small number of
#      exercises — become the VALIDATION set; the train cities become the TRAIN
#      set. No separate test set: validation is the final evaluation (whole
#      cities never trained on → geometric generalisation).
#
#   2. Fallback by exercise count (file absent): the N_EVAL_CITIES cities with
#      the fewest graphs are taken as validation, the rest as train.
#
#   3. Last resort (SPLIT_FROM_CITY_FILE=False, or too few cities): random
#      whole-city hold-out of N_TEST_CITIES for the test; validation carved
#      graph-level from the rest (the historical behaviour).
SPLIT_FROM_CITY_FILE = True   # CITY_SPLIT_FILE path is defined next to SHARD_DIR
N_EVAL_CITIES        = 4       # fallback-by-count: how many cities → validation

# The final test runs on whole cities the model never saw during training
# (true geometric generalisation). With 10 cities → 7 train / 3 test by default.
# Validation (early-stopping / LR scheduler) is carved out graph-level from the
# train cities, so the 7 train cities all keep contributing and the test cities
# stay completely clean.
N_TEST_CITIES = 3
# The drone-source altitude Z is extremely skewed (most points sit in [0.2, 1.3]).
# An unbalanced test would mostly report low-altitude performance. We subsample
# the test graphs into fixed-width Z bins and cap each bin so the test set has a
# roughly uniform altitude distribution — every Z range weighs ~equally.
EVAL_Z_BALANCE = True
EVAL_Z_BINS    = 8        # number of fixed-width Z bins over the test Z range
EVAL_Z_PER_BIN = 40     # max graphs per bin; None → median non-empty bin size

EVAL_EVERY  = 5
PATIENCE    = 50
MIN_DELTA   = 1e-4

LR_FACTOR   = 0.5
LR_PATIENCE = 10
LR_MIN      = 1e-6

KEEP_OLD_CHECKPOINTS = 3

# Per-class CE weights (violet, blue, yellow, orange, red, dark_red, occluded).
# Set to None to use auto-computed inverse-sqrt weights.
CLASS_WEIGHTS = [0.6581047128250365, 0.3402403449139341, 0.20343783507673832, 1.5696092757007185, 4.048243580680749, 5.7084092327128335, 0.6898965029646703]

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
SCRIPT_DIR  = Path(__file__).resolve().parent
REPO_ROOT   = SCRIPT_DIR.parents[1]
GENERATED   = REPO_ROOT / "dataset" / "data" / "generated"
OUT_DIR     = GENERATED / "processed"
SHARD_DIR   = OUT_DIR / "shards"
STATS_FILE  = OUT_DIR / "node_stats.json"
CITY_SPLIT_FILE = OUT_DIR / "city_split.json"   # eval/train city roles (gen_graphs)


# ─────────────────────────────────────────────────────────────────────────────
# Drones — acoustic signatures
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

DRONE_NORM = {
    "n_blades": (4.0, 6.0),
    "rpm":      (800.0, 7000.0),
    "lp_ref":   (16.0, 75.0),
    "slope":    (-1.3e-4, 4.5e-2),
}


# ─────────────────────────────────────────────────────────────────────────────
# Node feature stats (z-score). Overridden at import time by node_stats.json
# if it exists (computed by gen_graphs.py during dataset build).
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
    "cos_angles":          (0.4793,  0.5988),
    "grazing_angle":       (0.1860,  0.3684),
    "elevation_angle":     (0.9263,  0.0335),
    "obstacle_proximity":  (0.2643,  0.2055),
    "slope_discontinuity": (0.1996,  0.2058),
    "normal_x":            (0.0,     0.5),
    "normal_y":            (0.0,     0.5),
    "cos_horiz":           (0.0,     0.5),
    "is_occluded":         (0.3,     0.5),
    "first_obstacle_frac": (0.7,     0.3),
    "log_n_intersections": (0.4,     0.6),
    "log_dist_to_lit":     (0.5,     0.5),
    "log_n_lit_nearby":    (2.0,     1.5),
}

# Radius (mesh units) for the lit-neighbour ball query used by the
# reflection-proxy features. Mesh scale is 1 unit = 100 m, so 0.05 ≈ 5 m,
# the rough range over which 1-bounce reflection still contributes
# meaningfully in dense urban acoustics.
REFLECT_RADIUS = 0.05


# ─────────────────────────────────────────────────────────────────────────────
# Feature vector dimensions + canonical key order
# ─────────────────────────────────────────────────────────────────────────────
N_BANDS        = 24
NUM_FEATURES   = 22   # reflect_score retiré (faible importance SHAP+GNNExplainer)
DRONE_FEAT_DIM = 3 + N_BANDS + N_BANDS  # = 51

# reflect_score (ancien index 22) a été retiré. C'était la DERNIÈRE colonne, donc
# les shards générés en 23-features restent compatibles : dataio.load_sharded_dataset
# tronque la colonne excédentaire au chargement (pas besoin de régénérer le dataset).
FEAT_KEYS = [
    "log_dist", "cos_ns", "rel_x", "rel_y", "rel_z",
    "log_height", "log_area", "normal_z", "log_horiz_dist",
    "cos_angles", "grazing_angle", "elevation_angle",
    "obstacle_proximity", "slope_discontinuity",
    "normal_x", "normal_y", "cos_horiz",
    "is_occluded", "first_obstacle_frac", "log_n_intersections",
    "log_dist_to_lit", "log_n_lit_nearby",
]
assert len(FEAT_KEYS) == NUM_FEATURES, "FEAT_KEYS doit avoir NUM_FEATURES entrées"


def _load_stats_from_json() -> None:
    """If STATS_FILE exists, overwrite NODE_STATS with the values measured by
    gen_graphs.py during the dataset build. Lets infer.py use the right stats
    without code changes."""
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
# Class ↔ RGB mapping
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
# Filename pattern for NoiseMap_*.ply
# ─────────────────────────────────────────────────────────────────────────────
FNAME_RE = re.compile(
    r"^NoiseMap_"
    r"(?P<map>[^_]+(?:_[^_]+)*?)_"   # map_name : up to the three floats
    r"(?P<x>[+-]?\d+(?:\.\d+)?)_"
    r"(?P<y>[+-]?\d+(?:\.\d+)?)_"
    r"(?P<z>[+-]?\d+(?:\.\d+)?)_"
    r"(?P<drone>.+)"                  # drone_id : everything else (may contain _)
    r"\.ply$"
)
