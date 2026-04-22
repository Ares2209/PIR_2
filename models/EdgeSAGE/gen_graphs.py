#!/usr/bin/env python3
"""Build PyG graphs from NoiseMap PLY files for the GCN.

Node-level 7-class classification.
  nodes = triangle faces (only unknown-RGB faces are dropped)
  edges = faces sharing a triangle edge (manifold adjacency)
  y     = color class {0:violet, 1:blue, 2:yellow, 3:orange,
                       4:red, 5:dark_red, 6:occluded}

  x (10 features, ALL source-relative or scale-invariant):
    [0]  log10(dist + 1)           # distance log-normalisée [m]
    [1]  cos(normal, dir_to_src)   # visibilité angulaire [-1, 1]
    [2]  rel_centroid_x / dist     # direction normalisée source→face
    [3]  rel_centroid_y / dist
    [4]  rel_centroid_z / dist
    [5]  log10(height_agl + 1)     # hauteur face - point le plus bas du mesh [m]
    [6]  log10(area + 1)           # aire du triangle [m²]
    [7]  normal_z                  # orientation de la face (1=toit, -1=plafond, 0=mur)
    [8]  log10(horiz_dist + 1)     # distance horizontale (sol) source→face [m]
    [9]  occluded                  # 1 si un bâtiment bloque le rayon face→source, sinon 0
                                   # (ray-casting Möller-Trumbore; fortement corrélé au label c6)

  NOTE: on supprime les coordonnées absolues (fuite inter-villes).
        Les coordonnées relatives (dir_to_src) sont invariantes
        par translation → le modèle généralise entre villes.

  drone_feat (per-graph, shape (1, 51), normalisé):
    [n_blades_norm, rpm_min_norm, rpm_max_norm, LP_REF_norm(24), SLOPE_norm(24)]

Input : dataset/data/generated/{drone}/NoiseMap_<map>_<x>_<y>_<z>_<drone>.ply
Output: dataset/data/generated/processed/graphs.pt

Usage:
    python3 build_graphs.py
    python3 build_graphs.py --limit 20
    python3 build_graphs.py --workers 4
    python3 build_graphs.py --no-normalize   # désactive la normalisation (debug)
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import coalesce  # déduplique les arêtes

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
GENERATED  = SCRIPT_DIR
OUT_DIR    = GENERATED / "processed"
OUT_FILE   = OUT_DIR / "graphs.pt"

DRONES = ["F-4", "I2", "M2", "S-9"]

# ─────────────────────────────────────────────────────────────────────────────
# Drone acoustic signatures
# ─────────────────────────────────────────────────────────────────────────────
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
NUM_FEATURES   = 10  # voir header du fichier
DRONE_FEAT_DIM = 3 + N_BANDS + N_BANDS  # 51

# ─────────────────────────────────────────────────────────────────────────────
# Statistiques de normalisation (calculées offline sur le dataset complet,
# ou valeurs analytiques ci-dessous suffisantes pour bootstrapper)
# ─────────────────────────────────────────────────────────────────────────────

# Node features : (mean, std) pour chaque des 7 features
# Index : [log_dist, cos_ns, rel_x, rel_y, rel_z, log_height, log_area]
# Stats calculées sur le dataset complet par compute_node_stats.py
NODE_STATS = {
    "log_dist":       (1.2334,  0.2244),
    "cos_ns":         (-0.0256, 0.4906),
    "rel_x":          (0.0077,  0.6940),
    "rel_y":          (0.0087,  0.6962),
    "rel_z":          (-0.0385, 0.1788),
    "log_height":     (0.2712,  0.2985),
    "log_area":       (0.0280,  0.0197),
    "normal_z":       (0.4053,     0.6046),
    "log_horiz_dist": (1.22454,    0.2351),
    "occluded":       (0.7012, 0.4577),
}

# Drone features normalization ranges (min-max → [0,1])
DRONE_NORM = {
    "n_blades": (4.0, 6.0),
    "rpm":      (800.0, 7000.0),
    "lp_ref":   (16.0, 75.0),
    "slope":    (-1.3e-4, 4.5e-2),
}

# ─────────────────────────────────────────────────────────────────────────────
# Class mapping
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
NUM_CLASSES = 7

# Tolérance colorimétrique (certains logiciels arrondissent les RGB)
RGB_TOLERANCE = 8  # ±8 par canal

# ─────────────────────────────────────────────────────────────────────────────
# Filename pattern
# ─────────────────────────────────────────────────────────────────────────────
FNAME_RE = re.compile(
    r"^NoiseMap_(?P<map>.+?)_(?P<x>-?\d+(?:\.\d+)?)_(?P<y>-?\d+(?:\.\d+)?)_"
    r"(?P<z>-?\d+(?:\.\d+)?)_(?P<drone>[A-Za-z0-9-]+)\.ply$"
)

# ─────────────────────────────────────────────────────────────────────────────
# Drone acoustic signatures
# ─────────────────────────────────────────────────────────────────────────────

def _normalize_drone_vector(drone: str) -> np.ndarray:
    """
    Retourne le vecteur acoustique 51-dim NORMALISÉ en [0, 1] (float32).

    Normalisation min-max sur les plages connues de tous les drones.
    Cela évite que rpm (~3000) écrase n_blades (~4-6) ou slope (~1e-3).
    """
    n, rmin, rmax = META[drone]

    # n_blades : min=4, max=6
    n_norm    = (n    - DRONE_NORM["n_blades"][0]) / (
                 DRONE_NORM["n_blades"][1] - DRONE_NORM["n_blades"][0])
    rmin_norm = (rmin - DRONE_NORM["rpm"][0]) / (
                 DRONE_NORM["rpm"][1] - DRONE_NORM["rpm"][0])
    rmax_norm = (rmax - DRONE_NORM["rpm"][0]) / (
                 DRONE_NORM["rpm"][1] - DRONE_NORM["rpm"][0])

    lp  = np.array(LP_REF[drone], dtype=np.float32)
    lp_norm = (lp - DRONE_NORM["lp_ref"][0]) / (
               DRONE_NORM["lp_ref"][1] - DRONE_NORM["lp_ref"][0])

    sl  = np.array(SLOPE[drone], dtype=np.float32)
    sl_norm = (sl - DRONE_NORM["slope"][0]) / (
               DRONE_NORM["slope"][1] - DRONE_NORM["slope"][0])

    return np.concatenate(
        [[n_norm, rmin_norm, rmax_norm], lp_norm, sl_norm]
    ).astype(np.float32)

# ─────────────────────────────────────────────────────────────────────────────
# PLY parser
# ─────────────────────────────────────────────────────────────────────────────

def _parse_header(f) -> list[tuple[str, int, list[tuple[str, bool]]]]:
    """Parse PLY ASCII header; return list of (elem_name, count, props)."""
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
    """
    Parse an ASCII PLY file.

    Returns
    -------
    verts    : (N, 3) float32
    faces    : (M, 3) int64  — indices validés
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

    # ── Validation : indices de faces dans les bornes ─────────────────────────
    n_verts = verts.shape[0]
    valid_faces = (
        (faces[:, 0] >= 0) & (faces[:, 0] < n_verts) &
        (faces[:, 1] >= 0) & (faces[:, 1] < n_verts) &
        (faces[:, 2] >= 0) & (faces[:, 2] < n_verts)
    )
    n_invalid = int((~valid_faces).sum())
    if n_invalid > 0:
        warnings.warn(
            f"{path.name}: dropped {n_invalid} faces with out-of-bounds vertex indices"
        )
        faces    = faces[valid_faces]
        face_rgb = face_rgb[valid_faces]

    # ── Suppression des faces dégénérées (deux sommets identiques) ────────────
    degenerate = (
        (faces[:, 0] == faces[:, 1]) |
        (faces[:, 1] == faces[:, 2]) |
        (faces[:, 0] == faces[:, 2])
    )
    n_degen = int(degenerate.sum())
    if n_degen > 0:
        warnings.warn(
            f"{path.name}: dropped {n_degen} degenerate faces"
        )
        faces    = faces[~degenerate]
        face_rgb = face_rgb[~degenerate]

    return verts, faces, face_rgb

# ─────────────────────────────────────────────────────────────────────────────
# Graph helpers
# ─────────────────────────────────────────────────────────────────────────────

def _face_adjacency(faces: np.ndarray) -> np.ndarray:
    """
    Build symmetric face-adjacency edge_index (2, E), dédupliqué.

    Deux faces sont adjacentes ssi elles partagent une arête mesh (paire de
    sommets).  On utilise lexsort + comparaison consécutive : O(N log N).
    """
    n = faces.shape[0]
    if n == 0:
        return np.empty((2, 0), dtype=np.int64)

    e = np.concatenate([
        np.sort(faces[:, [0, 1]], axis=1),
        np.sort(faces[:, [1, 2]], axis=1),
        np.sort(faces[:, [0, 2]], axis=1),
    ], axis=0)                                        # (3N, 2)

    fid   = np.tile(np.arange(n, dtype=np.int64), 3)  # (3N,)
    order  = np.lexsort((e[:, 1], e[:, 0]))
    e_s    = e[order]
    fid_s  = fid[order]

    eq  = (e_s[1:, 0] == e_s[:-1, 0]) & (e_s[1:, 1] == e_s[:-1, 1])
    src = fid_s[:-1][eq]
    dst = fid_s[1:][eq]

    if src.size == 0:
        return np.empty((2, 0), dtype=np.int64)

    # Symétrique + dédoublonnage (cas de faces partagées en T-junction)
    s = np.concatenate([src, dst])
    d = np.concatenate([dst, src])

    # Suppression des self-loops éventuels
    no_self = s != d
    s, d = s[no_self], d[no_self]

    # Dédupliquation via tri
    pairs  = np.stack([s, d], axis=1)
    pairs  = np.unique(pairs, axis=0)
    return pairs.T.astype(np.int64)

def _classify_with_tolerance(rgb: np.ndarray) -> np.ndarray:
    """
    Map (N, 3) uint8 RGB → class indices; unknown → -1.

    Utilise une tolérance ±RGB_TOLERANCE par canal pour absorber les
    arrondis de rendu (antialiasing, compression PNG intermédiaire).
    """
    y = np.full(rgb.shape[0], -1, dtype=np.int64)
    rgb_f = rgb.astype(np.int16)  # évite l'overflow sur soustraction

    for (r, g, b), cls in RGB_TO_CLASS.items():
        mask = (
            (np.abs(rgb_f[:, 0] - r) <= RGB_TOLERANCE) &
            (np.abs(rgb_f[:, 1] - g) <= RGB_TOLERANCE) &
            (np.abs(rgb_f[:, 2] - b) <= RGB_TOLERANCE)
        )
        # En cas de conflit (deux classes dans la tolérance), la dernière gagne.
        # C'est rare ; on émet un avertissement si nécessaire.
        y[mask] = cls

    return y

# ─────────────────────────────────────────────────────────────────────────────
# Normalisation des features nœuds
# ─────────────────────────────────────────────────────────────────────────────

def _normalize_node_features(feats: np.ndarray) -> np.ndarray:
    """
    Standardisation (z-score) des 5 features nœuds avec les stats globales.

    Index :
      0  log_dist     1  cos_ns
      2  rel_x        3  rel_y     4  rel_z
    """
    s = NODE_STATS
    means = np.array([
        s["log_dist"][0], s["cos_ns"][0],
        s["rel_x"][0],    s["rel_y"][0],    s["rel_z"][0],
        s["log_height"][0], s["log_area"][0],
        s["normal_z"][0], s["log_horiz_dist"][0],
        s["occluded"][0],
    ], dtype=np.float32)

    stds = np.array([
        s["log_dist"][1], s["cos_ns"][1],
        s["rel_x"][1],    s["rel_y"][1],    s["rel_z"][1],
        s["log_height"][1], s["log_area"][1],
        s["normal_z"][1], s["log_horiz_dist"][1],
        s["occluded"][1],
    ], dtype=np.float32)

    return ((feats - means) / stds).astype(np.float32)

# ─────────────────────────────────────────────────────────────────────────────
# Ray-cast occlusion — BVH via trimesh sur le mesh de base (dataset/blender/).
#
# Justification : l'occlusion ne dépend que de la géométrie de la ville et de
# la position source. Pas besoin de re-lancer le ray-casting pour chaque
# combinaison (drone × position) — on charge une fois les 6 PLY de base et
# on re-utilise leur BVH pour les 2000 NoiseMap_*.ply.
# ─────────────────────────────────────────────────────────────────────────────

BLENDER_DIR = Path(__file__).resolve().parents[2] / "dataset" / "blender"
_BASE_MESH_CACHE: dict[str, "object"] = {}  # map_name -> trimesh.Trimesh (lazy)


def _load_base_mesh(map_name: str):
    """Charge (avec cache) le mesh de base depuis dataset/blender/<map_name>.ply."""
    if map_name in _BASE_MESH_CACHE:
        return _BASE_MESH_CACHE[map_name]
    import trimesh  # import local : évite la dépendance dure au module
    ply = BLENDER_DIR / f"{map_name}.ply"
    if not ply.exists():
        warnings.warn(
            f"base mesh for '{map_name}' not found at {ply} — "
            f"occlusion feature will be all zeros"
        )
        _BASE_MESH_CACHE[map_name] = None
        return None
    mesh = trimesh.load(str(ply), process=False, force="mesh")
    # Purge explicite des triangles (quasi-)dégénérés : ils causent des
    # divide-by-zero dans triangles.py:550 (barycentric NaN) pendant les
    # intersections. process=True ne suffit pas pour les triangles colinéaires
    # à aire très petite mais non nulle.
    areas = mesh.area_faces
    keep  = areas > 1e-8
    if not keep.all():
        mesh.update_faces(keep)
        mesh.remove_unreferenced_vertices()
    # Force la construction du BVH tout de suite (sinon 1er appel lent)
    _ = mesh.ray.intersects_any(
        ray_origins=np.array([[0.0, 0.0, 1e6]]),
        ray_directions=np.array([[0.0, 0.0, -1.0]]),
    )
    _BASE_MESH_CACHE[map_name] = mesh
    return mesh


_OCC_CHUNK = 4096  # taille de batch de rayons pour maîtriser la RAM


def _compute_occlusion(
    centroid: np.ndarray,
    src: np.ndarray,
    map_name: str,
) -> np.ndarray:
    """
    Pour chaque face (centroïde donné), teste si le rayon centroid→src est
    bloqué par la géométrie de la ville (mesh de base trimesh + BVH).

    Les rayons sont traités par chunks pour que l'empreinte mémoire de
    `intersects_location` (proportionnelle à n_rayons) reste bornée.

    Retourne (M,) uint8. Si le mesh de base n'est pas trouvé, renvoie des 0.
    """
    mesh = _load_base_mesh(map_name)
    M = centroid.shape[0]
    occ = np.zeros(M, dtype=np.uint8)
    if mesh is None or M == 0:
        return occ

    vec  = src[None, :].astype(np.float64) - centroid.astype(np.float64)
    dist = np.linalg.norm(vec, axis=1)
    safe = np.maximum(dist, 1e-9)
    dirs = vec / safe[:, None]
    eps  = 1e-3
    # Décale l'origine vers la source pour éviter l'auto-intersection avec la
    # face d'origine (quasi-coplanaire à celle du mesh de base).
    origins = centroid.astype(np.float64) + dirs * eps

    # errstate : supprime les RuntimeWarning divide/invalid émis par
    # trimesh.triangles.points_to_barycentric si un triangle résiduel est
    # quasi-dégénéré. Un NaN dans hit_dist donne mask=False donc pas de
    # fausse occlusion.
    with np.errstate(divide="ignore", invalid="ignore"):
        for start in range(0, M, _OCC_CHUNK):
            end = min(start + _OCC_CHUNK, M)
            hits, ray_idx, _ = mesh.ray.intersects_location(
                ray_origins=origins[start:end],
                ray_directions=dirs[start:end],
                multiple_hits=False,
            )
            if len(ray_idx) == 0:
                continue
            hit_dist = np.linalg.norm(hits - origins[start:end][ray_idx], axis=1)
            mask = hit_dist < (dist[start:end][ray_idx] - 2.0 * eps)
            occ[start + ray_idx[mask]] = 1
    return occ

# ─────────────────────────────────────────────────────────────────────────────
# Feature builder (partagé entre génération du dataset et inférence)
# ─────────────────────────────────────────────────────────────────────────────

def build_node_features(
    verts: np.ndarray,
    faces: np.ndarray,
    src: np.ndarray,
    drone: str,
    normalize: bool = True,
    map_name: str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build per-face node features + drone acoustic vector.

    Source unique de vérité pour la construction des 5 features du GCN ;
    utilisée par `_build_one` (dataset) et `infer.py` (inférence).

    Returns
    -------
    feats      : (M, 5) float32   — log_dist, cos_ns, rel_dir (x,y,z)
    drone_feat : (1, 51) float32  — signature acoustique normalisée
    centroid   : (M, 3) float32   — centroïdes absolus (utile pour plots)
    """
    
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    
    area = (0.5 * np.linalg.norm(
    np.cross(v1 - v0, v2 - v0), axis=1
    )).astype(np.float32)
    height_agl = (v0[:, 2] - v0[:, 2].min()).astype(np.float32)

    log_area = np.log10(area + 1.0)[:, None]
    log_height = np.log10(height_agl + 1.0)[:, None]
    
    centroid = ((v0 + v1 + v2) / 3.0).astype(np.float32)

    cross  = np.cross(v1 - v0, v2 - v0).astype(np.float32)
    norm2  = np.linalg.norm(cross, axis=1)
    safe_n = np.where(norm2 > 1e-12, norm2, 1.0)
    normal = (cross / safe_n[:, None]).astype(np.float32)

    vec        = (src[None, :] - centroid)
    dist_raw   = np.linalg.norm(vec, axis=1)
    dist       = np.where(dist_raw > 1e-6, dist_raw, 1e-6).astype(np.float32)
    dir_to_src = (vec / dist[:, None]).astype(np.float32)

    cos_ns = np.einsum("ij,ij->i", normal, dir_to_src).astype(np.float32)

    normal_z = normal[:, 2:3].astype(np.float32)          # (M, 1)
    horiz_dist = np.sqrt(vec[:, 0] ** 2 + vec[:, 1] ** 2).astype(np.float32)
    log_horiz_dist = np.log10(horiz_dist + 1.0)[:, None]  # (M, 1)

    if map_name is not None:
        occluded = _compute_occlusion(centroid, src, map_name)[:, None].astype(np.float32)
    else:
        occluded = np.zeros((centroid.shape[0], 1), dtype=np.float32)

    feats = np.concatenate([
        np.log10(dist + 1.0)[:, None],       # [0]  log_dist       (M, 1)
        cos_ns[:, None],                      # [1]  cos_ns         (M, 1)
        dir_to_src,                           # [2:5] rel_dir       (M, 3)
        log_height,                           # [5]  log_height     (M, 1)
        log_area,                             # [6]  log_area       (M, 1)
        normal_z,                             # [7]  normal_z       (M, 1)
        log_horiz_dist,                       # [8]  log_horiz_dist (M, 1)
        occluded,                             # [9]  occluded       (M, 1)
    ], axis=1).astype(np.float32)

    assert feats.shape[1] == NUM_FEATURES, (
        f"Feature dim mismatch: {feats.shape[1]} vs {NUM_FEATURES}"
    )

    if normalize:
        feats = _normalize_node_features(feats)

    drone_feat = _normalize_drone_vector(drone)[None, :]

    return feats, drone_feat, centroid

# ─────────────────────────────────────────────────────────────────────────────
# Core graph builder  (runs in worker process)
# ─────────────────────────────────────────────────────────────────────────────

def _build_one(args_tuple: tuple[str, bool]):
    """
    Build a single PyG Data object from one PLY file.

    Parameters
    ----------
    args_tuple : (path_str, normalize)

    Returns
    -------
    (feats_np, ei_np, y_np, drone_feat_np, meta) on success
    (None, error_message)                          on skip/failure
    """
    path_str, normalize = args_tuple
    path = Path(path_str)

    # ── Parse filename ────────────────────────────────────────────────────────
    fname_match = FNAME_RE.match(path.name)
    if not fname_match:
        return None, f"skip (bad name): {path.name}"

    drone    = fname_match.group("drone")
    map_name = fname_match.group("map")
    src      = np.array(
        [float(fname_match.group("x")),
         float(fname_match.group("y")),
         float(fname_match.group("z"))],
        dtype=np.float32,
    )

    if drone not in META:
        return None, f"skip (unknown drone '{drone}'): {path.name}"

    # ── Load geometry + colors ────────────────────────────────────────────────
    try:
        verts, faces, face_rgb = _parse_ply(path)
    except Exception as exc:
        return None, f"skip (parse error: {exc}): {path.name}"

    if faces.shape[0] == 0:
        return None, f"skip (no faces): {path.name}"

    # ── Labels ────────────────────────────────────────────────────────────────
    y_full = _classify_with_tolerance(face_rgb)
    keep   = y_full >= 0
    n_kept = int(keep.sum())
    n_total = len(y_full)

    if n_kept == 0:
        return None, f"skip (no valid face colours): {path.name}"

    # Avertissement si trop de faces non reconnues (>10%)
    n_dropped = n_total - n_kept
    if n_dropped > 0.10 * n_total:
        warnings.warn(
            f"{path.name}: {n_dropped}/{n_total} faces ont une couleur "
            f"non reconnue ({100*n_dropped/n_total:.1f}%)"
        )

    # ── Per-face geometry + feature matrix ───────────────────────────────────
    feats, drone_feat, _ = build_node_features(
        verts, faces, src, drone, normalize=normalize, map_name=map_name,
    )

    # ── Edge index (face adjacency) ───────────────────────────────────────────
    ei_full = _face_adjacency(faces)

    # Re-index to kept faces only
    new_idx = np.full(faces.shape[0], -1, dtype=np.int64)
    new_idx[keep] = np.arange(n_kept, dtype=np.int64)

    if ei_full.shape[1] > 0:
        edge_keep = (
            (new_idx[ei_full[0]] >= 0) &
            (new_idx[ei_full[1]] >= 0)
        )
        ei = np.stack(
            [new_idx[ei_full[0][edge_keep]],
             new_idx[ei_full[1][edge_keep]]],
            axis=0,
        )
    else:
        ei = np.empty((2, 0), dtype=np.int64)

    # ── Distribution des classes (utile pour pondération de la loss) ──────────
    y_kept   = y_full[keep]
    class_counts = np.bincount(y_kept, minlength=NUM_CLASSES)

    meta = {
        "path":          str(path),
        "drone":         drone,
        "map":           map_name,
        "source":        src.tolist(),
        "n_nodes":       n_kept,
        "n_edges":       ei.shape[1],
        "class_counts":  class_counts.tolist(),
    }

    return (feats[keep], ei, y_kept, drone_feat, meta), None

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build PyG graphs from NoiseMap PLY files."
    )
    ap.add_argument(
        "--workers", type=int,
        default=max(1, (os.cpu_count() or 2) - 1),
        help="Nombre de processus parallèles.",
    )
    ap.add_argument(
        "--limit", type=int, default=0,
        help="Traite seulement N fichiers par drone (debug).",
    )
    ap.add_argument(
        "--out", type=Path, default=OUT_FILE,
        help="Fichier .pt de sortie.",
    )
    ap.add_argument(
        "--no-normalize", action="store_true",
        help="Désactive la normalisation des features (debug).",
    )
    ap.add_argument(
        "--seed", type=int, default=42,
        help="Graine aléatoire pour la reproductibilité.",
    )
    args = ap.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    normalize = not args.no_normalize

    # ── Collect PLY files ─────────────────────────────────────────────────────
    files: list[Path] = []
    for drone in DRONES:
        drone_dir = GENERATED / drone
        if not drone_dir.is_dir():
            print(f"WARN: missing drone dir {drone_dir}", file=sys.stderr)
            continue
        plys = sorted(drone_dir.glob("NoiseMap_*.ply"))
        if args.limit:
            plys = plys[: args.limit]
        files.extend(plys)

    if not files:
        print("No PLY files found.", file=sys.stderr)
        sys.exit(1)

    print(
        f"Building graphs from {len(files)} files "
        f"with {args.workers} workers "
        f"(normalize={normalize})…"
    )
    t0 = time.perf_counter()

    graphs:  list[Data] = []
    metas:   list[dict] = []
    skipped = 0

    # ── Parallel processing ───────────────────────────────────────────────────
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        job_args  = [(str(p), normalize) for p in files]
        futures   = {
            executor.submit(_build_one, a): p
            for a, p in zip(job_args, files)
        }

        for i, fut in enumerate(as_completed(futures), 1):
            try:
                out, err = fut.result()
            except Exception as exc:
                skipped += 1
                print(
                    f"  ERROR [{futures[fut].name}]: {exc}",
                    file=sys.stderr,
                )
                continue

            if out is None:
                skipped += 1
                if err:
                    print(f"  {err}", file=sys.stderr)
            else:
                feats_np, ei_np, y_np, drone_feat_np, meta = out

                # Construction de l'objet PyG Data
                ei_tensor = torch.from_numpy(ei_np).long()

                # Dédupliquation finale via torch_geometric (sécurité)
                if ei_tensor.shape[1] > 0:
                    ei_tensor = coalesce(
                        ei_tensor,
                        num_nodes=int(feats_np.shape[0]),
                    )

                data = Data(
                    x          = torch.from_numpy(feats_np),
                    edge_index = ei_tensor,
                    y          = torch.from_numpy(y_np).long(),
                    drone_feat = torch.from_numpy(drone_feat_np),
                )
                data.num_nodes = int(data.x.shape[0])
                graphs.append(data)
                metas.append(meta)

            if i % 100 == 0 or i == len(futures):
                dt = time.perf_counter() - t0
                print(
                    f"  {i}/{len(futures)}  "
                    f"kept={len(graphs)}  skipped={skipped}  "
                    f"({dt:.1f}s)"
                )

    if not graphs:
        print("No graphs built — aborting.", file=sys.stderr)
        sys.exit(1)

    # ── Stats globales ────────────────────────────────────────────────────────
    total_nodes = sum(g.num_nodes for g in graphs)
    total_edges = sum(g.edge_index.shape[1] for g in graphs)

    # Distribution des classes sur le dataset complet
    all_counts     = np.array([m["class_counts"] for m in metas]).sum(axis=0)
    class_names    = [
        "violet", "blue", "yellow", "orange", "red", "dark_red", "occluded"
    ]

    print(f"\n{'─'*60}")
    print(
        f"Total : {len(graphs)} graphs | "
        f"{total_nodes:,} nodes | "
        f"{total_edges:,} directed edges"
    )
    print("\nDistribution des classes (tous graphes) :")
    for i, (name, cnt) in enumerate(zip(class_names, all_counts)):
        pct = 100.0 * cnt / max(all_counts.sum(), 1)
        bar = "█" * int(pct / 2)
        print(f"  [{i}] {name:<10} {cnt:>10,}  ({pct:5.1f}%)  {bar}")

    # Statistiques par ville (pour détecter le déséquilibre)
    from collections import defaultdict
    per_map: dict[str, list] = defaultdict(list)
    for m in metas:
        per_map[m["map"]].append(m["n_nodes"])

    print(f"\nNœuds par ville :")
    for map_name, nodes_list in sorted(per_map.items()):
        total = sum(nodes_list)
        print(
            f"  {map_name:<20}  {len(nodes_list):>5} graphes  "
            f"{total:>10,} nœuds  "
            f"(moy {total//len(nodes_list):,}/graphe)"
        )

    print(f"{'─'*60}\n")

    # ── Save ──────────────────────────────────────────────────────────────────
    args.out.parent.mkdir(parents=True, exist_ok=True)
    blob = {
        "graphs":             graphs,
        "num_classes":        NUM_CLASSES,
        "num_node_features":  NUM_FEATURES,
        "num_drone_features": DRONE_FEAT_DIM,
        "normalize":          normalize,
        "node_stats":         NODE_STATS,       # <-- pour inférence
        "drone_norm":         DRONE_NORM,       # <-- pour inférence
        "meta":               metas,
    }
    torch.save(blob, args.out)
    print(f"Wrote → {args.out}")

if __name__ == "__main__":
    main()
