"""Data I/O and feature engineering for the GAT pipeline.

Consolidates what used to live in four separate modules:
  - config.py        → PLY parser, face adjacency, RGB classifier, feature
                       normalisers (these were misfiled in `config.py`)
  - data.py          → dataset loading + map-level split + class weights +
                       DataLoader factory
  - plotting.py      → per-epoch history plots
  - gen_graphs.py    → `_build_node_features` (re-used by `infer.py`)
"""

from __future__ import annotations

import json
import math
import shutil
import time
import warnings
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data.distributed import DistributedSampler
from torch_geometric.loader import DataLoader

from models.GATv2.config import (
    CITY_SPLIT_FILE,
    CLASS_WEIGHTS,
    DRONE_NORM,
    EVAL_Z_BALANCE,
    EVAL_Z_BINS,
    EVAL_Z_PER_BIN,
    FEAT_KEYS,
    LP_REF,
    META,
    N_EVAL_CITIES,
    N_TEST_CITIES,
    NODE_STATS,
    NUM_CLASSES,
    NUM_FEATURES,
    REFLECT_RADIUS,
    RGB_TO_CLASS,
    RGB_TOLERANCE,
    SHARD_DIR,
    SLOPE,
    SPLIT_FROM_CITY_FILE,
)


# ═════════════════════════════════════════════════════════════════════════════
# PLY parser (ASCII only)
# ═════════════════════════════════════════════════════════════════════════════
def _parse_header(f) -> list[tuple[str, int, list[tuple[str, bool]]]]:
    """Parse a PLY ASCII header; returns [(elem_name, count, props), …]."""
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
    """Parse an ASCII PLY. Returns (verts, faces, face_rgb).

    verts    : (N, 3) float32
    faces    : (M, 3) int64   — validated indices
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


# ═════════════════════════════════════════════════════════════════════════════
# Face adjacency
# ═════════════════════════════════════════════════════════════════════════════
def _face_adjacency(faces: np.ndarray) -> np.ndarray:
    """Build a deduplicated, symmetric edge_index (2, E) between faces that
    share an edge."""
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


# ═════════════════════════════════════════════════════════════════════════════
# RGB → class (tolerant)
# ═════════════════════════════════════════════════════════════════════════════
def _classify_with_tolerance(rgb: np.ndarray) -> np.ndarray:
    """Map (N, 3) uint8 RGB → class; unknown → -1.

    Per-channel tolerance of ±RGB_TOLERANCE absorbs renderer rounding errors.
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


# ═════════════════════════════════════════════════════════════════════════════
# Feature normalisation
# ═════════════════════════════════════════════════════════════════════════════
def _normalize_drone_vector(drone: str) -> np.ndarray:
    """Return the 51-dim acoustic signature, normalised to [0, 1] (float32)."""
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


def _normalize_node_features(feats: np.ndarray) -> np.ndarray:
    """Z-score node features using global NODE_STATS."""
    means = np.array([NODE_STATS[k][0] for k in FEAT_KEYS], dtype=np.float32)
    stds  = np.array([NODE_STATS[k][1] for k in FEAT_KEYS], dtype=np.float32)
    stds  = np.where(stds > 1e-8, stds, 1.0)   # guardrail
    return ((feats - means) / stds).astype(np.float32)


# ═════════════════════════════════════════════════════════════════════════════
# Node feature builder (shared by gen_graphs.py and infer.py)
# ═════════════════════════════════════════════════════════════════════════════
def _ray_occlusion_features(
    verts:     np.ndarray,   # (N, 3) float32
    faces:     np.ndarray,   # (M, 3) int64
    centroids: np.ndarray,   # (M, 3) float32 — face centroids
    drone_pos: np.ndarray,   # (3,)   float32
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cast one ray per face (centroid → drone) and return three occlusion
    features. Uses trimesh's Embree backend, so cost is dominated by mesh
    construction; the ray cast itself is vectorised over all faces.

    Returns:
        is_occluded         (M,) float32 in {0, 1}
        first_obstacle_frac (M,) float32 in [0, 1]
                            fraction of (centroid→drone) distance to the first
                            blocking hit; 1.0 when the line of sight is clear.
        log_n_inter         (M,) float32 = log1p(#blocking hits along the ray)
    """
    import trimesh

    M = faces.shape[0]
    is_occluded = np.zeros(M, dtype=np.float32)
    first_obstacle_frac = np.ones(M, dtype=np.float32)
    log_n_inter = np.zeros(M, dtype=np.float32)
    if M == 0:
        return is_occluded, first_obstacle_frac, log_n_inter

    diff       = drone_pos[np.newaxis, :] - centroids        # (M, 3)
    total_dist = np.linalg.norm(diff, axis=1)                # (M,)
    total_dist = np.clip(total_dist, 1e-3, None)
    dirs       = (diff / total_dist[:, np.newaxis]).astype(np.float32)

    # Offset origin a bit along the ray to avoid hitting the source face itself.
    EPS = 1e-3
    origins = (centroids + EPS * dirs).astype(np.float32)

    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)

    locations, index_ray, _ = mesh.ray.intersects_location(
        ray_origins=origins,
        ray_directions=dirs,
        multiple_hits=True,
    )
    if locations.shape[0] == 0:
        return is_occluded, first_obstacle_frac, log_n_inter

    hit_dist  = np.linalg.norm(locations - origins[index_ray], axis=1)
    # Drop hits that lie past the drone (the drone is the endpoint of the ray).
    keep = hit_dist < total_dist[index_ray]
    idx_ray = index_ray[keep]
    hit_dist = hit_dist[keep]
    if idx_ray.size == 0:
        return is_occluded, first_obstacle_frac, log_n_inter

    counts = np.bincount(idx_ray, minlength=M).astype(np.float32)
    log_n_inter = np.log1p(counts).astype(np.float32)
    is_occluded = (counts > 0).astype(np.float32)

    min_dist = np.full(M, np.inf, dtype=np.float64)
    np.minimum.at(min_dist, idx_ray, hit_dist)
    hit = np.isfinite(min_dist)
    first_obstacle_frac[hit] = (
        min_dist[hit] / total_dist[hit]
    ).astype(np.float32)

    return is_occluded, first_obstacle_frac, log_n_inter


def _reflection_proxy_features(
    centroids:   np.ndarray,   # (M, 3) float32
    normals:     np.ndarray,   # (M, 3) float32 — unit normals
    is_occluded: np.ndarray,   # (M,)   float32 in {0, 1}
    radius:      float = REFLECT_RADIUS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Geometric proxies for 'is this face likely reached by reflected sound'.

    No ray-tracing is performed. The only inputs are the LOS mask already
    computed by `_ray_occlusion_features` plus mesh geometry, so the cost is
    dominated by a single KDTree build per mesh (negligible vs. ray casts).

    The idea: every face with direct LOS is a *candidate reflector*. A face
    in shadow that sits close to many well-oriented reflectors is likely
    reached by 1-bounce reflected sound; one deep in shadow probably isn't.

    Returns:
        log_dist_to_lit  (M,) float32 = log1p(distance to nearest lit face)
        log_n_lit_nearby (M,) float32 = log1p(# lit faces within `radius`)
        reflect_score    (M,) float32 in [0, 1] = mean cos(n_lit, lit→face)
                         over nearby lit faces — high when nearby lit
                         normals point toward this face.
    """
    from scipy.spatial import cKDTree

    M = centroids.shape[0]
    log_dist_to_lit  = np.zeros(M, dtype=np.float32)
    log_n_lit_nearby = np.zeros(M, dtype=np.float32)
    reflect_score    = np.zeros(M, dtype=np.float32)
    if M == 0:
        return log_dist_to_lit, log_n_lit_nearby, reflect_score

    lit_mask = is_occluded < 0.5
    if not lit_mask.any():
        # No lit face in the whole mesh — every face is in shadow. Leave the
        # zero defaults (post-z-score they collapse to a constant the model
        # can detect).
        return log_dist_to_lit, log_n_lit_nearby, reflect_score

    lit_centroids = centroids[lit_mask]
    lit_normals   = normals[lit_mask]
    tree = cKDTree(lit_centroids)

    dist_to_lit, _ = tree.query(centroids, k=1)
    log_dist_to_lit = np.log1p(dist_to_lit).astype(np.float32)

    neighbours = tree.query_ball_point(centroids, r=radius)

    for i, idx in enumerate(neighbours):
        if not idx:
            continue
        # log1p(count) — proxy for reflector density
        log_n_lit_nearby[i] = math.log1p(len(idx))

        # Orientation quality: a lit face whose normal points toward face i
        # is a plausible specular reflector. Skip self-matches (r ≈ 0).
        to_me = centroids[i] - lit_centroids[idx]
        r     = np.linalg.norm(to_me, axis=1)
        keep  = r > 1e-6
        if not keep.any():
            continue
        to_me = to_me[keep] / r[keep, np.newaxis]
        cos_r = (lit_normals[idx][keep] * to_me).sum(axis=1)
        reflect_score[i] = float(np.clip(cos_r, 0.0, 1.0).mean())

    return log_dist_to_lit, log_n_lit_nearby, reflect_score.astype(np.float32)


def _build_node_features(
    verts:    np.ndarray,   # (N, 3) float32 — original mesh vertices
    faces:    np.ndarray,   # (M, 3) int64   — vertex indices
    drone_pos: np.ndarray,  # (3,)   float32
    map_name: str,
) -> np.ndarray:
    """
    Build the (M, NUM_FEATURES) feature matrix for the faces of one graph.

    Canonical column order (must match `config.FEAT_KEYS`):
      0  log_dist           log distance to drone
      1  cos_ns             normal-source cosine
      2  rel_x              normalised relative X
      3  rel_y              normalised relative Y
      4  rel_z              normalised relative Z
      5  log_height         log(centroid height)
      6  log_area           log(face area)
      7  normal_z           Z normal component
      8  log_horiz_dist     log(horizontal distance)
      9  cos_angles         3D incidence cosine (= cos_ns)
     10  grazing_angle      normalised grazing angle
     11  elevation_angle    normalised elevation angle
     12  obstacle_proximity inverse normalised distance
     13  slope_discontinuity local slope discontinuity
     14  normal_x           X normal component
     15  normal_y           Y normal component
     16  cos_horiz          XY-plane normal-source cosine
     17  is_occluded        ray centroid→drone hits any face (0/1)
     18  first_obstacle_frac fraction of LOS distance to first hit (1=clear)
     19  log_n_intersections log1p of #blocking hits along the LOS
     20  log_dist_to_lit    log1p distance to nearest lit (LOS-direct) face
     21  log_n_lit_nearby   log1p count of lit faces within REFLECT_RADIUS
    """
    M = faces.shape[0]
    if M == 0:
        return np.empty((0, NUM_FEATURES), dtype=np.float32)

    # ── Centroids and face normals ───────────────────────────────────────────
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    centroids = (v0 + v1 + v2) / 3.0   # (M, 3)

    e1 = v1 - v0
    e2 = v2 - v0
    raw_normals = np.cross(e1, e2)      # (M, 3)
    norms       = np.linalg.norm(raw_normals, axis=1, keepdims=True)
    norms       = np.where(norms < 1e-8, 1.0, norms)   # avoid /0
    normals     = raw_normals / norms   # (M, 3) unitaires

    # ── Areas ─────────────────────────────────────────────────────────────────
    areas = np.linalg.norm(raw_normals, axis=1) * 0.5   # (M,)
    areas = np.clip(areas, 1e-6, None)

    # ── drone → face vectors ──────────────────────────────────────────────────
    diff      = centroids - drone_pos[np.newaxis, :]    # (M, 3)
    dists     = np.linalg.norm(diff, axis=1)            # (M,)
    dists     = np.clip(dists, 1e-3, None)
    dirs      = diff / dists[:, np.newaxis]             # (M, 3) unitaires

    log_dist  = np.log1p(dists).astype(np.float32)

    # cos(normal, source→face) — negative when the face faces the source
    cos_ns    = np.einsum("ij,ij->i", normals, -dirs).astype(np.float32)

    rel_xyz   = (diff / dists[:, np.newaxis]).astype(np.float32)   # (M, 3)

    log_height = np.log1p(np.abs(centroids[:, 2])).astype(np.float32)
    log_area   = np.log1p(areas).astype(np.float32)

    normal_z   = normals[:, 2].astype(np.float32)
    normal_x   = normals[:, 0].astype(np.float32)
    normal_y   = normals[:, 1].astype(np.float32)

    horiz_dist  = np.sqrt(diff[:, 0]**2 + diff[:, 1]**2)
    horiz_dist  = np.clip(horiz_dist, 1e-3, None)
    log_horiz_dist = np.log1p(horiz_dist).astype(np.float32)

    elev_angle  = np.arctan2(-diff[:, 2], horiz_dist)   # (M,)
    elev_norm   = (elev_angle / (math.pi / 2)).astype(np.float32)
    elev_norm   = np.clip(elev_norm, -1.0, 1.0)

    grazing     = (math.pi / 2 - np.abs(elev_angle)) / (math.pi / 2)
    grazing     = grazing.astype(np.float32)

    # cos_angles = full-3D normal-direction cosine; same as cos_ns but kept
    # separate to match FEAT_KEYS.
    cos_angles  = cos_ns.copy()

    # cos_horiz: XY-plane cosine (lateral occlusion-style signal, independent
    # of elevation). Complements cos_ns.
    horiz_dir   = diff[:, :2] / horiz_dist[:, np.newaxis]
    normal_h    = normals[:, :2]
    norm_h_norm = np.linalg.norm(normal_h, axis=1, keepdims=True)
    norm_h_norm = np.where(norm_h_norm < 1e-8, 1.0, norm_h_norm)
    normal_h_u  = normal_h / norm_h_norm
    cos_horiz   = np.einsum("ij,ij->i", normal_h_u, -horiz_dir).astype(np.float32)

    # Slope discontinuity: |Δnormal_z| averaged over 1-ring edges.
    adj = _face_adjacency(faces)
    if adj.shape[1] > 0:
        src_e, dst_e = adj[0], adj[1]
        delta_nz = np.abs(normal_z[src_e] - normal_z[dst_e])
        slope_disc = np.zeros(M, dtype=np.float32)
        np.add.at(slope_disc, src_e, delta_nz)
        degree = np.bincount(src_e, minlength=M).astype(np.float32)
        degree = np.where(degree < 1, 1.0, degree)
        slope_disc /= degree
    else:
        slope_disc = np.zeros(M, dtype=np.float32)

    obstacle_prox = (1.0 / (1.0 + dists)).astype(np.float32)

    # Ray-cast trio (centroid → drone): is_occluded, first_obstacle_frac,
    # log_n_intersections. One vectorised Embree call per mesh.
    is_occ, first_obs_frac, log_n_inter = _ray_occlusion_features(
        verts, faces, centroids, drone_pos,
    )

    # Reflection-proxy features. Reuses the LOS mask above — no extra ray casting.
    # (reflect_score, 3e sortie, retiré du modèle → ignoré.)
    log_dist_to_lit, log_n_lit_nearby, _reflect_score = _reflection_proxy_features(
        centroids, normals, is_occ,
    )

    feats = np.stack([
        log_dist,           # 0
        cos_ns,             # 1
        rel_xyz[:, 0],      # 2
        rel_xyz[:, 1],      # 3
        rel_xyz[:, 2],      # 4
        log_height,         # 5
        log_area,           # 6
        normal_z,           # 7
        log_horiz_dist,     # 8
        cos_angles,         # 9
        grazing,            # 10
        elev_norm,          # 11
        obstacle_prox,      # 12
        slope_disc,         # 13
        normal_x,           # 14
        normal_y,           # 15
        cos_horiz,          # 16
        is_occ,             # 17
        first_obs_frac,     # 18
        log_n_inter,        # 19
        log_dist_to_lit,    # 20
        log_n_lit_nearby,   # 21
    ], axis=1)              # (M, NUM_FEATURES)

    return feats.astype(np.float32)


# ═════════════════════════════════════════════════════════════════════════════
# Sharded dataset loader
# ═════════════════════════════════════════════════════════════════════════════
def load_sharded_dataset(shard_dir: Path = SHARD_DIR):
    """Load every shard into memory and return (graphs, metas).

    Peak memory ≈ total dataset size (no torch.save serialisation), which is
    substantially lighter than the old `torch.load(graphs.pt)` blob.

    metas[i] = {map, drone, class_counts, n_nodes, z} for graph i.
    """
    shard_paths = sorted(shard_dir.glob("shard_*.pt"))
    if not shard_paths:
        raise FileNotFoundError(f"Aucun shard trouvé dans {shard_dir}")

    graphs: list = []
    metas:  list[dict] = []

    for p in shard_paths:
        chunk = torch.load(p, weights_only=False)
        for g in chunk:
            # Compat rétro : les shards générés en 23 features ont reflect_score
            # en DERNIÈRE colonne (index 22). reflect_score ayant été retiré du
            # modèle, on tronque la/les colonne(s) excédentaire(s) pour aligner sur
            # config.NUM_FEATURES sans avoir à régénérer le dataset. No-op si les
            # shards sont déjà en NUM_FEATURES (régénérés).
            if g.x.shape[1] > NUM_FEATURES:
                g.x = g.x[:, :NUM_FEATURES].contiguous()
            y_np   = g.y.numpy()
            valid  = y_np >= 0
            counts = (np.bincount(y_np[valid], minlength=NUM_CLASSES).tolist()
                      if valid.any() else [0] * NUM_CLASSES)
            drone_pos = getattr(g, "drone_pos", None)
            z_src = float(drone_pos[2]) if drone_pos is not None else 0.0
            metas.append({
                "map":          getattr(g, "map_name", "unknown"),
                "drone":        getattr(g, "drone_id", "unknown"),
                "class_counts": counts,
                "n_nodes":      int(g.num_nodes),
                "z":            z_src,   # drone (source) altitude — used to
                                        # balance the test set over altitude
            })
            graphs.append(g)

    return graphs, metas


def load_dataset(shard_dir=SHARD_DIR):
    return load_sharded_dataset(shard_dir)


# ═════════════════════════════════════════════════════════════════════════════
# Map-level split + class weights + DataLoaders
# ═════════════════════════════════════════════════════════════════════════════
def _balance_indices_over_z(z_vals: np.ndarray, indices: list[int],
                            n_bins: int, per_bin, seed: int) -> list[int]:
    """Subsample `indices` so their source altitudes Z form a roughly uniform
    histogram. Z is split into `n_bins` fixed-width bins over [z.min, z.max];
    each bin is capped at `per_bin` graphs (sampled without replacement). When
    `per_bin is None` the cap is the *median* non-empty bin size — this flattens
    the dominant low-altitude bins while keeping the rare high-altitude graphs
    in full, so every Z range weighs ~equally without collapsing to the rarest
    bin."""
    idx = np.asarray(indices)
    z   = np.asarray(z_vals, dtype=np.float64)
    if idx.size == 0 or n_bins < 1:
        return list(indices)

    z_min, z_max = float(z.min()), float(z.max())
    if z_max <= z_min:                       # degenerate: all the same altitude
        return list(indices)

    edges  = np.linspace(z_min, z_max, n_bins + 1)
    bin_id = np.clip(np.digitize(z, edges[1:-1]), 0, n_bins - 1)

    sizes    = np.bincount(bin_id, minlength=n_bins)
    nonempty = sizes[sizes > 0]
    cap = int(per_bin) if per_bin is not None else int(np.median(nonempty))
    cap = max(cap, 1)

    rng    = np.random.default_rng(seed)
    chosen: list[int] = []
    for b in range(n_bins):
        members = idx[bin_id == b]
        if members.size > cap:
            members = rng.choice(members, size=cap, replace=False)
        chosen.extend(int(i) for i in members)
    return sorted(chosen)


def _pack_eval_train_split(map_names: list[str], eval_cities: set[str]
                           ) -> Optional[dict[str, tuple[set[str], list[int]]]]:
    """Assemble the split dict from a set of evaluation city names:

        val   = graphs whose city ∈ eval_cities
        train = every other graph
        test  = val   (final eval reloads the best ckpt and scores it on val)

    Validation cities are whole cities never seen at train time, so they already
    measure geometric generalisation — hence no separate test set. Returns None
    when the split is degenerate (val or train empty for the loaded shards)."""
    val_idx   = sorted(i for i, mn in enumerate(map_names) if mn in eval_cities)
    train_idx = sorted(i for i, mn in enumerate(map_names) if mn not in eval_cities)
    if not val_idx or not train_idx:
        return None
    val_maps   = {map_names[i] for i in val_idx}
    train_maps = {map_names[i] for i in train_idx}
    return {
        "train": (train_maps, train_idx),
        "val":   (val_maps,   val_idx),
        "test":  (set(val_maps), list(val_idx)),   # test == val, no data duplicated
    }


def _city_split_from_file(
    metas,
) -> Optional[dict[str, tuple[set[str], list[int]]]]:
    """Build the split from `city_split.json` (copied by gen_graphs.py from the
    `city_roles.json` written by generate_dataset.py): its `eval_cities` — the
    normal-condition cities generated with uniform-z and few exercises — become
    the validation set, every other city the train set.

    Returns None (→ caller falls back to count-based / random split) when the
    file is absent, unreadable, or lists no eval city present in `metas`."""
    if not CITY_SPLIT_FILE.exists():
        return None
    try:
        info = json.loads(CITY_SPLIT_FILE.read_text())
    except Exception as exc:
        warnings.warn(f"Could not read {CITY_SPLIT_FILE}: {exc}; "
                      f"falling back to count-based split.")
        return None

    # `eval_cities` is the current key; `partial_cities` kept for back-compat.
    eval_cities = set(info.get("eval_cities") or info.get("partial_cities") or [])
    if not eval_cities:
        return None

    split = _pack_eval_train_split([m["map"] for m in metas], eval_cities)
    if split is None:
        warnings.warn(
            f"{CITY_SPLIT_FILE.name}: no eval city present in the loaded shards "
            f"(or no train city left); falling back to count-based split."
        )
    return split


def _city_split_by_count(metas, n_eval: int = N_EVAL_CITIES
                         ) -> Optional[dict[str, tuple[set[str], list[int]]]]:
    """Fallback when the roles file is missing: pick the `n_eval` cities with the
    fewest graphs as validation (eval cities are generated with clearly fewer
    exercises), the rest as train. Ties broken by city name for determinism."""
    map_names = [m["map"] for m in metas]
    counts: dict[str, int] = {}
    for mn in map_names:
        counts[mn] = counts.get(mn, 0) + 1
    if len(counts) <= n_eval:
        return None
    eval_cities = set(sorted(counts, key=lambda c: (counts[c], c))[:n_eval])
    return _pack_eval_train_split(map_names, eval_cities)


def map_level_split(metas, val_ratio: float,
                    split_seed: int, *, n_test_cities: int = N_TEST_CITIES,
                    z_balance: bool = EVAL_Z_BALANCE, z_bins: int = EVAL_Z_BINS,
                    z_per_bin=EVAL_Z_PER_BIN,
                    ) -> dict[str, tuple[set[str], list[int]]]:
    """Build train/val/test index sets.

    When `SPLIT_FROM_CITY_FILE` is set, the split reuses the eval/train city
    roles decided at generation time: eval cities → validation, train cities →
    train (`_city_split_from_file`, or `_city_split_by_count` if the roles file
    is absent). Otherwise it falls back to the random whole-city hold-out below.

    Hold out `n_test_cities` whole cities for the final test so the model is
    scored on geometry it has never seen; the remaining cities are the training
    pool. Validation (early-stopping / LR scheduler) is carved out graph-level
    from the training pool, so every train city keeps contributing and the test
    cities stay completely clean (no leakage into model selection).

    The random-fallback test size is governed by `n_test_cities`; `val_ratio` is
    the graph-level fraction of the training pool reserved for validation.

    When `z_balance` is set, the test graphs are subsampled to a roughly uniform
    source-altitude (Z) histogram — see `_balance_indices_over_z`."""
    if SPLIT_FROM_CITY_FILE:
        from_file = _city_split_from_file(metas)
        if from_file is not None:
            return from_file
        by_count = _city_split_by_count(metas)
        if by_count is not None:
            return by_count

    map_names   = [m["map"] for m in metas]
    unique_maps = sorted(set(map_names))
    rng  = torch.Generator().manual_seed(split_seed)
    perm = torch.randperm(len(unique_maps), generator=rng).tolist()

    n_test     = min(max(1, n_test_cities), max(1, len(unique_maps) - 1))
    test_maps  = {unique_maps[i] for i in perm[:n_test]}
    train_maps = {unique_maps[i] for i in perm[n_test:]}

    test_idx_all = [i for i, m in enumerate(map_names) if m in test_maps]
    train_pool   = [i for i, m in enumerate(map_names) if m in train_maps]

    # ── Test set: hold-out cities, optionally balanced over altitude Z ────────
    if z_balance:
        z_vals   = np.array([metas[i]["z"] for i in test_idx_all], dtype=np.float64)
        test_idx = _balance_indices_over_z(z_vals, test_idx_all, z_bins,
                                           z_per_bin, split_seed)
    else:
        test_idx = test_idx_all

    # ── Validation: graph-level hold-out from the training pool ───────────────
    pool = np.array(train_pool)
    np.random.default_rng(split_seed).shuffle(pool)
    n_val     = max(1, int(round(len(pool) * val_ratio)))
    val_idx   = sorted(int(i) for i in pool[:n_val])
    train_idx = sorted(int(i) for i in pool[n_val:])
    val_maps  = {map_names[i] for i in val_idx}   # ⊆ train_maps (geometry shared)

    return {
        "train": (train_maps, train_idx),
        "val":   (val_maps,   val_idx),
        "test":  (test_maps,  test_idx),
    }


def compute_class_weights(metas, train_idx, num_classes: int = NUM_CLASSES
                          ) -> tuple[np.ndarray, np.ndarray, bool]:
    """Return (train_counts, class_weights, is_manual). Honours the manual
    `CLASS_WEIGHTS` override from `config.py`; otherwise inverse-sqrt of the
    train class distribution, normalised to mean=1."""
    counts = np.zeros(num_classes, dtype=np.int64)
    for i in train_idx:
        counts += np.asarray(metas[i]["class_counts"], dtype=np.int64)
    if CLASS_WEIGHTS is not None:
        weights = np.array(CLASS_WEIGHTS, dtype=np.float32)
        return counts, weights, True
    safe = np.maximum(counts, 1)
    raw  = 1.0 / np.sqrt(safe.astype(np.float64))
    weights = (raw / raw.mean()).astype(np.float32)
    return counts, weights, False


def build_loaders(graphs, split, batch_size: int, env, seed: int,
                  eval_batch_size: int | None = None):
    """Build (train_loader, val_loader, test_loader, train_sampler).

    Validation/test loaders run on rank 0 only — kept un-sharded so metrics
    don't need cross-rank gathering. `env` is a `DistEnv`. `eval_batch_size`
    defaults to `batch_size`; use a smaller value to cap the peak eval memory
    (the GATv2 attention tensor scales with the batch)."""
    if eval_batch_size is None:
        eval_batch_size = batch_size
    train_set = [graphs[i] for i in split["train"][1]]
    val_set   = [graphs[i] for i in split["val"][1]]
    test_set  = [graphs[i] for i in split["test"][1]]

    pin = env.device.type == "cuda"
    if env.is_distributed:
        train_sampler = DistributedSampler(
            train_set, num_replicas=env.world_size, rank=env.rank,
            shuffle=True, seed=seed, drop_last=False,
        )
        train_loader = DataLoader(train_set, batch_size=batch_size,
                                  sampler=train_sampler, pin_memory=pin)
    else:
        train_sampler = None
        gen = torch.Generator().manual_seed(seed)
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                                  generator=gen, pin_memory=pin)

    val_loader  = DataLoader(val_set,  batch_size=eval_batch_size, pin_memory=pin)
    test_loader = DataLoader(test_set, batch_size=eval_batch_size, pin_memory=pin)
    return train_loader, val_loader, test_loader, train_sampler


# ═════════════════════════════════════════════════════════════════════════════
# Per-run artefacts: timestamped folder + config dump + history plots
# ═════════════════════════════════════════════════════════════════════════════
_CLASS_NAMES = [
    "violet", "blue", "yellow", "orange", "red", "dark_red", "occluded",
]


def make_run_dir(plots_root: Path, stamp: str | None = None) -> tuple[Path, str]:
    """Create `plots_root/run_<stamp>/` and return (path, stamp).

    `stamp` defaults to YYYYMMDD_HHMMSS. The directory is created eagerly so
    every artefact (config dump, plots, …) lands in the same place even if
    training later crashes."""
    if stamp is None:
        stamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = plots_root / f"run_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir, stamp


def save_training_config(run_dir: Path, args, class_weights, train_counts,
                         train_maps, val_maps, test_maps,
                         track_metric: str, env, log) -> Path:
    """Dump a human-readable summary of the training configuration into
    `run_dir/config.txt`. Written at the start of training so it survives a
    crash."""
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "config.txt"

    cw = list(class_weights) if class_weights is not None else []
    counts = list(train_counts) if train_counts is not None else []
    total = int(sum(counts)) if counts else 0

    lines: list[str] = []
    lines.append("# GAT training run")
    lines.append(f"# started     : {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"# run dir     : {run_dir}")
    lines.append(f"# world size  : {env.world_size}  (distributed={env.is_distributed})")
    lines.append(f"# device      : {env.device}")
    lines.append("")

    lines.append("## CLI args")
    width = max((len(k) for k in vars(args)), default=0)
    for k, v in sorted(vars(args).items()):
        lines.append(f"  {k:<{width}} = {v}")
    lines.append("")

    lines.append("## Tracking metric")
    lines.append(f"  {track_metric}")
    lines.append("")

    if cw:
        lines.append("## Class weights (CE loss)")
        for i, w in enumerate(cw):
            name = _CLASS_NAMES[i] if i < len(_CLASS_NAMES) else f"class{i}"
            lines.append(f"  class {i} ({name:<9}) : {float(w):.4f}")
        lines.append("")

    if counts and total > 0:
        lines.append("## Train class counts")
        for i, c in enumerate(counts):
            name = _CLASS_NAMES[i] if i < len(_CLASS_NAMES) else f"class{i}"
            pct = 100.0 * c / total
            lines.append(f"  class {i} ({name:<9}) : {int(c):>12,d}  ({pct:5.2f}%)")
        lines.append(f"  {'TOTAL':<19}: {total:>12,d}")
        lines.append("")

    def _fmt_maps(label: str, maps) -> str:
        m = sorted(maps)
        return f"  {label:<7} ({len(m):2d} maps): " + ", ".join(m)

    lines.append("## Map-level split")
    lines.append(_fmt_maps("train", train_maps))
    lines.append(_fmt_maps("val",   val_maps))
    lines.append(_fmt_maps("test",  test_maps))
    lines.append("")

    path.write_text("\n".join(lines) + "\n")
    log.success(f"saved {path}")
    return path


def finalize_run_artifacts(run_dir: Path, ckpt_dir: Path, log) -> Path:
    """Move the run's artefacts (config.txt + plots) into the checkpoint
    directory they belong to. Called at the very end of training.

    Skipped when no checkpoint was actually written (e.g. training stopped
    before the first improvement) — in that case the artefacts stay in
    `plots/` since there's no ckpt to associate them with.

    If a previous run already left a folder of the same name under
    `ckpt_dir`, it is overwritten. Returns the final destination path
    (== `run_dir` if no move happened)."""
    if not run_dir.exists():
        log.warning(f"run_dir {run_dir} does not exist — nothing to move")
        return run_dir

    has_ckpt = any(ckpt_dir.glob("gat_*.pt")) if ckpt_dir.exists() else False
    if not has_ckpt:
        log.warning(
            f"no checkpoint found under {ckpt_dir} — leaving run artefacts "
            f"in {run_dir}"
        )
        return run_dir

    target = ckpt_dir / run_dir.name
    if target.exists():
        shutil.rmtree(target)
    shutil.move(str(run_dir), str(target))
    log.success(f"moved run artefacts → {target}")
    return target


def save_history_plots(hist, best_epoch, out_dir, log) -> None:
    """Per-epoch loss / accuracy / balanced-acc on top row, per-class
    precision / recall / F1 on bottom row. Plus a separate LR-schedule plot.

    Files are written as `out_dir/metrics.png` and `out_dir/lr.png` — the
    timestamp lives on `out_dir` itself (see `make_run_dir`)."""
    if not hist["epoch"]:
        log.warning("No epoch history recorded, skipping plots.")
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    ep = np.array(hist["epoch"])

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    for ax, key, title in zip(
        axes[0], ["loss", "acc", "bal"], ["Loss", "Accuracy", "Balanced Acc"]
    ):
        if key == "loss":
            ax.plot(ep, hist["train_loss"], label="train")
        ax.plot(ep, hist[f"val_{key}"],  label="val")
        if best_epoch:
            ax.axvline(best_epoch, color="red", linestyle="--",
                       alpha=0.5, label=f"best ep {best_epoch}")
        ax.set_xlabel("Epoch"); ax.set_ylabel(title)
        ax.set_title(f"{title} per epoch"); ax.grid(alpha=0.3); ax.legend()

    prec = np.array(hist["val_prec"])
    rec  = np.array(hist["val_rec"])
    f1   = np.array(hist["val_f1"])
    for ax, arr, title in zip(
        axes[1], [prec, rec, f1], ["Precision", "Recall", "F1"]
    ):
        for c in range(arr.shape[1]):
            ax.plot(ep, arr[:, c], label=f"class {c}")
        if best_epoch:
            ax.axvline(best_epoch, color="red", linestyle="--", alpha=0.5)
        ax.set_xlabel("Epoch"); ax.set_ylabel(title)
        ax.set_title(f"VAL — {title} per class")
        ax.set_ylim(-0.02, 1.02); ax.grid(alpha=0.3); ax.legend(fontsize=8)

    fig.tight_layout()
    p = out_dir / "metrics.png"
    fig.savefig(p, dpi=120); plt.close(fig)
    log.success(f"saved {p}")

    if hist.get("lr"):
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(ep, hist["lr"])
        if best_epoch:
            ax.axvline(best_epoch, color="red", linestyle="--",
                       alpha=0.5, label=f"best ep {best_epoch}")
            ax.legend()
        ax.set_xlabel("Epoch"); ax.set_ylabel("Learning rate")
        ax.set_yscale("log"); ax.set_title("Learning-rate schedule")
        ax.grid(alpha=0.3); fig.tight_layout()
        p = out_dir / "lr.png"
        fig.savefig(p, dpi=120); plt.close(fig)
        log.success(f"saved {p}")
