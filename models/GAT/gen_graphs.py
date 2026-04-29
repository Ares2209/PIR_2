#!/usr/bin/env python3
"""
gen_graphs.py — Génère le dataset de graphes en format shardé.

Format de sortie :
    processed/shards/shard_XXXX.pt   ← chaque shard = SHARD_SIZE graphes
    processed/node_stats.json         ← stats de normalisation globales

Plus jamais de rechargement global du dataset en mémoire.
Pic mémoire = 1 shard (~500 graphes) + 1 fichier PLY en cours.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
import sys
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch_geometric.data import Data

# ─────────────────────────────────────────────────────────────────────────────
# Chemins — corrigés (plus de triple nesting)
# ─────────────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT  = SCRIPT_DIR.parents[1]
GENERATED  = REPO_ROOT / "dataset" / "data" / "generated"
OUT_DIR    = GENERATED / "processed"           # ← était "generated/dataset/data/processed"
SHARD_DIR  = OUT_DIR / "shards"
STATS_FILE = OUT_DIR / "node_stats.json"

SHARD_SIZE = 500   # graphes par fichier shard
LOG_EVERY  = 100   # log de progression tous les N graphes


# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Imports locaux (config partagée)
# ─────────────────────────────────────────────────────────────────────────────
sys.path.insert(0, str(SCRIPT_DIR))
from config import (
    DRONES, META, LP_REF, SLOPE,
    N_BANDS, NUM_FEATURES, DRONE_FEAT_DIM,
    NODE_STATS, FEAT_KEYS, DRONE_NORM,
    RGB_TO_CLASS, NUM_CLASSES, RGB_TOLERANCE,
    FNAME_RE,
    _normalize_drone_vector,
    _parse_ply,
    _face_adjacency,
    _classify_with_tolerance,
    _normalize_node_features,
    _load_base_mesh,
    _OCC_CHUNK,
    BLENDER_DIR,
)

# ─────────────────────────────────────────────────────────────────────────────
# Construction des features nœuds
# ─────────────────────────────────────────────────────────────────────────────

def _build_node_features(
    verts:    np.ndarray,   # (N, 3) float32 — vertices originaux du mesh
    faces:    np.ndarray,   # (M, 3) int64   — indices vers verts
    drone_pos: np.ndarray,  # (3,)   float32
    map_name: str,
) -> np.ndarray:
    """
    Construit la matrice de features (M, 18) pour les faces d'un graphe.

    Features (dans l'ordre canonique FEAT_KEYS) :
      0  log_dist           distance log au drone
      1  cos_ns             cosinus normale-source
      2  rel_x              position relative X normalisée
      3  rel_y              position relative Y normalisée
      4  rel_z              position relative Z normalisée
      5  log_height         log(hauteur du centroïde)
      6  log_area           log(aire de la face)
      7  normal_z           composante Z de la normale
      8  log_horiz_dist     log(distance horizontale)
      9  occluded           flag d'occlusion ray-cast
     10  cos_angles         cosinus angle d'incidence horizontal
     11  grazing_angle      angle rasant normalisé
     12  elevation_angle    angle d'élévation normalisé
     13  obstacle_proximity proximité des obstacles
     14  slope_discontinuity discontinuité de pente
     15  normal_x           composante X de la normale
     16  normal_y           composante Y de la normale
     17  cos_horiz          cosinus angle horizontal drone→face
    """
    M = faces.shape[0]
    if M == 0:
        return np.empty((0, NUM_FEATURES), dtype=np.float32)

    # ── Centroïdes et normales des faces ─────────────────────────────────────
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    centroids = (v0 + v1 + v2) / 3.0   # (M, 3)

    e1 = v1 - v0
    e2 = v2 - v0
    raw_normals = np.cross(e1, e2)      # (M, 3)
    norms       = np.linalg.norm(raw_normals, axis=1, keepdims=True)
    norms       = np.where(norms < 1e-8, 1.0, norms)   # évite /0
    normals     = raw_normals / norms   # (M, 3) unitaires

    # ── Aires des faces ───────────────────────────────────────────────────────
    areas = np.linalg.norm(raw_normals, axis=1) * 0.5   # (M,)
    areas = np.clip(areas, 1e-6, None)

    # ── Vecteurs drone→face ───────────────────────────────────────────────────
    diff      = centroids - drone_pos[np.newaxis, :]    # (M, 3)
    dists     = np.linalg.norm(diff, axis=1)            # (M,)
    dists     = np.clip(dists, 1e-3, None)
    dirs      = diff / dists[:, np.newaxis]             # (M, 3) unitaires

    log_dist  = np.log1p(dists).astype(np.float32)

    # cos(normale, direction source→face) — négatif : face bien orientée
    cos_ns    = np.einsum("ij,ij->i", normals, -dirs).astype(np.float32)

    # Positions relatives normalisées par la distance
    rel_xyz   = (diff / dists[:, np.newaxis]).astype(np.float32)   # (M, 3)

    log_height = np.log1p(np.abs(centroids[:, 2])).astype(np.float32)
    log_area   = np.log1p(areas).astype(np.float32)

    normal_z   = normals[:, 2].astype(np.float32)
    normal_x   = normals[:, 0].astype(np.float32)
    normal_y   = normals[:, 1].astype(np.float32)

    horiz_dist  = np.sqrt(diff[:, 0]**2 + diff[:, 1]**2)
    horiz_dist  = np.clip(horiz_dist, 1e-3, None)
    log_horiz_dist = np.log1p(horiz_dist).astype(np.float32)

    # Angle d'élévation (drone vu depuis la face)
    elev_angle  = np.arctan2(-diff[:, 2], horiz_dist)   # (M,)
    elev_norm   = (elev_angle / (math.pi / 2)).astype(np.float32)
    elev_norm   = np.clip(elev_norm, -1.0, 1.0)

    # Angle rasant (complément de l'angle d'élévation)
    grazing     = (math.pi / 2 - np.abs(elev_angle)) / (math.pi / 2)
    grazing     = grazing.astype(np.float32)

    # cos_angles = cosinus 3D entre normale et direction face→drone
    # (= cos_ns ; conservé séparément pour cohérence avec FEAT_KEYS).
    cos_angles  = cos_ns.copy()

    # cos_horiz  = cosinus dans le plan XY uniquement (occlusion latérale,
    # indépendante de l'élévation). Apporte une info distincte de cos_ns.
    horiz_dir   = diff[:, :2] / horiz_dist[:, np.newaxis]   # (M, 2) unitaire
    normal_h    = normals[:, :2]
    norm_h_norm = np.linalg.norm(normal_h, axis=1, keepdims=True)
    norm_h_norm = np.where(norm_h_norm < 1e-8, 1.0, norm_h_norm)
    normal_h_u  = normal_h / norm_h_norm
    cos_horiz   = np.einsum("ij,ij->i", normal_h_u, -horiz_dir).astype(np.float32)

    # Discontinuité de pente (variance locale des normales via arêtes adjacentes)
    # Approximation : std de normal_z dans un voisinage 1-ring reconstruit
    # ici on utilise une version simplifiée : |Δnormal_z| moyenné sur les arêtes
    adj = _face_adjacency(faces)
    if adj.shape[1] > 0:
        src_e, dst_e = adj[0], adj[1]
        delta_nz = np.abs(normal_z[src_e] - normal_z[dst_e])
        # Accumulation par face source
        slope_disc = np.zeros(M, dtype=np.float32)
        np.add.at(slope_disc, src_e, delta_nz)
        degree = np.bincount(src_e, minlength=M).astype(np.float32)
        degree = np.where(degree < 1, 1.0, degree)
        slope_disc /= degree
    else:
        slope_disc = np.zeros(M, dtype=np.float32)

    # Proximité d'obstacles : inverse de la distance normalisée
    obstacle_prox = (1.0 / (1.0 + dists)).astype(np.float32)

    # ── Occlusion par ray-cast ────────────────────────────────────────────────
    occluded = _compute_occlusion(centroids, drone_pos, map_name)

    # ── Assemblage ────────────────────────────────────────────────────────────
    feats = np.stack([
        log_dist,           # 0
        cos_ns,             # 1
        rel_xyz[:, 0],      # 2  rel_x
        rel_xyz[:, 1],      # 3  rel_y
        rel_xyz[:, 2],      # 4  rel_z
        log_height,         # 5
        log_area,           # 6
        normal_z,           # 7
        log_horiz_dist,     # 8
        occluded,           # 9
        cos_angles,         # 10
        grazing,            # 11
        elev_norm,          # 12
        obstacle_prox,      # 13
        slope_disc,         # 14
        normal_x,           # 15
        normal_y,           # 16
        cos_horiz,          # 17
    ], axis=1)              # (M, 18)

    return feats.astype(np.float32)


def _compute_occlusion(
    centroids: np.ndarray,   # (M, 3)
    drone_pos: np.ndarray,   # (3,)
    map_name:  str,
) -> np.ndarray:             # (M,) float32 ∈ {0, 1}
    """
    Ray-cast depuis chaque centroïde vers le drone.
    Retourne 1.0 si occulté, 0.0 sinon.
    Utilise le BVH du mesh de base (chargé une fois par map_name).
    """
    mesh = _load_base_mesh(map_name)
    M    = centroids.shape[0]
    occ  = np.zeros(M, dtype=np.float32)

    if mesh is None:
        return occ

    target = drone_pos[np.newaxis, :]   # (1, 3)

    for start in range(0, M, _OCC_CHUNK):
        end   = min(start + _OCC_CHUNK, M)
        origs = centroids[start:end]            # (chunk, 3)
        dirs  = target - origs                  # (chunk, 3)
        dists = np.linalg.norm(dirs, axis=1, keepdims=True)
        dists = np.where(dists < 1e-6, 1.0, dists)
        dirs_u = dirs / dists                   # unitaires

        # Léger offset pour éviter auto-intersection
        origs_offset = origs + dirs_u * 1e-3

        try:
            hits = mesh.ray.intersects_any(
                ray_origins=origs_offset,
                ray_directions=dirs_u,
            )
            occ[start:end] = hits.astype(np.float32)
        except Exception as exc:
            warnings.warn(f"ray-cast error (map={map_name}): {exc}")

    return occ


# ─────────────────────────────────────────────────────────────────────────────
# Conversion PLY → Data PyG
# ─────────────────────────────────────────────────────────────────────────────

def ply_to_graph(path: Path) -> Optional[Data]:
    """
    Convertit un fichier NoiseMap_*.ply en objet Data PyG (non normalisé).
    Retourne None si le fichier est invalide ou contient trop peu de faces.
    """
    m = FNAME_RE.match(path.name)
    if m is None:
        warnings.warn(f"filename does not match pattern: {path.name}")
        return None

    map_name  = m.group("map")
    drone_pos = np.array(
        [float(m.group("x")), float(m.group("y")), float(m.group("z"))],
        dtype=np.float32,
    )
    drone_id  = m.group("drone")
    if drone_id not in META:
        warnings.warn(f"unknown drone '{drone_id}' in {path.name}")
        return None

    # ── Parse PLY ─────────────────────────────────────────────────────────────
    try:
        verts, faces, face_rgb = _parse_ply(path)
    except Exception as exc:
        warnings.warn(f"failed to parse {path.name}: {exc}")
        return None

    M = faces.shape[0]
    if M < 3:
        warnings.warn(f"{path.name}: only {M} faces after cleanup, skipping")
        return None

    # ── Labels ────────────────────────────────────────────────────────────────
    y = _classify_with_tolerance(face_rgb)
    n_unknown = int((y == -1).sum())
    if n_unknown > 0:
        warnings.warn(
            f"{path.name}: {n_unknown}/{M} faces have unknown RGB color"
        )
    if (y == -1).all():
        warnings.warn(f"{path.name}: all labels unknown, skipping")
        return None

    # ── Centroïdes (feature input) ────────────────────────────────────────────
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    centroids = (v0 + v1 + v2) / 3.0

    # ── Node features (brutes, non normalisées) ────────────────────────────────
    # NB: on passe `verts` (vertices originaux), pas `centroids` :
    # _build_node_features indexe verts via faces et recalcule lui-même
    # les centroïdes / normales / aires.
    node_feats = _build_node_features(verts, faces, drone_pos, map_name)

    # ── Adjacence ─────────────────────────────────────────────────────────────
    edge_index = _face_adjacency(faces)

    # ── Drone features ────────────────────────────────────────────────────────
    drone_vec = _normalize_drone_vector(drone_id)   # (51,) déjà normalisé

    # ── Assemblage PyG ────────────────────────────────────────────────────────
    data = Data(
        x          = torch.from_numpy(node_feats),              # (M, 18)
        edge_index = torch.from_numpy(edge_index).long(),       # (2, E)
        y          = torch.from_numpy(y).long(),                # (M,)
        drone_feat = torch.from_numpy(drone_vec).unsqueeze(0),  # (1, 51)
        pos        = torch.from_numpy(centroids),               # (M, 3)
        map_name   = map_name,
        drone_id   = drone_id,
        drone_pos  = torch.from_numpy(drone_pos),               # (3,)
    )
    return data


# ─────────────────────────────────────────────────────────────────────────────
# Stats en streaming (Welford one-pass)
# ─────────────────────────────────────────────────────────────────────────────

class _WelfordAccumulator:
    """
    Calcul en ligne de mean + M2 (pour std) selon l'algorithme de Welford.
    Fonctionne sur des batches (matrices 2D, accumulation par colonne).
    Pic mémoire : O(n_features) — indépendant du nombre de graphes.
    """

    def __init__(self, n_features: int):
        self.n       = 0
        self.mean    = np.zeros(n_features, dtype=np.float64)
        self.M2      = np.zeros(n_features, dtype=np.float64)
        self.n_feats = n_features

    def update_batch(self, x: np.ndarray) -> None:
        """x : (N, F) float array."""
        for row in x:
            self.n += 1
            delta      = row - self.mean
            self.mean += delta / self.n
            delta2     = row - self.mean
            self.M2   += delta * delta2

    def finalize(self) -> tuple[np.ndarray, np.ndarray]:
        """Retourne (mean, std) shape (F,)."""
        if self.n < 2:
            return self.mean.astype(np.float32), np.ones(self.n_feats, dtype=np.float32)
        variance = self.M2 / (self.n - 1)
        std = np.sqrt(np.maximum(variance, 1e-8))
        return self.mean.astype(np.float32), std.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Écriture shardée
# ─────────────────────────────────────────────────────────────────────────────

class ShardWriter:
    """
    Accumule des graphes et écrit un fichier shard tous les SHARD_SIZE graphes.
    Ne recharge jamais un shard existant.
    """

    def __init__(self, shard_dir: Path, shard_size: int):
        self.shard_dir  = shard_dir
        self.shard_size = shard_size
        self.shard_dir.mkdir(parents=True, exist_ok=True)

        self.buffer:      list[Data] = []
        self.shard_index: int        = 0
        self.total_graphs: int       = 0

    def add(self, g: Data) -> None:
        self.buffer.append(g)
        if len(self.buffer) >= self.shard_size:
            self._flush()

    def _flush(self) -> None:
        if not self.buffer:
            return
        path = self.shard_dir / f"shard_{self.shard_index:04d}.pt"
        torch.save(self.buffer, path)
        log.info(
            f"  → shard {self.shard_index:04d} écrit : "
            f"{len(self.buffer)} graphes → {path.name} "
            f"({path.stat().st_size / 1e6:.1f} MB)"
        )
        self.total_graphs += len(self.buffer)
        self.buffer        = []
        self.shard_index  += 1

    def close(self) -> int:
        """Flush le buffer restant. Retourne le nombre total de graphes."""
        self._flush()
        return self.total_graphs


# ─────────────────────────────────────────────────────────────────────────────
# Normalisation post-build en streaming shard par shard
# ─────────────────────────────────────────────────────────────────────────────

def compute_stats_from_shards(shard_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Passe 1 : calcule mean/std sur tous les shards sans tout charger.
    Pic mémoire = 1 shard à la fois.
    """
    acc = _WelfordAccumulator(NUM_FEATURES)
    shards = sorted(shard_dir.glob("shard_*.pt"))
    log.info(f"Calcul des stats sur {len(shards)} shards...")

    for shard_path in shards:
        graphs: list[Data] = torch.load(shard_path, weights_only=False)
        for g in graphs:
            acc.update_batch(g.x.numpy())
        del graphs   # libère immédiatement

    mean, std = acc.finalize()
    log.info(f"Stats calculées sur {acc.n:,} nœuds.")
    return mean, std


def normalize_shards_inplace(
    shard_dir: Path,
    mean: np.ndarray,
    std:  np.ndarray,
) -> None:
    """
    Passe 2 : normalise x dans chaque shard et réécrit le fichier.
    Pic mémoire = 1 shard à la fois.
    """
    mean_t = torch.from_numpy(mean)
    std_t  = torch.from_numpy(std)
    shards = sorted(shard_dir.glob("shard_*.pt"))
    log.info(f"Normalisation de {len(shards)} shards...")

    for shard_path in shards:
        graphs: list[Data] = torch.load(shard_path, weights_only=False)
        for g in graphs:
            g.x = ((g.x - mean_t) / std_t).float()
        torch.save(graphs, shard_path)
        del graphs
        log.info(f"  ✓ {shard_path.name} normalisé")


def save_stats_json(
    mean: np.ndarray,
    std:  np.ndarray,
    keys: list[str],
    out_path: Path,
) -> None:
    """Écrit node_stats.json lisible par config.py et infer.py."""
    stats = {k: [float(mean[i]), float(std[i])] for i, k in enumerate(keys)}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(stats, f, indent=2)
    log.info(f"Stats sauvegardées → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Boucle principale
# ─────────────────────────────────────────────────────────────────────────────

def build_dataset(
    generated_dir: Path,
    shard_dir:     Path,
    resume:        bool = True,
    normalize:     bool = True,
) -> None:
    """
    Construit le dataset shardé depuis les fichiers PLY générés.

    Architecture attendue :
        generated_dir/
            {drone_name}/
                NoiseMap_{ville}_{x}_{y}_{z}_{drone_name}.ply
                ...
    """
    # ── Collecte récursive de tous les PLY ────────────────────────────────────
    ply_files = sorted(generated_dir.rglob("NoiseMap_*.ply"))
    if not ply_files:
        log.error(
            f"Aucun fichier NoiseMap_*.ply trouvé (récursivement) dans {generated_dir}\n"
            f"Structure attendue : {generated_dir}/{{drone}}/NoiseMap_*.ply"
        )
        sys.exit(1)

    # Log par sous-dossier drone pour vérification
    drones_found: dict[str, int] = {}
    for p in ply_files:
        drone_folder = p.parent.name          # e.g. "DJI_Mavic3"
        drones_found[drone_folder] = drones_found.get(drone_folder, 0) + 1
    log.info(f"Trouvé {len(ply_files):,} fichiers PLY dans {generated_dir}")
    for drone_name, count in sorted(drones_found.items()):
        log.info(f"  {drone_name:30s} : {count:,} fichiers")

    # ── Calcul du point de reprise ─────────────────────────────────────────────
    existing_shards = sorted(shard_dir.glob("shard_*.pt")) if resume else []
    n_already_done  = 0

    if existing_shards and resume:
        # Charge le dernier shard pour connaître son vrai count
        last_shard_idx = len(existing_shards) - 1
        last_shard     = existing_shards[-1]
        last_graphs    = torch.load(last_shard, weights_only=False)
        last_count     = len(last_graphs)
        del last_graphs

        # Nombre de graphes = shards complets × SHARD_SIZE + dernier shard
        n_already_done = last_shard_idx * SHARD_SIZE + last_count

        # Si le dernier shard est incomplet, on le supprime et on reprend
        # depuis le début de ce shard (évite duplication)
        if last_count < SHARD_SIZE:
            log.info(
                f"Reprise : {last_shard.name} incomplet ({last_count} graphes), "
                f"suppression et reprise depuis ce shard."
            )
            last_shard.unlink()
            n_already_done = last_shard_idx * SHARD_SIZE
            existing_shards = existing_shards[:-1]

        log.info(
            f"Reprise depuis le graphe #{n_already_done} "
            f"({len(existing_shards)} shards complets déjà présents)"
        )

    # ── Création du ShardWriter (continue depuis l'état actuel) ───────────────
    writer = ShardWriter(shard_dir, SHARD_SIZE)
    writer.shard_index = len(existing_shards)   # reprend la numérotation

    n_ok   = 0
    n_skip = 0
    n_err  = 0

    for idx, ply_path in enumerate(ply_files):

        # Saute les PLY déjà intégrés dans des shards complets
        if idx < n_already_done:
            continue

        if idx % LOG_EVERY == 0:
            log.info(
                f"[{idx:>6}/{len(ply_files)}] "
                f"ok={n_ok} skip={n_skip} err={n_err} | {ply_path.name}"
            )

        g = ply_to_graph(ply_path)
        if g is None:
            n_err += 1
            continue

        # Filtre les graphes trop petits ou sans labels valides
        valid_labels = (g.y >= 0).sum().item()
        if valid_labels == 0:
            n_skip += 1
            continue

        writer.add(g)
        n_ok += 1

    total = writer.close()
    log.info(
        f"\n{'─'*60}\n"
        f"Build terminé : {total:,} graphes dans {writer.shard_index} shards\n"
        f"  ok={n_ok}  skip={n_skip}  err={n_err}\n"
        f"Sortie : {shard_dir}\n"
        f"{'─'*60}"
    )

    if total == 0:
        log.error("Aucun graphe valide produit. Vérifiez les PLY.")
        sys.exit(1)

    # ── Normalisation en streaming ─────────────────────────────────────────────
    if normalize:
        log.info("\n── Passe de normalisation (streaming) ──")
        mean, std = compute_stats_from_shards(shard_dir)
        save_stats_json(mean, std, FEAT_KEYS, STATS_FILE)
        normalize_shards_inplace(shard_dir, mean, std)
        log.info("Normalisation terminée.")
    else:
        log.info("Normalisation ignorée (--no-normalize).")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Génère le dataset de graphes en format shardé."
    )
    p.add_argument(
        "--generated-dir", type=Path, default=GENERATED,
        help="Dossier contenant les NoiseMap_*.ply (default: %(default)s)",
    )
    p.add_argument(
        "--out-dir", type=Path, default=OUT_DIR,
        help="Dossier de sortie processed/ (default: %(default)s)",
    )
    p.add_argument(
        "--shard-size", type=int, default=SHARD_SIZE,
        help="Graphes par shard (default: %(default)s)",
    )
    p.add_argument(
        "--no-resume", action="store_true",
        help="Repart de zéro (supprime les shards existants)",
    )
    p.add_argument(
        "--no-normalize", action="store_true",
        help="Saute la passe de normalisation",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    shard_dir = args.out_dir / "shards"

    if args.no_resume and shard_dir.exists():
        import shutil
        log.info(f"--no-resume : suppression de {shard_dir}")
        shutil.rmtree(shard_dir)

    build_dataset(
        generated_dir = args.generated_dir,
        shard_dir     = shard_dir,
        resume        = not args.no_resume,
        normalize     = not args.no_normalize,
    )
