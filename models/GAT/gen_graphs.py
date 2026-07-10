#!/usr/bin/env python3
"""
gen_graphs.py — Génère le dataset de graphes en format shardé.

Format de sortie :
    processed/shards/shard_XXXX.pt   ← chaque shard = SHARD_SIZE graphes
    processed/node_stats.json         ← stats de normalisation globales
    processed/shards/.manifest.txt    ← liste des PLYs déjà traités (resume)

Plus jamais de rechargement global du dataset en mémoire.
Pic mémoire = 1 shard (~500 graphes) + 1 fichier PLY en cours par worker.

Note : ce script est CPU-bound (parsing PLY, calcul de features, ray-casting
BVH). Le GPU n'est pas utilisé. La montée en charge passe par le
multi-process CPU (`--num-workers`, par défaut = `os.cpu_count()`).
"""

from __future__ import annotations

import os

# Limite les threads BLAS/OpenMP à 1 par process : avec mp.Pool, chaque worker
# fork()é hériterait sinon d'un pool de threads qui rentrerait en concurrence
# avec les autres workers (sursouscription → 30-50 % de perte). Doit être posé
# AVANT `import numpy`.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import logging
import multiprocessing as mp
import re
import sys
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch_geometric.data import Data
from tqdm import tqdm

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
MANIFEST_NAME = ".manifest.txt"   # liste des PLYs déjà traités (un chemin par ligne)


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
    FEAT_KEYS,
    RGB_TO_CLASS, NUM_CLASSES, RGB_TOLERANCE,
    FNAME_RE,
)
from dataio import (
    _build_node_features,
    _classify_with_tolerance,
    _face_adjacency,
    _normalize_drone_vector,
    _normalize_node_features,
    _parse_ply,
)

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
        x          = torch.from_numpy(node_feats),              # (M, 17)
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
# Workers multi-process (top-level pour être picklable par mp.Pool)
# ─────────────────────────────────────────────────────────────────────────────

def _worker_build_graph(path_str: str):
    """Construit un graphe à partir d'un chemin PLY.

    Retourne un tuple (path_str, status, graph_or_none) :
      - status == "ok"   : graphe valide
      - status == "skip" : aucun label valide
      - status == "err"  : parsing / ply_to_graph a échoué
    """
    try:
        g = ply_to_graph(Path(path_str))
    except Exception:
        return path_str, "err", None
    if g is None:
        return path_str, "err", None
    if (g.y >= 0).sum().item() == 0:
        return path_str, "skip", None
    return path_str, "ok", g


def _worker_partial_welford(shard_path_str: str):
    """Welford partiel sur un shard. Retourne (n, mean, M2)."""
    graphs = torch.load(Path(shard_path_str), weights_only=False)
    acc = _WelfordAccumulator(NUM_FEATURES)
    for g in graphs:
        acc.update_batch(g.x.numpy())
    return acc.n, acc.mean, acc.M2


def _combine_welford(a, b):
    """Combine deux accumulateurs Welford : (n, mean, M2)."""
    nA, meanA, M2_A = a
    nB, meanB, M2_B = b
    n = nA + nB
    if n == 0:
        return 0, np.zeros_like(meanA), np.zeros_like(M2_A)
    delta = meanB - meanA
    mean  = meanA + delta * (nB / n)
    M2    = M2_A + M2_B + (delta ** 2) * (nA * nB / n)
    return n, mean, M2


def _worker_normalize_shard(args):
    """Normalise un shard in-place. args = (shard_path_str, mean_np, std_np)."""
    shard_path_str, mean_np, std_np = args
    shard_path = Path(shard_path_str)
    mean_t = torch.from_numpy(mean_np)
    std_t  = torch.from_numpy(std_np)
    graphs: list[Data] = torch.load(shard_path, weights_only=False)
    for g in graphs:
        g.x = ((g.x - mean_t) / std_t).float()
    torch.save(graphs, shard_path)
    return shard_path.name


def compute_stats_parallel(shard_dir: Path, n_workers: int) -> tuple[np.ndarray, np.ndarray]:
    """Welford parallèle sur les shards. Pic mémoire = n_workers shards."""
    shards = sorted(shard_dir.glob("shard_*.pt"))
    log.info(f"Stats : {len(shards)} shards | {n_workers} workers")

    n_total = 0
    mean    = np.zeros(NUM_FEATURES, dtype=np.float64)
    M2      = np.zeros(NUM_FEATURES, dtype=np.float64)

    if n_workers <= 1 or len(shards) <= 1:
        for s in tqdm(shards, desc="welford", unit="shard"):
            n_total, mean, M2 = _combine_welford(
                (n_total, mean, M2), _worker_partial_welford(str(s))
            )
    else:
        with mp.Pool(processes=n_workers) as pool:
            it = pool.imap_unordered(_worker_partial_welford,
                                     [str(s) for s in shards])
            for partial in tqdm(it, total=len(shards), desc="welford", unit="shard"):
                n_total, mean, M2 = _combine_welford((n_total, mean, M2), partial)

    if n_total < 2:
        return mean.astype(np.float32), np.ones(NUM_FEATURES, dtype=np.float32)
    variance = M2 / (n_total - 1)
    std      = np.sqrt(np.maximum(variance, 1e-8))
    log.info(f"Stats calculées sur {n_total:,} nœuds.")
    return mean.astype(np.float32), std.astype(np.float32)


def normalize_shards_parallel(
    shard_dir: Path, mean: np.ndarray, std: np.ndarray, n_workers: int
) -> None:
    """Applique la normalisation aux shards en parallèle."""
    shards = sorted(shard_dir.glob("shard_*.pt"))
    log.info(f"Normalisation : {len(shards)} shards | {n_workers} workers")
    args_list = [(str(s), mean, std) for s in shards]

    if n_workers <= 1 or len(shards) <= 1:
        for a in tqdm(args_list, desc="normalize", unit="shard"):
            _worker_normalize_shard(a)
    else:
        with mp.Pool(processes=n_workers) as pool:
            it = pool.imap_unordered(_worker_normalize_shard, args_list)
            for _ in tqdm(it, total=len(shards), desc="normalize", unit="shard"):
                pass


# ─────────────────────────────────────────────────────────────────────────────
# Manifest (resume robuste, indépendant de l'ordre de traitement)
# ─────────────────────────────────────────────────────────────────────────────

def _load_manifest(shard_dir: Path) -> set[str]:
    p = shard_dir / MANIFEST_NAME
    if not p.exists():
        return set()
    return {ln.strip() for ln in p.read_text().splitlines() if ln.strip()}


def _bootstrap_manifest_from_shards(shard_dir: Path) -> set[str]:
    """Reconstruit un manifest depuis les shards existants (compat avec l'ancien
    mode séquentiel sans manifest). On dérive le chemin PLY attendu depuis
    map_name + drone_id + drone_pos, en s'appuyant sur la convention de nommage."""
    done: set[str] = set()
    shards = sorted(shard_dir.glob("shard_*.pt"))
    if not shards:
        return done
    log.info(f"Aucun manifest : reconstruction depuis {len(shards)} shards existants...")
    for s in tqdm(shards, desc="scan-shards", unit="shard"):
        try:
            graphs: list[Data] = torch.load(s, weights_only=False)
        except Exception:
            continue
        for g in graphs:
            map_name  = getattr(g, "map_name", None)
            drone_id  = getattr(g, "drone_id", None)
            drone_pos = getattr(g, "drone_pos", None)
            if map_name is None or drone_id is None or drone_pos is None:
                continue
            x, y, z = (float(v) for v in drone_pos.tolist())
            # Convention : NoiseMap_{map}_{x}_{y}_{z}_{drone}.ply, sous {drone}/
            fname = f"NoiseMap_{map_name}_{x}_{y}_{z}_{drone_id}.ply"
            done.add(fname)   # on stocke juste le nom de fichier (cf. _resume_filter)
    log.info(f"Manifest reconstruit : {len(done)} entrées (par nom de fichier).")
    return done


def _resume_filter(ply_files: list[Path], done: set[str]) -> list[Path]:
    """Retourne les PLYs non encore traités. Accepte un manifest contenant
    soit des chemins absolus, soit des noms de fichier (mode bootstrap)."""
    if not done:
        return list(ply_files)
    out = []
    for p in ply_files:
        if str(p) in done or p.name in done:
            continue
        out.append(p)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Boucle principale
# ─────────────────────────────────────────────────────────────────────────────

def _check_shard_feature_dim(shard_dir: Path) -> None:
    """If shards already exist, make sure their feature dim matches the
    current NUM_FEATURES. Catches the common pitfall of resuming on top of
    shards built by a previous version of `_build_node_features`."""
    existing = sorted(shard_dir.glob("shard_*.pt"))
    if not existing:
        return
    try:
        sample = torch.load(existing[0], weights_only=False)
    except Exception as exc:
        log.warning(f"Could not introspect {existing[0].name}: {exc}")
        return
    if not sample:
        return
    got = int(sample[0].x.shape[1])
    if got != NUM_FEATURES:
        log.error(
            f"Existing shards have x with {got} feature columns, but the "
            f"current code emits {NUM_FEATURES} (FEAT_KEYS changed). The two "
            f"are not compatible — Welford / normalisation would crash or "
            f"produce wrong stats.\n"
            f"  Fix: re-run with --no-resume to discard {shard_dir} and "
            f"rebuild from scratch."
        )
        sys.exit(2)


def build_dataset(
    generated_dir: Path,
    shard_dir:     Path,
    resume:        bool = True,
    normalize:     bool = True,
    num_workers:   int  = 1,
) -> None:
    """
    Construit le dataset shardé depuis les fichiers PLY générés.

    Architecture attendue :
        generated_dir/
            {drone_name}/
                NoiseMap_{ville}_{x}_{y}_{z}_{drone_name}.ply
                ...
    """
    _check_shard_feature_dim(shard_dir)
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

    shard_dir.mkdir(parents=True, exist_ok=True)

    # ── Resume via manifest ───────────────────────────────────────────────────
    done: set[str] = set()
    if resume:
        done = _load_manifest(shard_dir)
        if not done and any(shard_dir.glob("shard_*.pt")):
            # Compat avec l'ancien build séquentiel sans manifest
            done = _bootstrap_manifest_from_shards(shard_dir)

    todo = _resume_filter(ply_files, done)
    log.info(
        f"Resume : {len(done):,} déjà traités | "
        f"{len(todo):,} à traiter | total {len(ply_files):,}"
    )

    if not todo:
        log.info("Rien à faire — tous les PLYs sont dans le manifest.")
        if normalize:
            _run_normalize(shard_dir, num_workers)
        return

    # ── Tri par map_name pour la localité du cache BVH (chaque worker
    # voit des batches contigus de la même map → 1 chargement BVH partagé)
    def _map_key(p: Path) -> str:
        m = FNAME_RE.match(p.name)
        return m.group("map") if m else ""

    todo.sort(key=_map_key)

    # ── ShardWriter : on continue à la suite des shards existants ──────────────
    existing_shards = sorted(shard_dir.glob("shard_*.pt"))
    writer = ShardWriter(shard_dir, SHARD_SIZE)
    writer.shard_index = len(existing_shards)
    log.info(
        f"Reprise : shard_index initial = {writer.shard_index} "
        f"({len(existing_shards)} shards déjà présents)"
    )

    # ── Boucle parallèle ──────────────────────────────────────────────────────
    n_workers = max(1, num_workers)
    chunksize = max(1, min(64, len(todo) // (n_workers * 4) or 1))
    log.info(f"Build : {n_workers} workers | chunksize={chunksize}")

    n_ok = n_skip = n_err = 0
    todo_strs = [str(p) for p in todo]
    manifest_path = shard_dir / MANIFEST_NAME

    pbar = tqdm(total=len(todo_strs), desc="build", unit="ply", dynamic_ncols=True)

    def _consume(iterator):
        nonlocal n_ok, n_skip, n_err
        with manifest_path.open("a") as manifest_f:
            for path_str, status, g in iterator:
                if status == "ok":
                    writer.add(g)
                    manifest_f.write(f"{path_str}\n")
                    n_ok += 1
                elif status == "skip":
                    n_skip += 1
                else:
                    n_err += 1
                pbar.update(1)
                if (n_ok + n_skip + n_err) % LOG_EVERY == 0:
                    pbar.set_postfix(ok=n_ok, skip=n_skip, err=n_err)
                    manifest_f.flush()

    try:
        if n_workers == 1:
            _consume(_worker_build_graph(s) for s in todo_strs)
        else:
            with mp.Pool(processes=n_workers) as pool:
                _consume(pool.imap_unordered(
                    _worker_build_graph, todo_strs, chunksize=chunksize
                ))
    finally:
        pbar.close()

    total = writer.close()
    log.info(
        f"\n{'─'*60}\n"
        f"Build terminé : {total:,} nouveaux graphes "
        f"dans {writer.shard_index} shards (cumul)\n"
        f"  ok={n_ok}  skip={n_skip}  err={n_err}\n"
        f"Sortie : {shard_dir}\n"
        f"{'─'*60}"
    )

    if total == 0 and not existing_shards:
        log.error("Aucun graphe valide produit. Vérifiez les PLY.")
        sys.exit(1)

    if normalize:
        _run_normalize(shard_dir, num_workers)
    else:
        log.info("Normalisation ignorée (--no-normalize).")


def _run_normalize(shard_dir: Path, num_workers: int) -> None:
    """Stats + normalisation in-place, en parallèle."""
    log.info("\n── Passe de normalisation (parallèle) ──")
    mean, std = compute_stats_parallel(shard_dir, max(1, num_workers))
    save_stats_json(mean, std, FEAT_KEYS, STATS_FILE)
    normalize_shards_parallel(shard_dir, mean, std, max(1, num_workers))
    log.info("Normalisation terminée.")


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
    p.add_argument(
        "--num-workers", type=int, default=os.cpu_count() or 1,
        help="Nombre de processus pour le build / les stats / la normalisation "
             "(default: %(default)s = tous les cœurs CPU). 1 = mode séquentiel.",
    )
    return p.parse_args()

if __name__ == "__main__":
    args = _parse_args()

    shard_dir = args.out_dir / "shards"

    if args.no_resume and shard_dir.exists():
        import shutil
        log.info(f"--no-resume : suppression de {shard_dir} (manifest inclus)")
        shutil.rmtree(shard_dir)

    n_workers = max(1, args.num_workers)
    n_cpu     = os.cpu_count() or 1
    n_gpu     = 0
    try:
        import torch as _t
        if _t.cuda.is_available():
            n_gpu = _t.cuda.device_count()
    except Exception:
        pass
    log.info(
        f"Ressources : {n_cpu} CPU(s), {n_gpu} GPU(s) | "
        f"workers utilisés = {n_workers}"
    )
    if n_gpu > 0:
        log.info(
            "Note : ce script est CPU-bound (parsing PLY + ray-cast BVH + numpy). "
            "Le GPU n'apporterait qu'un gain marginal sur le ray-casting et serait "
            "annulé par les transferts ; on sature donc les CPU à la place."
        )

    build_dataset(
        generated_dir = args.generated_dir,
        shard_dir     = shard_dir,
        resume        = not args.no_resume,
        normalize     = not args.no_normalize,
        num_workers   = n_workers,
    )
