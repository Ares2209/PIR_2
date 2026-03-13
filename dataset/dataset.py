"""
Dataset pour la prédiction du SPL (dB) d'un drone sur un maillage urbain.

DataConfig est défini dans config.py (source de vérité unique).
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import numpy as np
import torch
from numba import jit, prange
from scipy.spatial import cKDTree
from torch.utils.data import Dataset

from training.config import DataConfig


# ---------------------------------------------------------------------------
# Fonctions Numba — features acoustiques
# ---------------------------------------------------------------------------

@jit(nopython=True, parallel=True, fastmath=True, cache=True)
def _nb_obstacle_proximity(
    points: np.ndarray,
    blocking_mask: np.ndarray,
    neighbor_dists: np.ndarray,
    neighbor_indices: np.ndarray,
    decay_radius: float,
) -> np.ndarray:
    """
    Proximité au voisin bloquant le plus proche : exp(-d / decay_radius).
    Détecte les effets de diffraction en bord d'obstacle.
    Sortie ∈ [0, 1].
    """
    n = points.shape[0]
    k = neighbor_indices.shape[1]
    out = np.zeros((n, 1), dtype=np.float64)
    for i in prange(n):
        min_d = 1e9
        for t in range(k):
            idx = neighbor_indices[i, t]
            if idx == -1:
                continue
            if blocking_mask[idx]:
                d = neighbor_dists[i, t]
                if d < min_d:
                    min_d = d
        out[i, 0] = 0.0 if min_d > 1e8 else math.exp(-min_d / decay_radius)
    return out


@jit(nopython=True, parallel=True, fastmath=True, cache=True)
def _nb_slope_discontinuity(
    normals: np.ndarray,
    neighbor_indices: np.ndarray,
) -> np.ndarray:
    """
    Discontinuité locale de la normale (≈ courbure discrète).
    Valeur haute = arête vive = zone de diffraction potentielle.
    Sortie ∈ [0, 1].
    """
    n = normals.shape[0]
    k = neighbor_indices.shape[1]
    out = np.zeros((n, 1), dtype=np.float64)
    for i in prange(n):
        nx, ny, nz = normals[i, 0], normals[i, 1], normals[i, 2]
        sum_cos = 0.0
        cnt = 0
        for t in range(1, k):
            idx = neighbor_indices[i, t]
            if idx == -1:
                continue
            cos_sim = nx*normals[idx,0] + ny*normals[idx,1] + nz*normals[idx,2]
            sum_cos += cos_sim
            cnt += 1
        if cnt > 0:
            out[i, 0] = min(1.0, max(0.0, (1.0 - sum_cos / cnt) * 2.0))
    return out


@jit(nopython=True, parallel=True, fastmath=True, cache=True)
def _nb_height_diff(
    points: np.ndarray,
    dist_to_source: np.ndarray,
    neighbor_indices: np.ndarray,
    neighbor_dists: np.ndarray,
    radius: float,
) -> np.ndarray:
    """
    Hauteur max des voisins plus proches de la source que le nœud courant.
    Indique si le nœud est dans l'ombre acoustique d'un obstacle.
    Sortie ∈ [-1, 1].
    """
    n = points.shape[0]
    k = neighbor_indices.shape[1]
    out = np.zeros((n, 1), dtype=np.float64)
    for i in prange(n):
        max_h = -1e9
        for t in range(k):
            idx = neighbor_indices[i, t]
            if idx == -1:
                continue
            if neighbor_dists[i, t] <= radius and dist_to_source[idx] < dist_to_source[i]:
                h = points[idx, 2]
                if h > max_h:
                    max_h = h
        if max_h > -1e8:
            out[i, 0] = min(1.0, max(-1.0, (max_h - points[i, 2]) / 10.0))
    return out


# ---------------------------------------------------------------------------
# Dataset principal
# ---------------------------------------------------------------------------

class DroneNoiseMeshDataset(Dataset):
    """
    Dataset pour la prédiction du SPL (dB) d'un drone sur un maillage urbain.

    Chaque nœud correspond au centroid d'une face triangulaire du maillage.
    Les features sont calculées à partir de la géométrie locale et de la
    position de la source (drone).

    Args:
        centroids:        Liste d'arrays (N, 3) — positions des centroids.
        normals:          Liste d'arrays (N, 3) — normales de faces (unitaires).
        areas:            Liste d'arrays (N,)   — aires des faces (m²).
        source_positions: Liste d'arrays (3,)   — positions du drone.
        spl_labels:       Liste d'arrays (N,)   — SPL en dB par nœud.
        config:           DataConfig depuis config.py (fournit k_neighbors_features).
        spl_mean:         Moyenne SPL pour normalisation (None = pas de normalisation).
        spl_std:          Écart-type SPL pour normalisation.
    """

    NUM_FEATURES: int = 13

    def __init__(
        self,
        centroids: List[np.ndarray],
        normals: List[np.ndarray],
        areas: List[np.ndarray],
        source_positions: List[np.ndarray],
        spl_labels: List[np.ndarray],
        config: DataConfig,
        spl_mean: Optional[float] = None,
        spl_std: Optional[float] = None,
    ):
        assert len(centroids) == len(normals) == len(areas) == \
               len(source_positions) == len(spl_labels), \
            "Toutes les listes doivent avoir la même longueur."

        self.centroids        = centroids
        self.normals          = normals
        self.areas            = areas
        self.source_positions = source_positions
        self.spl_labels       = spl_labels
        self.k_neighbors      = config.k_neighbors_features
        self.spl_mean         = spl_mean
        self.spl_std          = spl_std

    def __len__(self) -> int:
        return len(self.centroids)

    def __getitem__(self, idx: int) -> dict:
        points = self.centroids[idx]
        normals = self.normals[idx]
        areas = self.areas[idx]
        source = self.source_positions[idx]
        spl = self.spl_labels[idx].astype(np.float32)

        features = self._compute_features(points, normals, areas, source)

        if self.spl_mean is not None and self.spl_std is not None:
            spl = (spl - self.spl_mean) / (self.spl_std + 1e-8)

        return {
            'features':   torch.FloatTensor(features),   # (N, NUM_FEATURES)
            'labels':     torch.FloatTensor(spl),         # (N,)
            'points':     torch.FloatTensor(points),      # (N, 3) — pour collate_fn
            'num_points': len(points),
        }

    # ------------------------------------------------------------------
    # Calcul des features géométrico-acoustiques
    # ------------------------------------------------------------------

    def _compute_features(
        self,
        points: np.ndarray,
        normals: np.ndarray,
        areas: np.ndarray,
        source: np.ndarray,
    ) -> np.ndarray:
        """
        Calcule 13 features par nœud, en trois groupes :

        Groupe 1 — géométrie source/nœud (5 features) :
            1-3  Normales de face                           (nx, ny, nz)
            4    cos(θ) normale / direction source
            5    Angle rasant pondéré : (1-|cos θ|)·facing

        Groupe 2 — acoustique (4 features) :
            6    Élévation de la source (arctan normalisé)
            7    log10(distance)        → loi en 1/r²
            8    Ombre acoustique       (hauteur relative)
            9    Proximité obstacle     (diffraction)

        Groupe 3 — géométrie du maillage (4 features) :
            10   Discontinuité de pente (arêtes vives)
            11   cos(θ_horiz) — angle horizontal
            12   log10(aire de la face)
            13   Hauteur absolue normalisée
        """
        k_nn = min(self.k_neighbors, len(points))

        # KD-Tree (une seule construction par scène)
        tree = cKDTree(points)
        dists_k, inds_k = tree.query(points, k=k_nn)
        if inds_k.ndim == 1:
            inds_k  = inds_k.reshape(-1, 1)
            dists_k = dists_k.reshape(-1, 1)
        neigh_idx   = inds_k.astype(np.int64)
        neigh_dists = dists_k.astype(np.float64)

        # Vecteur et distance source → nœud
        vec_to_source = source - points                                        # (N, 3)
        dist_to_source = np.linalg.norm(vec_to_source, axis=1, keepdims=True) # (N, 1)
        dir_to_source  = vec_to_source / (dist_to_source + 1e-8)              # (N, 3)

        # Groupe 1
        cos_angles   = np.sum(normals * dir_to_source, axis=1, keepdims=True) # (N, 1)
        facing_source = (cos_angles > 0).astype(np.float32)
        grazing_angle = (1.0 - np.abs(cos_angles)) * facing_source            # (N, 1)

        # Groupe 2
        horiz_dist    = np.linalg.norm(points[:, :2] - source[:2], axis=1)
        elevation     = np.arctan2(source[2] - points[:, 2], horiz_dist + 1e-8)
        elevation_norm = (elevation / (np.pi / 2)).reshape(-1, 1).astype(np.float32)

        log_dist = np.log10(dist_to_source + 1.0).astype(np.float32)

        height_diff = _nb_height_diff(
            points.astype(np.float64),
            dist_to_source.ravel().astype(np.float64),
            neigh_idx, neigh_dists, radius=10.0,
        ).astype(np.float32)

        cos_block    = np.sum(normals * dir_to_source, axis=1)
        blocking_mask = cos_block > 0.3
        obstacle_proximity = _nb_obstacle_proximity(
            points.astype(np.float64),
            blocking_mask.astype(np.bool_),
            neigh_dists, neigh_idx, decay_radius=3.0,
        ).astype(np.float32)

        # Groupe 3
        slope_disc = _nb_slope_discontinuity(
            normals.astype(np.float64), neigh_idx,
        ).astype(np.float32)

        vec_horiz = vec_to_source.copy(); vec_horiz[:, 2] = 0.0
        vec_horiz /= (np.linalg.norm(vec_horiz, axis=1, keepdims=True) + 1e-8)
        normals_horiz = normals.copy(); normals_horiz[:, 2] = 0.0
        normals_horiz /= (np.linalg.norm(normals_horiz, axis=1, keepdims=True) + 1e-8)
        cos_horiz = np.sum(normals_horiz * vec_horiz, axis=1, keepdims=True).astype(np.float32)

        log_area    = np.log10(areas.reshape(-1, 1) + 1e-6).astype(np.float32)
        z_max       = np.max(np.abs(points[:, 2])) + 1e-8
        height_norm = (points[:, 2] / z_max).reshape(-1, 1).astype(np.float32)

        features = np.concatenate([
            normals,            # 1-3
            cos_angles,         # 4
            grazing_angle,      # 5
            elevation_norm,     # 6
            log_dist,           # 7
            height_diff,        # 8
            obstacle_proximity, # 9
            slope_disc,         # 10
            cos_horiz,          # 11
            log_area,           # 12
            height_norm,        # 13
        ], axis=1)

        assert features.shape[1] == self.NUM_FEATURES
        return features.astype(np.float32)

    # ------------------------------------------------------------------
    # Helpers de normalisation
    # ------------------------------------------------------------------

    @staticmethod
    def compute_spl_stats(spl_labels: List[np.ndarray]) -> Tuple[float, float]:
        """
        Calcule la moyenne et l'écart-type globaux des labels SPL.
        À appeler sur le split d'entraînement uniquement (pas de leakage).

        Usage :
            mean, std = DroneNoiseMeshDataset.compute_spl_stats(train_spl)
            train_ds = DroneNoiseMeshDataset(..., spl_mean=mean, spl_std=std)
            val_ds   = DroneNoiseMeshDataset(..., spl_mean=mean, spl_std=std)
        """
        all_vals = np.concatenate([s.ravel() for s in spl_labels])
        return float(all_vals.mean()), float(all_vals.std())


# ---------------------------------------------------------------------------
# collate_fn — assemble un batch + matrice d'adjacence KNN
# ---------------------------------------------------------------------------

def collate_fn(
    batch: list,
    k_adj: int = 16,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Assemble un batch de scènes de tailles variables et construit
    la matrice d'adjacence KNN pour le GAT.

    Args:
        batch: Liste de dicts retournés par DroneNoiseMeshDataset.__getitem__.
        k_adj: Nombre de voisins pour l'adjacence GAT (cf. DataConfig.k_neighbors_adj).

    Returns:
        features : (B, N_max, NUM_FEATURES)
        labels   : (B, N_max)
        mask     : (B, N_max) bool   — True = nœud réel
        adj      : (B, N_max, N_max) bool — adjacence KNN avec self-loops
    """
    max_N = max(item['num_points'] for item in batch)
    B     = len(batch)
    F     = batch[0]['features'].shape[1]

    features_pad = torch.zeros(B, max_N, F)
    labels_pad   = torch.zeros(B, max_N)
    mask         = torch.zeros(B, max_N, dtype=torch.bool)
    adj          = torch.zeros(B, max_N, max_N, dtype=torch.bool)

    for i, item in enumerate(batch):
        N = item['num_points']
        features_pad[i, :N] = item['features']
        labels_pad[i, :N]   = item['labels']
        mask[i, :N]          = True

        points_np = item['points'].numpy()
        k_eff = min(k_adj + 1, N)

        tree = cKDTree(points_np)
        _, knn_idx = tree.query(points_np, k=k_eff)
        if knn_idx.ndim == 1:
            knn_idx = knn_idx.reshape(-1, 1)

        knn_t   = torch.from_numpy(knn_idx.astype(np.int64))
        sub_adj = torch.zeros(N, N, dtype=torch.bool)
        sub_adj.scatter_(1, knn_t, True)
        sub_adj = sub_adj | sub_adj.t()
        sub_adj.fill_diagonal_(True)          # self-loops requis par GAT (Section 2.1)

        adj[i, :N, :N] = sub_adj

    return features_pad, labels_pad, mask, adj


def make_collate_fn(config: DataConfig):
    """
    Fabrique un collate_fn paramétré depuis DataConfig.
    Compatible avec le multiprocessing du DataLoader (pas de lambda).

    Usage :
        loader = DataLoader(ds, collate_fn=make_collate_fn(cfg.data))
    """
    k = config.k_neighbors_adj

    def _collate(batch):
        return collate_fn(batch, k_adj=k)

    return _collate