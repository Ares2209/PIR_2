"""Métriques de régression pour la prédiction SPL (dB) de drone sur maillage urbain.

Remplace les métriques de classification binaire (F1, MCC, IoU…) par des
métriques adaptées à la régression continue en dB :
    - MAE  : erreur absolue moyenne (dB)
    - RMSE : racine de l'erreur quadratique moyenne (dB)
    - R²   : coefficient de détermination
    - MAPE : erreur relative moyenne (%)
    - within_NdB : % de prédictions à moins de N dB de la vraie valeur
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Optional

import numpy as np
import torch


# ─────────────────────────────────────────────────────────────────────────────
# Conteneur de métriques
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RegressionMetrics:
    """Métriques de régression SPL pour un epoch ou un split."""

    mae:        float   # Erreur absolue moyenne (dB)
    rmse:       float   # Racine erreur quadratique moyenne (dB)
    r2:         float   # Coefficient de détermination R²
    mape:       float   # Erreur absolue relative moyenne (%)
    within_1dB: float   # % prédictions à ±1 dB
    within_3dB: float   # % prédictions à ±3 dB
    within_5dB: float   # % prédictions à ±5 dB
    n_samples:  int     # Nombre de nœuds évalués

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)

    def __str__(self) -> str:
        return (
            f"MAE={self.mae:.3f} dB  RMSE={self.rmse:.3f} dB  R²={self.r2:.4f}  "
            f"MAPE={self.mape:.1f}%  "
            f"±1dB={self.within_1dB:.1f}%  ±3dB={self.within_3dB:.1f}%  "
            f"n={self.n_samples:,}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Calcul des métriques depuis tableaux numpy
# ─────────────────────────────────────────────────────────────────────────────

def compute_regression_metrics(
    pred: np.ndarray,
    true: np.ndarray,
) -> RegressionMetrics:
    """
    Calcule les métriques de régression SPL.

    Args:
        pred: Prédictions en dB, shape (N,)
        true: Valeurs vraies en dB, shape (N,)

    Returns:
        RegressionMetrics
    """
    assert pred.shape == true.shape, (
        f"Shape mismatch : pred={pred.shape}, true={true.shape}"
    )

    n       = len(pred)
    abs_err = np.abs(pred - true)
    sq_err  = (pred - true) ** 2

    mae  = float(np.mean(abs_err))
    rmse = float(np.sqrt(np.mean(sq_err)))

    ss_res = float(np.sum(sq_err))
    ss_tot = float(np.sum((true - true.mean()) ** 2))
    r2     = float(1.0 - ss_res / (ss_tot + 1e-10))

    # MAPE : les valeurs SPL peuvent être proches de 0 → on ajoute 1e-3 en référence
    mape = float(np.mean(abs_err / (np.abs(true) + 1e-3)) * 100.0)

    within_1dB = float(np.mean(abs_err <= 1.0) * 100.0)
    within_3dB = float(np.mean(abs_err <= 3.0) * 100.0)
    within_5dB = float(np.mean(abs_err <= 5.0) * 100.0)

    return RegressionMetrics(
        mae=mae, rmse=rmse, r2=r2, mape=mape,
        within_1dB=within_1dB, within_3dB=within_3dB, within_5dB=within_5dB,
        n_samples=n,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Accumulateur par batch → métriques d'epoch
# ─────────────────────────────────────────────────────────────────────────────

class RegressionMetricsAccumulator:
    """
    Accumule prédictions et labels (dénormalisés en dB) sur plusieurs batches,
    puis calcule les métriques finales sur l'ensemble de l'epoch.

    Usage :
        acc = RegressionMetricsAccumulator()
        for batch in loader:
            pred, true, mask = ...
            acc.update(pred, true, mask, spl_mean, spl_std)
        metrics = acc.compute()
        acc.reset()
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self._preds: List[np.ndarray] = []
        self._trues: List[np.ndarray] = []

    @torch.no_grad()
    def update(
        self,
        pred:     torch.Tensor,
        true:     torch.Tensor,
        mask:     Optional[torch.Tensor] = None,
        spl_mean: float = 0.0,
        spl_std:  float = 1.0,
    ):
        """
        Ajoute un batch (dénormalisé en dB avant stockage).

        Args:
            pred:     (B, N) ou (B, N, 1) — prédictions normalisées
            true:     (B, N)              — labels normalisés
            mask:     (B, N) bool         — True = nœud réel (pas de padding)
            spl_mean: Moyenne SPL utilisée à la normalisation (dB)
            spl_std:  Écart-type SPL utilisé à la normalisation (dB)
        """
        if pred.dim() == 3:
            pred = pred.squeeze(-1)

        if mask is not None:
            m         = mask.bool()
            pred_vals = pred[m]
            true_vals = true[m]
        else:
            pred_vals = pred.flatten()
            true_vals = true.flatten()

        # Dénormalisation → dB réels
        pred_db = (pred_vals * spl_std + spl_mean).cpu().float().numpy()
        true_db = (true_vals * spl_std + spl_mean).cpu().float().numpy()

        self._preds.append(pred_db)
        self._trues.append(true_db)

    def compute(self) -> RegressionMetrics:
        """Calcule les métriques finales sur tous les batches accumulés."""
        if not self._preds:
            raise RuntimeError("Aucune donnée accumulée. Appelez update() d'abord.")
        pred_all = np.concatenate(self._preds)
        true_all = np.concatenate(self._trues)
        return compute_regression_metrics(pred_all, true_all)
