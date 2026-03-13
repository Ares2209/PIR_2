"""Fonctions de perte pour la prédiction SPL de drone sur maillage urbain.

DroneNoiseLoss : MSE sur les nœuds réels + régularisation de lissage spatial.
La régularisation pénalise les discontinuités abruptes de SPL entre nœuds
adjacents, physiquement justifié par la continuité de la propagation acoustique.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class DroneNoiseLoss(nn.Module):
    """
    MSE masqué + régularisation de lissage spatial (λ_smooth).

    La perte principale est calculée uniquement sur les nœuds réels (mask=True),
    ce qui permet de gérer les batches avec padding (scènes de tailles variables).

    La régularisation spatiale contraint les prédictions voisines à être cohérentes :
        L_smooth = MSE(h_i, mean_{j ∈ N(i)} h_j)
    ce qui correspond à un prior de continuité acoustique.

    Args:
        smooth_weight: Coefficient λ_smooth. 0.0 = MSE pur.
    """

    def __init__(self, smooth_weight: float = 0.01):
        super().__init__()
        self.smooth_weight = smooth_weight

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor,
        adj: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            logits:  (B, N, 1) ou (B, N) — prédictions brutes
            targets: (B, N)               — labels SPL normalisés
            mask:    (B, N) bool          — True = nœud réel (pas de padding)
            adj:     (B, N, N) bool       — adjacence KNN (optionnel, pour lissage)

        Returns:
            Scalaire — perte totale.
        """
        if logits.dim() == 3:
            logits = logits.squeeze(-1)   # (B, N)

        float_mask = mask.float()
        n_valid = float_mask.sum().clamp(min=1)

        # Perte principale : MSE sur nœuds réels uniquement
        loss = (F.mse_loss(logits, targets, reduction='none') * float_mask).sum() / n_valid

        # Régularisation spatiale : continuité acoustique entre voisins
        if adj is not None and self.smooth_weight > 0.0:
            deg = adj.float().sum(-1).clamp(min=1)            # (B, N)
            neighbor_mean = (
                torch.bmm(adj.float(), logits.unsqueeze(-1)).squeeze(-1) / deg
            )   # (B, N)
            smooth = F.mse_loss(
                logits * float_mask,
                neighbor_mean.detach() * float_mask,
            )
            loss = loss + self.smooth_weight * smooth

        return loss


def build_loss(smooth_weight: float = 0.01) -> DroneNoiseLoss:
    """Instancie la perte depuis un scalaire (compatible TrainingConfig.smooth_weight)."""
    return DroneNoiseLoss(smooth_weight=smooth_weight)
