"""
Graph Attention Network (GAT) pour la prédiction de l'empreinte sonore de drones.

Architecture : Veličković et al., "Graph Attention Networks", ICLR 2018.
ModelConfig est défini dans config.py (source de vérité unique).
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from training.config import ModelConfig
from training.Loss import DroneNoiseLoss


# ---------------------------------------------------------------------------
# Couche d'attention graphique
# ---------------------------------------------------------------------------

class GATLayer(nn.Module):
    """
    Couche d'attention graphique multi-têtes.

    Équations du papier :
        e_ij = LeakyReLU( ~a^T [Wh_i || Wh_j] )          (eq. 3)
        α_ij = softmax_j(e_ij)                             (eq. 2)
        h'_i = σ( Σ_j α_ij W h_j )                        (eq. 4)
        → concaténation des K têtes (eq. 5) ou moyenne (eq. 6)
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        leaky_slope: float = 0.2,
        concat_heads: bool = True,
        use_layer_norm: bool = True,
        use_skip: bool = True,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.out_dim = out_dim
        self.concat_heads = concat_heads

        # Transformation linéaire partagée W ∈ R^{K·F' × F}
        self.W = nn.Linear(in_dim, num_heads * out_dim, bias=False)

        # Vecteur d'attention ~a ∈ R^{2F'} par tête
        self.a = nn.Parameter(torch.empty(num_heads, 2 * out_dim))

        self.leaky_relu = nn.LeakyReLU(leaky_slope)
        self.dropout = nn.Dropout(dropout)

        out_features = num_heads * out_dim if concat_heads else out_dim
        self.layer_norm = nn.LayerNorm(out_features) if use_layer_norm else None

        self.use_skip = use_skip
        if use_skip:
            self.skip_proj = (
                nn.Linear(in_dim, out_features, bias=False)
                if in_dim != out_features else nn.Identity()
            )

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.W.weight, gain=1.0)
        nn.init.xavier_uniform_(self.a.data.view(self.num_heads, 2, self.out_dim))

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:   (B, N, F)        Features des nœuds.
            adj: (B, N, N) bool   Matrice d'adjacence avec self-loops.
        Returns:
            (B, N, K*F') si concat_heads, sinon (B, N, F').
        """
        B, N, _ = x.shape

        # 1. Wh : (B, N, K, F')
        Wh = self.W(x).view(B, N, self.num_heads, self.out_dim)

        # 2. Scores : ~a^T [Wh_i || Wh_j] = a_src · Wh_i + a_dst · Wh_j
        e_src = (Wh * self.a[:, :self.out_dim]).sum(-1)   # (B, N, K)
        e_dst = (Wh * self.a[:, self.out_dim:]).sum(-1)   # (B, N, K)
        e = self.leaky_relu(e_src.unsqueeze(2) + e_dst.unsqueeze(1))  # (B, N, N, K)

        # 3. Masquage des non-voisins
        e = e.masked_fill(~adj.unsqueeze(-1), float('-inf'))

        # 4. α_ij = softmax_j(e_ij)
        alpha = self.dropout(F.softmax(e, dim=2))          # (B, N, N, K)

        # 5. h'_i = Σ_j α_ij · Wh_j
        h_prime = (alpha.unsqueeze(-1) * Wh.unsqueeze(1)).sum(2)  # (B, N, K, F')

        # 6. Concat (eq. 5) ou moyenne (eq. 6)
        if self.concat_heads:
            h_prime = h_prime.reshape(B, N, self.num_heads * self.out_dim)
        else:
            h_prime = h_prime.mean(2)

        if self.use_skip:
            h_prime = h_prime + self.skip_proj(x)
        if self.layer_norm is not None:
            h_prime = self.layer_norm(h_prime)

        return h_prime


# ---------------------------------------------------------------------------
# Modèle complet
# ---------------------------------------------------------------------------

def _make_activation(name: str) -> nn.Module:
    options = {"elu": nn.ELU(), "relu": nn.ReLU(), "leaky_relu": nn.LeakyReLU(0.2)}
    if name not in options:
        raise ValueError(f"Activation inconnue '{name}'. Options : {list(options)}")
    return options[name]


class DroneNoiseGAT(nn.Module):
    """
    GAT multi-couches pour la prédiction du SPL drone sur maillage urbain.

    Entrée  : (B, N, input_dim)   features géométrico-acoustiques
    Adj     : (B, N, N) bool      matrice d'adjacence KNN avec self-loops
    Sortie  : (B, N, output_dim)  logits SPL par nœud
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.gat_layers = nn.ModuleList()
        self.inter_acts = nn.ModuleList()

        in_dim = config.input_dim

        # Couches intermédiaires : concaténation des têtes (eq. 5)
        for _ in range(config.num_layers - 1):
            self.gat_layers.append(GATLayer(
                in_dim=in_dim,
                out_dim=config.hidden_dim,
                num_heads=config.num_heads,
                dropout=config.dropout,
                leaky_slope=config.leaky_relu_slope,
                concat_heads=True,
                use_layer_norm=config.use_layer_norm,
                use_skip=config.use_skip_connections,
            ))
            self.inter_acts.append(nn.Sequential(
                _make_activation(config.activation),
                nn.Dropout(config.dropout),
            ))
            in_dim = config.num_heads * config.hidden_dim

        # Couche de sortie : moyenne des têtes (eq. 6)
        self.gat_layers.append(GATLayer(
            in_dim=in_dim,
            out_dim=config.hidden_dim,
            num_heads=config.output_heads,
            dropout=config.dropout,
            leaky_slope=config.leaky_relu_slope,
            concat_heads=False,
            use_layer_norm=config.use_layer_norm,
            use_skip=config.use_skip_connections,
        ))

        # Tête de prédiction MLP
        self.head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim // 2, config.output_dim),
        )

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:   (B, N, input_dim)
            adj: (B, N, N) bool
        Returns:
            (B, N, output_dim) — logits bruts.
        """
        for layer, act in zip(self.gat_layers[:-1], self.inter_acts):
            x = act(layer(x, adj))
        x = self.gat_layers[-1](x, adj)
        return self.head(x)

    def predict(
        self,
        x: torch.Tensor,
        adj: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Inférence avec dénormalisation optionnelle via le masque de padding."""
        out = self.forward(x, adj).squeeze(-1)
        if self.config.task == "binary":
            out = torch.sigmoid(out)
        if mask is not None:
            out = out * mask.float()
        return out

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# DroneNoiseLoss est défini dans training/Loss.py et importé en tête de fichier.

def build_model(config: ModelConfig) -> DroneNoiseGAT:
    """
    Instancie DroneNoiseGAT depuis un ModelConfig (issu de config.py).

    Usage :
        from config import Config
        from model import build_model
        cfg = Config()
        model = build_model(cfg.model)
    """
    model = DroneNoiseGAT(config)
    print(
        f"  DroneNoiseGAT — {model.num_parameters:,} paramètres\n"
        f"    {config.num_layers} couches  |  {config.num_heads} têtes  |  "
        f"hidden={config.hidden_dim}  |  task={config.task}"
    )
    return model