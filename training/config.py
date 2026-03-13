"""Configuration du projet DroneNoiseGAT.

Source de vérité unique pour tous les hyperparamètres.
Importé par model.py, drone_noise_dataset.py et main.py.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch


# ─────────────────────────────────────────────────────────────────────────────
# DRONE
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DroneConfig:
    """Paramètres physiques du drone source."""

    max_speed_ms: float = 20.0
    """Vitesse maximale en m/s."""

    sound_power_db: float = 85.0
    """Puissance acoustique de référence en dB."""

    flight_altitude_m: float = 50.0
    """Altitude de vol par défaut en mètres."""

    rotor_frequency_hz: float = 100.0
    """Fréquence fondamentale des rotors en Hz."""

    def __post_init__(self):
        if self.max_speed_ms <= 0:
            raise ValueError("max_speed_ms doit être > 0")
        if self.flight_altitude_m <= 0:
            raise ValueError("flight_altitude_m doit être > 0")
        if self.rotor_frequency_hz <= 0:
            raise ValueError("rotor_frequency_hz doit être > 0")

    def geometric_attenuation_db(self, distance_m: float) -> float:
        """
        Atténuation géométrique (loi en 1/r²) en dB.
        ΔL = 20 * log10(r)  (ref : 1 m)
        """
        if distance_m <= 0:
            raise ValueError("distance_m doit être > 0")
        return 20.0 * math.log10(max(distance_m, 1.0))

    def spl_at_distance(self, distance_m: float) -> float:
        """SPL estimé à une distance donnée (sans obstacles)."""
        return self.sound_power_db - self.geometric_attenuation_db(distance_m)


# ─────────────────────────────────────────────────────────────────────────────
# MODÈLE GAT
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ModelConfig:
    """Hyperparamètres du modèle DroneNoiseGAT."""

    # Entrée
    input_dim: int = 13
    """Dimension des features par nœud — doit correspondre à DroneNoiseMeshDataset.NUM_FEATURES."""

    # Architecture GAT (Veličković et al., ICLR 2018)
    hidden_dim: int = 64
    """Features par tête d'attention dans les couches intermédiaires."""

    num_heads: int = 8
    """Nombre de têtes d'attention K (eq. 5 du papier)."""

    num_layers: int = 3
    """Nombre de couches GAT empilées (dont 1 couche de sortie)."""

    output_dim: int = 1
    """Dimension de sortie par nœud (1 = SPL scalaire)."""

    output_heads: int = 1
    """Têtes sur la couche de sortie — moyennées, pas concaténées (eq. 6)."""

    # Régularisation
    dropout: float = 0.1
    """Dropout sur les features et les coefficients d'attention."""

    use_skip_connections: bool = True
    """Skip connections entre couches (Section 3.3 du papier)."""

    activation: str = "elu"
    """Activation intermédiaire : 'elu' | 'relu' | 'leaky_relu'."""

    leaky_relu_slope: float = 0.2
    """Pente négative du LeakyReLU du mécanisme d'attention."""

    use_layer_norm: bool = True
    """LayerNorm après chaque couche GAT."""

    # Tâche
    task: str = "regression"
    """'regression' pour SPL en dB continu."""

    def __post_init__(self):
        if self.activation not in ("elu", "relu", "leaky_relu"):
            raise ValueError(f"activation invalide : '{self.activation}'")
        if self.task not in ("regression", "binary"):
            raise ValueError(f"task invalide : '{self.task}'")
        if self.input_dim != 13:
            import warnings
            warnings.warn(
                f"input_dim={self.input_dim} ≠ 13 (NUM_FEATURES du dataset). "
                "Vérifiez la cohérence.",
                UserWarning,
            )

@dataclass
class TrainingConfig:
    """Hyperparamètres d'entraînement."""

    batch_size: int = 4
    """Nombre de scènes par batch (chaque scène a N nœuds variables)."""

    num_epochs: int = 200

    learning_rate: float = 1e-3

    weight_decay: float = 1e-4

    # Optimizer
    optimizer_betas: tuple = (0.9, 0.999)
    """Betas d'AdamW."""

    optimizer_eps: float = 1e-8
    """Epsilon d'AdamW."""

    # Scheduler
    scheduler: str = "cosine"
    """'cosine' (CosineAnnealingLR) | 'plateau' (ReduceLROnPlateau)."""

    # Early stopping
    patience: int = 30
    """Nombre d'epochs sans amélioration avant arrêt."""

    min_delta: float = 1e-4
    """Amélioration minimale considérée comme significative."""

    # Gradient
    gradient_clip: float = 1.0

    # Perte
    smooth_weight: float = 0.01
    """Poids de la régularisation spatiale entre nœuds voisins."""

    # Checkpointing
    save_every_n_epochs: int = 10
    keep_last_n_checkpoints: int = 3

    # Préchargement
    preload: Optional[str] = None
    """Chemin vers un checkpoint .pt à charger avant l'entraînement."""

    # Multi-GPU
    use_multi_gpu: bool = False
    gpu_ids: Optional[List[int]] = None

    def __post_init__(self):
        if self.batch_size <= 0:
            raise ValueError("batch_size doit être > 0")
        if self.num_epochs <= 0:
            raise ValueError("num_epochs doit être > 0")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate doit être > 0")
        if self.scheduler not in ("cosine", "plateau"):
            raise ValueError(f"scheduler invalide : '{self.scheduler}'")


# ─────────────────────────────────────────────────────────────────────────────
# DONNÉES
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DataConfig:
    """Configuration des données du maillage urbain."""

    data_dir: str = "data/drone_noise"
    """Dossier racine contenant les sous-dossiers centroids/, normals/, etc."""

    # Split
    train_ratio: float = 0.8
    random_seed: int = 42

    # KNN
    k_neighbors_features: int = 64
    """Voisins pour le calcul des features locales dans le dataset."""

    k_neighbors_adj: int = 16
    """Voisins pour la matrice d'adjacence GAT dans le collate_fn."""

    # DataLoader
    num_workers: int = 4
    pin_memory: bool = True

    def __post_init__(self):
        if not 0.0 < self.train_ratio < 1.0:
            raise ValueError("train_ratio doit être dans ]0, 1[")
        if self.k_neighbors_features < self.k_neighbors_adj:
            import warnings
            warnings.warn(
                "k_neighbors_features < k_neighbors_adj : "
                "l'adjacence utilisera plus de voisins que les features.",
                UserWarning,
            )


# ─────────────────────────────────────────────────────────────────────────────
# CHEMINS
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PathsConfig:
    """Chemins de sauvegarde."""

    model_folder: str = "checkpoints"
    model_basename: str = "drone_gat"
    experiment_name: str = "runs/drone_gat_v1"
    results_folder: str = "results"

    def __post_init__(self):
        for p in [self.model_folder, self.experiment_name, self.results_folder]:
            Path(p).mkdir(parents=True, exist_ok=True)

    def checkpoint_path(self, epoch: Optional[int] = None) -> Path:
        """Retourne le chemin vers un checkpoint (epoch=None → best_model)."""
        name = (
            f"{self.model_basename}_epoch{epoch:04d}.pt"
            if epoch is not None
            else f"{self.model_basename}_best.pt"
        )
        return Path(self.model_folder) / name

    def latest_checkpoint(self) -> Optional[Path]:
        """Retourne le checkpoint le plus récent, ou None."""
        import glob, os
        pattern = str(Path(self.model_folder) / f"{self.model_basename}_epoch*.pt")
        files = sorted(glob.glob(pattern), key=os.path.getmtime)
        return Path(files[-1]) if files else None


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG PRINCIPALE
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Config:
    """Configuration principale du projet DroneNoiseGAT."""

    model:    ModelConfig    = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data:     DataConfig     = field(default_factory=DataConfig)
    paths:    PathsConfig    = field(default_factory=PathsConfig)
    drone:    DroneConfig    = field(default_factory=DroneConfig)

    device:   str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    num_gpus: int = 0

    def __post_init__(self):
        if torch.cuda.is_available():
            self.device = "cuda"
            self.num_gpus = torch.cuda.device_count()
            if self.num_gpus > 1 and self.training.use_multi_gpu:
                if self.training.gpu_ids is None:
                    self.training.gpu_ids = list(range(self.num_gpus))
                print(f"  Multi-GPU : {self.num_gpus} GPUs")
                for i in self.training.gpu_ids:
                    print(f"    GPU {i} : {torch.cuda.get_device_name(i)}")
            else:
                self.num_gpus = 1
                print(f"  Single GPU : {torch.cuda.get_device_name(0)}")
        else:
            self.device = "cpu"
            self.num_gpus = 0
            print("  CPU mode")

    # ── Sérialisation ────────────────────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model":    asdict(self.model),
            "training": asdict(self.training),
            "data":     asdict(self.data),
            "paths":    asdict(self.paths),
            "drone":    asdict(self.drone),
            "device":   self.device,
        }

    def save(self, path: str):
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        print(f"  Config sauvegardée : {p}")

    @classmethod
    def load(cls, path: str) -> "Config":
        with open(path, "r") as f:
            d = json.load(f)
        return cls.from_dict(d)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Config":
        return cls(
            model    = ModelConfig(**d.get("model",    {})),
            training = TrainingConfig(**d.get("training", {})),
            data     = DataConfig(**d.get("data",     {})),
            paths    = PathsConfig(**d.get("paths",    {})),
            drone    = DroneConfig(**d.get("drone",    {})),
        )

    def print_summary(self):
        print(f"\n{'='*70}")
        print("  CONFIG — DroneNoiseGAT")
        print(f"{'='*70}")
        print(f"  Modèle   : {self.model.num_layers} couches GAT  |  "
              f"{self.model.num_heads} têtes  |  hidden={self.model.hidden_dim}  |  "
              f"input_dim={self.model.input_dim}")
        print(f"  Tâche    : {self.model.task}  |  skip={self.model.use_skip_connections}  |  "
              f"dropout={self.model.dropout}")
        print(f"  Train    : {self.training.num_epochs} epochs  |  "
              f"lr={self.training.learning_rate:.1e}  |  "
              f"bs={self.training.batch_size}  |  "
              f"scheduler={self.training.scheduler}")
        print(f"  Données  : {self.data.data_dir}  |  "
              f"split={self.data.train_ratio:.0%}/{1-self.data.train_ratio:.0%}  |  "
              f"k_adj={self.data.k_neighbors_adj}")
        print(f"  Drone    : {self.drone.sound_power_db} dBW  |  "
              f"alt={self.drone.flight_altitude_m} m")
        print(f"  Device   : {self.device}  ({self.num_gpus} GPU(s))")
        print(f"{'='*70}\n")