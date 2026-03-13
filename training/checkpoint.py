"""Module de gestion des checkpoints — DroneNoiseGAT (régression SPL)."""

import glob
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import torch


class CheckpointManager:
    """
    Gestionnaire de checkpoints avec sauvegarde automatique.

    Suit le meilleur modèle selon val_loss (MSE, à minimiser) et conserve
    les N derniers checkpoints périodiques pour économiser l'espace disque.
    """

    def __init__(
        self,
        model_folder:   str = 'checkpoints',
        model_basename: str = 'drone_gat_',
        keep_last_n:    int = 3,
    ):
        """
        Args:
            model_folder:   Dossier de sauvegarde des checkpoints.
            model_basename: Préfixe des fichiers de checkpoint.
            keep_last_n:    Nombre de checkpoints périodiques à conserver.
        """
        self.model_folder   = Path(model_folder)
        self.model_basename = model_basename
        self.keep_last_n    = keep_last_n

        self.model_folder.mkdir(parents=True, exist_ok=True)

        # Suivi du meilleur score (val_loss → à minimiser)
        self.best_val_loss = float('inf')

        print(f"  CheckpointManager initialisé:")
        print(f"    • Dossier   : {self.model_folder}")
        print(f"    • Basename  : {model_basename}")
        print(f"    • Conserver : {keep_last_n} derniers checkpoints")

    def save_checkpoint(
        self,
        model:     torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any],
        epoch:     int,
        metrics:   Optional[Dict[str, float]] = None,
        history:   Optional[Dict[str, list]]  = None,
        is_best:   bool = False,
        **extra_fields,
    ):
        """
        Sauvegarde un checkpoint complet.

        Args:
            model:      Modèle (supporte DataParallel / DistributedDataParallel).
            optimizer:  Optimizer AdamW.
            scheduler:  LR scheduler (ou None).
            epoch:      Numéro d'epoch courant.
            metrics:    Métriques de validation (val_loss, val_mae, val_rmse, val_r2…).
            history:    Historique complet d'entraînement.
            is_best:    Si True, sauvegarde aussi comme '<basename>best.pth'.
            **extra_fields: Champs supplémentaires (global_step, config, …).
        """
        # Support DataParallel / DistributedDataParallel
        if isinstance(model, (torch.nn.DataParallel,
                               getattr(torch.nn.parallel, 'DistributedDataParallel',
                                       torch.nn.DataParallel))):
            state_dict = model.module.state_dict()
        else:
            state_dict = model.state_dict()

        checkpoint = {
            'epoch':                epoch,
            'model_state_dict':     state_dict,
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'metrics':              metrics or {},
            'history':              history or {},
            'timestamp':            datetime.now().isoformat(),
        }
        if extra_fields:
            checkpoint.update(extra_fields)

        # Checkpoint périodique
        epoch_path = self.model_folder / f"{self.model_basename}{epoch:03d}.pth"
        torch.save(checkpoint, epoch_path)

        # Meilleur modèle
        if is_best:
            best_path = self.model_folder / f"{self.model_basename}best.pth"
            torch.save(checkpoint, best_path)
            m = metrics or {}
            self.best_val_loss = m.get('val_loss', extra_fields.get('val_loss', float('inf')))

        self._cleanup_old_checkpoints()

    def _cleanup_old_checkpoints(self):
        """Supprime les anciens checkpoints périodiques (hors 'best')."""
        if self.keep_last_n <= 0:
            return

        pattern = str(self.model_folder / f"{self.model_basename}*.pth")
        checkpoints = [
            f for f in glob.glob(pattern)
            if 'best' not in os.path.basename(f)
        ]
        checkpoints.sort(key=os.path.getmtime)

        if len(checkpoints) > self.keep_last_n:
            for path in checkpoints[:-self.keep_last_n]:
                try:
                    os.remove(path)
                except Exception as e:
                    print(f"    Erreur suppression {path}: {e}")

    def load_checkpoint(
        self,
        checkpoint_name: str = 'best',
        device:          str = 'cpu',
    ) -> Dict[str, Any]:
        """
        Charge un checkpoint.

        Args:
            checkpoint_name: 'best', 'latest', ou numéro d'epoch ex. '005'.
            device:          Device cible ('cpu', 'cuda', …).

        Returns:
            Dictionnaire du checkpoint.
        """
        if checkpoint_name == 'latest':
            checkpoint_path = self.get_latest_checkpoint()
            if checkpoint_path is None:
                raise FileNotFoundError("Aucun checkpoint périodique trouvé.")
        else:
            checkpoint_path = self.model_folder / f"{self.model_basename}{checkpoint_name}.pth"

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint introuvable : {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=device)

        m = checkpoint.get('metrics', {}) or {}
        print(f"  Checkpoint chargé : {checkpoint_path}")
        print(f"    • Epoch    : {checkpoint.get('epoch', '?')}")
        print(f"    • val_loss : {m.get('val_loss', '?')}")
        print(f"    • val_mae  : {m.get('val_mae',  '?')} dB")

        return checkpoint

    def get_latest_checkpoint(self) -> Optional[Path]:
        """Retourne le chemin du checkpoint périodique le plus récent."""
        pattern = str(self.model_folder / f"{self.model_basename}*.pth")
        checkpoints = [
            f for f in glob.glob(pattern)
            if 'best' not in os.path.basename(f)
        ]
        if not checkpoints:
            return None
        return Path(max(checkpoints, key=os.path.getmtime))

    def get_best_checkpoint_path(self) -> Path:
        """Retourne le chemin du meilleur checkpoint."""
        return self.model_folder / f"{self.model_basename}best.pth"

    def checkpoint_exists(self, checkpoint_name: str = 'best') -> bool:
        """Vérifie si un checkpoint existe."""
        return (self.model_folder / f"{self.model_basename}{checkpoint_name}.pth").exists()


def load_model_from_checkpoint(
    model:           torch.nn.Module,
    checkpoint_path: str,
    device:          str = 'cpu',
    load_optimizer:  bool = False,
    optimizer:       Optional[torch.optim.Optimizer] = None,
    scheduler:       Optional[Any] = None,
) -> tuple:
    """
    Charge un modèle DroneNoiseGAT depuis un checkpoint (fonction utilitaire).

    Args:
        model:           Modèle instancié (non chargé).
        checkpoint_path: Chemin vers le fichier .pth.
        device:          Device cible.
        load_optimizer:  Si True, restaure aussi l'optimizer et le scheduler.
        optimizer:       Optimizer (requis si load_optimizer=True).
        scheduler:       LR scheduler (optionnel).

    Returns:
        (model, history) — history peut être None si absent du checkpoint.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)

    m = checkpoint.get('metrics', {}) or {}
    print(f"  Modèle chargé : {checkpoint_path}")
    print(f"    • Epoch    : {checkpoint.get('epoch', '?')}")
    print(f"    • val_loss : {m.get('val_loss', '?')}")
    print(f"    • val_mae  : {m.get('val_mae',  '?')} dB")
    print(f"    • val_r2   : {m.get('val_r2',   '?')}")

    if load_optimizer and optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if scheduler is not None and checkpoint.get('scheduler_state_dict'):
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        print(f"    • Optimizer/Scheduler restaurés")

    return model, checkpoint.get('history', None)
