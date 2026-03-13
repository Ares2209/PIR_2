"""Module d'entraînement — DroneNoiseGAT (régression SPL).

Structure portée depuis le Trainer de classification de référence,
entièrement adaptée à :
    - une tâche de régression continue (MAE / RMSE / R² en dB)
    - des batchs 4-tenseurs : (features, labels, mask, adj)
    - la perte DroneNoiseLoss (MSE + régularisation spatiale)
    - l'early stopping sur val_loss (↓) au lieu de MCC (↑)

Toute la configuration est lue depuis config.py (source de vérité unique).
"""

from __future__ import annotations

import glob
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR, ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    from torch.utils.tensorboard import SummaryWriter
    _TB_AVAILABLE = True
except ImportError:
    _TB_AVAILABLE = False
    print("  [WARNING] TensorBoard non disponible — logs TB désactivés.")

from training.config import Config
from training.Loss import DroneNoiseLoss
from training.metrique import RegressionMetricsAccumulator
from models.GAT import DroneNoiseGAT

@dataclass
class TrainingState:
    """
    État complet de l'entraînement — sérialisable dans les checkpoints.

    Les séries temporelles vivent dans `history` ; les scalaires sont les
    références pour l'early stopping et le checkpointing.
    `__getattr__` permet d'écrire `self.state.val_loss` comme raccourci
    vers `self.state.history['val_loss']`.
    """

    epoch:            int   = 0
    global_step:      int   = 0
    patience_counter: int   = 0

    best_val_loss: float = float('inf')
    best_val_mae:  float = float('inf')
    best_val_rmse: float = float('inf')
    best_val_r2:   float = -float('inf')

    history: Dict[str, List] = field(default_factory=lambda: {
        'train_loss':       [],
        'val_loss':         [],
        'iteration_losses': [],   # loss par batch train
        'train_mae':        [],
        'val_mae':          [],
        'train_rmse':       [],
        'val_rmse':         [],
        'train_r2':         [],
        'val_r2':           [],
        'lr':               [],
    })

    # ── accès pratique : state.val_loss == state.history['val_loss'] ──────────
    def __getattr__(self, name: str):
        if name == 'history':
            raise AttributeError(name)
        history = object.__getattribute__(self, 'history')
        if name in history:
            return history[name]
        raise AttributeError(f"'TrainingState' n'a pas d'attribut '{name}'")

    def append_epoch(self, prefix: str, metrics: Dict[str, float]) -> None:
        """Ajoute les métriques d'une epoch dans l'historique."""
        for key in ('loss', 'mae', 'rmse', 'r2'):
            hist_key = f'{prefix}_{key}'
            if key in metrics and hist_key in self.history:
                self.history[hist_key].append(metrics[key])

    def get_history(self) -> Dict[str, List]:
        return {k: list(v) for k, v in self.history.items()}

def _regression_metrics(pred: np.ndarray, true: np.ndarray) -> Dict[str, float]:
    """
    MAE, RMSE et R² calculés sur des valeurs dénormalisées en dB.
    Délègue à training.metrique.compute_regression_metrics et retourne un dict
    plat compatible avec TrainingState.append_epoch().
    """
    from training.metrique import compute_regression_metrics
    m = compute_regression_metrics(pred, true)
    return {'mae': m.mae, 'rmse': m.rmse, 'r2': m.r2}

class Trainer:
    """
    Gestionnaire d'entraînement pour DroneNoiseGAT — régression SPL.

    Responsabilités :
        - boucle train / validate avec tqdm + fenêtre glissante pour la pbar
        - schedulers : cosine | plateau | onecycle
          (OneCycleLR dimensionné sur patience×2 pour correspondre à la durée réelle)
        - early stopping sur val_loss avec min_delta
        - checkpointing périodique + best model (optimizer + scheduler inclus)
        - TensorBoard (optionnel)
        - reprise depuis checkpoint (cfg.training.preload)
        - calcul de stats features (algorithme de Chan) sur le train set
        - callback _on_improvement isolé pour lisibilité

    Usage :
        trainer = Trainer(model, train_loader, val_loader, cfg, dataset_info)
        result  = trainer.train()
    """

    def __init__(
        self,
        model:        DroneNoiseGAT,
        train_loader: DataLoader,
        val_loader:   DataLoader,
        cfg:          Config,
        dataset_info: dict,
    ):
        self.cfg          = cfg
        self.device       = cfg.device
        self.dataset_info = dataset_info

        self.model        = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader   = val_loader

        self.criterion = self._setup_criterion()
        self.optimizer = self._setup_optimizer()
        self.scheduler = self._setup_scheduler()

        self.writer = (
            SummaryWriter(cfg.paths.experiment_name) if _TB_AVAILABLE else None
        )

        self.state     = TrainingState()
        self.best_ckpt = cfg.paths.checkpoint_path()   # ← PathsConfig.checkpoint_path()

        # Reprise depuis checkpoint
        if cfg.training.preload:
            self._load_checkpoint(cfg.training.preload)

        # Stats features (non bloquant)
        try:
            self.feature_stats = self._compute_feature_stats()
            print(f"  Stats features calculées : "
                  f"{self.feature_stats['n_nodes']:,} nœuds valides")
        except Exception as e:
            print(f"  [WARNING] Stats features non calculées : {e}")
            self.feature_stats = None

        self._print_init_summary()

    def _setup_criterion(self) -> DroneNoiseLoss:
        return DroneNoiseLoss(
            self.cfg.model,
            smooth_weight=self.cfg.training.smooth_weight,
        )

    def _setup_optimizer(self) -> AdamW:
        return AdamW(
            self.model.parameters(),
            lr           = self.cfg.training.learning_rate,
            betas        = self.cfg.training.optimizer_betas,
            eps          = self.cfg.training.optimizer_eps,
            weight_decay = self.cfg.training.weight_decay,
        )

    def _setup_scheduler(self):
        """
        Fabrique le scheduler selon cfg.training.scheduler.

        Pour OneCycleLR : dimensionnement sur patience×2 (horizon réaliste)
        afin que le cycle LR soit adapté à la durée effective d'entraînement
        plutôt qu'à num_epochs (souvent jamais atteint à cause de l'early stopping).
        """
        sched = self.cfg.training.scheduler

        if sched == 'cosine':
            return CosineAnnealingLR(
                self.optimizer,
                T_max   = self.cfg.training.num_epochs,
                eta_min = 1e-6,
            )

        elif sched == 'plateau':
            return ReduceLROnPlateau(
                self.optimizer,
                patience = 10,
                factor   = 0.5,
                verbose  = True,
            )

        elif sched == 'onecycle':
            patience         = self.cfg.training.patience
            realistic_epochs = max(50, patience * 2)
            total_steps      = len(self.train_loader) * realistic_epochs
            print(f"  OneCycleLR : {realistic_epochs} epochs estimées "
                  f"({total_steps} steps, patience={patience})")
            return OneCycleLR(
                self.optimizer,
                max_lr      = self.cfg.training.learning_rate,
                total_steps = total_steps,
                pct_start   = 0.3,
                last_epoch  = -1,
            )

        else:
            raise ValueError(
                f"scheduler inconnu : '{sched}'. Options : cosine | plateau | onecycle"
            )

    def _load_checkpoint(self, path: str):
        """Charge l'état complet depuis un checkpoint (modèle + optimizer + scheduler)."""
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt['state_dict'])
        if 'optimizer_state_dict' in ckpt:
            self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if 'scheduler_state_dict' in ckpt and self.scheduler is not None:
            try:
                self.scheduler.load_state_dict(ckpt['scheduler_state_dict'])
            except Exception:
                pass   # scheduler incompatible (changement de type) → on repart
        self.state.epoch          = ckpt.get('epoch', 0)
        self.state.global_step    = ckpt.get('global_step', 0)
        self.state.best_val_loss  = ckpt.get('val_loss', float('inf'))
        self.state.best_val_mae   = ckpt.get('val_mae',  float('inf'))
        self.state.best_val_rmse  = ckpt.get('val_rmse', float('inf'))
        self.state.best_val_r2    = ckpt.get('val_r2',  -float('inf'))
        print(f"  Checkpoint chargé : {path}  "
              f"(epoch {self.state.epoch}, "
              f"val_loss {self.state.best_val_loss:.4f}, "
              f"val_MAE {self.state.best_val_mae:.2f} dB)")

    def _print_init_summary(self):
        cfg = self.cfg
        sched_name = (
            self.scheduler.__class__.__name__ if self.scheduler else 'None'
        )
        print(f"\n{'='*70}")
        print(f"  Trainer initialisé — DroneNoiseGAT (régression SPL)")
        print(f"{'='*70}")
        print(f"  Device      : {self.device}")
        print(f"  Epochs      : {cfg.training.num_epochs}  |  "
              f"Patience : {cfg.training.patience}  |  "
              f"Min delta : {cfg.training.min_delta}")
        print(f"  Batch       : {cfg.training.batch_size}  |  "
              f"LR : {cfg.training.learning_rate:.1e}  |  "
              f"Scheduler : {sched_name}")
        print(f"  Smooth λ    : {cfg.training.smooth_weight}  |  "
              f"Grad clip : {cfg.training.gradient_clip}")
        print(f"  Optimizer   : AdamW  "
              f"β={cfg.training.optimizer_betas}  "
              f"ε={cfg.training.optimizer_eps:.0e}  "
              f"wd={cfg.training.weight_decay}")
        print(f"  Train       : {len(self.train_loader)} batches  |  "
              f"Val : {len(self.val_loader)} batches")
        print(f"{'='*70}\n")

    def _unpack_batch(
        self, batch: Tuple
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Dépaquète (features, labels, mask, adj) → tout sur self.device.
        Compatible avec le collate_fn de drone_noise_dataset.py.
        """
        features, labels, mask, adj = batch
        return (
            features.to(self.device),
            labels.to(self.device),
            mask.to(self.device),
            adj.to(self.device),
        )

    def _denorm_metrics(
        self,
        pred: torch.Tensor,
        true: torch.Tensor,
        mask: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Dénormalise pred et true en dB via les stats du dataset,
        puis calcule MAE / RMSE / R² sur les nœuds réels (mask=True).
        """
        std  = self.dataset_info['spl_std']
        mean = self.dataset_info['spl_mean']
        m    = mask.bool()
        pred_db = (pred[m] * std + mean).cpu().float().numpy()
        true_db = (true[m] * std + mean).cpu().float().numpy()
        return _regression_metrics(pred_db, true_db)

    def train_epoch(self) -> Dict[str, float]:
        """
        Effectue une epoch d'entraînement complète.

        Fenêtre glissante (window_size=5) sur les derniers batches pour
        afficher des métriques lissées dans la barre de progression tqdm,
        sans attendre la fin de l'epoch.
        """
        self.model.train()
        total_loss   = 0.0
        all_pred, all_true, all_mask = [], [], []
        log_every = max(1, len(self.train_loader) // 10)

        # Fenêtre glissante pour la pbar (évite le bruit batch-à-batch)
        window_pred:  List[torch.Tensor] = []
        window_true:  List[torch.Tensor] = []
        window_mask:  List[torch.Tensor] = []
        window_size   = 5

        pbar = tqdm(
            self.train_loader,
            desc  = f"  Epoch {self.state.epoch} [train]",
            leave = False,
        )

        for batch_idx, batch in enumerate(pbar):
            features, labels, mask, adj = self._unpack_batch(batch)

            self.optimizer.zero_grad()
            logits = self.model(features, adj)                     # (B, N, 1)
            loss   = self.criterion(logits, labels, mask, adj)
            loss.backward()
            nn.utils.clip_grad_norm_(
                self.model.parameters(), self.cfg.training.gradient_clip
            )
            self.optimizer.step()

            # OneCycleLR : step batch-level (en respectant total_steps)
            if isinstance(self.scheduler, OneCycleLR):
                if self.state.global_step < self.scheduler.total_steps:
                    self.scheduler.step()

            batch_loss = loss.item()
            total_loss += batch_loss
            self.state.history['iteration_losses'].append(batch_loss)

            with torch.no_grad():
                pred = logits.squeeze(-1).detach()

                # Accumulateurs globaux (métriques epoch complètes)
                all_pred.append(pred.cpu())
                all_true.append(labels.cpu())
                all_mask.append(mask.cpu())

                # Fenêtre glissante pour l'affichage tqdm
                window_pred.append(pred.cpu())
                window_true.append(labels.cpu())
                window_mask.append(mask.cpu())
                if len(window_pred) > window_size:
                    window_pred.pop(0)
                    window_true.pop(0)
                    window_mask.pop(0)

                w_metrics = self._denorm_metrics(
                    torch.cat(window_pred),
                    torch.cat(window_true),
                    torch.cat(window_mask),
                )

            # TensorBoard itération
            if (batch_idx + 1) % log_every == 0 and self.writer:
                self.writer.add_scalar('Train/Loss_iter',  batch_loss,         self.state.global_step)
                self.writer.add_scalar('Train/MAE_iter',   w_metrics['mae'],   self.state.global_step)
                self.writer.add_scalar('Train/RMSE_iter',  w_metrics['rmse'],  self.state.global_step)
                self.writer.add_scalar('Train/LR',         self._get_lr(),     self.state.global_step)

            self.state.global_step += 1

            pbar.set_postfix({
                'loss': f"{batch_loss:.4f}",
                'MAE':  f"{w_metrics['mae']:.2f} dB",   # lissé sur 5 batches
                'RMSE': f"{w_metrics['rmse']:.2f} dB",
                'lr':   f"{self._get_lr():.2e}",
            })

        # Métriques epoch sur tous les nœuds réels
        metrics = self._denorm_metrics(
            torch.cat(all_pred),
            torch.cat(all_true),
            torch.cat(all_mask),
        )
        metrics['loss'] = total_loss / len(self.train_loader)
        return metrics

    def validate(self) -> Dict[str, float]:
        """Évalue le modèle sur le set de validation."""
        self.model.eval()
        total_loss   = 0.0
        all_pred, all_true, all_mask = [], [], []

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="  Validation", leave=False):
                features, labels, mask, adj = self._unpack_batch(batch)
                logits = self.model(features, adj)
                loss   = self.criterion(logits, labels, mask, adj)
                total_loss += loss.item()

                pred = logits.squeeze(-1)
                all_pred.append(pred.cpu())
                all_true.append(labels.cpu())
                all_mask.append(mask.cpu())

        metrics = self._denorm_metrics(
            torch.cat(all_pred),
            torch.cat(all_true),
            torch.cat(all_mask),
        )
        metrics['loss'] = total_loss / len(self.val_loader)
        return metrics

    def _compute_feature_stats(self) -> Dict[str, Any]:
        """
        Calcule mean et std des features sur le train set via l'algorithme de
        Chan (mise à jour incrémentale, stable numériquement).
        Seuls les nœuds réels (mask=True) sont pris en compte.
        """
        total_n: int               = 0
        mean:    Optional[torch.Tensor] = None
        M2:      Optional[torch.Tensor] = None

        with torch.no_grad():
            for batch in tqdm(
                self.train_loader, desc="  Calcul stats features", leave=False
            ):
                features, _, mask, _ = batch       # (B, N_max, F), (B, N_max)
                B, N, F = features.shape
                m_flat  = mask.reshape(-1)          # (B*N,)
                f_flat  = features.reshape(-1, F).double()
                f_valid = f_flat[m_flat]            # (M, F)

                if f_valid.shape[0] == 0:
                    continue

                b_n    = f_valid.shape[0]
                b_mean = f_valid.mean(0)
                b_M2   = ((f_valid - b_mean) ** 2).sum(0)

                if mean is None:
                    total_n, mean, M2 = b_n, b_mean, b_M2
                else:
                    new_n  = total_n + b_n
                    delta  = b_mean - mean
                    mean   = (total_n * mean + b_n * b_mean) / new_n
                    M2     = M2 + b_M2 + delta ** 2 * (total_n * b_n / new_n)
                    total_n = new_n

        if total_n == 0:
            raise RuntimeError("Aucun nœud valide trouvé dans train_loader")

        std = torch.sqrt(torch.clamp(M2 / total_n, min=1e-8))
        return {'mean': mean.tolist(), 'std': std.tolist(), 'n_nodes': total_n}

    def _save_checkpoint(
        self,
        epoch: int,
        val_metrics: Dict[str, float],
        is_best: bool = False,
    ):
        """Sauvegarde l'état complet (modèle, optimizer, scheduler, config)."""
        payload = {
            'epoch':                epoch,
            'global_step':          self.state.global_step,
            'state_dict':           self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': (
                self.scheduler.state_dict()
                if hasattr(self.scheduler, 'state_dict') else None
            ),
            'val_loss':             val_metrics['loss'],
            'val_mae':              val_metrics['mae'],
            'val_rmse':             val_metrics['rmse'],
            'val_r2':               val_metrics['r2'],
            'dataset_info':         self.dataset_info,
            'feature_stats':        self.feature_stats,
            'config':               self.cfg.to_dict(),
        }

        path  = self.best_ckpt if is_best else self.cfg.paths.checkpoint_path(epoch)
        label = "Meilleur modèle" if is_best else f"Checkpoint epoch {epoch}"
        torch.save(payload, path)
        print(f"  💾 {label} sauvegardé : {path}")

        if not is_best:
            self._prune_old_checkpoints()

    def _prune_old_checkpoints(self):
        """Garde uniquement les N derniers checkpoints périodiques."""
        pattern = str(self.cfg.paths.checkpoint_path(0)).replace('epoch0000', 'epoch*')
        files   = sorted(glob.glob(pattern), key=os.path.getmtime)
        for old in files[:-self.cfg.training.keep_last_n_checkpoints]:
            Path(old).unlink(missing_ok=True)

    def _on_improvement(
        self,
        epoch:       int,
        val_metrics: Dict[str, float],
        delta_loss:  float,
    ):
        """
        Appelé quand val_loss s'améliore de plus de min_delta.
        Met à jour l'état, sauvegarde le best model, log la progression.
        """
        self.state.best_val_loss  = val_metrics['loss']
        self.state.best_val_mae   = val_metrics['mae']
        self.state.best_val_rmse  = val_metrics['rmse']
        self.state.best_val_r2    = val_metrics['r2']
        self.state.patience_counter = 0

        self._log_improvement(epoch, val_metrics, delta_loss)

        try:
            self._save_checkpoint(epoch, val_metrics, is_best=True)
        except Exception as e:
            print(f"  ❌ Erreur sauvegarde best model : {e}")

    def _log_tensorboard(self, epoch: int, train: Dict, val: Dict):
        if not self.writer:
            return
        for tag, key in [
            ('Loss',    'loss'),
            ('MAE_dB',  'mae'),
            ('RMSE_dB', 'rmse'),
            ('R2',      'r2'),
        ]:
            self.writer.add_scalars(tag, {'train': train[key], 'val': val[key]}, epoch)
        self.writer.add_scalar('Learning_Rate', self._get_lr(), epoch)

    def _log_console(
        self,
        epoch:      int,
        epoch_time: float,
        train:      Dict,
        val:        Dict,
    ):
        num_epochs = self.cfg.training.num_epochs
        patience   = self.cfg.training.patience

        print(f"\n{'='*70}")
        print(f"  Epoch {epoch}/{num_epochs} — {epoch_time:.1f}s  |  "
              f"LR {self._get_lr():.2e}  |  "
              f"Patience {self.state.patience_counter}/{patience}")
        print(f"{'='*70}")
        print(f"  {'Métrique':<12}  {'Train':>12}  {'Val':>12}")
        print(f"  {'-'*38}")
        for k, unit in [('loss', ''), ('mae', ' dB'), ('rmse', ' dB'), ('r2', '')]:
            tr = f"{train[k]:.4f}{unit}"
            vl = f"{val[k]:.4f}{unit}"
            print(f"  {k.upper():<12}  {tr:>12}  {vl:>12}")

        # Résumé des bests courants
        print(f"\n  Bests   loss={self.state.best_val_loss:.4f}  "
              f"MAE={self.state.best_val_mae:.2f} dB  "
              f"RMSE={self.state.best_val_rmse:.2f} dB  "
              f"R²={self.state.best_val_r2:.4f}")

    def _log_improvement(self, epoch: int, val: Dict, delta_loss: float):
        print(f"\n{'='*70}")
        print(f"  ✓ AMÉLIORATION — Epoch {epoch}")
        print(f"{'='*70}")
        print(f"    Δ Loss : {delta_loss:+.5f}")
        print(f"    MAE    : {val['mae']:.2f} dB  (best : {self.state.best_val_mae:.2f} dB)")
        print(f"    RMSE   : {val['rmse']:.2f} dB  (best : {self.state.best_val_rmse:.2f} dB)")
        print(f"    R²     : {val['r2']:.4f}        (best : {self.state.best_val_r2:.4f})")

    def _log_no_improvement(self, val: Dict, min_delta: float):
        patience = self.cfg.training.patience
        print(f"  ↔ Patience {self.state.patience_counter}/{patience}  "
              f"val_loss={val['loss']:.5f}  "
              f"(best={self.state.best_val_loss:.5f}, "
              f"amélioration requise > {min_delta:.5f})")

    def _log_early_stopping(self, epoch: int):
        patience = self.cfg.training.patience
        print(f"\n{'='*70}")
        print(f"  EARLY STOPPING — epoch {epoch}, patience {patience} épuisée")
        print(f"  Meilleur val_loss : {self.state.best_val_loss:.5f}")
        print(f"  Meilleur val_MAE  : {self.state.best_val_mae:.2f} dB")
        print(f"  Meilleur val_RMSE : {self.state.best_val_rmse:.2f} dB")
        print(f"  Meilleur val_R²   : {self.state.best_val_r2:.4f}")
        print(f"{'='*70}\n")

    def _log_final_summary(self, final_epoch: int, elapsed: float):
        num_epochs = self.cfg.training.num_epochs
        patience   = self.cfg.training.patience
        hist       = self.state.history

        best_i = int(np.argmin(hist['val_loss'])) if hist['val_loss'] else 0

        print(f"\n{'='*70}")
        print(f"  ENTRAÎNEMENT TERMINÉ — {self._format_time(elapsed)}")
        print(f"  Epochs : {final_epoch}/{num_epochs}  |  "
              f"Steps : {self.state.global_step}  |  "
              f"Early stopping : "
              f"{'Oui' if self.state.patience_counter >= patience else 'Non'}")
        print(f"\n  Meilleure epoch : {best_i + 1}")
        for k, unit in [('loss', ''), ('mae', ' dB'), ('rmse', ' dB'), ('r2', '')]:
            tr = hist.get(f'train_{k}', [])
            vl = hist.get(f'val_{k}',   [])
            if vl:
                tr_s = f"{tr[best_i]:.4f}{unit}" if tr else "—"
                print(f"    {k.upper():<6}  train={tr_s}  val={vl[best_i]:.4f}{unit}")
        print(f"{'='*70}\n")

    def train(self) -> Dict:
        """
        Boucle d'entraînement complète avec early stopping et checkpointing.

        Returns un dict avec :
            history, training_time, early_stopped,
            best_val_loss, best_val_mae, best_val_rmse, best_val_r2,
            best_epoch, final_epoch, dataset_info.
        """
        t0         = time.time()
        num_epochs = self.cfg.training.num_epochs
        patience   = self.cfg.training.patience
        min_delta  = self.cfg.training.min_delta
        save_every = self.cfg.training.save_every_n_epochs

        final_epoch = 0

        print(f"\n{'='*70}")
        print(f"  🚀 DÉBUT DE L'ENTRAÎNEMENT")
        print(f"{'='*70}")
        print(f"  Epochs : {num_epochs}  |  Patience : {patience}  |  Min delta : {min_delta}")
        print(f"  Critère d'arrêt : val_loss  ↓  (MSE normalisé + lissage spatial)")
        print(f"  Save every : {save_every} epochs  |  "
              f"Keep : {self.cfg.training.keep_last_n_checkpoints} checkpoints")
        print(f"{'='*70}\n")

        try:
            for epoch in range(1, num_epochs + 1):
                self.state.epoch = epoch
                epoch_start      = time.time()

                train_metrics = self.train_epoch()
                val_metrics   = self.validate()

                # Scheduler epoch-level
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['loss'])
                elif isinstance(self.scheduler, CosineAnnealingLR):
                    self.scheduler.step()
                # OneCycleLR : déjà steppé dans train_epoch()

                # Historique
                self.state.append_epoch('train', train_metrics)
                self.state.append_epoch('val',   val_metrics)
                self.state.history['lr'].append(self._get_lr())

                # Logging
                self._log_tensorboard(epoch, train_metrics, val_metrics)
                self._log_console(epoch, time.time() - epoch_start, train_metrics, val_metrics)

                # Early stopping : amélioration sur val_loss
                delta    = self.state.best_val_loss - val_metrics['loss']
                improved = delta > min_delta

                if improved:
                    self._on_improvement(epoch, val_metrics, delta)
                else:
                    self.state.patience_counter += 1
                    self._log_no_improvement(val_metrics, min_delta)

                # Checkpoint initial (epoch 1, si pas d'amélioration)
                if epoch == 1 and not improved:
                    try:
                        self._save_checkpoint(epoch, val_metrics, is_best=False)
                    except Exception as e:
                        print(f"  [WARNING] Checkpoint initial : {e}")

                # Checkpoint périodique
                if epoch % save_every == 0:
                    try:
                        self._save_checkpoint(epoch, val_metrics, is_best=False)
                        print(f"  Checkpoint périodique epoch {epoch}")
                    except Exception as e:
                        print(f"  [WARNING] Checkpoint périodique : {e}")

                final_epoch = epoch

                if self.state.patience_counter >= patience:
                    self._log_early_stopping(epoch)
                    break

        except KeyboardInterrupt:
            print(f"\n  Entraînement interrompu — epoch {final_epoch}, "
                  f"step {self.state.global_step}")

        except Exception as e:
            import traceback
            print(f"\n  ❌ ERREUR — epoch {final_epoch}")
            print(f"  {e}")
            traceback.print_exc()
            raise

        finally:
            if self.writer:
                self.writer.close()

        elapsed = time.time() - t0
        self._log_final_summary(final_epoch, elapsed)

        hist   = self.state.history
        best_i = int(np.argmin(hist['val_loss'])) if hist['val_loss'] else 0

        return {
            'history':       self.state.get_history(),
            'training_time': elapsed,
            'early_stopped': self.state.patience_counter >= patience,
            'best_val_loss': self.state.best_val_loss,
            'best_val_mae':  self.state.best_val_mae,
            'best_val_rmse': self.state.best_val_rmse,
            'best_val_r2':   self.state.best_val_r2,
            'best_epoch':    best_i + 1,
            'final_epoch':   final_epoch,
            'dataset_info':  self.dataset_info,
        }

    # =========================================================================
    # UTILITAIRES
    # =========================================================================

    def _get_lr(self) -> float:
        return self.optimizer.param_groups[0]['lr']

    @staticmethod
    def _format_time(seconds: float) -> str:
        h, rem = divmod(int(seconds), 3600)
        m, s   = divmod(rem, 60)
        if h:
            return f"{h}h {m}min {s}s"
        return f"{m}min {s}s" if m else f"{s}s"