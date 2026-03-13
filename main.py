#!/usr/bin/env python3
"""Script d'entraînement principal — DroneNoiseGAT.

Toute la configuration est lue depuis config.py.

Usage :
    python main.py
    python main.py --config results/config.json
    python main.py --data-dir /chemin/vers/donnees
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.utils.data import DataLoader

# ── Source de vérité unique ───────────────────────────────────────────────────
from training.config import Config
from dataset.dataset import DroneNoiseMeshDataset, make_collate_fn
from models.GAT import DroneNoiseGAT, build_model
from training.Loss import DroneNoiseLoss


def _load_scene_arrays(data_dir: Path, subdir: str):
    """Charge tous les .npy d'un sous-dossier, triés par nom."""
    folder = data_dir / subdir
    if not folder.exists():
        raise FileNotFoundError(f"Dossier introuvable : {folder}")
    files = sorted(folder.glob('*.npy'))
    if not files:
        raise FileNotFoundError(f"Aucun fichier .npy dans : {folder}")
    return [np.load(f) for f in files]


def build_dataloaders(cfg: Config):
    """
    Charge les données et construit les DataLoaders train/val.

    Structure attendue sous cfg.data.data_dir :
        centroids/   *.npy  (N, 3)
        normals/     *.npy  (N, 3)
        areas/       *.npy  (N,)
        sources/     *.npy  (3,)
        spl/         *.npy  (N,)

    Returns:
        train_loader, val_loader, dataset_info (dict)
    """
    data_dir = Path(cfg.data.data_dir)
    print(f"  Chargement depuis : {data_dir}")

    centroids = _load_scene_arrays(data_dir, 'centroids')
    normals   = _load_scene_arrays(data_dir, 'normals')
    areas     = _load_scene_arrays(data_dir, 'areas')
    sources   = _load_scene_arrays(data_dir, 'sources')
    spl       = _load_scene_arrays(data_dir, 'spl')

    n_scenes = len(centroids)
    sizes    = [len(c) for c in centroids]
    print(f"  Scènes : {n_scenes}  |  "
          f"nœuds médian : {int(np.median(sizes))}  |  "
          f"min/max : {min(sizes)}/{max(sizes)}")

    # Split reproductible (lu depuis DataConfig)
    indices = list(range(n_scenes))
    train_idx, val_idx = train_test_split(
        indices,
        test_size    = 1 - cfg.data.train_ratio,
        random_state = cfg.data.random_seed,
    )
    print(f"  Split : {len(train_idx)} train  /  {len(val_idx)} val")

    def _sel(lst, idx): return [lst[i] for i in idx]

    # Normalisation des labels : fittée sur le train uniquement (pas de leakage)
    spl_mean, spl_std = DroneNoiseMeshDataset.compute_spl_stats(_sel(spl, train_idx))
    print(f"  SPL train — mean : {spl_mean:.1f} dB  |  std : {spl_std:.1f} dB")

    train_ds = DroneNoiseMeshDataset(
        centroids        = _sel(centroids, train_idx),
        normals          = _sel(normals,   train_idx),
        areas            = _sel(areas,     train_idx),
        source_positions = _sel(sources,   train_idx),
        spl_labels       = _sel(spl,       train_idx),
        config           = cfg.data,        # ← DataConfig
        spl_mean         = spl_mean,
        spl_std          = spl_std,
    )
    val_ds = DroneNoiseMeshDataset(
        centroids        = _sel(centroids, val_idx),
        normals          = _sel(normals,   val_idx),
        areas            = _sel(areas,     val_idx),
        source_positions = _sel(sources,   val_idx),
        spl_labels       = _sel(spl,       val_idx),
        config           = cfg.data,
        spl_mean         = spl_mean,        # mêmes stats que le train
        spl_std          = spl_std,
    )

    collate = make_collate_fn(cfg.data)     # ← paramétré depuis DataConfig
    loader_kw = dict(
        batch_size  = cfg.training.batch_size,
        num_workers = cfg.data.num_workers,
        pin_memory  = cfg.data.pin_memory,
        collate_fn  = collate,
    )
    train_loader = DataLoader(train_ds, shuffle=True,  **loader_kw)
    val_loader   = DataLoader(val_ds,   shuffle=False, **loader_kw)

    dataset_info = {
        'n_scenes_train': len(train_idx),
        'n_scenes_val':   len(val_idx),
        'spl_mean':       spl_mean,
        'spl_std':        spl_std,
        'input_dim':      DroneNoiseMeshDataset.NUM_FEATURES,
    }
    return train_loader, val_loader, dataset_info


# ─────────────────────────────────────────────────────────────────────────────
# ENTRAÎNEMENT
# ─────────────────────────────────────────────────────────────────────────────

class GATTrainer:
    """
    Boucle d'entraînement pour DroneNoiseGAT.
    Lit tous ses hyperparamètres depuis TrainingConfig et PathsConfig.
    """

    def __init__(
        self,
        model: DroneNoiseGAT,
        train_loader: DataLoader,
        val_loader: DataLoader,
        cfg: Config,
        dataset_info: dict,
    ):
        self.model        = model.to(cfg.device)
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.cfg          = cfg
        self.dataset_info = dataset_info
        self.device       = cfg.device

        self.optimizer = AdamW(
            model.parameters(),
            lr           = cfg.training.learning_rate,
            weight_decay = cfg.training.weight_decay,
        )

        if cfg.training.scheduler == 'cosine':
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max   = cfg.training.num_epochs,
                eta_min = 1e-6,
            )
        else:  # 'plateau'
            self.scheduler = ReduceLROnPlateau(
                self.optimizer, patience=10, factor=0.5, verbose=True,
            )

        self.criterion = DroneNoiseLoss(
            cfg.model,
            smooth_weight = cfg.training.smooth_weight,   # ← TrainingConfig
        )

        self.best_val_loss    = float('inf')
        self.patience_counter = 0
        self.best_ckpt        = cfg.paths.checkpoint_path()    # ← PathsConfig

        # Chargement d'un checkpoint existant (cfg.training.preload)
        if cfg.training.preload:
            self._load_checkpoint(cfg.training.preload)

    def _load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt['state_dict'])
        print(f"  Checkpoint chargé : {path}  (epoch {ckpt.get('epoch', '?')})")

    def _step(self, features, labels, mask, adj, train: bool):
        features = features.to(self.device)
        labels   = labels.to(self.device)
        mask     = mask.to(self.device)
        adj      = adj.to(self.device)

        with torch.set_grad_enabled(train):
            logits = self.model(features, adj)
            loss   = self.criterion(logits, labels, mask, adj)

        if train:
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                self.model.parameters(), self.cfg.training.gradient_clip,
            )
            self.optimizer.step()

        pred = logits.squeeze(-1).detach()
        return loss.item(), pred, labels, mask

    def _epoch(self, loader: DataLoader, train: bool) -> dict:
        self.model.train(train)
        total_loss = 0.0
        all_pred, all_true = [], []
        std  = self.dataset_info['spl_std']
        mean = self.dataset_info['spl_mean']

        for features, labels, mask, adj in loader:
            loss, pred, true, m = self._step(features, labels, mask, adj, train)
            total_loss += loss
            m_dev = m.to(pred.device)
            # Dénormalisation en dB pour les métriques
            all_pred.append((pred[m_dev] * std + mean).cpu().numpy())
            all_true.append((true.to(pred.device)[m_dev] * std + mean).cpu().numpy())

        pred_np = np.concatenate(all_pred)
        true_np = np.concatenate(all_true)
        metrics = _regression_metrics(pred_np, true_np)
        metrics['loss'] = total_loss / len(loader)
        return metrics

    def train(self) -> dict:
        history = {k: [] for k in [
            'train_loss', 'val_loss',
            'train_mae',  'val_mae',
            'train_rmse', 'val_rmse',
            'train_r2',   'val_r2',
            'lr',
        ]}
        t0 = time.time()

        for epoch in range(1, self.cfg.training.num_epochs + 1):
            tr = self._epoch(self.train_loader, train=True)
            vl = self._epoch(self.val_loader,   train=False)

            if isinstance(self.scheduler, ReduceLROnPlateau):
                self.scheduler.step(vl['loss'])
            else:
                self.scheduler.step()

            lr = self.optimizer.param_groups[0]['lr']

            for split, m in [('train', tr), ('val', vl)]:
                for k in ['loss', 'mae', 'rmse', 'r2']:
                    history[f'{split}_{k}'].append(m[k])
            history['lr'].append(lr)

            print(
                f"Epoch {epoch:3d}/{self.cfg.training.num_epochs}  "
                f"loss {tr['loss']:.4f}/{vl['loss']:.4f}  "
                f"MAE {tr['mae']:.2f}/{vl['mae']:.2f} dB  "
                f"RMSE {tr['rmse']:.2f}/{vl['rmse']:.2f} dB  "
                f"R² {tr['r2']:.3f}/{vl['r2']:.3f}  "
                f"lr {lr:.2e}"
            )

            # Checkpoint périodique
            if epoch % self.cfg.training.save_every_n_epochs == 0:
                periodic_path = self.cfg.paths.checkpoint_path(epoch)
                torch.save({
                    'epoch':      epoch,
                    'state_dict': self.model.state_dict(),
                    'val_loss':   vl['loss'],
                    'val_mae':    vl['mae'],
                    'config':     self.cfg.to_dict(),
                }, periodic_path)
                self._prune_old_checkpoints()

            # Best model
            improved = vl['loss'] < self.best_val_loss - self.cfg.training.min_delta
            if improved:
                self.best_val_loss = vl['loss']
                self.patience_counter = 0
                torch.save({
                    'epoch':      epoch,
                    'state_dict': self.model.state_dict(),
                    'val_loss':   vl['loss'],
                    'val_mae':    vl['mae'],
                    'val_rmse':   vl['rmse'],
                    'val_r2':     vl['r2'],
                    'config':     self.cfg.to_dict(),
                }, self.best_ckpt)
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.cfg.training.patience:
                    print(f"\n  Early stopping — epoch {epoch}  "
                          f"(patience={self.cfg.training.patience})")
                    break

        return {
            'history':       history,
            'training_time': time.time() - t0,
            'early_stopped': self.patience_counter >= self.cfg.training.patience,
            'best_val_loss': self.best_val_loss,
            'dataset_info':  self.dataset_info,
        }

    def _prune_old_checkpoints(self):
        """Supprime les anciens checkpoints périodiques (garde les N derniers)."""
        import glob, os
        pattern = str(self.cfg.paths.checkpoint_path(0)).replace('epoch0000', 'epoch*')
        files = sorted(glob.glob(pattern), key=os.path.getmtime)
        for old in files[:-self.cfg.training.keep_last_n_checkpoints]:
            Path(old).unlink(missing_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# MÉTRIQUES
# ─────────────────────────────────────────────────────────────────────────────

def _regression_metrics(pred: np.ndarray, true: np.ndarray) -> dict:
    """MAE, RMSE, R² en dB sur les nœuds réels dénormalisés."""
    mae  = float(np.mean(np.abs(pred - true)))
    rmse = float(np.sqrt(np.mean((pred - true) ** 2)))
    ss_res = np.sum((true - pred) ** 2)
    ss_tot = np.sum((true - true.mean()) ** 2)
    r2 = float(1 - ss_res / (ss_tot + 1e-10))
    return {'mae': mae, 'rmse': rmse, 'r2': r2}


# ─────────────────────────────────────────────────────────────────────────────
# GRAPHIQUES ET SAUVEGARDE
# ─────────────────────────────────────────────────────────────────────────────

def _get(h: dict, key: str) -> list:
    return h.get(key, [])


def plot_training_results(history: dict, save_dir: str):
    """Produit les figures de suivi (régression SPL)."""
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    n = len(_get(history, 'train_loss'))
    if not n:
        return
    epochs = range(1, n + 1)

    fig, axes = plt.subplots(2, 3, figsize=(20, 10))
    axs = axes.flatten()

    def _pair(ax, tk, vk, title, ylabel, ylim=None):
        tr = _get(history, tk); vl = _get(history, vk)
        if tr: ax.plot(epochs, tr, label='Train', lw=2, marker='o', ms=3, color='#3498db')
        if vl: ax.plot(epochs, vl, label='Val',   lw=2, marker='s', ms=3, color='#e74c3c')
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.set_xlabel('Epoch'); ax.set_ylabel(ylabel)
        ax.legend(); ax.grid(True, alpha=0.3)
        if ylim: ax.set_ylim(ylim)

    _pair(axs[0], 'train_loss', 'val_loss',   'Loss (MSE normalisé)',  'Loss')
    _pair(axs[1], 'train_mae',  'val_mae',    'MAE (dB)',              'MAE (dB)')
    _pair(axs[2], 'train_rmse', 'val_rmse',   'RMSE (dB)',             'RMSE (dB)')
    _pair(axs[3], 'train_r2',   'val_r2',     'R²',                    'R²', ylim=[-0.1, 1.05])

    lr = _get(history, 'lr')
    if lr:
        axs[4].semilogy(epochs, lr, color='#2ecc71', lw=2)
        axs[4].set_title('Learning Rate', fontsize=13, fontweight='bold')
        axs[4].set_xlabel('Epoch'); axs[4].set_ylabel('LR')
        axs[4].grid(True, alpha=0.3)

    # Barres finales
    labels = ['MAE (dB)', 'RMSE (dB)', 'R²']
    keys   = [('train_mae', 'val_mae'), ('train_rmse', 'val_rmse'), ('train_r2', 'val_r2')]
    tr_v   = [(_get(history, k[0]) or [0])[-1] for k in keys]
    vl_v   = [(_get(history, k[1]) or [0])[-1] for k in keys]
    x, w   = np.arange(3), 0.35
    ax     = axs[5]
    b1 = ax.bar(x - w/2, tr_v, w, label='Train', color='#3498db', edgecolor='black')
    b2 = ax.bar(x + w/2, vl_v, w, label='Val',   color='#e74c3c', edgecolor='black')
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_title('Métriques finales', fontsize=13, fontweight='bold')
    ax.legend(); ax.grid(True, alpha=0.3, axis='y')
    for bar in [*b1, *b2]:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h, f'{h:.2f}',
                ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(f'{save_dir}/training_history.png', dpi=100, bbox_inches='tight')
    plt.close()

    # MAE détaillée
    fig, ax = plt.subplots(figsize=(10, 5))
    tr_mae = _get(history, 'train_mae'); vl_mae = _get(history, 'val_mae')
    if tr_mae:
        ax.plot(epochs, tr_mae, label='Train MAE', lw=2, color='#3498db', marker='o', ms=3)
        ax.axhline(min(tr_mae), color='#3498db', ls=':', alpha=0.6,
                   label=f'Min train : {min(tr_mae):.2f} dB')
    if vl_mae:
        ax.plot(epochs, vl_mae, label='Val MAE',   lw=2, color='#e74c3c', marker='s', ms=3)
        ax.axhline(min(vl_mae), color='#e74c3c', ls=':', alpha=0.6,
                   label=f'Min val : {min(vl_mae):.2f} dB')
    ax.set_title('Évolution MAE (dB)', fontsize=13, fontweight='bold')
    ax.set_xlabel('Epoch'); ax.set_ylabel('MAE (dB)')
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/mae_detailed.png', dpi=100, bbox_inches='tight')
    plt.close()
    print(f"  Graphiques : {save_dir}/")


def save_training_results(result: dict, cfg: Config):
    """Sauvegarde CSV + rapport texte dans cfg.paths.results_folder."""
    save_dir = cfg.paths.results_folder
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    history  = result.get('history', {})

    n = len(_get(history, 'train_loss'))
    if not n:
        return

    df = pd.DataFrame({'epoch': range(1, n + 1)})
    for key, vals in history.items():
        if isinstance(vals, list) and len(vals) == n:
            df[key] = vals
    df.to_csv(f'{save_dir}/training_history.csv', index=False)
    print(f"  CSV : {save_dir}/training_history.csv")

    info    = result.get('dataset_info', {})
    report  = Path(save_dir) / 'training_report.txt'
    best_i  = int(np.argmin(_get(history, 'val_loss'))) if _get(history, 'val_loss') else 0
    t       = result.get('training_time', 0)

    with open(report, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\nRAPPORT — DroneNoiseGAT\n" + "=" * 70 + "\n\n")
        f.write(f"Epochs           : {n}\n")
        f.write(f"Durée            : {t/3600:.2f}h ({t/60:.1f} min)\n")
        f.write(f"Early stopping   : {'Oui' if result.get('early_stopped') else 'Non'}\n\n")
        f.write(f"Scènes train     : {info.get('n_scenes_train', '?')}\n")
        f.write(f"Scènes val       : {info.get('n_scenes_val',   '?')}\n")
        f.write(f"SPL mean / std   : {info.get('spl_mean', 0):.1f} / "
                f"{info.get('spl_std', 0):.1f} dB\n\n")
        # Paramètres drone depuis DroneConfig
        d = cfg.drone
        f.write(f"Drone            : {d.sound_power_db} dBW  |  "
                f"alt={d.flight_altitude_m} m  |  "
                f"v_max={d.max_speed_ms} m/s\n\n")

        f.write(f"{'─'*70}\nMEILLEURE EPOCH (val_loss) : {best_i + 1}\n{'─'*70}\n")
        for k, unit in [('loss',''), ('mae',' dB'), ('rmse',' dB'), ('r2','')]:
            tr = _get(history, f'train_{k}'); vl = _get(history, f'val_{k}')
            if tr: f.write(f"  Train {k:5s} : {tr[best_i]:.4f}{unit}\n")
            if vl: f.write(f"  Val   {k:5s} : {vl[best_i]:.4f}{unit}\n")

        f.write(f"\n{'─'*70}\nDERNIÈRE EPOCH : {n}\n{'─'*70}\n")
        for k, unit in [('loss',''), ('mae',' dB'), ('rmse',' dB'), ('r2','')]:
            tr = _get(history, f'train_{k}'); vl = _get(history, f'val_{k}')
            if tr: f.write(f"  Train {k:5s} : {tr[-1]:.4f}{unit}\n")
            if vl: f.write(f"  Val   {k:5s} : {vl[-1]:.4f}{unit}\n")

        f.write("\n" + "=" * 70 + "\n")

    print(f"  Rapport : {report}")


# ─────────────────────────────────────────────────────────────────────────────
# RÉSUMÉ CONSOLE
# ─────────────────────────────────────────────────────────────────────────────

def print_training_summary(result: dict):
    history = result.get('history', {})
    t       = result.get('training_time', 0)
    n       = len(_get(history, 'train_loss'))
    best_i  = int(np.argmin(_get(history, 'val_loss'))) if _get(history, 'val_loss') else 0

    print(f"\n{'='*70}")
    print(f"  RÉSUMÉ — DroneNoiseGAT")
    print(f"{'='*70}")
    print(f"  Epochs : {n}  |  Durée : {t/3600:.2f}h ({t/60:.1f} min)  |  "
          f"Early stopping : {'Oui' if result.get('early_stopped') else 'Non'}")
    print(f"  Meilleure epoch (val_loss) : {best_i + 1}")
    for k, unit in [('mae', ' dB'), ('rmse', ' dB'), ('r2', '')]:
        tr = _get(history, f'train_{k}'); vl = _get(history, f'val_{k}')
        if vl:
            tr_str = f"{tr[best_i]:.4f}{unit}" if tr else "—"
            print(f"    {k.upper():5s}  train={tr_str}  val={vl[best_i]:.4f}{unit}")
    print(f"{'='*70}\n")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Entraînement DroneNoiseGAT")
    parser.add_argument('--config',   type=str, default=None,
                        help="Chemin vers un fichier config JSON sauvegardé")
    parser.add_argument('--data-dir', type=str, default=None,
                        help="Dossier racine des données (override cfg.data.data_dir)")
    args = parser.parse_args()

    # ── Config ───────────────────────────────────────────────────────────────
    cfg = Config.load(args.config) if args.config else Config()
    if args.data_dir:
        cfg.data.data_dir = args.data_dir

    cfg.print_summary()
    cfg.save(Path(cfg.paths.results_folder) / 'config.json')

    # ── Données ──────────────────────────────────────────────────────────────
    print("\n── Chargement des données ──")
    train_loader, val_loader, dataset_info = build_dataloaders(cfg)

    # Vérification cohérence input_dim
    if dataset_info['input_dim'] != cfg.model.input_dim:
        raise ValueError(
            f"input_dim config ({cfg.model.input_dim}) ≠ dataset "
            f"({dataset_info['input_dim']}). "
            f"Mettez input_dim={dataset_info['input_dim']} dans ModelConfig."
        )

    # ── Modèle ───────────────────────────────────────────────────────────────
    print("\n── Construction du modèle ──")
    model = build_model(cfg.model)    # ← ModelConfig depuis config.py

    # ── Entraînement ─────────────────────────────────────────────────────────
    print("\n── Entraînement ──")
    trainer = GATTrainer(model, train_loader, val_loader, cfg, dataset_info)
    result  = trainer.train()

    # ── Résultats ─────────────────────────────────────────────────────────────
    print_training_summary(result)

    print("\n── Sauvegarde ──")
    out_dir = cfg.paths.results_folder
    with open(f'{out_dir}/training_result.json', 'w') as f:
        json.dump(
            {k: v for k, v in result.items() if k != 'history'},
            f, indent=2, default=float,
        )

    save_training_results(result, cfg)              # ← passe cfg entier (accès drone)
    plot_training_results(result['history'], out_dir)

    print(f"\n  Best checkpoint : {trainer.best_ckpt}")
    print(f"  Résultats       : {out_dir}/\n")


if __name__ == '__main__':
    main()