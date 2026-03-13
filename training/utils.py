"""Utilitaires divers pour l'entraînement."""

import torch
import numpy as np
import random
from typing import Dict, Any


def set_seed(seed: int = 42):
    """Fixe toutes les graines aléatoires pour reproductibilité."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_device_info() -> Dict[str, Any]:
    """Retourne les informations sur le device disponible."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    info = {'device': device}
    
    if device == 'cuda':
        info['gpu_name'] = torch.cuda.get_device_name(0)
        info['cuda_version'] = torch.version.cuda
        info['total_memory_gb'] = torch.cuda.get_device_properties(0).total_memory / 1024**3
        info['num_gpus'] = torch.cuda.device_count()
    
    return info


def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    """Compte les paramètres du modèle."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total': total,
        'trainable': trainable,
        'frozen': total - trainable
    }


def format_time(seconds: float) -> str:
    """Formate une durée en secondes en format lisible."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}h {minutes}min {secs}s"
    elif minutes > 0:
        return f"{minutes}min {secs}s"
    else:
        return f"{secs}s"


def get_lr(optimizer: torch.optim.Optimizer) -> float:
    """Récupère le learning rate actuel."""
    for param_group in optimizer.param_groups:
        return param_group['lr']
    return 0.0