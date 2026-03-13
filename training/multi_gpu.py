
"""
Utilitaires pour l'entraînement multi-GPU.
"""

import torch
import torch.nn as nn
from typing import Optional, List

class MultiGPUWrapper:
    """Wrapper pour gérer facilement le multi-GPU."""
    
    def __init__(self, model: nn.Module, gpu_ids: Optional[List[int]] = None, 
                 backend: str = 'dp'):
        """
        Args:
            model: Le modèle PyTorch
            gpu_ids: Liste des GPU à utiliser (None = tous)
            backend: 'dp' pour DataParallel, 'ddp' pour DistributedDataParallel
        """
        self.model = model
        self.backend = backend
        self.is_multi_gpu = False
        
        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            
            if num_gpus > 1:
                if gpu_ids is None:
                    gpu_ids = list(range(num_gpus))
                
                if len(gpu_ids) > 1:
                    self.is_multi_gpu = True
                    self.gpu_ids = gpu_ids
                    
                    print(f"\n🎮 Configuration Multi-GPU ({backend.upper()})")
                    print(f"   Nombre de GPUs: {len(gpu_ids)}")
                    for gpu_id in gpu_ids:
                        props = torch.cuda.get_device_properties(gpu_id)
                        print(f"   - GPU {gpu_id}: {props.name}")
                        print(f"     Memory: {props.total_memory / 1e9:.2f} GB")
                    
                    # Choisir le backend
                    if backend == 'dp':
                        self.wrapped_model = nn.DataParallel(
                            model, 
                            device_ids=gpu_ids
                        )
                    elif backend == 'ddp':
                        # DistributedDataParallel nécessite une configuration plus complexe
                        raise NotImplementedError(
                            "DDP backend nécessite une initialisation via torch.distributed"
                        )
                    else:
                        raise ValueError(f"Backend inconnu: {backend}")
                    
                    # Déplacer sur le GPU principal
                    self.wrapped_model = self.wrapped_model.cuda(gpu_ids[0])
                    print(f"   ✅ Modèle distribué sur {len(gpu_ids)} GPUs\n")
                else:
                    # Un seul GPU
                    self.wrapped_model = model.cuda(gpu_ids[0])
                    print(f"\n🎮 Single GPU: {torch.cuda.get_device_name(gpu_ids[0])}\n")
            else:
                # Un seul GPU disponible
                self.wrapped_model = model.cuda(0)
                print(f"\n🎮 Single GPU: {torch.cuda.get_device_name(0)}\n")
        else:
            # CPU
            self.wrapped_model = model
            print("\n💻 CPU Mode\n")
    
    def get_model(self):
        """Retourne le modèle wrappé."""
        return self.wrapped_model
    
    def get_base_model(self):
        """Retourne le modèle de base (sans wrapper)."""
        if self.is_multi_gpu:
            return self.wrapped_model.module
        return self.wrapped_model
    
    def adjust_batch_size(self, base_batch_size: int) -> int:
        """Ajuste le batch size pour le multi-GPU."""
        if self.is_multi_gpu:
            adjusted = base_batch_size * len(self.gpu_ids)
            print(f"📊 Batch size ajusté pour multi-GPU:")
            print(f"   Base: {base_batch_size} → Effectif: {adjusted}")
            print(f"   ({base_batch_size} par GPU × {len(self.gpu_ids)} GPUs)")
            return adjusted
        return base_batch_size

def setup_multi_gpu(model: nn.Module, config) -> tuple:
    """
    Configure le modèle pour le multi-GPU et ajuste la config.
    
    Returns:
        (wrapped_model, adjusted_batch_size, device)
    """
    wrapper = MultiGPUWrapper(
        model=model,
        gpu_ids=config.training.gpu_ids,
        backend='dp',   # DataParallel — TrainingConfig ne supporte pas DDP
    )
    
    # Ajuster le batch size
    adjusted_batch_size = wrapper.adjust_batch_size(config.training.batch_size)
    
    # Device principal
    if wrapper.is_multi_gpu:
        device = torch.device(f'cuda:{wrapper.gpu_ids[0]}')
    elif torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    
    return wrapper.get_model(), adjusted_batch_size, device, wrapper

def calculate_effective_batch_size(base_batch_size: int, num_gpus: int, 
                                   gradient_accumulation: int = 1) -> dict:
    """
    Calcule les différentes tailles de batch.
    
    Returns:
        dict avec 'per_gpu', 'total_parallel', 'effective'
    """
    return {
        'per_gpu': base_batch_size,
        'total_parallel': base_batch_size * num_gpus,
        'effective': base_batch_size * num_gpus * gradient_accumulation
    }
