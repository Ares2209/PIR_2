"""Analyse d'importance des features node du GATv2 par la méthode SHAP.

Objectif
--------
Classer les 23 features node (`config.FEAT_KEYS`) par leur contribution aux
décisions du modèle, sur des villes que le modèle n'a jamais vues (split
`test` par défaut), et produire :
  • un CSV `feature_importance.csv` (importance globale + par classe) ;
  • un bar-plot `01_global_bar.png` (mean |SHAP| par feature) ;
  • un beeswarm `02_beeswarm.png` (signe + dispersion des attributions) ;
  • un bar-plot par classe `03_per_class_bar.png`.

Pourquoi Expected Gradients (et pas KernelSHAP / GradientExplainer natif)
-------------------------------------------------------------------------
Le modèle est un GNN : la prédiction d'un nœud dépend, par message-passing,
des features de ses voisins. On ne peut donc PAS traiter chaque nœud comme un
échantillon indépendant (ce que suppose `shap.KernelExplainer` ou l'API
`shap.GradientExplainer`, qui batchent les lignes de la matrice de features).

On implémente donc directement l'algorithme sous-jacent de
`shap.GradientExplainer` — les *Expected Gradients* (Erion et al. 2021, la
variante « attentes de gradients » des valeurs de Shapley, cf. Lundberg &
Lee 2017) — via autograd sur le graphe COMPLET, ce qui respecte le
message-passing. On réutilise ensuite la librairie `shap` uniquement pour ses
visualisations (`shap.Explanation`, `shap.plots.bar` / `beeswarm`).

Attribution
-----------
Pour un graphe, on définit le scalaire cible
    S = Σ_i  logit[i, ŷ_i]              (ŷ_i = classe prédite du nœud i)
c.-à-d. la confiance totale du modèle sur ses propres décisions. L'attribution
Expected-Gradients de la feature f du nœud i vers S est
    φ[i,f] = E_{b~bg, α~U(0,1)} [ (x[i,f] - b[i,f]) · ∂S/∂x[i,f] |_{b+α(x-b)} ]
Grâce au message-passing, φ[i,f] capture l'effet des features du nœud i sur la
décision de TOUS les nœuds (lui + ses voisins), pas seulement la sienne.

L'importance globale d'une feature est mean_i(|φ[i,f]|) agrégée sur tous les
nœuds de tous les graphes analysés.

Usage
-----
    python models/GATv2/shap_analysis.py                      # meilleur ckpt auto
    python models/GATv2/shap_analysis.py --ckpt /chemin/gat_xxx.pt
    python models/GATv2/shap_analysis.py --split test --n-graphs 20 \
        --max-nodes 2000 --n-baselines 4 --n-alphas 32
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# repo root importable (models/GATv2/shap_analysis.py -> repo root)
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np
import torch

import models.GATv2.config as config
from models.GATv2.config import (
    ATTN_DROPOUT,
    DRONE_FEAT_DIM,
    FEAT_KEYS,
    NUM_CLASSES,
    NUM_FEATURES,
)
from models.GATv2.core import (
    GATv2 as GAT,
    find_best_checkpoint,
    get_logger,
    make_local_env,
    unwrap_module,
)
from models.GATv2.dataio import load_dataset, map_level_split

log = get_logger("gat.shap")

SCRIPT_DIR = Path(__file__).resolve().parent
CKPT_DIR   = SCRIPT_DIR / "checkpoints"
CLASS_NAMES = [
    "violet", "blue", "yellow", "orange", "red", "dark_red", "occluded",
]


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", type=str, default=None,
                   help="Checkpoint .pt (défaut: meilleur sous GATv2/checkpoints, "
                        "sinon passer un ckpt GAT compatible — l'archi est identique).")
    p.add_argument("--split", choices=["test", "val", "train"], default="test",
                   help="Split analysé (défaut: test = villes jamais vues).")
    p.add_argument("--n-graphs", type=int, default=15,
                   help="Nombre de graphes échantillonnés dans le split.")
    p.add_argument("--max-nodes", type=int, default=2000,
                   help="Plafond de nœuds attribués par graphe (échantillonnés).")
    p.add_argument("--n-baselines", type=int, default=4,
                   help="Nombre de baselines (références) par estimation EG.")
    p.add_argument("--n-alphas", type=int, default=32,
                   help="Points d'interpolation α le long du chemin baseline→x.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default=None,
                   help="Dossier de sortie (défaut: GATv2/ana_shap).")
    p.add_argument("--no-plots", action="store_true",
                   help="N'écrire que le CSV (pas de figures matplotlib/shap).")
    return p


# ─────────────────────────────────────────────────────────────────────────────
# Chargement du modèle
# ─────────────────────────────────────────────────────────────────────────────
def load_model(ckpt_path: Path, device: torch.device):
    log.info(f"Chargement du checkpoint : {ckpt_path.name}")
    ckpt = torch.load(ckpt_path, weights_only=False, map_location="cpu")

    feat_dim = int(ckpt.get("num_node_features", NUM_FEATURES))
    if feat_dim != NUM_FEATURES:
        log.error(f"Checkpoint entraîné avec {feat_dim} features node, code={NUM_FEATURES} "
                  f"(FEAT_KEYS a changé) — incompatible.")
        sys.exit(2)

    a = ckpt.get("args", {})
    model = GAT(
        num_node_features  = ckpt.get("num_node_features", NUM_FEATURES),
        num_drone_features = ckpt.get("num_drone_features", DRONE_FEAT_DIM),
        hidden_channels    = a.get("hidden_channels", config.HIDDEN_CHANNELS),
        out_channels       = ckpt.get("num_classes", NUM_CLASSES),
        dropout            = 0.0,           # analyse déterministe
        attn_dropout       = 0.0,
        num_layers         = a.get("num_layers", config.NUM_LAYERS),
        num_heads          = a.get("num_heads", config.NUM_HEADS),
        grad_checkpoint    = False,
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Expected Gradients pour un graphe
# ─────────────────────────────────────────────────────────────────────────────
def _forward_logits(model, x, edge_index, drone_feat, batch):
    return model(x, edge_index, drone_feat, batch)


def expected_gradients_graph(model, graph, device, *, n_baselines: int,
                             n_alphas: int, max_nodes: int, rng: np.random.Generator):
    """Retourne (phi, x_np, pred) pour un graphe.

    phi   : (n_sel, NUM_FEATURES) attributions Expected-Gradients (nœuds sélectionnés)
    x_np  : (n_sel, NUM_FEATURES) valeurs de features (normalisées) des mêmes nœuds
    pred  : (n_sel,) classe prédite des mêmes nœuds
    Baseline = vecteur de features moyen (≈ 0, features z-scorées) + baselines
    tirées aléatoirement parmi les nœuds réels du graphe (attente sur la
    distribution des données, comme GradientExplainer)."""
    x_full = graph.x.to(device).float()                       # (N, F)
    edge_index = graph.edge_index.to(device)
    drone_feat = graph.drone_feat.to(device).float()          # (1, DRONE_FEAT_DIM)
    N = x_full.shape[0]
    batch = torch.zeros(N, dtype=torch.long, device=device)

    # Classe prédite par nœud (cible de l'attribution).
    with torch.no_grad():
        logits = _forward_logits(model, x_full, edge_index, drone_feat, batch)
        pred = logits.argmax(dim=-1)                          # (N,)

    # Nœuds valides (label connu) — on ignore les nœuds masqués (-1) s'ils existent.
    y = getattr(graph, "y", None)
    if y is not None:
        valid_mask = (y.to(device) >= 0)
    else:
        valid_mask = torch.ones(N, dtype=torch.bool, device=device)
    valid_idx = torch.nonzero(valid_mask, as_tuple=False).squeeze(-1)
    if valid_idx.numel() == 0:
        return None

    # Sélection des nœuds à attribuer (plafonnée).
    sel = valid_idx
    if sel.numel() > max_nodes:
        pick = rng.choice(sel.cpu().numpy(), size=max_nodes, replace=False)
        sel = torch.as_tensor(np.sort(pick), device=device)
    n_sel = sel.numel()

    # Ensemble de baselines : la moyenne (zéros) + des nœuds réels tirés au hasard.
    base_vectors = [torch.zeros(NUM_FEATURES, device=device)]
    if n_baselines > 1:
        pool = rng.choice(N, size=n_baselines - 1, replace=(N < n_baselines - 1))
        for j in pool:
            base_vectors.append(x_full[int(j)].detach())

    # Accumulateur d'attributions sur tous les nœuds (on ne garde que `sel`).
    phi_acc = torch.zeros(N, NUM_FEATURES, device=device)
    n_terms = 0

    # Cible scalaire S = Σ_i logit[i, ŷ_i]. Son gradient p/r à x donne, pour la
    # ligne i, l'effet des features du nœud i sur toutes les décisions (i + voisins).
    for b_vec in base_vectors:
        baseline = b_vec.unsqueeze(0).expand(N, -1)           # (N, F)
        delta = (x_full - baseline)                           # (N, F)
        for a_i in range(n_alphas):
            # α tiré uniformément (EG stochastique) plutôt qu'une grille fixe.
            alpha = float(rng.uniform(0.0, 1.0))
            x_interp = (baseline + alpha * delta).detach().requires_grad_(True)
            out = _forward_logits(model, x_interp, edge_index, drone_feat, batch)
            S = out.gather(1, pred.unsqueeze(1)).sum()
            grad = torch.autograd.grad(S, x_interp)[0]        # (N, F)
            phi_acc += grad.detach() * delta
            n_terms += 1

    phi = (phi_acc / max(n_terms, 1))[sel]                    # (n_sel, F)
    x_np = x_full[sel].detach().cpu().numpy()
    return phi.cpu().numpy(), x_np, pred[sel].cpu().numpy()


# ─────────────────────────────────────────────────────────────────────────────
# Sorties : CSV + figures
# ─────────────────────────────────────────────────────────────────────────────
def write_csv(out_dir: Path, global_imp: np.ndarray, per_class_imp: np.ndarray):
    """global_imp: (F,), per_class_imp: (NUM_CLASSES, F). Trie par importance globale."""
    order = np.argsort(-global_imp)
    path = out_dir / "feature_importance.csv"
    header = ["rank", "feature", "mean_abs_shap"] + \
             [f"cls_{c}_{CLASS_NAMES[c]}" for c in range(NUM_CLASSES)]
    lines = [",".join(header)]
    for rank, f in enumerate(order):
        row = [str(rank + 1), FEAT_KEYS[f], f"{global_imp[f]:.6e}"] + \
              [f"{per_class_imp[c, f]:.6e}" for c in range(NUM_CLASSES)]
        lines.append(",".join(row))
    path.write_text("\n".join(lines) + "\n")
    log.success(f"CSV écrit : {path}")
    return order


def make_plots(out_dir: Path, phi_all: np.ndarray, x_all: np.ndarray,
               global_imp: np.ndarray, per_class_imp: np.ndarray, order: np.ndarray):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    feat_names = np.array(FEAT_KEYS)

    # ── 01 — bar global (mean |SHAP|) ────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 8))
    y = np.arange(len(order))[::-1]
    ax.barh(y, global_imp[order], color="#3b6ea5", edgecolor="black", linewidth=0.4)
    ax.set_yticks(y)
    ax.set_yticklabels(feat_names[order])
    ax.set_xlabel("Importance moyenne |SHAP| (Expected Gradients)")
    ax.set_title("Importance globale des features node — GATv2")
    for yi, f in zip(y, order):
        ax.text(global_imp[f], yi, f"  {global_imp[f]:.3g}", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "01_global_bar.png", dpi=140)
    plt.close(fig)

    # ── 02 — beeswarm via la lib shap ────────────────────────────────────────
    try:
        import shap
        expl = shap.Explanation(values=phi_all, data=x_all,
                                feature_names=list(FEAT_KEYS))
        shap.plots.beeswarm(expl, max_display=NUM_FEATURES, show=False)
        fig = plt.gcf()
        fig.set_size_inches(10, 9)
        fig.suptitle("SHAP beeswarm — features node (valeur z-scorée)", y=1.0)
        fig.tight_layout()
        fig.savefig(out_dir / "02_beeswarm.png", dpi=140, bbox_inches="tight")
        plt.close(fig)
    except Exception as e:
        log.warning(f"Beeswarm shap ignoré ({e}).")

    # ── 03 — heatmap importance par classe ───────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 9))
    data = per_class_imp[:, order].T                         # (F, C)
    im = ax.imshow(data, aspect="auto", cmap="viridis")
    ax.set_yticks(np.arange(len(order)))
    ax.set_yticklabels(feat_names[order])
    ax.set_xticks(np.arange(NUM_CLASSES))
    ax.set_xticklabels([f"{c}\n{CLASS_NAMES[c]}" for c in range(NUM_CLASSES)])
    ax.set_title("Importance |SHAP| par classe prédite")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="mean |SHAP|")
    fig.tight_layout()
    fig.savefig(out_dir / "03_per_class_bar.png", dpi=140)
    plt.close(fig)

    log.success(f"Figures écrites dans : {out_dir}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    args = build_parser().parse_args()
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    env = make_local_env()
    device = env.device
    log.info(f"Device : {device}")

    # ── Checkpoint ────────────────────────────────────────────────────────────
    if args.ckpt:
        ckpt_path = Path(args.ckpt)
    else:
        ckpt_path = find_best_checkpoint(CKPT_DIR)
        if ckpt_path is None:
            log.error(f"Aucun checkpoint sous {CKPT_DIR}. Entraîne GATv2 d'abord "
                      f"ou passe --ckpt (un checkpoint GAT compatible convient, "
                      f"l'architecture est identique).")
            sys.exit(1)
    if not ckpt_path.exists():
        log.error(f"Checkpoint introuvable : {ckpt_path}")
        sys.exit(1)

    model = load_model(ckpt_path, device)

    # ── Dataset + split ───────────────────────────────────────────────────────
    log.info("Chargement du dataset shardé…")
    graphs, metas = load_dataset()
    split = map_level_split(metas, config.VAL_RATIO, config.SPLIT_SEED)
    maps, idx = split[args.split]
    log.info(f"Split '{args.split}' : {len(maps)} villes, {len(idx)} graphes.")
    if not idx:
        log.error(f"Split '{args.split}' vide.")
        sys.exit(1)

    # Échantillonnage des graphes analysés.
    idx = list(idx)
    if len(idx) > args.n_graphs:
        idx = sorted(rng.choice(idx, size=args.n_graphs, replace=False).tolist())
    log.info(f"Attribution SHAP sur {len(idx)} graphes "
             f"(≤{args.max_nodes} nœuds/graphe, {args.n_baselines} baselines × "
             f"{args.n_alphas} α).")

    # ── Boucle d'attribution ──────────────────────────────────────────────────
    phi_chunks, x_chunks, pred_chunks = [], [], []
    for k, gi in enumerate(idx, 1):
        res = expected_gradients_graph(
            model, graphs[gi], device,
            n_baselines=args.n_baselines, n_alphas=args.n_alphas,
            max_nodes=args.max_nodes, rng=rng)
        if res is None:
            log.warning(f"  graphe {gi} : aucun nœud valide — ignoré.")
            continue
        phi, x_np, pred = res
        phi_chunks.append(phi)
        x_chunks.append(x_np)
        pred_chunks.append(pred)
        log.info(f"  [{k}/{len(idx)}] graphe {gi} ({metas[gi]['map']}) : "
                 f"{phi.shape[0]} nœuds attribués.")

    if not phi_chunks:
        log.error("Aucune attribution produite.")
        sys.exit(1)

    phi_all  = np.concatenate(phi_chunks, axis=0)             # (n, F)
    x_all    = np.concatenate(x_chunks, axis=0)               # (n, F)
    pred_all = np.concatenate(pred_chunks, axis=0)            # (n,)
    log.info(f"Total : {phi_all.shape[0]} nœuds attribués.")

    # ── Agrégation ────────────────────────────────────────────────────────────
    global_imp = np.abs(phi_all).mean(axis=0)                # (F,)
    per_class_imp = np.zeros((NUM_CLASSES, NUM_FEATURES))
    for c in range(NUM_CLASSES):
        m = pred_all == c
        if m.any():
            per_class_imp[c] = np.abs(phi_all[m]).mean(axis=0)

    # ── Sorties ───────────────────────────────────────────────────────────────
    out_dir = Path(args.out) if args.out else (SCRIPT_DIR / "ana_shap")
    out_dir.mkdir(parents=True, exist_ok=True)
    order = write_csv(out_dir, global_imp, per_class_imp)

    log.success("Top features (importance globale) :")
    for rank, f in enumerate(order[:10], 1):
        log.info(f"  {rank:2d}. {FEAT_KEYS[f]:<22} {global_imp[f]:.4e}")

    if not args.no_plots:
        make_plots(out_dir, phi_all, x_all, global_imp, per_class_imp, order)


if __name__ == "__main__":
    main()
