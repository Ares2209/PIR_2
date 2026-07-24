"""Explication structurelle du GATv2 par GNNExplainer (PyTorch Geometric).

Objectif — complément de `shap_analysis.py`
--------------------------------------------
L'analyse SHAP (Expected Gradients, cf. `shap_analysis.py`) répond à « quelles
features de nœud comptent ». Elle ne dit RIEN sur la structure : quels voisins /
quelles arêtes du graphe portent la décision. C'est précisément ce que
GNNExplainer (Ying et al. 2019) apprend — un masque sur les arêtes ET sur les
features de nœud maximisant l'information mutuelle avec la prédiction du modèle.

Ce script produit donc :
  • `subgraphs/node_XXXX.png` — pour quelques nœuds exemples (un par classe si
    possible), le sous-graphe important autour du nœud (arêtes pondérées par
    leur masque) : la vue STRUCTURELLE, absente de SHAP ;
  • `feature_importance_gnnexplainer.csv` + `01_global_bar.png` — le classement
    des 23 features node (masque de features `common_attributes`, moyenné sur les
    nœuds expliqués). À lire comme un RECOUPEMENT secondaire du CSV SHAP : le
    masque de features de GNNExplainer est notoirement moins fiable/discriminant
    que l'attribution par gradients (SHAP) — un accord sur le top des features est
    un bon signal de robustesse, un désaccord ne disqualifie pas SHAP ;
  • `02_per_class_bar.png` — importance de features par classe prédite.

Coût / champ réceptif
---------------------
GNNExplainer optimise un masque par nœud expliqué. Le champ réceptif du modèle
est `num_layers` sauts (ici 6). On extrait donc le sous-graphe k-hop autour de
chaque nœud (`--num-hops`, défaut = num_layers) et on l'explique isolément, ce
qui garde le calcul traitable. Un plafond `--max-subgraph-nodes` saute les nœuds
dont le sous-graphe explose. Réduire `--num-hops` accélère (au prix de la
fidélité : la prédiction sur le sous-graphe tronqué peut différer du graphe
complet).

Usage
-----
    python models/GATv2/gnn_explainer.py                          # meilleur ckpt auto
    python models/GATv2/gnn_explainer.py --n-graphs 4 --nodes-per-graph 8 \
        --epochs 100 --num-hops 3
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Réduit la fragmentation de l'allocateur CUDA (cartes 11 GiB) — AVANT import torch.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np
import torch
from torch_geometric.explain import Explainer, GNNExplainer
from torch_geometric.utils import k_hop_subgraph

import models.GATv2.config as config
from models.GATv2.config import FEAT_KEYS, NUM_CLASSES, NUM_FEATURES
from models.GATv2.core import find_best_checkpoint, get_logger, make_local_env
from models.GATv2.dataio import load_dataset, map_level_split
# DRY : on réutilise le chargeur de checkpoint de l'analyse SHAP (archi identique).
from models.GATv2.shap_analysis import CLASS_NAMES, load_model

log = get_logger("gat.gnnexplainer")

SCRIPT_DIR = Path(__file__).resolve().parent
CKPT_DIR   = SCRIPT_DIR / "checkpoints"


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", type=str, default=None,
                   help="Checkpoint .pt (défaut: meilleur sous GATv2/checkpoints).")
    p.add_argument("--split", choices=["test", "val", "train"], default="test",
                   help="Split analysé (défaut: test = villes jamais vues).")
    p.add_argument("--n-graphs", type=int, default=4,
                   help="Nombre de graphes échantillonnés dans le split.")
    p.add_argument("--nodes-per-graph", type=int, default=8,
                   help="Nœuds expliqués par graphe (un GNNExplainer chacun — coûteux).")
    p.add_argument("--epochs", type=int, default=100,
                   help="Époques d'optimisation du masque GNNExplainer par nœud.")
    p.add_argument("--num-hops", type=int, default=None,
                   help="Sauts du sous-graphe expliqué (défaut: num_layers du modèle).")
    p.add_argument("--max-subgraph-nodes", type=int, default=5000,
                   help="Saute un nœud si son sous-graphe k-hop dépasse ce plafond.")
    p.add_argument("--topk-edges", type=int, default=20,
                   help="Arêtes les plus importantes affichées dans les sous-graphes.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default=None,
                   help="Dossier de sortie (défaut: GATv2/ana_gnnexplainer).")
    p.add_argument("--no-plots", action="store_true",
                   help="N'écrire que le CSV (pas de figures).")
    return p


# ─────────────────────────────────────────────────────────────────────────────
# Explainer
# ─────────────────────────────────────────────────────────────────────────────
def build_explainer(model, epochs: int) -> Explainer:
    """Explainer PyG configuré pour la classification de nœuds (logits bruts).

    node_mask_type='common_attributes' → UN masque de features [1, F] partagé par
        tous les nœuds. On évite volontairement 'attributes' ([N, F], un masque
        par nœud) : agréger [N, F] sur les dizaines de nœuds du sous-graphe (champ
        réceptif 6 sauts) noie les différences entre features sous la taille du
        sous-graphe (toutes les features finissent ≈ n_sub × 0,5). 'common_attributes'
        donne directement un vecteur d'importance de features discriminant.
    edge_mask_type='object'     → masque [E] (importance par arête, inchangé).
    explanation_type='model'    → on explique la prédiction du MODÈLE (pas la
                                   vérité terrain), cohérent avec SHAP.
    """
    return Explainer(
        model=model,
        algorithm=GNNExplainer(epochs=epochs),
        explanation_type="model",
        node_mask_type="common_attributes",
        edge_mask_type="object",
        model_config=dict(
            mode="multiclass_classification",
            task_level="node",
            return_type="raw",        # le modèle sort des logits bruts
        ),
    )


def explain_node(explainer, x_sub, edge_index_sub, drone_feat, seed_local: int):
    """Explication d'un nœud (index local `seed_local`) sur son sous-graphe.

    drone_feat et batch sont passés en kwargs FIXES : GNNExplainer ne perturbe
    que x et edge_index, la signature-drone reste celle du graphe d'origine.
    Retourne l'objet Explanation.
    """
    n = x_sub.shape[0]
    batch = torch.zeros(n, dtype=torch.long, device=x_sub.device)
    return explainer(
        x_sub, edge_index_sub,
        index=seed_local,
        drone_feat=drone_feat,
        batch=batch,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Sorties : CSV + figures
# ─────────────────────────────────────────────────────────────────────────────
def write_csv(out_dir: Path, global_imp: np.ndarray, per_class_imp: np.ndarray):
    order = np.argsort(-global_imp)
    path = out_dir / "feature_importance_gnnexplainer.csv"
    header = ["rank", "feature", "mean_node_mask"] + \
             [f"cls_{c}_{CLASS_NAMES[c]}" for c in range(NUM_CLASSES)]
    lines = [",".join(header)]
    for rank, f in enumerate(order):
        row = [str(rank + 1), FEAT_KEYS[f], f"{global_imp[f]:.6e}"] + \
              [f"{per_class_imp[c, f]:.6e}" for c in range(NUM_CLASSES)]
        lines.append(",".join(row))
    path.write_text("\n".join(lines) + "\n")
    log.success(f"CSV écrit : {path}")
    return order


def plot_feature_importance(out_dir: Path, global_imp, per_class_imp, order):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    feat_names = np.array(FEAT_KEYS)

    # ── bar global (même style que shap_analysis 01) ─────────────────────────
    fig, ax = plt.subplots(figsize=(9, 8))
    y = np.arange(len(order))[::-1]
    ax.barh(y, global_imp[order], color="#3b6ea5", edgecolor="black", linewidth=0.4)
    ax.set_yticks(y)
    ax.set_yticklabels(feat_names[order])
    ax.set_xlabel("Importance moyenne du masque de features (GNNExplainer)")
    ax.set_title("Importance globale des features node — GATv2 (GNNExplainer)")
    for yi, f in zip(y, order):
        ax.text(global_imp[f], yi, f"  {global_imp[f]:.3g}", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "01_global_bar.png", dpi=140)
    plt.close(fig)

    # ── heatmap par classe (même style que shap_analysis 03) ─────────────────
    fig, ax = plt.subplots(figsize=(10, 9))
    data = per_class_imp[:, order].T
    im = ax.imshow(data, aspect="auto", cmap="viridis")
    ax.set_yticks(np.arange(len(order)))
    ax.set_yticklabels(feat_names[order])
    ax.set_xticks(np.arange(NUM_CLASSES))
    ax.set_xticklabels([f"{c}\n{CLASS_NAMES[c]}" for c in range(NUM_CLASSES)])
    ax.set_title("Importance du masque de features par classe prédite")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="mean node mask")
    fig.tight_layout()
    fig.savefig(out_dir / "02_per_class_bar.png", dpi=140)
    plt.close(fig)
    log.success(f"Figures features écrites dans : {out_dir}")


# Palette des 7 classes, alignée sur CLASS_NAMES (violet→dark_red + occluded).
_CLASS_COLORS = ["#7b3fbf", "#2c6fbf", "#e5c100", "#e8850c",
                 "#d62728", "#7f1414", "#888888"]


def plot_subgraph(out_dir: Path, expl, subset_np, seed_local: int, seed_global: int,
                  pred_sub_np: np.ndarray, topk_edges: int):
    """Dessine le sous-graphe expliqué : arêtes top-k pondérées par le masque,
    nœuds colorés par classe prédite, nœud-cible surligné."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import networkx as nx

    edge_index = expl.edge_index.cpu().numpy()               # (2, E) indices locaux
    edge_mask  = expl.edge_mask.detach().cpu().numpy()       # (E,)
    if edge_mask.size == 0:
        return

    # On ne garde que les top-k arêtes pour la lisibilité.
    k = min(topk_edges, edge_mask.size)
    keep = np.argsort(-edge_mask)[:k]
    max_w = float(edge_mask[keep].max()) or 1.0

    G = nx.DiGraph()
    n_sub = subset_np.shape[0]
    G.add_nodes_from(range(n_sub))
    for e in keep:
        u, v = int(edge_index[0, e]), int(edge_index[1, e])
        G.add_edge(u, v, w=float(edge_mask[e]))

    # Sous-ensemble de nœuds effectivement reliés (+ la cible) pour un layout net.
    nodes_shown = set().union(*[set(e) for e in G.edges]) if G.number_of_edges() else set()
    nodes_shown.add(seed_local)
    H = G.subgraph(nodes_shown)

    pos = nx.spring_layout(H, seed=0, k=0.8)
    fig, ax = plt.subplots(figsize=(8, 7))

    node_colors = [_CLASS_COLORS[int(pred_sub_np[nd])] for nd in H.nodes]
    sizes = [420 if nd == seed_local else 160 for nd in H.nodes]
    edgecols = ["black" if nd == seed_local else "none" for nd in H.nodes]
    nx.draw_networkx_nodes(H, pos, node_color=node_colors, node_size=sizes,
                           edgecolors=edgecols, linewidths=1.6, ax=ax)
    widths = [3.0 * H[u][v]["w"] / max_w for u, v in H.edges]
    nx.draw_networkx_edges(H, pos, width=widths, edge_color="#555555",
                           arrows=True, arrowsize=8, alpha=0.7, ax=ax)

    cls = int(pred_sub_np[seed_local])
    ax.set_title(f"GNNExplainer — nœud {seed_global} "
                 f"(classe prédite {cls}: {CLASS_NAMES[cls]})\n"
                 f"{k} arêtes les plus importantes du sous-graphe")
    ax.axis("off")
    fig.tight_layout()
    sub_dir = out_dir / "subgraphs"
    sub_dir.mkdir(exist_ok=True)
    fig.savefig(sub_dir / f"node_{seed_global:05d}.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


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
                      f"ou passe --ckpt.")
            sys.exit(1)
    if not ckpt_path.exists():
        log.error(f"Checkpoint introuvable : {ckpt_path}")
        sys.exit(1)

    model = load_model(ckpt_path, device)
    num_hops = args.num_hops if args.num_hops is not None else len(model.convs)
    log.info(f"Champ réceptif expliqué : {num_hops} sauts.")
    explainer = build_explainer(model, args.epochs)

    # ── Dataset + split ───────────────────────────────────────────────────────
    log.info("Chargement du dataset shardé…")
    graphs, metas = load_dataset()
    split = map_level_split(metas, config.VAL_RATIO, config.SPLIT_SEED)
    maps, idx = split[args.split]
    log.info(f"Split '{args.split}' : {len(maps)} villes, {len(idx)} graphes.")
    if not idx:
        log.error(f"Split '{args.split}' vide.")
        sys.exit(1)

    idx = list(idx)
    if len(idx) > args.n_graphs:
        idx = sorted(rng.choice(idx, size=args.n_graphs, replace=False).tolist())
    log.info(f"GNNExplainer sur {len(idx)} graphes × ≤{args.nodes_per_graph} nœuds "
             f"({args.epochs} époques/nœud).")

    out_dir = Path(args.out) if args.out else (SCRIPT_DIR / "ana_gnnexplainer")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Accumulateurs d'importance de features (par nœud expliqué).
    fmask_sum   = np.zeros(NUM_FEATURES)
    fmask_n     = 0
    per_class_sum = np.zeros((NUM_CLASSES, NUM_FEATURES))
    per_class_n   = np.zeros(NUM_CLASSES, dtype=int)
    # Nœuds déjà illustrés en sous-graphe (un exemple par classe prédite).
    plotted_classes: set[int] = set()

    for gk, gi in enumerate(idx, 1):
        graph = graphs[gi]
        x_full = graph.x.to(device).float()
        edge_index = graph.edge_index.to(device)
        drone_feat = graph.drone_feat.to(device).float()
        N = x_full.shape[0]
        batch_full = torch.zeros(N, dtype=torch.long, device=device)

        # Prédictions du modèle sur le graphe complet (choix des nœuds + labels viz).
        with torch.no_grad():
            pred_full = model(x_full, edge_index, drone_feat, batch_full).argmax(-1)

        # Nœuds valides (label ≥ 0), échantillonnés.
        y = getattr(graph, "y", None)
        valid = (y.to(device) >= 0) if y is not None else torch.ones(N, dtype=torch.bool, device=device)
        valid_idx = torch.nonzero(valid, as_tuple=False).squeeze(-1)
        if valid_idx.numel() == 0:
            log.warning(f"  graphe {gi} : aucun nœud valide — ignoré.")
            continue
        sel = valid_idx.cpu().numpy()
        if sel.size > args.nodes_per_graph:
            sel = np.sort(rng.choice(sel, size=args.nodes_per_graph, replace=False))

        n_done = 0
        for node in sel:
            node = int(node)
            # Sous-graphe k-hop autour du nœud (indices relabellisés en local).
            subset, ei_sub, mapping, _ = k_hop_subgraph(
                node, num_hops, edge_index, relabel_nodes=True,
                num_nodes=N, flow="source_to_target")
            if subset.numel() > args.max_subgraph_nodes:
                continue  # sous-graphe trop gros → on saute (cf. --max-subgraph-nodes)

            x_sub = x_full[subset]
            seed_local = int(mapping.item())

            expl = explain_node(explainer, x_sub, ei_sub, drone_feat, seed_local)

            # node_mask : (1, F) avec 'common_attributes' → vecteur d'importance
            # par feature. .mean(dim=0) reste correct si on repasse à 'attributes'
            # (moyenne, pas somme : la somme réintroduirait la taille du sous-graphe).
            fmask = expl.node_mask.detach().abs().mean(dim=0).cpu().numpy()  # (F,)
            cls = int(pred_full[node].item())
            fmask_sum += fmask
            fmask_n   += 1
            per_class_sum[cls] += fmask
            per_class_n[cls]   += 1
            n_done += 1

            # Un sous-graphe illustratif par classe prédite (les plus utiles).
            if not args.no_plots and cls not in plotted_classes:
                with torch.no_grad():
                    n_s = x_sub.shape[0]
                    pred_sub = model(
                        x_sub, ei_sub, drone_feat,
                        torch.zeros(n_s, dtype=torch.long, device=device)
                    ).argmax(-1).cpu().numpy()
                plot_subgraph(out_dir, expl, subset.cpu().numpy(), seed_local,
                              node, pred_sub, args.topk_edges)
                plotted_classes.add(cls)

        log.info(f"  [{gk}/{len(idx)}] graphe {gi} ({metas[gi]['map']}) : "
                 f"{n_done} nœuds expliqués.")

    if fmask_n == 0:
        log.error("Aucun nœud expliqué (sous-graphes trop gros ? baisse --num-hops "
                  "ou monte --max-subgraph-nodes).")
        sys.exit(1)

    log.info(f"Total : {fmask_n} nœuds expliqués.")

    # ── Agrégation (moyenne des masques) ──────────────────────────────────────
    global_imp = fmask_sum / fmask_n
    per_class_imp = np.zeros((NUM_CLASSES, NUM_FEATURES))
    for c in range(NUM_CLASSES):
        if per_class_n[c] > 0:
            per_class_imp[c] = per_class_sum[c] / per_class_n[c]

    order = write_csv(out_dir, global_imp, per_class_imp)
    log.success("Top features (masque GNNExplainer) :")
    for rank, f in enumerate(order[:10], 1):
        log.info(f"  {rank:2d}. {FEAT_KEYS[f]:<22} {global_imp[f]:.4e}")

    if not args.no_plots:
        plot_feature_importance(out_dir, global_imp, per_class_imp, order)
        log.success(f"Sous-graphes exemples : {out_dir / 'subgraphs'}")


if __name__ == "__main__":
    main()
