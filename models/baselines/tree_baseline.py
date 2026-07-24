"""Baselines tabulaires (arbre de décision / RandomForest / HistGradientBoosting)
pour la classification de nœuds SPL — à comparer directement au GATv2.

Idée
────
Un arbre ou une forêt ne « lit » pas un graphe : il traite chaque nœud comme une
ligne indépendante d'un tableau. On aplatit donc les graphes en (X, y) :

    features "brutes"    = [x_node(22) ; drone_feat(51)]                 → 73 cols
    features "voisinage" = brutes + mean/max des x des voisins directs   → +44 = 117 cols

La version "voisinage" donne à l'arbre une partie de l'information que le GNN
obtient par message passing (sans en être un) → baseline « forte ».

Le chargement des shards ET le split ville/altitude sont réutilisés depuis
`models/GATv2/dataio.py`, donc le découpage train/test est EXACTEMENT celui du
GNN : la comparaison des scores est honnête.

Usage
─────
    ./venv-local/bin/python models/baselines/tree_baseline.py
    ./venv-local/bin/python models/baselines/tree_baseline.py --features raw
    ./venv-local/bin/python models/baselines/tree_baseline.py \
        --models rf,hgb --max-train-nodes 2_000_000
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
    recall_score,
)
from sklearn.tree import DecisionTreeClassifier
from torch_geometric.utils import scatter

# ── Réutilise le pipeline GATv2 (chargement + split identiques au GNN) ──────────
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from models.GATv2 import config, dataio  # noqa: E402

RARE_CLASSES = [3, 4]   # mêmes classes "rares" que le GNN (core.py evaluate)


# ═══════════════════════════════════════════════════════════════════════════════
# Construction des matrices tabulaires
# ═══════════════════════════════════════════════════════════════════════════════
def _graph_rows(g, with_neighbors: bool) -> tuple[np.ndarray, np.ndarray]:
    """Transforme un graphe PyG en (X, y) au niveau nœud.

    X = [x_node(22) ; drone_feat(51) ({; mean_voisins(22) ; max_voisins(22)})]
    Les nœuds sans étiquette (y < 0) sont retirés.
    """
    x = g.x.float()                                    # (N, 22)
    n = x.shape[0]
    drone = g.drone_feat.float().expand(n, -1)         # (N, 51) broadcast du drone

    parts = [x, drone]
    if with_neighbors:
        src, dst = g.edge_index                        # (2, E), déjà symétrique
        mean_nbr = scatter(x[src], dst, dim=0, dim_size=n, reduce="mean")
        max_nbr  = scatter(x[src], dst, dim=0, dim_size=n, reduce="max")
        # Nœuds isolés : scatter renvoie 0 → on garde 0 (pas de voisin = neutre).
        parts += [mean_nbr, max_nbr]

    X = torch.cat(parts, dim=1).numpy()
    y = g.y.numpy()
    valid = y >= 0
    return X[valid], y[valid]


def iter_graphs():
    """Itère les graphes un shard à la fois (pic RAM ≈ un shard, pas tout le
    dataset). L'ordre reproduit celui de dataio.load_sharded_dataset, donc l'index
    global d'un graphe est cohérent avec les indices renvoyés par le split."""
    for p in sorted(config.SHARD_DIR.glob("shard_*.pt")):
        chunk = torch.load(p, weights_only=False)
        for g in chunk:
            if g.x.shape[1] > config.NUM_FEATURES:
                g.x = g.x[:, :config.NUM_FEATURES].contiguous()
            yield g
        del chunk


def scan_metas() -> list[dict]:
    """Construit la liste des métas (léger : quelques scalaires par graphe) en
    streaming, sans conserver les graphes en mémoire. Réplique le format de
    dataio.load_sharded_dataset pour rester compatible avec map_level_split."""
    metas: list[dict] = []
    for g in iter_graphs():
        y = g.y.numpy()
        valid = y >= 0
        counts = (np.bincount(y[valid], minlength=config.NUM_CLASSES).tolist()
                  if valid.any() else [0] * config.NUM_CLASSES)
        dp = getattr(g, "drone_pos", None)
        metas.append({
            "map":          getattr(g, "map_name", "unknown"),
            "drone":        getattr(g, "drone_id", "unknown"),
            "class_counts": counts,
            "n_nodes":      int(g.num_nodes),
            "z":            float(dp[2]) if dp is not None else 0.0,
        })
    return metas


def build_xy_streaming(keep_idx: set[int], with_neighbors: bool, frac: float,
                       seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Empile en (X, y) uniquement les graphes dont l'index global ∈ keep_idx,
    en sous-échantillonnant CHAQUE graphe à la fraction `frac` AVANT accumulation.

    Le pic mémoire reste borné : on ne matérialise jamais tous les nœuds bruts —
    `frac` est calculé en amont pour que le total tienne sous le plafond voulu.
    Le broadcast du vecteur drone (51 dims) est transitoire, par graphe.
    """
    rng = np.random.default_rng(seed)
    Xs, ys = [], []
    for gi, g in enumerate(iter_graphs()):
        if gi not in keep_idx:
            continue
        X, y = _graph_rows(g, with_neighbors)
        if frac < 1.0 and X.shape[0] > 0:
            k = max(1, int(round(X.shape[0] * frac)))
            sel = rng.choice(X.shape[0], size=k, replace=False)
            X, y = X[sel], y[sel]
        Xs.append(X.astype(np.float32))
        ys.append(y.astype(np.int64))
    return np.concatenate(Xs, axis=0), np.concatenate(ys, axis=0)


def _valid_nodes(metas, indices) -> int:
    return int(sum(sum(metas[i]["class_counts"]) for i in indices))


def _frac_for(metas, indices, cap: int | None) -> float:
    if not cap:
        return 1.0
    total = _valid_nodes(metas, indices)
    return min(1.0, cap / total) if total > cap else 1.0


def feature_names(with_neighbors: bool) -> list[str]:
    names = list(config.FEAT_KEYS)                      # 22 features nœud
    names += [f"drone_{k}" for k in range(config.DRONE_FEAT_DIM)]   # 51
    if with_neighbors:
        names += [f"nbr_mean_{k}" for k in config.FEAT_KEYS]
        names += [f"nbr_max_{k}"  for k in config.FEAT_KEYS]
    return names


# ═══════════════════════════════════════════════════════════════════════════════
# Modèles
# ═══════════════════════════════════════════════════════════════════════════════
def class_weight_dict() -> dict[int, float]:
    w = config.CLASS_WEIGHTS
    if w is None:
        return "balanced"          # sklearn calcule l'inverse-fréquence
    return {c: float(w[c]) for c in range(len(w))}


def model_factories(names: list[str], seed: int, n_jobs: int) -> dict:
    """Renvoie des *fabriques* (pas des instances) : chaque modèle n'est
    matérialisé qu'au moment de son fit dans la boucle, puis libéré — sinon une
    forêt déjà entraînée (~plusieurs GB) resterait en RAM pendant le fit suivant
    et ferait exploser le plafond mémoire."""
    cw = class_weight_dict()
    zoo = {
        "tree": lambda: DecisionTreeClassifier(
            max_depth=25, min_samples_leaf=20,
            class_weight=cw, random_state=seed),
        "rf": lambda: RandomForestClassifier(
            n_estimators=200, max_depth=None, min_samples_leaf=20,
            max_features="sqrt", class_weight=cw, n_jobs=n_jobs,
            random_state=seed),
        "hgb": lambda: HistGradientBoostingClassifier(
            max_iter=400, learning_rate=0.08, max_leaf_nodes=63,
            l2_regularization=1.0, early_stopping=True,
            class_weight=cw, random_state=seed),
    }
    unknown = [n for n in names if n not in zoo]
    if unknown:
        raise SystemExit(f"Modèle(s) inconnu(s): {unknown}; dispo: {list(zoo)}")
    return {n: zoo[n] for n in names}


# ═══════════════════════════════════════════════════════════════════════════════
# Évaluation — mêmes métriques que le GNN (core.py::evaluate)
# ═══════════════════════════════════════════════════════════════════════════════
def evaluate(preds, targets, num_classes: int) -> dict:
    rare = [c for c in RARE_CLASSES if c < num_classes]
    return {
        "acc":      float((preds == targets).mean()),
        "mcc":      float(matthews_corrcoef(targets, preds)),
        "bal_acc":  float(balanced_accuracy_score(targets, preds)),
        "macro_f1": float(f1_score(targets, preds, average="macro", zero_division=0)),
        "kappa":    float(cohen_kappa_score(targets, preds)),
        "rare_rec": float(recall_score(targets, preds, labels=rare,
                                       average="macro", zero_division=0)) if rare else 0.0,
    }


def format_per_class(preds, targets, num_classes: int) -> str:
    labels = list(range(num_classes))
    prec, rec, f1, sup = precision_recall_fscore_support(
        targets, preds, labels=labels, zero_division=0)
    lines = [f"  {'class':<9}{'precision':>11}{'recall':>11}{'f1':>11}{'support':>11}"]
    for c in labels:
        lines.append(f"  class {c:<3}{prec[c]:>11.4f}{rec[c]:>11.4f}"
                     f"{f1[c]:>11.4f}{int(sup[c]):>11d}")
    return "\n".join(lines)


def save_confusion(preds, targets, num_classes: int, path: Path, title: str) -> None:
    cm = confusion_matrix(targets, preds, labels=list(range(num_classes)))
    cm_norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(num_classes)); ax.set_yticks(range(num_classes))
    ax.set_xlabel("prédit"); ax.set_ylabel("vrai"); ax.set_title(title)
    for i in range(num_classes):
        for j in range(num_classes):
            ax.text(j, i, f"{cm_norm[i, j]:.2f}", ha="center", va="center",
                    color="white" if cm_norm[i, j] > 0.5 else "black", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", default="tree,rf,hgb",
                   help="liste séparée par des virgules parmi: tree, rf, hgb")
    p.add_argument("--features", default="both", choices=["raw", "nbr", "both"],
                   help="raw = nœud isolé ; nbr = + agrégats voisins ; both = les deux")
    p.add_argument("--max-train-nodes", type=int, default=3_000_000,
                   help="borne le nb de nœuds d'entraînement (mémoire/temps). 0 = tout")
    p.add_argument("--max-test-nodes", type=int, default=2_000_000,
                   help="borne le nb de nœuds de test (mémoire). 0 = tout")
    p.add_argument("--n-jobs", type=int, default=-1, help="cœurs pour RandomForest")
    p.add_argument("--seed", type=int, default=config.SEED)
    p.add_argument("--out", default=None, help="dossier de sortie (défaut: results/run_<ts>)")
    args = p.parse_args()

    model_names = [m.strip() for m in args.models.split(",") if m.strip()]
    feat_modes = {"raw": [False], "nbr": [True], "both": [False, True]}[args.features]

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out) if args.out else Path(__file__).resolve().parent / "results" / f"run_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "baseline.log"
    log_lines: list[str] = []

    def log(msg: str = "") -> None:
        print(msg)
        log_lines.append(msg)

    # ── Métas + split (streaming, identiques au GNN) ───────────────────────────
    log(f"[load] scan métas depuis {config.SHARD_DIR} (streaming, un shard à la fois)")
    metas = scan_metas()
    split = dataio.map_level_split(metas, config.VAL_RATIO, config.SPLIT_SEED)
    train_idx = set(split["train"][1])
    test_idx = set(split["test"][1])
    log(f"[split] {len(train_idx)} graphes train / {len(test_idx)} graphes test "
        f"(villes test: {sorted(split['test'][0])})")
    counts, weights, is_manual = dataio.compute_class_weights(metas, list(train_idx))
    log(f"[weights] class_weights={'manuels' if is_manual else 'auto'} -> "
        f"{np.round(weights, 3).tolist()}")
    log(f"[weights] counts train par classe = {counts.tolist()}")

    # Fractions de sous-échantillonnage calculées AVANT de matérialiser X → borne
    # le pic mémoire quels que soient le nb de nœuds réels.
    frac_tr = _frac_for(metas, train_idx, args.max_train_nodes or None)
    frac_te = _frac_for(metas, test_idx, args.max_test_nodes or None)
    log(f"[data] nœuds valides train={_valid_nodes(metas, train_idx):,} "
        f"test={_valid_nodes(metas, test_idx):,} | frac_train={frac_tr:.3f} "
        f"frac_test={frac_te:.3f}")

    summary: list[dict] = []
    for with_nbr in feat_modes:
        tag = "nbr" if with_nbr else "raw"
        log("\n" + "═" * 78)
        log(f"FEATURES = {tag}  ({len(feature_names(with_nbr))} colonnes)")
        log("═" * 78)

        t0 = time.time()
        Xtr, ytr = build_xy_streaming(train_idx, with_nbr, frac_tr, args.seed)
        Xte, yte = build_xy_streaming(test_idx, with_nbr, frac_te, args.seed)
        log(f"[data] X_train={Xtr.shape}  X_test={Xte.shape}  "
            f"(construit en {time.time() - t0:.1f}s)")

        for name, make in model_factories(model_names, args.seed, args.n_jobs).items():
            model = make()
            t0 = time.time()
            model.fit(Xtr, ytr)
            fit_s = time.time() - t0
            preds = model.predict(Xte)
            m = evaluate(preds, yte, config.NUM_CLASSES)
            log(f"\n── {name.upper()} [{tag}] ── (fit {fit_s:.1f}s)")
            log(f"  acc={m['acc']:.4f}  bal_acc={m['bal_acc']:.4f}  "
                f"mcc={m['mcc']:.4f}  macro_f1={m['macro_f1']:.4f}  "
                f"kappa={m['kappa']:.4f}  rare_rec={m['rare_rec']:.4f}")
            log(format_per_class(preds, yte, config.NUM_CLASSES))
            save_confusion(preds, yte, config.NUM_CLASSES,
                           out_dir / f"cm_{name}_{tag}.png",
                           f"{name} [{tag}] — confusion (norm. par ligne)")
            summary.append({"model": name, "features": tag, "fit_s": round(fit_s, 1), **m})
            # Flush incrémental : un arrêt tardif ne perd pas les modèles déjà faits.
            log_path.write_text("\n".join(log_lines) + "\n")
            (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
            # Libère le modèle (forêt/boosting) AVANT le fit suivant → pic mémoire
            # borné à X_train + X_test + un seul modèle.
            del model, preds
            gc.collect()

    # ── Récapitulatif comparatif ───────────────────────────────────────────────
    log("\n" + "═" * 78)
    log("RÉCAPITULATIF  (à comparer aux scores test du GATv2)")
    log("═" * 78)
    hdr = f"  {'model':<7}{'feat':<6}{'acc':>9}{'bal_acc':>9}{'mcc':>9}{'macro_f1':>10}{'kappa':>9}{'rare':>8}{'fit_s':>8}"
    log(hdr)
    for r in sorted(summary, key=lambda d: -d["macro_f1"]):
        log(f"  {r['model']:<7}{r['features']:<6}{r['acc']:>9.4f}{r['bal_acc']:>9.4f}"
            f"{r['mcc']:>9.4f}{r['macro_f1']:>10.4f}{r['kappa']:>9.4f}"
            f"{r['rare_rec']:>8.4f}{r['fit_s']:>8.1f}")

    log_path.write_text("\n".join(log_lines) + "\n")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    log(f"\n[done] résultats dans {out_dir}")


if __name__ == "__main__":
    main()
