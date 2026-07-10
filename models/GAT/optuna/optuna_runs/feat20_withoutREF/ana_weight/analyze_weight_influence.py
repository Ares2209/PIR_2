"""Analyse l'influence du vecteur CLASS_WEIGHTS (w0..w6) sur la métrique
optimisée par l'étude Optuna `gat-class-weights`.

Lit la DB SQLite produite par `optuna_search.py` et génère :
  - un CSV des trials (trials.csv)
  - un résumé texte (summary.txt)
  - un best.json local
  - des figures PNG dans figures/ :
      01_optim_history.png         best-so-far + valeurs par trial
      02_param_importance.png      importances fANOVA par classe
      03_marginal_weights.png      pour chaque classe : weight (log) vs value
      04_top_weights.png           profils des top-K trials (normalisés)
      05_corr_matrix.png           corrélation Spearman (w_i, value) + pairwise
      06_best_vs_current.png       comparaison best vs CLASS_WEIGHTS de config
      07_learning_curves.png       courbes intermédiaires colorées par value
      08_weight_distribution.png   distribution des poids (top vs bottom)

Usage :
    python analyze_weight_influence.py
    python /home/ldena/Bureau/PIR-local/PIR/models/GAT/optuna/optuna_runs/ana_weight/analyze_weight_influence.py --study gat-class-weights --db gat-class-weights.db --out ./figures
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd

_HERE = Path(__file__).resolve().parent

# Make `config` importable (lives in ../../../GAT/).
_GAT_DIR = _HERE.parent.parent.parent
if str(_GAT_DIR) not in sys.path:
    sys.path.insert(0, str(_GAT_DIR))
try:
    import config as _gat_config  # noqa: E402
    _NUM_CLASSES = int(_gat_config.NUM_CLASSES)
    _CURRENT_WEIGHTS = list(_gat_config.CLASS_WEIGHTS)
except Exception:
    _gat_config = None
    _NUM_CLASSES = 7
    _CURRENT_WEIGHTS = None

_CLASS_NAMES = ["violet", "blue", "yellow", "orange", "red",
                "dark_red", "occluded"][:_NUM_CLASSES]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--db", type=Path, default=_HERE / "gat-class-weights.db",
                   help="Chemin vers la DB SQLite Optuna.")
    p.add_argument("--study", type=str, default="gat-class-weights",
                   help="Nom de l'étude dans la DB.")
    p.add_argument("--out", type=Path, default=_HERE / "figures",
                   help="Dossier de sortie pour les figures.")
    p.add_argument("--metric-label", type=str, default="bal",
                   help="Label de la métrique pour les axes.")
    p.add_argument("--top-k", type=int, default=10,
                   help="Nombre de trials top considérés pour les figures "
                        "comparatives (04, 08).")
    p.add_argument("--use-normalized", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="Utiliser les poids normalisés (user_attr w{c}_norm) "
                        "quand ils existent. Sinon utilise les params bruts.")
    return p


def _load_study(db: Path, name: str) -> optuna.Study:
    if not db.exists():
        raise FileNotFoundError(f"DB introuvable : {db}")
    return optuna.load_study(study_name=name, storage=f"sqlite:///{db}")


def _trials_df(study: optuna.Study, use_norm: bool) -> pd.DataFrame:
    rows = []
    for t in study.trials:
        row = {
            "number": t.number,
            "value": t.value if t.value is not None else np.nan,
            "state": t.state.name,
            "duration_min": (t.duration.total_seconds() / 60.0
                             if t.duration is not None else np.nan),
        }
        for c in range(_NUM_CLASSES):
            raw = t.params.get(f"w{c}", np.nan)
            row[f"w{c}_raw"] = raw
            norm_attr = t.user_attrs.get(f"w{c}_norm")
            if use_norm and norm_attr is not None:
                row[f"w{c}"] = float(norm_attr)
            else:
                row[f"w{c}"] = raw
        rows.append(row)
    return pd.DataFrame(rows)


def _weights_matrix(df: pd.DataFrame) -> np.ndarray:
    """(N, _NUM_CLASSES) matrix of (effective) weights for completed trials."""
    return df[[f"w{c}" for c in range(_NUM_CLASSES)]].to_numpy(dtype=float)


# ─────────────────────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────────────────────
def plot_optim_history(df: pd.DataFrame, out: Path, metric: str) -> None:
    done = df[df["state"] == "COMPLETE"].sort_values("number")
    pruned = df[df["state"] == "PRUNED"]
    best_so_far = done["value"].cummax()

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.scatter(done["number"], done["value"], s=30, c="tab:blue",
               label="complete", zorder=3)
    ax.scatter(pruned["number"], pruned["value"], s=30, c="tab:red",
               marker="x", label="pruned (last reported)", zorder=3)
    ax.plot(done["number"], best_so_far, "k-", lw=1.5,
            label="best so far", zorder=2)
    ax.set_xlabel("trial #")
    ax.set_ylabel(metric)
    ax.set_title("Historique d'optimisation — class weights")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "01_optim_history.png", dpi=140)
    plt.close(fig)


def plot_param_importance(study: optuna.Study, out: Path) -> None:
    try:
        imp = optuna.importance.get_param_importances(study)
    except Exception as e:
        print(f"[warn] importance non calculable : {e}")
        return
    pretty = {}
    for k, v in imp.items():
        if k.startswith("w") and k[1:].isdigit():
            i = int(k[1:])
            label = f"w{i} ({_CLASS_NAMES[i]})" if i < len(_CLASS_NAMES) else k
        else:
            label = k
        pretty[label] = float(v)

    names = list(pretty.keys())
    vals = [pretty[n] for n in names]
    order = np.argsort(vals)
    names = [names[i] for i in order]
    vals = [vals[i] for i in order]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh(names, vals, color="tab:purple")
    for y, v in enumerate(vals):
        ax.text(v + max(vals) * 0.01, y, f"{v:.2f}", va="center", fontsize=9)
    ax.set_xlabel("importance (fANOVA, complete trials)")
    ax.set_title("Importance des poids de classe")
    ax.set_xlim(0, max(vals) * 1.18 if max(vals) > 0 else 1)
    fig.tight_layout()
    fig.savefig(out / "02_param_importance.png", dpi=140)
    plt.close(fig)


def plot_marginal_weights(df: pd.DataFrame, out: Path, metric: str) -> None:
    done = df[df["state"] == "COMPLETE"].copy()
    if len(done) < 3:
        return

    n_cols = 4
    n_rows = int(np.ceil(_NUM_CLASSES / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.0 * n_cols,
                                                       3.2 * n_rows),
                             squeeze=False)

    y = done["value"].to_numpy()
    for c in range(_NUM_CLASSES):
        ax = axes[c // n_cols][c % n_cols]
        x = done[f"w{c}"].to_numpy()
        # Pearson en log car les poids sont samplés en log-uniforme
        with np.errstate(invalid="ignore"):
            rho = np.corrcoef(np.log(np.clip(x, 1e-6, None)), y)[0, 1]

        ax.scatter(x, y, s=22, alpha=0.7, color="tab:blue")

        # tendance via moyenne mobile sur x trié
        order = np.argsort(x)
        xs, ys = x[order], y[order]
        k = max(3, len(xs) // 6)
        if len(xs) >= k:
            kernel = np.ones(k) / k
            ax.plot(np.convolve(xs, kernel, mode="valid"),
                    np.convolve(ys, kernel, mode="valid"),
                    color="tab:orange", lw=1.4,
                    label=f"moy. mob. (k={k})")
        title = (f"w{c} — {_CLASS_NAMES[c]}"
                 if c < len(_CLASS_NAMES) else f"w{c}")
        ax.set_title(f"{title}\nρ_log = {rho:+.2f}", fontsize=10)
        ax.set_xscale("log")
        ax.set_xlabel(f"w{c}")
        ax.set_ylabel(metric)
        ax.grid(alpha=0.3)
        if c == 0:
            ax.legend(loc="lower right", fontsize=8)

    # cacher les axes non utilisés
    for k in range(_NUM_CLASSES, n_rows * n_cols):
        axes[k // n_cols][k % n_cols].axis("off")

    fig.suptitle("Effet marginal de chaque poids de classe",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(out / "03_marginal_weights.png", dpi=140,
                bbox_inches="tight")
    plt.close(fig)


def plot_top_weights(df: pd.DataFrame, out: Path, top_k: int,
                     use_norm: bool) -> None:
    done = df[df["state"] == "COMPLETE"].copy()
    if done.empty:
        return
    top = done.nlargest(min(top_k, len(done)), "value")
    W = top[[f"w{c}" for c in range(_NUM_CLASSES)]].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    x = np.arange(_NUM_CLASSES)
    cmap = plt.get_cmap("viridis")
    vmin, vmax = float(top["value"].min()), float(top["value"].max())

    for i, (_, row) in enumerate(top.iterrows()):
        c = cmap((row["value"] - vmin) / max(1e-9, vmax - vmin))
        ax.plot(x, W[i], "-o", color=c, alpha=0.85, lw=1.4,
                label=f"#{int(row['number'])} ({row['value']:.4f})")

    # médiane top-K
    ax.plot(x, np.median(W, axis=0), "k--", lw=2.0,
            label=f"médiane top-{len(top)}")

    if _CURRENT_WEIGHTS is not None and len(_CURRENT_WEIGHTS) == _NUM_CLASSES:
        ax.plot(x, _CURRENT_WEIGHTS, "r:^", lw=1.8,
                label="config.CLASS_WEIGHTS")

    ax.set_xticks(x)
    ax.set_xticklabels([f"w{c}\n{_CLASS_NAMES[c]}" if c < len(_CLASS_NAMES)
                        else f"w{c}" for c in x], fontsize=9)
    ax.set_yscale("log")
    label = "poids normalisés (mean=1)" if use_norm else "poids bruts"
    ax.set_ylabel(label)
    ax.set_title(f"Profils des top-{len(top)} trials")
    ax.grid(alpha=0.3, which="both")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8,
              borderaxespad=0.0)
    fig.tight_layout()
    fig.savefig(out / "04_top_weights.png", dpi=140,
                bbox_inches="tight")
    plt.close(fig)


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3:
        return float("nan")
    ra = pd.Series(a).rank().to_numpy()
    rb = pd.Series(b).rank().to_numpy()
    return float(np.corrcoef(ra, rb)[0, 1])


def plot_corr_matrix(df: pd.DataFrame, out: Path, metric: str) -> None:
    done = df[df["state"] == "COMPLETE"].copy()
    if len(done) < 3:
        return
    cols = [f"w{c}" for c in range(_NUM_CLASSES)] + ["value"]
    M = np.zeros((len(cols), len(cols)), dtype=float)
    arrs = {col: done[col].to_numpy(dtype=float) for col in cols}
    for i, ci in enumerate(cols):
        for j, cj in enumerate(cols):
            M[i, j] = _spearman(arrs[ci], arrs[cj])

    labels = []
    for c in range(_NUM_CLASSES):
        labels.append(f"w{c}\n{_CLASS_NAMES[c]}"
                      if c < len(_CLASS_NAMES) else f"w{c}")
    labels.append(metric)

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(M, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(len(cols)))
    ax.set_yticks(range(len(cols)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)
    for i in range(len(cols)):
        for j in range(len(cols)):
            v = M[i, j]
            ax.text(j, i, f"{v:+.2f}", ha="center", va="center",
                    fontsize=8,
                    color="white" if abs(v) > 0.5 else "black")
    plt.colorbar(im, ax=ax, label="Spearman ρ")
    ax.set_title(f"Corrélations Spearman (poids ↔ poids, poids ↔ {metric})")
    fig.tight_layout()
    fig.savefig(out / "05_corr_matrix.png", dpi=140)
    plt.close(fig)


def plot_best_vs_current(df: pd.DataFrame, out: Path, use_norm: bool) -> None:
    done = df[df["state"] == "COMPLETE"]
    if done.empty:
        return
    best_row = done.loc[done["value"].idxmax()]
    best_w = np.array([best_row[f"w{c}"] for c in range(_NUM_CLASSES)],
                      dtype=float)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(_NUM_CLASSES)
    w = 0.36
    ax.bar(x - w / 2, best_w, width=w, color="tab:green",
           label=f"best Optuna (#{int(best_row['number'])}, "
                 f"{best_row['value']:.4f})")
    if _CURRENT_WEIGHTS is not None and len(_CURRENT_WEIGHTS) == _NUM_CLASSES:
        ax.bar(x + w / 2, _CURRENT_WEIGHTS, width=w, color="tab:red",
               alpha=0.85, label="config.CLASS_WEIGHTS")
    ax.set_xticks(x)
    ax.set_xticklabels([f"w{c}\n{_CLASS_NAMES[c]}" if c < len(_CLASS_NAMES)
                        else f"w{c}" for c in x], fontsize=9)
    ax.set_yscale("log")
    ax.set_ylabel("poids normalisés (mean=1)" if use_norm else "poids bruts")
    ax.set_title("Best trial vs config actuelle")
    for xi, v in zip(x - w / 2, best_w):
        ax.text(xi, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    if _CURRENT_WEIGHTS is not None and len(_CURRENT_WEIGHTS) == _NUM_CLASSES:
        for xi, v in zip(x + w / 2, _CURRENT_WEIGHTS):
            ax.text(xi, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    ax.grid(alpha=0.3, axis="y", which="both")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out / "06_best_vs_current.png", dpi=140)
    plt.close(fig)


def plot_learning_curves(study: optuna.Study, out: Path,
                         metric: str) -> None:
    trials = [t for t in study.trials
              if t.state == optuna.trial.TrialState.COMPLETE
              and t.intermediate_values]
    if not trials:
        return
    vmin = min(t.value for t in trials)
    vmax = max(t.value for t in trials)
    cmap = plt.get_cmap("viridis")

    fig, ax = plt.subplots(figsize=(9, 4.5))
    for t in trials:
        epochs = sorted(t.intermediate_values.keys())
        vals = [t.intermediate_values[e] for e in epochs]
        c = cmap((t.value - vmin) / max(1e-9, vmax - vmin))
        ax.plot(epochs, vals, color=c, alpha=0.65, lw=1.0)
    ax.set_xlabel("epoch")
    ax.set_ylabel(metric)
    ax.set_title(f"Courbes d'apprentissage par trial "
                 f"(couleur = {metric} final)")
    ax.grid(alpha=0.3)
    sm = plt.cm.ScalarMappable(cmap=cmap,
                               norm=plt.Normalize(vmin=vmin, vmax=vmax))
    plt.colorbar(sm, ax=ax, label=f"{metric} final")
    fig.tight_layout()
    fig.savefig(out / "07_learning_curves.png", dpi=140)
    plt.close(fig)


def plot_weight_distribution(df: pd.DataFrame, out: Path, top_k: int,
                             use_norm: bool) -> None:
    done = df[df["state"] == "COMPLETE"].copy()
    if len(done) < 2 * top_k:
        # garde-fou : pas la peine de séparer top/bottom si trop peu de trials
        top_k = max(1, len(done) // 3)

    sorted_done = done.sort_values("value", ascending=False)
    top = sorted_done.head(top_k)
    bot = sorted_done.tail(top_k)

    top_W = top[[f"w{c}" for c in range(_NUM_CLASSES)]].to_numpy(dtype=float)
    bot_W = bot[[f"w{c}" for c in range(_NUM_CLASSES)]].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(10, 5))
    positions = np.arange(_NUM_CLASSES)
    width = 0.38
    bp_top = ax.boxplot(
        [top_W[:, c] for c in range(_NUM_CLASSES)],
        positions=positions - width / 2, widths=width, patch_artist=True,
        showfliers=False, medianprops=dict(color="black", lw=1.4),
    )
    for box in bp_top["boxes"]:
        box.set_facecolor("tab:green")
        box.set_alpha(0.7)
    bp_bot = ax.boxplot(
        [bot_W[:, c] for c in range(_NUM_CLASSES)],
        positions=positions + width / 2, widths=width, patch_artist=True,
        showfliers=False, medianprops=dict(color="black", lw=1.4),
    )
    for box in bp_bot["boxes"]:
        box.set_facecolor("tab:red")
        box.set_alpha(0.7)

    ax.set_xticks(positions)
    ax.set_xticklabels([f"w{c}\n{_CLASS_NAMES[c]}" if c < len(_CLASS_NAMES)
                        else f"w{c}" for c in positions], fontsize=9)
    ax.set_yscale("log")
    ax.set_ylabel("poids normalisés (mean=1)" if use_norm else "poids bruts")
    ax.set_title(f"Distribution des poids — top-{top_k} (vert) "
                 f"vs bottom-{top_k} (rouge)")
    ax.grid(alpha=0.3, axis="y", which="both")
    # légende custom
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(facecolor="tab:green", alpha=0.7, label=f"top-{top_k}"),
        Patch(facecolor="tab:red", alpha=0.7, label=f"bottom-{top_k}"),
    ], loc="upper right")
    fig.tight_layout()
    fig.savefig(out / "08_weight_distribution.png", dpi=140)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Textual summary
# ─────────────────────────────────────────────────────────────────────────────
def write_summary(study: optuna.Study, df: pd.DataFrame, out: Path,
                  metric: str, use_norm: bool) -> str:
    done = df[df["state"] == "COMPLETE"]
    pruned = df[df["state"] == "PRUNED"]
    best = study.best_trial

    lines: list[str] = []
    p = lines.append
    p(f"Study           : {study.study_name}")
    p(f"Metric          : {metric} (maximize)")
    p(f"Weight scale    : {'normalized (mean=1)' if use_norm else 'raw'}")
    p(f"Total trials    : {len(df)}  "
      f"(complete={len(done)}, pruned={len(pruned)})")
    p(f"Best trial      : #{best.number}  value={best.value:.4f}")
    raw_w = np.array([best.params[f"w{c}"] for c in range(_NUM_CLASSES)],
                     dtype=float)
    norm_w = raw_w / raw_w.mean()
    p("  raw weights        : [" + ", ".join(f"{w:.4f}" for w in raw_w) + "]")
    p("  normalized weights : [" + ", ".join(f"{w:.4f}" for w in norm_w) + "]")
    p("  → CLASS_WEIGHTS = [" + ", ".join(f"{w:.3f}" for w in norm_w) + "]")
    if _CURRENT_WEIGHTS is not None and len(_CURRENT_WEIGHTS) == _NUM_CLASSES:
        p("  config actuel     : [" +
          ", ".join(f"{w:.4f}" for w in _CURRENT_WEIGHTS) + "]")
    p("")

    # Top-5 trials
    cols = ["number", "value"] + [f"w{c}" for c in range(_NUM_CLASSES)] \
        + ["duration_min"]
    top = done.nlargest(5, "value")[cols]
    p("Top 5 trials")
    p(top.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    p("")

    # Corrélations Spearman log(w_i) vs value
    p(f"Corrélations Spearman log(w_i) ↔ {metric}")
    for c in range(_NUM_CLASSES):
        col = f"w{c}"
        if len(done) >= 3:
            x = np.log(np.clip(done[col].to_numpy(dtype=float), 1e-6, None))
            y = done["value"].to_numpy()
            ra = pd.Series(x).rank().to_numpy()
            rb = pd.Series(y).rank().to_numpy()
            rho = float(np.corrcoef(ra, rb)[0, 1])
            name = (_CLASS_NAMES[c] if c < len(_CLASS_NAMES) else "")
            p(f"  w{c} ({name:<9s}) : ρ = {rho:+.3f}")
    p("")

    # Importance fANOVA
    try:
        imp = optuna.importance.get_param_importances(study)
        p("Importance des poids (fANOVA)")
        for k, v in sorted(imp.items(), key=lambda kv: -kv[1]):
            label = k
            if k.startswith("w") and k[1:].isdigit():
                i = int(k[1:])
                if i < len(_CLASS_NAMES):
                    label = f"{k} ({_CLASS_NAMES[i]})"
            p(f"  {label:<22s} : {float(v):.3f}")
    except Exception as e:
        p(f"[importance non calculable: {e}]")

    text = "\n".join(lines) + "\n"
    (out / "summary.txt").write_text(text)
    return text


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    opts = build_parser().parse_args()
    opts.out.mkdir(parents=True, exist_ok=True)

    study = _load_study(opts.db, opts.study)
    df = _trials_df(study, opts.use_normalized)
    df.to_csv(opts.out / "trials.csv", index=False)

    plot_optim_history(df, opts.out, opts.metric_label)
    plot_param_importance(study, opts.out)
    plot_marginal_weights(df, opts.out, opts.metric_label)
    plot_top_weights(df, opts.out, opts.top_k, opts.use_normalized)
    plot_corr_matrix(df, opts.out, opts.metric_label)
    plot_best_vs_current(df, opts.out, opts.use_normalized)
    plot_learning_curves(study, opts.out, opts.metric_label)
    plot_weight_distribution(df, opts.out, opts.top_k, opts.use_normalized)

    text = write_summary(study, df, opts.out, opts.metric_label,
                         opts.use_normalized)
    print(text)
    print(f"→ figures + CSV + summary écrits dans {opts.out}")

    # best.json local (pratique pour scripts en aval)
    best = study.best_trial
    raw = np.array([best.params[f"w{c}"] for c in range(_NUM_CLASSES)],
                   dtype=float)
    norm = (raw / raw.mean()).tolist()
    (opts.out / "best.json").write_text(json.dumps({
        "metric":             opts.metric_label,
        "value":              best.value,
        "trial":              best.number,
        "raw_weights":        raw.tolist(),
        "normalized_weights": norm,
        "params":             best.params,
    }, indent=2))


if __name__ == "__main__":
    main()
