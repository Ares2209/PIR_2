"""Analyse l'influence de l'architecture (num_layers, num_heads, attn_dropout)
sur la métrique optimisée par l'étude Optuna `gat-arch-v1`.

Lit la DB SQLite produite par `optuna_search_arch.py` et génère :
  - un CSV des trials (trials.csv)
  - un résumé texte (summary.txt)
  - des figures PNG dans figures/ :
      01_optim_history.png         best-so-far + valeurs par trial
      02_param_importance.png      importances fANOVA
      03_marginal_layers.png       value vs num_layers (box + scatter)
      04_marginal_heads.png        value vs num_heads  (box + scatter)
      05_marginal_attn_drop.png    value vs attn_dropout (scatter + lowess)
      06_heatmap_layers_heads.png  moyenne de la value par (layers, heads)
      07_learning_curves.png       courbes intermédiaires colorées par layers
      08_duration_vs_arch.png      durée par trial vs architecture

Usage :
    python analyze_arch_influence.py                       # défauts
    python /home/ldena/Bureau/PIR-local/PIR/models/GAT/optuna/optuna_runs/ana_arch/analyze_arch_influence.py --study gat-arch-v1 --db gat-arch-v1.db --out ./figures
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd

_HERE = Path(__file__).resolve().parent


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--db", type=Path, default=_HERE / "gat-arch-v1.db",
                   help="Chemin vers la DB SQLite Optuna.")
    p.add_argument("--study", type=str, default="gat-arch-v1",
                   help="Nom de l'étude dans la DB.")
    p.add_argument("--out", type=Path, default=_HERE / "figures",
                   help="Dossier de sortie pour les figures.")
    p.add_argument("--metric-label", type=str, default="f1m+mcc",
                   help="Label de la métrique pour les axes.")
    return p


def _load_study(db: Path, name: str) -> optuna.Study:
    if not db.exists():
        raise FileNotFoundError(f"DB introuvable : {db}")
    return optuna.load_study(study_name=name, storage=f"sqlite:///{db}")


def _trials_df(study: optuna.Study) -> pd.DataFrame:
    df = study.trials_dataframe(attrs=("number", "value", "params",
                                       "state", "duration"))
    df["duration_min"] = df["duration"].dt.total_seconds() / 60.0
    df = df.rename(columns={
        "params_num_layers":   "num_layers",
        "params_num_heads":    "num_heads",
        "params_attn_dropout": "attn_dropout",
    })
    return df


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
    ax.set_title("Historique d'optimisation")
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
    names = list(imp.keys())
    vals = [float(imp[n]) for n in names]
    order = np.argsort(vals)
    names = [names[i] for i in order]
    vals = [vals[i] for i in order]

    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.barh(names, vals, color="tab:purple")
    for y, v in enumerate(vals):
        ax.text(v + 0.005, y, f"{v:.2f}", va="center")
    ax.set_xlabel("importance (fANOVA, complete trials)")
    ax.set_title("Importance des hyperparamètres")
    ax.set_xlim(0, max(vals) * 1.15)
    fig.tight_layout()
    fig.savefig(out / "02_param_importance.png", dpi=140)
    plt.close(fig)


def _box_scatter_discrete(df: pd.DataFrame, col: str, metric: str,
                          path: Path, title: str) -> None:
    done = df[df["state"] == "COMPLETE"]
    levels = sorted(done[col].dropna().unique().astype(int).tolist())
    groups = [done.loc[done[col] == lv, "value"].to_numpy() for lv in levels]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.boxplot(groups, positions=levels, widths=0.55, showfliers=False,
               medianprops=dict(color="black", lw=1.5))
    # scatter individuel avec jitter
    rng = np.random.default_rng(0)
    for lv, g in zip(levels, groups):
        jitter = rng.uniform(-0.15, 0.15, size=len(g))
        ax.scatter(lv + jitter, g, s=22, alpha=0.7, color="tab:blue")
    # moyenne par niveau
    means = [float(g.mean()) if len(g) else np.nan for g in groups]
    ax.plot(levels, means, "o-", color="tab:orange", lw=1.5,
            label=f"mean {metric}")
    ax.set_xticks(levels)
    ax.set_xlabel(col)
    ax.set_ylabel(metric)
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_marginal_attn_dropout(df: pd.DataFrame, out: Path,
                               metric: str) -> None:
    done = df[df["state"] == "COMPLETE"].dropna(subset=["attn_dropout"])
    x = done["attn_dropout"].to_numpy()
    y = done["value"].to_numpy()

    # moyenne mobile triée
    order = np.argsort(x)
    xs, ys = x[order], y[order]
    k = max(3, len(xs) // 6)
    if len(xs) >= k:
        kernel = np.ones(k) / k
        ys_smooth = np.convolve(ys, kernel, mode="valid")
        xs_smooth = np.convolve(xs, kernel, mode="valid")
    else:
        xs_smooth, ys_smooth = xs, ys

    # régression linéaire indicative
    slope, intercept = np.polyfit(x, y, 1)
    xline = np.linspace(x.min(), x.max(), 50)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.scatter(x, y, s=28, alpha=0.75, color="tab:blue", label="trials")
    ax.plot(xs_smooth, ys_smooth, color="tab:orange", lw=1.5,
            label=f"moyenne mobile (k={k})")
    ax.plot(xline, slope * xline + intercept, "--", color="gray",
            label=f"linfit (pente={slope:+.2f})")
    ax.set_xlabel("attn_dropout")
    ax.set_ylabel(metric)
    ax.set_title("Effet marginal de attn_dropout")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "05_marginal_attn_drop.png", dpi=140)
    plt.close(fig)


def plot_heatmap_layers_heads(df: pd.DataFrame, out: Path,
                              metric: str) -> None:
    done = df[df["state"] == "COMPLETE"].copy()
    done["num_layers"] = done["num_layers"].astype(int)
    done["num_heads"] = done["num_heads"].astype(int)

    pivot = done.pivot_table(index="num_layers", columns="num_heads",
                             values="value", aggfunc="mean")
    counts = done.pivot_table(index="num_layers", columns="num_heads",
                              values="value", aggfunc="count").fillna(0).astype(int)

    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(pivot.values, aspect="auto", cmap="viridis",
                   origin="lower")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel("num_heads")
    ax.set_ylabel("num_layers")
    ax.set_title(f"Moyenne {metric} par (num_layers, num_heads)")
    for i, lv in enumerate(pivot.index):
        for j, hv in enumerate(pivot.columns):
            v = pivot.values[i, j]
            n = counts.values[i, j]
            if np.isnan(v):
                ax.text(j, i, "·", ha="center", va="center", color="white")
            else:
                ax.text(j, i, f"{v:.3f}\nn={n}", ha="center", va="center",
                        color="white" if v < pivot.values[~np.isnan(pivot.values)].mean()
                        else "black", fontsize=8)
    plt.colorbar(im, ax=ax, label=metric)
    fig.tight_layout()
    fig.savefig(out / "06_heatmap_layers_heads.png", dpi=140)
    plt.close(fig)


def plot_learning_curves(study: optuna.Study, out: Path,
                         metric: str) -> None:
    trials = [t for t in study.trials
              if t.state == optuna.trial.TrialState.COMPLETE
              and t.intermediate_values]
    if not trials:
        return
    cmap = plt.get_cmap("viridis")
    layers = np.array([t.params["num_layers"] for t in trials])
    lmin, lmax = int(layers.min()), int(layers.max())

    fig, ax = plt.subplots(figsize=(9, 4.5))
    for t in trials:
        epochs = sorted(t.intermediate_values.keys())
        vals = [t.intermediate_values[e] for e in epochs]
        c = cmap((t.params["num_layers"] - lmin) / max(1, lmax - lmin))
        ax.plot(epochs, vals, color=c, alpha=0.65, lw=1.0)
    ax.set_xlabel("epoch")
    ax.set_ylabel(metric)
    ax.set_title(f"Courbes d'apprentissage par trial "
                 f"(couleur = num_layers, {lmin} → {lmax})")
    ax.grid(alpha=0.3)
    sm = plt.cm.ScalarMappable(cmap=cmap,
                               norm=plt.Normalize(vmin=lmin, vmax=lmax))
    plt.colorbar(sm, ax=ax, label="num_layers")
    fig.tight_layout()
    fig.savefig(out / "07_learning_curves.png", dpi=140)
    plt.close(fig)


def plot_duration(df: pd.DataFrame, out: Path) -> None:
    done = df[df["state"] == "COMPLETE"].copy()
    done["arch"] = done["num_layers"].astype(int).astype(str) + "L×" \
                 + done["num_heads"].astype(int).astype(str) + "H"
    g = done.groupby("arch")["duration_min"].agg(["mean", "count"]).reset_index()
    g = g.sort_values("mean")

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(g["arch"], g["mean"], color="tab:cyan")
    for x, (m, n) in enumerate(zip(g["mean"], g["count"])):
        ax.text(x, m, f"n={n}", ha="center", va="bottom", fontsize=8)
    ax.set_xlabel("architecture (layers × heads)")
    ax.set_ylabel("durée moyenne (min)")
    ax.set_title("Coût d'entraînement par configuration")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out / "08_duration_vs_arch.png", dpi=140)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Textual summary
# ─────────────────────────────────────────────────────────────────────────────
def write_summary(study: optuna.Study, df: pd.DataFrame, out: Path,
                  metric: str) -> str:
    done = df[df["state"] == "COMPLETE"]
    pruned = df[df["state"] == "PRUNED"]
    best = study.best_trial

    lines: list[str] = []
    p = lines.append
    p(f"Study           : {study.study_name}")
    p(f"Metric          : {metric} (maximize)")
    p(f"Total trials    : {len(df)}  "
      f"(complete={len(done)}, pruned={len(pruned)})")
    p(f"Best trial      : #{best.number}  value={best.value:.4f}")
    p(f"  num_layers    : {best.params['num_layers']}")
    p(f"  num_heads     : {best.params['num_heads']}")
    p(f"  attn_dropout  : {best.params['attn_dropout']:.4f}")
    p("")

    # Top-5
    top = done.nlargest(5, "value")[
        ["number", "value", "num_layers", "num_heads", "attn_dropout",
         "duration_min"]
    ]
    p("Top 5 trials")
    p(top.to_string(index=False,
                    float_format=lambda x: f"{x:.4f}"))
    p("")

    # Effet marginal — moyennes par niveau
    for col in ("num_layers", "num_heads"):
        g = done.groupby(col)["value"].agg(["mean", "std", "count"])
        p(f"Effet marginal de {col}")
        p(g.to_string(float_format=lambda x: f"{x:.4f}"))
        p("")

    # Corrélation attn_dropout / value
    if len(done) >= 3:
        rho = np.corrcoef(done["attn_dropout"], done["value"])[0, 1]
        slope, _ = np.polyfit(done["attn_dropout"], done["value"], 1)
        p(f"attn_dropout vs {metric} : "
          f"Pearson={rho:+.3f}  | pente lin.={slope:+.3f}")
        p("")

    # Importance fANOVA
    try:
        imp = optuna.importance.get_param_importances(study)
        p("Importance des paramètres (fANOVA)")
        for k, v in sorted(imp.items(), key=lambda kv: -kv[1]):
            p(f"  {k:<15s} : {float(v):.3f}")
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
    df = _trials_df(study)
    df.to_csv(opts.out / "trials.csv", index=False)

    plot_optim_history(df, opts.out, opts.metric_label)
    plot_param_importance(study, opts.out)
    _box_scatter_discrete(df, "num_layers", opts.metric_label,
                          opts.out / "03_marginal_layers.png",
                          "Effet marginal de num_layers")
    _box_scatter_discrete(df, "num_heads", opts.metric_label,
                          opts.out / "04_marginal_heads.png",
                          "Effet marginal de num_heads")
    plot_marginal_attn_dropout(df, opts.out, opts.metric_label)
    plot_heatmap_layers_heads(df, opts.out, opts.metric_label)
    plot_learning_curves(study, opts.out, opts.metric_label)
    plot_duration(df, opts.out)

    text = write_summary(study, df, opts.out, opts.metric_label)
    print(text)
    print(f"→ figures + CSV + summary écrits dans {opts.out}")

    # Dump aussi un best.json local (pratique pour scripts en aval)
    best = study.best_trial
    (opts.out / "best.json").write_text(json.dumps({
        "metric": opts.metric_label,
        "value": best.value,
        "trial": best.number,
        "params": best.params,
    }, indent=2))


if __name__ == "__main__":
    main()
