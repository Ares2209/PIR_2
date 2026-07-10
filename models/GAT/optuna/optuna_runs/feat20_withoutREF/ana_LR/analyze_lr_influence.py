"""Analyse l'influence du learning rate (et optionnellement du weight_decay)
sur la métrique optimisée par l'étude Optuna `gat-lr`.

Lit la DB SQLite produite par `optuna_search_lr.py` et génère :
  - un CSV des trials (trials.csv)
  - un résumé texte (summary.txt)
  - un best.json local
  - des figures PNG dans figures/ :
      01_optim_history.png         best-so-far + valeurs par trial
      02_param_importance.png      importances fANOVA
      03_marginal_lr.png           value vs lr (échelle log + moy. mobile + linfit)
      04_marginal_wd.png           value vs weight_decay (si présent)
      05_scatter_lr_wd.png         scatter 2D (lr, wd) coloré par value
      06_learning_curves.png       courbes intermédiaires colorées par lr
      07_duration_vs_lr.png        durée par trial vs lr (bucketisé)
      08_pruned_vs_complete.png    distribution lr selon état du trial

Usage :
    python analyze_lr_influence.py
    python /home/ldena/Bureau/PIR-local/PIR/models/GAT/optuna/optuna_runs/ana_LR/analyze_lr_influence.py \\
        --study gat-lr --db gat-lr.db --out ./figures
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
    p.add_argument("--db", type=Path, default=_HERE / "gat-lr.db",
                   help="Chemin vers la DB SQLite Optuna.")
    p.add_argument("--study", type=str, default="gat-lr",
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
    rename = {"params_lr": "lr"}
    if "params_weight_decay" in df.columns:
        rename["params_weight_decay"] = "weight_decay"
    df = df.rename(columns=rename)
    return df


def _has_wd(df: pd.DataFrame) -> bool:
    return "weight_decay" in df.columns and df["weight_decay"].notna().any()


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
    ax.set_xlim(0, max(vals) * 1.15 if vals else 1.0)
    fig.tight_layout()
    fig.savefig(out / "02_param_importance.png", dpi=140)
    plt.close(fig)


def _marginal_log_x(df: pd.DataFrame, col: str, metric: str, path: Path,
                    title: str, xlabel: str) -> None:
    """Scatter en log-x + moyenne mobile (en log-space) + fit linéaire log-x."""
    done = df[df["state"] == "COMPLETE"].dropna(subset=[col])
    if done.empty:
        return
    x = done[col].to_numpy(dtype=float)
    y = done["value"].to_numpy(dtype=float)
    logx = np.log10(x)

    order = np.argsort(logx)
    xs, ys = logx[order], y[order]
    k = max(3, len(xs) // 6)
    if len(xs) >= k:
        kernel = np.ones(k) / k
        ys_smooth = np.convolve(ys, kernel, mode="valid")
        xs_smooth = np.convolve(xs, kernel, mode="valid")
    else:
        xs_smooth, ys_smooth = xs, ys

    slope, intercept = np.polyfit(logx, y, 1)
    xline = np.linspace(logx.min(), logx.max(), 80)

    # marquer le best
    best_idx = int(np.argmax(y))

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.scatter(x, y, s=28, alpha=0.75, color="tab:blue", label="trials")
    ax.plot(10 ** xs_smooth, ys_smooth, color="tab:orange", lw=1.6,
            label=f"moyenne mobile (k={k})")
    ax.plot(10 ** xline, slope * xline + intercept, "--", color="gray",
            label=f"linfit log-x (pente={slope:+.3f})")
    ax.scatter([x[best_idx]], [y[best_idx]], s=120, marker="*",
               color="tab:red", edgecolor="black", zorder=5,
               label=f"best ({xlabel}={x[best_idx]:.2e})")
    ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(metric)
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_scatter_lr_wd(df: pd.DataFrame, out: Path, metric: str) -> None:
    done = df[df["state"] == "COMPLETE"].dropna(subset=["lr", "weight_decay"])
    if done.empty:
        return
    x = done["lr"].to_numpy(dtype=float)
    y = done["weight_decay"].to_numpy(dtype=float)
    c = done["value"].to_numpy(dtype=float)
    best_idx = int(np.argmax(c))

    fig, ax = plt.subplots(figsize=(7.5, 5))
    sc = ax.scatter(x, y, c=c, s=60, cmap="viridis",
                    edgecolor="white", linewidth=0.5)
    ax.scatter([x[best_idx]], [y[best_idx]], s=220, marker="*",
               facecolor="none", edgecolor="red", linewidth=2,
               label=f"best (#{int(done.iloc[best_idx]['number'])} : {c[best_idx]:.4f})")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("lr")
    ax.set_ylabel("weight_decay")
    ax.set_title(f"Trials dans l'espace (lr, weight_decay), couleur = {metric}")
    ax.grid(alpha=0.3, which="both")
    ax.legend(loc="upper right", fontsize=9)
    plt.colorbar(sc, ax=ax, label=metric)
    fig.tight_layout()
    fig.savefig(out / "05_scatter_lr_wd.png", dpi=140)
    plt.close(fig)


def plot_learning_curves(study: optuna.Study, out: Path, metric: str) -> None:
    trials = [t for t in study.trials
              if t.state == optuna.trial.TrialState.COMPLETE
              and t.intermediate_values]
    if not trials:
        return
    cmap = plt.get_cmap("viridis")
    lrs = np.array([t.params["lr"] for t in trials])
    log_lrs = np.log10(lrs)
    lmin, lmax = float(log_lrs.min()), float(log_lrs.max())

    fig, ax = plt.subplots(figsize=(9, 4.5))
    for t, llr in zip(trials, log_lrs):
        epochs = sorted(t.intermediate_values.keys())
        vals = [t.intermediate_values[e] for e in epochs]
        c = cmap((llr - lmin) / max(1e-9, lmax - lmin))
        ax.plot(epochs, vals, color=c, alpha=0.7, lw=1.0)
    ax.set_xlabel("epoch")
    ax.set_ylabel(metric)
    ax.set_title(f"Courbes d'apprentissage par trial "
                 f"(couleur = log10(lr), {lmin:.2f} → {lmax:.2f})")
    ax.grid(alpha=0.3)
    sm = plt.cm.ScalarMappable(cmap=cmap,
                               norm=plt.Normalize(vmin=lmin, vmax=lmax))
    plt.colorbar(sm, ax=ax, label="log10(lr)")
    fig.tight_layout()
    fig.savefig(out / "06_learning_curves.png", dpi=140)
    plt.close(fig)


def plot_duration_vs_lr(df: pd.DataFrame, out: Path) -> None:
    done = df[df["state"] == "COMPLETE"].dropna(subset=["lr", "duration_min"])
    if done.empty:
        return
    x = done["lr"].to_numpy(dtype=float)
    y = done["duration_min"].to_numpy(dtype=float)

    # bucketisation en log
    nb = min(6, max(3, len(x) // 5))
    edges = np.logspace(np.log10(x.min()), np.log10(x.max()), nb + 1)
    bins = np.digitize(x, edges[1:-1])
    centers = 0.5 * (edges[:-1] + edges[1:])
    means = [y[bins == i].mean() if np.any(bins == i) else np.nan
             for i in range(nb)]
    counts = [int(np.sum(bins == i)) for i in range(nb)]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.scatter(x, y, s=24, alpha=0.6, color="tab:blue", label="trials")
    ax.plot(centers, means, "o-", color="tab:orange", lw=1.8,
            label="moyenne par bucket")
    for c, m, n in zip(centers, means, counts):
        if not np.isnan(m):
            ax.text(c, m, f"n={n}", fontsize=8, ha="center", va="bottom")
    ax.set_xscale("log")
    ax.set_xlabel("lr")
    ax.set_ylabel("durée (min)")
    ax.set_title("Durée d'entraînement vs lr (les très petits/grands LR pruneront tôt)")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(out / "07_duration_vs_lr.png", dpi=140)
    plt.close(fig)


def plot_pruned_vs_complete(df: pd.DataFrame, out: Path) -> None:
    """Distribution du lr selon que le trial a été pruné ou non.
    Indique la 'fenêtre saine' de LR."""
    sub = df.dropna(subset=["lr"]).copy()
    if sub.empty:
        return
    sub["log_lr"] = np.log10(sub["lr"].astype(float))
    states = ["COMPLETE", "PRUNED", "FAIL"]
    colors = {"COMPLETE": "tab:blue", "PRUNED": "tab:red", "FAIL": "tab:gray"}

    fig, ax = plt.subplots(figsize=(9, 4.5))
    bins = np.linspace(sub["log_lr"].min(), sub["log_lr"].max(), 18)
    for st in states:
        s = sub[sub["state"] == st]
        if s.empty:
            continue
        ax.hist(s["log_lr"], bins=bins, alpha=0.55,
                label=f"{st} (n={len(s)})", color=colors.get(st, "k"))
    ax.set_xlabel("log10(lr)")
    ax.set_ylabel("# trials")
    ax.set_title("Distribution du lr par état du trial")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out / "08_pruned_vs_complete.png", dpi=140)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Textual summary
# ─────────────────────────────────────────────────────────────────────────────
def write_summary(study: optuna.Study, df: pd.DataFrame, out: Path,
                  metric: str) -> str:
    done = df[df["state"] == "COMPLETE"]
    pruned = df[df["state"] == "PRUNED"]
    best = study.best_trial
    has_wd = _has_wd(df)

    lines: list[str] = []
    p = lines.append
    p(f"Study           : {study.study_name}")
    p(f"Metric          : {metric} (maximize)")
    p(f"Total trials    : {len(df)}  "
      f"(complete={len(done)}, pruned={len(pruned)})")
    p(f"Best trial      : #{best.number}  value={best.value:.4f}")
    p(f"  lr            : {best.params['lr']:.3e}")
    if has_wd and "weight_decay" in best.params:
        p(f"  weight_decay  : {best.params['weight_decay']:.3e}")
    p("")

    # Top-5
    cols = ["number", "value", "lr"]
    if has_wd:
        cols.append("weight_decay")
    cols.append("duration_min")
    top = done.nlargest(5, "value")[cols]
    p("Top 5 trials")
    p(top.to_string(index=False,
                    float_format=lambda x: f"{x:.4e}" if abs(x) < 1e-2 else f"{x:.4f}"))
    p("")

    # Stats par décade de lr (utile en log)
    if not done.empty:
        decades = np.floor(np.log10(done["lr"].astype(float)))
        g = done.groupby(decades)["value"].agg(["mean", "std", "count"])
        g.index = [f"1e{int(d)} … 1e{int(d)+1}" for d in g.index]
        p("Effet marginal de lr (par décade)")
        p(g.to_string(float_format=lambda x: f"{x:.4f}"))
        p("")

    # Corrélations
    if len(done) >= 3:
        log_lr = np.log10(done["lr"].astype(float))
        rho_lr = float(np.corrcoef(log_lr, done["value"])[0, 1])
        slope_lr, _ = np.polyfit(log_lr, done["value"], 1)
        p(f"log10(lr) vs {metric} : "
          f"Pearson={rho_lr:+.3f} | pente lin.={slope_lr:+.3f}")
        if has_wd:
            sub = done.dropna(subset=["weight_decay"])
            if len(sub) >= 3:
                log_wd = np.log10(sub["weight_decay"].astype(float))
                rho_wd = float(np.corrcoef(log_wd, sub["value"])[0, 1])
                slope_wd, _ = np.polyfit(log_wd, sub["value"], 1)
                p(f"log10(wd) vs {metric} : "
                  f"Pearson={rho_wd:+.3f} | pente lin.={slope_wd:+.3f}")
        p("")

    # Pruning rate par décade de lr
    if not df.empty:
        d2 = df.dropna(subset=["lr"]).copy()
        d2["decade"] = np.floor(np.log10(d2["lr"].astype(float))).astype(int)
        rate = d2.groupby("decade").apply(
            lambda s: (s["state"] == "PRUNED").mean()
        ).rename("prune_rate")
        n = d2.groupby("decade").size().rename("n")
        prune_df = pd.concat([n, rate], axis=1)
        prune_df.index = [f"1e{int(d)} … 1e{int(d)+1}" for d in prune_df.index]
        p("Taux de pruning par décade de lr")
        p(prune_df.to_string(float_format=lambda x: f"{x:.2f}"))
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

    has_wd = _has_wd(df)

    plot_optim_history(df, opts.out, opts.metric_label)
    plot_param_importance(study, opts.out)
    _marginal_log_x(df, "lr", opts.metric_label,
                    opts.out / "03_marginal_lr.png",
                    "Effet marginal du learning rate", "lr")
    if has_wd:
        _marginal_log_x(df, "weight_decay", opts.metric_label,
                        opts.out / "04_marginal_wd.png",
                        "Effet marginal du weight_decay", "weight_decay")
        plot_scatter_lr_wd(df, opts.out, opts.metric_label)
    plot_learning_curves(study, opts.out, opts.metric_label)
    plot_duration_vs_lr(df, opts.out)
    plot_pruned_vs_complete(df, opts.out)

    text = write_summary(study, df, opts.out, opts.metric_label)
    print(text)
    print(f"→ figures + CSV + summary écrits dans {opts.out}")

    best = study.best_trial
    (opts.out / "best.json").write_text(json.dumps({
        "metric": opts.metric_label,
        "value":  best.value,
        "trial":  best.number,
        "params": best.params,
    }, indent=2))


if __name__ == "__main__":
    main()
