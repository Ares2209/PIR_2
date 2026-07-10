"""Diagnostic: measure alignment between the ray-cast `is_occluded` feature
and the renderer-painted class 6 label.

Hypothesis: large GAT confusions between class 6 (occluded) and classes 0/1/2
are driven by a noisy feature, not by an architectural limitation. Run this
once and compare `agree` to the test accuracy on class 6.

Usage:
    python check_occlusion_alignment.py
"""

from __future__ import annotations

import numpy as np

from config import FEAT_KEYS, NODE_STATS
from dataio import load_sharded_dataset


COL_IS_OCC      = FEAT_KEYS.index("is_occluded")
COL_FIRST_FRAC  = FEAT_KEYS.index("first_obstacle_frac")
COL_LOG_N_INTER = FEAT_KEYS.index("log_n_intersections")

OCC_CLASS = 6


def _unz(col: np.ndarray, key: str) -> np.ndarray:
    """Invert the z-score applied at dataset-build time."""
    mean, std = NODE_STATS[key]
    return col * std + mean


def main() -> None:
    graphs, _ = load_sharded_dataset()

    total = 0
    agree = 0
    miss_occ = 0      # label = occluded, ray says clear
    false_occ = 0     # ray says blocked, label != occluded
    # Per-true-class breakdown of where the ray says "blocked" (useful to see
    # which visible classes get spuriously flagged as occluded by the feature).
    false_occ_by_class = np.zeros(7, dtype=np.int64)
    # When the label is occluded, distribution of first_obstacle_frac / n_inter.
    fof_when_occ_label: list[float] = []
    nin_when_occ_label: list[float] = []
    fof_when_visible:    list[float] = []

    for g in graphs:
        x = g.x.numpy()
        y = g.y.numpy()

        is_occ_raw = _unz(x[:, COL_IS_OCC],     "is_occluded")
        fof_raw    = _unz(x[:, COL_FIRST_FRAC], "first_obstacle_frac")
        nin_raw    = _unz(x[:, COL_LOG_N_INTER], "log_n_intersections")

        # Binarise: built as exactly 0/1, so 0.5 is a safe threshold post-unz.
        occ_feat  = (is_occ_raw > 0.5).astype(np.int8)
        occ_label = (y == OCC_CLASS).astype(np.int8)

        valid = y >= 0
        occ_feat  = occ_feat[valid]
        occ_label = occ_label[valid]
        y_valid   = y[valid]
        fof_v     = fof_raw[valid]
        nin_v     = nin_raw[valid]

        total     += occ_feat.size
        agree     += int((occ_feat == occ_label).sum())
        miss_occ  += int(((occ_feat == 0) & (occ_label == 1)).sum())
        false_occ += int(((occ_feat == 1) & (occ_label == 0)).sum())

        fo_mask = (occ_feat == 1) & (occ_label == 0)
        if fo_mask.any():
            for c in range(7):
                false_occ_by_class[c] += int(((y_valid == c) & fo_mask).sum())

        occ_lbl_mask = occ_label == 1
        if occ_lbl_mask.any():
            fof_when_occ_label.extend(fof_v[occ_lbl_mask].tolist())
            nin_when_occ_label.extend(nin_v[occ_lbl_mask].tolist())
        if (~occ_lbl_mask).any():
            fof_when_visible.extend(fof_v[~occ_lbl_mask].tolist())

    print(f"\n=== is_occluded feature vs class-6 label ===")
    print(f"total nodes      : {total:,}")
    print(f"agree            : {agree:,}  ({100 * agree / total:.3f}%)")
    print(f"miss_occ         : {miss_occ:,}  ({100 * miss_occ / total:.3f}%)"
          "   (label=occluded, ray says clear)")
    print(f"false_occ        : {false_occ:,}  ({100 * false_occ / total:.3f}%)"
          "   (ray says blocked, label!=occluded)")

    print(f"\nfalse_occ broken down by true class:")
    names = ["violet", "blue", "yellow", "orange", "red", "dark_red", "occluded"]
    for c, n in enumerate(names):
        if c == OCC_CLASS:
            continue
        pct = 100 * false_occ_by_class[c] / max(1, false_occ)
        print(f"  class {c} ({n:<9}) : {int(false_occ_by_class[c]):>12,d}  ({pct:5.2f}%)")

    if fof_when_occ_label:
        fof_occ = np.array(fof_when_occ_label)
        nin_occ = np.array(nin_when_occ_label)
        fof_vis = np.array(fof_when_visible)
        print(f"\nfirst_obstacle_frac distribution (raw, 1.0 = clear LOS):")
        print(f"  when label=occluded  : "
              f"mean={fof_occ.mean():.3f}  median={np.median(fof_occ):.3f}  "
              f"frac_at_1={float((fof_occ >= 0.999).mean()):.3%}")
        print(f"  when label=visible   : "
              f"mean={fof_vis.mean():.3f}  median={np.median(fof_vis):.3f}  "
              f"frac_at_1={float((fof_vis >= 0.999).mean()):.3%}")
        print(f"\nlog_n_intersections when label=occluded:")
        print(f"  mean={nin_occ.mean():.3f}  median={np.median(nin_occ):.3f}  "
              f"frac_zero={float((nin_occ < 1e-3).mean()):.3%}")

    print()
    print("Interpretation:")
    print("  agree > 98%  : feature is clean, GAT under-uses it (architecture issue)")
    print("  agree 80-95% : feature is noisy, drives the 0/1/2 ↔ 6 confusions")
    print("  miss_occ high: ray from centroid misses obstacles the renderer sees")
    print("  false_occ high: ray says blocked but renderer paints visible (label bleed)")


if __name__ == "__main__":
    main()
