"""
phase2_figures.py -- all Phase-2 figures, built from the saved JSON/parquet artifacts.
Each panel is guarded so the script runs with whatever artifacts exist so far.

  figures/phase2_accuracy.png      predicted vs true m_s (val/test_extrap) + residuals, by regime
  figures/phase2_gradient.png      gradient verification: directional scatter + per-base cosine
  figures/phase2_calibration.png   reliability diagram (raw vs calibrated) + width vs m_s
  figures/phase2_modes.png         mode-convergence curves + 40->138 drift by regime
  figures/phase2_dimensionality.png  headline: true-solves-to-target vs dimension, by method
"""
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, "experiments")
import phase2_data as D

REG_COL = {"marginal": "#d62728", "mid": "#ff7f0e", "stable": "#2ca02c", "very_stable": "#1f77b4"}


def _load(p):
    return json.load(open(p)) if os.path.exists(p) else None


def fig_accuracy():
    import pandas as pd
    if not os.path.exists("data/phase2_predictions.parquet"):
        return
    df = pd.read_parquet("data/phase2_predictions.parquet")
    met = _load("data/phase2_train_metrics.json")
    fig, ax = plt.subplots(1, 2, figsize=(12, 5.4))
    for k, split in enumerate(["val", "test_extrap"]):
        dd = df[df.split == split]
        for name in REG_COL:
            m = dd.regime == name
            if m.sum():
                ax[k].scatter(dd.m_s[m], dd.m_s_pred[m], s=14, c=REG_COL[name], alpha=0.6, label=name)
        lim = [min(dd.m_s.min(), dd.m_s_pred.min()), max(dd.m_s.max(), dd.m_s_pred.max())]
        ax[k].plot(lim, lim, "k--", lw=1)
        ax[k].set_xscale("log"); ax[k].set_yscale("log")
        r2 = met[split]["accuracy"]["all"]["r2"] if met else float("nan")
        r2log = met[split]["accuracy_log"]["all_r2_log"] if met else float("nan")
        ax[k].set_title(f"{split}: R²={r2:.3f} (orig), {r2log:.3f} (log m_s)")
        ax[k].set_xlabel("true m_s"); ax[k].set_ylabel("predicted m_s"); ax[k].legend(fontsize=8)
    fig.suptitle("Phase 2 — differentiable m_s(shape) surrogate accuracy (split honored)", fontsize=12)
    fig.tight_layout()
    fig.savefig("figures/phase2_accuracy.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("saved figures/phase2_accuracy.png")


def fig_gradient():
    g2 = _load("data/phase2_gradcheck2.json")          # in-distribution ascent (headline)
    g = _load("data/phase2_gradcheck.json")            # per-axis cosine (secondary diagnostic)
    if not g2 and not g:
        return
    fig, ax = plt.subplots(1, 2, figsize=(12, 5.2))
    # LEFT: decision-relevant ascent test (true m_s rises along the surrogate gradient), by regime
    if g2:
        names = [n for n, _, _ in D.REGIMES if n in g2["by_regime"]]
        asc = [g2["by_regime"][n]["ascent_rate"] * 100 for n in names]
        br = [g2["by_regime"][n]["beats_random_rate"] * 100 for n in names]
        xx = np.arange(len(names)); w = 0.38
        ax[0].bar(xx - w / 2, asc, w, color="#2ca02c", label="ascent (m_s↑ along +∇)")
        ax[0].bar(xx + w / 2, br, w, color="#1f77b4", label="beats random direction")
        ax[0].axhline(50, color="k", ls=":", lw=1)
        ax[0].set_xticks(xx); ax[0].set_xticklabels(names, fontsize=8)
        ax[0].set_ylabel("% of held-out bases")
        ax[0].set_title(f"in-distribution gradient ascent vs true solver\n"
                        f"overall ascent {g2['ascent_rate']*100:.0f}%, beats-random {g2['beats_random_rate']*100:.0f}%")
        ax[0].legend(fontsize=8)
    # RIGHT: per-axis cosine diagnostic (harsh; diluted by weak controls) + note
    if g:
        cos_b = [r["cos_B"] for r in g["per_base"]]
        cos_d = [r["cos_dir"] for r in g["per_base"]]
        ax[1].hist(cos_d, bins=np.linspace(-1, 1, 21), alpha=0.6, label="shape-dir (descriptor FD)")
        ax[1].hist(cos_b, bins=np.linspace(-1, 1, 21), alpha=0.6, label="control-grad (composed)")
        ax[1].axvline(0.0, color="k", ls=":", lw=1)
        ax[1].set_title("per-axis gradient cosine (harsh diagnostic)\n"
                        "diluted by weak/redundant controls — see ledger")
        ax[1].set_xlabel("cosine(surrogate, true)"); ax[1].set_ylabel("# bases"); ax[1].legend(fontsize=8)
    fig.suptitle("Phase 2 — surrogate gradient: usable ascent direction, solver-confirmed", fontsize=12)
    fig.tight_layout()
    fig.savefig("figures/phase2_gradient.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("saved figures/phase2_gradient.png")


def fig_calibration():
    import pandas as pd
    c = _load("data/phase2_calibration.json")
    if not c or not os.path.exists("data/phase2_predictions.parquet"):
        return
    df = pd.read_parquet("data/phase2_predictions.parquet")
    fig, ax = plt.subplots(1, 2, figsize=(12, 5.2))
    for split, ls in [("val", "-"), ("test_extrap", "--")]:
        b = c[split]
        ax[0].plot(b["reliability_grid"], b["reliability_raw"], ls, color="#d62728", alpha=0.8,
                   label=f"{split} raw")
        ax[0].plot(b["reliability_grid"], b["reliability_cal"], ls, color="#2ca02c", alpha=0.8,
                   label=f"{split} calibrated")
    ax[0].plot([0, 1], [0, 1], "k:", lw=1)
    ax[0].set_xlabel("nominal central coverage"); ax[0].set_ylabel("empirical coverage")
    ax[0].set_title(f"reliability (global recal scale s={c['recal_scale_global']:.2f})")
    ax[0].legend(fontsize=8)
    for name in REG_COL:
        m = df.regime == name
        if m.sum():
            ax[1].scatter(df.m_s[m], df.tot_std[m] * c["recal_scale_global"], s=12,
                          c=REG_COL[name], alpha=0.5, label=name)
    ax[1].set_xscale("log")
    ax[1].set_xlabel("true m_s"); ax[1].set_ylabel("calibrated predictive σ (log m_s)")
    ax[1].set_title("predictive width widens toward m_s→0 and the sparse stable tail")
    ax[1].legend(fontsize=8)
    fig.suptitle("Phase 2 — honest calibrated uncertainty (post-hoc recalibrated)", fontsize=12)
    fig.tight_layout()
    fig.savefig("figures/phase2_calibration.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("saved figures/phase2_calibration.png")


def fig_modes():
    m = _load("data/phase2_modes.json")
    s = _load("data/phase2_modes_summary.json")
    if not m:
        return
    modes = m["modes"]
    fig, ax = plt.subplots(1, 2, figsize=(12, 5.2))
    # curves: a few shapes per regime, normalized to their 138 value
    shown = {k: 0 for k in REG_COL}
    for r in m["recs"]:
        reg = r["regime"]
        if shown.get(reg, 99) >= 5:
            continue
        ys = [r["ms"].get(str(md)) for md in modes]
        if any(y is None or not np.isfinite(y) or y <= 0 for y in ys):
            continue
        ref = ys[-1]
        ax[0].plot(modes, [y / ref for y in ys], "-o", color=REG_COL[reg], alpha=0.5, ms=3)
        shown[reg] += 1
    ax[0].axhline(1.0, color="k", ls=":", lw=1)
    ax[0].set_xlabel("retained passive modes"); ax[0].set_ylabel("m_s / m_s(138 modes)")
    ax[0].set_title("mode convergence (138 = all passive structures)")
    for name in REG_COL:
        ax[0].plot([], [], "-o", color=REG_COL[name], label=name)
    ax[0].legend(fontsize=8)
    if s:
        names = [n for n, _, _ in D.REGIMES if n in s["by_regime"]]
        drift = [s["by_regime"][n]["drift_40_to_138"]["median"] * 100 for n in names]
        ax[1].bar(names, drift, color=[REG_COL[n] for n in names])
        ax[1].axhline(0, color="k", lw=0.8)
        ax[1].set_ylabel("median 40→138 m_s drift [%]")
        ax[1].set_title("shipped (40-mode) labels are systematically low\n(largest near m_s→0)")
        for i, v in enumerate(drift):
            ax[1].text(i, v, f"{v:+.0f}%", ha="center", va="bottom", fontsize=9)
    fig.suptitle("Phase 2 — mode-convergence study (component 0)", fontsize=12)
    fig.tight_layout()
    fig.savefig("figures/phase2_modes.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("saved figures/phase2_modes.png")


def fig_dim():
    s = _load("data/phase2_dim_summary.json")
    if not s:
        return
    methods = s["methods"]; d_list = s["d_list"]
    mcol = {"surrogate": "#2ca02c", "heuristic": "#9467bd", "cma": "#ff7f0e", "random": "#d62728"}
    fig, ax = plt.subplots(1, 2, figsize=(12, 5.2))
    for meth in methods:
        xs, ys, lo, hi, succ = [], [], [], [], []
        for d in d_list:
            g = s["grid"][str(d)].get(meth)
            if g:
                xs.append(d); ys.append(g["solves_median"])
                lo.append(g["solves_iqr"][0]); hi.append(g["solves_iqr"][1])
                succ.append(g["success_rate"] * 100)
        if xs:
            ax[0].plot(xs, ys, "-o", color=mcol.get(meth), label=meth)
            ax[0].fill_between(xs, lo, hi, color=mcol.get(meth), alpha=0.15)
            ax[1].plot(xs, succ, "-o", color=mcol.get(meth), label=meth)
    ax[0].axhline(s["budget"], color="k", ls=":", lw=1, label=f"budget={s['budget']}")
    ax[0].set_xlabel("true design dimension d (top-d control PCs)")
    ax[0].set_ylabel("true solves to reach m_s≥%.1f (median, IQR)" % s["target"])
    ax[0].set_title(f"true solves to target (local navigation)\n"
                    f"(effective control dim ≈ {s['effective_dim']:.1f})")
    ax[0].legend(fontsize=8)
    # right panel: robust headline = best true m_s reached at the fixed budget vs d
    for meth in methods:
        xs, ys, lo, hi = [], [], [], []
        for d in d_list:
            g = s["grid"][str(d)].get(meth)
            if g:
                xs.append(d); ys.append(g["best_median"])
                lo.append(g["best_iqr"][0]); hi.append(g["best_iqr"][1])
        if xs:
            ax[1].plot(xs, ys, "-o", color=mcol.get(meth), label=meth)
            ax[1].fill_between(xs, lo, hi, color=mcol.get(meth), alpha=0.15)
    ax[1].axhline(s["target"], color="k", ls=":", lw=1, label=f"target={s['target']}")
    ax[1].set_xlabel("true design dimension d")
    ax[1].set_ylabel(f"best true m_s at budget={s['budget']} (median, IQR)")
    ax[1].set_title("best m_s reached at fixed budget\n(gradient holds up as d grows; search degrades)")
    ax[1].legend(fontsize=8)
    fig.suptitle("Phase 2 — dimensionality experiment (control/PCA space, true-solver confirmed)", fontsize=12)
    fig.tight_layout()
    fig.savefig("figures/phase2_dimensionality.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("saved figures/phase2_dimensionality.png")


if __name__ == "__main__":
    for fn in (fig_accuracy, fig_gradient, fig_calibration, fig_modes, fig_dim):
        try:
            fn()
        except Exception as e:
            print(f"{fn.__name__} skipped: {type(e).__name__}: {e}")
