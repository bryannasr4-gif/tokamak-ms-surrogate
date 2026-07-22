"""
phase25b_figures.py -- figures for the Phase-2.5b foundation-hardening pass:
  (1) q95 distribution + q95-vs-m_s (kappa-mediated);
  (2) true high-kappa extrapolation, RMSE_log RESOLVED BY REGIME (the honest, un-confounded view);
  (3) gradient ascent at the converged 80 modes vs 40 modes, by regime with Wilson 95% CIs.
Saves figures/phase25b_{q95,extrap,gradient80}.png. No solves.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, "experiments")
import phase2_data as D

REG_COL = {"marginal": "#d62728", "mid": "#ff7f0e", "stable": "#2ca02c", "very_stable": "#1f77b4"}


def fig_q95():
    df = pd.read_parquet("data/dataset_v1_80q.parquet")
    df["regime"] = df["m_s"].apply(D.regime_of)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
    ax[0].hist(df["q95"], bins=40, color="#4c72b0")
    ax[0].axvspan(3, 10, color="green", alpha=0.08, label="ST physical band 3-10")
    ax[0].set_xlabel("q95 (safety factor at ψ_n=0.95)"); ax[0].set_ylabel("count")
    ax[0].set_title(f"q95: median {df['q95'].median():.2f}, 100% in [3,10]"); ax[0].legend(fontsize=8)
    for name in ("marginal", "mid", "stable", "very_stable"):
        s = df[df.regime == name]
        ax[1].scatter(s["q95"], s["m_s"], s=6, alpha=0.4, color=REG_COL[name], label=name)
    ax[1].set_yscale("log"); ax[1].set_xlabel("q95"); ax[1].set_ylabel("m_s (log)")
    ax[1].set_title("q95 vs m_s: corr(q95,log m_s)=-0.48,\nbut κ-mediated (partial corr |κ = +0.16)")
    ax[1].legend(fontsize=7)
    fig.suptitle("Phase-2.5b q95 feature (dataset_v1_80q): physically sane, κ-mediated", fontsize=11)
    fig.tight_layout(); fig.savefig("figures/phase25b_q95.png", dpi=140, bbox_inches="tight"); plt.close(fig)
    print("saved figures/phase25b_q95.png")


def fig_extrap():
    e = json.load(open("data/phase2_extrap_kappa.json"))
    # regime-resolved RMSE_log: in-dist val vs high-kappa tail (same model) + canonical corner
    regimes = ["marginal", "mid"]
    val = [e["val"]["log"].get(r, {}).get("rmse", np.nan) for r in regimes]
    tail = [e["test_extrap_kappa"]["log"].get(r, {}).get("rmse", np.nan) for r in regimes]
    corner = {"marginal": 0.223, "mid": 0.059}      # canonical A80 corner, computed regime-resolved
    cor = [corner[r] for r in regimes]
    x = np.arange(len(regimes)); w = 0.27
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - w, val, w, color="#9ecae1", label="in-dist val (κ<2.078)")
    ax.bar(x, tail, w, color="#d62728", label="held-out high-κ TAIL (extrapolation)")
    ax.bar(x + w, cor, w, color="#9467bd", label="corner split (canonical, mostly in-hull)")
    for i, (a, b, c) in enumerate(zip(val, tail, cor)):
        for off, v in zip((-w, 0, w), (a, b, c)):
            ax.text(i + off, v + 0.004, f"{v:.3f}", ha="center", fontsize=7)
    ax.set_xticks(x); ax.set_xticklabels([f"{r}\n(tail n={e['test_extrap_kappa']['log'][r]['n']})" for r in regimes])
    ax.set_ylabel("RMSE_log (lower = better)")
    ax.set_title("True high-κ extrapolation, RESOLVED BY REGIME (un-confounds the aggregate)\n"
                 "mid: modest penalty (0.063→0.076); marginal: tail predicted BETTER than in-dist\n"
                 "(high-κ marginals follow the smooth κ→m_s trend); aggregate is regime-mix-confounded")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig("figures/phase25b_extrap.png", dpi=140, bbox_inches="tight"); plt.close(fig)
    print("saved figures/phase25b_extrap.png")


def fig_gradient80():
    g80 = json.load(open("data/phase2_gradcheck2_80.json"))
    g40 = json.load(open("data/phase2_gradcheck2.json"))
    cats = ["marginal", "mid", "design\n(marg+mid)"]
    # 80-mode rates + Wilson CIs
    r80 = [g80["by_regime"]["marginal"]["ascent"], g80["by_regime"]["mid"]["ascent"], g80["design_regime_ascent"]]
    rate80 = [x["rate"] for x in r80]
    lo80 = [x["rate"] - x["wilson95"][0] for x in r80]
    hi80 = [x["wilson95"][1] - x["rate"] for x in r80]
    n80 = [x["n"] for x in r80]
    # 40-mode rates (no CI stored; design = 15/20)
    rate40 = [g40["by_regime"]["marginal"]["ascent_rate"], g40["by_regime"]["mid"]["ascent_rate"], 15 / 20]
    x = np.arange(len(cats)); w = 0.35
    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.bar(x - w / 2, rate40, w, color="#9ecae1", label="40 modes (old labels)")
    ax.bar(x + w / 2, rate80, w, color="#d62728", yerr=[lo80, hi80], capsize=4,
           label="80 modes (converged, Wilson 95%)")
    ax.axhline(0.5, ls="--", color="gray", lw=1, label="chance")
    for i, (n, rt) in enumerate(zip(n80, rate80)):
        ax.text(i + w / 2, rt + 0.03, f"n={n}", ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(cats); ax.set_ylim(0, 1.05)
    ax.set_ylabel("solver-confirmed ascent rate (true m_s↑ along +∇)")
    ax.set_title("Gradient ascent: 40 vs converged 80 modes (held-out bases)\n"
                 "design-regime ascent 75%→55% (Wilson 34–74%, spans chance): usable but WEAKER direction;\n"
                 "the solver-confirmed design loop (Task A) is robust to this — the gradient is a direction, not a Jacobian")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig("figures/phase25b_gradient80.png", dpi=140, bbox_inches="tight"); plt.close(fig)
    print("saved figures/phase25b_gradient80.png")


if __name__ == "__main__":
    for fn in (fig_q95, fig_extrap, fig_gradient80):
        try:
            fn()
        except Exception as e:
            print(f"{fn.__name__} FAILED: {type(e).__name__}: {e}")
