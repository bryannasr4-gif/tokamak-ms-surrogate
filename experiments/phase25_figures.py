"""
phase25_figures.py -- Phase-2.5 hardening figures.
  figures/phase25_crossmachine.png   A-surrogate -> Machine B transfer (Spearman 0.99; affine rescale)
  figures/phase25_kappa.png          kappa-constrained 'beats heuristics' (median gain by method)
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
import phase2_model as M


def _load(p):
    return json.load(open(p)) if os.path.exists(p) else None


def fig_crossmachine():
    cm = _load("data/phase25_crossmachine.json")
    if not cm or not os.path.exists("data/dataset_v2_B.parquet"):
        return
    a = pd.read_parquet("data/dataset_v1_80.parquet")
    b = pd.read_parquet("data/dataset_v2_B.parquet")
    mrg = a[["idx", "m_s"] + D.SHAPE_FEATURES].rename(columns={"m_s": "m_s_A"}).merge(
        b[["idx", "m_s"]].rename(columns={"m_s": "m_s_B"}), on="idx")
    models, _ = M.load_ensemble()  # A80 surrogate
    predA = M.ensemble_predict(models, mrg[D.SHAPE_FEATURES].values.astype(float))["mean"][:, 0]
    logB = np.log(mrg["m_s_B"].values)
    a_, b_ = cm["A_to_B_affine_coef"]
    fig, ax = plt.subplots(1, 2, figsize=(12, 5.2))
    ax[0].scatter(predA, logB, s=8, alpha=0.4, c="#1f77b4")
    xs = np.array([predA.min(), predA.max()])
    ax[0].plot(xs, a_ * xs + b_, "r-", lw=2,
               label=f"affine (held-out R²={cm['A_to_B_affine_logR2_val']:.2f} val/{cm['A_to_B_affine_logR2_test']:.2f} corner)")
    ax[0].plot(xs, xs, "k--", lw=1, label="identity (naive transfer)")
    ax[0].set_xlabel("Machine-A surrogate log m_s (on B shapes)"); ax[0].set_ylabel("true Machine-B log m_s")
    ax[0].set_title(f"surrogate→B transfer (Spearman {cm['A_to_B_spearman']:.3f}); but the trivial\n"
                    f"TRUE-LABEL rescale does better (Spearman {cm['truelabel_spearman']:.3f}) — "
                    f"surrogate adds nothing")
    ax[0].legend(fontsize=8)
    # m_s_A vs m_s_B directly
    ax[1].scatter(mrg["m_s_A"], mrg["m_s_B"], s=8, alpha=0.4, c="#2ca02c")
    lim = [min(mrg.m_s_A.min(), mrg.m_s_B.min()), max(mrg.m_s_A.max(), mrg.m_s_B.max())]
    ax[1].plot(lim, lim, "k--", lw=1)
    ax[1].set_xscale("log"); ax[1].set_yscale("log")
    ax[1].set_xlabel("Machine A m_s (138 passives)"); ax[1].set_ylabel("Machine B m_s (56 passives)")
    ax[1].set_title(f"same shapes, different conducting structure\nmedian m_s_B/m_s_A={np.median(mrg.m_s_B/mrg.m_s_A):.2f}")
    fig.suptitle("Phase 2.5 — surrogate generalises across conducting structures (up to a rescale)", fontsize=12)
    fig.tight_layout()
    fig.savefig("figures/phase25_crossmachine.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("saved figures/phase25_crossmachine.png")


def fig_kappa():
    s = _load("data/phase25_kappa_summary.json")
    r = _load("data/phase25_kappa_results.json")
    if not s or not r:
        return
    # NEW analyzer schema (Phase-2.5b): per_method_median[m] = {pooled,marginal,mid};
    # beats-rates + two-sided p live under by_regime[pooled/marginal/mid].vs_{heuristic,gradfree}.
    pm = s["per_method_median"]
    def pmed(m):
        v = pm.get(m, 0)
        return v["pooled"] if isinstance(v, dict) else v
    pool = s["by_regime"]["pooled"]
    recs0 = r["recs"]
    heur_methods = sorted({x["method"] for x in recs0 if x["method"].startswith("heuristic")})
    best_heur = max(heur_methods, key=lambda m: pmed(m)) if heur_methods else None
    order = ["surrogate", "random", "cma"] + ([best_heur] if best_heur else [])
    labels = ["surrogate\n(learned ∇m_s)", "random", "CMA-ES",
              f"best single lever\n({best_heur.split(':')[1] if best_heur else ''})"]
    cols = ["#2ca02c", "#d62728", "#ff7f0e", "#9467bd"]
    vals = [pmed(m) for m in order]
    fig, ax = plt.subplots(1, 2, figsize=(12, 5.2))
    ax[0].bar(range(len(order)), vals, color=cols)
    ax[0].set_xticks(range(len(order))); ax[0].set_xticklabels(labels, fontsize=8, rotation=15, ha="right")
    ax[0].set_ylabel("median true m_s gain (κ held fixed ±0.04)")
    vh = pool["vs_heuristic"]; vg = pool["vs_gradfree"]
    n = s["n_starts_total"]
    ax[0].set_title(f"κ-CONSTRAINED design, all methods EQUAL budget 18 (n={n}: 10 marginal + 10 mid)\n"
                    f"vs gradient-free {vg['wins']}/{vg['n']} (Wilcoxon p={vg['wilcoxon_p_two_sided']:.4f}, significant)\n"
                    f"vs best fair single-lever heuristic {vh['wins']}/{vh['n']} "
                    f"(Wilcoxon p={vh['wilcoxon_p_two_sided']:.3f}, suggestive/NS; n.s. within regimes)")
    # per-start surrogate vs best-heuristic vs best-gradfree
    recs = r["recs"]
    starts = sorted(set(x["start_i"] for x in recs))
    HEUR = heur_methods; GF = ["random", "cma"]
    def g(si, m):
        v = [x["gain"] for x in recs if x["start_i"] == si and x["method"] == m]
        return max(v) if v else np.nan
    sur = [g(si, "surrogate") for si in starts]
    bh = [np.nanmax([g(si, h) for h in HEUR]) for si in starts]
    bg = [np.nanmax([g(si, x) for x in GF]) for si in starts]
    x = np.arange(len(starts)); w = 0.27
    ax[1].bar(x - w, sur, w, color="#2ca02c", label="surrogate")
    ax[1].bar(x, bg, w, color="#ff7f0e", label="best gradient-free")
    ax[1].bar(x + w, bh, w, color="#9467bd", label="best heuristic")
    nwin = sum(1 for a, b in zip(sur, bh) if a > b)
    ax[1].set_xlabel("start (marginal/mid)"); ax[1].set_ylabel("true m_s gain")
    ax[1].set_title(f"per start: surrogate beats best fair heuristic in {nwin}/{len(starts)}"); ax[1].legend(fontsize=8)
    fig.suptitle("Phase 2.5b — with κ fixed (n=20), the learned gradient robustly beats gradient-free search;\n"
                 "its edge over the best fair single-lever heuristic is positive but within noise (Wilcoxon p=0.064)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig("figures/phase25_kappa.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("saved figures/phase25_kappa.png")


if __name__ == "__main__":
    for fn in (fig_crossmachine, fig_kappa):
        try:
            fn()
        except Exception as e:
            print(f"{fn.__name__} skipped: {type(e).__name__}: {e}")
