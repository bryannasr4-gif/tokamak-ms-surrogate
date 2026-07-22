"""
device2_design_figures.py -- figures for the Device-C UNCONSTRAINED design comparison (Phase 5 C4).

Panels: (A) per-start best m_s surrogate vs reduce_kappa (paired scatter, diagonal = tie), by regime;
(B) win-rate by regime with Wilson 95% CI (surrogate vs reduce_kappa and vs cma); (C) gain
distribution by method and regime; (D) median true-m_s ascent trajectory vs #solves per method.

  python experiments/device2_design_figures.py --framing zeroshot
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "experiments"))
import phase2_data as D

METHODS = ["surrogate", "reduce_kappa", "cma"]
COL = {"surrogate": "#1f77b4", "reduce_kappa": "#d62728", "cma": "#7f7f7f"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--framing", default="zeroshot")
    args = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    recs = []
    for f in glob.glob(os.path.join(ROOT, "data", "device2_design_results", f"{args.framing}_job*.json")):
        try:
            recs.append(json.load(open(f)))
        except Exception:
            pass
    by_start = {}
    for r in recs:
        s = by_start.setdefault(r["start_id"], dict(ms_start=r["ms_start"], traj={}, best={}, gain={}))
        s["ms_start"] = r["ms_start"]
        s["traj"][r["method"]] = r["traj"]
        s["best"][r["method"]] = r["best_ms"]
        s["gain"][r["method"]] = r["gain"]
    for s in by_start.values():
        s["regime"] = D.regime_of(s["ms_start"])
    starts = list(by_start.values())
    summ = json.load(open(os.path.join(ROOT, "data", f"device2_design_{args.framing}_summary.json")))

    fig, ax = plt.subplots(2, 2, figsize=(13, 11))
    regcol = {"marginal": "#d62728", "mid": "#1f77b4", "stable": "#2ca02c", "very_stable": "#9467bd"}

    # (A) paired scatter surrogate vs reduce_kappa
    a = ax[0, 0]
    for s in starts:
        if "surrogate" in s["best"] and "reduce_kappa" in s["best"]:
            a.scatter(s["best"]["reduce_kappa"], s["best"]["surrogate"], s=44,
                      color=regcol.get(s["regime"], "k"), edgecolor="k", lw=0.4, zorder=3)
    lim = [0, max([max(s["best"].values()) for s in starts if s["best"]] + [1.2]) * 1.05]
    a.plot(lim, lim, "k--", lw=1, zorder=1)
    a.set_xlim(lim); a.set_ylim(lim)
    a.set_xlabel("reduce_kappa  best true m_s"); a.set_ylabel("surrogate  best true m_s")
    a.set_title("(A) Per-start final m_s: surrogate vs reduce_kappa\n(above diagonal = surrogate wins)")
    for reg, c in regcol.items():
        if any(s["regime"] == reg for s in starts):
            a.scatter([], [], color=c, label=reg)
    a.legend(fontsize=8, title="start regime")

    # (B) win-rate by regime with Wilson CI
    b = ax[0, 1]
    cohort = summ.get("ALL", {})
    regs = [r for r in ["marginal", "mid", "stable", "POOLED"] if r in cohort]
    x = np.arange(len(regs)); w = 0.35
    for k, (comp, off, c, lab) in enumerate([("surrogate_vs_reduce_kappa", -w / 2, "#1f77b4", "vs reduce_kappa"),
                                             ("surrogate_vs_cma", w / 2, "#7f7f7f", "vs cma")]):
        wr = [cohort[r][comp]["win_rate"] for r in regs]
        lo = [cohort[r][comp]["win_rate"] - cohort[r][comp]["wilson95"][0] for r in regs]
        hi = [cohort[r][comp]["wilson95"][1] - cohort[r][comp]["win_rate"] for r in regs]
        b.bar(x + off, wr, w, color=c, label=lab, alpha=0.85, yerr=[lo, hi], capsize=4)
    b.axhline(0.5, color="k", ls="--", lw=1)
    b.set_xticks(x); b.set_xticklabels([f"{r}\nn={cohort[r]['surrogate_vs_reduce_kappa']['n']}" for r in regs])
    b.set_ylim(0, 1.05); b.set_ylabel("surrogate win-rate"); b.legend(fontsize=8)
    b.set_title("(B) Surrogate win-rate by regime (Wilson 95% CI)\n0.5 = tie")

    # (C) gain distribution by method x regime
    c = ax[1, 0]
    order = ["marginal", "mid", "stable"]
    present = [r for r in order if any(s["regime"] == r for s in starts)]
    pos = 0; ticks = []; lab = []
    for reg in present:
        for mi, m in enumerate(METHODS):
            g = [s["gain"][m] for s in starts if s["regime"] == reg and m in s["gain"]]
            if g:
                c.boxplot(g, positions=[pos], widths=0.7, patch_artist=True,
                          boxprops=dict(facecolor=COL[m], alpha=0.7), medianprops=dict(color="k"))
            pos += 1
        ticks.append(pos - 2); lab.append(reg); pos += 1
    c.set_xticks(ticks); c.set_xticklabels(lab)
    c.axhline(0, color="k", lw=0.8)
    c.set_ylabel("gain in true m_s (best - start)")
    c.set_title("(C) m_s gain by method x regime\n(blue=surrogate, red=reduce_kappa, grey=cma)")

    # (D) median trajectory vs solves
    d = ax[1, 1]
    for m in METHODS:
        curves = []
        for s in starts:
            if m in s["traj"]:
                t = np.array(s["traj"][m]); curves.append((t[:, 0], t[:, 1]))
        if not curves:
            continue
        maxn = int(max(c0[-1] for c0, _ in curves))
        grid = np.arange(1, maxn + 1)
        M_ = []
        for ns, ms in curves:
            M_.append(np.interp(grid, ns, ms))
        med = np.median(np.array(M_), axis=0)
        d.plot(grid, med, color=COL[m], lw=2, label=m)
    d.set_xlabel("# true 80-mode solves"); d.set_ylabel("median best true m_s")
    d.set_title("(D) Median true-m_s ascent vs solver budget"); d.legend(fontsize=9)

    plt.suptitle(f"Device-C UNCONSTRAINED design comparison ({args.framing}) — learned m_s vs reduce-kappa vs gradient-free",
                 fontsize=12, y=0.997)
    plt.tight_layout(rect=[0, 0, 1, 0.985])
    out = os.path.join(ROOT, "figures", f"phase5_design_{args.framing}.png")
    plt.savefig(out, dpi=130)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
