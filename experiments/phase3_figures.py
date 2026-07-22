"""
phase3_figures.py -- Phase-3 efficiency / amortization figures (the headline story).

figures/phase3_efficiency.png (4 panels), leading with the UNCONFOUNDED metric (true-solves-to-target
+ reach-rate; the final-margin "gain" is confounded by CMA population-overshoot of the stop-at-target
rule, so it is NOT the headline):
  A. cumulative reach-rate of m*=1.0 vs number of true solves, per method (MARGINAL starts -- the
     discriminating m_s->0 regime): the surrogate design loop stabilizes plasmas in far fewer solves.
  B. median true-solves-to-target per method (among reachers), resolved by regime; reach count annotated.
  C. reach-rate of m*=1.0 per method with Wilson 95% CIs (pooled / marginal / mid).
  D. paired true-solves-to-target, surrogate vs best gradient-free (censored at budget+1); points BELOW
     the diagonal = surrogate reaches the target in fewer expensive solves.
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
GRADFREE = ["cma", "random", "nelder"]
COL = dict(surrogate="#d1495b", heuristic="#edae49", cma="#00798c", random="#66a182", nelder="#2e4057")


def main():
    recs = json.load(open("data/phase3_results.json"))["recs"]
    S = json.load(open("data/phase3_summary.json"))
    setup = json.load(open("data/phase3_setup.json"))
    budget = setup["budget"]; methods = setup["methods"]; CENS = budget + 1
    regime_of = {r["start_i"]: r["regime"] for r in recs}
    starts = sorted(set(r["start_i"] for r in recs))

    def rec_of(si, m):
        rs = [r for r in recs if r["start_i"] == si and r["method"] == m]
        return rs[0] if rs else None

    def s2t(si, m):
        r = rec_of(si, m); return None if not r else r["solves_to_target"].get("1.0")

    fig, ax = plt.subplots(2, 2, figsize=(13.5, 10.5))

    # --- A. cumulative reach-rate vs true solves (MARGINAL starts) ---
    a = ax[0, 0]
    marg = [si for si in starts if regime_of[si] == "marginal"]
    bmax = budget
    # overlay the FAIR-CMA control (pop6, budget 60, graded penalty) if present -- shows even a fair,
    # well-resourced CMA reaches the targets only SLOWLY (the efficiency gap is not a starvation artifact)
    fair = None
    if os.path.exists("data/phase3_cmafair_results.json"):
        fair = json.load(open("data/phase3_cmafair_results.json"))["recs"]
        bmax = max(bmax, max(x["budget"] for x in fair))
    ns = np.arange(1, bmax + 1)
    for m in methods:
        frac = [sum(1 for si in marg if (s2t(si, m) is not None and s2t(si, m) <= n)) / len(marg)
                for n in np.arange(1, budget + 1)]
        a.step(np.arange(1, budget + 1), frac, where="post", lw=2.4, color=COL[m], label=m)
    if fair:
        for tag, ls in [("cma_pop6_b60", "--")]:
            sub = [x for x in fair if x["tag"] == tag]
            nb = max(x["budget"] for x in sub)
            frac = [sum(1 for x in sub if (x["solves_to_target"]["1.0"] is not None and x["solves_to_target"]["1.0"] <= n)) / len(sub)
                    for n in np.arange(1, nb + 1)]
            a.step(np.arange(1, nb + 1), frac, where="post", lw=2.0, ls=ls, color="#5c4d7d",
                   label="fair-CMA (pop6, 2×budget)")
    a.axvline(budget, ls=":", color="0.6", lw=1)
    a.set_xlabel("true 80-mode solves (expensive)"); a.set_ylabel("fraction of marginal starts stabilized (m_s≥1.0)")
    a.set_title("A. amortization: stabilize marginal plasmas in ~4× fewer expensive solves\n"
                "(fair-CMA reaches the same 9/10 but needs ~2× budget & median 26 vs 6 solves)")
    a.legend(fontsize=7.5, loc="lower right"); a.grid(alpha=0.3); a.set_ylim(-0.03, 1.03)

    # --- B. median solves-to-target per method, by regime ---
    b = ax[0, 1]
    labels = ["marginal", "mid"]
    x = np.arange(len(labels)); w = 0.15
    for i, m in enumerate(methods):
        meds, anns = [], []
        for lab in labels:
            sub = [si for si in starts if regime_of[si] == lab]
            sv = [s2t(si, m) for si in sub]
            reach = [v for v in sv if v is not None]
            meds.append(np.median(reach) if reach else CENS)
            anns.append(f"{len(reach)}/{len(sub)}")
        bars = b.bar(x + (i - 2) * w, meds, w, color=COL[m], label=m)
        for xi, (bar, an) in enumerate(zip(bars, anns)):
            b.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.4, an,
                   ha="center", va="bottom", fontsize=6, rotation=90)
    b.axhline(budget, ls=":", color="0.5", lw=1, label=f"budget={budget}")
    b.set_xticks(x); b.set_xticklabels([f"{l}\nstart" for l in labels])
    b.set_ylabel("median true-solves-to-target (reachers)")
    b.set_title("B. expensive solves to reach m*=1.0 (annot: reach count)")
    b.legend(fontsize=7, ncol=2); b.grid(alpha=0.3, axis="y")

    # --- C. reach-rate with Wilson CIs ---
    c = ax[1, 0]
    show = ["surrogate", "heuristic", "best_gradfree"]
    x = np.arange(3); w = 0.26
    for j, lab in enumerate(["pooled", "marginal", "mid"]):
        e = S["by_regime"].get(lab, {}); rr = e.get("reach_rate", {})
        fr = np.array([rr.get(m, {}).get("frac", np.nan) for m in show])
        lo = [rr.get(m, {}).get("wilson95", [np.nan, np.nan])[0] for m in show]
        hi = [rr.get(m, {}).get("wilson95", [np.nan, np.nan])[1] for m in show]
        c.bar(x + (j - 1) * w, fr, w, yerr=[fr - np.array(lo), np.array(hi) - fr], capsize=3,
              color=["#444", "#999", "#ccc"][j], label=lab, edgecolor="k", lw=0.5)
    c.set_xticks(x); c.set_xticklabels(["surrogate", "reduce-κ\nheuristic", "best\ngradient-free"])
    c.set_ylabel("reach-rate of m*=1.0 (Wilson 95%)")
    c.set_title("C. reach-rate by method (pooled / marginal / mid)")
    c.legend(fontsize=8); c.grid(alpha=0.3, axis="y"); c.set_ylim(0, 1.08)

    # --- D. paired solves-to-target scatter: surrogate vs best gradient-free (censored) ---
    d = ax[1, 1]
    for lab, mk in [("marginal", "o"), ("mid", "s")]:
        sub = [si for si in starts if regime_of[si] == lab]
        sx, sy = [], []
        for si in sub:
            sv = s2t(si, "surrogate")
            gf = [s2t(si, g) for g in GRADFREE]; gf = [v for v in gf if v is not None]
            sy.append(sv if sv is not None else CENS)
            sx.append(min(gf) if gf else CENS)
        jit = (np.random.RandomState(0).rand(len(sx)) - 0.5) * 0.0  # no jitter (determinism)
        d.scatter(np.array(sx), np.array(sy), marker=mk, s=70, color=COL["surrogate"],
                  edgecolor="k", lw=0.5, alpha=0.85, label=f"{lab} start")
    d.plot([0, CENS], [0, CENS], "k--", lw=1, label="parity")
    d.axvline(CENS, ls=":", color="0.5", lw=1); d.axhline(CENS, ls=":", color="0.5", lw=1)
    d.text(CENS, 1.5, "  gradient-free\n  never reached", fontsize=6.5, va="bottom", color="0.4")
    am = S["amortization"]["pooled"]
    d.set_xlabel("best gradient-free solves-to-target (censored)")
    d.set_ylabel("surrogate solves-to-target")
    d.set_title(f"D. solves-to-target: surrogate faster in {am['paired_solves_surrogate_faster']}/{am['n']} "
                f"(Wilcoxon p={am['paired_solves_wilcoxon_p']:.4f})\nbelow parity = surrogate cheaper")
    d.legend(fontsize=8); d.grid(alpha=0.3); d.set_xlim(0, CENS + 1); d.set_ylim(0, CENS + 1)

    fig.text(0.5, 0.005,
             "'best gradient-free' = per-start oracle best-of-3 (CMA/random/Nelder-Mead) envelope; CMA alone reaches "
             "14/20 at median 15 solves.  Solves-to-target = SERIAL expensive-solve count; under parallel CMA-population "
             "evaluation the pooled per-query edge is within noise (5/20).  The reach-rate gap (A,C) is framing-invariant.",
             ha="center", fontsize=6.8, color="0.3")
    plt.tight_layout(rect=[0, 0.02, 1, 1])
    plt.savefig("figures/phase3_efficiency.png", dpi=140)
    print("saved figures/phase3_efficiency.png")


if __name__ == "__main__":
    main()
