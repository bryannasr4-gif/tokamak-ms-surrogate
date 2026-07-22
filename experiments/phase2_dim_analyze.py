"""
phase2_dim_analyze.py -- aggregate the Phase-2 dimensionality experiment (the headline).

For each (dimension d, method) over the N_START starts, computes the median + IQR of:
  * true-solves-to-target (censored at budget if not reached) -- the headline metric;
  * best true m_s at the budget;
  * success rate (fraction of starts that reached the target).
The claim under test: the surrogate-gradient reaches the target with FAR fewer true solves than
gradient-free search (CMA-ES, random), and the GAP GROWS with the true design dimension d, while
the rigid kappa heuristic plateaus. Saves data/phase2_dim_summary.json.
"""
import json
import os
import sys

import numpy as np

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def main():
    with open("data/phase2_dim_results.json") as f:
        recs = json.load(f)["recs"]
    with open("data/phase2_dim_setup.json") as f:
        setup = json.load(f)
    budget = setup["budget"]; target = setup["target"]
    d_list = setup["d_list"]; methods = setup["methods"]

    def agg(sel):
        # solves-to-target: n_solves if reached else censored at budget (worst case)
        solves = np.array([r["n_solves"] if r["reached"] else budget for r in sel], float)
        best = np.array([r["best_ms"] for r in sel], float)
        reached = np.array([1.0 if r["reached"] else 0.0 for r in sel])
        return dict(n=len(sel),
                    solves_median=float(np.median(solves)), solves_iqr=[float(np.percentile(solves, 25)),
                                                                        float(np.percentile(solves, 75))],
                    solves_mean=float(np.mean(solves)),
                    best_median=float(np.median(best)), best_iqr=[float(np.percentile(best, 25)),
                                                                  float(np.percentile(best, 75))],
                    success_rate=float(np.mean(reached)))

    summary = {"budget": budget, "target": target, "effective_dim": setup["effective_dim"],
               "cum_var": setup["cum_var"], "d_list": d_list, "methods": methods, "grid": {}}
    for d in d_list:
        summary["grid"][str(d)] = {}
        for meth in methods:
            sel = [r for r in recs if r["d"] == d and r["method"] == meth]
            if sel:
                summary["grid"][str(d)][meth] = agg(sel)

    with open("data/phase2_dim_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"effective control dim (participation ratio) = {setup['effective_dim']:.2f}")
    print(f"target m_s={target}, budget={budget} true solves; {len(recs)} runs\n")
    print(f"{'d':>3} | " + " | ".join(f"{m:>22s}" for m in methods))
    print("-" * (6 + 25 * len(methods)))
    for d in d_list:
        cells = []
        for meth in methods:
            g = summary["grid"][str(d)].get(meth)
            if g:
                cells.append(f"{g['solves_median']:4.0f}sv {g['success_rate']*100:3.0f}% ms{g['best_median']:.2f}")
            else:
                cells.append(" " * 22)
        print(f"{d:>3} | " + " | ".join(f"{c:>22s}" for c in cells))
    print("\n(cells: median true-solves-to-target | success% | median best m_s)")
    print("Saved data/phase2_dim_summary.json")


if __name__ == "__main__":
    main()
