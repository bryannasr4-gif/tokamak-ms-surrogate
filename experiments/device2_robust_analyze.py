"""
device2_robust_analyze.py -- analyze the robust-design confirmation (Phase 5 #2).

Per (start, design in {nominal, robust, reduce_kappa}) computes TRUE-solver robustness metrics under
the operational-uncertainty ensemble: nominal-center m_s, mean, worst-case (min, INCLUDING perturbations
that fell off the diverted manifold = m_s 0, the honest worst case), CVaR@25%, P(m_s>0.15 design margin),
and converged-fraction (itself a robustness measure). Then pairs designs PER START (robust vs nominal,
robust vs reduce_kappa) on true worst-case m_s + P(m_s>0.15), resolved by regime, two-sided Wilcoxon.

The headline the surrogate uniquely enables: does the robust design have a higher TRUE worst-case m_s
than the nominal optimum (and the reduce-kappa heuristic), accepting a small nominal-m_s 'price of
robustness'? reduce-kappa and gradient-free cannot cheaply optimize this objective.

  python experiments/device2_robust_analyze.py
"""
import glob
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "experiments"))
import phase2_data as D


def metrics(center, pert):
    p = np.array(pert, dtype=float)               # 0.0 entries = failed/off-manifold perturbations
    good = p[p > 0]
    conv = float(len(good) / len(p)) if len(p) else 0.0
    worst_incl = float(p.min()) if len(p) else 0.0          # honest worst case (failures count as 0)
    worst_conv = float(good.min()) if len(good) else 0.0
    cvar = float(np.mean(np.sort(p)[:max(1, len(p) // 4)])) if len(p) else 0.0  # mean of worst 25% (incl 0s)
    return dict(center=float(center), mean=float(good.mean()) if len(good) else 0.0,
                worst=worst_incl, worst_converged=worst_conv, cvar25=cvar,
                p_above_015=float(np.mean(p >= 0.15)), conv_frac=conv, M=len(p))


def wilcoxon_pair(starts, key, a, b):
    from scipy.stats import wilcoxon
    pairs = [s for s in starts if a in s["m"] and b in s["m"]]
    if not pairs:
        return None
    da = np.array([s["m"][a][key] for s in pairs]); db = np.array([s["m"][b][key] for s in pairs])
    diff = da - db; nz = diff[diff != 0]
    wins = int(np.sum(diff > 0))
    p = float(wilcoxon(nz, alternative="two-sided").pvalue) if len(nz) else float("nan")
    return dict(n=len(pairs), a_better=wins, win_rate=wins / len(pairs), wilcoxon_p=p,
                median_a=float(np.median(da)), median_b=float(np.median(db)),
                median_diff=float(np.median(diff)))


def main():
    files = glob.glob(os.path.join(ROOT, "data", "device2_robust_results", "start*_*.json"))
    recs = []
    for f in files:
        try:
            recs.append(json.load(open(f)))
        except Exception:
            pass
    if not recs:
        print("no robust results yet"); sys.exit(1)

    # true 80-mode start regime from the Phase-5 retrained results
    msstart = {}
    for f in glob.glob(os.path.join(ROOT, "data", "device2_design_results", "retrained_job*.json")):
        r = json.load(open(f)); msstart[r["start_id"]] = r["ms_start"]

    by_start = {}
    for r in recs:
        s = by_start.setdefault(r["start_id"], dict(start_id=r["start_id"], cohort=r["cohort"], m={}))
        s["m"][r["design"]] = metrics(r["center_ms"], r["pert_ms"])
    for s in by_start.values():
        s["regime"] = D.regime_of(msstart.get(s["start_id"], 1.0))
    starts = list(by_start.values())

    DES = ["nominal", "robust", "reduce_kappa"]
    out = dict(n_starts=len(starts),
               n_complete=sum(1 for s in starts if all(d in s["m"] for d in DES)))
    print(f"=== ROBUST DESIGN CONFIRMATION (Device-C, true 80-mode perturbations) ===")
    print(f"starts: {len(starts)}  (all-3-designs complete: {out['n_complete']})")

    # median price of robustness + worst-case lift, pooled and by regime
    for reg in [name for name, _, _ in D.REGIMES] + ["POOLED"]:
        sub = starts if reg == "POOLED" else [s for s in starts if s["regime"] == reg]
        sub = [s for s in sub if all(d in s["m"] for d in DES)]
        if len(sub) < 2:
            continue
        out[reg] = {}
        print(f"\n----- {reg} (n={len(sub)}) -----")
        # NOTE: P(m_s>0.15) collapses to conv_frac on this data (the 0.15 design margin never binds a
        # converged solution -- they differ on 7/1920 perturbations), so report conv_frac only (one DOF).
        # worst_converged (worst m_s among CONVERGED perturbations) is the discriminating graded metric
        # when the unconditional worst-case is floor-saturated at 0 by off-manifold failures.
        for key, lab in [("worst", "true WORST-CASE m_s (incl fails=0)"),
                         ("worst_converged", "worst m_s among CONVERGED"), ("cvar25", "CVaR25 m_s"),
                         ("conv_frac", "diverted-converged frac"),
                         ("center", "nominal-center m_s")]:
            med = {dn: float(np.median([s["m"][dn][key] for s in sub])) for dn in DES}
            rn = wilcoxon_pair(sub, key, "robust", "nominal")
            rk = wilcoxon_pair(sub, key, "robust", "reduce_kappa")
            out[reg][key] = dict(median=med, robust_vs_nominal=rn, robust_vs_reduce_kappa=rk)
            print(f"  {lab:26s} med nom {med['nominal']:.3f} | rob {med['robust']:.3f} | redK {med['reduce_kappa']:.3f}"
                  f"   robust>nominal {rn['a_better']}/{rn['n']} p={rn['wilcoxon_p']:.3f}"
                  f" | robust>reduceK {rk['a_better']}/{rk['n']} p={rk['wilcoxon_p']:.3f}")

    json.dump(out, open(os.path.join(ROOT, "data", "device2_robust_summary.json"), "w"), indent=2)
    print("\nSaved data/device2_robust_summary.json")


if __name__ == "__main__":
    main()
