"""
phase2_modes_analyze.py -- Phase-2 component 0: the mode-convergence study.

dataset_v1 labels are at fix_n_vessel_modes=40, which Phase 0 flagged as NOT mode-converged
(the dominant systematic; 40->80 ~ +10%). Here we replayed a stratified subset at modes
{40,80,120,138} (138 = ALL passive structures = the fully-converged reference) and quantify:
  * the 40->138 drift (the full systematic error of the shipped labels), resolved by regime;
  * whether m_s is converged by 80 / 120 (residual |m-138|/138);
  * a per-regime multiplicative 40->138 correction factor.
Output drives the Phase-2 decision: re-label at the converged count, or report modes=40 with the
measured systematic band. Saves data/phase2_modes_summary.json.
"""
import json
import os
import sys

import numpy as np

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, "experiments")
import phase2_data as D

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def main():
    with open("data/phase2_modes.json") as f:
        blob = json.load(f)
    modes = [str(m) for m in blob["modes"]]
    ref = "138"
    rows = []
    for r in blob["recs"]:
        ms = r["ms"]
        if ms.get(ref) is None or not np.isfinite(ms[ref]) or ms[ref] <= 0:
            continue
        msf = {m: ms[m] for m in modes if ms.get(m) is not None and np.isfinite(ms[m]) and ms[m] > 0}
        if "40" not in msf or ref not in msf:
            continue
        rows.append(dict(regime=r["regime"], m_s_stored=r["m_s_stored"], ms=msf))
    print(f"{len(rows)} shapes with valid 40 & 138 labels\n")

    def rel(a, b):
        return (a - b) / b

    summary = {"n": len(rows), "modes": blob["modes"], "reference_modes": 138, "by_regime": {}, "overall": {}}
    # overall + per regime
    for scope, sel in [("overall", rows)] + [(name, [r for r in rows if r["regime"] == name])
                                             for name, _, _ in D.REGIMES]:
        if len(sel) < 2:
            continue
        d40 = np.array([rel(r["ms"]["40"], r["ms"][ref]) for r in sel])
        block = dict(n=len(sel),
                     drift_40_to_138=dict(median=float(np.median(d40)), mean=float(np.mean(d40)),
                                          p10=float(np.percentile(d40, 10)), p90=float(np.percentile(d40, 90))),
                     corr_factor_40_to_138=dict(median=float(np.median(
                         [r["ms"][ref] / r["ms"]["40"] for r in sel]))))
        for m in ("80", "120"):
            dd = np.array([abs(rel(r["ms"][m], r["ms"][ref])) for r in sel if m in r["ms"]])
            if len(dd):
                block[f"resid_{m}_vs_138_abs"] = dict(median=float(np.median(dd)),
                                                      max=float(np.max(dd)), n=int(len(dd)))
        if scope == "overall":
            summary["overall"] = block
        else:
            summary["by_regime"][scope] = block

    with open("data/phase2_modes_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    o = summary["overall"]
    print("=== MODE CONVERGENCE (m_s drift vs 138-mode reference) ===")
    print(f" overall (n={o['n']}): 40->138 median drift = {o['drift_40_to_138']['median']*100:+.1f}% "
          f"(p10..p90 {o['drift_40_to_138']['p10']*100:+.1f}..{o['drift_40_to_138']['p90']*100:+.1f}%)")
    print(f"   residual |80-138|/138 median={o.get('resid_80_vs_138_abs',{}).get('median',float('nan'))*100:.1f}% "
          f"max={o.get('resid_80_vs_138_abs',{}).get('max',float('nan'))*100:.1f}% | "
          f"|120-138| median={o.get('resid_120_vs_138_abs',{}).get('median',float('nan'))*100:.1f}%")
    print(" by regime:")
    for name, _, _ in D.REGIMES:
        b = summary["by_regime"].get(name)
        if not b:
            continue
        print(f"   {name:11s} n={b['n']:2d}  40->138={b['drift_40_to_138']['median']*100:+5.1f}%  "
              f"|80-138|={b.get('resid_80_vs_138_abs',{}).get('median',float('nan'))*100:4.1f}%  "
              f"|120-138|={b.get('resid_120_vs_138_abs',{}).get('median',float('nan'))*100:4.1f}%  "
              f"x-factor={b['corr_factor_40_to_138']['median']:.2f}")
    print("\nSaved data/phase2_modes_summary.json")


if __name__ == "__main__":
    main()
