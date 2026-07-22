"""
phase4_gallery_analyze.py -- analyze the kappa-constrained gallery run (data/phase4_gallery_results.json):
  * verify kappa stayed locked (|kappa_final - kappa_start| <= KTOL) for every accepted surrogate design;
  * the kappa_nudge CONFOUND: median m_s bought by the residual +-KTOL kappa freedom alone, and the
    surrogate's EXCESS over it = the genuine SECONDARY-lever (learned-m_s) contribution;
  * surrogate vs gradient-free (cma) at fixed kappa, with full recording (re-confirms the headline);
  * which secondary levers the surrogate moved (start->best descriptor deltas), per regime;
  * pick representative gallery panels (best / typical / a near-tie or failure) for the shape figure.
Saves data/phase4_gallery_summary.json.
"""
import json
import os
import sys

import numpy as np

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.getcwd(), "experiments"))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
import phase4_gallery_lib as G

KTOL = G.KTOL


def by(recs, method):
    return {r["start_i"]: r for r in recs if r["method"] == method}


def med(x):
    x = [v for v in x if v is not None and np.isfinite(v)]
    return float(np.median(x)) if x else float("nan")


def main():
    blob = json.load(open("data/phase4_gallery_results.json"))
    recs = blob["recs"]
    sur = by(recs, "surrogate"); nud = by(recs, "kappa_nudge"); cma = by(recs, "cma")
    starts = sorted(sur)

    out = dict(ktol=KTOL, budget=blob.get("budget"), n_starts=len(starts), per_start=[])
    # kappa-lock verification + confound decomposition
    drift_ok = 0
    sur_gain, nud_gain, cma_gain, excess = [], [], [], []
    for si in starts:
        s = sur[si]; n = nud.get(si, {}); c = cma.get(si, {})
        kd = s.get("kappa_drift", float("nan"))
        ok = np.isfinite(kd) and kd <= KTOL + 1e-9
        drift_ok += int(ok)
        sg, ng, cg = s.get("gain", np.nan), n.get("gain", np.nan), c.get("gain", np.nan)
        sur_gain.append(sg); nud_gain.append(ng); cma_gain.append(cg)
        if np.isfinite(sg) and np.isfinite(ng):
            excess.append(sg - ng)
        # secondary-lever deltas (start->best) for the surrogate
        deltas = {}
        if s.get("accepted"):
            d0 = s["accepted"][0]["desc"]; d1 = s["best_desc"]
            if d0 and d1:
                for f in ("kappa", "sq_uo", "sq_lo", "gap_inner", "gap_outer", "li", "betap", "delta"):
                    if f in d0 and f in d1:
                        deltas[f] = float(d1[f] - d0[f])
        out["per_start"].append(dict(start_i=si, regime=s["regime"], m_s_start=s["m_s_start"],
                                     surrogate_gain=sg, kappa_nudge_gain=ng, cma_gain=cg,
                                     kappa_drift=kd, kappa_locked=bool(ok),
                                     surrogate_best=s.get("best_ms"), deltas=deltas,
                                     n_accept=len(s.get("accepted", [])) - 1))
    out["kappa_lock_verified"] = dict(n_ok=drift_ok, n=len(starts), max_drift=float(
        np.nanmax([sur[si].get("kappa_drift", np.nan) for si in starts])))
    out["median_gain"] = dict(surrogate=med(sur_gain), kappa_nudge=med(nud_gain), cma=med(cma_gain),
                              surrogate_excess_over_kappa_nudge=med(excess))
    # by regime
    out["by_regime"] = {}
    for reg in ("marginal", "mid"):
        idx = [i for i, si in enumerate(starts) if sur[si]["regime"] == reg]
        if idx:
            out["by_regime"][reg] = dict(
                n=len(idx),
                surrogate=med([sur_gain[i] for i in idx]),
                kappa_nudge=med([nud_gain[i] for i in idx]),
                cma=med([cma_gain[i] for i in idx]),
                excess=med([sur_gain[i] - nud_gain[i] for i in idx
                            if np.isfinite(sur_gain[i]) and np.isfinite(nud_gain[i])]))
    # pick gallery panels: best marginal, a typical mid, and the smallest-gain (honesty)
    cand = [p for p in out["per_start"] if p["kappa_locked"] and np.isfinite(p["surrogate_gain"])]
    marg = sorted([p for p in cand if p["regime"] == "marginal"], key=lambda p: -p["surrogate_gain"])
    mid = sorted([p for p in cand if p["regime"] == "mid"], key=lambda p: -p["surrogate_gain"])
    worst = sorted(cand, key=lambda p: p["surrogate_gain"])[:1]
    panels = []
    if marg:
        panels.append(("best_marginal", marg[0]["start_i"]))
    if mid:
        panels.append(("typical_mid", mid[len(mid) // 2]["start_i"]))
    if worst:
        panels.append(("smallest_gain", worst[0]["start_i"]))
    out["gallery_panels"] = panels
    json.dump(out, open("data/phase4_gallery_summary.json", "w"), indent=2)

    print(f"=== kappa-constrained gallery ({out['n_starts']} starts, budget {out['budget']}, "
          f"KTOL {KTOL}) ===")
    kl = out["kappa_lock_verified"]
    print(f"kappa lock verified: {kl['n_ok']}/{kl['n']} surrogate designs within KTOL "
          f"(max drift {kl['max_drift']:.4f})")
    mg = out["median_gain"]
    print(f"\nmedian gain:  surrogate {mg['surrogate']:+.3f}   kappa_nudge (residual-kappa only) "
          f"{mg['kappa_nudge']:+.3f}   gradient-free {mg['cma']:+.3f}")
    print(f"  -> surrogate EXCESS over residual-kappa-nudge (the SECONDARY-lever / learned-m_s "
          f"contribution): {mg['surrogate_excess_over_kappa_nudge']:+.3f}")
    for reg, d in out["by_regime"].items():
        print(f"  [{reg}] n={d['n']}: surrogate {d['surrogate']:+.3f}  kappa_nudge {d['kappa_nudge']:+.3f}  "
              f"gradient-free {d['cma']:+.3f}  excess {d['excess']:+.3f}")
    print(f"\ngallery panels: {panels}")
    print("Saved data/phase4_gallery_summary.json")


if __name__ == "__main__":
    main()
