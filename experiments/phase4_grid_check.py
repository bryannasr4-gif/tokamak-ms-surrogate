"""
phase4_grid_check.py -- verify the 65x65 grid systematic ON THE MARGINAL BAND at the converged 80
modes (Phase-4 quality audit, labels-protocol item). PHASE0_PROTOCOL §3.3 only checked grid at 40
modes; the shipped labels + all Phase 3/4 results are 80-mode, so we re-verify the grid sensitivity
where the paper actually headlines (the m_s->0 boundary, e.g. the gallery hero start m_s~0.19).

For ~15-20 marginal-focused shapes: solve at 65x65 AND 129x129 (both 80 modes, OMP=1, cold forward
solve via phase15_lib.forward_label) and report m_s(129)/m_s(65), median + MAX, resolved by m_s band.
A small shift (<~3%) means the 65x65 marginal m_s is defensible as-is; a large shift means carry a
marginal-regime grid band on the absolute m_s narrative. Saves data/phase4_grid_check.json.
"""
import json
import os
import sys
import time

import numpy as np

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.getcwd(), "experiments"))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import pandas as pd
import phase15_lib as L
import phase2_data as D


def solve_at(tok, u_row, nx):
    """forward_label at grid nx x nx, 80 modes, from a dataset row's controls."""
    c = {k: float(u_row[k]) for k in ["paxis", "Ip_target", "fvac", "alpha_m", "alpha_n"]}
    I = np.zeros(12); I[0] = 5000.0
    for i, name in enumerate(L.ACTIVE_COILS):
        if i == 0:
            continue
        I[i] = float(u_row[f"I_{name}"])
    rec = L.forward_label(tok, I, c["paxis"], c["Ip_target"], c["fvac"], c["alpha_m"], c["alpha_n"],
                          fix_n_modes=80, nx=nx, ny=nx)
    return float(rec["m_s"]), float(rec["kappa"])


def main():
    tok = L.load_machine()
    df = pd.read_parquet("data/dataset_v1_80.parquet")
    # marginal-focused selection: ~10 at m_s in [0.11,0.30] (incl. the lowest), ~5 in [0.30,0.45],
    # ~3 mid/stable anchors to confirm the §3.3 -5..-7% stable trend reproduces at 80 modes.
    pools = [("deep_marginal", df[(df.m_s >= 0.11) & (df.m_s < 0.30)], 10),
             ("near_marginal", df[(df.m_s >= 0.30) & (df.m_s < 0.45)], 5),
             ("mid_stable", df[(df.m_s >= 0.8) & (df.m_s < 2.0)], 3)]
    rows = []
    for name, pool, k in pools:
        pool = pool.sort_values("m_s")
        idx = np.linspace(0, len(pool) - 1, min(k, len(pool))).round().astype(int)
        for i in idx:
            rows.append((name, pool.iloc[i]))

    out = dict(n=len(rows), shapes=[])
    t0 = time.time()
    for band, r in rows:
        try:
            ms65, k65 = solve_at(tok, r, 65)
            ms129, k129 = solve_at(tok, r, 129)
            ratio = ms129 / ms65 if ms65 > 0 else float("nan")
            out["shapes"].append(dict(band=band, m_s_stored=float(r["m_s"]), m_s_65=ms65, m_s_129=ms129,
                                      ratio_129_over_65=ratio, pct_shift=100 * (ratio - 1),
                                      kappa_65=k65, kappa_129=k129))
            print(f"[{band:13s}] stored {r['m_s']:.3f}  m_s65={ms65:.4f} m_s129={ms129:.4f}  "
                  f"shift {100*(ratio-1):+.1f}%  ({(time.time()-t0)/60:.1f}min)", flush=True)
        except Exception as e:
            print(f"[{band}] FAILED {type(e).__name__}: {e}", flush=True)

    # summary resolved by band
    out["by_band"] = {}
    for band in ("deep_marginal", "near_marginal", "mid_stable"):
        sh = [s for s in out["shapes"] if s["band"] == band and np.isfinite(s["pct_shift"])]
        if sh:
            p = [abs(s["pct_shift"]) for s in sh]
            signed = [s["pct_shift"] for s in sh]
            out["by_band"][band] = dict(n=len(sh), median_abs_pct=float(np.median(p)),
                                        max_abs_pct=float(np.max(p)), median_signed_pct=float(np.median(signed)))
    allsh = [abs(s["pct_shift"]) for s in out["shapes"] if np.isfinite(s["pct_shift"])]
    out["overall_median_abs_pct"] = float(np.median(allsh)) if allsh else float("nan")
    out["overall_max_abs_pct"] = float(np.max(allsh)) if allsh else float("nan")
    json.dump(out, open("data/phase4_grid_check.json", "w"), indent=2)

    print("\n=== 65x65 -> 129x129 grid shift on m_s (80 modes), by band ===")
    for band, d in out["by_band"].items():
        print(f"  {band:14s} n={d['n']}: median |shift| {d['median_abs_pct']:.1f}%  "
              f"max |shift| {d['max_abs_pct']:.1f}%  (median signed {d['median_signed_pct']:+.1f}%)")
    print(f"  OVERALL: median |shift| {out['overall_median_abs_pct']:.1f}%  max {out['overall_max_abs_pct']:.1f}%")
    print("Saved data/phase4_grid_check.json")


if __name__ == "__main__":
    main()
