"""
device2_portone_crosscheck.py -- rule #4 / Phase-0-style INDEPENDENT cross-check of Device-C m_s.

For a few converged Device-C equilibria (controls drawn from the kill-gate probe), rebuild the
linearisation at 80 modes and INDEPENDENTLY recompute the Portone stability margin three
algebraically-distinct ways (phase0_lib.independent_stability_margins) -- NONE calling FreeGSNKE's
own calculate_stability_margin. Confirm FreeGSNKE's reported Device-C m_s matches the independent
recompute to <5% (Phase-0 bar) BEFORE trusting any Device-C label in the design comparison.

Mirrors phase15_lib.forward_label's cold forward solve (expanded Device-C grid) but keeps the
nl_solver so we can read the raw M0/dM/M matrix blocks. Run thread-pinned (OMP=1).

  python experiments/device2_portone_crosscheck.py --n 5 --modes 80
"""
import argparse
import json
import os
import sys

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "experiments"))
import phase15_lib as L
import phase0_lib as P0
import device2_killgate as KG


def solve_with_nls(tok, active_currents, paxis, Ip, fvac, alpha_m, alpha_n, fix_n_modes):
    """Cold forward solve (mirrors forward_label) but RETURNS the nl_solver for matrix access."""
    from freegsnke import equilibrium_update, GSstaticsolver, nonlinear_solve
    from freegsnke.jtor_update import ConstrainPaxisIp
    eq = equilibrium_update.Equilibrium(tokamak=tok, nx=L.NX, ny=L.NY, **L.GRID)
    profiles = ConstrainPaxisIp(eq=eq, paxis=float(paxis), Ip=float(Ip), fvac=float(fvac),
                                alpha_m=float(alpha_m), alpha_n=float(alpha_n))
    solver = GSstaticsolver.NKGSsolver(eq=eq)
    eq.tokamak.set_all_coil_currents(np.asarray(active_currents, dtype=float))
    ok = False
    for step in (2.5, 1.5, 1.0):
        try:
            eq.plasma_psi = eq.create_psi_plasma_default(adaptive_centre=True)
            with np.errstate(divide="raise", invalid="raise", over="raise"):
                solver.forward_solve(eq=eq, profiles=profiles, target_relative_tolerance=L.FWD_TOL,
                                     max_solving_iterations=120, step_size=step, verbose=False, suppress=True)
            if solver.relative_change <= 10 * L.FWD_TOL and np.all(np.isfinite(eq.plasma_psi)):
                ok = True
                break
        except Exception:
            continue
    if not ok or bool(getattr(eq, "flag_limiter", True)) or float(eq.intersectsWall()):
        return None, None
    with np.errstate(divide="raise", invalid="raise", over="raise"):
        nls = nonlinear_solve.nl_solver(eq=eq, profiles=profiles, GSStaticSolver=solver,
                                        plasma_resistivity=1e-6, fix_n_vessel_modes=fix_n_modes, verbose=False)
    return eq, nls


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--modes", type=int, default=80)
    ap.add_argument("--rmax", type=float, default=2.8)
    args = ap.parse_args()

    import pandas as pd
    import pickle
    import _blas_guard
    _blas_guard.assert_pinned()                    # refuse to run real solves unpinned (locked protocol)
    with open(os.path.join(ROOT, "data", "device2_anchors.pkl"), "rb") as f:
        L.GRID = pickle.load(f)["meta"]["grid"]    # Device-C grid: single source of truth (anchors meta)
    tok = KG.load_device_c()
    df = pd.read_parquet(os.path.join(ROOT, "data", "device2_probe.parquet"))
    # span regimes: pick samples across the m_s range
    df = df.sort_values("m_s").reset_index(drop=True)
    pick = np.linspace(0, len(df) - 1, args.n).round().astype(int)

    results = []
    worst = 0.0
    print(f"Device-C independent Portone cross-check ({args.modes} modes), {args.n} shapes:")
    for i in pick:
        r = df.iloc[int(i)]
        ac = np.array([float(r[f"I_{c}"]) for c in L.ACTIVE_COILS])
        eq, nls = solve_with_nls(tok, ac, r["paxis"], r["Ip_target"], r["fvac"],
                                 r["alpha_m"], r["alpha_n"], args.modes)
        if nls is None:
            print(f"  shape {int(i)}: re-solve failed; skip")
            continue
        reported = float(np.ravel(np.asarray(nls.linearised_sol.stability_margin).real).max())
        ind = P0.independent_stability_margins(nls)
        def toppos(p):
            return float(p[0]) if len(p) else float("nan")
        mA, mB, mC = toppos(ind["pos_A"]), toppos(ind["pos_B"]), toppos(ind["pos_C"])
        pcts = [100 * abs(m - reported) / abs(reported) for m in (mA, mB, mC) if np.isfinite(m)]
        wp = max(pcts) if pcts else float("nan")
        worst = max(worst, wp)
        kappa = float(np.asarray(eq.separatrix())[:, 1].ptp() / np.asarray(eq.separatrix())[:, 0].ptp())
        print(f"  shape {int(i):3d}  kappa~{kappa:.3f}  FreeGSNKE m_s={reported:.6f}  "
              f"A={mA:.6f} B={mB:.6f} C={mC:.6f}  worst |Δ|={wp:.3f}%  n_pos={len(ind['pos_A'])}")
        results.append(dict(idx=int(i), reported=reported, method_A=mA, method_B=mB, method_C=mC,
                            worst_pct=wp, n_positive=int(len(ind["pos_A"])),
                            identity_M0pdM=ind["identity_M0pdM_eq_M"],
                            identity_Lstar=ind["identity_Lstar_eq_LmS"]))
    gate = worst < 5.0
    out = dict(modes=args.modes, n=len(results), worst_pct_diff=worst,
               gate_match_under_5pct=bool(gate), shapes=results)
    json.dump(out, open(os.path.join(ROOT, "data", "device2_portone_crosscheck.json"), "w"), indent=2)
    print(f"\n==> max % diff (independent vs FreeGSNKE) over {len(results)} Device-C shapes = "
          f"{worst:.4f}%   GATE (<5%): {'PASS' if gate else 'FAIL'}")
    print("Saved data/device2_portone_crosscheck.json")


if __name__ == "__main__":
    main()
