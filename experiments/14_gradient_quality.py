"""
14_gradient_quality.py -- Pre-Phase-1 de-risk: is d(m_s)/d(shape) well-defined (not noise)?

The single biggest risk flagged by the novelty/feasibility scoping: m_s is the largest
positive eigenvalue of a matrix assembled from a NOISY finite-difference Jacobian dIy/dI,
so the LABEL may be noisy and its gradient d(m_s)/d(shape) may be garbage near the
controllability boundary -- which would kill the gradient-USED contribution.

Cheap decisive check: take fine steps in a shape knob (zscale ~ elongation) around a base
point in the interesting region (kappa ~ 1.97, m_s ~ 0.52) and verify m_s(shape) is SMOOTH
at fine scale, i.e. the central finite-difference slope is consistent across step sizes
(a noise-dominated label would give an erratic, step-size-dependent slope).

If m_s is smooth here, the true-solver gradient exists and a learned surrogate gradient
has something real to match (Phase 1). If it is jagged, we learn that in an hour, not a month.
"""
import json
import os
import time
import traceback

import numpy as np

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from freegsnke import build_machine, equilibrium_update, GSstaticsolver, nonlinear_solve
from freegsnke.inverse import Inverse_optimizer
from freegsnke.jtor_update import ConstrainPaxisIp

MC = "machine_configs/MAST-U"
FIX_N_MODES = 40


def evaluate(tokamak, zscale):
    eq = equilibrium_update.Equilibrium(tokamak=tokamak, Rmin=0.1, Rmax=2.0, Zmin=-2.2, Zmax=2.2, nx=65, ny=129)
    profiles = ConstrainPaxisIp(eq=eq, paxis=8e3, Ip=6e5, fvac=0.5, alpha_m=1.8, alpha_n=1.2)
    solver = GSstaticsolver.NKGSsolver(eq=eq)
    eq.tokamak.set_coil_current("Solenoid", 5000)
    eq.tokamak["Solenoid"].control = False
    Rx, Zx, Rout, Rin = 0.6, 1.1 * zscale, 1.4, 0.34
    Zt = [Zx, -Zx, 0.0, 0.0, 2.0 * zscale, -2.0 * zscale, 1.62 * zscale, -1.62 * zscale]
    isoflux_set = np.array([[[Rx, Rx, Rin, Rout, 1.0, 1.0, 0.8, 0.8], Zt]])
    lims = [[5e3, 9e3, 9e3, 7e3, 7e3, 5e3, 4e3, 5e3, 0.0, 0.0, None],
            [-5e3, -9e3, -9e3, -7e3, -7e3, -5e3, -4e3, -5e3, -10e3, -10e3, None]]
    constrain = Inverse_optimizer(null_points=[[Rx, Rx], [Zx, -Zx]], isoflux_set=isoflux_set, coil_current_limits=lims)
    constrain.mu_coils = 1e5
    solver.solve(eq=eq, profiles=profiles, constrain=constrain, target_relative_tolerance=1e-6,
                 target_relative_psit_update=1e-3, verbose=False, l2_reg=np.array([1e-12] * 10 + [1e-6]))
    b = np.asarray(eq.separatrix()); R, Z = b[:, 0], b[:, 1]
    kappa = float((Z.max() - Z.min()) / (R.max() - R.min()))
    nls = nonlinear_solve.nl_solver(eq=eq, profiles=profiles, GSStaticSolver=solver,
                                    plasma_resistivity=1e-6, fix_n_vessel_modes=FIX_N_MODES, verbose=False)
    sm = np.ravel(np.asarray(nls.linearised_sol.stability_margin).real)
    gr = np.ravel(np.asarray(nls.linearised_sol.growth_rates).real)
    return kappa, (float(sm.max()) if sm.size else float("nan")), (float(gr.max()) if gr.size else 0.0)


def main():
    tokamak = build_machine.tokamak(
        active_coils_path=f"{MC}/MAST-U_like_active_coils.pickle",
        passive_coils_path=f"{MC}/MAST-U_like_passive_coils.pickle",
        limiter_path=f"{MC}/MAST-U_like_limiter.pickle",
        wall_path=f"{MC}/MAST-U_like_wall.pickle",
    )
    zs = [0.930, 0.945, 0.960, 0.975, 0.990]
    rows = []
    for z in zs:
        t0 = time.time()
        try:
            kap, ms, g = evaluate(tokamak, z)
            rows.append({"zscale": z, "kappa": kap, "m_s": ms, "gamma": g, "t": time.time() - t0})
            print(f"zscale={z:.3f}  kappa={kap:.4f}  m_s={ms:.4f}  gamma={g:8.2f}  ({time.time()-t0:.0f}s)")
        except Exception as e:
            print(f"zscale={z:.3f} FAILED {type(e).__name__}: {e}")
            traceback.print_exc()
        with open("data/14_gradient_quality.json", "w") as f:
            json.dump(rows, f, indent=2)

    # finite-difference slope d(m_s)/d(kappa) via central differences on consecutive points
    ok = [r for r in rows if np.isfinite(r["m_s"])]
    print("\n--- gradient quality (central differences of m_s wrt kappa) ---")
    for i in range(1, len(ok) - 1):
        dk = ok[i + 1]["kappa"] - ok[i - 1]["kappa"]
        dms = ok[i + 1]["m_s"] - ok[i - 1]["m_s"]
        slope = dms / dk if dk else float("nan")
        print(f"  at kappa={ok[i]['kappa']:.4f}:  d(m_s)/d(kappa) ~ {slope:.3f}")
    if len(ok) >= 3:
        slopes = []
        for i in range(1, len(ok) - 1):
            dk = ok[i + 1]["kappa"] - ok[i - 1]["kappa"]
            slopes.append((ok[i + 1]["m_s"] - ok[i - 1]["m_s"]) / dk)
        slopes = np.array(slopes)
        print(f"\n  slope range: {slopes.min():.3f} .. {slopes.max():.3f}")
        print(f"  monotone m_s(kappa)? {all(np.diff([r['m_s'] for r in ok]) < 0)}")
        print("  -> SMOOTH/CONSISTENT slope => true-solver gradient is well-defined (Phase-1 precondition met)."
              if np.all(slopes < 0) else "  -> sign flips in slope => investigate label noise / mode count.")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
