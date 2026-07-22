"""
11_growth_rate_smoke.py -- FEASIBILITY de-risk for the stability-margin pivot.

Goal (cheapest decisive test first, per the project's prior lessons):
  (1) Confirm FreeGSNKE computes a vertical-instability growth rate gamma for a
      realistic MAST-U (spherical tokamak) free-boundary equilibrium on this laptop.
  (2) Benchmark the COST of one gamma evaluation (inverse solve + linearisation
      Jacobian build). This sets the data-generation budget for any surrogate.
  (3) Probe the equilibrium geometry API so the next script can scan shape -> gamma.

Reproduces the official example01a (diverted inverse solve) + example10 (growth rate).
Machine description: real MAST-U_like pickles in machine_configs/MAST-U/.
"""
import json
import os
import time
import traceback

import numpy as np

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # project root

from freegsnke import build_machine, equilibrium_update, GSstaticsolver, nonlinear_solve
from freegsnke.inverse import Inverse_optimizer
from freegsnke.jtor_update import ConstrainPaxisIp

MC = "machine_configs/MAST-U"


def shape_params(eq):
    """Best-effort extraction of geometry + global quantities (probes the API)."""
    out = {}
    for name in [
        "geometricAxis", "magneticAxis", "minorRadius", "aspectRatio",
        "elongation", "triangularity", "poloidalBeta", "internalInductance",
        "plasmaCurrent", "Rcurrent", "Zcurrent",
    ]:
        try:
            v = getattr(eq, name)()
            out[name] = (float(np.ravel(v)[0]) if np.ndim(v) else float(v)) if name != "magneticAxis" else list(np.ravel(v)[:2])
        except Exception as e:
            out[name] = f"ERR {type(e).__name__}: {e}"
    # boundary-based elongation as a robust fallback
    for meth in ["separatrix", "closed_boundary"]:
        try:
            b = np.asarray(getattr(eq, meth)())
            R, Z = b[:, 0], b[:, 1]
            out["kappa_box"] = float((Z.max() - Z.min()) / (R.max() - R.min()))
            out["bbox"] = [float(R.min()), float(R.max()), float(Z.min()), float(Z.max())]
            out["boundary_method"] = meth
            break
        except Exception as e:
            out[f"{meth}_err"] = f"ERR {type(e).__name__}: {e}"
    return out


def main():
    results = {}

    # --- build machine -----------------------------------------------------
    tokamak = build_machine.tokamak(
        active_coils_path=f"{MC}/MAST-U_like_active_coils.pickle",
        passive_coils_path=f"{MC}/MAST-U_like_passive_coils.pickle",
        limiter_path=f"{MC}/MAST-U_like_limiter.pickle",
        wall_path=f"{MC}/MAST-U_like_wall.pickle",
    )
    results["n_active_coils"] = int(tokamak.n_active_coils)
    results["n_passive_coils"] = int(tokamak.n_passive_coils)
    results["active_coil_names"] = list(tokamak.coils_list[: tokamak.n_active_coils])
    print("MACHINE:", results["n_active_coils"], "active,", results["n_passive_coils"], "passive")
    print("ACTIVE COILS:", results["active_coil_names"])

    eq = equilibrium_update.Equilibrium(
        tokamak=tokamak, Rmin=0.1, Rmax=2.0, Zmin=-2.2, Zmax=2.2, nx=65, ny=129
    )
    profiles = ConstrainPaxisIp(eq=eq, paxis=8e3, Ip=6e5, fvac=0.5, alpha_m=1.8, alpha_n=1.2)
    solver = GSstaticsolver.NKGSsolver(eq=eq)

    # --- baseline diverted inverse solve (example01a) ----------------------
    eq.tokamak.set_coil_current("Solenoid", 5000)
    eq.tokamak["Solenoid"].control = False
    Rx, Zx, Rout, Rin = 0.6, 1.1, 1.4, 0.34
    null_points = [[Rx, Rx], [Zx, -Zx]]
    isoflux_set = np.array([[
        [Rx, Rx, Rin, Rout, 1.0, 1.0, 0.8, 0.8],
        [Zx, -Zx, 0.0, 0.0, 2.0, -2.0, 1.62, -1.62],
    ]])
    coil_current_limits = [
        [5e3, 9e3, 9e3, 7e3, 7e3, 5e3, 4e3, 5e3, 0.0, 0.0, None],
        [-5e3, -9e3, -9e3, -7e3, -7e3, -5e3, -4e3, -5e3, -10e3, -10e3, None],
    ]
    constrain = Inverse_optimizer(
        null_points=null_points, isoflux_set=isoflux_set, coil_current_limits=coil_current_limits
    )
    constrain.mu_coils = 1e5

    t0 = time.time()
    solver.solve(
        eq=eq, profiles=profiles, constrain=constrain,
        target_relative_tolerance=1e-6, target_relative_psit_update=1e-3,
        verbose=True, l2_reg=np.array([1e-12] * 10 + [1e-6]),
    )
    results["inverse_solve_seconds"] = time.time() - t0
    print(f"\nINVERSE SOLVE: {results['inverse_solve_seconds']:.1f} s")

    results["geometry"] = shape_params(eq)
    print("GEOMETRY:", json.dumps(results["geometry"], indent=2, default=str))
    results["currents"] = {k: float(v) for k, v in eq.tokamak.getCurrents().items()}

    # --- growth rate / stability margin (example10) ------------------------
    t0 = time.time()
    nls = nonlinear_solve.nl_solver(
        eq=eq, profiles=profiles, GSStaticSolver=solver,
        plasma_resistivity=1e-6, fix_n_vessel_modes=40, verbose=True,
    )
    results["linearisation_seconds"] = time.time() - t0
    gr = np.asarray(nls.linearised_sol.growth_rates).real
    sm = np.asarray(nls.linearised_sol.stability_margin).real
    results["growth_rates"] = [float(x) for x in np.ravel(gr)]
    results["stability_margin"] = [float(x) for x in np.ravel(sm)]
    results["n_retained_modes"] = int(nls.dIydI.shape[1])
    print(f"\nLINEARISATION+GROWTH: {results['linearisation_seconds']:.1f} s")
    print("RETAINED MODES (Jacobian cols):", results["n_retained_modes"])
    print("GROWTH RATES [1/s]:", results["growth_rates"])
    print("STABILITY MARGIN m_s:", results["stability_margin"])

    os.makedirs("data", exist_ok=True)
    with open("data/11_growth_rate_smoke.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print("\nSaved -> data/11_growth_rate_smoke.json")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        print("\nSMOKE TEST FAILED -- see traceback above.")
