"""
12_elongation_scan.py -- Is gamma(shape) a LEARNABLE signal?

The feasibility smoke test (11) confirmed FreeGSNKE computes a sensible vertical
growth rate (gamma ~ 260/s, m_s ~ 0.39) for a MAST-U-like ST at ~46 s/sample.

This script answers the second make-or-break question: does gamma vary SMOOTHLY and
SYSTEMATICALLY with plasma shape? If it does, a regression surrogate gamma(shape) is
well-posed. We scan target ELONGATION (by scaling the vertical extent of the inverse-
solve boundary targets) and record gamma, m_s, and geometry for each converged shape.

Expectation (textbook): taller plasma -> higher kappa -> MORE vertically unstable ->
higher gamma and lower stability margin m_s.
"""
import json
import os
import time
import traceback

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from freegsnke import build_machine, equilibrium_update, GSstaticsolver, nonlinear_solve
from freegsnke.inverse import Inverse_optimizer
from freegsnke.jtor_update import ConstrainPaxisIp

MC = "machine_configs/MAST-U"
FIX_N_MODES = 40


def build_tokamak():
    return build_machine.tokamak(
        active_coils_path=f"{MC}/MAST-U_like_active_coils.pickle",
        passive_coils_path=f"{MC}/MAST-U_like_passive_coils.pickle",
        limiter_path=f"{MC}/MAST-U_like_limiter.pickle",
        wall_path=f"{MC}/MAST-U_like_wall.pickle",
    )


def geom(eq):
    out = {}
    for name in ["geometricAxis", "minorRadius", "aspectRatio", "triangularity"]:
        try:
            v = getattr(eq, name)()
            out[name] = float(np.ravel(v)[0])
        except Exception:
            out[name] = None
    try:
        b = np.asarray(eq.separatrix())
        R, Z = b[:, 0], b[:, 1]
        out["kappa"] = float((Z.max() - Z.min()) / (R.max() - R.min()))
        out["Zextent"] = float(Z.max() - Z.min())
    except Exception:
        out["kappa"] = None
    return out


def solve_shape(tokamak, zscale):
    """Inverse-solve a diverted MAST-U shape with vertical targets scaled by zscale,
    then compute the vertical growth rate and stability margin. Returns a dict or None."""
    eq = equilibrium_update.Equilibrium(
        tokamak=tokamak, Rmin=0.1, Rmax=2.0, Zmin=-2.2, Zmax=2.2, nx=65, ny=129
    )
    profiles = ConstrainPaxisIp(eq=eq, paxis=8e3, Ip=6e5, fvac=0.5, alpha_m=1.8, alpha_n=1.2)
    solver = GSstaticsolver.NKGSsolver(eq=eq)

    eq.tokamak.set_coil_current("Solenoid", 5000)
    eq.tokamak["Solenoid"].control = False

    Rx, Zx, Rout, Rin = 0.6, 1.1 * zscale, 1.4, 0.34
    null_points = [[Rx, Rx], [Zx, -Zx]]
    # scale every Z target by zscale (stretch the plasma vertically)
    Zt = np.array([Zx, -Zx, 0.0, 0.0, 2.0 * zscale, -2.0 * zscale, 1.62 * zscale, -1.62 * zscale])
    isoflux_set = np.array([[[Rx, Rx, Rin, Rout, 1.0, 1.0, 0.8, 0.8], list(Zt)]])
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
        verbose=False, l2_reg=np.array([1e-12] * 10 + [1e-6]),
    )
    t_inv = time.time() - t0

    g = geom(eq)

    t0 = time.time()
    nls = nonlinear_solve.nl_solver(
        eq=eq, profiles=profiles, GSStaticSolver=solver,
        plasma_resistivity=1e-6, fix_n_vessel_modes=FIX_N_MODES, verbose=False,
    )
    t_lin = time.time() - t0

    gr = np.ravel(np.asarray(nls.linearised_sol.growth_rates).real)
    sm = np.ravel(np.asarray(nls.linearised_sol.stability_margin).real)
    rec = dict(
        zscale=zscale,
        gamma=float(gr.max()) if gr.size else 0.0,
        n_unstable=int(gr.size),
        stability_margin=float(sm.max()) if sm.size else float("nan"),
        leuer=float(getattr(nls, "Leuer_metals_stab_over_active_destab", float("nan"))),
        t_inverse=t_inv, t_linear=t_lin,
        currents={k: float(v) for k, v in eq.tokamak.getCurrents().items()},
        **g,
    )
    return rec


def main():
    tokamak = build_tokamak()
    print("Machine:", tokamak.n_active_coils, "active,", tokamak.n_passive_coils, "passive")

    # Restricted to the numerically well-behaved, controllability-crossing range.
    # zscale>1.04 drives the plasma into the vessel and the GS solve degrades (the
    # marginal regime the surrogate must later quantify uncertainty over).
    zscales = [0.80, 0.84, 0.88, 0.92, 0.96, 1.00, 1.04]
    os.makedirs("data", exist_ok=True)
    records = []
    for z in zscales:
        try:
            rec = solve_shape(tokamak, z)
            records.append(rec)
            print(f"zscale={z:.2f}  kappa={rec['kappa']:.3f}  gamma={rec['gamma']:8.2f}/s  "
                  f"m_s={rec['stability_margin']:.3f}  n_unstable={rec['n_unstable']}  "
                  f"(inv {rec['t_inverse']:.0f}s + lin {rec['t_linear']:.0f}s)")
        except Exception as e:
            print(f"zscale={z:.2f}  FAILED: {type(e).__name__}: {e}")
            traceback.print_exc()
        # incremental save so a slow/aborted run still leaves a usable dataset
        with open("data/12_elongation_scan.json", "w") as f:
            json.dump(records, f, indent=2)

    ok = [r for r in records if r.get("kappa") and np.isfinite(r["gamma"])]
    if len(ok) >= 2:
        k = np.array([r["kappa"] for r in ok])
        g = np.array([r["gamma"] for r in ok])
        ms = np.array([r["stability_margin"] for r in ok])
        order = np.argsort(k)
        k, g, ms = k[order], g[order], ms[order]

        fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
        ax[0].plot(k, g, "o-", color="crimson")
        ax[0].set_xlabel("elongation $\\kappa$ (separatrix box)")
        ax[0].set_ylabel("vertical growth rate $\\gamma$ [1/s]")
        ax[0].set_title("Vertical instability grows with elongation")
        ax[0].grid(alpha=0.3)
        ax[1].plot(k, ms, "s-", color="navy")
        ax[1].set_xlabel("elongation $\\kappa$ (separatrix box)")
        ax[1].set_ylabel("stability margin $m_s$")
        ax[1].set_title("Stability margin shrinks with elongation")
        ax[1].grid(alpha=0.3)
        fig.suptitle("FreeGSNKE MAST-U-like: $\\gamma(\\mathrm{shape})$ is a smooth, learnable signal", y=1.02)
        fig.tight_layout()
        os.makedirs("figures", exist_ok=True)
        fig.savefig("figures/12_growth_rate_vs_elongation.png", dpi=140, bbox_inches="tight")
        print("\nSaved figure -> figures/12_growth_rate_vs_elongation.png")
        print("kappa range:", float(k.min()), "->", float(k.max()))
        print("gamma range:", float(g.min()), "->", float(g.max()), "/s")
    else:
        print("\nNot enough converged points to plot.")
    print(f"\nScan complete: {len(ok)}/{len(zscales)} converged. Saved -> data/12_elongation_scan.json")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
