"""
phase1_generate.py -- Phase-1 dataset: true FreeGSNKE m_s(shape) over a 2-knob shape slice.

Shape knobs (the "(kappa, wall gap)" slice from SCOPING_vertical_stability.md):
  knob1 = zscale : vertical stretch of the boundary targets  -> elongation
  knob2 = dR     : rigid radial shift of all R targets [m]    -> plasma-wall gap

Each grid point: inverse-solve to the target shape -> build the linearisation (3051 x 53
finite-difference Jacobian dIy/dI) -> lumped-circuit eigenvalue -> {m_s, gamma, geometry}.

Chunked for parallelism: run several copies with different --chunk so an N-core laptop
fills the grid ~N x faster. Pin BLAS threads to 1 per process (set in the launch command)
to avoid oversubscription. Saves incrementally to data/phase1_chunk_{chunk}.json.
"""
import argparse
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
NX, NY = 65, 65  # proven fast+accurate in the smoke test (~46 s/label)

# Shape-knob grid (well-behaved, cleanly-unstable region: kappa ~1.82-2.05)
ZSCALES = np.round(np.linspace(0.88, 1.00, 8), 4)
DRS = np.round(np.linspace(-0.04, 0.04, 8), 4)


def build_tokamak():
    return build_machine.tokamak(
        active_coils_path=f"{MC}/MAST-U_like_active_coils.pickle",
        passive_coils_path=f"{MC}/MAST-U_like_passive_coils.pickle",
        limiter_path=f"{MC}/MAST-U_like_limiter.pickle",
        wall_path=f"{MC}/MAST-U_like_wall.pickle",
    )


def evaluate(tokamak, zscale, dR):
    """Inverse-solve a shape (zscale, dR) and return {m_s, gamma, kappa, ...} or raise."""
    eq = equilibrium_update.Equilibrium(tokamak=tokamak, Rmin=0.1, Rmax=2.0, Zmin=-2.2, Zmax=2.2, nx=NX, ny=NY)
    profiles = ConstrainPaxisIp(eq=eq, paxis=8e3, Ip=6e5, fvac=0.5, alpha_m=1.8, alpha_n=1.2)
    solver = GSstaticsolver.NKGSsolver(eq=eq)
    eq.tokamak.set_coil_current("Solenoid", 5000)
    eq.tokamak["Solenoid"].control = False

    Rx, Zx = 0.6 + dR, 1.1 * zscale
    Rin, Rout = 0.34 + dR, 1.4 + dR
    RlegA, RlegB = 1.0 + dR, 0.8 + dR
    null_points = [[Rx, Rx], [Zx, -Zx]]
    R_targets = [Rx, Rx, Rin, Rout, RlegA, RlegA, RlegB, RlegB]
    Z_targets = [Zx, -Zx, 0.0, 0.0, 2.0 * zscale, -2.0 * zscale, 1.62 * zscale, -1.62 * zscale]
    isoflux_set = np.array([[R_targets, Z_targets]])
    lims = [[5e3, 9e3, 9e3, 7e3, 7e3, 5e3, 4e3, 5e3, 0.0, 0.0, None],
            [-5e3, -9e3, -9e3, -7e3, -7e3, -5e3, -4e3, -5e3, -10e3, -10e3, None]]
    constrain = Inverse_optimizer(null_points=null_points, isoflux_set=isoflux_set, coil_current_limits=lims)
    constrain.mu_coils = 1e5

    solver.solve(eq=eq, profiles=profiles, constrain=constrain, target_relative_tolerance=1e-6,
                 target_relative_psit_update=1e-3, verbose=False, l2_reg=np.array([1e-12] * 10 + [1e-6]))

    b = np.asarray(eq.separatrix()); R, Z = b[:, 0], b[:, 1]
    kappa = float((Z.max() - Z.min()) / (R.max() - R.min()))
    Rgeo = float(eq.geometricAxis()[0]); a = float(eq.minorRadius())

    nls = nonlinear_solve.nl_solver(eq=eq, profiles=profiles, GSStaticSolver=solver,
                                    plasma_resistivity=1e-6, fix_n_vessel_modes=FIX_N_MODES, verbose=False)
    sm = np.ravel(np.asarray(nls.linearised_sol.stability_margin).real)
    gr = np.ravel(np.asarray(nls.linearised_sol.growth_rates).real)
    return dict(
        zscale=float(zscale), dR=float(dR), kappa=kappa, Rgeo=Rgeo, minor=a,
        aspect=float(eq.aspectRatio()), tri=float(eq.triangularity()),
        m_s=float(sm.max()) if sm.size else float("nan"),
        gamma=float(gr.max()) if gr.size else 0.0,
        n_unstable=int(gr.size),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nchunks", type=int, default=1)
    ap.add_argument("--chunk", type=int, default=0)
    args = ap.parse_args()

    grid = [(z, d) for z in ZSCALES for d in DRS]
    mine = [(i, zd) for i, zd in enumerate(grid) if i % args.nchunks == args.chunk]

    os.makedirs("data", exist_ok=True)
    out = f"data/phase1_chunk_{args.chunk}.json"
    tokamak = build_tokamak()
    recs = []
    for i, (z, d) in mine:
        t0 = time.time()
        try:
            r = evaluate(tokamak, z, d)
            r["idx"] = i; r["t"] = time.time() - t0
            recs.append(r)
            print(f"[chunk {args.chunk}] idx={i:2d} z={z:.3f} dR={d:+.3f} "
                  f"kappa={r['kappa']:.3f} m_s={r['m_s']:.3f} gamma={r['gamma']:7.1f} ({r['t']:.0f}s)",
                  flush=True)
        except Exception as e:
            print(f"[chunk {args.chunk}] idx={i:2d} z={z:.3f} dR={d:+.3f} FAILED {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()
        with open(out, "w") as f:
            json.dump(recs, f, indent=2)
    print(f"[chunk {args.chunk}] DONE {len(recs)}/{len(mine)} -> {out}", flush=True)


if __name__ == "__main__":
    main()
