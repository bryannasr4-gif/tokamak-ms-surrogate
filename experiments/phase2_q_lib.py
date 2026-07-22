"""
phase2_q_lib.py -- LIGHT forward-only q-profile extraction (Phase-2.5b, Task B).

Adds the safety factor q95 (and q0-proxy / qmin) as a physics FEATURE to dataset_v1_80, with NO
m_s linearisation (the expensive part). We replay each shape's stored controls through the SAME
crash-proofed forward free-boundary solve as phase15_lib.forward_label (verified bit-exact replay:
m_s and kappa reproduce the stored 80-mode labels to all printed digits), then read q off the
converged equilibrium via eq.q(psinorm).

FreeGSNKE/FreeGS q API (verified on a converged eq, 2026-06-22):
  eq.q(psinorm) takes an ARRAY of normalised psi in (0,1) and returns the safety factor array.
  Docstring warns q is "problematic" at psinorm = 0 (axis) and 1 (separatrix) -> we evaluate on a
  grid in [0.05, 0.95] and report q95 (at 0.95), qmin (over the grid), q05 (axis proxy).

The forward solve here is forward-only (~5 s) vs the full label (~25 s at 80 modes).
"""
import numpy as np

import phase15_lib as L

# q-profile sampling grid: strictly inside (0,1); avoid the axis/separatrix where q is unreliable.
# np.linspace(0.05, 0.95, 19) -> step 0.05, so 0.05 is index 0 and 0.95 is index -1 (exact grid
# points, no interpolation). NOTE: eq.q() must be called with a length>1 array -- a length-1 array
# trips a removed np.asscalar() in freegs4e under numpy 1.26 -- so we evaluate the whole grid once.
PSIN_GRID = np.linspace(0.05, 0.95, 19)
I95 = 18      # index of psinorm=0.95 in PSIN_GRID
I05 = 0       # index of psinorm=0.05 in PSIN_GRID


def _forward_solve_eq(tok, active_currents, paxis, Ip, fvac, alpha_m, alpha_n,
                      nx=L.NX, ny=L.NY, fwd_tol=L.FWD_TOL):
    """Replicate phase15_lib.forward_label's crash-proofed FORWARD solve VERBATIM and return the
    converged eq (which forward_label discards). Deterministic given controls + locked protocol,
    so the eq is identical to the one the stored label was computed on."""
    from freegsnke import equilibrium_update, GSstaticsolver
    from freegsnke.jtor_update import ConstrainPaxisIp

    eq = equilibrium_update.Equilibrium(tokamak=tok, nx=nx, ny=ny, **L.GRID)
    profiles = ConstrainPaxisIp(eq=eq, paxis=float(paxis), Ip=float(Ip), fvac=float(fvac),
                                alpha_m=float(alpha_m), alpha_n=float(alpha_n))
    solver = GSstaticsolver.NKGSsolver(eq=eq)
    eq.tokamak.set_all_coil_currents(np.asarray(active_currents, dtype=float))

    ok = False
    for step in (2.5, 1.5, 1.0):
        try:
            eq.plasma_psi = eq.create_psi_plasma_default(adaptive_centre=True)
            with np.errstate(divide="raise", invalid="raise", over="raise"):
                solver.forward_solve(eq=eq, profiles=profiles, target_relative_tolerance=fwd_tol,
                                     max_solving_iterations=120, step_size=step,
                                     verbose=False, suppress=True)
            if (solver.relative_change <= 10 * fwd_tol
                    and np.all(np.isfinite(eq.plasma_psi))):
                ok = True
                break
        except Exception:
            continue
    if not ok:
        raise RuntimeError(f"forward solve did not converge (rel={getattr(solver,'relative_change',float('nan')):.1e})")
    if bool(getattr(eq, "flag_limiter", True)):
        raise RuntimeError("limited plasma (not diverted)")
    if float(eq.intersectsWall()):
        raise RuntimeError("plasma intersects wall")
    return eq


def forward_q(tok, active_currents, paxis, Ip, fvac, alpha_m, alpha_n):
    """Forward-solve and extract q95 / qmin / q05 + a verification kappa. Raises on failure."""
    eq = _forward_solve_eq(tok, active_currents, paxis, Ip, fvac, alpha_m, alpha_n)
    qgrid = np.asarray(eq.q(PSIN_GRID), dtype=float)
    q95 = float(qgrid[I95])
    q05 = float(qgrid[I05])
    if not np.all(np.isfinite(qgrid)):
        raise RuntimeError("non-finite q profile")
    # verification kappa from the separatrix (must match the stored descriptor -> replay sanity)
    sep = np.asarray(eq.separatrix())
    R, Z = sep[:, 0], sep[:, 1]
    kappa = float((Z.max() - Z.min()) / (R.max() - R.min()))
    return dict(q95=q95, qmin=float(qgrid.min()), q05=q05,
                q_at_grid_min_psin=float(PSIN_GRID[int(np.argmin(qgrid))]),
                kappa_check=kappa)
