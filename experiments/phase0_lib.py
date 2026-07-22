"""
phase0_lib.py -- canonical protocol-locked FreeGSNKE engine for Phase 0 (and beyond).

Single source of truth for: building / serializing the MAST-U machine, and solving one
shape -> {m_s, gamma, geometry, linearisation matrices} at a PARAMETERIZED numerical
protocol (grid, retained passive modes, inverse-solve tolerance).

The shape parameterization (zscale, dR), coil currents, profiles and isoflux targets are
IDENTICAL to experiments/phase1_generate.py so Phase-0 numbers compare directly to Phase 1.
Phase 0 sweeps the *numerical* protocol (nx, ny, fix_n_modes, inv_tol, BLAS threads) around
that fixed shape physics to quantify the m_s/gamma reproducibility floor.

BLAS threads MUST be pinned BEFORE importing this module (set OMP_NUM_THREADS etc. in env).
"""
import os
import pickle

import numpy as np

# Project root, regardless of CWD.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MC = os.path.join(ROOT, "machine_configs", "MAST-U")

# --- LOCKED PROTOCOL DEFAULTS (the values Phase 0 will confirm/adjust) ---------
DEF_NX, DEF_NY = 65, 65
DEF_FIX_N_MODES = 40
DEF_INV_TOL = 1e-6
DEF_PSIT_UPDATE = 1e-3

# Canonical "baseline" shape used for the cross-check and noise floor anchor.
CANON = dict(zscale=1.00, dR=0.0)


def build_tokamak():
    """Build the MAST-U_like machine from the open tutorial pickles.

    NOTE (audit 4.3): freegsnke/refine_passive.py uses a module-level LatinHypercube(seed=42)
    whose state advances per call, so building the machine TWICE in one process diverges.
    Build ONCE per process (or load the serialized machine, below).
    """
    from freegsnke import build_machine
    return build_machine.tokamak(
        active_coils_path=f"{MC}/MAST-U_like_active_coils.pickle",
        passive_coils_path=f"{MC}/MAST-U_like_passive_coils.pickle",
        limiter_path=f"{MC}/MAST-U_like_limiter.pickle",
        wall_path=f"{MC}/MAST-U_like_wall.pickle",
    )


def save_machine(path):
    """Build once and pickle the full tokamak object so every downstream run reuses the
    IDENTICAL machine (matrices R, M, passive refinement) -- no rebuild, no LHS-state drift."""
    tok = build_tokamak()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(tok, f, protocol=pickle.HIGHEST_PROTOCOL)
    return tok


def load_machine(path):
    """Load the serialized tokamak. Falls back to building if the pickle is missing."""
    if not os.path.exists(path):
        return build_machine_or_raise(path)
    with open(path, "rb") as f:
        return pickle.load(f)


def build_machine_or_raise(path):
    raise FileNotFoundError(
        f"Serialized machine not found at {path}. Run phase0_serialize_machine.py first."
    )


def solve_equilibrium(tokamak, zscale=1.0, dR=0.0, nx=DEF_NX, ny=DEF_NY,
                      fix_n_modes=DEF_FIX_N_MODES, inv_tol=DEF_INV_TOL,
                      psit_update=DEF_PSIT_UPDATE, return_nls=False):
    """Inverse-solve shape (zscale, dR) at the given numerical protocol, then build the
    linearisation. Returns a result dict; if return_nls, also returns the nl_solver so the
    caller can access the raw linearisation matrices (M0matrix, dMmatrix, Mmatrix).

    Physics (shape targets, coils, profiles) is IDENTICAL to phase1_generate.evaluate.
    """
    from freegsnke import equilibrium_update, GSstaticsolver, nonlinear_solve
    from freegsnke.inverse import Inverse_optimizer
    from freegsnke.jtor_update import ConstrainPaxisIp

    eq = equilibrium_update.Equilibrium(tokamak=tokamak, Rmin=0.1, Rmax=2.0,
                                        Zmin=-2.2, Zmax=2.2, nx=nx, ny=ny)
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
    constrain = Inverse_optimizer(null_points=null_points, isoflux_set=isoflux_set,
                                  coil_current_limits=lims)
    constrain.mu_coils = 1e5

    solver.solve(eq=eq, profiles=profiles, constrain=constrain,
                 target_relative_tolerance=inv_tol, target_relative_psit_update=psit_update,
                 verbose=False, l2_reg=np.array([1e-12] * 10 + [1e-6]))

    b = np.asarray(eq.separatrix()); R, Z = b[:, 0], b[:, 1]
    kappa = float((Z.max() - Z.min()) / (R.max() - R.min()))

    nls = nonlinear_solve.nl_solver(eq=eq, profiles=profiles, GSStaticSolver=solver,
                                    plasma_resistivity=1e-6, fix_n_vessel_modes=fix_n_modes,
                                    verbose=False)
    sm = np.ravel(np.asarray(nls.linearised_sol.stability_margin).real)
    gr = np.ravel(np.asarray(nls.linearised_sol.growth_rates).real)
    res = dict(
        zscale=float(zscale), dR=float(dR), nx=int(nx), ny=int(ny),
        fix_n_modes=int(fix_n_modes), inv_tol=float(inv_tol),
        kappa=kappa, Rgeo=float(eq.geometricAxis()[0]), minor=float(eq.minorRadius()),
        aspect=float(eq.aspectRatio()), tri=float(eq.triangularity()),
        n_independent_vars=int(nls.linearised_sol.n_independent_vars),
        n_jacobian_cols=int(nls.dIydI.shape[1]),
        m_s=float(sm.max()) if sm.size else float("nan"),
        n_positive_margins=int(sm.size),
        gamma=float(gr.max()) if gr.size else 0.0,
        n_unstable=int(gr.size),
    )
    if return_nls:
        return res, nls
    return res


def independent_stability_margins(nls):
    """Independently recompute the Portone stability-margin eigenvalues from the converged
    linearisation, three algebraically-distinct ways, WITHOUT calling FreeGSNKE's
    calculate_stability_margin. Returns a dict of full spectra (complex) + diagnostics.

      L     = M0matrix[:n,:n]            (pure-metal inductance block, normal-mode basis)
      S     = -dMmatrix[:n,:n]           (plasma-mediated coupling block)
      Lstar = Mmatrix[:n,:n] = L - S     (plasma-modified metal block of the full M)

      method_A : eig(L^{-1} S) - 1           (FreeGSNKE's own form, recomputed)
      method_B : eig(-L^{-1} Lstar)          (the "-L^{-1}L*" form the scoping flagged)
      method_C : eig(S, L) - 1               (generalized eigenproblem; no explicit inverse)
    """
    from scipy.linalg import eig as geig

    ls = nls.linearised_sol
    n = int(ls.n_independent_vars)
    M0 = np.asarray(ls.M0matrix)
    dM = np.asarray(ls.dMmatrix)
    Mfull = np.asarray(ls.Mmatrix)
    L = M0[:n, :n]
    S = -dM[:n, :n]
    Lstar = Mfull[:n, :n]

    A = np.linalg.solve(L, S) - np.eye(n)
    ev_A = np.linalg.eigvals(A)
    ev_B = np.linalg.eigvals(-np.linalg.solve(L, Lstar))
    ev_C = geig(S, L, right=False) - 1.0  # generalized eigenvalues mu, margin = mu - 1

    def toppos(ev):
        r = np.real(ev)
        pos = np.sort(r[r > 0])[::-1]
        return pos

    return dict(
        n=n,
        identity_M0pdM_eq_M=float(np.max(np.abs((M0 + dM)[:n, :n] - Lstar))),
        identity_Lstar_eq_LmS=float(np.max(np.abs(Lstar - (L - S)))),
        ev_A=ev_A, ev_B=ev_B, ev_C=ev_C,
        max_imag_A=float(np.max(np.abs(np.imag(ev_A)))),
        pos_A=toppos(ev_A), pos_B=toppos(ev_B), pos_C=toppos(ev_C),
        reported=np.ravel(np.asarray(ls.stability_margin).real),
        all_reported=np.ravel(np.asarray(ls.all_stability_margins).real),
    )
