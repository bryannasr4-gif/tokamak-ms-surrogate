"""
phase15_probe.py -- de-risk the FORWARD-sampling path before building the engine.

Verifies, end to end, on the serialized Phase-0 machine at the locked protocol:
  1. An INVERSE solve at the baseline shape -> capture the converged coil-current vector
     I_ref (this anchors the forward sampling distribution in the diverted-ST regime).
  2. A pure FORWARD solve (fixed coils = I_ref, constrain=None) reproduces the SAME
     equilibrium + m_s (cross-check that forward and inverse agree at the anchor).
  3. Discover/print the full shape-descriptor API actually available on the converged eq
     (elongation, triangularity, squareness, gaps, li, betap, ...) + diverted/limiter flag.
  4. Time both solves. Confirm pandas/pyarrow are importable.

Run thread-pinned: OMP/OPENBLAS/MKL/NUMEXPR/VECLIB = 1 (set in the launch command).
"""
import os
import sys
import time

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "experiments"))
import phase0_lib as P0

SER = os.path.join(ROOT, "machine_configs", "MAST-U", "serialized_tokamak.pkl")


def descriptors(eq):
    """Probe every plausible shape-descriptor accessor; report value or the error."""
    out = {}
    sep = np.asarray(eq.separatrix())
    R, Z = sep[:, 0], sep[:, 1]
    out["n_sep_pts"] = len(sep)
    out["kappa_from_sep"] = float((Z.max() - Z.min()) / (R.max() - R.min()))
    out["Rmax_sep"], out["Rmin_sep"] = float(R.max()), float(R.min())
    out["Zmax_sep"], out["Zmin_sep"] = float(Z.max()), float(Z.min())
    for name, fn in [
        ("geometricAxis", lambda: np.asarray(eq.geometricAxis()).tolist()),
        ("magneticAxis", lambda: np.asarray(eq.magneticAxis()).tolist()),
        ("minorRadius", eq.minorRadius),
        ("aspectRatio", eq.aspectRatio),
        ("triangularity", eq.triangularity),
        ("triangularity_upper", eq.triangularity_upper),
        ("triangularity_lower", eq.triangularity_lower),
        ("squareness", eq.squareness),
        ("plasmaCurrent", eq.plasmaCurrent),
        ("internalInductance2", lambda: eq.internalInductance2()),
        ("internalInductance3", lambda: eq.internalInductance3()),
        ("poloidalBeta2", lambda: eq.poloidalBeta2()),
        ("poloidalBeta3", lambda: eq.poloidalBeta3()),
        ("intersectsWall", eq.intersectsWall),
    ]:
        try:
            v = fn()
            out[name] = v if isinstance(v, (list, tuple)) else float(np.real(v))
        except Exception as e:
            out[name] = f"ERR {type(e).__name__}: {e}"
    # topology flags
    for attr in ("flag_limiter", "solved"):
        out[attr] = getattr(eq, attr, "MISSING")
    try:
        out["xpt"] = np.asarray(eq.xpt).tolist()
    except Exception as e:
        out["xpt"] = f"ERR {e}"
    return out


def main():
    print(f"BLAS threads OMP={os.environ.get('OMP_NUM_THREADS')}")
    from freegsnke import equilibrium_update, GSstaticsolver, nonlinear_solve
    from freegsnke.inverse import Inverse_optimizer
    from freegsnke.jtor_update import ConstrainPaxisIp

    tok = P0.load_machine(SER)
    print("Loaded serialized machine.")

    # --- 1. INVERSE solve at baseline -> reference coil currents -------------
    zscale, dR = 1.00, 0.0
    eq = equilibrium_update.Equilibrium(tokamak=tok, Rmin=0.1, Rmax=2.0, Zmin=-2.2, Zmax=2.2, nx=65, ny=65)
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
    t0 = time.time()
    solver.solve(eq=eq, profiles=profiles, constrain=constrain, target_relative_tolerance=1e-6,
                 target_relative_psit_update=1e-3, verbose=False, l2_reg=np.array([1e-12] * 10 + [1e-6]))
    t_inv = time.time() - t0
    I_ref = eq.tokamak.getCurrentsVec().copy()
    labels = list(eq.tokamak.getCurrents().keys())
    print(f"\n[INVERSE] {t_inv:.1f}s  -> coil currents (A):")
    for lab, cur in zip(labels, I_ref):
        print(f"    {lab:10s} {cur:12.1f}")

    t0 = time.time()
    nls = nonlinear_solve.nl_solver(eq=eq, profiles=profiles, GSStaticSolver=solver,
                                    plasma_resistivity=1e-6, fix_n_vessel_modes=40, verbose=False)
    t_lin = time.time() - t0
    ms_inv = float(np.ravel(np.asarray(nls.linearised_sol.stability_margin).real).max())
    print(f"[INVERSE] linearization {t_lin:.1f}s  m_s={ms_inv:.5f}")
    print("[INVERSE] descriptors:")
    for k, v in descriptors(eq).items():
        print(f"    {k:22s} {v}")

    # --- 2. FORWARD solve with fixed coils = I_ref --------------------------
    eqf = equilibrium_update.Equilibrium(tokamak=tok, Rmin=0.1, Rmax=2.0, Zmin=-2.2, Zmax=2.2, nx=65, ny=65)
    proff = ConstrainPaxisIp(eq=eqf, paxis=8e3, Ip=6e5, fvac=0.5, alpha_m=1.8, alpha_n=1.2)
    solverf = GSstaticsolver.NKGSsolver(eq=eqf)
    eqf.tokamak.set_all_coil_currents(I_ref)
    t0 = time.time()
    solverf.forward_solve(eq=eqf, profiles=proff, target_relative_tolerance=1e-8, verbose=False, suppress=True)
    t_fwd = time.time() - t0
    print(f"\n[FORWARD] {t_fwd:.1f}s  relative_change={getattr(solverf,'relative_change',np.nan):.2e}")
    t0 = time.time()
    nlsf = nonlinear_solve.nl_solver(eq=eqf, profiles=proff, GSStaticSolver=solverf,
                                     plasma_resistivity=1e-6, fix_n_vessel_modes=40, verbose=False)
    t_linf = time.time() - t0
    ms_fwd = float(np.ravel(np.asarray(nlsf.linearised_sol.stability_margin).real).max())
    print(f"[FORWARD] linearization {t_linf:.1f}s  m_s={ms_fwd:.5f}")
    print(f"[CROSS-CHECK] m_s inverse={ms_inv:.5f}  forward={ms_fwd:.5f}  "
          f"rel.diff={abs(ms_fwd-ms_inv)/ms_inv*100:.3f}%")
    print("[FORWARD] descriptors:")
    for k, v in descriptors(eqf).items():
        print(f"    {k:22s} {v}")

    # --- 4. deps ------------------------------------------------------------
    try:
        import pandas, pyarrow
        print(f"\npandas {pandas.__version__}, pyarrow {pyarrow.__version__} OK")
    except Exception as e:
        print(f"\npandas/pyarrow IMPORT ERROR: {e}")
    print(f"\nTIMINGS: inverse={t_inv:.1f}s lin={t_lin:.1f}s | forward={t_fwd:.1f}s lin={t_linf:.1f}s")

    # --- archive the forward-vs-inverse cross-check (the headline 0.000% claim) ---
    import json
    out = dict(
        baseline=dict(zscale=zscale, dR=dR, grid=[65, 65], fix_n_vessel_modes=40, OMP=1),
        m_s_inverse=ms_inv, m_s_forward=ms_fwd,
        rel_diff_pct=abs(ms_fwd - ms_inv) / ms_inv * 100.0,
        coil_currents_active=dict(zip(labels[:12], [float(x) for x in I_ref[:12]])),
        timings_s=dict(inverse=t_inv, inv_lin=t_lin, forward=t_fwd, fwd_lin=t_linf),
        note="Forward solve with fixed coils = inverse-anchor currents reproduces the inverse m_s; "
             "evidence for the dataset_v1 forward-sampling validity (DATASET.md / RESULTS.md Phase 1.5).",
    )
    with open(os.path.join(ROOT, "data", "phase15_probe.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved data/phase15_probe.json  (m_s inv={ms_inv:.6f} fwd={ms_fwd:.6f} "
          f"rel.diff={out['rel_diff_pct']:.4f}%)")


if __name__ == "__main__":
    main()
