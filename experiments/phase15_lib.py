"""
phase15_lib.py -- FORWARD-sampling data engine (Phase 1.5).

Root-cause fix for the Phase-0 noise floor: instead of the ill-conditioned INVERSE solve
(target shape -> coil currents) at data-generation time, we sample the CONTROL inputs
(active coil currents + plasma-profile parameters) and do a well-conditioned FORWARD
free-boundary GS solve (constrain=None). Forward labels are bit-reproducible at the locked
protocol (verified in phase15_probe: forward m_s reproduces the inverse anchor to 0.000%).

Pipeline per sample:
  control (12 active currents + paxis,Ip,fvac,alpha_m,alpha_n)
    -> forward_solve (fixed coils)                          [well-conditioned, smooth]
    -> keep iff CONVERGED and DIVERTED (flag_limiter False) [reject limited / failed]
    -> shape descriptors (kappa, delta, squareness, R_geo, a, gaps, li, betap, ...)
    -> labels (m_s, gamma, Leuer, instability timescale, n_unstable)

The sampling distribution is anchored to the diverted-ST manifold by a one-time coarse set
of INVERSE solves (build_anchors): those anchors only CENTER the forward distribution; every
labelled sample is a clean forward solve. We then interpolate between anchors + add correlated
current jitter + independently vary profile params to fill a continuous, well-conditioned
volume of control space (incl. shapes the inverse solver would not target).

BLAS threads MUST be pinned to 1 in the environment BEFORE importing numpy/this module.
Reuses the serialized Phase-0 machine (phase0_lib.load_machine) and the locked protocol.
"""
import os
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SER_MACHINE = os.path.join(ROOT, "machine_configs", "MAST-U", "serialized_tokamak.pkl")

# --- LOCKED NUMERICAL PROTOCOL (PHASE0_PROTOCOL.md) ---------------------------
NX, NY = 65, 65
FIX_N_MODES = 40
FWD_TOL = 1e-8          # forward GS tolerance; forward solve is well-conditioned so a tight
                        # tol is cheap and removes solver-residual scatter (no inverse to chase)
GRID = dict(Rmin=0.1, Rmax=2.0, Zmin=-2.2, Zmax=2.2)

# The 12 MAST-U active coils (the rest of the machine current_vec is passive structure = 0).
ACTIVE_COILS = ["Solenoid", "PX", "D1", "D2", "D3", "Dp", "D5", "D6", "D7", "P4", "P5", "P6"]
N_ACTIVE = len(ACTIVE_COILS)

# Nominal profile params (Phase-1 / Phase-0 continuity).
NOM = dict(paxis=8e3, Ip=6e5, fvac=0.5, alpha_m=1.8, alpha_n=1.2)


def load_machine():
    import phase0_lib as P0
    return P0.load_machine(SER_MACHINE)


# ---------------------------------------------------------------- descriptors
def _limiter_RZ(tok):
    lim = tok.limiter
    return np.asarray(lim.R), np.asarray(lim.Z)


def _seg_dist(points, polyR, polyZ):
    """Min distance from each point (N,2) to the closed polyline (limiter), point-to-segment."""
    P = np.asarray(points)
    A = np.stack([polyR, polyZ], axis=1)
    B = np.roll(A, -1, axis=0)          # next vertex (closed loop)
    AB = B - A                          # (M,2)
    AB2 = np.einsum("md,md->m", AB, AB) + 1e-30
    # for each point p, project onto each segment
    AP = P[:, None, :] - A[None, :, :]              # (N,M,2)
    t = np.clip(np.einsum("nmd,md->nm", AP, AB) / AB2, 0.0, 1.0)
    proj = A[None] + t[:, :, None] * AB[None]       # (N,M,2)
    d = np.linalg.norm(P[:, None, :] - proj, axis=2)  # (N,M)
    return d.min(axis=1)                            # (N,)


def descriptors(eq, tok):
    """Shape descriptors of a converged equilibrium. Raises on degenerate separatrix."""
    sep = np.asarray(eq.separatrix())
    R, Z = sep[:, 0], sep[:, 1]
    if len(sep) < 40 or not np.all(np.isfinite(sep)):
        raise ValueError("degenerate separatrix")
    Rmax, Rmin = float(R.max()), float(R.min())
    Zmax, Zmin = float(Z.max()), float(Z.min())
    a = (Rmax - Rmin) / 2.0
    Rgeo = (Rmax + Rmin) / 2.0
    kappa = (Zmax - Zmin) / (Rmax - Rmin)

    sq = eq.squareness()                       # (upper-out, upper-in, lower-out, lower-in)
    limR, limZ = _limiter_RZ(tok)
    d = _seg_dist(sep, limR, limZ)             # per-separatrix-point gap to limiter
    inb = R < Rgeo
    upp = Z > 0.0
    gap_min = float(d.min())
    gap_inner = float(d[inb].min()) if inb.any() else gap_min
    gap_outer = float(d[~inb].min()) if (~inb).any() else gap_min
    gap_top = float(d[upp].min()) if upp.any() else gap_min
    gap_bot = float(d[~upp].min()) if (~upp).any() else gap_min

    rec = dict(
        kappa=float(kappa),
        delta=float(eq.triangularity()),
        delta_upper=float(eq.triangularity_upper()),
        delta_lower=float(eq.triangularity_lower()),
        sq_uo=float(sq[0]), sq_ui=float(sq[1]), sq_lo=float(sq[2]), sq_li=float(sq[3]),
        Rgeo=float(Rgeo), a=float(a),
        aspect=float(eq.aspectRatio()),
        Rmag=float(np.asarray(eq.magneticAxis())[0]),
        gap_min=gap_min, gap_inner=gap_inner, gap_outer=gap_outer,
        gap_top=gap_top, gap_bot=gap_bot,
        li=float(eq.internalInductance2()),
        betap=float(eq.poloidalBeta2()),
        Zaxis=float(np.asarray(eq.magneticAxis())[1]),
    )
    return rec


# ---------------------------------------------------------------- forward label
def forward_label(tok, active_currents, paxis, Ip, fvac, alpha_m, alpha_n,
                  fix_n_modes=FIX_N_MODES, nx=NX, ny=NY, fwd_tol=FWD_TOL,
                  with_linearisation=True):
    """One FORWARD-sampled labelled sample. Returns a result dict on success.

    Raises on non-convergence / degeneracy / limited plasma so the caller can record a
    failure and move on. `active_currents` is the length-12 active-coil vector (A).
    """
    from freegsnke import equilibrium_update, GSstaticsolver, nonlinear_solve
    from freegsnke.jtor_update import ConstrainPaxisIp

    eq = equilibrium_update.Equilibrium(tokamak=tok, nx=nx, ny=ny, **GRID)
    profiles = ConstrainPaxisIp(eq=eq, paxis=float(paxis), Ip=float(Ip), fvac=float(fvac),
                                alpha_m=float(alpha_m), alpha_n=float(alpha_n))
    solver = GSstaticsolver.NKGSsolver(eq=eq)
    eq.tokamak.set_all_coil_currents(np.asarray(active_currents, dtype=float))

    # Forward free-boundary solve (constrain=None). Escalate robustness on failure.
    #
    # CRASH-PROOFING (Phase-1.5 pilot lesson): a forward trajectory can drive the core plasma
    # current toward zero, so ConstrainPaxisIp's normalization hits Ip/I_R with I_R->0. The
    # resulting inf/NaN, if it reaches the native multigrid solver, HARD-CRASHES the process
    # (no Python traceback). We force numpy to RAISE on that divide so it surfaces as a
    # catchable FloatingPointError *before* any NaN propagates -- caught here -> clean reject.
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
        except Exception:        # incl. FloatingPointError from the errstate guard
            continue
    if not ok:
        raise RuntimeError(f"forward solve did not converge (rel={getattr(solver,'relative_change',float('nan')):.1e})")

    if bool(getattr(eq, "flag_limiter", True)):
        raise RuntimeError("limited plasma (not diverted)")
    if float(eq.intersectsWall()):
        raise RuntimeError("plasma intersects wall")

    rec = descriptors(eq, tok)
    rec.update(dict(
        paxis=float(paxis), Ip_target=float(Ip), fvac=float(fvac),
        alpha_m=float(alpha_m), alpha_n=float(alpha_n),
        Ip=float(eq.plasmaCurrent()),
        fwd_rel_change=float(solver.relative_change),
    ))
    for i, lab in enumerate(ACTIVE_COILS):
        rec[f"I_{lab}"] = float(active_currents[i])

    if with_linearisation:
        with np.errstate(divide="raise", invalid="raise", over="raise"):
            nls = nonlinear_solve.nl_solver(eq=eq, profiles=profiles, GSStaticSolver=solver,
                                            plasma_resistivity=1e-6, fix_n_vessel_modes=fix_n_modes,
                                            verbose=False)
        ls = nls.linearised_sol
        sm = np.ravel(np.asarray(ls.stability_margin).real)
        gr = np.ravel(np.asarray(ls.growth_rates).real)
        # m_s = the (single) positive Portone margin. For these diverted ST equilibria there is
        # exactly ONE positive eigenvalue (the n=0 vertical mode), so sm.max()==sm.min(); we store
        # n_positive_margins to assert that. If a future config (e.g. more retained modes) ever
        # yields >1 positive margin, revisit: the marginal (smallest positive) one is the physical
        # controllability boundary, not the max. Kept identical to phase0_lib.solve_equilibrium.
        rec.update(dict(
            m_s=float(sm.max()) if sm.size else float("nan"),
            n_positive_margins=int(sm.size),
            gamma=float(gr.max()) if gr.size else 0.0,
            n_unstable=int(gr.size),
            tau_inst=float(1.0 / gr.max()) if gr.size and gr.max() > 0 else float("inf"),
            leuer=float(getattr(nls, "Leuer_metals_stab_over_active_destab", np.nan)),
        ))
    return rec


# ---------------------------------------------------------------- anchors
def _inverse_currents(tok, zscale, dR, nx=NX, ny=NY):
    """INVERSE-solve the Phase-1 (zscale, dR) target shape and return the converged active-coil
    currents + kappa. Used ONCE to seed the forward sampler (not as a label)."""
    from freegsnke import equilibrium_update, GSstaticsolver
    from freegsnke.inverse import Inverse_optimizer
    from freegsnke.jtor_update import ConstrainPaxisIp

    eq = equilibrium_update.Equilibrium(tokamak=tok, nx=nx, ny=ny, **GRID)
    profiles = ConstrainPaxisIp(eq=eq, paxis=NOM["paxis"], Ip=NOM["Ip"], fvac=NOM["fvac"],
                                alpha_m=NOM["alpha_m"], alpha_n=NOM["alpha_n"])
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
    solver.solve(eq=eq, profiles=profiles, constrain=constrain, target_relative_tolerance=1e-6,
                 target_relative_psit_update=1e-3, verbose=False,
                 l2_reg=np.array([1e-12] * 10 + [1e-6]))
    I = eq.tokamak.getCurrentsVec()[:N_ACTIVE].copy()
    sep = np.asarray(eq.separatrix()); R, Z = sep[:, 0], sep[:, 1]
    kappa = float((Z.max() - Z.min()) / (R.max() - R.min()))
    return I, kappa


def build_anchors(tok, zscales, dRs, verbose=True):
    """One-time coarse INVERSE solves over a (zscale, dR) grid -> anchor active-current vectors
    that CENTER the forward sampling distribution on the diverted-ST manifold. Anchors are NOT
    labels (they only seed the sampler); per-sample forward solves are the clean labels.
    """
    anchors = []
    for z in zscales:
        for d in dRs:
            try:
                I, kappa = _inverse_currents(tok, float(z), float(d))
            except Exception as e:
                if verbose:
                    print(f"  anchor z={z:.3f} dR={d:+.3f} FAILED {type(e).__name__}: {e}", flush=True)
                continue
            anchors.append(dict(zscale=float(z), dR=float(d), I=I, kappa=kappa))
            if verbose:
                print(f"  anchor z={z:.3f} dR={d:+.3f} kappa={kappa:.3f}  "
                      f"I[P4,P5]={I[9]:.0f},{I[10]:.0f}", flush=True)
    return anchors


# ---------------------------------------------------------------- sampler
def sample_control(rng, anchors, jitter=0.03, scale_sigma=0.035):
    """Draw one control vector: interpolate two random anchors, add correlated current jitter,
    and independently sample profile params. Ranges are deliberately conservative -- the bulk
    of the shape/m_s variety comes from interpolating anchors ACROSS the kappa axis, while the
    current jitter + profile params fill the local volume; wide profile ranges mostly add
    degenerate (near-zero-core) forward trajectories, so they are kept moderate."""
    n = len(anchors)
    i, j = rng.choice(n, size=2, replace=(n < 2))
    t = rng.uniform(0.0, 1.0)
    base_I = (1 - t) * anchors[i]["I"] + t * anchors[j]["I"]
    s = float(np.exp(rng.normal(0.0, scale_sigma)))         # global current scale (log-normal)
    per = rng.normal(0.0, jitter, size=N_ACTIVE)            # per-coil relative jitter
    I = base_I * s * (1.0 + per)
    I[0] = 5000.0                                           # Solenoid fixed (as in the inverse setup)
    return dict(
        active_currents=I,
        paxis=float(rng.uniform(6.0e3, 1.0e4)),
        Ip=float(NOM["Ip"] * s * rng.uniform(0.92, 1.08)),
        fvac=float(rng.uniform(0.42, 0.58)),
        alpha_m=float(rng.uniform(1.5, 2.1)),     # profile peakedness -> drives li/betap breadth
        alpha_n=float(rng.uniform(1.05, 1.45)),   # (crash-guarded, so wide ranges are safe)
    )
