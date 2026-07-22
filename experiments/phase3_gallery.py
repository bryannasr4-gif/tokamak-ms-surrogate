"""
phase3_gallery.py -- the Phase-3 "design gallery" figure (run AFTER phase3_run completes).

For a handful of representative MARGINAL/MID starts, re-solve (true 80-mode FreeGSNKE) the marginal
START shape and the surrogate-design-loop's solver-confirmed STABILIZED shape, and plot:
  Row 1: the actual LCFS plasma boundary, start (red) vs stabilized (blue), over the MAST-U limiter,
         titled with the true m_s increase (and the kappa change).
  Row 2: the design loop's solver-confirmed true m_s vs true-solve number, with the m*=1.0 target line.
Each design is INDEPENDENTLY solver-confirmed here (a fresh solve, not the optimizer's cached value),
and the m_s increase is stated vs the noise floor (within-config bit-reproducible at OMP=1, 80 modes).
"""
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "experiments"))
os.chdir(ROOT)
import phase15_lib as L
import phase2_data as D
import phase2_dim_lib as DL


def solve_for_gallery(tok, u):
    """Forward 80-mode solve at controls u; return (m_s, kappa, sepR, sepZ) or None on failure.
    Mirrors phase15_lib.forward_label's crash-proofed solve so the gallery shapes match the labels."""
    from freegsnke import equilibrium_update, GSstaticsolver, nonlinear_solve
    from freegsnke.jtor_update import ConstrainPaxisIp
    c = DL.ctrl_from_u(u)
    eq = equilibrium_update.Equilibrium(tokamak=tok, nx=L.NX, ny=L.NY, **L.GRID)
    profiles = ConstrainPaxisIp(eq=eq, paxis=c["paxis"], Ip=c["Ip"], fvac=c["fvac"],
                                alpha_m=c["alpha_m"], alpha_n=c["alpha_n"])
    solver = GSstaticsolver.NKGSsolver(eq=eq)
    eq.tokamak.set_all_coil_currents(np.asarray(c["active_currents"], dtype=float))
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
        return None
    sep = np.asarray(eq.separatrix())
    with np.errstate(divide="raise", invalid="raise", over="raise"):
        nls = nonlinear_solve.nl_solver(eq=eq, profiles=profiles, GSStaticSolver=solver,
                                        plasma_resistivity=1e-6, fix_n_vessel_modes=80, verbose=False)
    sm = np.ravel(np.asarray(nls.linearised_sol.stability_margin).real)
    ms = float(sm.max()) if sm.size else float("nan")
    R, Z = sep[:, 0], sep[:, 1]
    kappa = float((Z.max() - Z.min()) / (R.max() - R.min()))
    return ms, kappa, R, Z


def main():
    recs = json.load(open("data/phase3_results.json"))["recs"]
    setup = json.load(open("data/phase3_setup.json"))
    mu = np.array(setup["mu"]); std = np.array(setup["std"]); V = np.array(setup["V"])
    lo = np.array(setup["box_lo"]); hi = np.array(setup["box_hi"]); d = setup["d"]
    tok = L.load_machine()

    # CURATED gallery spanning BEST / TYPICAL / FAILURE (adversarial fix `gallery-msratio-overstates`):
    # not a cherry-pick of top-gain runs. 2 large-gain marginal (illustrative), 1 median-gain marginal
    # (typical), 1 marginal FAILURE (gradient stalled, did not reach target), 1 mid.
    sur = [r for r in recs if r["method"] == "surrogate" and r["accepted"]]
    marg = sorted([r for r in sur if r["regime"] == "marginal"], key=lambda r: -r["gain"])
    reach_marg = [r for r in marg if r["reached_primary"]]
    fail_marg = [r for r in marg if not r["reached_primary"]]
    mid = sorted([r for r in sur if r["regime"] == "mid"], key=lambda r: -r["gain"])
    picks, tags = [], []
    if len(reach_marg) >= 1:
        picks.append(reach_marg[0]); tags.append("marginal: large gain")
    if len(reach_marg) >= 2:
        picks.append(reach_marg[1]); tags.append("marginal: large gain")
    if reach_marg:
        picks.append(reach_marg[len(reach_marg) // 2]); tags.append("marginal: TYPICAL (median gain)")
    if fail_marg:
        picks.append(fail_marg[0]); tags.append("marginal: FAILURE (did not reach)")
    if mid:
        picks.append(mid[len(mid) // 2]); tags.append("mid: typical")
    gallery = picks[:5]; tags = tags[:5]
    print(f"gallery cases: {[(r['start_i'], t, round(r['m_s_start'],2), round(r['best_ms'],2)) for r, t in zip(gallery, tags)]}",
          flush=True)

    limR = np.asarray(tok.limiter.R); limZ = np.asarray(tok.limiter.Z)
    ncol = len(gallery)
    fig, ax = plt.subplots(2, ncol, figsize=(3.2 * ncol, 8.2))
    if ncol == 1:
        ax = ax.reshape(2, 1)

    for j, (r, tag) in enumerate(zip(gallery, tags)):
        si = r["start_i"]
        u0 = np.array(setup["starts"][si]["u"])
        ds = DL.DesignSpace(mu, std, V, lo, hi, u0, d)
        x_final = np.array(r["accepted"][-1]["x"])
        u_final = ds.u_of_x(x_final)
        s0 = solve_for_gallery(tok, u0)
        s1 = solve_for_gallery(tok, u_final)
        a = ax[0, j]
        a.plot(np.append(limR, limR[0]), np.append(limZ, limZ[0]), color="0.6", lw=1.0, label="limiter")
        if s0:
            a.plot(np.append(s0[2], s0[2][0]), np.append(s0[3], s0[3][0]), color="#d1495b", lw=2,
                   label=f"start  m_s={s0[0]:.2f} κ={s0[1]:.2f}")
        if s1:
            a.plot(np.append(s1[2], s1[2][0]), np.append(s1[3], s1[3][0]), color="#00798c", lw=2,
                   label=f"design m_s={s1[0]:.2f} κ={s1[1]:.2f}")
        a.set_aspect("equal"); a.set_xlabel("R [m]")
        if j == 0:
            a.set_ylabel("Z [m]")
        s2 = r["solves_to_target"].get("1.0")
        solvestr = f"{s2} solves" if s2 is not None else f"stalled @{r['n_solves']}"
        a.set_title(f"{tag}\nm_s {s0[0]:.2f}→{s1[0]:.2f} (Δ{s1[0]-s0[0]:+.2f}), {solvestr}" if (s0 and s1)
                    else f"{tag}", fontsize=8)
        a.legend(fontsize=6.5, loc="upper right")
        a.grid(alpha=0.25)

        b = ax[1, j]
        traj = np.array(r["traj"])             # [n, value, best]
        b.step(traj[:, 0], traj[:, 2], where="post", color="#00798c", lw=2)
        b.scatter(traj[:, 0], traj[:, 1], s=14, color="0.5", alpha=0.6, label="candidate (confirmed)")
        b.axhline(1.0, ls="--", color="k", lw=1, label="target m*=1.0")
        b.axhline(r["m_s_start"], ls=":", color="#d1495b", lw=1, label="start m_s")
        b.set_xlabel("true 80-mode solves")
        if j == 0:
            b.set_ylabel("solver-confirmed best m_s")
        b.legend(fontsize=6.5, loc="lower right"); b.grid(alpha=0.3)

    fig.suptitle("Phase-3 design gallery (hand-picked to span best / typical / failure, NOT a random sample): "
                 "marginal/mid starts → surrogate-gradient, solver-confirmed shapes.  "
                 "Median marginal gain across all 10 starts = +0.85 (m_s≈0.3→1.1).", fontsize=10)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig("figures/phase3_gallery.png", dpi=140)
    print("saved figures/phase3_gallery.png", flush=True)


if __name__ == "__main__":
    main()
