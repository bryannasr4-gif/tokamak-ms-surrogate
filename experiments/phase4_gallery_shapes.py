"""
phase4_gallery_shapes.py -- re-solve the chosen gallery panels (start shape + surrogate-optimized
design) at the locked 80-mode protocol and extract the LCFS (separatrix) outlines + key descriptors,
so the gallery figure shows REAL solver-confirmed before->after shapes at the SAME kappa. A handful
of solves; run after the gallery + after the solve pool is free.

The gallery Recorder stores the best point in PC-SCORE (x) space (the field named best_u holds x);
we reconstruct the DesignSpace and map x->controls->equilibrium to get the separatrix. Saves
data/phase4_gallery_shapes.json.
"""
import json
import os
import sys

import numpy as np

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.getcwd(), "experiments"))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import phase15_lib as L
import phase2_dim_lib as DL
from freegsnke import equilibrium_update, GSstaticsolver
from freegsnke.jtor_update import ConstrainPaxisIp


def solve_lcfs(tok, u):
    """Forward-solve controls u and return separatrix (R,Z) + descriptors + m_s (80 modes)."""
    c = DL.ctrl_from_u(u)
    rec = L.forward_label(tok, c["active_currents"], c["paxis"], c["Ip"], c["fvac"],
                          c["alpha_m"], c["alpha_n"], fix_n_modes=80)
    # re-solve once more (forward only) to grab the separatrix object cleanly
    eq = equilibrium_update.Equilibrium(tokamak=tok, nx=L.NX, ny=L.NY, **L.GRID)
    prof = ConstrainPaxisIp(eq=eq, paxis=c["paxis"], Ip=c["Ip"], fvac=c["fvac"],
                            alpha_m=c["alpha_m"], alpha_n=c["alpha_n"])
    solver = GSstaticsolver.NKGSsolver(eq=eq)
    eq.tokamak.set_all_coil_currents(np.asarray(c["active_currents"], float))
    eq.plasma_psi = eq.create_psi_plasma_default(adaptive_centre=True)
    with np.errstate(divide="raise", invalid="raise", over="raise"):
        solver.forward_solve(eq=eq, profiles=prof, target_relative_tolerance=L.FWD_TOL,
                             max_solving_iterations=120, step_size=1.5, verbose=False, suppress=True)
    sep = np.asarray(eq.separatrix())
    return dict(R=sep[:, 0].tolist(), Z=sep[:, 1].tolist(), m_s=float(rec["m_s"]),
                kappa=float(rec["kappa"]), delta=float(rec["delta"]),
                sq_uo=float(rec.get("sq_uo", np.nan)), li=float(rec.get("li", np.nan)),
                betap=float(rec.get("betap", np.nan)), gap_outer=float(rec.get("gap_outer", np.nan)))


def main():
    summ = json.load(open("data/phase4_gallery_summary.json"))
    res = {r["start_i"]: r for r in json.load(open("data/phase4_gallery_results.json"))["recs"]
           if r["method"] == "surrogate"}
    S = json.load(open("data/phase25_kappa_setup.json"))
    mu = np.array(S["mu"]); std = np.array(S["std"]); V = np.array(S["V"])
    lo = np.array(S["box_lo"]); hi = np.array(S["box_hi"])
    tok = L.load_machine()

    # limiter outline for context
    limR, limZ = L._limiter_RZ(tok)
    out = dict(limiter=dict(R=limR.tolist(), Z=limZ.tolist()), panels=[])
    for label, si in summ["gallery_panels"]:
        s = S["starts"][si]
        ds = DL.DesignSpace(mu, std, V, lo, hi, np.array(s["u"]), 12)
        best_x = np.array(res[si]["best_u"])              # stored in PC-score (x) space
        u_start = ds.u_of_x(ds.x0)
        u_best = ds.u_of_x(best_x)
        try:
            start_shape = solve_lcfs(tok, u_start)
            best_shape = solve_lcfs(tok, u_best)
        except Exception as e:
            print(f"panel {label} (start {si}) solve FAILED: {e}", flush=True)
            continue
        out["panels"].append(dict(label=label, start_i=si, regime=s["regime"],
                                  start=start_shape, best=best_shape,
                                  gain=res[si]["gain"]))
        print(f"panel {label} start{si} ({s['regime']}): m_s {start_shape['m_s']:.3f} -> "
              f"{best_shape['m_s']:.3f}  kappa {start_shape['kappa']:.3f} -> {best_shape['kappa']:.3f}",
              flush=True)
    json.dump(out, open("data/phase4_gallery_shapes.json", "w"))
    print("Saved data/phase4_gallery_shapes.json")


if __name__ == "__main__":
    main()
