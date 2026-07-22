"""
device2_killgate.py -- the GO/NO-GO kill-gate for the genuine-second-device test (Device-C, the
higher-aspect-ratio machine from device2_build.py).

THE QUESTION: on a device where the aspect ratio is conventional, is kappa STILL the dominant m_s
lever? We forward-sample a few hundred diverted equilibria on Device-C and measure:
  * corr(kappa, log m_s)              -- MAST-U is -0.875; if |corr| is materially WEAKER (< ~0.75)
                                         kappa is de-dominated  => GO (the unconstrained learned-m_s
                                         vs reduce-kappa test is worth running).
  * per-lever m_s sensitivity         -- does a SECONDARY lever (squareness/gaps/li) rival kappa in
                                         |d log m_s / d (standardized descriptor)|? If yes => GO.
If kappa still dominates (|corr| ~ 0.8+, no rival lever) => NO-GO: report "kappa-dominance is a
property of tokamak vertical stability generally, not a MAST-U artifact" (a real, publishable
scoping result) and DO NOT spend a full pipeline reproducing the tie.

This reuses phase15_lib.forward_label (machine-agnostic) for labels. Two Device-C specifics:
  (1) the grid is EXPANDED (Device-C plasma sits at larger R) -- we override phase15_lib.GRID;
  (2) the inverse-solve isoflux/null TARGETS are shifted by the same R-transform as the machine.

NOTE FOR THE MAC SESSION: this is a STARTER. Getting converged diverted equilibria on a new machine
is iterative -- if the anchor inverse solves or the forward samples do not converge, tune: the grid
bounds, the isoflux target R/Z, the coil-current limits, and the profile ranges. The protocol +
analysis below are fixed; the equilibrium tuning is your job. Aim for >=150 converged diverted
samples spanning a range of kappa before reading the corr(kappa, m_s) verdict.
"""
import argparse
import json
import os
import pickle
import sys

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "experiments"))
import phase15_lib as L
import phase0_lib as P0

MCC = os.path.join(ROOT, "machine_configs", "Device-C")


def load_device_c():
    return P0.load_machine(os.path.join(MCC, "serialized_tokamak_C.pkl"))


def device_c_inverse_currents(tok, zscale, dR_extra, dShift):
    """MAST-U inverse-solve targets shifted radially by dShift (= r0_new - r0_old) for Device-C.
    Mirrors phase15_lib._inverse_currents but with every R target += dShift. Returns (I, kappa)."""
    from freegsnke import equilibrium_update, GSstaticsolver
    from freegsnke.inverse import Inverse_optimizer
    from freegsnke.jtor_update import ConstrainPaxisIp
    eq = equilibrium_update.Equilibrium(tokamak=tok, nx=L.NX, ny=L.NY, **L.GRID)
    profiles = ConstrainPaxisIp(eq=eq, paxis=L.NOM["paxis"], Ip=L.NOM["Ip"], fvac=L.NOM["fvac"],
                                alpha_m=L.NOM["alpha_m"], alpha_n=L.NOM["alpha_n"])
    solver = GSstaticsolver.NKGSsolver(eq=eq)
    eq.tokamak.set_coil_current("Solenoid", 5000)
    eq.tokamak["Solenoid"].control = False
    s = dShift
    Rx, Zx = 0.6 + dR_extra + s, 1.1 * zscale
    Rin, Rout = 0.34 + dR_extra + s, 1.4 + dR_extra + s
    RlegA, RlegB = 1.0 + dR_extra + s, 0.8 + dR_extra + s
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
    I = eq.tokamak.getCurrentsVec()[:L.N_ACTIVE].copy()
    sep = np.asarray(eq.separatrix()); R, Z = sep[:, 0], sep[:, 1]
    return I, float((Z.max() - Z.min()) / (R.max() - R.min()))


def main():
    import _blas_guard
    _blas_guard.assert_pinned()                    # refuse to run real solves unpinned (locked protocol)
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200, help="target converged diverted samples")
    ap.add_argument("--dshift", type=float, default=0.7, help="r0_new - r0_old used in device2_build")
    ap.add_argument("--rmax", type=float, default=2.8, help="expanded grid Rmax for Device-C")
    ap.add_argument("--workers", type=int, default=1, help="(this starter is single-process; parallelize like phase15_run)")
    args = ap.parse_args()

    # (1) EXPAND the grid for Device-C (plasma sits at larger R)
    L.GRID = dict(Rmin=0.1, Rmax=args.rmax, Zmin=-2.2, Zmax=2.2)
    tok = load_device_c()

    # (2) anchors on Device-C (shifted targets). Tune the (zscale,dR) grid + dshift if these fail.
    print("Building Device-C anchors (shifted inverse solves)...", flush=True)
    anchors = []
    for z in (0.85, 0.95, 1.0, 1.05):
        for d in (-0.1, 0.0, 0.1):
            try:
                I, k = device_c_inverse_currents(tok, z, d, args.dshift)
                anchors.append(dict(zscale=z, dR=d, I=I, kappa=k))
                print(f"  anchor z={z} dR={d} kappa={k:.3f}", flush=True)
            except Exception as e:
                print(f"  anchor z={z} dR={d} FAILED {type(e).__name__}: {str(e)[:60]}", flush=True)
    if len(anchors) < 2:
        print("\nKILL-GATE BLOCKED: too few anchors converged. TUNE (Mac): grid bounds, target R/Z, "
              "coil limits, dshift. The machine is valid (device2_build verified it) -- this is "
              "equilibrium tuning for the new geometry.", flush=True)
        return

    # forward-sample + label (reuses the crash-proofed forward_label; 40 modes for the cheap probe)
    rng = np.random.default_rng([20260625, 0])
    rows, attempts = [], 0
    while len(rows) < args.n and attempts < args.n * 4:
        attempts += 1
        c = L.sample_control(rng, anchors)
        try:
            rec = L.forward_label(tok, c["active_currents"], c["paxis"], c["Ip"], c["fvac"],
                                  c["alpha_m"], c["alpha_n"], fix_n_modes=40)
            if np.isfinite(rec.get("m_s", np.nan)) and rec["m_s"] > 0:
                rows.append(rec)
                if len(rows) % 25 == 0:
                    print(f"  {len(rows)} converged ({attempts} attempts)", flush=True)
        except Exception:
            continue

    if len(rows) < 30:
        print(f"\nKILL-GATE BLOCKED: only {len(rows)} converged samples. Tune the sampler/grid (Mac).", flush=True)
        return

    import pandas as pd
    df = pd.DataFrame(rows)
    logms = np.log(df["m_s"].values)
    from scipy.stats import spearmanr, pearsonr
    corr_k = float(spearmanr(df["kappa"], logms).statistic)
    # per-lever sensitivity = |corr(standardized descriptor, log m_s)| (cheap proxy for d log m_s/d feat)
    levers = ["kappa", "sq_uo", "sq_lo", "gap_inner", "gap_outer", "li", "betap", "delta"]
    sens = {}
    for f in levers:
        if f in df:
            sens[f] = abs(float(spearmanr(df[f], logms).statistic))
    kdom = abs(corr_k)
    rival = max((v for f, v in sens.items() if f != "kappa"), default=0.0)
    rival_lever = max((f for f in sens if f != "kappa"), key=lambda f: sens[f], default=None)
    go = (kdom < 0.75) or (rival >= 0.7 * kdom)

    out = dict(n=len(rows), dshift=args.dshift, corr_kappa_logms=corr_k,
               lever_sensitivity=sens, kappa_dominance=kdom, best_rival_lever=rival_lever,
               best_rival_sensitivity=rival, GO=bool(go),
               kappa_range=[float(df["kappa"].min()), float(df["kappa"].max())],
               aspect_median=float(df["aspect"].median()) if "aspect" in df else None)
    # NOTE: this single-process STARTER writes *_starter paths so it can NEVER clobber the canonical
    # artifacts (data/device2_killgate.json + device2_probe.parquet) produced by the parallel
    # device2_killgate_run.py + device2_killgate_analyze.py pipeline (seed 20260626, 11 chunks).
    json.dump(out, open(os.path.join(ROOT, "data", "device2_killgate_starter.json"), "w"), indent=2)
    df.to_parquet(os.path.join(ROOT, "data", "device2_probe_starter.parquet"))

    print(f"\n=== DEVICE-C KILL-GATE (n={len(rows)}, A_median~{out['aspect_median']}) ===")
    print(f"corr(kappa, log m_s) = {corr_k:+.3f}  (MAST-U = -0.875; |corr| here = {kdom:.3f})")
    print("per-lever |corr(feat, log m_s)|:")
    for f, v in sorted(sens.items(), key=lambda kv: -kv[1]):
        print(f"   {f:10s} {v:.3f}")
    print(f"best secondary lever = {rival_lever} ({rival:.3f}); kappa dominance = {kdom:.3f}")
    print(f"\nVERDICT: {'GO' if go else 'NO-GO'} -- "
          + ("kappa is de-dominated (or a secondary lever rivals it): RUN the unconstrained "
             "learned-m_s vs reduce-kappa design comparison (zero-shot transfer)."
             if go else
             "kappa still dominant: report 'kappa-dominance is general to tokamak vertical "
             "stability, not a MAST-U artifact' (a real scoping result) and STOP; OR escalate the "
             "aspect ratio (bigger --dshift in device2_build) and re-probe."))
    print("Saved data/device2_killgate_starter.json + data/device2_probe_starter.parquet "
          "(canonical artifacts come from device2_killgate_run.py + device2_killgate_analyze.py)")


if __name__ == "__main__":
    main()
