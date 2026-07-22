"""
phase0_ms_crosscheck.py -- Phase 0, Part 1: INDEPENDENT cross-check of FreeGSNKE's m_s.

For one converged MAST-U equilibrium at the locked protocol, extract the L (=M0) and
S (=-dM) circuit blocks and INDEPENDENTLY recompute the Portone-2005 stability margin
three algebraically-distinct ways (eig(L^-1 S - I); the -L^-1 L* form; the generalized
eigenproblem eig(S,L)-1). Confirm FreeGSNKE's reported m_s matches to <5% and pin down the
sign/convention. Then a 3-point elongation scan fixes the empirical convention direction
(does m_s rise or fall as the plasma destabilises?).

Run thread-pinned:  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python ...
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import phase0_lib as P0

BLAS = {k: os.environ.get(k, "<unset>") for k in
        ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]}


def jsonable(x):
    if isinstance(x, np.ndarray):
        if np.iscomplexobj(x):
            return [[float(np.real(v)), float(np.imag(v))] for v in np.ravel(x)]
        return [float(v) for v in np.ravel(x)]
    return x


def main():
    print(f"BLAS threads: {BLAS}")
    out = {"blas": BLAS, "protocol": dict(nx=P0.DEF_NX, ny=P0.DEF_NY,
           fix_n_modes=P0.DEF_FIX_N_MODES, inv_tol=P0.DEF_INV_TOL)}

    print("Building machine (once)...")
    tok = P0.build_tokamak()

    # --- (1) canonical shape: rigorous independent recomputation ---------------
    print(f"Solving canonical shape {P0.CANON} at locked protocol...")
    res, nls = P0.solve_equilibrium(tok, **P0.CANON, return_nls=True)
    ind = P0.independent_stability_margins(nls)

    reported_max = float(res["m_s"])
    out["canonical"] = res
    out["n_independent_vars"] = ind["n"]
    out["identities"] = dict(M0pdM_minus_M=ind["identity_M0pdM_eq_M"],
                             Lstar_minus_LmS=ind["identity_Lstar_eq_LmS"])
    out["max_imag_eig_A"] = ind["max_imag_A"]

    def cmp(name, pos):
        if pos.size == 0:
            print(f"  {name}: NO positive eigenvalues!")
            return dict(max_pos=None, n_pos=0, pct_diff=None)
        mp = float(pos[0])
        pct = 100.0 * abs(mp - reported_max) / abs(reported_max) if reported_max else float("nan")
        print(f"  {name}: max+={mp:.6f}  n_pos={pos.size}  vs reported {reported_max:.6f}  "
              f"-> {pct:.3f}%   (top: {np.round(pos[:4],4).tolist()})")
        return dict(max_pos=mp, n_pos=int(pos.size), pct_diff=float(pct), top=jsonable(pos[:6]))

    print(f"\nFreeGSNKE reported m_s (max positive) = {reported_max:.6f}")
    print(f"FreeGSNKE all_stability_margins (>0 kept) = {np.round(ind['reported'],4).tolist()}")
    print(f"FreeGSNKE full all_stability_margins      = {np.round(np.sort(ind['all_reported']),4).tolist()}")
    print(f"identity |(M0+dM)-M|_metal = {ind['identity_M0pdM_eq_M']:.2e}   "
          f"|Lstar-(L-S)| = {ind['identity_Lstar_eq_LmS']:.2e}   "
          f"max|Im(eig A)| = {ind['max_imag_A']:.2e}")
    print("\nIndependent recomputations (max positive real eigenvalue):")
    out["method_A_LinvS_minus_I"] = cmp("A: eig(L^-1 S) - 1   ", ind["pos_A"])
    out["method_B_neg_Linv_Lstar"] = cmp("B: eig(-L^-1 L*)     ", ind["pos_B"])
    out["method_C_generalized"] = cmp("C: eig(S,L) - 1      ", ind["pos_C"])

    out["gamma"] = res["gamma"]
    out["n_unstable"] = res["n_unstable"]
    print(f"\ngrowth rate gamma = {res['gamma']:.3f} /s  (n_unstable modes = {res['n_unstable']})")

    worst = max(v["pct_diff"] for v in
                [out["method_A_LinvS_minus_I"], out["method_B_neg_Linv_Lstar"], out["method_C_generalized"]]
                if v["pct_diff"] is not None)
    out["max_pct_diff_vs_reported"] = float(worst)
    gate1 = worst < 5.0
    out["gate_match_under_5pct"] = bool(gate1)
    print(f"\n==> max % diff (independent vs FreeGSNKE) = {worst:.4f}%   "
          f"GATE (<5%): {'PASS' if gate1 else 'FAIL'}")

    # --- (2) convention direction: 3-point elongation scan ---------------------
    print("\nElongation mini-scan (fixes the sign convention direction)...")
    scan = []
    for z in (0.90, 0.95, 1.00):
        r = P0.solve_equilibrium(tok, zscale=z, dR=0.0)
        scan.append(r)
        print(f"  zscale={z:.2f}  kappa={r['kappa']:.3f}  gamma={r['gamma']:8.2f} /s  "
              f"m_s={r['m_s']:.4f}  n_pos_margins={r['n_positive_margins']}")
    out["elongation_scan"] = scan
    # direction: does m_s fall as gamma rises?
    g = np.array([s["gamma"] for s in scan]); m = np.array([s["m_s"] for s in scan])
    falls = bool(m[-1] < m[0] and g[-1] > g[0])
    out["m_s_falls_as_gamma_rises"] = falls
    print(f"\n==> as gamma rises {g[0]:.1f}->{g[-1]:.1f}/s, m_s goes {m[0]:.3f}->{m[-1]:.3f}  "
          f"=> bigger m_s == {'MORE STABLE' if falls else 'unclear'}")

    os.makedirs(os.path.join(P0.ROOT, "data"), exist_ok=True)
    path = os.path.join(P0.ROOT, "data", "phase0_ms_crosscheck.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2, default=jsonable)
    print(f"\nSaved -> {path}")


if __name__ == "__main__":
    main()
