"""
audit_reproducibility.py -- quantify the m_s / gamma LABEL reproducibility.

The audit found that the SAME target shape gives different m_s/gamma depending on the
inverse-solve PATH (cold vs warm start), because the vertical mode is physically sensitive
to the exact converged equilibrium. This script isolates the effects on ONE canonical shape
(zscale=1.0, dR=0) at the Phase-1 protocol (65x65, fix_n_vessel_modes=40):

  cold_A, cold_B : two independent fresh-tokamak solves     -> determinism + cold spread
  warm           : same shape but warm-started from z=0.90  -> warm-start sensitivity

Reports the m_s and gamma spread. This is the label-noise floor any surrogate must be
read against, and it is a required rigor number for the paper.
"""
import json
import os
import numpy as np

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sys
sys.path.insert(0, os.path.join(os.getcwd(), "experiments"))
from phase1_generate import build_tokamak, evaluate

CANON = (1.00, 0.0)

def run():
    out = {}
    # cold A
    tA = build_tokamak(); rA = evaluate(tA, *CANON)
    out["cold_A"] = rA
    print(f"cold_A : m_s={rA['m_s']:.4f}  gamma={rA['gamma']:.2f}  kappa={rA['kappa']:.4f}")
    # cold B (independent fresh machine -> determinism)
    tB = build_tokamak(); rB = evaluate(tB, *CANON)
    out["cold_B"] = rB
    print(f"cold_B : m_s={rB['m_s']:.4f}  gamma={rB['gamma']:.2f}  kappa={rB['kappa']:.4f}")
    # warm: reuse tB, solve a shorter shape first, then the canonical shape
    _ = evaluate(tB, 0.90, 0.0)
    rW = evaluate(tB, *CANON)
    out["warm_from_0p90"] = rW
    print(f"warm   : m_s={rW['m_s']:.4f}  gamma={rW['gamma']:.2f}  kappa={rW['kappa']:.4f}")

    ms = np.array([rA["m_s"], rB["m_s"], rW["m_s"]])
    gm = np.array([rA["gamma"], rB["gamma"], rW["gamma"]])
    out["m_s_spread_pct"] = float(100 * (ms.max() - ms.min()) / ms.mean())
    out["gamma_spread_pct"] = float(100 * (gm.max() - gm.min()) / gm.mean())
    out["cold_determinism_m_s_absdiff"] = float(abs(rA["m_s"] - rB["m_s"]))
    print(f"\nm_s spread across paths:   {out['m_s_spread_pct']:.1f}%")
    print(f"gamma spread across paths: {out['gamma_spread_pct']:.1f}%")
    print(f"cold determinism |Δm_s|:   {out['cold_determinism_m_s_absdiff']:.2e}")
    with open("data/audit_reproducibility.json", "w") as f:
        json.dump(out, f, indent=2)
    print("Saved -> data/audit_reproducibility.json")

if __name__ == "__main__":
    run()
