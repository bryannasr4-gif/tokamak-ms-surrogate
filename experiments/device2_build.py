"""
device2_build.py -- build "Device C": a HIGHER-ASPECT-RATIO controlled variant of the MAST-U-like
machine, for the Phase-5 genuine-second-device test (does the learned m_s beat the reduce-kappa
heuristic UNCONSTRAINED on a device where kappa is NOT the dominant lever?).

WHY this construction (credibility): a fully hand-invented conventional tokamak risks a referee
rejecting the geometry as unphysical, and is hard to get converging. Instead we apply a SINGLE,
DOCUMENTED geometric transform to the validated MAST-U FreeGSNKE build -- a radial shift that moves
the machine from spherical-tokamak aspect ratio (A~1.6) toward conventional (A~2.5-3.2) while
keeping the coil/vessel TOPOLOGY (so inverse/forward solves still converge with correspondingly
shifted targets). This ISOLATES the aspect-ratio variable: the kill-gate then asks whether raising A
weakens the kappa->m_s dominance (corr(kappa, log m_s), MAST-U = -0.875). A controlled aspect-ratio
study is arguably CLEANER than two unrelated machines -- it changes one physical knob.

TRANSFORM (parameterized, documented): every R coordinate of every conductor (active coils, passive
structures, limiter, wall) is mapped R -> R0_new + (R - R0_old)*scale, with R0_old ~ the machine's
geometric centre. Default raises R0 (bigger major radius, same-ish minor radius -> higher A) without
distorting Z. The serialized tokamak is bit-reproducible.

NOTE FOR THE MAC SESSION: this builds the MACHINE (verified here). Getting CONVERGED DIVERTED
equilibria on it is the iterative part you must do (shift the inverse-solve isoflux targets in the
sampler by the same R-shift; tune coil-current limits). The kill-gate harness (device2_killgate.py)
is where you verify convergence + measure corr(kappa, m_s). Escalate the transform (bigger R-shift)
if A is not yet conventional; keep it physical (coils must stay outside the vessel, vessel outside
the plasma).
"""
import argparse
import copy
import os
import pickle
import sys

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MC = os.path.join(ROOT, "machine_configs", "MAST-U")
MCC = os.path.join(ROOT, "machine_configs", "Device-C")


def _shift_R(R, r0_old, r0_new, scale):
    """R -> r0_new + (R - r0_old)*scale, elementwise (list or scalar)."""
    a = np.asarray(R, dtype=float)
    return (r0_new + (a - r0_old) * scale).tolist() if a.ndim else float(r0_new + (a - r0_old) * scale)


def transform_active(active, r0_old, r0_new, scale):
    out = copy.deepcopy(active)
    for name, coil in out.items():
        if "R" in coil and not isinstance(coil["R"], dict):     # Solenoid-style flat coil
            coil["R"] = _shift_R(coil["R"], r0_old, r0_new, scale)
        else:                                                    # nested sub-elements
            for sub, el in coil.items():
                if isinstance(el, dict) and "R" in el:
                    el["R"] = _shift_R(el["R"], r0_old, r0_new, scale)
    return out


def transform_passive(passive, r0_old, r0_new, scale):
    out = copy.deepcopy(passive)
    for e in out:
        e["R"] = _shift_R(e["R"], r0_old, r0_new, scale)
    return out


def transform_polyline(poly, r0_old, r0_new, scale):
    out = copy.deepcopy(poly)
    for p in out:
        p["R"] = float(_shift_R(p["R"], r0_old, r0_new, scale))
    return out


def build(r0_old=0.9, r0_new=1.5, scale=1.0, name="Device-C"):
    """Load MAST-U data, apply the radial transform, build + serialize the new FreeGSNKE tokamak."""
    from freegsnke import build_machine
    os.makedirs(MCC, exist_ok=True)
    A = pickle.load(open(f"{MC}/MAST-U_like_active_coils.pickle", "rb"))
    P = pickle.load(open(f"{MC}/MAST-U_like_passive_coils.pickle", "rb"))
    Lim = pickle.load(open(f"{MC}/MAST-U_like_limiter.pickle", "rb"))
    Wall = pickle.load(open(f"{MC}/MAST-U_like_wall.pickle", "rb"))

    A2 = transform_active(A, r0_old, r0_new, scale)
    P2 = transform_passive(P, r0_old, r0_new, scale)
    Lim2 = transform_polyline(Lim, r0_old, r0_new, scale)
    Wall2 = transform_polyline(Wall, r0_old, r0_new, scale)

    tok = build_machine.tokamak(active_coils_data=A2, passive_coils_data=P2,
                                limiter_data=Lim2, wall_data=Wall2)
    ser = os.path.join(MCC, "serialized_tokamak_C.pkl")
    with open(ser, "wb") as f:
        pickle.dump(tok, f, protocol=pickle.HIGHEST_PROTOCOL)
    # also save the transform params + the shifted data (so the sampler can shift its targets)
    meta = dict(r0_old=r0_old, r0_new=r0_new, scale=scale, n_active=len(A2), n_passive=len(P2))
    pickle.dump(dict(active=A2, passive=P2, limiter=Lim2, wall=Wall2, meta=meta),
                open(os.path.join(MCC, "Device-C_data.pkl"), "wb"))
    return tok, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--r0_old", type=float, default=0.9, help="approx MAST-U geometric centre R")
    ap.add_argument("--r0_new", type=float, default=1.5, help="new major radius centre (raise for higher A)")
    ap.add_argument("--scale", type=float, default=1.0, help="minor-radius scale about the centre")
    args = ap.parse_args()
    tok, meta = build(args.r0_old, args.r0_new, args.scale)
    print(f"Built Device-C: shift R0 {args.r0_old}->{args.r0_new}, scale {args.scale}; "
          f"{meta['n_active']} active coils, {meta['n_passive']} passive structures.")
    # quick geometry sanity: limiter R-range before/after (=> rough aspect ratio change)
    Lim = pickle.load(open(f"{MC}/MAST-U_like_limiter.pickle", "rb"))
    R_old = np.array([p["R"] for p in Lim])
    R_new = _shift_R(R_old, args.r0_old, args.r0_new, args.scale)
    a_old = (R_old.max() - R_old.min()) / 2; R0_old = (R_old.max() + R_old.min()) / 2
    a_new = (np.array(R_new).max() - np.array(R_new).min()) / 2; R0_new = (np.array(R_new).max() + np.array(R_new).min()) / 2
    print(f"  limiter R: [{R_old.min():.2f},{R_old.max():.2f}] (R0~{R0_old:.2f}, a~{a_old:.2f}, A~{R0_old/a_old:.2f}) "
          f"-> [{np.array(R_new).min():.2f},{np.array(R_new).max():.2f}] "
          f"(R0~{R0_new:.2f}, a~{a_new:.2f}, A~{R0_new/a_new:.2f})")
    print(f"  serialized -> machine_configs/Device-C/serialized_tokamak_C.pkl")
    print("  NEXT (Mac): device2_killgate_anchors.py -> device2_killgate_run.py -> device2_killgate_analyze.py "
          "(canonical parallel kill-gate; device2_killgate.py is a single-process starter only).")


if __name__ == "__main__":
    main()
