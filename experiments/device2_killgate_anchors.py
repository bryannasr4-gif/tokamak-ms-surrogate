"""
device2_killgate_anchors.py -- ONE-TIME builder of the Device-C forward-sampling anchors.

The forward sampler (phase15_lib.sample_control) interpolates between anchor active-current
vectors to CENTER the sampling distribution on Device-C's diverted-ST manifold. We build a
(zscale, dR) grid of shifted inverse solves (device2_killgate.device_c_inverse_currents) that
spans a RANGE of kappa, so the forward probe covers low->high elongation for a tight
corr(kappa, log m_s) estimate. Anchors are NOT labels; every probe sample is a clean forward
solve. Run thread-pinned (OMP=1). Saves data/device2_anchors.pkl.

  python experiments/device2_killgate_anchors.py --dshift 0.7 --rmax 2.8
"""
import argparse
import os
import pickle
import sys
import time

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "experiments"))
import phase15_lib as L
import device2_killgate as KG


def main():
    import _blas_guard
    _blas_guard.assert_pinned()                    # refuse to run inverse solves unpinned (locked protocol)
    ap = argparse.ArgumentParser()
    ap.add_argument("--dshift", type=float, default=0.7, help="r0_new - r0_old used in device2_build")
    ap.add_argument("--rmax", type=float, default=2.8, help="expanded grid Rmax for Device-C")
    # widened zscale grid (vs the killgate starter's 0.85-1.05) to span more kappa
    ap.add_argument("--zscales", type=float, nargs="+",
                    default=[0.78, 0.85, 0.92, 1.0, 1.07, 1.14])
    ap.add_argument("--dRs", type=float, nargs="+", default=[-0.1, 0.0, 0.1])
    ap.add_argument("--out", type=str, default=os.path.join(ROOT, "data", "device2_anchors.pkl"))
    args = ap.parse_args()

    # EXPAND the grid for Device-C (plasma sits at larger R) -- both the anchor inverse solves
    # and every forward label read this module global, so set it BEFORE any solve.
    L.GRID = dict(Rmin=0.1, Rmax=args.rmax, Zmin=-2.2, Zmax=2.2)

    print(f"Building Device-C anchors: zscales={args.zscales} dRs={args.dRs} dshift={args.dshift} "
          f"rmax={args.rmax}", flush=True)
    anchors = []
    t0 = time.time()
    for z in args.zscales:
        for d in args.dRs:
            ta = time.time()
            try:
                # COLD inverse solve: reload a FRESH machine per anchor so leftover coil currents
                # from the previous solve never seed (and break) the next one -- the locked
                # protocol's cold-solve discipline (PHASE0 §3.2 warm-start fragility).
                tok = KG.load_device_c()
                I, k = KG.device_c_inverse_currents(tok, float(z), float(d), args.dshift)
                anchors.append(dict(zscale=float(z), dR=float(d), I=I, kappa=float(k)))
                print(f"  anchor z={z:.2f} dR={d:+.2f}  kappa={k:.3f}  ({time.time()-ta:.1f}s)", flush=True)
            except Exception as e:
                print(f"  anchor z={z:.2f} dR={d:+.2f}  FAILED {type(e).__name__}: {str(e)[:70]} "
                      f"({time.time()-ta:.1f}s)", flush=True)
    if len(anchors) < 2:
        print("\nBLOCKED: <2 anchors converged -- tune grid/targets/limits (Mac).", flush=True)
        sys.exit(1)

    kappas = sorted(a["kappa"] for a in anchors)
    meta = dict(dshift=args.dshift, rmax=args.rmax, grid=L.GRID,
                n_anchors=len(anchors), kappa_min=kappas[0], kappa_max=kappas[-1])
    with open(args.out, "wb") as f:
        pickle.dump(dict(anchors=anchors, meta=meta), f)
    print(f"\n{len(anchors)} anchors built in {(time.time()-t0)/60:.1f} min; "
          f"kappa range [{kappas[0]:.3f}, {kappas[-1]:.3f}] -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
