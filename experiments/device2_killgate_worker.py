"""
device2_killgate_worker.py -- ONE chunk of the Device-C kill-gate forward probe (run many in
parallel via device2_killgate_run.py). Mirrors phase15_generate.py, with two Device-C specifics:
  (1) the grid is EXPANDED (Rmax) because the plasma sits at larger R;
  (2) labels use the CHEAP 40-mode probe (corr is a relative rank measure at fixed modes).

Each chunk: loads its own Device-C machine copy, loads the shared anchors, draws controls from a
chunk-deterministic RNG, forward-labels (40 modes), keeps converged+diverted+(m_s>0) samples, and
saves incrementally to data/device2_kg_chunk_{chunk}.json. Run thread-pinned (OMP=1).

  python experiments/device2_killgate_worker.py --chunk 0 --nchunks 11 --target 300 --seed 20260626
"""
import argparse
import json
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk", type=int, default=0)
    ap.add_argument("--nchunks", type=int, default=11)
    ap.add_argument("--target", type=int, default=300, help="total converged samples wanted")
    ap.add_argument("--seed", type=int, default=20260626)
    ap.add_argument("--modes", type=int, default=40, help="cheap probe modes")
    ap.add_argument("--max-attempts-factor", type=float, default=5.0)
    ap.add_argument("--anchors", type=str, default=os.path.join(ROOT, "data", "device2_anchors.pkl"))
    args = ap.parse_args()

    with open(args.anchors, "rb") as f:
        blob = pickle.load(f)
    anchors = blob["anchors"]
    rmax = blob["meta"]["rmax"]
    # EXPAND the grid for Device-C BEFORE any solve (module global read by forward_label).
    L.GRID = dict(Rmin=0.1, Rmax=rmax, Zmin=-2.2, Zmax=2.2)
    tok = KG.load_device_c()

    target_per_chunk = int(np.ceil(args.target / args.nchunks))
    max_attempts = int(target_per_chunk * args.max_attempts_factor)
    rng = np.random.default_rng([args.seed, args.chunk])

    out = os.path.join(ROOT, "data", f"device2_kg_chunk_{args.chunk}.json")
    recs, fails = [], {}
    t0 = time.time()
    attempts = 0
    while len(recs) < target_per_chunk and attempts < max_attempts:
        attempts += 1
        c = L.sample_control(rng, anchors)
        try:
            rec = L.forward_label(tok, c["active_currents"], c["paxis"], c["Ip"], c["fvac"],
                                  c["alpha_m"], c["alpha_n"], fix_n_modes=args.modes,
                                  with_linearisation=True)
            ms = rec.get("m_s", float("nan"))
            if not (np.isfinite(ms) and ms > 0):
                raise RuntimeError("non-finite or non-positive m_s")
            rec["chunk"] = args.chunk
            rec["attempt"] = attempts
            recs.append(rec)
            if len(recs) % 5 == 0:
                rate = len(recs) / attempts
                print(f"[c{args.chunk}] {len(recs)}/{target_per_chunk} kept (yield {rate:.0%}, "
                      f"{(time.time()-t0)/max(len(recs),1):.0f}s/keep) last m_s={ms:.3f} "
                      f"kappa={rec['kappa']:.3f}", flush=True)
        except Exception as e:
            key = type(e).__name__ + ":" + str(e).split("(")[0][:30]
            fails[key] = fails.get(key, 0) + 1
        if attempts % 5 == 0:
            with open(out, "w") as f:
                json.dump(dict(chunk=args.chunk, seed=args.seed, nchunks=args.nchunks, modes=args.modes,
                               attempts=attempts, fails=fails, recs=recs), f)
    with open(out, "w") as f:
        json.dump(dict(chunk=args.chunk, seed=args.seed, nchunks=args.nchunks, modes=args.modes,
                       attempts=attempts, fails=fails, recs=recs), f)
    print(f"[c{args.chunk}] DONE {len(recs)} kept / {attempts} attempts "
          f"in {(time.time()-t0)/60:.1f} min -> {out}", flush=True)
    print(f"[c{args.chunk}] fails: {fails}", flush=True)


if __name__ == "__main__":
    main()
