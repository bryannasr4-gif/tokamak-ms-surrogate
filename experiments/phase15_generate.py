"""
phase15_generate.py -- ONE chunk of the forward-sampling generation (run many in parallel).

Each chunk:
  * loads its OWN copy of the serialized machine (bit-reproducible; no shared mutable state),
  * loads the shared anchors (data/phase15_anchors.pkl),
  * draws control vectors from a chunk-deterministic RNG (np.random.default_rng([seed, chunk])),
  * forward-labels each and KEEPS the converged+diverted ones,
  * saves incrementally to data/phase15_chunk_{chunk}.json,
  * stops once it has collected `target_per_chunk` converged samples (or hits max attempts).

Reproducible: the full dataset is determined by (seed, nchunks, target, locked protocol),
since every forward solve is bit-reproducible at OMP=1. Run thread-pinned (OMP=1) per process.

  python phase15_generate.py --chunk 0 --nchunks 14 --target 3500 --seed 20260619
"""
import argparse
import json
import os
import sys
import time
import traceback

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "experiments"))
import phase15_lib as L
import pickle


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk", type=int, default=0)
    ap.add_argument("--nchunks", type=int, default=1)
    ap.add_argument("--target", type=int, default=3500, help="total converged samples wanted")
    ap.add_argument("--seed", type=int, default=20260619)
    ap.add_argument("--max-attempts-factor", type=float, default=4.0)
    args = ap.parse_args()

    target_per_chunk = int(np.ceil(args.target / args.nchunks))
    max_attempts = int(target_per_chunk * args.max_attempts_factor)

    tok = L.load_machine()
    with open(os.path.join(ROOT, "data", "phase15_anchors.pkl"), "rb") as f:
        anchors = pickle.load(f)
    rng = np.random.default_rng([args.seed, args.chunk])

    out = os.path.join(ROOT, "data", f"phase15_chunk_{args.chunk}.json")
    recs, fails = [], {}
    t0 = time.time()
    attempts = 0
    while len(recs) < target_per_chunk and attempts < max_attempts:
        attempts += 1
        c = L.sample_control(rng, anchors)
        try:
            rec = L.forward_label(tok, **c, with_linearisation=True)
            if not np.isfinite(rec.get("m_s", np.nan)):
                raise RuntimeError("non-finite m_s")
            rec["chunk"] = args.chunk
            rec["attempt"] = attempts
            recs.append(rec)
            if len(recs) % 10 == 0:
                rate = len(recs) / attempts
                print(f"[c{args.chunk}] {len(recs)}/{target_per_chunk} kept "
                      f"(yield {rate:.0%}, {(time.time()-t0)/max(len(recs),1):.0f}s/keep) "
                      f"last m_s={rec['m_s']:.3f} kappa={rec['kappa']:.3f}", flush=True)
        except Exception as e:
            key = type(e).__name__ + ":" + str(e).split("(")[0][:30]
            fails[key] = fails.get(key, 0) + 1
        if attempts % 5 == 0:
            with open(out, "w") as f:
                json.dump(dict(chunk=args.chunk, seed=args.seed, nchunks=args.nchunks,
                               attempts=attempts, fails=fails, recs=recs), f)
    with open(out, "w") as f:
        json.dump(dict(chunk=args.chunk, seed=args.seed, nchunks=args.nchunks,
                       attempts=attempts, fails=fails, recs=recs), f)
    print(f"[c{args.chunk}] DONE {len(recs)} kept / {attempts} attempts "
          f"in {(time.time()-t0)/60:.1f} min -> {out}", flush=True)
    print(f"[c{args.chunk}] fails: {fails}", flush=True)


if __name__ == "__main__":
    main()
