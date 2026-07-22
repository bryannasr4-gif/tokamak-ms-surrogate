"""
device2_gen_worker.py -- ONE chunk of a general Device-C forward-sampling generation. Reuses the
kill-gate anchors + sampler; supports:
  * --linearise 0  -> FORWARD-ONLY (no Jacobian/m_s; ~5x cheaper) for ShapeMap (controls->shape)
                      training data. Keeps converged+diverted samples (shape descriptors only).
  * --linearise 1  -> full label at --modes (80 for the retrained-surrogate dataset; m_s>0 kept).
Writes data/{outprefix}_chunk_{chunk}.json incrementally. Run thread-pinned (OMP=1).

  python experiments/device2_gen_worker.py --chunk 0 --nchunks 11 --target 1200 --seed 20260626 \
         --linearise 0 --outprefix device2_shapegen
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
    ap.add_argument("--target", type=int, default=1200)
    ap.add_argument("--seed", type=int, default=20260626)
    ap.add_argument("--modes", type=int, default=80)
    ap.add_argument("--linearise", type=int, default=1, help="1=full m_s label; 0=forward-only (shape)")
    ap.add_argument("--outprefix", type=str, default="device2_gen")
    ap.add_argument("--max-attempts-factor", type=float, default=5.0)
    ap.add_argument("--anchors", type=str, default=os.path.join(ROOT, "data", "device2_anchors.pkl"))
    args = ap.parse_args()

    with open(args.anchors, "rb") as f:
        blob = pickle.load(f)
    anchors = blob["anchors"]
    L.GRID = dict(Rmin=0.1, Rmax=blob["meta"]["rmax"], Zmin=-2.2, Zmax=2.2)
    tok = KG.load_device_c()

    lin = bool(args.linearise)
    target_per_chunk = int(np.ceil(args.target / args.nchunks))
    max_attempts = int(target_per_chunk * args.max_attempts_factor)
    # distinct RNG stream per generation type so shape-gen samples != kill-gate samples
    rng = np.random.default_rng([args.seed, args.chunk, int(lin)])

    out = os.path.join(ROOT, "data", f"{args.outprefix}_chunk_{args.chunk}.json")
    recs, fails = [], {}
    t0 = time.time(); attempts = 0
    while len(recs) < target_per_chunk and attempts < max_attempts:
        attempts += 1
        c = L.sample_control(rng, anchors)
        try:
            rec = L.forward_label(tok, c["active_currents"], c["paxis"], c["Ip"], c["fvac"],
                                  c["alpha_m"], c["alpha_n"], fix_n_modes=args.modes,
                                  with_linearisation=lin)
            if lin:
                ms = rec.get("m_s", float("nan"))
                if not (np.isfinite(ms) and ms > 0):
                    raise RuntimeError("non-finite/non-positive m_s")
            rec["chunk"] = args.chunk
            recs.append(rec)
            if len(recs) % 20 == 0:
                extra = f"m_s={rec.get('m_s', float('nan')):.3f}" if lin else f"kappa={rec['kappa']:.3f}"
                print(f"[c{args.chunk}] {len(recs)}/{target_per_chunk} kept "
                      f"({(time.time()-t0)/max(len(recs),1):.0f}s/keep) {extra}", flush=True)
        except Exception as e:
            key = type(e).__name__ + ":" + str(e).split("(")[0][:30]
            fails[key] = fails.get(key, 0) + 1
        if attempts % 10 == 0:
            with open(out, "w") as f:
                json.dump(dict(chunk=args.chunk, seed=args.seed, nchunks=args.nchunks, modes=args.modes,
                               linearise=int(lin), attempts=attempts, fails=fails, recs=recs), f)
    with open(out, "w") as f:
        json.dump(dict(chunk=args.chunk, seed=args.seed, nchunks=args.nchunks, modes=args.modes,
                       linearise=int(lin), attempts=attempts, fails=fails, recs=recs), f)
    print(f"[c{args.chunk}] DONE {len(recs)} kept / {attempts} attempts in "
          f"{(time.time()-t0)/60:.1f} min -> {out}; fails={fails}", flush=True)


if __name__ == "__main__":
    main()
