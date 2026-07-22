"""
phase4_gallery_worker.py -- one round-robin chunk of the Phase-4 kappa-constrained gallery run.
Each job = one (start, method) kappa-constrained design loop with FULL recording. Resume-safe:
per-chunk file written atomically after every job; a relaunch skips finished (start, method) jobs.
"""
import argparse
import json
import os
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
import phase2_model as M
import phase2_dim_lib as DL
import phase4_gallery_lib as G

METHODS = ["surrogate", "kappa_nudge", "cma"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk", type=int, default=0)
    ap.add_argument("--nchunks", type=int, default=1)
    ap.add_argument("--budget", type=int, default=18)
    args = ap.parse_args()

    S = json.load(open(os.path.join(ROOT, "data", "phase25_kappa_setup.json")))
    starts = S["starts"]
    mu = np.array(S["mu"]); std = np.array(S["std"]); V = np.array(S["V"])
    lo = np.array(S["box_lo"]); hi = np.array(S["box_hi"])
    d = 12

    jobs = [(si, m) for si in range(len(starts)) for m in METHODS]
    mine = [j for k, j in enumerate(jobs) if k % args.nchunks == args.chunk]

    tok = L.load_machine()
    models, _ = M.load_ensemble()
    smap, _ = M.load_shapemap()

    out = os.path.join(ROOT, "data", f"phase4_gallery_chunk_{args.chunk}.json")
    recs, done = [], set()
    if os.path.exists(out):
        try:
            recs = json.load(open(out)).get("recs", [])
            done = {(r["start_i"], r["method"]) for r in recs}
        except Exception:
            recs, done = [], set()
    mine = [j for j in mine if (j[0], j[1]) not in done]

    t0 = time.time()
    for (si, method) in mine:
        s = starts[si]
        ds = DL.DesignSpace(mu, std, V, lo, hi, np.array(s["u"]), d)
        m = 0.05 * (ds.box_hi - ds.box_lo)
        ds.box_lo = np.minimum(ds.box_lo, ds.x0 - m)
        ds.box_hi = np.maximum(ds.box_hi, ds.x0 + m)
        try:
            if method == "cma":
                r = G.run_cma(tok, ds, args.budget, s["kappa_start"], None, seed=1000 + 7 * si + d)
            else:
                r = G.run_surrogate(tok, models, smap, ds, args.budget, s["kappa_start"], None,
                                    kind=method, seed=1000 + 7 * si + d)
            r.update(dict(start_i=si, method=method, regime=s["regime"], idx=s.get("idx")))
        except Exception as e:
            r = dict(start_i=si, method=method, regime=s["regime"], best_ms=0.0, gain=0.0,
                     n_solves=args.budget, error=f"{type(e).__name__}:{str(e)[:80]}")
        recs.append(r)
        print(f"[c{args.chunk}] start{si:2d} {method:12s} ms0={s['m_s_start']:.3f} "
              f"best={r.get('best_ms',0):.3f} gain={r.get('gain',0):+.3f} "
              f"kdrift={r.get('kappa_drift', float('nan')):.3f} n={r.get('n_solves','?')} "
              f"({(time.time()-t0)/60:.1f}min)", flush=True)
        tmp = out + ".tmp"
        json.dump(dict(chunk=args.chunk, recs=recs), open(tmp, "w"))
        os.replace(tmp, out)
    print(f"[c{args.chunk}] DONE {len(recs)} in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
