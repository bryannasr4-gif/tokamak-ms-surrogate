"""
phase25_kappa_worker.py -- one chunk of the kappa-constrained "beats heuristics" experiment.
Reuses the PCA control setup (data/phase2_dim_setup.json) + the (retrained) shape surrogate +
shapemap. Each job runs one (start x d x method) kappa-constrained optimization. Round-robin.
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
import phase25_kappa_lib as KL


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk", type=int, default=0)
    ap.add_argument("--nchunks", type=int, default=1)
    args = ap.parse_args()
    with open(os.path.join(ROOT, "data", "phase25_kappa_setup.json")) as f:
        S = json.load(f)
    with open(os.path.join(ROOT, "data", "phase25_kappa_jobs.json")) as f:
        jobs = json.load(f)["jobs"]
    mine = [j for k, j in enumerate(jobs) if k % args.nchunks == args.chunk]

    tok = L.load_machine()
    models, _ = M.load_ensemble()       # the (retrained) shape surrogate
    smap, _ = M.load_shapemap()
    mu = np.array(S["mu"]); std = np.array(S["std"]); V = np.array(S["V"])
    lo = np.array(S["box_lo"]); hi = np.array(S["box_hi"]); budget = S["budget"]

    out = os.path.join(ROOT, "data", f"phase25_kappa_chunk_{args.chunk}.json")
    # RESUME-SAFE (power-loss hardening): load any recs already on disk for this chunk and skip the
    # (start,d,method) jobs already done. Chunk files are saved incrementally per job, so progress
    # survives an interruption; a relaunch only redoes the unfinished jobs.
    recs = []
    done = set()
    if os.path.exists(out):
        try:
            recs = json.load(open(out)).get("recs", [])
            done = {(r["start_i"], r["d"], r["method"]) for r in recs}
            print(f"[c{args.chunk}] RESUME: {len(recs)} recs already done, skipping those", flush=True)
        except Exception:
            recs, done = [], set()
    mine = [j for j in mine if (j["start_i"], j["d"], j["method"]) not in done]
    t0 = time.time()
    for j in mine:
        s = S["starts"][j["start_i"]]
        ds = DL.DesignSpace(mu, std, V, lo, hi, np.array(s["u"]), j["d"])
        # Phase-2.5b firming fix: the extreme MARGINAL starts (high-kappa, 2% distribution tail)
        # can have PC scores OUTSIDE the dataset p2-p98 box. CMA-ES requires its start strictly
        # within the bounds (else "argument of inverse must be within the given bounds"), and a box
        # that excludes the start distorts every method's local search. The start IS a valid
        # in-distribution shape, so expand the box per-start to contain x0 with a small margin.
        m = 0.05 * (ds.box_hi - ds.box_lo)
        ds.box_lo = np.minimum(ds.box_lo, ds.x0 - m)
        ds.box_hi = np.maximum(ds.box_hi, ds.x0 + m)
        try:
            r = KL.run_constrained(tok, models, smap, ds, budget, j["method"],
                                   s["kappa_start"], rng=j.get("seed", 0))
        except Exception as e:
            r = dict(traj=[], n_solves=budget, best_ms=0.0, kappa_start=s["kappa_start"],
                     error=f"{type(e).__name__}:{str(e)[:60]}")
        rec = dict(start_i=j["start_i"], d=j["d"], method=j["method"], regime=s["regime"],
                   m_s_start=s["m_s_start"], best_ms=r["best_ms"], n_solves=r["n_solves"],
                   gain=r["best_ms"] - s["m_s_start"], error=r.get("error"))
        recs.append(rec)
        print(f"[c{args.chunk}] start{j['start_i']} d={j['d']} {j['method']:16s} "
              f"start_ms={s['m_s_start']:.3f} best={rec['best_ms']:.3f} gain={rec['gain']:+.3f} "
              f"(new {len(recs)-len(done)}/{len(mine)}, {(time.time()-t0)/max(len(recs)-len(done),1):.0f}s/job)",
              flush=True)
        # ATOMIC save (power-loss hardening): write to a temp file then os.replace, so a power cut
        # mid-write cannot truncate/corrupt the chunk file and lose all of this chunk's progress.
        tmp = out + ".tmp"
        with open(tmp, "w") as f:
            json.dump(dict(chunk=args.chunk, recs=recs), f)
        os.replace(tmp, out)
    print(f"[c{args.chunk}] DONE {len(recs)} in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
