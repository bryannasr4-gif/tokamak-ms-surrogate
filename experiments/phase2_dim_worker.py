"""
phase2_dim_worker.py -- one chunk of the Phase-2 dimensionality experiment.

Loads the serialized machine + trained surrogate/shapemap once, reads data/phase2_dim_setup.json
+ data/phase2_dim_jobs.json, and runs its round-robin slice of (start x dim x method) optimizer
jobs (phase2_dim_lib). Each job records #true-solves-to-target, best true m_s, and the trajectory.
Run thread-pinned (OMP=1).
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
import phase2_data as D
import phase2_model as M
import phase2_dim_lib as DL


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk", type=int, default=0)
    ap.add_argument("--nchunks", type=int, default=1)
    args = ap.parse_args()

    with open(os.path.join(ROOT, "data", "phase2_dim_setup.json")) as f:
        S = json.load(f)
    with open(os.path.join(ROOT, "data", "phase2_dim_jobs.json")) as f:
        jobs = json.load(f)["jobs"]
    mine = [j for k, j in enumerate(jobs) if k % args.nchunks == args.chunk]

    tok = L.load_machine()
    models, _ = M.load_ensemble()
    smap, _ = M.load_shapemap()
    mu = np.array(S["mu"]); std = np.array(S["std"]); V = np.array(S["V"])
    box_lo = np.array(S["box_lo"]); box_hi = np.array(S["box_hi"])
    target, budget = S["target"], S["budget"]

    out = os.path.join(ROOT, "data", f"phase2_dim_chunk_{args.chunk}.json")
    recs = []
    t0 = time.time()
    for j in mine:
        s = S["starts"][j["start_i"]]
        u0 = np.array(s["u"])
        ds = DL.DesignSpace(mu, std, V, box_lo, box_hi, u0, j["d"])
        meth = j["method"]
        try:
            if meth == "surrogate":
                r = DL.run_gradient(tok, models, smap, ds, target, budget, kind="surrogate")
            elif meth == "heuristic":
                r = DL.run_gradient(tok, models, smap, ds, target, budget, kind="heuristic")
            elif meth == "cma":
                r = DL.run_cma(tok, ds, target, budget, j["seed"])
            elif meth == "random":
                r = DL.run_random(tok, ds, target, budget, j["seed"])
            else:
                continue
        except Exception as e:
            r = dict(traj=[], n_solves=budget, best_ms=0.0, reached=False,
                     error=f"{type(e).__name__}:{str(e)[:80]}")
        rec = dict(job=j["job"], start_i=j["start_i"], d=j["d"], method=meth,
                   m_s_start=s["m_s_start"], **{k: r[k] for k in
                   ("n_solves", "best_ms", "reached") if k in r}, traj=r.get("traj", []),
                   error=r.get("error"))
        recs.append(rec)
        print(f"[c{args.chunk}] start{j['start_i']} d={j['d']} {meth:9s} "
              f"-> n_solves={rec['n_solves']} best_ms={rec['best_ms']:.3f} reached={rec['reached']} "
              f"({len(recs)}/{len(mine)}, {(time.time()-t0)/max(len(recs),1):.0f}s/job)", flush=True)
        with open(out, "w") as f:
            json.dump(dict(chunk=args.chunk, recs=recs), f)
    print(f"[c{args.chunk}] DONE {len(recs)} jobs in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
