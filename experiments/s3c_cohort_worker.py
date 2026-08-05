"""s3c_cohort_worker.py -- one round-robin chunk of unit S3c (pre-registered fixed-n third cohort).

Frozen design: data/audit/strategy/S4A_S3C_DESIGN.md (unit S3c); pre-registration data/s3c_prereg.json.
The loop body is phase25_kappa_lib.run_constrained -- the SAME function object that produced the
original 20 starts and both power top-up batches -- so the only variables changed versus those
batches are the cohort (fresh disjoint starts) and the arm set (2 pre-named arms).
Setup transcribed verbatim from phase4_power_worker.py. Resume-safe atomic per-chunk writes.
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
    ap.add_argument("--jobs", default=os.path.join(ROOT, "data", "phase4_power3_jobs.json"))
    ap.add_argument("--stem", default="s3c_chunk")
    args = ap.parse_args()

    S = json.load(open(os.path.join(ROOT, "data", "phase4_power3_setup.json")))
    jobs = json.load(open(args.jobs))["jobs"]
    mine = [j for k, j in enumerate(jobs) if k % args.nchunks == args.chunk]

    tok = L.load_machine()
    models, _ = M.load_ensemble()
    smap, _ = M.load_shapemap()
    mu = np.array(S["mu"]); std = np.array(S["std"]); V = np.array(S["V"])
    lo = np.array(S["box_lo"]); hi = np.array(S["box_hi"]); budget = S["budget"]

    out = os.path.join(ROOT, "data", f"{args.stem}_{args.chunk}.json")
    recs, done = [], set()
    if os.path.exists(out):
        try:
            recs = json.load(open(out)).get("recs", [])
            done = {(r["start_i"], r["method"]) for r in recs}
        except Exception:
            recs, done = [], set()
    mine = [j for j in mine if (j["start_i"], j["method"]) not in done]

    t0 = time.time()
    for j in mine:
        s = S["starts"][j["start_i"]]
        ds = DL.DesignSpace(mu, std, V, lo, hi, np.array(s["u"]), 12)
        m = 0.05 * (ds.box_hi - ds.box_lo)
        ds.box_lo = np.minimum(ds.box_lo, ds.x0 - m)
        ds.box_hi = np.maximum(ds.box_hi, ds.x0 + m)
        try:
            r = KL.run_constrained(tok, models, smap, ds, budget, j["method"],
                                   s["kappa_start"], rng=j.get("seed", 0))
        except Exception as e:
            r = dict(best_ms=0.0, n_solves=budget, error=f"{type(e).__name__}:{str(e)[:80]}")
        rec = dict(start_i=j["start_i"], idx=s["idx"], d=12, method=j["method"],
                   regime=s["regime"], m_s_start=s["m_s_start"], kappa_start=s["kappa_start"],
                   best_ms=r["best_ms"], n_solves=r["n_solves"],
                   gain=r["best_ms"] - s["m_s_start"], error=r.get("error"))
        recs.append(rec)
        print(f"[c{args.chunk}] start{j['start_i']:2d} idx{s['idx']:5d} {j['method']:22s} "
              f"ms0={s['m_s_start']:.3f} best={rec['best_ms']:.4f} gain={rec['gain']:+.4f} "
              f"({(time.time() - t0) / 60:.1f}min)", flush=True)
        tmp = out + ".tmp"
        json.dump(dict(chunk=args.chunk, recs=recs), open(tmp, "w"))
        os.replace(tmp, out)
    print(f"[c{args.chunk}] DONE {len(recs)} in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
