"""
phase4_power_worker.py -- one round-robin chunk of the Phase-4 power top-up. Identical loop +
recording format as phase25_kappa_worker (KL.run_constrained), reading the phase4_power_* setup/jobs,
so results pool with data/phase25_kappa_results.json. Resume-safe atomic per-chunk writes.
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
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    tag = args.tag
    S = json.load(open(os.path.join(ROOT, "data", f"phase4_power{tag}_setup.json")))
    jobs = json.load(open(os.path.join(ROOT, "data", f"phase4_power{tag}_jobs.json")))["jobs"]
    mine = [j for k, j in enumerate(jobs) if k % args.nchunks == args.chunk]

    tok = L.load_machine()
    models, _ = M.load_ensemble()
    smap, _ = M.load_shapemap()
    mu = np.array(S["mu"]); std = np.array(S["std"]); V = np.array(S["V"])
    lo = np.array(S["box_lo"]); hi = np.array(S["box_hi"]); budget = S["budget"]

    out = os.path.join(ROOT, "data", f"phase4_power{tag}_chunk_{args.chunk}.json")
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
            r = dict(best_ms=0.0, n_solves=budget, error=f"{type(e).__name__}:{str(e)[:60]}")
        rec = dict(start_i=j["start_i"], d=12, method=j["method"], regime=s["regime"],
                   m_s_start=s["m_s_start"], best_ms=r["best_ms"], n_solves=r["n_solves"],
                   gain=r["best_ms"] - s["m_s_start"], error=r.get("error"))
        recs.append(rec)
        print(f"[c{args.chunk}] start{j['start_i']} {j['method']:16s} ms0={s['m_s_start']:.3f} "
              f"best={rec['best_ms']:.3f} gain={rec['gain']:+.3f} ({(time.time()-t0)/60:.1f}min)", flush=True)
        tmp = out + ".tmp"
        json.dump(dict(chunk=args.chunk, recs=recs), open(tmp, "w"))
        os.replace(tmp, out)
    print(f"[c{args.chunk}] DONE {len(recs)} in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
