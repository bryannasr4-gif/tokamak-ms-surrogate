"""
phase3_worker.py -- one chunk of the Phase-3 design-loop experiment. Each job runs one
(start x method) solver-confirmed design optimization. Resume-safe + atomic-write (power-loss
hardening, REUSED from phase25_kappa_worker): progress is saved per job and a relaunch only redoes
unfinished jobs.
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
import phase3_lib as P3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk", type=int, default=0)
    ap.add_argument("--nchunks", type=int, default=1)
    args = ap.parse_args()
    S = json.load(open(os.path.join(ROOT, "data", "phase3_setup.json")))
    jobs = json.load(open(os.path.join(ROOT, "data", "phase3_jobs.json")))["jobs"]
    mine = [j for k, j in enumerate(jobs) if k % args.nchunks == args.chunk]

    tok = L.load_machine()
    models, _ = M.load_ensemble()       # the clean 80-mode shape surrogate
    smap, _ = M.load_shapemap()
    mu = np.array(S["mu"]); std = np.array(S["std"]); V = np.array(S["V"])
    lo = np.array(S["box_lo"]); hi = np.array(S["box_hi"])
    budget = S["budget"]; d = S["d"]; ranges = S["ranges"]

    out = os.path.join(ROOT, "data", f"phase3_chunk_{args.chunk}.json")
    recs, done = [], set()
    if os.path.exists(out):
        try:
            recs = json.load(open(out)).get("recs", [])
            done = {(r["start_i"], r["method"]) for r in recs}
            print(f"[c{args.chunk}] RESUME: {len(recs)} recs already done", flush=True)
        except Exception:
            recs, done = [], set()
    mine = [j for j in mine if (j["start_i"], j["method"]) not in done]
    n_pre = len(recs)
    t0 = time.time()
    for j in mine:
        s = S["starts"][j["start_i"]]
        ds = DL.DesignSpace(mu, std, V, lo, hi, np.array(s["u"]), d)
        # Phase-2.5b fix: extreme marginal starts (high-kappa, 2% PC-score tail) can have x0 OUTSIDE
        # the dataset p2-p98 box -> CMA-ES requires its start strictly inside the bounds and a box
        # excluding the start distorts every method's local search. Expand the box per-start to
        # contain x0 with a small margin, IDENTICALLY for all methods.
        m = 0.05 * (ds.box_hi - ds.box_lo)
        ds.box_lo = np.minimum(ds.box_lo, ds.x0 - m)
        ds.box_hi = np.maximum(ds.box_hi, ds.x0 + m)
        # Same idea for the in-range descriptor guard: a few high-kappa marginal starts sit just
        # ABOVE the dataset kappa-p99, so expand the [p1,p99] guard per-start to contain the start's
        # OWN (exact) descriptors with a small margin -- a start is a valid design point and must
        # never be self-invalid. Applied IDENTICALLY to all methods for this start.
        sranges = {}
        for f, (rlo, rhi) in ranges.items():
            v = s.get("desc", {}).get(f)
            if v is None:
                sranges[f] = [rlo, rhi]
            else:
                pad = 0.02 * (rhi - rlo)
                sranges[f] = [min(rlo, v - pad), max(rhi, v + pad)]
        try:
            r = P3.run_one(tok, models, smap, ds, budget, sranges, s, j["method"], j.get("seed", 0))
            err = None
        except Exception as e:
            r = dict(m_s_start=s["m_s_start"], kappa_start=s["kappa_start"], regime=s["regime"],
                     n_solves=budget, best_ms=0.0, best_desc=None, gain=0.0, reached_primary=False,
                     solves_to_target={f"{t:.1f}": None for t in P3.TARGETS},
                     reject={}, traj=[], accepted=[])
            err = f"{type(e).__name__}:{str(e)[:80]}"
        rec = dict(start_i=j["start_i"], method=j["method"], seed=j.get("seed", 0),
                   error=err, **r)
        recs.append(rec)
        nnew = len(recs) - n_pre
        print(f"[c{args.chunk}] start{j['start_i']:2d}({s['regime'][:4]}) {j['method']:10s} "
              f"start={s['m_s_start']:.3f} best={rec['best_ms']:.3f} gain={rec['gain']:+.3f} "
              f"nsolve={rec['n_solves']} reached={rec['reached_primary']} "
              f"(new {nnew}/{len(mine)}, {(time.time() - t0) / max(nnew, 1):.0f}s/job)", flush=True)
        tmp = out + ".tmp"
        with open(tmp, "w") as f:
            json.dump(dict(chunk=args.chunk, recs=recs), f)
        os.replace(tmp, out)
    print(f"[c{args.chunk}] DONE {len(recs)} recs in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
