"""
device2_design_worker.py -- ONE striped chunk of the Device-C UNCONSTRAINED design comparison.
Resume-safe: each (framing, job) writes its own result file and is SKIPPED if already present, so a
sleep/restart loses at most the in-flight run. Run thread-pinned (OMP=1); every step 80-mode solved.

  python experiments/device2_design_worker.py --chunk 0 --nchunks 11 --framing zeroshot \
        --surrogate surrogate --shapemap shapemap_C
"""
import argparse
import json
import os
import sys

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
import device2_killgate as KG
import device2_design_lib as DZ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk", type=int, default=0)
    ap.add_argument("--nchunks", type=int, default=11)
    ap.add_argument("--framing", type=str, default="zeroshot")
    ap.add_argument("--surrogate", type=str, default="surrogate", help="ensemble name (shape->m_s)")
    ap.add_argument("--shapemap", type=str, default="shapemap_C", help="ShapeMap name (controls->shape)")
    ap.add_argument("--rmax", type=float, default=2.8)
    args = ap.parse_args()

    setup = json.load(open(os.path.join(ROOT, "data", "device2_design_setup.json")))
    jobs = json.load(open(os.path.join(ROOT, "data", "device2_design_jobs.json")))
    d, budget = setup["d"], setup["budget"]
    mu = np.array(setup["mu"]); std = np.array(setup["std"]); V = np.array(setup["V"])
    lo = np.array(setup["box_lo"]); hi = np.array(setup["box_hi"])
    starts = {s["id"]: s for s in setup["starts"]}

    # Device-C machine (expanded grid from the frozen anchors meta = single source of truth) + models
    import pickle
    with open(os.path.join(ROOT, "data", "device2_anchors.pkl"), "rb") as f:
        L.GRID = pickle.load(f)["meta"]["grid"]
    tok = KG.load_device_c()
    models, _ = M.load_ensemble(args.surrogate)
    smap, _ = M.load_shapemap(args.shapemap)

    outdir = os.path.join(ROOT, "data", "device2_design_results")
    os.makedirs(outdir, exist_ok=True)

    my = [j for k, j in enumerate(jobs) if k % args.nchunks == args.chunk]
    for j in my:
        out = os.path.join(outdir, f"{args.framing}_job{j['jid']}.json")
        if os.path.exists(out):
            continue
        s = starts[j["start_id"]]
        ds = DL.DesignSpace(mu, std, V, lo, hi, np.array(s["u0"], dtype=np.float64), d)
        res = DZ.run_method(tok, ds, budget, j["method"], models=models, smap=smap, seed=j["seed"])
        rec = dict(framing=args.framing, jid=j["jid"], start_id=j["start_id"], cohort=j["cohort"],
                   band=j["band"], method=j["method"], seed=j["seed"],
                   ms_start=res["m_s_start"], kappa_start=res["kappa_start"],
                   best_ms=res["best_ms"], gain=res["gain"], kappa_final=res["kappa_final"],
                   kappa_drift=res["kappa_drift"], n_solves=res["n_solves"],
                   best_u=res["best_u"], best_desc=res["best_desc"], traj=res["traj"],
                   reject=res["reject"])
        tmp = out + ".tmp"      # ATOMIC save (power-loss/sleep hardening): write then os.replace,
        with open(tmp, "w") as f:  # so an interrupted run leaves NO file (recomputed) -- never a
            json.dump(rec, f)      # truncated file that skip-if-exists would skip forever.
        os.replace(tmp, out)
        print(f"[c{args.chunk}] {args.framing} job{j['jid']} start{j['start_id']} {j['method']:12s} "
              f"ms {res['m_s_start']:.3f}->{res['best_ms']:.3f} (gain {res['gain']:+.3f}, "
              f"kappa {res['kappa_start']:.3f}->{res['kappa_final']:.3f})", flush=True)
    print(f"[c{args.chunk}] done {len(my)} assigned jobs", flush=True)


if __name__ == "__main__":
    main()
