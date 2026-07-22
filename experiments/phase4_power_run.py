"""
phase4_power_run.py -- POWER TOP-UP for the kappa-constrained learned-m_s comparison (Phase-4
LEAD-c; addresses the Phase-3 open item "n=10/regime is at the resolution floor"). Adds NEW
stratified marginal+mid starts (DISJOINT from the original 20 in data/phase25_kappa_setup.json) and
runs the IDENTICAL method set + loop (phase25_kappa_lib.run_constrained, same KTOL/budget/PCA setup),
so the new runs POOL cleanly with data/phase25_kappa_results.json for a combined n.

Parallel, thread-pinned, resume-safe (atomic per-chunk writes). Run AFTER the gallery (avoid
over-subscribing cores).
  OMP_NUM_THREADS=1 ... python phase4_power_run.py --nworkers 10 --budget 18 --nmarg 12 --nmid 6
"""
import argparse
import glob
import json
import os
import subprocess
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import sys
sys.path.insert(0, os.path.join(ROOT, "experiments"))
import phase2_data as D
import phase25_kappa_lib as KL

PY = os.path.join(ROOT, "fusion-env", "Scripts", "python.exe")
WORKER = os.path.join(ROOT, "experiments", "phase4_power_worker.py")
BLAS_ENV = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
METHODS = (["surrogate", "random", "cma"] +
           [f"heuristic:{f}{s}" for f in ["sq_uo", "gap_outer", "li", "betap"] for s in ["+", "-"]])
MS_FLOOR, MARG_HI, MID_HI = 0.05, 0.4, 1.0


def build(budget, nmarg, nmid, labels, tag=""):
    import pandas as pd
    import glob as _glob
    setup = json.load(open(os.path.join(ROOT, "data", "phase2_dim_setup.json")))
    orig = json.load(open(os.path.join(ROOT, "data", "phase25_kappa_setup.json")))
    used_idx = {s["idx"] for s in orig["starts"]}
    # exclude starts already used by ANY prior power batch (disjoint pooling)
    for fn in _glob.glob(os.path.join(ROOT, "data", "phase4_power*_setup.json")):
        try:
            used_idx |= {s["idx"] for s in json.load(open(fn))["starts"]}
        except Exception:
            pass
    df = pd.read_parquet(labels)
    ctrl = D.CONTROL_FEATURES

    def stratify_new(pool, n):
        pool = pool[~pool["idx"].isin(used_idx)].sort_values("m_s").reset_index(drop=True)
        # interleaved quantiles offset from the original (0.025..0.975) to land on fresh shapes
        qs = np.linspace(0.025, 0.975, n)
        idxs = sorted(set(int(round(q * (len(pool) - 1))) for q in qs))
        return [pool.iloc[i] for i in idxs[:n]]

    marg = df[(df["m_s"] >= MS_FLOOR) & (df["m_s"] < MARG_HI)]
    mid = df[(df["m_s"] >= MARG_HI) & (df["m_s"] < MID_HI)]
    picked = stratify_new(marg, nmarg) + stratify_new(mid, nmid)
    starts = []
    for r in picked:
        starts.append(dict(idx=int(r["idx"]), m_s_start=float(r["m_s"]), kappa_start=float(r["kappa"]),
                           regime="marginal" if r["m_s"] < MARG_HI else "mid",
                           u=[float(r[c]) for c in ctrl]))
    nm = sum(1 for s in starts if s["regime"] == "marginal")
    print(f"NEW starts (batch '{tag}'): {len(starts)} ({nm} marginal, {len(starts)-nm} mid); "
          f"disjoint from original 20 + all prior power batches", flush=True)
    out = dict(mu=setup["mu"], std=setup["std"], V=setup["V"], box_lo=setup["box_lo"],
               box_hi=setup["box_hi"], budget=budget, starts=starts, ktol=KL.KTOL)
    json.dump(out, open(os.path.join(ROOT, "data", f"phase4_power{tag}_setup.json"), "w"))
    jobs = []
    for si in range(len(starts)):
        for m in METHODS:
            jobs.append(dict(job=len(jobs), start_i=si, method=m, seed=2000 + 7 * si + len(tag) * 13))
    json.dump(dict(jobs=jobs), open(os.path.join(ROOT, "data", f"phase4_power{tag}_jobs.json"), "w"))
    print(f"{len(starts)} starts x {len(METHODS)} methods = {len(jobs)} jobs", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nworkers", type=int, default=10)
    ap.add_argument("--budget", type=int, default=18)
    ap.add_argument("--nmarg", type=int, default=12)
    ap.add_argument("--nmid", type=int, default=6)
    ap.add_argument("--tag", default="", help="batch tag, e.g. '2' -> phase4_power2_* files (disjoint pooling)")
    ap.add_argument("--labels", default=os.path.join(ROOT, "data", "dataset_v1_80.parquet"))
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    tag = args.tag
    if not args.resume:
        for fn in glob.glob(os.path.join(ROOT, "data", f"phase4_power{tag}_chunk_*.json")):
            os.remove(fn)
        build(args.budget, args.nmarg, args.nmid, args.labels, tag=tag)
    env = dict(os.environ)
    for k in BLAS_ENV:
        env[k] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    logdir = os.path.join(ROOT, "data", "phase2_logs"); os.makedirs(logdir, exist_ok=True)
    procs = []
    t0 = time.time()
    for ch in range(args.nworkers):
        logf = open(os.path.join(logdir, f"p4power{tag}_chunk_{ch}.log"), "a" if args.resume else "w")
        p = subprocess.Popen([PY, WORKER, "--chunk", str(ch), "--nchunks", str(args.nworkers), "--tag", tag],
                             stdout=logf, stderr=subprocess.STDOUT, env=env, cwd=ROOT)
        procs.append((p, logf))
    done = 0
    while done < len(procs):
        time.sleep(15)
        done = sum(1 for p, _ in procs if p.poll() is not None)
    for _, lf in procs:
        lf.close()
    recs = []
    for fn in sorted(glob.glob(os.path.join(ROOT, "data", f"phase4_power{tag}_chunk_*.json"))):
        recs += json.load(open(fn))["recs"]
    json.dump(dict(n=len(recs), recs=recs),
              open(os.path.join(ROOT, "data", f"phase4_power{tag}_results.json"), "w"))
    print(f"\nALL done in {(time.time()-t0)/60:.1f} min; {len(recs)} runs", flush=True)


if __name__ == "__main__":
    main()
