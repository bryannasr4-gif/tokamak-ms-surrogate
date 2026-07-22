"""
phase3_run.py -- set up + launch the Phase-3 solver-confirmed design-loop experiment.

Poses >=20 stratified MARGINAL+MID design tasks (raise true m_s toward m*=1.0, holding shape
descriptors in range) and runs, at the SAME true-solve budget, the differentiable surrogate design
loop vs the reduce-kappa heuristic vs gradient-free search on the true solver (CMA / random /
Nelder-Mead). Resume-safe atomic-write worker pool (REUSED from phase25_kappa_*; a power loss must
not lose or duplicate work).

  ./fusion-env/Scripts/python.exe experiments/phase3_run.py --nworkers 10 --budget 30
  ... --resume   (redo only unfinished jobs; MUST use the same --nworkers)
"""
import argparse
import glob
import json
import os
import subprocess
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "experiments"))
import phase2_data as D

PY = os.path.join(ROOT, "fusion-env", "Scripts", "python.exe")
WORKER = os.path.join(ROOT, "experiments", "phase3_worker.py")
BLAS_ENV = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")

D_DIM = 12                                   # d=12: realistic many-lever design space (Phase-2 effective dim 5.5)
METHODS = ["surrogate", "heuristic", "cma", "random", "nelder"]
N_START_MARG = 10
N_START_MID = 10
MS_FLOOR = 0.05
MARG_HI = 0.4
MID_HI = 1.0


def _stratify(pool, n):
    """n rows spread across a stratum (sorted by m_s) at evenly-spaced quantiles incl. the edges.
    Deterministic (matches phase25_kappa_run)."""
    pool = pool.sort_values("m_s").reset_index(drop=True)
    qs = np.linspace(0.05, 0.95, n)
    idxs = sorted(set(int(round(q * (len(pool) - 1))) for q in qs))
    k = 0
    while len(idxs) < n and k < len(pool):
        if k not in idxs:
            idxs.append(k)
        k += 1
    return [pool.iloc[i] for i in sorted(idxs)[:n]]


def build(budget, label_parquet):
    import pandas as pd
    setup = json.load(open(os.path.join(ROOT, "data", "phase2_dim_setup.json")))
    ranges = json.load(open(os.path.join(ROOT, "data", "phase3_desc_ranges.json")))
    df = pd.read_parquet(label_parquet)
    ctrl = D.CONTROL_FEATURES
    marg = df[(df["m_s"] >= MS_FLOOR) & (df["m_s"] < MARG_HI)]
    mid = df[(df["m_s"] >= MARG_HI) & (df["m_s"] < MID_HI)]
    picked = _stratify(marg, N_START_MARG) + _stratify(mid, N_START_MID)
    guard = list(ranges.keys())
    starts = []
    for r in picked:
        starts.append(dict(idx=int(r.get("idx", r.name)), m_s_start=float(r["m_s"]),
                           kappa_start=float(r["kappa"]),
                           regime="marginal" if r["m_s"] < MARG_HI else "mid",
                           u=[float(r[c]) for c in ctrl],
                           desc={f: float(r[f]) for f in guard}))   # start's own descriptors (exact)
    nmarg = sum(1 for s in starts if s["regime"] == "marginal")
    print(f"starts: {len(starts)} ({nmarg} marginal, {len(starts) - nmarg} mid); "
          f"m_s [{min(s['m_s_start'] for s in starts):.3f}, {max(s['m_s_start'] for s in starts):.3f}]", flush=True)
    out = dict(mu=setup["mu"], std=setup["std"], V=setup["V"], box_lo=setup["box_lo"],
               box_hi=setup["box_hi"], budget=budget, d=D_DIM, starts=starts, ranges=ranges,
               methods=METHODS, target=1.0)
    json.dump(out, open(os.path.join(ROOT, "data", "phase3_setup.json"), "w"))
    jobs = []
    for si in range(len(starts)):
        for m in METHODS:
            jobs.append(dict(job=len(jobs), start_i=si, method=m, seed=4000 + 13 * si))
    json.dump(dict(jobs=jobs), open(os.path.join(ROOT, "data", "phase3_jobs.json"), "w"))
    print(f"{len(starts)} starts x {len(METHODS)} methods = {len(jobs)} jobs, budget {budget}, d={D_DIM}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nworkers", type=int, default=10)
    ap.add_argument("--budget", type=int, default=30)
    ap.add_argument("--labels", default=os.path.join(ROOT, "data", "dataset_v1_80q.parquet"))
    ap.add_argument("--resume", action="store_true",
                    help="reuse existing setup/jobs + completed chunk recs; redo only unfinished jobs "
                         "(use the SAME --nworkers as the interrupted run)")
    args = ap.parse_args()
    if args.resume:
        print("RESUME mode: keeping chunk files + setup/jobs; redoing only unfinished jobs", flush=True)
    else:
        for fn in glob.glob(os.path.join(ROOT, "data", "phase3_chunk_*.json")):
            os.remove(fn)
        labels = args.labels if os.path.exists(args.labels) else D.PARQUET
        build(args.budget, labels)
    env = dict(os.environ)
    for k in BLAS_ENV:
        env[k] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    logdir = os.path.join(ROOT, "data", "phase2_logs"); os.makedirs(logdir, exist_ok=True)
    procs = []
    t0 = time.time()
    for ch in range(args.nworkers):
        logf = open(os.path.join(logdir, f"phase3_chunk_{ch}.log"), "a" if args.resume else "w")
        p = subprocess.Popen([PY, WORKER, "--chunk", str(ch), "--nchunks", str(args.nworkers)],
                             stdout=logf, stderr=subprocess.STDOUT, env=env, cwd=ROOT)
        procs.append((p, logf))
    done = 0
    while done < len(procs):
        time.sleep(15)
        done = sum(1 for p, _ in procs if p.poll() is not None)
    for _, lf in procs:
        lf.close()
    recs = []
    for fn in sorted(glob.glob(os.path.join(ROOT, "data", "phase3_chunk_*.json"))):
        recs += json.load(open(fn))["recs"]
    json.dump(dict(n=len(recs), recs=recs),
              open(os.path.join(ROOT, "data", "phase3_results.json"), "w"))
    print(f"\nALL done in {(time.time() - t0) / 60:.1f} min; {len(recs)} runs", flush=True)


if __name__ == "__main__":
    main()
