"""
phase25_kappa_run.py -- set up + launch the kappa-constrained "beats heuristics" experiment.
Holds kappa fixed and asks whether the learned m_s-gradient raises m_s via SECONDARY levers more
than (a) single-secondary-lever heuristics and (b) gradient-free search. Marginal/mid starts (room
to improve), d in {8,12}. Run AFTER the surrogate is retrained on the clean 80-mode labels.
  python phase25_kappa_run.py --nworkers 11 --budget 20
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
import phase25_kappa_lib as KL

PY = os.path.join(ROOT, "fusion-env", "Scripts", "python.exe")
WORKER = os.path.join(ROOT, "experiments", "phase25_kappa_worker.py")
BLAS_ENV = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
D_LIST = [12]      # d=12 gives the secondary levers room; this test is about levers, not d-scaling
# FAIR heuristic set (M6 fix): each secondary lever in BOTH signs, full budget, ranked by its own
# descriptor change (not the surrogate). "best heuristic" = best across all of these per start.
METHODS = (["surrogate", "random", "cma"] +
           [f"heuristic:{f}{s}" for f in ["sq_uo", "gap_outer", "li", "betap"] for s in ["+", "-"]])
# Phase-2.5b firming: GUARANTEE marginal-regime coverage. Stratify ~20 starts so ~half are
# marginal (m_s in [0.05,0.4), the m_s->0 design regime where the gradient is most load-bearing)
# and ~half mid (m_s in [0.4,1.0)). The earlier headline rested on only 1 marginal start.
N_START_MARG = 10
N_START_MID = 10
N_START = N_START_MARG + N_START_MID
MS_FLOOR = 0.05          # lowered from 0.2 to reach the marginal band (dataset_v1_80 min m_s=0.112)
MARG_HI = 0.4
MID_HI = 1.0


def _stratify(pool, n):
    """Pick n rows spread across a stratum (sorted by m_s) at evenly-spaced quantiles, including
    near the stratum edges so the near-boundary (m_s->0) shapes are covered. Deterministic."""
    pool = pool.sort_values("m_s").reset_index(drop=True)
    qs = np.linspace(0.05, 0.95, n)
    idxs = sorted(set(int(round(q * (len(pool) - 1))) for q in qs))
    # if rounding collapsed any duplicates, backfill with the densest unused indices
    k = 0
    while len(idxs) < n and k < len(pool):
        if k not in idxs:
            idxs.append(k)
        k += 1
    return [pool.iloc[i] for i in sorted(idxs)[:n]]


def build(budget, label_parquet):
    """Use the clean 80-mode labels to pick STRATIFIED marginal+mid starts (Phase-2.5b firming);
    reuse phase2 PCA control setup."""
    import pandas as pd
    setup = json.load(open(os.path.join(ROOT, "data", "phase2_dim_setup.json")))
    df = pd.read_parquet(label_parquet)
    ctrl = D.CONTROL_FEATURES
    marg_pool = df[(df["m_s"] >= MS_FLOOR) & (df["m_s"] < MARG_HI)]
    mid_pool = df[(df["m_s"] >= MARG_HI) & (df["m_s"] < MID_HI)]
    picked = _stratify(marg_pool, N_START_MARG) + _stratify(mid_pool, N_START_MID)
    starts = []
    for r in picked:
        starts.append(dict(idx=int(r.get("idx", r.name)), m_s_start=float(r["m_s"]),
                           kappa_start=float(r["kappa"]),
                           regime="marginal" if r["m_s"] < MARG_HI else "mid",
                           u=[float(r[c]) for c in ctrl]))
    nmarg = sum(1 for s in starts if s["regime"] == "marginal")
    print(f"starts: {len(starts)} total ({nmarg} marginal, {len(starts)-nmarg} mid); "
          f"m_s range [{min(s['m_s_start'] for s in starts):.3f}, {max(s['m_s_start'] for s in starts):.3f}]",
          flush=True)
    out = dict(mu=setup["mu"], std=setup["std"], V=setup["V"], box_lo=setup["box_lo"],
               box_hi=setup["box_hi"], budget=budget, starts=starts, ktol=KL.KTOL)
    json.dump(out, open(os.path.join(ROOT, "data", "phase25_kappa_setup.json"), "w"))
    jobs = []
    for si in range(len(starts)):
        for d in D_LIST:
            for m in METHODS:
                jobs.append(dict(job=len(jobs), start_i=si, d=d, method=m, seed=1000 + 7 * si + d))
    json.dump(dict(jobs=jobs), open(os.path.join(ROOT, "data", "phase25_kappa_jobs.json"), "w"))
    print(f"{len(starts)} starts x {len(D_LIST)} d x {len(METHODS)} methods = {len(jobs)} jobs", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nworkers", type=int, default=11)
    ap.add_argument("--budget", type=int, default=20)
    ap.add_argument("--labels", default=os.path.join(ROOT, "data", "dataset_v1_80.parquet"))
    ap.add_argument("--resume", action="store_true",
                    help="reuse existing setup/jobs + completed chunk recs; only redo unfinished jobs "
                         "(MUST use the same --nworkers as the interrupted run so job->chunk mapping matches)")
    args = ap.parse_args()
    if args.resume:
        print("RESUME mode: keeping existing chunk files + setup/jobs; redoing only unfinished jobs", flush=True)
    else:
        for fn in glob.glob(os.path.join(ROOT, "data", "phase25_kappa_chunk_*.json")):
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
        logf = open(os.path.join(logdir, f"kappa_chunk_{ch}.log"), "a" if args.resume else "w")
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
    for fn in sorted(glob.glob(os.path.join(ROOT, "data", "phase25_kappa_chunk_*.json"))):
        recs += json.load(open(fn))["recs"]
    json.dump(dict(n=len(recs), recs=recs), open(os.path.join(ROOT, "data", "phase25_kappa_results.json"), "w"), indent=1)
    print(f"\nALL done in {(time.time()-t0)/60:.1f} min; {len(recs)} runs", flush=True)


if __name__ == "__main__":
    main()
