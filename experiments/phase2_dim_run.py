"""
phase2_dim_run.py -- set up + launch the Phase-2 dimensionality experiment (the headline).

1. PCA-orthogonalize the CONTROL space (standardized) over the dataset; report the effective
   dimensionality (participation ratio) and per-PC variance. The top-d PCs define the d-dim
   in-distribution search space (the only honest "increasing true dimension" axis).
2. Pick N_START marginal held-out starts; target m_s = 1.0 (marginal -> controllably stable).
3. Enumerate jobs (start x d x method) and launch thread-pinned workers; assemble results.

Run in the BACKGROUND (solver-heavy; ~1-2 h). Needs the trained models (data/phase2_models).
  python phase2_dim_run.py --nworkers 11 --budget 24
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
WORKER = os.path.join(ROOT, "experiments", "phase2_dim_worker.py")
BLAS_ENV = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")

D_LIST = [2, 4, 8, 12]      # span below/around/above the effective control dim (~5.5)
METHODS = ["surrogate", "heuristic", "cma", "random"]
N_START = 10                # Phase-2.5 power bump (was 5); now on clean 80-mode labels/surrogate
TARGET = 1.0


def build_setup(budget):
    from sklearn.decomposition import PCA
    df = D.load()
    ctrl = D.CONTROL_FEATURES
    U = df[ctrl].values.astype(np.float64)
    mu, std = U.mean(0), U.std(0) + 1e-12
    Z = (U - mu) / std
    pca = PCA(n_components=len(ctrl), svd_solver="full").fit(Z)
    V = pca.components_.T                                  # (16,16) columns = PCs
    scores = Z @ V                                         # (N,16)
    box_lo = np.percentile(scores, 2, axis=0)
    box_hi = np.percentile(scores, 98, axis=0)
    ev = pca.explained_variance_                          # eigenvalues
    part_ratio = float((ev.sum() ** 2) / (ev ** 2).sum())  # participation ratio (effective dim)
    var_ratio = pca.explained_variance_ratio_

    # marginal held-out starts spanning the marginal m_s range. Use the CLEAN 80-mode labels if
    # available (Phase 2.5) so m_s_start / "marginal" are at the converged mode count; controls
    # (hence the PCA above) are label-independent.
    import os as _os
    import pandas as pd
    _clean = _os.path.join(ROOT, "data", "dataset_v1_80.parquet")
    df_lab = pd.read_parquet(_clean) if _os.path.exists(_clean) else df
    marg = df_lab[df_lab.m_s < 0.4]
    marg = marg.sort_values("m_s")
    pick = np.linspace(0.1, 0.9, N_START)
    starts = []
    for q in pick:
        r = marg.iloc[int(q * (len(marg) - 1))]
        starts.append(dict(idx=int(r.name), m_s_start=float(r["m_s"]),
                           kappa_start=float(r["kappa"]),
                           u=[float(r[c]) for c in ctrl]))

    setup = dict(controls=ctrl, mu=mu.tolist(), std=std.tolist(), V=V.tolist(),
                 box_lo=box_lo.tolist(), box_hi=box_hi.tolist(),
                 d_list=D_LIST, methods=METHODS, target=TARGET, budget=budget,
                 starts=starts,
                 effective_dim=part_ratio,
                 var_ratio=var_ratio.tolist(),
                 cum_var=np.cumsum(var_ratio).tolist())
    with open(os.path.join(ROOT, "data", "phase2_dim_setup.json"), "w") as f:
        json.dump(setup, f)

    jobs = []
    for s_i, s in enumerate(starts):
        for d in D_LIST:
            for meth in METHODS:
                jobs.append(dict(job=len(jobs), start_i=s_i, d=d, method=meth,
                                 seed=1000 + 31 * s_i + d))
    with open(os.path.join(ROOT, "data", "phase2_dim_jobs.json"), "w") as f:
        json.dump(dict(jobs=jobs), f)
    print(f"effective control dim (participation ratio) = {part_ratio:.2f} / {len(ctrl)}", flush=True)
    print(f"cum var explained by top {D_LIST}: " +
          ", ".join(f"d={d}:{np.cumsum(var_ratio)[d-1]:.2f}" for d in D_LIST), flush=True)
    print(f"{len(jobs)} jobs ({N_START} starts x {len(D_LIST)} dims x {len(METHODS)} methods)", flush=True)
    return len(jobs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nworkers", type=int, default=11)
    ap.add_argument("--budget", type=int, default=24)
    args = ap.parse_args()

    for fn in glob.glob(os.path.join(ROOT, "data", "phase2_dim_chunk_*.json")):
        os.remove(fn)
    build_setup(args.budget)

    env = dict(os.environ)
    for k in BLAS_ENV:
        env[k] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    logdir = os.path.join(ROOT, "data", "phase2_logs")
    os.makedirs(logdir, exist_ok=True)

    procs = []
    t0 = time.time()
    for ch in range(args.nworkers):
        logf = open(os.path.join(logdir, f"dim_chunk_{ch}.log"), "w")
        cmd = [PY, WORKER, "--chunk", str(ch), "--nchunks", str(args.nworkers)]
        p = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, env=env, cwd=ROOT)
        procs.append((p, logf))
        print(f"launched dim chunk {ch} (pid {p.pid})", flush=True)
    done = 0
    while done < len(procs):
        time.sleep(15)
        done = sum(1 for p, _ in procs if p.poll() is not None)
    for _, lf in procs:
        lf.close()

    recs = []
    for fn in sorted(glob.glob(os.path.join(ROOT, "data", "phase2_dim_chunk_*.json"))):
        with open(fn) as f:
            recs += json.load(f)["recs"]
    out = os.path.join(ROOT, "data", "phase2_dim_results.json")
    with open(out, "w") as f:
        json.dump(dict(n=len(recs), recs=recs), f, indent=2)
    print(f"\nALL {len(procs)} workers done in {(time.time()-t0)/60:.1f} min; "
          f"{len(recs)} runs -> {out}", flush=True)


if __name__ == "__main__":
    main()
