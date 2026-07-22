"""
phase2_modes_run.py -- launch the Phase-2 mode-convergence study.

1. Build a deterministic STRATIFIED subset of dataset_v1 (n_per_regime rows from each m_s
   regime, so the marginal m_s->0 band -- where truncation bites hardest -- is well sampled).
2. Write data/phase2_modes_subset.json (row indices + stored controls to replay).
3. Launch N thread-pinned workers (phase2_modes_worker.py); wait.
4. Assemble data/phase2_modes.json (per-shape m_s/gamma at each mode count).

Run in the BACKGROUND (pure solver; ~30-40 min). No surrogate needed.
  python phase2_modes_run.py --nworkers 11 --nper 15
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
WORKER = os.path.join(ROOT, "experiments", "phase2_modes_worker.py")
BLAS_ENV = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")


def build_subset(nper, seed=20260620):
    df = D.load()
    rng = np.random.default_rng(seed)
    rows = []
    for name, lo, hi in D.REGIMES:
        pool = df[(df["m_s"] >= lo) & (df["m_s"] < hi)]
        take = min(nper, len(pool))
        pick = rng.choice(pool.index.values, size=take, replace=False)
        for idx in pick:
            r = df.loc[idx]
            rep = D.replay_kwargs(r)
            rows.append(dict(idx=int(idx), regime=name, m_s_stored=float(r["m_s"]),
                             kappa_stored=float(r["kappa"]),
                             replay=dict(active_currents=rep["active_currents"].tolist(),
                                         paxis=rep["paxis"], Ip=rep["Ip"], fvac=rep["fvac"],
                                         alpha_m=rep["alpha_m"], alpha_n=rep["alpha_n"])))
    # interleave strata so round-robin chunking is load-balanced
    rows.sort(key=lambda r: (r["idx"] % 1000))
    out = os.path.join(ROOT, "data", "phase2_modes_subset.json")
    with open(out, "w") as f:
        json.dump(dict(seed=seed, nper=nper, rows=rows), f)
    print(f"subset: {len(rows)} shapes ({nper}/regime) -> {out}", flush=True)
    return len(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nworkers", type=int, default=11)
    ap.add_argument("--nper", type=int, default=15)
    args = ap.parse_args()

    # clean old chunk files
    for fn in glob.glob(os.path.join(ROOT, "data", "phase2_modes_chunk_*.json")):
        os.remove(fn)
    build_subset(args.nper)

    env = dict(os.environ)
    for k in BLAS_ENV:
        env[k] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    logdir = os.path.join(ROOT, "data", "phase2_logs")
    os.makedirs(logdir, exist_ok=True)

    procs = []
    t0 = time.time()
    for ch in range(args.nworkers):
        logf = open(os.path.join(logdir, f"modes_chunk_{ch}.log"), "w")
        cmd = [PY, WORKER, "--chunk", str(ch), "--nchunks", str(args.nworkers)]
        p = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, env=env, cwd=ROOT)
        procs.append((p, logf))
        print(f"launched modes chunk {ch} (pid {p.pid})", flush=True)

    done = 0
    while done < len(procs):
        time.sleep(10)
        done = sum(1 for p, _ in procs if p.poll() is not None)
    for _, lf in procs:
        lf.close()

    # assemble
    recs = []
    for fn in sorted(glob.glob(os.path.join(ROOT, "data", "phase2_modes_chunk_*.json"))):
        with open(fn) as f:
            recs += json.load(f)["recs"]
    out = os.path.join(ROOT, "data", "phase2_modes.json")
    with open(out, "w") as f:
        json.dump(dict(modes=[40, 80, 120, 138], n=len(recs), recs=recs), f, indent=2)
    print(f"\nALL {len(procs)} workers done in {(time.time()-t0)/60:.1f} min; "
          f"{len(recs)} shapes -> {out}", flush=True)


if __name__ == "__main__":
    main()
