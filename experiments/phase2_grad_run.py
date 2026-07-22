"""
phase2_grad_run.py -- launch the Phase-2 gradient-verification probe set.

1. Build a stratified set of held-out BASE equilibria (val + test_extrap rows; weighted toward
   the marginal/mid regimes where gradient-design matters most).
2. Define per-control finite-difference steps = rel * std(control over the dataset) so every
   control gets a comparable, in-distribution perturbation.
3. Launch N thread-pinned workers (phase2_grad_worker.py); assemble data/phase2_grad_probes.json.

Run in the BACKGROUND (pure solver; ~30-45 min). Needs no trained model (the analyzer applies
the surrogate afterwards).
  python phase2_grad_run.py --nworkers 11 --rel 0.04
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
WORKER = os.path.join(ROOT, "experiments", "phase2_grad_worker.py")
BLAS_ENV = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")

# bases per regime (held out); marginal/mid weighted (gradient-design matters most there)
NPER = {"marginal": 10, "mid": 10, "stable": 8, "very_stable": 2}


def build_bases(rel, seed=20260620):
    df = D.load()
    held = df[df.split.isin(["val", "test_extrap"])]
    rng = np.random.default_rng(seed)
    bases = []
    for name, _, _ in D.REGIMES:
        pool = held[held.regime == name]
        take = min(NPER[name], len(pool))
        for idx in rng.choice(pool.index.values, take, replace=False):
            r = df.loc[idx]
            u = [float(r[c]) for c in D.CONTROL_FEATURES]
            bases.append(dict(idx=int(idx), regime=name, split=str(r["split"]), u=u))
    steps = [rel * float(df[c].std()) for c in D.CONTROL_FEATURES]
    out = os.path.join(ROOT, "data", "phase2_grad_bases.json")
    with open(out, "w") as f:
        json.dump(dict(rel=rel, controls=D.CONTROL_FEATURES, steps=steps, bases=bases), f)
    print(f"bases: {len(bases)} held-out equilibria; steps rel={rel} -> {out}", flush=True)
    return len(bases)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nworkers", type=int, default=11)
    ap.add_argument("--rel", type=float, default=0.04)
    args = ap.parse_args()

    for fn in glob.glob(os.path.join(ROOT, "data", "phase2_grad_chunk_*.json")):
        os.remove(fn)
    build_bases(args.rel)

    env = dict(os.environ)
    for k in BLAS_ENV:
        env[k] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    logdir = os.path.join(ROOT, "data", "phase2_logs")
    os.makedirs(logdir, exist_ok=True)

    procs = []
    t0 = time.time()
    for ch in range(args.nworkers):
        logf = open(os.path.join(logdir, f"grad_chunk_{ch}.log"), "w")
        cmd = [PY, WORKER, "--chunk", str(ch), "--nchunks", str(args.nworkers)]
        p = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, env=env, cwd=ROOT)
        procs.append((p, logf))
        print(f"launched grad chunk {ch} (pid {p.pid})", flush=True)
    done = 0
    while done < len(procs):
        time.sleep(10)
        done = sum(1 for p, _ in procs if p.poll() is not None)
    for _, lf in procs:
        lf.close()

    recs = []
    for fn in sorted(glob.glob(os.path.join(ROOT, "data", "phase2_grad_chunk_*.json"))):
        with open(fn) as f:
            recs += json.load(f)["recs"]
    out = os.path.join(ROOT, "data", "phase2_grad_probes.json")
    with open(out, "w") as f:
        json.dump(dict(controls=D.CONTROL_FEATURES, n=len(recs), recs=recs), f)
    print(f"\nALL {len(procs)} workers done in {(time.time()-t0)/60:.1f} min; "
          f"{len(recs)} bases -> {out}", flush=True)


if __name__ == "__main__":
    main()
