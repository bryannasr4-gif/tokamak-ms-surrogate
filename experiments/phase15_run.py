"""
phase15_run.py -- launch the forward-sampling generation across N thread-pinned chunk workers.

Each worker (phase15_generate.py) is an isolated subprocess pinned to OMP=1 (so labels are
bit-reproducible and there is no BLAS oversubscription). The parent just launches one worker
per chunk, sets each child's BLAS env to 1, and waits. Incremental per-chunk JSON lets you
inspect / resume. Designed to be run in the BACKGROUND (it can take ~1-2 h).

  python phase15_run.py --nchunks 14 --target 3500 --seed 20260619
"""
import argparse
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = os.path.join(ROOT, "fusion-env", "Scripts", "python.exe")
WORKER = os.path.join(ROOT, "experiments", "phase15_generate.py")

BLAS_ENV = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nchunks", type=int, default=14)
    ap.add_argument("--target", type=int, default=3500)
    ap.add_argument("--seed", type=int, default=20260619)
    args = ap.parse_args()

    env = dict(os.environ)
    for k in BLAS_ENV:
        env[k] = "1"
    env["PYTHONIOENCODING"] = "utf-8"

    procs = []
    logdir = os.path.join(ROOT, "data", "phase15_logs")
    os.makedirs(logdir, exist_ok=True)
    t0 = time.time()
    for ch in range(args.nchunks):
        logf = open(os.path.join(logdir, f"chunk_{ch}.log"), "w")
        cmd = [PY, WORKER, "--chunk", str(ch), "--nchunks", str(args.nchunks),
               "--target", str(args.target), "--seed", str(args.seed)]
        p = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, env=env, cwd=ROOT)
        procs.append((p, logf))
        print(f"launched chunk {ch} (pid {p.pid})", flush=True)

    print(f"\n{args.nchunks} workers running; target {args.target} converged samples total.",
          flush=True)
    done = 0
    while done < len(procs):
        time.sleep(10)
        done = sum(1 for p, _ in procs if p.poll() is not None)
    for _, lf in procs:
        lf.close()
    print(f"\nALL {len(procs)} workers finished in {(time.time()-t0)/60:.1f} min.", flush=True)


if __name__ == "__main__":
    main()
