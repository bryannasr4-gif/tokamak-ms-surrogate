"""
device2_gen_run.py -- launch a general Device-C generation across N thread-pinned chunk workers
(macOS: fusion-env/bin/python + all 5 BLAS vars pinned). See device2_gen_worker.py.

  python experiments/device2_gen_run.py --nchunks 11 --target 1200 --seed 20260626 \
         --linearise 0 --outprefix device2_shapegen
"""
import argparse
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = os.path.join(ROOT, "fusion-env", "bin", "python")
WORKER = os.path.join(ROOT, "experiments", "device2_gen_worker.py")
BLAS_ENV = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nchunks", type=int, default=11)
    ap.add_argument("--target", type=int, default=1200)
    ap.add_argument("--seed", type=int, default=20260626)
    ap.add_argument("--modes", type=int, default=80)
    ap.add_argument("--linearise", type=int, default=1)
    ap.add_argument("--outprefix", type=str, default="device2_gen")
    args = ap.parse_args()

    env = dict(os.environ)
    for k in BLAS_ENV:
        env[k] = "1"
    env["PYTHONIOENCODING"] = "utf-8"

    logdir = os.path.join(ROOT, "data", f"{args.outprefix}_logs")
    os.makedirs(logdir, exist_ok=True)
    procs = []
    t0 = time.time()
    for ch in range(args.nchunks):
        logf = open(os.path.join(logdir, f"chunk_{ch}.log"), "w")
        cmd = [PY, WORKER, "--chunk", str(ch), "--nchunks", str(args.nchunks),
               "--target", str(args.target), "--seed", str(args.seed), "--modes", str(args.modes),
               "--linearise", str(args.linearise), "--outprefix", args.outprefix]
        p = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, env=env, cwd=ROOT)
        procs.append((p, logf))
        print(f"launched chunk {ch} (pid {p.pid})", flush=True)
    print(f"\n{args.nchunks} workers; target {args.target} (linearise={args.linearise}, "
          f"modes={args.modes}, prefix={args.outprefix}).", flush=True)
    done = 0
    while done < len(procs):
        time.sleep(10)
        done = sum(1 for p, _ in procs if p.poll() is not None)
    for _, lf in procs:
        lf.close()
    print(f"\nALL {len(procs)} workers finished in {(time.time()-t0)/60:.1f} min.", flush=True)


if __name__ == "__main__":
    main()
