"""
device2_design_run.py -- launch the Device-C UNCONSTRAINED design comparison across N thread-pinned,
resume-safe workers (macOS: fusion-env/bin/python + 5 BLAS vars). Re-runnable: completed per-job
result files are skipped, so a sleep/restart just resumes.

  python experiments/device2_design_run.py --nchunks 11 --framing zeroshot \
        --surrogate surrogate --shapemap shapemap_C
"""
import argparse
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = os.path.join(ROOT, "fusion-env", "bin", "python")
WORKER = os.path.join(ROOT, "experiments", "device2_design_worker.py")
BLAS_ENV = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nchunks", type=int, default=11)
    ap.add_argument("--framing", type=str, default="zeroshot")
    ap.add_argument("--surrogate", type=str, default="surrogate")
    ap.add_argument("--shapemap", type=str, default="shapemap_C")
    args = ap.parse_args()

    env = dict(os.environ)
    for k in BLAS_ENV:
        env[k] = "1"
    env["PYTHONIOENCODING"] = "utf-8"

    logdir = os.path.join(ROOT, "data", f"device2_design_logs_{args.framing}")
    os.makedirs(logdir, exist_ok=True)
    procs = []
    t0 = time.time()
    for ch in range(args.nchunks):
        logf = open(os.path.join(logdir, f"chunk_{ch}.log"), "w")
        cmd = [PY, WORKER, "--chunk", str(ch), "--nchunks", str(args.nchunks),
               "--framing", args.framing, "--surrogate", args.surrogate, "--shapemap", args.shapemap]
        p = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, env=env, cwd=ROOT)
        procs.append((p, logf))
        print(f"launched chunk {ch} (pid {p.pid})", flush=True)
    print(f"\n{args.nchunks} workers; framing={args.framing} surrogate={args.surrogate} "
          f"shapemap={args.shapemap}.", flush=True)
    done = 0
    while done < len(procs):
        time.sleep(10)
        done = sum(1 for p, _ in procs if p.poll() is not None)
    for _, lf in procs:
        lf.close()
    print(f"\nALL {len(procs)} workers finished in {(time.time()-t0)/60:.1f} min.", flush=True)


if __name__ == "__main__":
    main()
