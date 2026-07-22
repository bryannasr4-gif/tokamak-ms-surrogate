"""
tier1_resolve_run.py -- launch the Tier-1 shape-anchored re-solve across N BLAS-pinned chunk workers.

Mirrors phase15_run.py: one subprocess per chunk, each pinned to OMP=1, resume-safe via per-slice JSON
in data/tier1_resolved/. Run in the background.

  python tier1_resolve_run.py --nchunks 10 --source selection
"""
import argparse, os, subprocess, sys, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = os.path.join(ROOT, "fusion-env", "Scripts", "python.exe")
WORKER = os.path.join(ROOT, "experiments", "tier1_resolve_worker.py")
BLAS_ENV = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nchunks", type=int, default=10)
    ap.add_argument("--source", default="selection", choices=["selection", "pool"])
    ap.add_argument("--max_it", type=int, default=60)
    args = ap.parse_args()

    env = dict(os.environ)
    for k in BLAS_ENV:
        env[k] = "1"
    env["PYTHONIOENCODING"] = "utf-8"

    logdir = os.path.join(ROOT, "data", "tier1_logs")
    os.makedirs(logdir, exist_ok=True)
    procs = []
    t0 = time.time()
    for ch in range(args.nchunks):
        logf = open(os.path.join(logdir, f"chunk_{ch}.log"), "w")
        cmd = [PY, WORKER, "--chunk", str(ch), "--nchunks", str(args.nchunks),
               "--source", args.source, "--max_it", str(args.max_it)]
        p = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, env=env, cwd=ROOT)
        procs.append((p, logf))
        print(f"launched chunk {ch} (pid {p.pid})", flush=True)

    print(f"\n{args.nchunks} workers running (source={args.source}).", flush=True)
    done = 0
    while done < len(procs):
        time.sleep(10)
        done = sum(1 for p, _ in procs if p.poll() is not None)
    for _, lf in procs:
        lf.close()
    print(f"\nALL {len(procs)} workers finished in {(time.time()-t0)/60:.1f} min.", flush=True)


if __name__ == "__main__":
    main()
