"""
phase4_gallery_run.py -- launch the Phase-4 kappa-constrained gallery run (parallel, thread-pinned,
resume-safe) on the SAME 20 stratified starts used in the Phase-2.5b n=20 comparison
(data/phase25_kappa_setup.json), so the gallery is consistent with the published statistics.
Three methods per start (surrogate / kappa_nudge / cma) with FULL trajectory + descriptor + control
recording. Assembles data/phase4_gallery_results.json.
  OMP_NUM_THREADS=1 ... python phase4_gallery_run.py --nworkers 10 --budget 18
"""
import argparse
import glob
import json
import os
import subprocess
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = os.path.join(ROOT, "fusion-env", "Scripts", "python.exe")
WORKER = os.path.join(ROOT, "experiments", "phase4_gallery_worker.py")
BLAS_ENV = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
OUT = os.path.join(ROOT, "data", "phase4_gallery_results.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nworkers", type=int, default=10)
    ap.add_argument("--budget", type=int, default=18)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    if not args.resume:
        for fn in glob.glob(os.path.join(ROOT, "data", "phase4_gallery_chunk_*.json")):
            os.remove(fn)
    env = dict(os.environ)
    for k in BLAS_ENV:
        env[k] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    logdir = os.path.join(ROOT, "data", "phase2_logs"); os.makedirs(logdir, exist_ok=True)
    procs = []
    t0 = time.time()
    for ch in range(args.nworkers):
        logf = open(os.path.join(logdir, f"p4gallery_chunk_{ch}.log"), "a" if args.resume else "w")
        p = subprocess.Popen([PY, WORKER, "--chunk", str(ch), "--nchunks", str(args.nworkers),
                              "--budget", str(args.budget)],
                             stdout=logf, stderr=subprocess.STDOUT, env=env, cwd=ROOT)
        procs.append((p, logf))
    done = 0
    while done < len(procs):
        time.sleep(15)
        done = sum(1 for p, _ in procs if p.poll() is not None)
    for _, lf in procs:
        lf.close()
    recs = []
    for fn in sorted(glob.glob(os.path.join(ROOT, "data", "phase4_gallery_chunk_*.json"))):
        recs += json.load(open(fn))["recs"]
    json.dump(dict(n=len(recs), budget=args.budget, recs=recs), open(OUT, "w"))
    print(f"\nALL done in {(time.time()-t0)/60:.1f} min; {len(recs)} runs -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
