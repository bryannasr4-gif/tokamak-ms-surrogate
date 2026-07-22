"""Launch the Tier-1 in-distribution control across N BLAS-pinned chunk workers (see tier1_indist_worker)."""
import argparse, os, subprocess, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = os.path.join(ROOT, "fusion-env", "Scripts", "python.exe")
WORKER = os.path.join(ROOT, "experiments", "tier1_indist_worker.py")
BLAS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
ap = argparse.ArgumentParser(); ap.add_argument("--nchunks", type=int, default=6); args = ap.parse_args()
env = dict(os.environ)
for k in BLAS:
    env[k] = "1"
env["PYTHONIOENCODING"] = "utf-8"
logdir = os.path.join(ROOT, "data", "tier1_logs"); os.makedirs(logdir, exist_ok=True)
procs = []
for ch in range(args.nchunks):
    lf = open(os.path.join(logdir, f"indist_{ch}.log"), "w")
    p = subprocess.Popen([PY, WORKER, "--chunk", str(ch), "--nchunks", str(args.nchunks)],
                         stdout=lf, stderr=subprocess.STDOUT, env=env, cwd=ROOT)
    procs.append((p, lf)); print(f"launched indist chunk {ch} (pid {p.pid})", flush=True)
t0 = time.time(); done = 0
while done < len(procs):
    time.sleep(10); done = sum(1 for p, _ in procs if p.poll() is not None)
for _, lf in procs:
    lf.close()
print(f"ALL {len(procs)} indist workers finished in {(time.time()-t0)/60:.1f} min.", flush=True)
