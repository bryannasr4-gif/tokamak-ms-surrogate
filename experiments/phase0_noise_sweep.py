"""
phase0_noise_sweep.py -- Phase 0, Part 2: quantify the m_s / gamma reproducibility floor.

One-at-a-time (OAT) sweep around the candidate locked protocol
  baseline = {threads=1, tol=1e-6, grid=65x65, modes=40}
along four numerical axes:
  - BLAS threads {1,2,4,8}   x 10 shapes   (HEADLINE: the dominant ~10% floor)
  - inverse tol  {1e-6,1e-8,1e-10} x 3 shapes x threads{1,8}  (does tighter tol shrink the floor?)
  - grid {65x65,65x129,129x129}    x 3 shapes (threads=1)     (resolution shift + cost)
  - modes {20,40,80}               x 3 shapes (threads=1)     (mode-count convergence)

Each job is an ISOLATED cold subprocess (phase0_solve_one.py) that loads the serialized
machine and does ONE solve -> no warm-start path dependence (audit 4.2). BLAS threads are
set per-subprocess via env. A thread-budget scheduler keeps sum(active threads) <= BUDGET so
1-thread jobs pack onto the 16 cores while 8-thread jobs still get room. m_s VALUES are
independent of contention (BLAS thread COUNT fixes the FP reduction order regardless of cores).

Saves incrementally to data/phase0_noise_sweep.json. Run thread-pinned-irrelevant (the parent
sets each child's threads); just: PYTHONIOENCODING=utf-8 python experiments/phase0_noise_sweep.py
"""
import json
import os
import subprocess
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = os.path.join(ROOT, "fusion-env", "Scripts", "python.exe")
WORKER = os.path.join(ROOT, "experiments", "phase0_solve_one.py")
OUT = os.path.join(ROOT, "data", "phase0_noise_sweep.json")

BUDGET = 14            # max sum of nominal threads running at once (16 logical cpus)
JOB_TIMEOUT = 1200     # s per job (129x129 can be minutes)

# 10 representative shapes spanning the controllable->marginal m_s range, incl. wall-gap dR.
SHAPES = [
    (0.85, 0.0), (0.88, 0.0), (0.91, 0.0), (0.94, 0.0), (0.97, 0.0),
    (1.00, 0.0), (1.02, 0.0),
    (0.94, 0.03), (0.94, -0.03), (0.88, 0.03),
]
# 3 shapes used for the secondary (tol/grid/modes) axes: high / mid / low m_s.
REP = [(0.88, 0.0), (0.94, 0.0), (1.00, 0.0)]

BASE = dict(tol=1e-6, nx=65, ny=65, modes=40)


def make_jobs():
    jobs = {}  # key -> dict, dedup identical configs

    def add(zscale, dR, threads, tol, nx, ny, modes, axis):
        key = (zscale, dR, threads, tol, nx, ny, modes)
        if key not in jobs:
            jobs[key] = dict(zscale=zscale, dR=dR, threads=threads, tol=tol,
                             nx=nx, ny=ny, modes=modes, axes=[axis])
        elif axis not in jobs[key]["axes"]:
            jobs[key]["axes"].append(axis)

    # threads axis: all 10 shapes x {1,2,4,8}
    for (z, d) in SHAPES:
        for th in (1, 2, 4, 8):
            add(z, d, th, BASE["tol"], BASE["nx"], BASE["ny"], BASE["modes"], "threads")
    # tol axis: rep shapes x tol x {1,8}
    for (z, d) in REP:
        for tol in (1e-6, 1e-8, 1e-10):
            for th in (1, 8):
                add(z, d, th, tol, BASE["nx"], BASE["ny"], BASE["modes"], "tol")
    # grid axis: rep shapes x grids x threads=1
    for (z, d) in REP:
        for (nx, ny) in ((65, 65), (65, 129), (129, 129)):
            add(z, d, 1, BASE["tol"], nx, ny, BASE["modes"], "grid")
    # modes axis: rep shapes x modes x threads=1
    for (z, d) in REP:
        for m in (20, 40, 80):
            add(z, d, 1, BASE["tol"], BASE["nx"], BASE["ny"], m, "modes")

    return list(jobs.values())


def launch(job, tmpdir, idx):
    env = dict(os.environ)
    th = str(job["threads"])
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        env[k] = th
    env["PYTHONIOENCODING"] = "utf-8"
    outf = open(os.path.join(tmpdir, f"job_{idx}.out"), "w+")
    cmd = [PY, WORKER, "--zscale", str(job["zscale"]), "--dR", str(job["dR"]),
           "--threads", th, "--tol", repr(job["tol"]),
           "--nx", str(job["nx"]), "--ny", str(job["ny"]), "--modes", str(job["modes"])]
    p = subprocess.Popen(cmd, stdout=outf, stderr=subprocess.STDOUT, env=env, cwd=ROOT)
    return dict(proc=p, outf=outf, job=job, idx=idx, t0=time.time())


def harvest(run, results):
    run["outf"].flush(); run["outf"].seek(0)
    text = run["outf"].read(); run["outf"].close()
    rec = None
    for line in text.splitlines():
        if line.startswith("RESULT "):
            rec = json.loads(line[len("RESULT "):])
    if rec is None:
        rec = dict(run["job"], ok=False, error="no RESULT line", tail=text[-400:])
    rec["axes"] = run["job"]["axes"]
    rec["wall_s"] = round(time.time() - run["t0"], 1)
    results.append(rec)
    tag = f"z={rec['zscale']:.2f} dR={rec.get('dR',0):+.2f} th={rec['threads']} " \
          f"tol={rec['tol']:.0e} {rec['nx']}x{rec['ny']} m{rec['modes']}"
    if rec.get("ok"):
        print(f"[{len(results):3d}] OK  {tag}  m_s={rec['m_s']:.4f} gamma={rec['gamma']:7.1f} "
              f"({rec['wall_s']:.0f}s)", flush=True)
    else:
        print(f"[{len(results):3d}] FAIL {tag}  {rec.get('error')}", flush=True)
    with open(OUT, "w") as f:
        json.dump(dict(budget=BUDGET, shapes=SHAPES, rep=REP, base=BASE, results=results), f, indent=2)


def main():
    jobs = make_jobs()
    # schedule heaviest first (129x129, modes=80) so they overlap with many light jobs
    jobs.sort(key=lambda j: (j["nx"] * j["ny"], j["modes"], j["threads"]), reverse=True)
    print(f"{len(jobs)} unique jobs; BUDGET={BUDGET} threads; out -> {OUT}")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    results = []
    tmpdir = tempfile.mkdtemp(prefix="phase0sweep_")
    t_start = time.time()
    pending = list(jobs)
    running = []
    used = 0
    idx = 0
    while pending or running:
        # launch while budget allows (always allow at least one job to run)
        progressed = True
        while progressed:
            progressed = False
            for j in list(pending):
                if used + j["threads"] <= BUDGET or not running:
                    pending.remove(j)
                    run = launch(j, tmpdir, idx); idx += 1
                    running.append(run); used += j["threads"]
                    progressed = True
                    if not (pending and used < BUDGET):
                        break
        # poll
        time.sleep(1.0)
        for run in list(running):
            done = run["proc"].poll() is not None
            timedout = (time.time() - run["t0"]) > JOB_TIMEOUT
            if timedout and not done:
                run["proc"].kill(); done = True
            if done:
                running.remove(run); used -= run["job"]["threads"]
                harvest(run, results)
    n_ok = sum(1 for r in results if r.get("ok"))
    print(f"\nDONE {n_ok}/{len(results)} ok in {(time.time()-t_start)/60:.1f} min -> {OUT}")


if __name__ == "__main__":
    main()
