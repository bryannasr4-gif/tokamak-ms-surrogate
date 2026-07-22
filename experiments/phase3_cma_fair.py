"""
phase3_cma_fair.py -- FAIR-CMA control answering the adversarial finding `cma-starved-too-few-
generations`: the main run's CMA used the DEFAULT popsize (=11 at d=12) so budget 30 bought only
~2.5 generations -- CMA never reached its exploitation phase. Here we give CMA a SMALLER popsize
(6 -> ~5 generations at budget 30, ~10 at budget 60), a GRADED out-of-range penalty, and also 2x
the budget, on the 10 MARGINAL starts (the discriminating regime). If a fair, well-resourced CMA
STILL cannot match the surrogate's marginal efficiency (9/10 reach in median 6 solves), the
amortization gap is not a CMA-starvation artifact.

Dual mode: launcher when run with no --chunk; worker when given --chunk (resume-safe, atomic write).
  ./fusion-env/Scripts/python.exe experiments/phase3_cma_fair.py --nworkers 10
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

PY = os.path.join(ROOT, "fusion-env", "Scripts", "python.exe")
SELF = os.path.abspath(__file__)
BLAS_ENV = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
CONFIGS = [dict(tag="cma_pop6_b30", popsize=6, budget=30),
           dict(tag="cma_pop6_b60", popsize=6, budget=60)]


def build():
    S = json.load(open(os.path.join(ROOT, "data", "phase3_setup.json")))
    marg = [i for i, s in enumerate(S["starts"]) if s["regime"] == "marginal"]
    jobs = []
    for si in marg:
        for cfg in CONFIGS:
            jobs.append(dict(job=len(jobs), start_i=si, **cfg, seed=4000 + 13 * si))
    json.dump(dict(jobs=jobs), open(os.path.join(ROOT, "data", "phase3_cmafair_jobs.json"), "w"))
    print(f"{len(marg)} marginal starts x {len(CONFIGS)} configs = {len(jobs)} jobs", flush=True)
    return jobs


def worker(chunk, nchunks):
    import phase15_lib as L
    import phase2_dim_lib as DL
    import phase3_lib as P3
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    S = json.load(open(os.path.join(ROOT, "data", "phase3_setup.json")))
    jobs = json.load(open(os.path.join(ROOT, "data", "phase3_cmafair_jobs.json")))["jobs"]
    mine = [j for k, j in enumerate(jobs) if k % nchunks == chunk]
    tok = L.load_machine()
    mu = np.array(S["mu"]); std = np.array(S["std"]); V = np.array(S["V"])
    lo = np.array(S["box_lo"]); hi = np.array(S["box_hi"]); d = S["d"]; ranges = S["ranges"]
    out = os.path.join(ROOT, "data", f"phase3_cmafair_chunk_{chunk}.json")
    recs, done = [], set()
    if os.path.exists(out):
        try:
            recs = json.load(open(out)).get("recs", [])
            done = {(r["start_i"], r["tag"]) for r in recs}
        except Exception:
            recs, done = [], set()
    mine = [j for j in mine if (j["start_i"], j["tag"]) not in done]
    t0 = time.time()
    for j in mine:
        s = S["starts"][j["start_i"]]
        ds = DL.DesignSpace(mu, std, V, lo, hi, np.array(s["u"]), d)
        mrg = 0.05 * (ds.box_hi - ds.box_lo)
        ds.box_lo = np.minimum(ds.box_lo, ds.x0 - mrg); ds.box_hi = np.maximum(ds.box_hi, ds.x0 + mrg)
        sranges = {}
        for f, (rlo, rhi) in ranges.items():
            v = s.get("desc", {}).get(f)
            sranges[f] = [rlo, rhi] if v is None else [min(rlo, v - 0.02 * (rhi - rlo)),
                                                       max(rhi, v + 0.02 * (rhi - rlo))]
        try:
            r = P3.run_cma_fair(tok, ds, j["budget"], sranges, s["m_s_start"], s["kappa_start"],
                                s["regime"], j["seed"], popsize=j["popsize"])
            err = None
        except Exception as e:
            r = dict(m_s_start=s["m_s_start"], best_ms=0.0, gain=0.0, n_solves=j["budget"],
                     reached_primary=False, solves_to_target={f"{t:.1f}": None for t in P3.TARGETS})
            err = f"{type(e).__name__}:{str(e)[:80]}"
        rec = dict(start_i=j["start_i"], tag=j["tag"], budget=j["budget"], popsize=j["popsize"],
                   regime=s["regime"], m_s_start=s["m_s_start"], best_ms=r["best_ms"],
                   gain=r["gain"], n_solves=r["n_solves"], reached_primary=r["reached_primary"],
                   solves_to_target=r["solves_to_target"], error=err)
        recs.append(rec)
        print(f"[c{chunk}] start{j['start_i']:2d} {j['tag']:12s} start={s['m_s_start']:.3f} "
              f"best={rec['best_ms']:.3f} reached={rec['reached_primary']} "
              f"s2t={rec['solves_to_target']['1.0']} nsolve={rec['n_solves']}", flush=True)
        tmp = out + ".tmp"
        with open(tmp, "w") as f:
            json.dump(dict(chunk=chunk, recs=recs), f)
        os.replace(tmp, out)
    print(f"[c{chunk}] DONE {len(recs)} in {(time.time() - t0) / 60:.1f} min", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk", type=int, default=-1)
    ap.add_argument("--nchunks", type=int, default=1)
    ap.add_argument("--nworkers", type=int, default=10)
    args = ap.parse_args()
    if args.chunk >= 0:
        worker(args.chunk, args.nchunks)
        return
    for fn in glob.glob(os.path.join(ROOT, "data", "phase3_cmafair_chunk_*.json")):
        os.remove(fn)
    build()
    env = dict(os.environ)
    for k in BLAS_ENV:
        env[k] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    logdir = os.path.join(ROOT, "data", "phase2_logs"); os.makedirs(logdir, exist_ok=True)
    procs = []
    t0 = time.time()
    for ch in range(args.nworkers):
        logf = open(os.path.join(logdir, f"phase3_cmafair_chunk_{ch}.log"), "w")
        p = subprocess.Popen([PY, SELF, "--chunk", str(ch), "--nchunks", str(args.nworkers)],
                             stdout=logf, stderr=subprocess.STDOUT, env=env, cwd=ROOT)
        procs.append((p, logf))
    done = 0
    while done < len(procs):
        time.sleep(15)
        done = sum(1 for p, _ in procs if p.poll() is not None)
    for _, lf in procs:
        lf.close()
    recs = []
    for fn in sorted(glob.glob(os.path.join(ROOT, "data", "phase3_cmafair_chunk_*.json"))):
        recs += json.load(open(fn))["recs"]
    json.dump(dict(n=len(recs), recs=recs), open(os.path.join(ROOT, "data", "phase3_cmafair_results.json"), "w"))
    print(f"\nALL fair-CMA done in {(time.time() - t0) / 60:.1f} min; {len(recs)} runs", flush=True)


if __name__ == "__main__":
    main()
