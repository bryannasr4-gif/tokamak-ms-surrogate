"""
phase2_q_run.py -- launch the q95 forward-only re-solve over all dataset_v1_80 shapes, then
assemble data/dataset_v1_80q.parquet (= dataset_v1_80 + q95/qmin/q05 columns) and report q
ranges + corr(q95, m_s) / corr(q95, kappa). Background; ~1-2 h with 11 workers.
  python phase2_q_run.py --nworkers 11
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
WORKER = os.path.join(ROOT, "experiments", "phase2_q_worker.py")
BLAS_ENV = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
DS80 = os.path.join(ROOT, "data", "dataset_v1_80.parquet")
DS80Q = os.path.join(ROOT, "data", "dataset_v1_80q.parquet")


def build_rows():
    import pandas as pd
    df = pd.read_parquet(DS80)
    rows = []
    for _, r in df.iterrows():
        I = np.array([5000.0] + [float(r[f"I_{c}"]) for c in D.ACTIVE_COILS if c != "Solenoid"])
        rows.append(dict(idx=int(r["idx"]), kappa=float(r["kappa"]), I=I.tolist(),
                         paxis=float(r["paxis"]), Ip=float(r["Ip_target"]), fvac=float(r["fvac"]),
                         alpha_m=float(r["alpha_m"]), alpha_n=float(r["alpha_n"])))
    with open(os.path.join(ROOT, "data", "phase2_q_rows.json"), "w") as f:
        json.dump(dict(n=len(rows), rows=rows), f)
    print(f"{len(rows)} shapes to re-solve for q", flush=True)


def assemble():
    import pandas as pd
    recs = []
    for fn in sorted(glob.glob(os.path.join(ROOT, "data", "phase2_q_chunk_*.json"))):
        recs += json.load(open(fn))["recs"]
    qdf = pd.DataFrame(recs)
    nfail = int(qdf["err"].notna().sum()) if "err" in qdf.columns else 0
    print(f"assembled {len(qdf)} q-records; {nfail} failed", flush=True)
    # kappa replay sanity
    if "kappa_match" in qdf.columns:
        km = qdf["kappa_match"].dropna()
        print(f"  kappa replay match: median={km.median():.2e} max={km.max():.2e} "
              f"(should be ~machine-zero; replay bit-exact)", flush=True)
    df = pd.read_parquet(DS80)
    qcols = ["idx", "q95", "qmin", "q05", "qmin_psin"]
    qkeep = qdf[[c for c in qcols if c in qdf.columns]].copy()
    merged = df.merge(qkeep, on="idx", how="left")
    nmiss = int(merged["q95"].isna().sum())
    print(f"  merged: {len(merged)} rows, {nmiss} missing q95 (failed re-solves)", flush=True)
    merged.to_parquet(DS80Q)

    # ---- report (resolved, physical sanity) ----
    g = merged.dropna(subset=["q95"])
    rep = dict(
        n=len(merged), n_q=len(g), n_fail=nmiss,
        q95=dict(min=float(g["q95"].min()), p05=float(g["q95"].quantile(0.05)),
                 median=float(g["q95"].median()), p95=float(g["q95"].quantile(0.95)),
                 max=float(g["q95"].max())),
        qmin=dict(min=float(g["qmin"].min()), median=float(g["qmin"].median()), max=float(g["qmin"].max())),
        q05=dict(min=float(g["q05"].min()), median=float(g["q05"].median()), max=float(g["q05"].max())),
        frac_q95_in_3_10=float(((g["q95"] >= 3) & (g["q95"] <= 10)).mean()),
        frac_qmin_gt_1=float((g["qmin"] > 1.0).mean()),
        corr_q95_ms=float(g["q95"].corr(g["m_s"])),
        corr_q95_logms=float(g["q95"].corr(np.log(g["m_s"]))),
        spearman_q95_ms=float(g["q95"].corr(g["m_s"], method="spearman")),
        corr_q95_kappa=float(g["q95"].corr(g["kappa"])),
        corr_qmin_ms=float(g["qmin"].corr(g["m_s"])),
    )
    # resolved corr(q95, m_s) by regime
    rep["by_regime"] = {}
    for name, lo, hi in D.REGIMES:
        sel = g[(g["m_s"] >= lo) & (g["m_s"] < hi)]
        if len(sel) >= 5:
            rep["by_regime"][name] = dict(n=int(len(sel)), q95_median=float(sel["q95"].median()),
                                          corr_q95_ms=float(sel["q95"].corr(sel["m_s"])))
    with open(os.path.join(ROOT, "data", "phase2_q_summary.json"), "w") as f:
        json.dump(rep, f, indent=2)
    print("\n=== q95 SUMMARY (dataset_v1_80q) ===")
    print(f"  n={rep['n_q']} ({rep['n_fail']} failed)")
    print(f"  q95:  min={rep['q95']['min']:.2f}  p05={rep['q95']['p05']:.2f}  median={rep['q95']['median']:.2f}"
          f"  p95={rep['q95']['p95']:.2f}  max={rep['q95']['max']:.2f}")
    print(f"  qmin: min={rep['qmin']['min']:.2f}  median={rep['qmin']['median']:.2f}  max={rep['qmin']['max']:.2f}")
    print(f"  frac q95 in [3,10] = {rep['frac_q95_in_3_10']*100:.1f}%   frac qmin>1 = {rep['frac_qmin_gt_1']*100:.1f}%")
    print(f"  corr(q95,m_s)={rep['corr_q95_ms']:+.3f}  corr(q95,log m_s)={rep['corr_q95_logms']:+.3f}"
          f"  Spearman={rep['spearman_q95_ms']:+.3f}  corr(q95,kappa)={rep['corr_q95_kappa']:+.3f}")
    for name, d in rep["by_regime"].items():
        print(f"    {name:11s} n={d['n']:4d}  q95_med={d['q95_median']:.2f}  corr(q95,m_s)={d['corr_q95_ms']:+.3f}")
    print(f"Saved {DS80Q} and data/phase2_q_summary.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nworkers", type=int, default=11)
    ap.add_argument("--assemble-only", action="store_true")
    args = ap.parse_args()
    if args.assemble_only:
        assemble()
        return
    for fn in glob.glob(os.path.join(ROOT, "data", "phase2_q_chunk_*.json")):
        os.remove(fn)
    build_rows()
    env = dict(os.environ)
    for k in BLAS_ENV:
        env[k] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    logdir = os.path.join(ROOT, "data", "phase2_logs")
    os.makedirs(logdir, exist_ok=True)
    procs = []
    t0 = time.time()
    for ch in range(args.nworkers):
        logf = open(os.path.join(logdir, f"q_chunk_{ch}.log"), "w")
        p = subprocess.Popen([PY, WORKER, "--chunk", str(ch), "--nchunks", str(args.nworkers)],
                             stdout=logf, stderr=subprocess.STDOUT, env=env, cwd=ROOT)
        procs.append((p, logf))
    done = 0
    while done < len(procs):
        time.sleep(15)
        done = sum(1 for p, _ in procs if p.poll() is not None)
    for _, lf in procs:
        lf.close()
    assemble()
    print(f"\nALL done in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
