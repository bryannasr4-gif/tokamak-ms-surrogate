"""
phase25_relabel_run.py -- launch the combined re-label (Machine A @80) + Machine B pass over all
3298 dataset_v1 shapes, then assemble data/dataset_v1_80.parquet (clean A labels) and
data/dataset_v2_B.parquet (Machine-B labels on matched shapes). Background; ~5 h.
  python phase25_relabel_run.py --nworkers 11
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
WORKER = os.path.join(ROOT, "experiments", "phase25_relabel_worker.py")
BLAS_ENV = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")


def build_rows():
    df = D.load()
    rows = []
    for idx, r in df.iterrows():
        rep = D.replay_kwargs(r)
        rows.append(dict(idx=int(idx), split=str(r["split"]), m_s_A40=float(r["m_s"]),
                         I=rep["active_currents"].tolist(), paxis=rep["paxis"], Ip=rep["Ip"],
                         fvac=rep["fvac"], alpha_m=rep["alpha_m"], alpha_n=rep["alpha_n"]))
    with open(os.path.join(ROOT, "data", "phase25_relabel_rows.json"), "w") as f:
        json.dump(dict(n=len(rows), rows=rows), f)
    print(f"{len(rows)} rows to re-label", flush=True)
    return len(rows)


def assemble():
    import pandas as pd
    recs = []
    for fn in sorted(glob.glob(os.path.join(ROOT, "data", "phase25_relabel_chunk_*.json"))):
        recs += json.load(open(fn))["recs"]
    df = pd.DataFrame(recs)
    # carry the controls + split from v1 by idx
    v1 = D.load().reset_index().rename(columns={"index": "idx"})
    ctrl_cols = D.CONTROL_FEATURES
    df = df.merge(v1[["idx"] + ctrl_cols], on="idx", how="left")
    ok = df[df["m_s_A80"].notna() & df["m_s_B"].notna()].copy()
    print(f"assembled {len(df)} rows; {len(ok)} with both A80 & B labels", flush=True)
    # dataset_v1_80: clean Machine-A labels at 80 modes (same shapes/splits as v1)
    a = ok.copy()
    a["m_s"] = a["m_s_A80"]; a["gamma"] = a["gamma_A80"]; a["leuer"] = a["leuer_A"]
    a.to_parquet(os.path.join(ROOT, "data", "dataset_v1_80.parquet"))
    # dataset_v2_B: Machine-B labels on matched shapes
    b = ok.copy()
    b["m_s"] = b["m_s_B"]; b["gamma"] = b["gamma_B"]; b["leuer"] = b["leuer_B"]
    b.to_parquet(os.path.join(ROOT, "data", "dataset_v2_B.parquet"))
    print(f"saved dataset_v1_80.parquet & dataset_v2_B.parquet ({len(ok)} rows each)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nworkers", type=int, default=11)
    args = ap.parse_args()
    for fn in glob.glob(os.path.join(ROOT, "data", "phase25_relabel_chunk_*.json")):
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
        logf = open(os.path.join(logdir, f"relabel_chunk_{ch}.log"), "w")
        p = subprocess.Popen([PY, WORKER, "--chunk", str(ch), "--nchunks", str(args.nworkers)],
                             stdout=logf, stderr=subprocess.STDOUT, env=env, cwd=ROOT)
        procs.append((p, logf))
        print(f"launched relabel chunk {ch} (pid {p.pid})", flush=True)
    done = 0
    while done < len(procs):
        time.sleep(20)
        done = sum(1 for p, _ in procs if p.poll() is not None)
    for _, lf in procs:
        lf.close()
    assemble()
    print(f"\nALL done in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
