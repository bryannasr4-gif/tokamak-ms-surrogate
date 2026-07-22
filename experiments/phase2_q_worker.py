"""
phase2_q_worker.py -- one chunk of the q95 re-solve over dataset_v1_80 (Phase-2.5b, Task B).
Replays each assigned shape's stored controls through a LIGHT forward-only solve and extracts
q95/qmin/q05 (no m_s linearisation). Round-robin chunking; thread-pinned (OMP=1). Saves
incrementally to data/phase2_q_chunk_{chunk}.json.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "experiments"))
import phase15_lib as L
import phase2_data as D
import phase2_q_lib as Q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk", type=int, default=0)
    ap.add_argument("--nchunks", type=int, default=1)
    args = ap.parse_args()

    with open(os.path.join(ROOT, "data", "phase2_q_rows.json")) as f:
        rows = json.load(f)["rows"]
    mine = [r for k, r in enumerate(rows) if k % args.nchunks == args.chunk]

    tok = L.load_machine()
    out = os.path.join(ROOT, "data", f"phase2_q_chunk_{args.chunk}.json")
    recs = []
    t0 = time.time()
    for r in mine:
        I = np.array(r["I"], float)
        rec = dict(idx=r["idx"], kappa_stored=r["kappa"])
        try:
            q = Q.forward_q(tok, I, r["paxis"], r["Ip"], r["fvac"], r["alpha_m"], r["alpha_n"])
            rec.update(q95=q["q95"], qmin=q["qmin"], q05=q["q05"],
                       qmin_psin=q["q_at_grid_min_psin"],
                       kappa_check=q["kappa_check"],
                       kappa_match=float(abs(q["kappa_check"] - r["kappa"])))
        except Exception as e:
            rec["err"] = f"{type(e).__name__}:{str(e)[:60]}"
        recs.append(rec)
        if len(recs) % 20 == 0:
            print(f"[c{args.chunk}] {len(recs)}/{len(mine)} idx={rec['idx']} "
                  f"q95={rec.get('q95')} kmatch={rec.get('kappa_match')} "
                  f"({(time.time()-t0)/max(len(recs),1):.1f}s/shape)", flush=True)
            with open(out, "w") as f:
                json.dump(dict(chunk=args.chunk, recs=recs), f)
    with open(out, "w") as f:
        json.dump(dict(chunk=args.chunk, recs=recs), f)
    nfail = sum(1 for x in recs if "err" in x)
    print(f"[c{args.chunk}] DONE {len(recs)} ({nfail} failed) in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
