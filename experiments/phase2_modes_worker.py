"""
phase2_modes_worker.py -- one chunk of the Phase-2 mode-convergence study.

Replays a stratified subset of dataset_v1 rows (by their stored CONTROLS) through the true
solver at several retained-passive-mode counts {40,80,120,138} to measure the residual
systematic mode-truncation drift of m_s (the dominant Phase-0 floor; labels are at 40).
138 = all passive structures retained = the fully-converged reference.

Each chunk processes round-robin row indices (idx % nchunks == chunk) from the shared subset
file data/phase2_modes_subset.json and writes data/phase2_modes_chunk_{chunk}.json.
Run thread-pinned (OMP=1) per process, like phase15_generate.py.
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

MODES = [40, 80, 120, 138]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk", type=int, default=0)
    ap.add_argument("--nchunks", type=int, default=1)
    args = ap.parse_args()

    with open(os.path.join(ROOT, "data", "phase2_modes_subset.json")) as f:
        subset = json.load(f)["rows"]      # list of {idx, regime, m_s_stored, replay:{...}}
    mine = [r for k, r in enumerate(subset) if k % args.nchunks == args.chunk]

    tok = L.load_machine()
    out = os.path.join(ROOT, "data", f"phase2_modes_chunk_{args.chunk}.json")
    recs = []
    t0 = time.time()
    for r in mine:
        rep = r["replay"]
        I = np.array(rep["active_currents"], float)
        row = dict(idx=r["idx"], regime=r["regime"], m_s_stored=r["m_s_stored"],
                   kappa_stored=r.get("kappa_stored"), ms={}, gamma={})
        for m in MODES:
            try:
                res = L.forward_label(tok, I, rep["paxis"], rep["Ip"], rep["fvac"],
                                      rep["alpha_m"], rep["alpha_n"], fix_n_modes=m)
                row["ms"][str(m)] = float(res["m_s"])
                row["gamma"][str(m)] = float(res["gamma"])
            except Exception as e:
                row["ms"][str(m)] = None
                row["gamma"][str(m)] = None
                row.setdefault("errs", {})[str(m)] = f"{type(e).__name__}:{str(e)[:60]}"
        recs.append(row)
        print(f"[c{args.chunk}] idx={r['idx']} {r['regime']:11s} "
              f"ms40={row['ms'].get('40')} ms138={row['ms'].get('138')} "
              f"({len(recs)}/{len(mine)}, {(time.time()-t0)/max(len(recs),1):.0f}s/shape)",
              flush=True)
        with open(out, "w") as f:
            json.dump(dict(chunk=args.chunk, modes=MODES, recs=recs), f)
    print(f"[c{args.chunk}] DONE {len(recs)} shapes in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
