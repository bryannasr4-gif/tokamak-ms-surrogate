"""
phase25_relabel_worker.py -- one chunk of the Phase-2.5 combined re-label + Machine-B pass.

For each assigned dataset_v1 row, replay its CONTROLS through:
  * Machine A (original 138-passive MAST-U) at the CONVERGED 80 modes -> clean m_s_A80 (fixes the
    -13.5% 40-mode bias), plus the shape descriptors (recomputed) and gamma/leuer.
  * Machine B (coil-cases removed, 56 passives) at all 56 modes -> m_s_B on the SAME shape (the
    cross-machine conducting-structure-dependence label).
Because passives sit at zero current in the static GS solve, both machines see the same plasma
shape; only the linearisation (m_s) differs. Writes data/phase25_relabel_chunk_{chunk}.json.
Round-robin chunking; run thread-pinned (OMP=1).
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
import phase0_lib as P0
import phase15_lib as L
import phase2_data as D

SER_A = os.path.join(ROOT, "machine_configs", "MAST-U", "serialized_tokamak.pkl")
SER_B = os.path.join(ROOT, "machine_configs", "MAST-U-B", "serialized_tokamak_B.pkl")
MODES_A = 80          # converged count for Machine A (138 passives)
MODES_B = 200         # clamps to all 56 passives of Machine B = converged reference for B


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk", type=int, default=0)
    ap.add_argument("--nchunks", type=int, default=1)
    args = ap.parse_args()

    with open(os.path.join(ROOT, "data", "phase25_relabel_rows.json")) as f:
        rows = json.load(f)["rows"]
    mine = [r for k, r in enumerate(rows) if k % args.nchunks == args.chunk]

    tokA = P0.load_machine(SER_A)
    tokB = P0.load_machine(SER_B)
    out = os.path.join(ROOT, "data", f"phase25_relabel_chunk_{args.chunk}.json")
    recs = []
    t0 = time.time()
    for r in mine:
        I = np.array(r["I"], float)
        rec = dict(idx=r["idx"], split=r["split"], m_s_A40=r["m_s_A40"])
        try:
            rA = L.forward_label(tokA, I, r["paxis"], r["Ip"], r["fvac"], r["alpha_m"], r["alpha_n"],
                                 fix_n_modes=MODES_A)
            for f in D.SHAPE_FEATURES:
                rec[f] = float(rA[f])
            rec.update(m_s_A80=float(rA["m_s"]), gamma_A80=float(rA["gamma"]), leuer_A=float(rA["leuer"]))
        except Exception as e:
            rec["errA"] = f"{type(e).__name__}:{str(e)[:50]}"
        try:
            rB = L.forward_label(tokB, I, r["paxis"], r["Ip"], r["fvac"], r["alpha_m"], r["alpha_n"],
                                 fix_n_modes=MODES_B)
            rec.update(m_s_B=float(rB["m_s"]), gamma_B=float(rB["gamma"]), leuer_B=float(rB["leuer"]),
                       kappa_B=float(rB["kappa"]))
        except Exception as e:
            rec["errB"] = f"{type(e).__name__}:{str(e)[:50]}"
        recs.append(rec)
        if len(recs) % 10 == 0:
            print(f"[c{args.chunk}] {len(recs)}/{len(mine)} "
                  f"msA80={rec.get('m_s_A80')} msB={rec.get('m_s_B')} "
                  f"({(time.time()-t0)/max(len(recs),1):.0f}s/shape)", flush=True)
        if len(recs) % 5 == 0:
            with open(out, "w") as f:
                json.dump(dict(chunk=args.chunk, recs=recs), f)
    with open(out, "w") as f:
        json.dump(dict(chunk=args.chunk, recs=recs), f)
    print(f"[c{args.chunk}] DONE {len(recs)} in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
