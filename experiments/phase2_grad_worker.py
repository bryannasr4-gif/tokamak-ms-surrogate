"""
phase2_grad_worker.py -- one chunk of the Phase-2 gradient-verification probe set.

For each assigned BASE equilibrium (a held-out dataset row, replayed by its controls) we solve
the base and a finite-difference perturbation of EACH of the 16 live controls, recording the
realized SHAPE descriptors (20) + log m_s for the base and every perturbation. From these the
analyzer (phase2_grad_analyze.py) builds:
  * the realized shape Jacobian J = d(shape)/d(u)   (20x16, finite differences)
  * the true control-space gradient g_true = d(log m_s)/d(u)  (16, finite differences)
  * a directional-derivative test along the REALIZED shape directions (the only realizable ones)
and compares them to the surrogate's autodiff gradients. Because labels are bit-reproducible at
the locked protocol (OMP=1), these finite differences are clean (no label-noise floor).

Round-robin chunking over data/phase2_grad_bases.json. Run thread-pinned (OMP=1).
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


def solve_point(tok, ctrl):
    """Forward-label one control vector; return {shape feats..., logms, gamma, m_s} or None."""
    try:
        rec = L.forward_label(tok, ctrl["active_currents"], ctrl["paxis"], ctrl["Ip"],
                              ctrl["fvac"], ctrl["alpha_m"], ctrl["alpha_n"], fix_n_modes=40)
        if not np.isfinite(rec.get("m_s", np.nan)) or rec["m_s"] <= 0:
            return None
        out = {f: float(rec[f]) for f in D.SHAPE_FEATURES}
        out["m_s"] = float(rec["m_s"]); out["logms"] = float(np.log(rec["m_s"]))
        out["gamma"] = float(rec["gamma"])
        return out
    except Exception:
        return None


def ctrl_from_u(u, idx_map):
    """Build a forward_label control dict from the 16-vector u (CONTROL_FEATURES order)."""
    I = np.zeros(12)
    I[0] = 5000.0  # Solenoid fixed
    prof = {}
    for name, val in zip(D.CONTROL_FEATURES, u):
        if name.startswith("I_"):
            I[idx_map[name]] = val
        else:
            prof[name] = val
    return dict(active_currents=I, paxis=prof["paxis"], Ip=prof["Ip_target"],
                fvac=prof["fvac"], alpha_m=prof["alpha_m"], alpha_n=prof["alpha_n"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk", type=int, default=0)
    ap.add_argument("--nchunks", type=int, default=1)
    args = ap.parse_args()

    with open(os.path.join(ROOT, "data", "phase2_grad_bases.json")) as f:
        blob = json.load(f)
    bases, steps = blob["bases"], np.array(blob["steps"])
    idx_map = {f"I_{c}": i for i, c in enumerate(D.ACTIVE_COILS)}
    mine = [b for k, b in enumerate(bases) if k % args.nchunks == args.chunk]

    tok = L.load_machine()
    out = os.path.join(ROOT, "data", f"phase2_grad_chunk_{args.chunk}.json")
    recs = []
    t0 = time.time()
    for b in mine:
        u0 = np.array(b["u"], float)
        base = solve_point(tok, ctrl_from_u(u0, idx_map))
        rec = dict(idx=b["idx"], regime=b["regime"], split=b["split"], u=u0.tolist(),
                   base=base, probes=[])
        if base is not None:
            for k in range(len(u0)):
                up = u0.copy(); up[k] += steps[k]
                pt = solve_point(tok, ctrl_from_u(up, idx_map))
                rec["probes"].append(dict(k=k, du=float(steps[k]), pt=pt))
        recs.append(rec)
        nok = sum(1 for p in rec["probes"] if p["pt"] is not None)
        print(f"[c{args.chunk}] idx={b['idx']} {b['regime']:11s} base={'ok' if base else 'FAIL'} "
              f"probes_ok={nok}/{len(u0)} ({len(recs)}/{len(mine)}, "
              f"{(time.time()-t0)/max(len(recs),1):.0f}s/base)", flush=True)
        with open(out, "w") as f:
            json.dump(dict(chunk=args.chunk, recs=recs), f)
    print(f"[c{args.chunk}] DONE {len(recs)} bases in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
