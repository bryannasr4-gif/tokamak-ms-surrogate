"""s4a_li_worker.py -- one round-robin chunk of unit S4a (zero-l_i gradient ablation).

Frozen design: data/audit/strategy/S4A_S3C_DESIGN.md (unit S4a) + AMENDMENT A1 (council-before).
Setup is TRANSCRIBED VERBATIM from phase4_gallery_worker.py so each ablation arm differs from the
banked gallery surrogate arm in exactly the intended way.

  --mode smoke       : zero_feat_i=None  -> must reproduce the banked gallery bit-exactly (ALL 20)
  --mode ablate      : ARM A, gradient l_i channel zeroed (the manuscript's claim)
  --mode ablate_full : ARM B, gradient zeroed AND line-search scoring pins l_i at its start value
                       (the 'pure' ablation required by amendment A1)
Resume-safe: per-chunk file written atomically after every job.
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
import phase2_model as M
import phase2_data as D
import phase2_dim_lib as DL
import phase4_gallery_lib as G

LI_I = D.SHAPE_FEATURES.index("li")          # 18, asserted by the preflight
SHAM_SEL = os.path.join(ROOT, "data", "s4a_sham_selection.json")


def modes():
    """mode -> (zero_feat_i, mask_value, out-stem, method). The sham channel (amendment A1-12) is
    read from the FROZEN selection file, which is written from gradient geometry alone before any
    ablation arm runs."""
    m = {"smoke":       (None, False, "s4a_smoke_chunk",  "surrogate"),
         "ablate":      (LI_I, False, "s4a_li_chunk",     "surrogate_zero_li"),
         "ablate_full": (LI_I, True,  "s4a_lifull_chunk", "surrogate_zero_li_value")}
    if os.path.exists(SHAM_SEL):
        s = json.load(open(SHAM_SEL))
        m["sham"] = (int(s["sham_feat_i"]), False, "s4a_sham_chunk",
                     f"surrogate_zero_{s['sham_feat']}")
    return m


MODES = modes()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk", type=int, default=0)
    ap.add_argument("--nchunks", type=int, default=1)
    ap.add_argument("--budget", type=int, default=18)
    ap.add_argument("--mode", choices=["smoke", "ablate", "ablate_full", "sham"], required=True)
    args = ap.parse_args()
    if args.mode not in MODES:
        raise SystemExit(f"mode '{args.mode}' requires the frozen sham selection ({SHAM_SEL})")
    zero_feat_i, mask_value, stem, method = MODES[args.mode]

    S = json.load(open(os.path.join(ROOT, "data", "phase25_kappa_setup.json")))
    starts = S["starts"]
    mu = np.array(S["mu"]); std = np.array(S["std"]); V = np.array(S["V"])
    lo = np.array(S["box_lo"]); hi = np.array(S["box_hi"])
    d = 12

    jobs = list(range(len(starts)))          # all 20 starts in every mode (A1 fix: full smoke)
    mine = [si for k, si in enumerate(jobs) if k % args.nchunks == args.chunk]
    out = os.path.join(ROOT, "data", f"{stem}_{args.chunk}.json")

    tok = L.load_machine()
    models, _ = M.load_ensemble()
    smap, _ = M.load_shapemap()

    recs, done = [], set()
    if os.path.exists(out):
        try:
            recs = json.load(open(out)).get("recs", [])
            done = {r["start_i"] for r in recs}
        except Exception:
            recs, done = [], set()
    mine = [si for si in mine if si not in done]

    t0 = time.time()
    for si in mine:
        s = starts[si]
        ds = DL.DesignSpace(mu, std, V, lo, hi, np.array(s["u"]), d)
        m = 0.05 * (ds.box_hi - ds.box_lo)
        ds.box_lo = np.minimum(ds.box_lo, ds.x0 - m)
        ds.box_hi = np.maximum(ds.box_hi, ds.x0 + m)
        try:
            r = G.run_surrogate(tok, models, smap, ds, args.budget, s["kappa_start"], None,
                                kind="surrogate", seed=1000 + 7 * si + d,
                                zero_feat_i=zero_feat_i, mask_value=mask_value)
            r.update(dict(start_i=si, method=method, regime=s["regime"], idx=s.get("idx"),
                          zero_feat_i=(None if zero_feat_i is None else int(zero_feat_i)),
                          mask_value=bool(mask_value)))
        except Exception as e:
            r = dict(start_i=si, method="error", regime=s["regime"], idx=s.get("idx"),
                     best_ms=0.0, gain=0.0, n_solves=args.budget,
                     error=f"{type(e).__name__}:{str(e)[:100]}")
        recs.append(r)
        dc = r.get("dir_cos") or []
        cosmed = float(np.median([c[0] for c in dc])) if dc else float("nan")
        print(f"[c{args.chunk}] start{si:2d} {args.mode:11s} ms0={s['m_s_start']:.3f} "
              f"best={r.get('best_ms', 0):.4f} gain={r.get('gain', 0):+.4f} "
              f"kdrift={r.get('kappa_drift', float('nan')):.4f} n={r.get('n_solves', '?')} "
              f"cos={cosmed:.5f} ({(time.time() - t0) / 60:.1f}min)", flush=True)
        tmp = out + ".tmp"
        json.dump(dict(chunk=args.chunk, mode=args.mode, recs=recs), open(tmp, "w"))
        os.replace(tmp, out)
    print(f"[c{args.chunk}] DONE {len(recs)} in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
