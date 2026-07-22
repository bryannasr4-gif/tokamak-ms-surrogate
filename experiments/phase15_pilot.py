"""
phase15_pilot.py -- de-risk the forward-sampling distribution BEFORE the full run.

Phase A (fast, no linearisation): build anchors, draw N control vectors, forward-solve only.
  Measures: convergence yield, DIVERTED fraction, and shape-descriptor coverage (kappa, delta,
  gaps, li, betap). This is the make-or-break of Phase 1.5 -- can we forward-sample a clean,
  well-covered diverted-ST dataset?
Phase B (slow): take M of the converged controls and add the m_s/gamma linearisation, to
  confirm the LABEL (m_s) spans controllable->marginal and to time the full per-label cost.

Anchors are pickled to data/phase15_anchors.pkl for reuse. Run thread-pinned (OMP=1).
"""
import os
import pickle
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

ANCHOR_PKL = os.path.join(ROOT, "data", "phase15_anchors.pkl")
ZSCALES = [0.80, 0.85, 0.90, 0.94, 0.97, 1.00, 1.02]
DRS = [-0.04, 0.0, 0.04]


def get_anchors(tok, rebuild=False):
    if os.path.exists(ANCHOR_PKL) and not rebuild:
        with open(ANCHOR_PKL, "rb") as f:
            return pickle.load(f)
    print(f"Building anchors over {len(ZSCALES)}x{len(DRS)} (zscale,dR) grid ...", flush=True)
    t0 = time.time()
    anchors = L.build_anchors(tok, ZSCALES, DRS)
    anchors = [a for a in anchors if a["I"] is not None and np.all(np.isfinite(a["I"]))]
    os.makedirs(os.path.dirname(ANCHOR_PKL), exist_ok=True)
    with open(ANCHOR_PKL, "wb") as f:
        pickle.dump(anchors, f)
    print(f"Built {len(anchors)} anchors in {time.time()-t0:.0f}s (kappa "
          f"{min(a['kappa'] for a in anchors):.3f}-{max(a['kappa'] for a in anchors):.3f})",
          flush=True)
    return anchors


def main():
    n_fast = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    n_lin = int(sys.argv[2]) if len(sys.argv) > 2 else 12
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 0

    tok = L.load_machine()
    anchors = get_anchors(tok)
    rng = np.random.default_rng(seed)

    # ---- Phase A: forward-solve only (fast) -------------------------------
    print(f"\n=== Phase A: {n_fast} forward samples (no linearisation) ===", flush=True)
    ok, fails = [], {}
    t0 = time.time()
    for k in range(n_fast):
        c = L.sample_control(rng, anchors)
        try:
            rec = L.forward_label(tok, **c, with_linearisation=False)
            ok.append(rec)
        except Exception as e:
            key = str(e).split("(")[0][:40]
            fails[key] = fails.get(key, 0) + 1
    dtA = time.time() - t0
    print(f"  yield: {len(ok)}/{n_fast} converged+diverted in {dtA:.0f}s "
          f"({dtA/n_fast:.2f}s/sample)", flush=True)
    if fails:
        print("  rejections:", {k: v for k, v in sorted(fails.items(), key=lambda x: -x[1])}, flush=True)
    if ok:
        def span(key):
            v = np.array([r[key] for r in ok if np.isfinite(r.get(key, np.nan))])
            return f"{v.min():.3f}..{v.max():.3f} (med {np.median(v):.3f})" if v.size else "n/a"
        for key in ["kappa", "delta", "sq_uo", "Rgeo", "a", "gap_inner", "gap_outer",
                    "gap_top", "li", "betap", "Ip"]:
            print(f"    {key:10s} {span(key)}", flush=True)

    # ---- Phase B: add the m_s/gamma label on a subset ---------------------
    print(f"\n=== Phase B: {n_lin} samples WITH linearisation (m_s label) ===", flush=True)
    rng2 = np.random.default_rng(seed + 1000)
    labelled, tlin = [], []
    for k in range(n_lin):
        c = L.sample_control(rng2, anchors)
        t0 = time.time()
        try:
            rec = L.forward_label(tok, **c, with_linearisation=True)
            dt = time.time() - t0
            tlin.append(dt)
            labelled.append(rec)
            print(f"  [{k:2d}] kappa={rec['kappa']:.3f} delta={rec['delta']:+.3f} "
                  f"m_s={rec['m_s']:.3f} gamma={rec['gamma']:7.1f} li={rec['li']:.3f} "
                  f"betap={rec['betap']:.3f} ({dt:.0f}s)", flush=True)
        except Exception as e:
            print(f"  [{k:2d}] REJECT {str(e)[:60]}", flush=True)
    if labelled:
        ms = np.array([r["m_s"] for r in labelled])
        print(f"\n  m_s span: {ms.min():.3f}..{ms.max():.3f} (med {np.median(ms):.3f}); "
              f"marginal(<0.4): {(ms<0.4).sum()}, stable(>1.0): {(ms>1.0).sum()}", flush=True)
        print(f"  per-label cost: median {np.median(tlin):.0f}s, max {np.max(tlin):.0f}s", flush=True)
    # persist the pilot for inspection
    out = os.path.join(ROOT, "data", "phase15_pilot.pkl")
    with open(out, "wb") as f:
        pickle.dump(dict(fast=ok, labelled=labelled, fails=fails), f)
    print(f"\nSaved {out}", flush=True)


if __name__ == "__main__":
    main()
