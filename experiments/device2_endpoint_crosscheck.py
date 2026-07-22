"""
device2_endpoint_crosscheck.py -- rule #4 applied to the DESIGN ENDPOINTS. The win/loss verdict is
driven by each run's best_ms (an 80-mode forward_label solve at the best controls best_u). The
Device-C labeler was already cross-checked vs the independent Portone eigenvalue recompute on probe
shapes (0.000%); this re-confirms it on the actual design endpoints that carry the headline, spanning
regimes + the highest-gain runs.

  python experiments/device2_endpoint_crosscheck.py --framing retrained --n 6
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "experiments"))
import phase15_lib as L
import phase2_model as M
import phase2_dim_lib as DL
import phase0_lib as P0
import device2_killgate as KG
import device2_portone_crosscheck as PC


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--framing", default="retrained")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--rmax", type=float, default=2.8)
    args = ap.parse_args()

    recs = [json.load(open(f)) for f in
            glob.glob(os.path.join(ROOT, "data", "device2_design_results", f"{args.framing}_job*.json"))]
    # pick endpoints: top-gain runs + a spread across start m_s
    recs = [r for r in recs if r.get("best_u")]
    recs.sort(key=lambda r: -r["gain"])
    pick = recs[:args.n // 2]                                   # highest-gain endpoints
    rest = sorted(recs[args.n // 2:], key=lambda r: r["best_ms"])
    pick += [rest[int(i)] for i in np.linspace(0, len(rest) - 1, args.n - len(pick))] if rest else []

    # best_u is the d-dim PC-score x (GRecorder); map back to the 16-dim control via the DesignSpace.
    setup = json.load(open(os.path.join(ROOT, "data", "device2_design_setup.json")))
    starts = {s["id"]: s for s in setup["starts"]}
    mu = np.array(setup["mu"]); std = np.array(setup["std"]); V = np.array(setup["V"])
    lo = np.array(setup["box_lo"]); hi = np.array(setup["box_hi"]); d = setup["d"]

    import pickle
    import _blas_guard
    _blas_guard.assert_pinned()                    # refuse to run real solves unpinned (locked protocol)
    with open(os.path.join(ROOT, "data", "device2_anchors.pkl"), "rb") as f:
        L.GRID = pickle.load(f)["meta"]["grid"]    # Device-C grid: single source of truth (anchors meta)
    tok = KG.load_device_c()
    worst = 0.0
    out = []
    print(f"Endpoint Portone cross-check ({args.framing}, {len(pick)} endpoints, 80 modes):")
    for r in pick:
        s = starts[r["start_id"]]
        ds = DL.DesignSpace(mu, std, V, lo, hi, np.array(s["u0"], dtype=np.float64), d)
        u = ds.u_of_x(np.array(r["best_u"]))                   # PC-score x -> 16-dim control u
        c = DL.ctrl_from_u(u)
        eq, nls = PC.solve_with_nls(tok, c["active_currents"], c["paxis"], c["Ip"], c["fvac"],
                                    c["alpha_m"], c["alpha_n"], 80)
        if nls is None:
            print(f"  job{r['jid']} ({r['method']}): endpoint re-solve failed; skip"); continue
        reported = float(np.ravel(np.asarray(nls.linearised_sol.stability_margin).real).max())
        ind = P0.independent_stability_margins(nls)
        mA = float(ind["pos_A"][0]) if len(ind["pos_A"]) else float("nan")
        pct = 100 * abs(mA - reported) / abs(reported) if reported else float("nan")
        worst = max(worst, pct)
        print(f"  job{r['jid']:3d} {r['method']:12s} stored best_ms={r['best_ms']:.4f} "
              f"re-solve={reported:.4f} indep={mA:.4f} |Δ|={pct:.3f}% n_pos={len(ind['pos_A'])}")
        out.append(dict(jid=r["jid"], method=r["method"], stored_best_ms=r["best_ms"],
                        resolve_ms=reported, independent_ms=mA, pct_diff=pct))
    gate = worst < 5.0
    json.dump(dict(framing=args.framing, n=len(out), worst_pct=worst,
                   gate_under_5pct=bool(gate), endpoints=out),
              open(os.path.join(ROOT, "data", f"device2_endpoint_crosscheck_{args.framing}.json"), "w"), indent=2)
    print(f"\n==> worst |Δ| (independent vs FreeGSNKE) = {worst:.4f}%  GATE(<5%): {'PASS' if gate else 'FAIL'}")


if __name__ == "__main__":
    main()
