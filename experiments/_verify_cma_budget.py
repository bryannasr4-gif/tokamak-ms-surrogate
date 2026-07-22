"""VERIFICATION SCRIPT (independent referee): re-run CMA on the 10 MARGINAL starts at a larger
budget to test the 'cma-starved-too-few-generations' finding. Same harness / per-start box+guard
expansion as phase3_worker.py. Writes data/_verify_cma_budget.json incrementally (resume-safe).
NOT part of the project pipeline."""
import json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiments"))
import phase15_lib as L
import phase2_model as M
import phase2_dim_lib as DL
import phase3_lib as P3

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUDGET = int(os.environ.get("VBUDGET", "60"))
OUT = os.path.join(ROOT, "data", f"_verify_cma_budget_{BUDGET}.json")

def main():
    S = json.load(open(os.path.join(ROOT, "data", "phase3_setup.json")))
    tok = L.load_machine()
    models, _ = M.load_ensemble()
    smap, _ = M.load_shapemap()
    mu = np.array(S["mu"]); std = np.array(S["std"]); V = np.array(S["V"])
    lo = np.array(S["box_lo"]); hi = np.array(S["box_hi"]); ranges = S["ranges"]
    recs, done = [], set()
    if os.path.exists(OUT):
        recs = json.load(open(OUT)).get("recs", [])
        done = {r["start_i"] for r in recs}
    marg = [(i, s) for i, s in enumerate(S["starts"]) if s["regime"] == "marginal"]
    for i, s in marg:
        if i in done:
            continue
        ds = DL.DesignSpace(mu, std, V, lo, hi, np.array(s["u"]), S["d"])
        m = 0.05 * (ds.box_hi - ds.box_lo)
        ds.box_lo = np.minimum(ds.box_lo, ds.x0 - m)
        ds.box_hi = np.maximum(ds.box_hi, ds.x0 + m)
        sranges = {}
        for f, (rlo, rhi) in ranges.items():
            v = s.get("desc", {}).get(f)
            sranges[f] = [rlo, rhi] if v is None else [min(rlo, v - 0.02*(rhi-rlo)), max(rhi, v + 0.02*(rhi-rlo))]
        t0 = time.time()
        r = P3.run_cma(tok, ds, BUDGET, sranges, s["m_s_start"], s["kappa_start"], s["regime"], seed=0)
        s2t = r["solves_to_target"].get("1.0")
        rec = dict(start_i=i, budget=BUDGET, n_solves=r["n_solves"], best_ms=r["best_ms"],
                   reached=r["reached_primary"], s2t_1p0=s2t)
        recs.append(rec)
        print(f"start{i:2d} budget={BUDGET} n={r['n_solves']:3d} best={r['best_ms']:.3f} "
              f"reach@={s2t} ({(time.time()-t0)/60:.1f}min)", flush=True)
        json.dump(dict(recs=recs), open(OUT + ".tmp", "w")); os.replace(OUT + ".tmp", OUT)
    print("DONE", flush=True)

if __name__ == "__main__":
    main()
