"""VERIFICATION (independent referee): CMA on the 10 MARGINAL starts at the SAME budget=30 but with a
SMALL popsize (fix (a) of the 'cma-starved' finding) so CMA gets ~6 generations instead of ~2.7 to
adapt its covariance. Same harness / per-start box+guard expansion as phase3_worker.py.
Writes data/_verify_cma_popsize.json. NOT part of the project pipeline."""
import json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiments"))
import cma
import phase15_lib as L
import phase2_model as M
import phase2_dim_lib as DL
import phase3_lib as P3

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUDGET = 30
POPSIZE = int(os.environ.get("VPOP", "5"))
OUT = os.path.join(ROOT, "data", f"_verify_cma_pop{POPSIZE}.json")


def run_cma_pop(tok, ds, budget, ranges, m_s_start, kappa_start, regime, seed, popsize):
    rc = P3.Recorder(m_s_start, kappa_start, regime)
    P3._eval_start(tok, ds, ranges, rc)
    es = cma.CMAEvolutionStrategy(list(ds.x0), P3.MAX_STEP, {
        "bounds": [list(ds.box_lo), list(ds.box_hi)], "seed": int(seed) % (2**31),
        "popsize": popsize, "verbose": -9, "maxfevals": budget - 1})
    while rc.n < budget and rc.best < P3.PRIMARY_TARGET and not es.stop():
        sols = es.ask()
        fit = []
        for s in sols:
            if rc.n >= budget:
                fit.append(1e3); continue
            xc = ds.clip(s)
            val, info = P3.true_eval(tok, ds.u_of_x(xc), ranges)
            accept = info["ok"] and val > rc.best
            rc.log(val, info, xc if accept else None, accept)
            fit.append(-val if info["ok"] else 1e3)
        es.tell(sols, fit)
    return rc.result()


def main():
    S = json.load(open(os.path.join(ROOT, "data", "phase3_setup.json")))
    tok = L.load_machine(); models, _ = M.load_ensemble(); smap, _ = M.load_shapemap()
    mu = np.array(S["mu"]); std = np.array(S["std"]); V = np.array(S["V"])
    lo = np.array(S["box_lo"]); hi = np.array(S["box_hi"]); ranges = S["ranges"]
    recs, done = [], set()
    if os.path.exists(OUT):
        recs = json.load(open(OUT)).get("recs", []); done = {r["start_i"] for r in recs}
    marg = [(i, s) for i, s in enumerate(S["starts"]) if s["regime"] == "marginal"]
    for i, s in marg:
        if i in done:
            continue
        ds = DL.DesignSpace(mu, std, V, lo, hi, np.array(s["u"]), S["d"])
        m = 0.05 * (ds.box_hi - ds.box_lo)
        ds.box_lo = np.minimum(ds.box_lo, ds.x0 - m); ds.box_hi = np.maximum(ds.box_hi, ds.x0 + m)
        sranges = {}
        for f, (rlo, rhi) in ranges.items():
            v = s.get("desc", {}).get(f)
            sranges[f] = [rlo, rhi] if v is None else [min(rlo, v-0.02*(rhi-rlo)), max(rhi, v+0.02*(rhi-rlo))]
        t0 = time.time()
        r = run_cma_pop(tok, ds, BUDGET, sranges, s["m_s_start"], s["kappa_start"], s["regime"], 0, POPSIZE)
        s2t = r["solves_to_target"].get("1.0")
        recs.append(dict(start_i=i, popsize=POPSIZE, n_solves=r["n_solves"], best_ms=r["best_ms"],
                         reached=r["reached_primary"], s2t_1p0=s2t))
        print(f"start{i:2d} pop={POPSIZE} n={r['n_solves']:3d} best={r['best_ms']:.3f} reach@={s2t} "
              f"({(time.time()-t0)/60:.1f}min)", flush=True)
        json.dump(dict(recs=recs), open(OUT + ".tmp", "w")); os.replace(OUT + ".tmp", OUT)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
