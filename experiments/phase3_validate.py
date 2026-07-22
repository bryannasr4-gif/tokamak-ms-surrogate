"""Quick end-to-end validation of all 5 Phase-3 methods on one marginal start, tiny budget."""
import os, sys, json, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiments"))
import numpy as np, pandas as pd
import phase15_lib as L, phase2_model as M, phase2_data as D, phase2_dim_lib as DL, phase3_lib as P3

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
tok = L.load_machine(); models,_ = M.load_ensemble(); smap,_ = M.load_shapemap()
setup = json.load(open(os.path.join(ROOT,"data","phase2_dim_setup.json")))
ranges = json.load(open(os.path.join(ROOT,"data","phase3_desc_ranges.json")))
df = pd.read_parquet(os.path.join(ROOT,"data","dataset_v1_80q.parquet"))
mu=np.array(setup['mu']); std=np.array(setup['std']); V=np.array(setup['V'])
lo=np.array(setup['box_lo']); hi=np.array(setup['box_hi'])

row = df[(df.m_s>=0.15)&(df.m_s<0.4)].sort_values('m_s').iloc[5]
u0 = np.array([float(row[c]) for c in D.CONTROL_FEATURES])
start = dict(m_s_start=float(row['m_s']), kappa_start=float(row['kappa']), regime="marginal")
print(f"start m_s={start['m_s_start']:.3f} kappa={start['kappa_start']:.3f}\n", flush=True)

for method in ["surrogate","heuristic","random","cma","nelder"]:
    ds = DL.DesignSpace(mu,std,V,lo,hi,u0.copy(),12)
    m = 0.05*(ds.box_hi-ds.box_lo); ds.box_lo=np.minimum(ds.box_lo,ds.x0-m); ds.box_hi=np.maximum(ds.box_hi,ds.x0+m)
    t0=time.time()
    r = P3.run_one(tok, models, smap, ds, 4, ranges, start, method, seed=123)
    assert set(["n_solves","best_ms","gain","solves_to_target","reject","traj","accepted"]).issubset(r), "missing keys"
    assert r["n_solves"]<=4, f"budget overrun {r['n_solves']}"
    assert len(r["traj"])==r["n_solves"], "traj length mismatch"
    print(f"{method:10s} nsolve={r['n_solves']} best={r['best_ms']:.3f} gain={r['gain']:+.3f} "
          f"reject={r['reject']} naccept={len(r['accepted'])} s2t={r['solves_to_target']} "
          f"({time.time()-t0:.0f}s)", flush=True)
print("\nVALIDATE OK", flush=True)
