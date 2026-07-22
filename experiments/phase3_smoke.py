"""phase3_smoke.py -- validate the full Phase-3 stack loads + time one 80-mode true solve."""
import os, sys, time, json
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiments"))
import numpy as np
import phase15_lib as L
import phase2_model as M
import phase2_data as D
import phase2_dim_lib as DL
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
t0 = time.time()
tok = L.load_machine()
models, _ = M.load_ensemble()
smap, _ = M.load_shapemap()
setup = json.load(open(os.path.join(ROOT, "data", "phase2_dim_setup.json")))
df = pd.read_parquet(os.path.join(ROOT, "data", "dataset_v1_80q.parquet"))
print(f"loaded machine+models+shapemap+setup+data in {time.time()-t0:.1f}s", flush=True)
print(f"data: {df.shape}, columns incl m_s={'m_s' in df.columns}, kappa={'kappa' in df.columns}, idx={'idx' in df.columns}", flush=True)
print(f"m_s range [{df['m_s'].min():.4f}, {df['m_s'].max():.4f}]; "
      f"marginal(<0.4)={int((df['m_s']<0.4).sum())}, mid[0.4,1)={int(((df['m_s']>=0.4)&(df['m_s']<1.0)).sum())}", flush=True)

# pick a marginal start, build d=12 design space, time one true solve
mu=np.array(setup['mu']); std=np.array(setup['std']); V=np.array(setup['V'])
lo=np.array(setup['box_lo']); hi=np.array(setup['box_hi'])
row = df[(df['m_s']>=0.15)&(df['m_s']<0.4)].sort_values('m_s').iloc[5]
u0 = np.array([float(row[c]) for c in D.CONTROL_FEATURES])
ds = DL.DesignSpace(mu, std, V, lo, hi, u0, 12)
print(f"\nstart row idx: m_s_label={row['m_s']:.4f}, kappa_label={row['kappa']:.4f}", flush=True)

t1 = time.time()
ms = DL.true_ms(tok, ds.u_of_x(ds.x0), fix_n_modes=80)
print(f"true_ms (80 modes) at start = {ms:.4f}  in {time.time()-t1:.1f}s  (label was {row['m_s']:.4f})", flush=True)

# verify forward_label returns full descriptors (for in-range guard + gallery)
c = DL.ctrl_from_u(ds.u_of_x(ds.x0))
t2 = time.time()
rec = L.forward_label(tok, c['active_currents'], c['paxis'], c['Ip'], c['fvac'], c['alpha_m'], c['alpha_n'], fix_n_modes=80)
print(f"forward_label keys: m_s={rec['m_s']:.4f} kappa={rec['kappa']:.4f} delta={rec['delta']:.4f} "
      f"gap_inner={rec['gap_inner']:.4f} gap_outer={rec['gap_outer']:.4f} li={rec['li']:.4f} in {time.time()-t2:.1f}s", flush=True)

# one surrogate gradient step direction sanity
g = DL._grad_x(models, smap, ds, ds.x0.copy())
print(f"\nsurrogate grad_x norm={np.linalg.norm(g):.4f} (nonzero={np.linalg.norm(g)>1e-9})", flush=True)
print("SMOKE OK", flush=True)
