"""
phase4_pareto.py -- the cost/accuracy Pareto (Phase-4 rigor item 1). Surrogate inference + autodiff
gradient vs the full FreeGSNKE 80-mode linearised Jacobian (the expensive ~tens-of-perturbed-solves
stability calculation). Wall-clock + accuracy, at the locked protocol (OMP=1, serialized machine).

Reports:
  * t_true   : one full forward + 80-mode linearisation (the m_s label / Jacobian) -- median over N
  * t_fwd    : forward GS solve only (no linearisation) -- the cheaper partial-physics reference
  * t_surr   : surrogate ensemble m_s prediction (the amortized inference)
  * t_grad   : composed autodiff gradient d(log m_s)/d(controls) via Surrogate(ShapeMap(u))
  * speedups : t_true / t_surr and t_true / (t_surr + t_grad)
  * accuracy : held-out RMSE_log + R^2 (the accuracy delivered at that inference cost)
  * break-even: offline training cost = 3254 true solves (a shared project asset); per-query saving.
Saves data/phase4_pareto.json.
"""
import json
import os
import sys
import time

import numpy as np

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.getcwd(), "experiments"))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import pandas as pd
import torch
import phase2_data as D
import phase2_model as M
import phase2_dim_lib as DL
import phase15_lib as L

N_TRUE = 8           # timed full solves (expensive; keep modest)
N_INFER = 2000       # surrogate inference repeats for a stable timing


def main():
    df = pd.read_parquet("data/dataset_v1_80q.parquet")
    models, meta = M.load_ensemble()
    smap, _ = M.load_shapemap()
    tok = L.load_machine()

    # --- accuracy (held-out) ---
    held = df[df["split"] != "train"]
    X = held[D.SHAPE_FEATURES].values.astype(np.float32)
    pred = M.ensemble_predict(models, X)
    pred_logms = pred["mean"][:, 0]
    true_logms = np.log(held["m_s"].values)
    rmse_log = float(np.sqrt(np.mean((pred_logms - true_logms) ** 2)))
    ss_res = float(np.sum((true_logms - pred_logms) ** 2))
    ss_tot = float(np.sum((true_logms - true_logms.mean()) ** 2))
    r2_log = 1.0 - ss_res / ss_tot

    # --- timing: surrogate inference (single query) ---
    x1 = torch.tensor(X[:1], dtype=torch.float32)
    with torch.no_grad():                       # warmup
        for _ in range(50):
            _ = torch.stack([m(x1)[:, 0] for m in models]).mean()
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(N_INFER):
            _ = torch.stack([m(x1)[:, 0] for m in models]).mean()
    t_surr = (time.perf_counter() - t0) / N_INFER

    # --- timing: composed autodiff gradient d(log m_s)/d(controls) via ShapeMap ---
    setup = json.load(open("data/phase2_dim_setup.json"))
    mu = np.array(setup["mu"]); std = np.array(setup["std"]); V = np.array(setup["V"])
    lo = np.array(setup["box_lo"]); hi = np.array(setup["box_hi"])
    u0 = held[D.CONTROL_FEATURES].values[0].astype(float)
    ds = DL.DesignSpace(mu, std, V, lo, hi, u0, 12)
    for _ in range(20):                          # warmup
        _ = DL._grad_x(models, smap, ds, ds.x0)
    t0 = time.perf_counter()
    NG = 200
    for _ in range(NG):
        _ = DL._grad_x(models, smap, ds, ds.x0)
    t_grad = (time.perf_counter() - t0) / NG

    # --- timing: full true solve (forward + 80-mode linearisation) and forward-only ---
    rows = held[(held["m_s"] > 0.3) & (held["m_s"] < 2.0)].head(N_TRUE)
    t_true, t_fwd = [], []
    for _, r in rows.iterrows():
        u = np.array([float(r[c]) for c in D.CONTROL_FEATURES])
        c = DL.ctrl_from_u(u)
        try:
            t0 = time.perf_counter()
            L.forward_label(tok, c["active_currents"], c["paxis"], c["Ip"], c["fvac"],
                            c["alpha_m"], c["alpha_n"], fix_n_modes=80, with_linearisation=True)
            t_true.append(time.perf_counter() - t0)
            t0 = time.perf_counter()
            L.forward_label(tok, c["active_currents"], c["paxis"], c["Ip"], c["fvac"],
                            c["alpha_m"], c["alpha_n"], fix_n_modes=80, with_linearisation=False)
            t_fwd.append(time.perf_counter() - t0)
        except Exception:
            continue

    t_true_med = float(np.median(t_true))
    t_fwd_med = float(np.median(t_fwd))
    out = dict(
        n_heldout=int(len(held)), rmse_log=rmse_log, r2_log=r2_log,
        t_true_solve_s=t_true_med, t_forward_only_s=t_fwd_med,
        t_surr_infer_s=t_surr, t_grad_s=t_grad,
        speedup_infer=t_true_med / t_surr,
        speedup_infer_plus_grad=t_true_med / (t_surr + t_grad),
        training_cost_true_solves=3254,
        n_true=len(t_true),
    )
    json.dump(out, open("data/phase4_pareto.json", "w"), indent=2)

    print("=== Cost / accuracy Pareto (locked protocol, OMP=1, 80 modes) ===")
    print(f"held-out accuracy: RMSE_log {rmse_log:.3f}  log-R^2 {r2_log:.3f}  (n={len(held)})\n")
    print(f"t_true  (forward + 80-mode Jacobian/linearisation): {t_true_med*1e3:8.1f} ms  (n={len(t_true)})")
    print(f"t_fwd   (forward GS solve only, no linearisation)  : {t_fwd_med*1e3:8.1f} ms")
    print(f"t_surr  (ensemble m_s inference, single query)     : {t_surr*1e6:8.1f} us")
    print(f"t_grad  (composed autodiff d log m_s / d controls) : {t_grad*1e6:8.1f} us")
    print(f"\nspeedup, inference only       : {out['speedup_infer']:,.0f}x")
    print(f"speedup, inference + gradient : {out['speedup_infer_plus_grad']:,.0f}x")
    print(f"\nlinearisation overhead = t_true - t_fwd = {(t_true_med - t_fwd_med)*1e3:.0f} ms "
          f"({100*(t_true_med-t_fwd_med)/t_true_med:.0f}% of the full solve)")
    print("Saved data/phase4_pareto.json")


if __name__ == "__main__":
    main()
