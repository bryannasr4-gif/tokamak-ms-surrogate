"""
phase4_ablations_light.py -- the no-retrain / no-solve ablations (Phase-4 rigor item 3, part 1):
  (A) ENSEMBLE SIZE: evaluate held-out accuracy + gradient-direction stability using the first k of
      the 8 trained members (k=1,2,4,8). No retraining -- a valid ensemble-size sweep by subsetting.
  (B) INPUT-NOISE ROBUSTNESS (rigor item 4): perturb the held-out SHAPE inputs with relative Gaussian
      noise (sigma = 1/3/5 % of per-feature std) and measure (i) prediction RMSE_log degradation and
      (ii) gradient-direction cosine vs the unperturbed gradient -- both the accuracy and the GRADIENT
      sensitivity, as the prompt requires.
  (C) MODE-COUNT sensitivity: summarized from the existing 60-shape mode study (data/phase2_modes_*).
Saves data/phase4_ablations_light.json.
"""
import json
import os
import sys

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


def r2_rmse_log(models_subset, X, true_logms):
    Xt = torch.as_tensor(X, dtype=torch.float32)
    with torch.no_grad():
        pl = torch.stack([m(Xt)[:, 0] for m in models_subset]).mean(0).numpy()
    rmse = float(np.sqrt(np.mean((pl - true_logms) ** 2)))
    r2 = 1.0 - np.sum((true_logms - pl) ** 2) / np.sum((true_logms - true_logms.mean()) ** 2)
    return rmse, float(r2), pl


def main():
    df = pd.read_parquet("data/dataset_v1_80q.parquet")
    models, meta = M.load_ensemble()
    smap, _ = M.load_shapemap()
    held = df[df["split"] != "train"]
    X = held[D.SHAPE_FEATURES].values.astype(np.float32)
    true_logms = np.log(held["m_s"].values)
    marg = held["m_s"].values < 0.4

    out = {}

    # (A) ENSEMBLE SIZE
    ens = []
    for k in [1, 2, 4, 8]:
        sub = models[:k]
        rmse, r2, pl = r2_rmse_log(sub, X, true_logms)
        rmse_m = float(np.sqrt(np.mean((pl[marg] - true_logms[marg]) ** 2)))
        ens.append(dict(k=k, rmse_log=rmse, r2_log=r2, rmse_log_marginal=rmse_m))
    out["ensemble_size"] = ens

    # gradient-direction stability vs ensemble size (composed control gradient at several starts)
    setup = json.load(open("data/phase2_dim_setup.json"))
    mu = np.array(setup["mu"]); std = np.array(setup["std"]); V = np.array(setup["V"])
    lo = np.array(setup["box_lo"]); hi = np.array(setup["box_hi"])
    starts = held[D.CONTROL_FEATURES].values[:30].astype(float)
    cos_vs_full = {1: [], 2: [], 4: []}
    for u0 in starts:
        ds = DL.DesignSpace(mu, std, V, lo, hi, u0, 12)
        g8 = DL._grad_x(models, smap, ds, ds.x0); g8 /= np.linalg.norm(g8) + 1e-12
        for k in [1, 2, 4]:
            gk = DL._grad_x(models[:k], smap, ds, ds.x0); gk /= np.linalg.norm(gk) + 1e-12
            cos_vs_full[k].append(float(np.dot(gk, g8)))
    out["grad_dir_cos_vs_full8"] = {str(k): dict(median=float(np.median(v)),
                                                 p10=float(np.percentile(v, 10))) for k, v in cos_vs_full.items()}

    # (B) INPUT-NOISE ROBUSTNESS (accuracy + gradient direction)
    rng = np.random.default_rng(20260624)
    feat_std = held[D.SHAPE_FEATURES].values.std(0) + 1e-9
    base_rmse, _, _ = r2_rmse_log(models, X, true_logms)
    # baseline shape-space gradients on the 30 starts
    g0s = []
    Xt = torch.as_tensor(X[:30], dtype=torch.float32)
    g0_all = M.ms_grad_shape(models, X[:30])           # (30, 20) d log m_s / d shape
    noise_rows = []
    for sig in [0.01, 0.03, 0.05]:
        rmses, cosg = [], []
        for rep in range(5):
            Xn = X + rng.normal(0, sig, X.shape).astype(np.float32) * feat_std
            rmse, _, _ = r2_rmse_log(models, Xn, true_logms)
            rmses.append(rmse)
        # gradient direction stability under input noise
        for rep in range(5):
            Xn30 = X[:30] + rng.normal(0, sig, X[:30].shape).astype(np.float32) * feat_std
            gn = M.ms_grad_shape(models, Xn30)
            for i in range(30):
                a, b = g0_all[i], gn[i]
                cosg.append(float(np.dot(a, b) / ((np.linalg.norm(a) + 1e-12) * (np.linalg.norm(b) + 1e-12))))
        noise_rows.append(dict(sigma_frac=sig, rmse_log_mean=float(np.mean(rmses)),
                               rmse_log_delta=float(np.mean(rmses) - base_rmse),
                               grad_cos_median=float(np.median(cosg)),
                               grad_cos_p10=float(np.percentile(cosg, 10))))
    out["input_noise"] = dict(baseline_rmse_log=base_rmse, rows=noise_rows)

    # (C) MODE-COUNT sensitivity (from the existing study)
    ms = json.load(open("data/phase2_modes_summary.json"))
    out["mode_count"] = dict(
        converged_modes=80,
        drift_40_to_138_by_regime={k: ms["by_regime"][k]["drift_40_to_138"]["median"]
                                   for k in ms["by_regime"]},
        resid_80_vs_138_median={k: ms["by_regime"][k]["resid_80_vs_138_abs"]["median"]
                                for k in ms["by_regime"]})

    json.dump(out, open("data/phase4_ablations_light.json", "w"), indent=2)

    print("=== (A) ENSEMBLE SIZE (held-out) ===")
    for e in ens:
        print(f"  k={e['k']}: RMSE_log {e['rmse_log']:.3f}  log-R2 {e['r2_log']:.3f}  "
              f"marginal RMSE_log {e['rmse_log_marginal']:.3f}")
    print("  gradient-direction cosine vs full-8 ensemble (median / p10):")
    for k, v in out["grad_dir_cos_vs_full8"].items():
        print(f"    k={k}: {v['median']:.3f} / {v['p10']:.3f}")
    print("\n=== (B) INPUT-NOISE ROBUSTNESS ===")
    print(f"  baseline RMSE_log {base_rmse:.3f}")
    for r in noise_rows:
        print(f"  sigma={r['sigma_frac']*100:.0f}% of feat-std: RMSE_log {r['rmse_log_mean']:.3f} "
              f"(+{r['rmse_log_delta']:.3f})  grad-dir cos median {r['grad_cos_median']:.3f} "
              f"p10 {r['grad_cos_p10']:.3f}")
    print("\n=== (C) MODE-COUNT (existing study) ===")
    print(f"  converged = 80 modes; 40->138 drift by regime: {out['mode_count']['drift_40_to_138_by_regime']}")
    print("Saved data/phase4_ablations_light.json")


if __name__ == "__main__":
    main()
