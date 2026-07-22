"""
phase4_ablations_train.py -- the RETRAINING ablations (Phase-4 rigor item 3, part 2). NO solver.
Run AFTER the gallery completes (it retrains several small ensembles on CPU; avoid contending with
the solve pool). All on the canonical 80-mode labels (dataset_v1_80q), split honored.

  (A) DATASET-SIZE LEARNING CURVE: train on random {12.5,25,50,100}% of the train split; held-out
      RMSE_log + marginal RMSE_log. Shows how much data the accuracy needs (the offline label cost).
  (B) SHAPE PARAMETERIZATION: train on nested feature sets
      {kappa} -> {kappa,delta} -> {+gaps} -> {+squareness} -> full(20); held-out RMSE_log by regime.
      Shows what each descriptor group adds OVER kappa alone (the dominant lever).

Training config = the CANONICAL surrogate config (n_models=8, epochs=2600, warmup=800) so the
ablation numbers are directly comparable to the headline model -- no speed compromise. (An earlier
pass used a reduced 4x1500 config for speed; this is the full-quality re-run.)
Saves data/phase4_ablations_train.json.
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
import phase2_train as T

N_MODELS = 8        # canonical surrogate config (no speed compromise)
EPOCHS = 2600
WARMUP = 800
SEED = 20260624


def rmse_log(models, X, true_logms):
    Xt = torch.as_tensor(X, dtype=torch.float32)
    with torch.no_grad():
        pl = torch.stack([m(Xt)[:, 0] for m in models]).mean(0).numpy()
    return float(np.sqrt(np.mean((pl - true_logms) ** 2)))


def main():
    df = pd.read_parquet("data/dataset_v1_80q.parquet")
    df["regime"] = df["m_s"].apply(D.regime_of)
    tr = df[df.split == "train"].reset_index(drop=True)
    held = df[df.split != "train"]
    feat_full = D.SHAPE_FEATURES
    Xh_full = held[feat_full].values.astype(np.float64)
    true_logms = np.log(held["m_s"].values)
    marg = held["m_s"].values < 0.4
    rng = np.random.default_rng(SEED)
    out = {"config": dict(n_models=N_MODELS, epochs=EPOCHS, warmup=WARMUP)}

    # (A) learning curve
    Ttr_full = np.column_stack([np.log(tr["m_s"].values), np.log(tr["gamma"].values)])
    lc = []
    for frac in [0.125, 0.25, 0.5, 1.0]:
        n = int(round(frac * len(tr)))
        idx = rng.choice(len(tr), n, replace=False)
        models, _ = T.train_surrogate(tr.iloc[idx][feat_full].values.astype(np.float64),
                                      Ttr_full[idx], din=len(feat_full),
                                      n_models=N_MODELS, epochs=EPOCHS, warmup=WARMUP, seed0=SEED)
        r = rmse_log(models, Xh_full, true_logms)
        rm = rmse_log(models, Xh_full[marg], true_logms[marg])
        lc.append(dict(frac=frac, n=n, rmse_log=r, rmse_log_marginal=rm))
        print(f"[LC] frac={frac:.3f} n={n}: RMSE_log {r:.3f} marginal {rm:.3f}", flush=True)
    out["learning_curve"] = lc

    # (B) shape parameterization (nested feature sets)
    feat_sets = {
        "kappa": ["kappa"],
        "kappa_delta": ["kappa", "delta"],
        "plus_gaps": ["kappa", "delta", "gap_inner", "gap_outer", "gap_min"],
        "plus_squareness": ["kappa", "delta", "gap_inner", "gap_outer", "gap_min",
                            "sq_uo", "sq_ui", "sq_lo", "sq_li"],
        "plus_li_betap": ["kappa", "delta", "gap_inner", "gap_outer", "gap_min",
                          "sq_uo", "sq_ui", "sq_lo", "sq_li", "li", "betap"],
        "full20": feat_full,
    }
    sp = []
    for name, feats in feat_sets.items():
        Xtr = tr[feats].values.astype(np.float64)
        models, _ = T.train_surrogate(Xtr, Ttr_full, din=len(feats),
                                      n_models=N_MODELS, epochs=EPOCHS, warmup=WARMUP, seed0=SEED)
        Xh = held[feats].values.astype(np.float64)
        r = rmse_log(models, Xh, true_logms)
        rm = rmse_log(models, Xh[marg], true_logms[marg])
        # by regime
        byreg = {}
        for rn, a, b in D.REGIMES:
            mm = (held["m_s"].values >= a) & (held["m_s"].values < b)
            if mm.sum() >= 8:
                byreg[rn] = rmse_log(models, Xh[mm], true_logms[mm])
        sp.append(dict(name=name, n_feats=len(feats), rmse_log=r, rmse_log_marginal=rm, by_regime=byreg))
        print(f"[SP] {name:16s} ({len(feats)} feats): RMSE_log {r:.3f} marginal {rm:.3f}", flush=True)
    out["shape_parameterization"] = sp

    json.dump(out, open("data/phase4_ablations_train.json", "w"), indent=2)
    print("\nSaved data/phase4_ablations_train.json")


if __name__ == "__main__":
    main()
