"""
phase2_extrap_kappa.py -- a TRUE univariate extrapolation split (Phase-2.5b, Task B-ii).

The existing `test_extrap` is a JOINT high-kappa AND high-delta CORNER -- ~89% of it sits inside
the training convex hull (interpolation-with-a-gap), and the canonical surrogate was trained with
those points present along each axis. That is NOT real extrapolation.

Here we define a genuinely held-out UNIVARIATE high-kappa tail: hold out the top ~7% by kappa
(kappa >= the 93rd percentile) and keep the ENTIRE train/val pool strictly BELOW that threshold.
The surrogate never sees a single high-kappa shape -> it must EXTRAPOLATE in kappa, the dominant
m_s lever (corr(kappa,m_s)=-0.73). We retrain (same architecture/hyperparameters as the canonical
80-mode surrogate, via phase2_train.train_surrogate -- NO solves) and report accuracy on the tail,
resolved by regime, vs the in-distribution val of the SAME model (the clean extrapolation gap) and
vs the corner split. Expectation (honest): tail accuracy is WORSE than the corner.

Adds a `split_kappa` column {train, val, test_extrap_kappa} to data/dataset_v1_80q.parquet (keeping
the original `split` corner column intact). Saves model surrogate_kappa_extrap + metrics json.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, "experiments")
import phase2_data as D
import phase2_model as M
import phase2_train as T

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

KAPPA_Q = 0.93           # top ~7% by kappa = the held-out univariate extrapolation tail
VAL_FRAC = 0.15
DS80 = "data/dataset_v1_80.parquet"
DS80Q = "data/dataset_v1_80q.parquet"


def resolved_log(df_split, logms_true, logms_pred):
    """log-units R2/RMSE resolved by m_s regime."""
    out = dict(all=dict(n=int(len(logms_true)), r2=T.r2(logms_true, logms_pred),
                        rmse=T.rmse(logms_true, logms_pred)))
    reg = df_split["regime"].values
    for name, _, _ in D.REGIMES:
        m = reg == name
        if m.sum() >= 5:
            out[name] = dict(n=int(m.sum()), r2=T.r2(logms_true[m], logms_pred[m]),
                             rmse=T.rmse(logms_true[m], logms_pred[m]))
    return out


def main():
    df = pd.read_parquet(DS80)
    df["regime"] = df["m_s"].apply(D.regime_of)
    feat = D.SHAPE_FEATURES

    thresh = float(df["kappa"].quantile(KAPPA_Q))
    tail = df["kappa"] >= thresh
    pool = df[~tail].copy()
    rng = np.random.default_rng(0)
    val_mask = rng.random(len(pool)) < VAL_FRAC
    split_kappa = pd.Series("train", index=df.index)
    split_kappa[tail] = "test_extrap_kappa"
    pool_idx = pool.index.values
    split_kappa.loc[pool_idx[val_mask]] = "val"
    df["split_kappa"] = split_kappa.values

    tr = df[df.split_kappa == "train"]
    va = df[df.split_kappa == "val"]
    ta = df[df.split_kappa == "test_extrap_kappa"]
    print(f"kappa threshold (q{int(KAPPA_Q*100)}) = {thresh:.4f}")
    print(f"train {len(tr)}  val {len(va)}  test_extrap_kappa {len(ta)} "
          f"(tail kappa {ta['kappa'].min():.3f}-{ta['kappa'].max():.3f}; "
          f"train kappa max {tr['kappa'].max():.3f})")
    print("tail regime composition:", dict(ta["regime"].value_counts()))
    print(f"tail m_s range [{ta['m_s'].min():.3f}, {ta['m_s'].max():.3f}] median {ta['m_s'].median():.3f}")

    # ---- retrain (held-out tail) ----
    X = tr[feat].values.astype(np.float64)
    Ttr = np.column_stack([np.log(tr["m_s"].values), np.log(tr["gamma"].values)])
    print("training kappa-extrapolation surrogate (tail strictly held out)...")
    models, meta = T.train_surrogate(X, Ttr, din=len(feat))
    M.save_ensemble(models, meta, name="surrogate_kappa_extrap")

    out = dict(kappa_threshold=thresh, kappa_q=KAPPA_Q,
               n=dict(train=len(tr), val=len(va), test_extrap_kappa=len(ta)),
               train_kappa_max=float(tr["kappa"].max()),
               tail_kappa_range=[float(ta["kappa"].min()), float(ta["kappa"].max())],
               tail_regime={str(k): int(v) for k, v in ta["regime"].value_counts().items()},
               tail_ms_median=float(ta["m_s"].median()))
    for nm, dd in [("val", va), ("test_extrap_kappa", ta)]:
        pr = M.ensemble_predict(models, dd[feat].values.astype(np.float64))
        logms_pred = pr["mean"][:, 0]
        logms_true = np.log(dd["m_s"].values)
        ms_true, ms_pred = dd["m_s"].values, np.exp(logms_pred)
        out[nm] = dict(log=resolved_log(dd, logms_true, logms_pred),
                       orig=T.resolved(dd, ms_true, ms_pred),
                       rms_z=float(np.sqrt(np.mean(((logms_true - logms_pred) / pr["tot_std"][:, 0]) ** 2))))

    # corner-split (A80) comparison from the retrain artifact
    try:
        cm = json.load(open("data/phase25_crossmachine.json"))
        out["corner_A80_val_logR2"] = cm.get("A80_val_logR2")
        out["corner_A80_test_extrap_logR2"] = cm.get("A80_test_extrap_logR2")
    except Exception:
        pass

    json.dump(out, open("data/phase2_extrap_kappa.json", "w"), indent=2,
              default=lambda o: o.item() if hasattr(o, "item") else str(o))

    # add split_kappa to the 80q parquet (or 80 if 80q not yet built)
    target = DS80Q if os.path.exists(DS80Q) else DS80
    base = pd.read_parquet(target)
    base = base.merge(df[["idx", "split_kappa"]], on="idx", how="left")
    base.to_parquet(target)
    print(f"added split_kappa column to {target}")

    # ---- print ----
    print("\n=== TRUE UNIVARIATE HIGH-KAPPA EXTRAPOLATION (held-out tail) ===")
    for nm in ("val", "test_extrap_kappa"):
        a = out[nm]["log"]["all"]
        print(f"  {nm:20s} log-R2={a['r2']:+.3f} RMSE_log={a['rmse']:.3f} "
              f"orig-R2={out[nm]['orig']['all']['r2']:+.3f} RMS_z={out[nm]['rms_z']:.2f}")
        for name, _, _ in D.REGIMES:
            if name in out[nm]["log"]:
                r = out[nm]["log"][name]
                print(f"      {name:11s} n={r['n']:3d} log-R2={r['r2']:+.3f} RMSE_log={r['rmse']:.3f}")
    print(f"\n  extrapolation gap (same model): in-dist val log-R2={out['val']['log']['all']['r2']:+.3f} "
          f"-> high-kappa tail log-R2={out['test_extrap_kappa']['log']['all']['r2']:+.3f}")
    if "corner_A80_test_extrap_logR2" in out and out["corner_A80_test_extrap_logR2"] is not None:
        print(f"  vs the CORNER split (canonical A80): corner test log-R2={out['corner_A80_test_extrap_logR2']:.3f} "
              f"(easier: ~89% inside train hull)")
    print("\nSaved data/phase2_extrap_kappa.json + surrogate_kappa_extrap.pt")


if __name__ == "__main__":
    main()
