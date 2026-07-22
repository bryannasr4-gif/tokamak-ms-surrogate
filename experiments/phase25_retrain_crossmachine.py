"""
phase25_retrain_crossmachine.py -- after the combined re-label completes:
  1. Retrain the canonical shape surrogate m_s(shape) + the (machine-independent) ShapeMap on the
     CLEAN 80-mode Machine-A labels (data/dataset_v1_80.parquet) -> overwrites data/phase2_models/
     {surrogate,shapemap}.pt so all post-hardening experiments use converged labels.
  2. Train a Machine-B m_s surrogate (data/dataset_v2_B.parquet; same shapes, different conducting
     structure) -> surrogate_B.
  3. CROSS-MACHINE generalization: how well does the Machine-A surrogate transfer to Machine B
     (same shapes, m_s shifted ~-40%)? Report naive RMSE, Spearman (does the shape->m_s ORDERING
     transfer?), and post-affine-rescale R² (is it just a global scale, or different physics?).
Saves data/phase25_crossmachine.json. No solves.
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

import numpy as np
from scipy.stats import spearmanr


def r2(y, yh):
    return float(1 - np.sum((y - yh) ** 2) / np.sum((y - y.mean()) ** 2))


def regime_split(df):
    df = df.copy()
    df["regime"] = df["m_s"].apply(D.regime_of)
    return df


def train_on(parquet, name):
    df = regime_split(pd.read_parquet(parquet))
    tr = df[df.split == "train"]
    feat = D.SHAPE_FEATURES
    X = tr[feat].values.astype(np.float64)
    Ttr = np.column_stack([np.log(tr["m_s"].values), np.log(tr["gamma"].values)])
    models, meta = T.train_surrogate(X, Ttr, din=len(feat))
    M.save_ensemble(models, meta, name=name)
    return df, models, feat


def main():
    out = {}
    # 1. canonical clean-label retrain (Machine A, 80 modes) -> default surrogate + shapemap
    print("retraining canonical surrogate on clean 80-mode Machine-A labels...")
    dfA, modelsA, feat = train_on("data/dataset_v1_80.parquet", "surrogate")
    tr = dfA[dfA.split == "train"]
    smap, smeta = T.train_shapemap(tr[D.CONTROL_FEATURES].values.astype(np.float64),
                                   tr[feat].values.astype(np.float64),
                                   din=len(D.CONTROL_FEATURES), dout=len(feat))
    M.save_shapemap(smap, smeta)
    for split in ("val", "test_extrap"):
        dd = dfA[dfA.split == split]
        pr = M.ensemble_predict(modelsA, dd[feat].values.astype(np.float64))["mean"][:, 0]
        out[f"A80_{split}_logR2"] = r2(np.log(dd["m_s"].values), pr)
    print(f"  A80 surrogate val log-R2={out['A80_val_logR2']:.3f} test={out['A80_test_extrap_logR2']:.3f}")

    # 2. Machine-B surrogate
    print("training Machine-B surrogate...")
    dfB, modelsB, _ = train_on("data/dataset_v2_B.parquet", "surrogate_B")
    for split in ("val", "test_extrap"):
        dd = dfB[dfB.split == split]
        pr = M.ensemble_predict(modelsB, dd[feat].values.astype(np.float64))["mean"][:, 0]
        out[f"B_{split}_logR2"] = r2(np.log(dd["m_s"].values), pr)
    print(f"  B surrogate val log-R2={out['B_val_logR2']:.3f} test={out['B_test_extrap_logR2']:.3f}")

    # 3. cross-machine transfer: A-surrogate on B's shapes/labels (matched by idx)
    mrg = dfA[["idx", "m_s"] + feat].rename(columns={"m_s": "m_s_A"}).merge(
        dfB[["idx", "m_s"]].rename(columns={"m_s": "m_s_B"}), on="idx")
    Xb = mrg[feat].values.astype(np.float64)
    predA_on_B = M.ensemble_predict(modelsA, Xb)["mean"][:, 0]      # A-surrogate's log m_s on B shapes
    logB = np.log(mrg["m_s_B"].values); logA = np.log(mrg["m_s_A"].values)
    spl = mrg["split"].values
    tr = spl == "train"; va = spl == "val"; te = spl == "test_extrap"
    out["A_to_B_naive_logR2"] = r2(logB, predA_on_B)
    out["A_to_B_spearman"] = float(spearmanr(predA_on_B, logB).statistic)
    # affine rescale: FIT on train, EVALUATE held-out (the earlier in-sample fit was circular).
    a, b = np.polyfit(predA_on_B[tr], logB[tr], 1)
    out["A_to_B_affine_coef"] = [float(a), float(b)]
    out["A_to_B_affine_logR2_val"] = r2(logB[va], a * predA_on_B[va] + b)
    out["A_to_B_affine_logR2_test"] = r2(logB[te], a * predA_on_B[te] + b)
    # TRIVIAL true-label baselines (no surrogate): is the surrogate adding anything to transfer?
    ca, cb = np.polyfit(logA[tr], logB[tr], 1)
    out["truelabel_affine_logR2_val"] = r2(logB[va], ca * logA[va] + cb)
    out["truelabel_affine_logR2_test"] = r2(logB[te], ca * logA[te] + cb)
    out["truelabel_spearman"] = float(spearmanr(logA, logB).statistic)
    shift = float(np.mean(logB[tr] - logA[tr]))
    out["truelabel_shift_only_logR2_val"] = r2(logB[va], logA[va] + shift)
    # marginal-band degradation
    mb = mrg["m_s_B"].values < 0.5
    out["marginal_band_spearman_truelabel"] = float(spearmanr(logA[mb], logB[mb]).statistic)
    out["mean_logms_shift_A_to_B"] = float(np.mean(logB - logA))
    out["n_matched"] = int(len(mrg))

    json.dump(out, open("data/phase25_crossmachine.json", "w"), indent=2)
    print("\n=== CROSS-MACHINE (A-surrogate -> Machine B) ===")
    print(f"  naive transfer log-R2 = {out['A_to_B_naive_logR2']:.3f} (expect LOW: m_s shifted)")
    print(f"  Spearman(A-pred, B-true) = {out['A_to_B_spearman']:.3f} (HIGH => shape ordering transfers)")
    print(f"  after global affine rescale: log-R2 = {out['A_to_B_affine_logR2']:.3f} "
          f"(HIGH => mostly a scale change, not different physics)")
    print(f"  mean log m_s shift A->B = {out['mean_logms_shift_A_to_B']:+.3f}")
    print("Saved data/phase25_crossmachine.json")


if __name__ == "__main__":
    main()
