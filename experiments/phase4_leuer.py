"""
phase4_leuer.py -- the rigid Leuer-parameter PHYSICS baseline (Phase-4 rigor item 2). NO new solves.

The Leuer ratio (rigid-displacement stabilizing/destabilizing inductance ratio) is a cheap physics
proxy for vertical stability that needs NO full FD Jacobian. dataset_v1_80q already stores it
(`leuer`). We quantify it as a predictor of the converged 80-mode m_s and show what the LEARNED,
non-rigid m_s(shape) surrogate adds OVER this rank ceiling. Per RESULTS.md Phase-1.5 advice: Leuer
is a strong RANK predictor (Spearman) but Pearson is wrecked by a few outliers, so we report
Spearman + the marginal-vs-stable AUC, with outliers clipped, resolved by regime.

Reports, on the held-out split (val + test_extrap, never used to fit the surrogate):
  * Spearman(leuer, m_s) raw and after clipping leuer to [p1,p99]
  * AUC for the marginal (m_s<0.4) vs controllable (m_s>=0.4) decision: leuer vs surrogate
  * Spearman(prediction, m_s) for leuer vs surrogate (the rank ceiling the learned m_s lifts)
Saves data/phase4_leuer.json.
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
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
import phase2_data as D
import phase2_model as M


def auc_safe(y, score):
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, score))


def main():
    df = pd.read_parquet("data/dataset_v1_80q.parquet")
    # held-out = everything not in train (val + test_extrap), so the surrogate never saw it
    held = df[df["split"] != "train"].copy()
    ms = held["m_s"].values
    leuer = held["leuer"].values

    # surrogate predictions (mean ensemble log m_s -> m_s) on the same rows
    models, meta = M.load_ensemble()
    X = held[D.SHAPE_FEATURES].values.astype(np.float32)
    pred = M.ensemble_predict(models, X)
    pred_logms = pred["mean"][:, 0]
    pred_ms = np.exp(pred_logms)

    # clip leuer outliers to [p1,p99] (RESULTS advice: a few outliers wreck Pearson; Spearman robust)
    lo, hi = np.percentile(leuer, [1, 99])
    leuer_clip = np.clip(leuer, lo, hi)

    out = dict(n_heldout=int(len(held)))
    # rank correlation (higher leuer = more stable => positive Spearman expected)
    out["spearman_leuer_raw"] = float(spearmanr(leuer, ms).statistic)
    out["spearman_leuer_clipped"] = float(spearmanr(leuer_clip, ms).statistic)
    out["pearson_leuer_raw"] = float(np.corrcoef(leuer, ms)[0, 1])
    out["spearman_surrogate"] = float(spearmanr(pred_ms, ms).statistic)

    # marginal detection: y=1 if marginal (m_s<0.4). leuer LOW => marginal, so score = -leuer.
    y_marg = (ms < 0.4).astype(int)
    out["n_marginal_heldout"] = int(y_marg.sum())
    out["auc_marginal_leuer"] = auc_safe(y_marg, -leuer_clip)
    out["auc_marginal_surrogate"] = auc_safe(y_marg, -pred_ms)

    # resolve by regime: Spearman within each regime (rank ceiling resolved)
    out["by_regime"] = {}
    for name, a, b in D.REGIMES:
        m = (ms >= a) & (ms < b)
        if m.sum() >= 8:
            out["by_regime"][name] = dict(
                n=int(m.sum()),
                spearman_leuer=float(spearmanr(leuer_clip[m], ms[m]).statistic),
                spearman_surrogate=float(spearmanr(pred_ms[m], ms[m]).statistic))

    # the learned-m_s GAIN over the Leuer rank ceiling (held-out, all regimes)
    out["rank_gain_surrogate_over_leuer"] = out["spearman_surrogate"] - out["spearman_leuer_clipped"]
    out["auc_gain_surrogate_over_leuer"] = out["auc_marginal_surrogate"] - out["auc_marginal_leuer"]

    json.dump(out, open("data/phase4_leuer.json", "w"), indent=2)

    print("=== Leuer rigid-parameter baseline vs the learned m_s(shape) surrogate (held-out) ===")
    print(f"held-out rows: {out['n_heldout']} ({out['n_marginal_heldout']} marginal m_s<0.4)\n")
    print(f"Spearman(., m_s):  leuer raw {out['spearman_leuer_raw']:.3f} | "
          f"leuer clipped {out['spearman_leuer_clipped']:.3f} | surrogate {out['spearman_surrogate']:.3f}")
    print(f"Pearson(leuer, m_s) raw = {out['pearson_leuer_raw']:.3f}  (outlier-wrecked, as expected)")
    print(f"marginal-vs-controllable AUC:  leuer {out['auc_marginal_leuer']:.3f} | "
          f"surrogate {out['auc_marginal_surrogate']:.3f}")
    print("\n  Spearman by regime (leuer / surrogate):")
    for nm, d in out["by_regime"].items():
        print(f"    {nm:12s} n={d['n']:4d}  leuer {d['spearman_leuer']:+.3f}  surrogate {d['spearman_surrogate']:+.3f}")
    print(f"\nlearned-m_s gain over Leuer rank ceiling: Spearman {out['rank_gain_surrogate_over_leuer']:+.3f}, "
          f"marginal AUC {out['auc_gain_surrogate_over_leuer']:+.3f}")
    print("Saved data/phase4_leuer.json")


if __name__ == "__main__":
    main()
