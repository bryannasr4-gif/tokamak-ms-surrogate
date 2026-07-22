"""
phase2_calibrate.py -- Phase-2 component 2: honest uncertainty calibration.

The raw heteroscedastic deep ensemble is OVER-confident (RMS_z ~ 1.9 val / 2.9 test; worse in
the marginal m_s<0.4 bin). That is the expected deep-ensemble failure mode. We apply standard
post-hoc variance recalibration: fit a single global scale s on the VAL set so RMS_z->1, then
report CALIBRATED coverage on the clean held-out test_extrap corner, resolved by regime. We
also report the residual per-regime miscalibration (a diagnostic / limitation) and the
predictive-width-vs-m_s trend that shows the model WIDENS (abstains) toward the m_s->0 boundary.

Loads data/phase2_predictions.parquet (written by phase2_train.py). Saves
data/phase2_calibration.json (+ reliability-curve arrays for the figure).
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import norm, spearmanr

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, "experiments")
import phase2_data as D

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def coverage(z, levels):
    out = {}
    for lv in levels:
        k = norm.ppf(0.5 + lv / 2)
        out[f"cov_{int(lv*100)}"] = float(np.mean(np.abs(z) <= k))
    out["rms_z"] = float(np.sqrt(np.mean(z ** 2)))
    out["n"] = int(len(z))
    return out


def reliability_curve(z, grid):
    """Empirical central-interval coverage at each nominal level in grid."""
    return [float(np.mean(np.abs(z) <= norm.ppf(0.5 + lv / 2))) for lv in grid]


def main():
    df = pd.read_parquet("data/phase2_predictions.parquet")
    df["logms_true"] = np.log(df["m_s"])
    df["z_raw"] = (df["logms_true"] - df["logms_mean"]) / df["tot_std"]

    va = df[df.split == "val"]; te = df[df.split == "test_extrap"]

    # --- global recalibration scale fit on VAL (s = RMS_z so recalibrated RMS_z->1) ---
    s_global = float(np.sqrt(np.mean(va["z_raw"] ** 2)))
    df["z_cal"] = df["z_raw"] / s_global
    va = df[df.split == "val"]; te = df[df.split == "test_extrap"]

    levels = (0.5, 0.9, 0.95)
    grid = list(np.linspace(0.05, 0.99, 20))
    out = {"recal_scale_global": s_global,
           "method": "global variance scaling fit on val (s=RMS_z_val); reported on test_extrap"}

    for nm, dd in [("val", va), ("test_extrap", te)]:
        blk = {"raw": coverage(dd["z_raw"].values, levels),
               "calibrated": coverage(dd["z_cal"].values, levels),
               "reliability_grid": grid,
               "reliability_raw": reliability_curve(dd["z_raw"].values, grid),
               "reliability_cal": reliability_curve(dd["z_cal"].values, grid),
               "by_regime_calibrated": {}}
        for name, lo, hi in D.REGIMES:
            m = dd["regime"].values == name
            if m.sum() >= 5:
                blk["by_regime_calibrated"][name] = coverage(dd["z_cal"].values[m], levels)
        # predictive width vs m_s (abstention story): does tot_std grow as m_s->0?
        blk["width_vs_logms_spearman"] = float(spearmanr(dd["m_s"], dd["tot_std"]).statistic)
        blk["median_totstd_by_regime"] = {
            name: float(np.median(dd["tot_std"].values[dd["regime"].values == name]))
            for name, _, _ in D.REGIMES if (dd["regime"].values == name).sum() >= 5}
        out[nm] = blk

    with open("data/phase2_calibration.json", "w") as f:
        json.dump(out, f, indent=2)

    print(f"Global recalibration scale (fit on val): s={s_global:.2f}\n")
    for nm in ("val", "test_extrap"):
        b = out[nm]
        print(f"=== {nm} ===")
        print(f"  raw       : RMS_z={b['raw']['rms_z']:.2f} cov50={b['raw']['cov_50']:.2f} "
              f"cov90={b['raw']['cov_90']:.2f} cov95={b['raw']['cov_95']:.2f}")
        print(f"  calibrated: RMS_z={b['calibrated']['rms_z']:.2f} cov50={b['calibrated']['cov_50']:.2f} "
              f"cov90={b['calibrated']['cov_90']:.2f} cov95={b['calibrated']['cov_95']:.2f}")
        print(f"  width vs m_s Spearman={b['width_vs_logms_spearman']:+.2f} "
              f"(negative => widens toward m_s->0 = abstention)")
        print(f"  median tot_std by regime: " +
              ", ".join(f"{k}={v:.3f}" for k, v in b["median_totstd_by_regime"].items()))
        print("  calibrated coverage90 by regime: " +
              ", ".join(f"{k}={v['cov_90']:.2f}" for k, v in b["by_regime_calibrated"].items()))
        print()
    print("Saved data/phase2_calibration.json")


if __name__ == "__main__":
    main()
