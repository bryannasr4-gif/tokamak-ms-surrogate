"""
phase2_data.py -- shared data hygiene + feature definitions for Phase 2.

Single source of truth for: loading dataset_v1, the column-hygiene drops (constants +
redundants, per DATASET.md / RESULTS.md Phase-1.5 advice), the surrogate feature sets
(SHAPE = the headline m_s(shape) inputs; CONTROL = the independently-steerable inputs for
the dimensionality experiment), the m_s regime bins, and the control-vector extractor used
to REPLAY any dataset row through the true solver (phase15_lib.forward_label).

Verified on the shipped data (2026-06-20):
  * constants (zero variance): I_Solenoid=5000, n_unstable=1, n_positive_margins=1  -> DROP
  * redundant: Ip == Ip_target (<=6e-16 rel), tau_inst == 1/gamma (exact)            -> DROP one
  * I_P6 DOES vary (min -4.17, max 0.15, std 0.84) -- small magnitude but a live control;
    DATASET.md's "near-constant" note is misleading. Keep it as a control.
  * up-down symmetry: sq_uo==sq_lo and gap_top==gap_bot to ~1.0 corr (kept; not exact dupes).
"""
import os
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARQUET = os.path.join(ROOT, "data", "dataset_v1.parquet")

# Active coils in phase15_lib order (the order forward_label expects).
ACTIVE_COILS = ["Solenoid", "PX", "D1", "D2", "D3", "Dp", "D5", "D6", "D7", "P4", "P5", "P6"]

# --- column hygiene ----------------------------------------------------------
DROP_CONSTANT = ["I_Solenoid", "n_unstable", "n_positive_margins"]
DROP_REDUNDANT = ["Ip", "tau_inst"]          # Ip==Ip_target; tau_inst==1/gamma

# --- feature sets ------------------------------------------------------------
# The headline surrogate maps a low-D plasma SHAPE -> m_s. These are all OUTPUTS of the
# solve (emergent geometry + global profile descriptors), the quantities a designer reads off.
SHAPE_FEATURES = [
    "kappa", "delta", "delta_upper", "delta_lower",
    "sq_uo", "sq_ui", "sq_lo", "sq_li",
    "Rgeo", "a", "aspect", "Rmag", "Zaxis",
    "gap_min", "gap_inner", "gap_outer", "gap_top", "gap_bot",
    "li", "betap",
]

# The independently-steerable CONTROL inputs we sample directly (coil currents + profile
# params). Solenoid is held fixed (constant) so it is excluded -> 16 live controls.
CONTROL_FEATURES = [f"I_{c}" for c in ACTIVE_COILS if c != "Solenoid"] + [
    "paxis", "Ip_target", "fvac", "alpha_m", "alpha_n",
]

# --- m_s regimes (resolve every metric across these; never aggregate) --------
REGIMES = [
    ("marginal", 0.0, 0.4),     # m_s -> 0 controllability boundary (the floor is worst here)
    ("mid", 0.4, 1.0),
    ("stable", 1.0, 3.0),
    ("very_stable", 3.0, np.inf),
]


def regime_of(ms):
    for name, lo, hi in REGIMES:
        if lo <= ms < hi:
            return name
    return "very_stable"


def load(verify=True):
    """Load dataset_v1 with hygiene columns identified (not yet dropped). Returns the full df."""
    df = pd.read_parquet(PARQUET)
    if verify:
        assert df["I_Solenoid"].nunique() == 1, "I_Solenoid no longer constant"
        assert (np.abs(df["Ip"] - df["Ip_target"]) / df["Ip_target"]).max() < 1e-12
        assert np.allclose(df["tau_inst"], 1.0 / df["gamma"], equal_nan=True)
    df["regime"] = df["m_s"].apply(regime_of)
    return df


def clean(df):
    """Return df with hygiene columns dropped (keep one of each redundant pair)."""
    return df.drop(columns=[c for c in DROP_CONSTANT + DROP_REDUNDANT if c in df.columns])


def control_vector(row):
    """Length-12 active-coil current vector (phase15 order) from a dataset row, for REPLAY."""
    return np.array([row[f"I_{c}"] for c in ACTIVE_COILS], dtype=float)


def replay_kwargs(row):
    """kwargs for phase15_lib.forward_label to reproduce this row's equilibrium."""
    return dict(
        active_currents=control_vector(row),
        paxis=float(row["paxis"]), Ip=float(row["Ip_target"]), fvac=float(row["fvac"]),
        alpha_m=float(row["alpha_m"]), alpha_n=float(row["alpha_n"]),
    )


if __name__ == "__main__":
    df = load()
    print("loaded", df.shape)
    print("split:", dict(df["split"].value_counts()))
    print("regimes:", dict(df["regime"].value_counts()))
    print("shape features:", len(SHAPE_FEATURES), "control features:", len(CONTROL_FEATURES))
    print("clean shape:", clean(df).shape)
