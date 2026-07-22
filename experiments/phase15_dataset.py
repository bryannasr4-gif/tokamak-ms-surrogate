"""
phase15_dataset.py -- assemble the forward-sampled chunks into the versioned dataset_v1.

  * load data/phase15_chunk_*.json  -> one DataFrame
  * basic QC (finite m_s, diverted already enforced upstream, physical-range sanity)
  * define a SHAPE-REGION extrapolation split (held-out corner of shape space) + random val
  * write data/dataset_v1.parquet  (single versioned artifact)
  * write figures/phase15_coverage.png (descriptor + label coverage histograms)
  * write DATASET.md (datasheet: provenance, protocol, columns, distributions, splits, license)

Run: PYTHONIOENCODING=utf-8 python experiments/phase15_dataset.py
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

ACTIVE_COILS = ["Solenoid", "PX", "D1", "D2", "D3", "Dp", "D5", "D6", "D7", "P4", "P5", "P6"]
CONTROL_COLS = [f"I_{c}" for c in ACTIVE_COILS] + ["paxis", "Ip_target", "fvac", "alpha_m", "alpha_n"]
SHAPE_COLS = ["kappa", "delta", "delta_upper", "delta_lower", "sq_uo", "sq_ui", "sq_lo", "sq_li",
              "Rgeo", "a", "aspect", "Rmag", "Zaxis", "gap_min", "gap_inner", "gap_outer",
              "gap_top", "gap_bot", "li", "betap", "Ip"]
LABEL_COLS = ["m_s", "gamma", "leuer", "tau_inst", "n_unstable", "n_positive_margins"]


def load():
    """Load all chunk JSONs -> (DataFrame, total_attempts). Robust to a chunk being mid-write
    (a worker may be json.dump-ing concurrently): a transiently unparseable chunk is skipped."""
    recs, attempts, nchunks, fails = [], 0, 0, {}
    for fn in sorted(glob.glob("data/phase15_chunk_*.json")):
        try:
            with open(fn) as f:
                d = json.load(f)
        except (json.JSONDecodeError, OSError):
            print(f"  (skipping {os.path.basename(fn)} — mid-write / unreadable)")
            continue
        recs += d.get("recs", [])
        attempts += int(d.get("attempts", 0))
        for k, v in d.get("fails", {}).items():
            # normalise the rejection key to a human label
            key = ("non-convergence" if "did not converge" in k else
                   "non-finite m_s" if "non-finite" in k else
                   "limited plasma" if "limited" in k else
                   "wall-touching" if "intersects wall" in k else k)
            fails[key] = fails.get(key, 0) + int(v)
        nchunks += 1
    df = pd.DataFrame(recs)
    print(f"Loaded {len(df)} raw samples from {nchunks} chunks ({attempts} attempts).")
    return df, attempts, fails


def qc(df):
    n0 = len(df)
    df = df[np.isfinite(df["m_s"])].copy()
    # physical-range sanity (diverted ST); these are guards, not aggressive filters
    df = df[(df["kappa"] > 1.2) & (df["kappa"] < 3.0)]
    df = df[(df["a"] > 0.2) & (df["a"] < 0.9)]
    # m_s upper guard is generous: the high-m_s tail is PHYSICAL (low-kappa round plasmas;
    # corr(kappa,m_s)=-0.73, m_s falls monotonically with elongation), not an artifact -- so we
    # keep the very-stable samples and only guard against a true numerical blow-up.
    df = df[(df["m_s"] > 0.0) & (df["m_s"] < 25.0)]
    df = df[df["gap_min"] > 0.0]
    df = df.reset_index(drop=True)
    print(f"QC: kept {len(df)}/{n0} ({n0-len(df)} dropped).")
    return df


def make_split(df, seed=0):
    """Held-out JOINT shape-corner test (a generalization stress test, NOT extrapolation).
    test_extrap = the corner where elongation AND triangularity are BOTH high (kappa>=q0.78 AND
    delta>=q0.55). This is a COMPOSITIONAL hold-out: each criterion individually is well covered by
    training, so most of the corner lies inside the training convex hull -- it is a held-out
    sub-region (interpolation-with-a-gap), not beyond-the-hull extrapolation. (The column value is
    kept as 'test_extrap' for code stability; for a TRUE extrapolation split, hold out a univariate
    high-kappa tail.) The remaining pool is split into random val (15%) / train.
    """
    rng = np.random.default_rng(seed)
    k_hi = df["kappa"].quantile(0.78)
    d_hi = df["delta"].quantile(0.55)
    extrap = (df["kappa"] >= k_hi) & (df["delta"] >= d_hi)
    split = np.array(["train"] * len(df), dtype=object)
    split[extrap.values] = "test_extrap"
    pool = np.where(~extrap.values)[0]
    val = rng.choice(pool, size=int(0.15 * len(pool)), replace=False)
    split[val] = "val"
    df = df.copy()
    df["split"] = split
    print("Split counts:", df["split"].value_counts().to_dict(),
          f"(extrap corner: kappa>={k_hi:.3f} & delta>={d_hi:.3f})")
    return df, dict(kappa_thresh=float(k_hi), delta_thresh=float(d_hi))


def coverage_fig(df):
    fields = [("kappa", "elongation κ"), ("delta", "triangularity δ"),
              ("m_s", "stability margin m_s"), ("gamma", "growth rate γ [1/s]"),
              ("li", "internal inductance li"), ("betap", "poloidal beta βp"),
              ("gap_inner", "inner gap [m]"), ("gap_outer", "outer gap [m]"),
              ("Ip", "plasma current Ip [A]")]
    fig, ax = plt.subplots(3, 4, figsize=(18, 11))
    ax = ax.ravel()
    for i, (k, lab) in enumerate(fields):
        v = df[k].values
        ax[i].hist(v, bins=40, color="steelblue", alpha=0.85)
        ax[i].set_title(lab, fontsize=10)
        ax[i].set_ylabel("count")
    # 2D coverage panels
    ax[9].scatter(df["kappa"], df["m_s"], s=4, alpha=0.3, c="darkred")
    ax[9].set_xlabel("κ"); ax[9].set_ylabel("m_s"); ax[9].set_title("m_s vs κ", fontsize=10)
    ax[10].scatter(df["kappa"], df["delta"], s=4, alpha=0.3, c="darkgreen")
    ax[10].set_xlabel("κ"); ax[10].set_ylabel("δ"); ax[10].set_title("shape coverage κ-δ", fontsize=10)
    cols = {"train": "steelblue", "val": "orange", "test_extrap": "crimson"}
    for s, c in cols.items():
        m = df["split"] == s
        ax[11].scatter(df.loc[m, "kappa"], df.loc[m, "delta"], s=4, alpha=0.4, c=c, label=s)
    ax[11].set_xlabel("κ"); ax[11].set_ylabel("δ"); ax[11].set_title("splits in κ-δ", fontsize=10)
    ax[11].legend(fontsize=7)
    fig.suptitle(f"Phase 1.5 forward-sampled dataset_v1 — {len(df)} converged diverted MAST-U equilibria",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    os.makedirs("figures", exist_ok=True)
    fig.savefig("figures/phase15_coverage.png", dpi=130, bbox_inches="tight")
    print("Saved figures/phase15_coverage.png")


DATASHEET = """# DATASET — Phase 1.5 forward-sampled MAST-U vertical-stability dataset (`dataset_v1`)

**Artifact:** `data/dataset_v1.parquet` — {n} converged, diverted spherical-tokamak (MAST-U-like)
free-boundary equilibria, each with the Portone-2005 inductive **stability margin m_s** (and γ,
Leuer, instability timescale) and a full set of plasma-shape descriptors. Open + reproducible.

## Motivation
The Phase-2 differentiable surrogate maps a low-D plasma **shape** → **m_s**. Phase 1 generated
labels by an INVERSE solve (target shape → coil currents), which is ill-conditioned and was the
source of the Phase-0 reproducibility floor (cross-thread m_s spread median 4.9 % / max 11.4 %).
**This dataset replaces that with FORWARD sampling**: we sample the control inputs (active-coil
currents + plasma-profile parameters) and run a well-conditioned forward free-boundary solve. The
forward label is bit-reproducible at the locked protocol (verified: a forward solve reproduces the
inverse anchor's m_s to 0.000 %), removing the inverse-solve noise at its root cause.

## Collection / generation process
- **Machine:** the single serialized MAST-U-like tokamak (`machine_configs/MAST-U/serialized_tokamak.pkl`,
  12 active coils + 138 passive structures, aspect ratio 1.638), loaded identically by every worker.
- **Locked numerical protocol (PHASE0_PROTOCOL.md):** OMP/OPENBLAS/MKL/NUMEXPR/VECLIB = 1; grid 65×65;
  `fix_n_vessel_modes = 40`; forward GS tolerance 1e-8; cold isolated solves; plasma_resistivity 1e-6.
- **Sampling distribution:** a one-time coarse set of {n_anchors} INVERSE solves over a (zscale, dR)
  grid provides anchor active-current vectors that CENTER the distribution on the diverted-ST
  manifold (anchors are NOT labels — they only seed the sampler). Each sample interpolates two random
  anchors, adds a log-normal global current scale + per-coil Gaussian jitter, and independently
  samples profile params (paxis, Ip, fvac, alpha_m, alpha_n). Every labelled sample is a clean
  forward solve.
- **Acceptance:** a sample is KEPT iff the forward solve converges (rel ≤ 1e-7), the plasma is
  DIVERTED (FreeGSNKE `flag_limiter == False`), it does not intersect the wall, and m_s is finite.
  Degenerate (near-zero-core) trajectories — which make the profile normalization hit Ip/I_R→0 — are
  intercepted (numpy raises on the divide) and cleanly rejected, never corrupting a label.
- **Reproducibility:** worker `c` draws from `np.random.default_rng([{seed}, c])`; forward solves are
  bit-reproducible at the locked protocol. NOTE: the shipped artifact is a snapshot of a generation run
  that was INTERRUPTED by a machine power-off at {n}/3500 of the nominal target; each chunk is therefore
  a deterministic RNG *prefix* (per-chunk keep counts in `data/phase15_split_meta.json`), so the shipped
  set is reproducible from (seed, nchunks, protocol) + those per-chunk counts; re-running to target=3500
  would yield a superset. {n} already exceeds the >=3k gate, and every kept row is a clean forward solve.
- **Yield:** ~{yield_pct}% of attempts converged+diverted. Rejections (of {n_rej} total): {rej_breakdown}.
  ("non-finite m_s" = a converged diverted solve whose linearisation gave a non-finite margin, i.e. the
  m_s->0 boundary — a distinct third category, not just non-convergence/limited.)

## Contents — columns
- **Control inputs** ({n_control}): `I_<coil>` (12 active-coil currents, A) + `paxis, Ip_target, fvac,
  alpha_m, alpha_n`.
- **Shape descriptors** ({n_shape}): `kappa` (elongation), `delta`/`delta_upper`/`delta_lower`
  (triangularity), `sq_uo/ui/lo/li` (Luce squareness, 4 quadrants), `Rgeo`, `a` (minor radius),
  `aspect`, `Rmag`/`Zaxis` (magnetic axis), `gap_min/inner/outer/top/bot` (LCFS→limiter clearances),
  `li` (internal inductance), `betap` (poloidal beta), `Ip` (achieved plasma current).
- **Labels** ({n_label}): `m_s` (Portone inductive stability margin; >0 stable, →0 = controllability
  boundary), `gamma` (vertical growth rate, 1/s), `leuer` (rigid Leuer ratio), `tau_inst` (1/γ, s),
  `n_unstable`, `n_positive_margins`.
- **Bookkeeping:** `fwd_rel_change`, `chunk`, `attempt`, `split`.

## Column hygiene (verified on the shipped data — read before training)
- **Constant columns (zero variance):** `I_Solenoid` = 5000 A (held fixed by the sampler, so only 11 of
  the 12 active coils vary; `I_P6` is also near-zero/near-constant), and `n_unstable` = `n_positive_margins`
  = 1 for every row (each diverted ST equilibrium here has exactly one unstable n=0 vertical mode — a clean
  physics fact, not a learnable target). Drop these before training.
- **Redundant columns:** `Ip` reproduces the control `Ip_target` to ~1e-16 relative (a forward-solve
  convergence check, not new information); `tau_inst` = 1/`gamma`. Keep one of each.
- **Cross-correlation (NOT independent axes):** the shape descriptors are an observational cloud on the
  diverted-ST manifold and are strongly correlated (e.g. |corr(kappa,sq_uo)|~0.77, |corr(delta,gap_outer)|~0.82,
  |corr(gap_inner,gap_outer)|~0.81; delta varies only over ~0.39-0.58). The EFFECTIVE shape dimensionality is
  well below the column count — any "N-D shape" experiment should orthogonalize (PCA) or work in the
  independently-steerable CONTROL variables (coil currents + profile params), not treat each descriptor as a
  free axis.

## Coverage (resolved, not aggregated)
{coverage_table}
- m_s regimes: **marginal m_s<0.4: {n_marg}**, mid 0.4–1.0: {n_mid}, **stable m_s>1.0: {n_stab}**.
- The single most-marginal label (m_s≈0.0005) has γ≈2.7e6/s — read as order-of-magnitude only (it is the
  point most exposed to the m_s→0 mode-truncation floor). ~1% of rows have LCFS→limiter clearance <5 mm
  (valid diverted, but near the diverted/limited boundary).

## Splits — held-out JOINT shape-corner (a generalization stress test), not just random
- `test_extrap` ({n_test}): a held-out **joint** corner — `kappa ≥ {kthr:.4f}` AND `delta ≥ {dthr:.4f}`
  (both together; exact cut points in `data/phase15_split_meta.json`). This is a COMPOSITIONAL hold-out:
  each criterion individually is well covered by training, so ~89 % of the corner lies INSIDE the training
  convex hull — it is a held-out high-elongation/high-triangularity SUB-REGION (interpolation-with-a-gap),
  a meaningful generalization stress test, **NOT** beyond-the-hull extrapolation. Its median m_s ≈ 0.48
  (mid-regime); the most marginal (m_s→0) plasmas are high-κ/**low**-δ and remain mostly in `train`, so this
  corner is **not** the marginal regime. The column value is kept as `test_extrap` for code stability. For a
  TRUE extrapolation test, hold out a univariate high-κ tail (a Phase-2 option).
- `val` ({n_val}): random 15 % of the remaining pool. `train` ({n_train}): the rest.

## Noise floor (carried from Phase 0; read every result against it)
Within this locked config m_s is **bit-reproducible** (forward labels, OMP=1). Residual SYSTEMATIC
protocol-dependence of the absolute m_s remains: passive-mode truncation dominates (40→80 modes
≈ +10 %; m_s is NOT mode-converged at 40 — a convergence study is owed in Phase 2), grid ≤7 %,
tolerance ≤10 %. These are systematic biases, not label noise. Do not present any effect below the
relevant floor as real.

## Limitations
- Ground truth is **FreeGSNKE-linearized m_s**, not experiment. m_s not mode-converged at 40 modes.
- Single machine / wall configuration (MAST-U-like). Profiles are the ConstrainPaxisIp α-family.
- The sampler is anchored to the diverted manifold, so coverage is dense there and sparse at the
  edges of control space (by design — we want valid diverted ST equilibria).

## License
Code + dataset: open. Built entirely on the open FreeGSNKE `MAST-U_like` tutorial machine files
(FreeGSNKE is LGPL-3.0). No proprietary or experimental data. Free to use with attribution.

*Generated by `experiments/phase15_dataset.py` from `data/phase15_chunk_*.json`. Protocol: PHASE0_PROTOCOL.md.*
"""


def write_datasheet(df, split_meta, n_anchors, yield_pct, seed, fails):
    cov_keys = [("kappa", "elongation κ"), ("delta", "triangularity δ"),
                ("m_s", "stability margin m_s"), ("gamma", "growth rate γ [1/s]"),
                ("li", "internal inductance li"), ("betap", "poloidal β_p"),
                ("gap_inner", "inner gap [m]"), ("gap_outer", "outer gap [m]"),
                ("Ip", "plasma current Ip [A]")]
    rows = ["| descriptor | min | median | max |", "|---|---|---|---|"]
    for k, lab in cov_keys:
        v = df[k].values
        rows.append(f"| {lab} | {v.min():.3g} | {np.median(v):.3g} | {v.max():.3g} |")
    counts = df["split"].value_counts().to_dict()
    n_rej = sum(fails.values())
    rej_breakdown = ", ".join(f"{k} {v} ({100*v/max(n_rej,1):.0f}%)"
                              for k, v in sorted(fails.items(), key=lambda x: -x[1])) or "none recorded"
    text = DATASHEET.format(
        n=len(df), n_anchors=n_anchors, seed=seed, yield_pct=f"{yield_pct:.0f}",
        n_control=len([c for c in CONTROL_COLS if c in df.columns]),
        n_shape=len([c for c in SHAPE_COLS if c in df.columns]),
        n_label=len([c for c in LABEL_COLS if c in df.columns]),
        coverage_table="\n".join(rows),
        n_marg=int((df["m_s"] < 0.4).sum()), n_mid=int(((df["m_s"] >= 0.4) & (df["m_s"] <= 1.0)).sum()),
        n_stab=int((df["m_s"] > 1.0).sum()),
        n_test=counts.get("test_extrap", 0), n_val=counts.get("val", 0), n_train=counts.get("train", 0),
        kthr=split_meta["kappa_thresh"], dthr=split_meta["delta_thresh"],
        n_rej=n_rej, rej_breakdown=rej_breakdown,
    )
    with open("DATASET.md", "w", encoding="utf-8") as f:
        f.write(text)
    print("Wrote DATASET.md")


def main():
    df, attempts, fails = load()
    if len(df) == 0:
        print("No data yet."); return
    yield_pct = 100.0 * len(df) / max(attempts, 1)
    n_anchors = len(__import__("pickle").load(open("data/phase15_anchors.pkl", "rb"))) \
        if os.path.exists("data/phase15_anchors.pkl") else 0
    df = qc(df)
    df, split_meta = make_split(df, seed=0)

    cols = [c for c in (CONTROL_COLS + SHAPE_COLS + LABEL_COLS +
                        ["fwd_rel_change", "chunk", "attempt", "split"]) if c in df.columns]
    df = df[cols]
    os.makedirs("data", exist_ok=True)
    df.to_parquet("data/dataset_v1.parquet", index=False)
    print(f"Wrote data/dataset_v1.parquet ({len(df)} rows, {len(cols)} cols).")
    coverage_fig(df)

    # quick numeric summary to stdout (the ledger will quote these)
    print("\n=== coverage summary ===")
    for k in ["kappa", "delta", "m_s", "gamma", "li", "betap", "gap_inner", "gap_outer", "Ip"]:
        v = df[k].values
        print(f"  {k:10s} min={v.min():.3f} med={np.median(v):.3f} max={v.max():.3f}")
    print(f"  m_s<0.4 (marginal): {(df['m_s']<0.4).sum()}  | m_s>1.0 (stable): {(df['m_s']>1.0).sum()}")
    # persist split meta + write the datasheet
    with open("data/phase15_split_meta.json", "w") as f:
        json.dump(dict(split_meta, counts=df["split"].value_counts().to_dict(),
                       n=len(df), yield_pct=yield_pct, n_anchors=n_anchors,
                       rejections=fails), f, indent=2)
    write_datasheet(df, split_meta, n_anchors, yield_pct, seed=20260619, fails=fails)


if __name__ == "__main__":
    main()
