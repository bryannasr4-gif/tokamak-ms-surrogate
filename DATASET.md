# DATASET — Phase 1.5 forward-sampled MAST-U vertical-stability dataset (`dataset_v1`)

**Artifact:** `data/dataset_v1.parquet` — 3298 converged, diverted spherical-tokamak (MAST-U-like)
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
- **Sampling distribution:** a one-time coarse set of 21 INVERSE solves over a (zscale, dR)
  grid provides anchor active-current vectors that CENTER the distribution on the diverted-ST
  manifold (anchors are NOT labels — they only seed the sampler). Each sample interpolates two random
  anchors, adds a log-normal global current scale + per-coil Gaussian jitter, and independently
  samples profile params (paxis, Ip, fvac, alpha_m, alpha_n). Every labelled sample is a clean
  forward solve.
- **Acceptance:** a sample is KEPT iff the forward solve converges (rel ≤ 1e-7), the plasma is
  DIVERTED (FreeGSNKE `flag_limiter == False`), it does not intersect the wall, and m_s is finite.
  Degenerate (near-zero-core) trajectories — which make the profile normalization hit Ip/I_R→0 — are
  intercepted (numpy raises on the divide) and cleanly rejected, never corrupting a label.
- **Reproducibility:** worker `c` draws from `np.random.default_rng([20260619, c])`; forward solves are
  bit-reproducible at the locked protocol. NOTE: the shipped artifact is a snapshot of a generation run
  that was INTERRUPTED by a machine power-off at 3298/3500 of the nominal target; each chunk is therefore
  a deterministic RNG *prefix* (per-chunk keep counts in `data/phase15_split_meta.json`), so the shipped
  set is reproducible from (seed, nchunks, protocol) + those per-chunk counts; re-running to target=3500
  would yield a superset. 3298 already exceeds the >=3k gate, and every kept row is a clean forward solve.
- **Yield:** ~89% of attempts converged+diverted. Rejections (of 400 total): non-convergence 246 (62%), non-finite m_s 93 (23%), limited plasma 61 (15%).
  ("non-finite m_s" = a converged diverted solve whose linearisation gave a non-finite margin, i.e. the
  m_s->0 boundary — a distinct third category, not just non-convergence/limited.)

## Contents — columns
- **Control inputs** (17): `I_<coil>` (12 active-coil currents, A) + `paxis, Ip_target, fvac,
  alpha_m, alpha_n`.
- **Shape descriptors** (21): `kappa` (elongation), `delta`/`delta_upper`/`delta_lower`
  (triangularity), `sq_uo/ui/lo/li` (Luce squareness, 4 quadrants), `Rgeo`, `a` (minor radius),
  `aspect`, `Rmag`/`Zaxis` (magnetic axis), `gap_min/inner/outer/top/bot` (LCFS→limiter clearances),
  `li` (internal inductance), `betap` (poloidal beta), `Ip` (achieved plasma current).
- **Labels** (6): `m_s` (Portone inductive stability margin; >0 stable, →0 = controllability
  boundary), `gamma` (vertical growth rate, 1/s), `leuer` (rigid Leuer ratio), `tau_inst` (1/γ, s),
  `n_unstable`, `n_positive_margins`.
- **Bookkeeping:** `fwd_rel_change`, `chunk`, `attempt`, `split`.
- **Safety-factor features (Phase 2.5b, in `dataset_v1_80q.parquet` only):** `q95` (safety factor at
  ψ_norm=0.95), `qmin` (min over the [0.05,0.95] grid), `q05` (≈ on-axis q proxy). Extracted by re-solving
  each shape FORWARD-ONLY (no m_s linearisation) and reading `eq.q(ψ_n)`; κ reproduces bit-exactly so q is
  read off the SAME equilibrium as the m_s label. **Physically sane:** q95 median 5.58, **100 % in the ST band
  [3,10]**; qmin>1 for 99.3 % (avoids m=1). corr(q95,κ)=+0.61, corr(q95,log m_s)=−0.48 — **but κ-MEDIATED:
  partial corr controlling for κ is only +0.16**, so q95 carries little m_s signal independent of shape.
  (`qmin ≡ q05`: q is monotone-increasing on the grid.)

## Dataset versions (which file to use)
- `dataset_v1.parquet` (3298 rows): original Phase-1.5 set, **40-mode** m_s labels (systematically low ~14 %).
- `dataset_v1_80.parquet` (3254 rows): **CANONICAL** — same shapes/controls re-labelled at the converged
  **80 modes** (Phase 2.5). Use this for all m_s work. Columns include `m_s`(=`m_s_A80`), `m_s_A40`, `gamma`,
  `leuer`, the shape descriptors, controls, `split`, `idx`.
- `dataset_v1_80q.parquet` (3254 rows): canonical **+ q95/qmin/q05 + the `split_kappa` column** (Phase 2.5b).
  Use this when you want q or the true high-κ extrapolation split.

## Column hygiene (verified on the shipped data — read before training)
- **Constant columns (zero variance):** `I_Solenoid` = 5000 A (held fixed by the sampler, so only 11 of
  the 12 active coils vary), and `n_unstable` = `n_positive_margins` = 1 for every row (each diverted ST
  equilibrium here has exactly one unstable n=0 vertical mode — a clean physics fact, not a learnable target).
  Drop these before training. **CORRECTION (Phase 2):** `I_P6` is NOT constant — it varies (−4.17..0.15 A,
  std 0.84), small in magnitude but a live control (do not drop it). An earlier note here was wrong.
- **Redundant columns:** `Ip` reproduces the control `Ip_target` to ~1e-16 relative (a forward-solve
  convergence check, not new information); `tau_inst` = 1/`gamma`. Keep one of each.
- **Cross-correlation (NOT independent axes):** the shape descriptors are an observational cloud on the
  diverted-ST manifold and are strongly correlated (e.g. |corr(kappa,sq_uo)|~0.77, |corr(delta,gap_outer)|~0.82,
  |corr(gap_inner,gap_outer)|~0.81; delta varies only over ~0.39-0.58). The EFFECTIVE shape dimensionality is
  well below the column count — any "N-D shape" experiment should orthogonalize (PCA) or work in the
  independently-steerable CONTROL variables (coil currents + profile params), not treat each descriptor as a
  free axis.

## Coverage (resolved, not aggregated)
| descriptor | min | median | max |
|---|---|---|---|
| elongation κ | 1.58 | 1.9 | 2.23 |
| triangularity δ | 0.393 | 0.489 | 0.581 |
| stability margin m_s | 0.000503 | 0.73 | 15.4 |
| growth rate γ [1/s] | 0.413 | 59.6 | 2.67e+06 |
| internal inductance li | 0.833 | 0.985 | 1.18 |
| poloidal β_p | 0.113 | 0.24 | 0.519 |
| inner gap [m] | 0.00116 | 0.0785 | 0.167 |
| outer gap [m] | 0.000641 | 0.156 | 0.414 |
| plasma current Ip [A] | 4.89e+05 | 5.98e+05 | 7.29e+05 |
- m_s regimes: **marginal m_s<0.4: 711**, mid 0.4–1.0: 1427, **stable m_s>1.0: 1160**.
- The single most-marginal label (m_s≈0.0005) has γ≈2.7e6/s — read as order-of-magnitude only (it is the
  point most exposed to the m_s→0 mode-truncation floor). ~1% of rows have LCFS→limiter clearance <5 mm
  (valid diverted, but near the diverted/limited boundary).

## Splits — TWO held-out schemes (document which one a result uses)
**(1) `split` — JOINT shape-corner (a generalization stress test, NOT true extrapolation):**
- `test_extrap` (187): a held-out **joint** corner — `kappa ≥ 2.0049` AND `delta ≥ 0.4930`
  (both together; exact cut points in `data/phase15_split_meta.json`). This is a COMPOSITIONAL hold-out:
  each criterion individually is well covered by training, so ~89 % of the corner lies INSIDE the training
  convex hull — it is a held-out high-elongation/high-triangularity SUB-REGION (interpolation-with-a-gap),
  a meaningful generalization stress test, **NOT** beyond-the-hull extrapolation. Its median m_s ≈ 0.48
  (mid-regime, **94 % mid / only 7 marginal**); the most marginal (m_s→0) plasmas are high-κ/**low**-δ and
  remain mostly in `train`, so this corner is **not** the marginal regime.
- `val` (466), `train` (2645). (At 80 modes the kept counts are 465 / 2602 / 187 of 3254.)

**(2) `split_kappa` — TRUE univariate high-κ extrapolation (Phase 2.5b; in `dataset_v1_80q.parquet`):**
- `test_extrap_kappa` (228): the **top ~7 % by elongation** (κ ≥ 2.078, the 93rd percentile). The entire
  train/val pool is held strictly BELOW that threshold (**train κ-max = 2.078**), so the surrogate never sees
  a single higher-κ shape and must EXTRAPOLATE in κ — the dominant m_s lever (corr(κ,log m_s) = −0.875). This
  IS beyond-hull extrapolation along κ. Composition: 163 marginal + 65 mid (71 % marginal), m_s median 0.30.
- `val` (467), `train` (2559). **Honest accuracy (regime-resolved RMSE_log, retrained surrogate):** MID
  in-dist 0.063 → tail **0.076** (modest +21 % extrapolation penalty); MARGINAL in-dist 0.242 → tail **0.133**
  (tail predicted *better* — high-κ marginals follow the smooth κ→m_s trend). The aggregate (tail 0.119 vs corner
  0.072) is **regime-mix-confounded** (corner 94 % mid, tail 71 % marginal); resolve by regime. The smooth
  surrogate extrapolates the κ trend with no collapse on the held-out tail.

## Noise floor (carried from Phase 0; read every result against it)
Within this locked config m_s is **bit-reproducible** (forward labels, OMP=1). Residual SYSTEMATIC
protocol-dependence of the absolute m_s remains: passive-mode truncation dominates, grid ≤7 %,
tolerance ≤10 %. These are systematic biases, not label noise. Do not present any effect below the
relevant floor as real.
**MODE CONVERGENCE — RESOLVED (Phase 2):** a 60-shape study at modes {40,80,120,138} (138 = all 138
passive structures = converged reference) shows these **40-mode labels are systematically LOW by median
−13.5 %**, regime-dependent: **marginal −27 %, mid −15 %, stable −13 %, very_stable −9 %** (worst at
m_s→0). **m_s is converged by 80 modes** (|m_s(80)−m_s(138)| median 0.4 %, max 3.4 %). This bias **cancels
in every fixed-modes relative/gradient comparison**, but absolute m_s values should be **re-labeled at 80
modes before publication** (a v2 dataset). See `data/phase2_modes_summary.json`, `figures/phase2_modes.png`.

## Limitations
- Ground truth is **FreeGSNKE-linearized m_s**, not experiment. m_s not mode-converged at 40 modes.
- Single machine / wall configuration (MAST-U-like). Profiles are the ConstrainPaxisIp α-family.
- The sampler is anchored to the diverted manifold, so coverage is dense there and sparse at the
  edges of control space (by design — we want valid diverted ST equilibria).

## License
Code + dataset: open. Built entirely on the open FreeGSNKE `MAST-U_like` tutorial machine files
(FreeGSNKE is LGPL-3.0). No proprietary or experimental data. Free to use with attribution.

*Generated by `experiments/phase15_dataset.py` from `data/phase15_chunk_*.json`. Protocol: PHASE0_PROTOCOL.md.*
