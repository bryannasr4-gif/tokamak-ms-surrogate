# Tier-1 Pre-Registration — Real-Data Anchoring of the Differentiable m_s Surrogate

**Status:** frozen before the production re-solve/analysis run. Commit this file's git hash; nothing
downstream (slice choice, metrics, thresholds, framing) may be tuned after seeing solver/surrogate output.
**Date frozen:** 2026-06-30. **Machine:** Windows-canonical (all solves here; never mixed with Mac/Device-C).

---

## 0. Framing (the honest scope — read first)

This is an **emulator out-of-distribution / shape-coverage / solver-consistency STRESS-TEST** of the
learned, differentiable Portone-m_s surrogate against **real, open, EFIT reconstructions from the ORIGINAL
MAST spherical tokamak** (the predecessor of MAST-U; a *different machine* — different coils and passive
vessel; MAST reached higher elongation). It is **NOT** a validation of vertical-stability physics against
experiment, and there is **no experimentally-measured m_s label**.

Why real MAST and not MAST-U: MAST-U EFIT++ reconstructions are **not publicly downloadable** (UKAEA-gated).
FAIR-MAST openly hosts original-MAST EFIT (campaigns M5–M9). Data via the FAIR-MAST S3 Zarr (anon,
`https://s3.echo.stfc.ac.uk`), read with xarray; the `efm` group provides per-slice stored LCFS
(`lcfs_r/lcfs_z`), derived scalars (`elongation`, `triang_upper/lower`, `li`, `betap`, `q_95`,
`minor_radius`, `geom_axis_rc/zc`, `magnetic_axis_r/z`), real profiles (`pprime`, `ffprime` on `psi_norm`),
`plasma_current_x` (Ip, A), `bvac_r/bvac_val` (→ fvac), X-points (`xpoint1/2_rc/zc`), and quality
(`final_chisq`, `cnvrgd_times`).

**Reference template:** arXiv:2407.12432 (Pentland et al.) — FreeGSNKE forward-GS validated vs MAST-U
EFIT++. We follow its spirit (treat EFIT as a realistic *shape prior*, not ground truth) but our re-solve
is **cross-machine** (MAST shapes on the MAST-U model), so agreement will be looser than their same-machine
numbers, and the resulting m_s is **model-vs-model** (see must-not-claim).

## 1. The three tests

- **(a) Shape-realism / coverage.** Do real MAST equilibrium *shapes* fall inside our synthetic MAST-U
  training distribution? Compute real descriptors (our-convention, from `lcfs_r/lcfs_z` + EFIT `li`,
  `betap`) and score each real slice's **Mahalanobis distance** in full standardized descriptor space AND
  **k-NN distance** to training points; threshold at the **99th-percentile training self-distance**. Report
  the **OOD fraction** (not just in/out of a 2-D PCA hull). PCA scatter is a *visual only* (state variance
  explained). **No solve needed for (a)** — it is the robust primary result.
- **(b) Emulator self-consistency.** Re-solve each real LCFS on the fixed MAST-U model (shape-anchored
  inverse solve, real profiles), then compare the **surrogate's predicted m_s** to **FreeGSNKE's own
  80-mode m_s** on that re-solved geometry. Judged **relative to the solver's own 40↔80-mode label
  ambiguity**, not against reality. This bounds surrogate *emulation* error on real-shape-derived geometry.
- **(c) Descriptor fidelity.** Recompute ALL descriptors from the raw LCFS with the **same code** on both
  the native-EFIT boundary and our re-solved boundary. Drop cross-machine gaps and q (limiter/Ip/Bt differ).

## 2. Slice selection (deterministic, metadata-only, frozen)

1. **Shot set:** original-MAST campaign **M9** (latest MAST, closest to MAST-U), shots with a *complete*
   `efm` (has `lcfs_r`, `elongation`, `li`, `betap`, `pprime`). Shot list frozen in
   `data/tier1_shotlist.json` (published with candidate count).
2. **Per-slice quality gate:** finite `final_chisq`; flat-top only (|Ip| ≥ 0.7·max|Ip| for the shot AND
   |dIp/dt| small); `final_chisq` below the shot's own converged-window level; away from ramp-up/ramp-down/
   disruption; valid closed `lcfs_r/lcfs_z`; `li ∈ [0.5,1.6]`, `betap > 0`, `elongation ∈ [1.2,3.0]`.
3. **Deduplicate/target:** farthest-point sampling in standardized `(kappa, delta, li, betap)` to a target
   of **~60 slices** (pull ≥100 candidates first), **deliberately including the high-κ tail** (the OOD
   region). Report full candidate list + per-slice drop reasons + **yield = kept/candidates**.
4. **Independence:** report the number of **independent shots**; cluster/CI by shot (flat-top slices within
   a shot are autocorrelated — never treat as i.i.d. n).

## 3. Re-solve protocol (test b)

Locked numerical protocol (continuity with the trained surrogate): serialized MAST-U machine
(`machine_configs/MAST-U/serialized_tokamak.pkl`), OMP=1, 65×65, Rmin=0.1 (note deviation from the paper's
0.06 — verified real LCFS inner point > 0.1 so no clipping), cold solves. Shape-anchored inverse solve:
`lcfs_r/lcfs_z` subsampled to ~40 points as one `isoflux_set`; EFIT X-points as `null_points`; MAST-U
coil-current limits + `l2_reg`. **Profiles:** ingest real EFIT `pprime`/`ffprime` via
`GeneralPprimeFFprime` (real Ip + fvac, `Ip_logic=True`); fall back to a fixed profile only if that fails,
recorded per slice. Compute `descriptors()` + FreeGSNKE m_s at **both 40 and 80 vessel modes**. Crash-proof
(np.errstate guard) + resume-safe (per-slice output files).

## 4. Acceptance bands (pre-registered)

- **Intrinsic-ambiguity anchor:** report the 40↔80-mode m_s spread ON the real-shape re-solved equilibria
  (synthetic reference: median rel |diff| = 0.141, p90 0.268, p95 0.324). Every (b) number is judged
  relative to this.
- **(b) in-hull PASS:** median |rel m_s error| (surrogate vs FreeGSNKE-80) ≤ 0.14 AND p90 ≤ 0.27
  → "within solver's own ambiguity." Looser → honest emulator-OOD degradation (still reported).
- **(b) out-of-hull:** NO threshold; report |rel error| vs OOD distance (smooth rise = graceful; cliff =
  honest negative). This conditional is the scientifically valuable result.
- **(a) coverage:** OOD fraction above the 99th-pctile training self-distance; also 95th. No single pass
  threshold — the honest headline is the in-hull vs out-of-hull residual conditional.
- **Profile-match gate (counts toward b only):** |li_resolved − li_EFIT| ≤ 0.1 AND |betap_resolved −
  betap_EFIT| ≤ 0.1; mismatched slices reported separately. li ±0.1 sensitivity band on m_s reported.
- **(c) definitional consistency:** after same-code recompute on both sides, |κ_ours − κ_EFIT| ≤ 0.03 and
  |δ_ours − δ_EFIT| ≤ 0.03; larger ⇒ methodology bug, not physics.
- **Inverse-solve yield floor:** overall convergence ≥ ~70%; high-κ subgroup yield must not trail low-κ by
  > ~20 pts, else the "covers marginal ST" claim is unsupported (report as limitation).
- **Boundary shape agreement (informational, arXiv:2407.12432 metrics):** report ζ (max point-to-point LCFS
  distance) and η (area mismatch); ours will exceed their same-machine < 1 cm because we re-solve
  cross-machine.

## 5. What we WILL claim / MUST NOT claim

**Can claim:** (i) real MAST shapes fall [X%] inside / [Y%] beyond our synthetic training distribution
(quantified); (ii) on real-shape-derived geometries the surrogate stays consistent with its own solver to
within [metric], [tighter than/comparable to] the solver's 40↔80 ambiguity; (iii) "to our knowledge, first
grounding of a differentiable Portone-m_s surrogate against real open EFIT shapes via coverage +
solver-consistency" (hedge; DECAF touches real-EFIT vertical stability — scope the 'first' tightly).

**Must NOT claim:** "validated"/"experimentally validated"; "MAST-U" (it is original MAST); "recovered the
real discharges' m_s" (re-solved m_s is model-vs-model, passive-structure-dominated — a fiction for real
stability); "accuracy vs real data" (there is no real m_s label → it is CONSISTENCY + COVERAGE);
"cross-device transfer demonstrated" (physics underneath is still MAST-U — the genuine 2nd device remains
owed = Option 3); "novel OOD method"; "first real-data fusion-ML anchoring" (DECAF predates).

## 6. Deliverables

`experiments/tier1_*.py` (data lib + re-solve + analysis), `data/tier1_*.{json,parquet}` (candidate list,
per-slice results, yield, dropped-set characterization), `figures/tier1_{coverage,consistency,fidelity,
gallery}.png`, a ledger entry in `RESULTS.md`, and an adversarial review pass before any headline is
finalized.

## AMENDMENT 2026-07-13 — profile-ingestion deviation (ConstrainBetapIp), documented post-hoc

*All text above this heading is the original frozen pre-registration, byte-identical to the version
frozen at commit `09df502` (verified by diff against `git show 09df502:TIER1_PREREG.md` on
2026-07-13). Nothing above was edited; this amendment is append-only.*

1. **The pre-registered sentence deviated from (§3, verbatim):** "**Profiles:** ingest real EFIT
   `pprime`/`ffprime` via `GeneralPprimeFFprime` (real Ip + fvac, `Ip_logic=True`); fall back to a
   fixed profile only if that fails, recorded per slice."

2. **What the pipeline actually did, and when it entered the code.** The Tier-1 re-solve worker is
   `experiments/tier1_resolve_worker.py` (confirmed by reading it: `resolve_slice()` performs the
   shape-anchored inverse solve and the 40/80-mode m_s computation for every Tier-1 slice).
   `git log --follow --format="%h %ad %s" -S "ConstrainBetapIp" -- experiments/tier1_resolve_worker.py`
   returns exactly one commit: **`09df502` (2026-06-30)** — the pre-registration/pipeline freeze
   commit itself. Inspection of that commit's diff confirms FUNCTIONAL use (not a comment): it adds
   `profiles = ConstrainBetapIp(eq=eq, betap=real_betap, Ip=Ip, fvac=fvac, alpha_m=1.8, alpha_n=1.2)`
   as the primary profile constructor in `resolve_slice()` (with `ConstrainPaxisIp` as the
   exception-path fallback). So `ConstrainBetapIp` — not `GeneralPprimeFFprime` — was the pipeline
   DEFAULT from the first committed version of the worker.

3. **Ordering evidence (topological, not date-based):** `git merge-base --is-ancestor 09df502 83859f8`
   returns TRUE (exit 0) — the commit introducing functional `ConstrainBetapIp` use is a topological
   ancestor of `83859f8`, the commit that produced the test-b analysis output (defined as
   `data/tier1_analysis.json` + the RESULTS.md Tier-1 entry). Dates: `09df502` = 2026-06-30;
   `83859f8` = 2026-07-01. The deviation therefore PRECEDES all test-b analysis output; no analysis
   was produced under the pre-registered profile path.

   **Mechanism (as recorded in the ledger):** `GeneralPprimeFFprime` required a runtime typo patch
   (`tier1_lib.patch_freegs4e_profile_bug`) AND is unsupported by `nl_solver` (no
   `n_profiles_parameters`), so the fixed-parametric-profile fallback was **convergence-forced** —
   consistent with the prereg's fall-back clause ("fall back to a fixed profile only if that fails")
   but BROADER than it: the fallback became the pipeline default for every slice, rather than a
   recorded per-slice exception.

4. **Consequences for the pre-registered analysis:**
   - `ConstrainBetapIp`'s betap target is not `descriptors()`' poloidalBeta2, so the re-solve βp is
     systematically inflated: the in-distribution control measures the pipeline's own inflation at
     **~4.8×** (`data/tier1_indist_summary.json`, n=24 recompute 2026-07-13: `betap_inflation` = 4.82;
     the original n=20 value was 4.85). On the real slices the re-solved βp median is 1.248 vs real
     EFIT 0.278 (~4.5×; recomputed from `data/tier1_resolved/*.json`).
   - The pre-registered §4 profile-match gate (|dli| ≤ 0.1 AND |dbetap| ≤ 0.1) passes **0/55**
     status-ok slices (recomputed from `data/tier1_resolved/*.json`, 2026-07-13) — the gate is
     **VOIDED as a gate outcome** (a consequence of the profile-family deviation, NOT a surrogate
     accuracy failure). Test (b) is therefore outside the pre-registered profile-match regime, as
     disclosed in the RESULTS.md Tier-1 entry.

5. Documented 2026-07-13, Stage-0 ledger-integrity pass (EXEC-S0.2 TASK b; gate G-S0.2-2).
