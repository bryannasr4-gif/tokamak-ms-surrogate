<!-- DOI badge (Zenodo concept DOI, all versions): -->
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21499720.svg)](https://doi.org/10.5281/zenodo.21499720)

# tokamak-ms-surrogate

**A learned, amortized, *differentiable* surrogate for the Portone-2005 vertical
stability margin `m_s` of a spherical tokamak (MAST-U), whose gradient is *used* to
drive solver-confirmed shape design — on open, reproducible FreeGSNKE data.**

This repository is a **fresh-history public snapshot** of a private research project.
It contains the code, synthetic datasets, verified numerical artifacts, and figures
behind the results below. Every headline number in the claims map is read directly
from a named JSON artifact in `data/` (see the reproduction map).

> **What this is not.** This is **not** a "first neural stability surrogate" (learned
> γ surrogates predate it — Pertnet, Wai/Boyer/Kolemen 2022). The defensible novelty is
> the **conjunction**: learned + amortized + *differentiable, gradient used for
> solver-confirmed shape design* + open synthetic data + the Portone `m_s` target on a
> spherical tokamak. Please read the **Scope & honest limitations** section — the
> negative results are part of the contribution.

---

## Relationship to the manuscript (read this first)

A manuscript drawing on this work is in preparation for *Plasma Physics and Controlled
Fusion*. **This snapshot is not the manuscript's complete artifact set, and the two are
deliberately not identical in coverage.** Specifically:

- **The confirmatory statistic is here.** The pre-registered 28-start fixed-κ cohort —
  **25/28, two-sided Wilcoxon p = 2.4×10⁻⁵** — ships with its pre-registration, its raw
  per-start records and the code that produced it (claim 1 below).
- **The manuscript's conventional-aspect transfer result is *not* here.** The manuscript
  reports that the surrogate fails to transfer to a conventional-aspect (DIII-D-geometry)
  equilibrium at `A = 2.80`, ≈22σ out of distribution, where its gradient saturates into
  a numb, weakly anti-aligned signal. **Those artifacts and that machine geometry are held
  until the Stage-2 pre-registration (PREREG v2) is frozen**, and are not in this
  snapshot. What *is* here is the earlier Device-C zero-shot transfer failure (claim 7),
  which is a weaker test on a radial transform of MAST-U rather than a structurally
  distinct machine. Do not read claim 7 as the manuscript's transfer result; it is not.

If you are checking a claim you read in the manuscript and cannot find its artifact here,
that is the reason, and it is deliberate rather than an oversight.

> **Erratum, 2026-08-04.** Earlier versions of this README named the 20-start cohort
> (16/20, p = 0.0017) as "the confirmatory statistic". **That was wrong.** That cohort is
> **exploratory** and the manuscript relabels it as such: it was chosen after the fact,
> which is precisely why the pre-registered 28-start cohort was run. The confirmatory
> statistic is 25/28, p = 2.4×10⁻⁵. No underlying data changed; the label was incorrect
> and is corrected here.

---

## Stability margin `m_s` (what the surrogate predicts)

`m_s` is the Portone-2005 **inductive** vertical stability margin (Portone, *Nucl.
Fusion* 45 (2005) 926; the multi-machine form FreeGSNKE evaluates is Humphreys et al.,
*Nucl. Fusion* 49 (2009) 115003). It is **dimensionless and resistivity-independent**.
Convention used throughout: **`m_s` > 0 = stable** (passively/resistively unstable but
controllable); `m_s` = 0 = ideal-MHD marginal; **bigger `m_s` = more stable**;
`m_s` → 0 is the controllability boundary. Independent recomputation matches FreeGSNKE
to 0.000% (three ways). Practical design thresholds referenced in the literature:
`m_s` ≳ 0.15 (FUSE), ~0.26 (TCV/C-Mod control). `m_s*` = 1.0 in the design experiments
is a **demonstration target**, not an operational spec.

**Absolute `m_s` magnitudes are not reportable from this work.** They are
simulator-configuration dependent and are not recovered by refining the vessel basis (see
limitations). Every claim below is *relative and ordinal*: paired win-rates between arms
at equal budget. Where a median gain is quoted it is an internal simulator Δ`m_s` at the
locked protocol, given **for scale only** — the inferential content is the win-rate.

---

## Reproduction map (claim → protocol → script → artifact → figure)

Every number below is read from the cited JSON artifact (all cited artifacts are present
in this snapshot). Statistics are two-sided; `n` fixed in advance per family. Each row is
labelled **confirmatory**, **supporting** or **exploratory**, and the confirmatory row is
never pooled with the others.

| # | Headline claim | Label | Protocol | Script | Artifact (number) | Figure |
|---|----------------|-------|----------|--------|-------------------|--------|
| 1 | **Fixed-κ learned-`m_s` differentiator — the headline result.** With κ held to ±0.04 (reduce-κ lever disabled) so elongation cannot do the work, the learned-`m_s` gradient drives the *secondary* levers (squareness, gaps, internal inductance) and beats the best **realisable** single fixed physics lever: **25/28, two-sided Wilcoxon p = 2.4×10⁻⁵** (sign p = 2.7×10⁻⁵; win-rate 0.89, Wilson 95 % [0.73, 0.96]; median paired diff +0.218, scale only). The cohort's design, `n`, test and six ordered outcome bands were **frozen in version control before any of its data existed**; `n` fixed in advance, no top-ups, no early stopping, no cohort replacement. Outcome band reached: `R-REPLICATED`. | **confirmatory** | 28 fresh marginal starts (`m_s` ∈ [0.05, 0.4)), disjoint from all 56 prior starts, drawn from a pool of 396; κ ±0.04; budget 18 true solves per arm per start. Pre-registered family of one, so no correction applies within it. | `experiments/s3c_cohort_run.py`, `experiments/s3c_cohort_worker.py`; independent recompute `experiments/verify_s3c_recompute.py` | `data/s3c_third_cohort.json` → `primary` (`n` 28, `wilcoxon_p` 2.3805e-05, `median_delta` 0.21806), `secondary` (`wins_surrogate` 25, `sign_p` 2.744e-05, `wilson95` [0.728, 0.963]); pre-registration `data/s3c_prereg.json` (its `prereg_sha256` is recorded in the cohort file) | `figures/phase4_kappa_differentiator.png` (drawn from the pooled n=56 set, therefore **supporting**, not this row) |
| 2 | **The pre-named lever was still the right comparator.** On this cohort the lever named in advance remained the **best of 8** candidate fixed levers (rank 1 of 8), so the confirmatory comparison was not made easy by picking a weak baseline after the fact. | exploratory | Post-hoc sweep of all 8 levers on the confirmatory cohort; zero new solves. | `experiments/s3c_council_checks.py` | `data/s3c_lever_sweep.json` (`pre_named_rank` 1, `pre_named_was_still_best` true) | — |
| 3 | **Pooled context (not the headline).** Over `n = 56` starts from two disjoint batches: vs best realisable fixed lever **45/56** (p = 4.4×10⁻⁶); vs derivative-free search on the true solver **50/56** (p = 3.5×10⁻⁹); vs the *unrealisable* per-start oracle lever **42/56** (p = 6.5×10⁻⁴). The oracle cell is **exploratory and deliberately not headlined**: at the pre-top-up `n` it was non-significant and a data-dependent `n` increase moved it. | supporting (oracle row: exploratory) | κ ±0.04, budget 18. | `experiments/phase4_power_run.py`, `experiments/phase4_kappa_pooled.py` | `data/phase4_kappa_pooled.json` → `by_regime.pooled` (`vs_best_fixed_lever` 45/56 p 4.381e-06; `vs_best_gradfree` 50/56 p 3.511e-09; `vs_oracle_lever` 42/56 p 6.504e-04) | `figures/phase4_kappa_differentiator.png` |
| 4 | **Solve-efficiency of the design loop.** The gradient design loop reaches `m_s*` = 1.0 in a median **4 vs 13.5** expensive *serial* solves (faster **17/20**, Wilcoxon **p = 0.0009**); robust to a fair-CMA control ⇒ not a search-starvation artifact. *(A Phase-3 result with κ unconstrained. Reported here as efficiency; it is **not** the confirmatory claim — see claim 1.)* | supporting | 20 stratified marginal+mid design tasks, d=12 PCA control, target 1.0, budget 30, **every step 80-mode solver-confirmed**. | `experiments/phase3_lib.py`, `experiments/phase3_analyze.py` | `data/phase3_summary.json` → `amortization.pooled` (`surrogate_median_solves` 4.0, `best_gradfree_median_solves` 13.5, `paired_solves_surrogate_faster` 17, `paired_solves_wilcoxon_p` 8.85e-4) | `figures/phase3_efficiency.png` |
| 5 | **Accuracy-at-speed vs the rigid rank ceiling.** The surrogate reaches held-out **log-R² = 0.971** and its rank agreement (Spearman **0.984**) exceeds the rigid Leuer metric's ceiling (**0.917**). *(Reported as accuracy-at-speed, not as a headline "×" number.)* | supporting | 652 held-out shapes; Spearman(Leuer) vs Spearman(surrogate). | `experiments/phase4_pareto.py`, `experiments/phase4_leuer.py` | `data/phase4_pareto.json` (`r2_log` 0.9714); `data/phase4_leuer.json` (`spearman_leuer_raw` 0.9168, `spearman_surrogate` 0.9845) | `figures/phase4_pareto.png`, `figures/phase4_leuer.png` |
| 6 | **Internal-inductance lever ablation.** The `l_i` lever is separated from the geometric levers, with a sham-selection control and a smoke test, to check the fixed-κ win is not an artifact of one channel. | exploratory | Ablation + sham control. | `experiments/s4a_li_ablation.py`, `experiments/s4a_li_worker.py` | `data/s4a_li_ablation.json`, `data/s4a_sham_selection.json`, `data/s4a_li_smoke.json`, raw `data/s4a_li_raw.json` | — |
| 7 | **Zero-shot transfer to Device-C FAILS.** Applied zero-shot to a higher-aspect-ratio device (Device-C, A ≈ 2.97), the learned `m_s` fails as an absolute model: **log-space R² = −6.4**. Weights are machine-specific and need per-device retraining (~700 solves), after which the retrained model only *ties* reduce-κ. **This is not the manuscript's transfer result** — see "Relationship to the manuscript". | supporting | MAST-U surrogate applied zero-shot to Device-C; 40 starts. | `experiments/device2_surrogate_train.py`, `experiments/device2_design_analyze.py` | `data/phase5_zeroshot_transfer_diag.json` (`log_space_R2` −6.393); `data/device2_surrogate_C_eval.json` | `figures/phase5_design_zeroshot.png` |
| 8 | **Quantified synthetic↔real domain gap.** Real MAST EFIT shapes are **100% out-of-distribution** of the synthetic MAST-U training cloud (convex hull: 0 inside), driven by an aspect-ratio offset plus an independent triangularity mismatch. | supporting | Geometric κ/δ/aspect coverage, identical code both sides, no re-solve; 357 real slices vs 3254 training. | `experiments/tier1_coverage.py` | `data/tier1_coverage.json` (`ood_frac_99` 1.0, `convex_hull_ood` 1.0) | `figures/tier1_coverage.png` |
| 9 | **Uncertainty-gated abstention flag.** On the real (OOD) shapes the surrogate's epistemic uncertainty is elevated ~**25×** vs its in-distribution baseline (**computed**: 0.7145 ÷ 0.0281 = 25.4 → 25× at 2 s.f.). A binary domain-shift flag; it does not rank OOD severity. | exploratory | Ensemble epistemic std on real slices ÷ the in-dist baseline median. | `experiments/tier1_analyze.py`, `experiments/tier1_epistemic_baseline.py` | `data/tier1_analysis.json` (`consistency.median_epi_std` 0.7145, `indist_epistemic_median` 0.02811; ratio computed at build time = 25.42); `data/tier1_epistemic_baseline.json` | `figures/tier1_consistency.png` |

---

## Scope & honest limitations (each with its artifact)

Omitting these would misrepresent the work; they are load-bearing.

- **Absolute margins are not recoverable and are not reported.** They are
  simulator-configuration dependent and are **not** recovered by refining the vessel
  basis. Every claim here is relative and ordinal.
- **Cross-device transfer fails.** log-space R² = −6.4 zero-shot on Device-C; the weights
  are machine-specific (`data/phase5_zeroshot_transfer_diag.json`). The manuscript reports
  a stronger and more adverse test on a conventional-aspect (DIII-D-geometry) equilibrium
  whose artifacts are held until PREREG v2 (see above).
- **Surrogate-only optimization goes off-manifold; confirm-in-loop is essential.** The
  raw single-step gradient ascends the true landscape only ~40% of the time pooled (~55%
  in the design band); it is the solver **confirm-and-reject loop** that delivers the
  wins (`data/phase2_gradcheck2_80.json`; the loop still rejects ~1-in-4 marginal/mid
  steps as off-manifold, `data/phase3_summary.json` → `reject_mix`).
- **Unconstrained-κ tie.** With κ *unconstrained*, κ-geometry dominates and the learned
  `m_s` differentiator ties the reduce-κ heuristic: on the single ST manifold surrogate
  vs heuristic is within noise (`data/phase3_summary.json`), and on Device-C the
  retrained model ties reduce-κ (pooled 21/40 = 52.5%, Wilcoxon p=0.92,
  `data/device2_design_retrained_summary.json` → `ALL.POOLED`). **The fixed-κ result
  (claim 1) is the load-bearing one, and it is scoped to marginal starts against one
  named lever.**
- **Generality across structurally distinct devices is NOT established.** Device-C is a
  documented higher-aspect-ratio *radial transform* of MAST-U and shares its passive/coil
  topology by construction, so these results bound κ-dominance as **robust-to-aspect-ratio,
  not general** (`data/device2_design_retrained_summary.json`; `SCOPING_vertical_stability.md`).
- **Single profile family.** The surrogate is `shape → m_s` at a **fixed profile family**;
  profile-family generalization is untested (`DATASET.md`, `data/dataset_v1_80q.parquet`).
- **Synthetic and simulator-internal by declared scope.** Nothing here is validated
  against measured vertical-stability data. The real-MAST artifacts (claims 8–9) quantify
  a *domain gap*; they are not a validation.
- **Marginal-band grid systematic ~10%.** Re-solving deep-marginal endpoints at 129²
  shifts `m_s` by a median ~9.5% (`data/phase4_grid_check.json` → `overall_median_abs_pct`);
  treat single deep-marginal `m_s` values at 65² as grid-sensitive.
- **Reproducibility is single-platform.** Labels are bit-reproducible to 12 digits within
  a fixed BLAS-pinned configuration; **single-platform**. Cross-platform (Mac vs Windows)
  `m_s` agrees to Δ = −0.0015% (`data/phase5_A4_mac_crossplatform.json`).
  *(Errata covering every document in this snapshot: all "bit-reproducible" statements refer
  to this single-platform, BLAS-pinned sense; cross-platform agreement is ~0.0015%, not exact.)*

---

## Phase-numbering key

| Phase | What | Code / data prefix |
|---|---|---|
| 0 | Label-trust & locked numerical protocol | `phase0_*` |
| 1 | Gradient de-risk (feasibility) | `phase1_*`, `11_*`, `12_*`, `14_*` |
| 1.5 | Forward-sampling data engine → `dataset_v1` | `phase15_*` |
| 2 / 2.5 / 2.5b | Surrogate, calibration, dimensionality, κ-headline, q95, 80-mode gradient | `phase2_*`, `phase25_*`, `phase25b_*` |
| 3 | Solver-confirmed differentiable **design loop** | `phase3_*` |
| 4 | Rigor layer + fixed-κ differentiator (pooled / exploratory cohorts) | `phase4_*` |
| 5 | **Device-C** (documented higher-aspect-ratio radial transform of MAST-U): unconstrained learned-`m_s` vs reduce-κ + zero-shot transfer test | **`device2_*` scripts/data/figures = Phase-5 Device-C**; `phase5_*` data |
| S3c / S4a | **The pre-registered confirmatory cohort** (`s3c_*`) and the internal-inductance ablation (`s4a_*`) | `s3c_*`, `s4a_*`, `s34_preflight.py` |

**Note:** in this repository, all files named `device2_*` implement **Phase 5 / Device-C**.

---

## Environment (locked)

- venv: Python **3.12.10**; **FreeGS 0.8.2**, **FreeGSNKE 3.0.1**, numpy 1.26.4,
  scipy 1.15.3, pandas 3.0.3, pyarrow 24.0.0, torch, shapely (`requirements.txt`).
- **Always pin BLAS threads identically**: `OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=
  MKL_NUM_THREADS=NUMEXPR_NUM_THREADS=VECLIB_MAXIMUM_THREADS=1`. Windows console: set
  `PYTHONIOENCODING=utf-8`.
- Machine: the FreeGSNKE `MAST-U_like` tutorial configuration
  (`machine_configs/MAST-U/serialized_tokamak.pkl`; 12 active coils, 138 passive
  structures, aspect ratio 1.638). Full protocol in `PHASE0_PROTOCOL.md`.

---

## Data & provenance

- Canonical training set: `data/dataset_v1_80q.parquet` (3254 converged diverted MAST-U
  equilibria; 80-mode `m_s` labels + q95). Datasheet: `DATASET.md`.
- The Tier-1 real-data artifacts (`data/tier1_coverage.json`, `data/tier1_selection.json`,
  `data/tier1_analysis.json`, `data/tier1_pool.json`, `data/tier1_shotlist.json`,
  `data/tier1_resolved/`) are **derived from the open FAIR-MAST EFIT dataset** and are
  redistributed under **CC-BY-SA-4.0** (share-alike; see License).
- `data/s3c_prereg.json` ships **read-only (mode 0444)** on purpose: it is a
  pre-registration whose SHA-256 is recorded inside `data/s3c_third_cohort.json`. Verify
  it rather than edit it.

### Pointers that do not resolve in this snapshot

Some shipped artifacts carry `design` and `amendment` fields naming paths under
`data/audit/strategy/` and `data/research/council/`, and `experiments/verify_s3c_recompute.py`
writes its report under `data/tracks_ab/`. **Those paths live in the private provenance
archive and are intentionally absent here.** They are left in place rather than scrubbed:
`data/s3c_prereg.json` is hash-committed, and rewriting a pre-registration to tidy its
references would break the exact chain it exists to prove. A dangling pointer is the
honest cost of that choice.

### Provenance note (why this is a fresh-history snapshot)

This public repository is a **fresh-history snapshot**, not the private research repo made
public: the private history's early commits contain unrelated personal documents and
working notes, so publishing history would leak them. The Zenodo DOI plus this snapshot
provide the public timestamp; the private repository remains the provenance archive. All
new pre-registration freezes happen in this public repository going forward. **The Tier-1
pre-registration freeze hash (09df502) is verifiable in the private provenance archive;
publicly verifiable pre-registration begins with PREREG v2.**

`RESULTS.md` and `SCOPING_vertical_stability.md` in this snapshot are the **2026-07-22
vetted public revisions**. The private research log has advanced past them into material
held until PREREG v2, so they are pinned here rather than refreshed. **Use the
reproduction map above, not `RESULTS.md`, as the current claim index.**

---

## License

- **Code** (`experiments/`, `src/`): **MIT** — see `LICENSE` (© 2026 Bryan Nasr).
- **Synthetic data** (`data/dataset_v1*.parquet`, `data/dataset_v2_B.parquet`, and the
  synthetic-run JSON artifacts): **CC-BY-4.0**.
- **FAIR-MAST-EFIT-derived Tier-1 artifacts** (`data/tier1_coverage.json`,
  `data/tier1_selection.json`, `data/tier1_analysis.json`, `data/tier1_pool.json`,
  `data/tier1_shotlist.json`, `data/tier1_resolved/`, `figures/tier1_*.png`):
  **CC-BY-SA-4.0** (share-alike, per FAIR-MAST terms).

---

## Citation

If you use this work, please cite it (see `CITATION.cff`):

```
Nasr, B. (2026). tokamak-ms-surrogate: a differentiable vertical-stability-margin
surrogate for MAST-U (dataset + code). Zenodo. https://doi.org/10.5281/zenodo.21499720
```

### Key references

- A. Portone, *Nucl. Fusion* **45**, 926 (2005). doi:10.1088/0029-5515/45/8/021 — origin of `m_s`.
- D. A. Humphreys et al., *Nucl. Fusion* **49**, 115003 (2009) — the multi-machine inductive form FreeGSNKE evaluates.
- N. C. Amorisco et al., "FreeGSNKE," *Phys. Plasmas* **31**, 042517 (2024). doi:10.1063/5.0188467.
- K. Jackson et al., FreeGSNKE software paper, *SoftwareX* (2024); and the *IEEE Trans. Plasma Sci.* (2025) companion.
- S. Wai, M. D. Boyer, E. Kolemen (Pertnet), *Nucl. Fusion* **62**, 086042 (2022), arXiv:2202.13915 — learned NSTX-U γ; the direct precedent.

---

*Snapshot generated by a manifest-driven builder (enumerated include patterns,
default-exclude) maintained in the private provenance archive. File inventory:
`PUBLIC_MANIFEST.md`.*
