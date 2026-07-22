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

---

## Reproduction map (claim → protocol → script → artifact → figure)

Every number below is read from the cited JSON artifact (all are present in this
snapshot). Statistics are two-sided; `n` fixed in advance per family.

| # | Headline claim | Protocol | Script | Artifact (number) | Figure |
|---|----------------|----------|--------|-------------------|--------|
| 1 | **~4× solve-efficiency.** The gradient design loop reaches `m_s*`=1.0 in a median **4 vs 13.5** expensive *serial* solves (faster **17/20**, Wilcoxon **p=0.0009**); robust to a fair-CMA control ⇒ not a search-starvation artifact. | 20 stratified marginal+mid design tasks, d=12 PCA control, target 1.0, budget 30, **every step 80-mode solver-confirmed**. | `experiments/phase3_lib.py`, `experiments/phase3_analyze.py` | `data/phase3_summary.json` → `amortization.pooled` (`surrogate_median_solves` 4.0, `best_gradfree_median_solves` 13.5, `paired_solves_surrogate_faster` 17, `paired_solves_wilcoxon_p` 8.85e-4) | `figures/phase3_efficiency.png` |
| 2 | **Fixed-κ learned-`m_s` differentiator.** At κ held ±0.04 (reduce-κ lever disabled) the learned-`m_s` gradient beats every *realizable* baseline. **Confirmatory statistic = the pre-registered out-of-sample replication: 16/20 (Wilcoxon p=0.0017).** Supporting (data-dependent top-ups disclosed, no α-spending): pooled n=56 vs best fixed lever 45/56, vs gradient-free 50/56 (both p≤1e-5). | κ ±0.04, budget 18; replication cohort = fresh 20 marginal starts, out of sample. | `experiments/phase4_power_run.py`, `experiments/phase4_kappa_pooled.py` | `data/phase4_kappa_pooled.json` → `marginal_replication` (n 20, wins 16, `wilcoxon_p` 1.69e-3, `sign_p` 1.18e-2); `by_regime.pooled` (`vs_best_fixed_lever` 45/56, `vs_best_gradfree` 50/56) | `figures/phase4_kappa_differentiator.png` |
| 3 | **Accuracy-at-speed vs the rigid rank ceiling.** The surrogate reaches held-out **log-R² = 0.971** and its rank agreement (Spearman **0.984**) exceeds the rigid Leuer metric's ceiling (**0.917**). *(Reported as accuracy-at-speed, not as a headline "×" number.)* | 652 held-out shapes; Spearman(Leuer) vs Spearman(surrogate). | `experiments/phase4_pareto.py`, `experiments/phase4_leuer.py` | `data/phase4_pareto.json` (`r2_log` 0.9714); `data/phase4_leuer.json` (`spearman_leuer_raw` 0.9168, `spearman_surrogate` 0.9845) | `figures/phase4_pareto.png`, `figures/phase4_leuer.png` |
| 4 | **Quantified synthetic↔real domain gap.** Real MAST EFIT shapes are **100% out-of-distribution** of the synthetic MAST-U training cloud (convex hull: 0 inside), driven by an aspect-ratio offset plus an independent triangularity mismatch. | Geometric κ/δ/aspect coverage, identical code both sides, no re-solve; 357 real slices vs 3254 training. | `experiments/tier1_coverage.py` | `data/tier1_coverage.json` (`ood_frac_99` 1.0, `convex_hull_ood` 1.0) | `figures/tier1_coverage.png` |
| 5 | **Uncertainty-gated abstention flag.** On the real (OOD) shapes the surrogate's epistemic uncertainty is elevated ~**25×** vs its in-distribution baseline (**computed**: 0.7145 ÷ 0.0281 = 25.4 → 25× at 2 s.f.). A binary domain-shift flag (does not rank OOD severity; `confirmatory:false`). | Ensemble epistemic std on real slices ÷ the in-dist baseline median. | `experiments/tier1_analyze.py`, `experiments/tier1_epistemic_baseline.py` | `data/tier1_analysis.json` (`consistency.median_epi_std` 0.7145, `indist_epistemic_median` 0.02811; their ratio, computed at build time, = 25.42); `data/tier1_epistemic_baseline.json` | `figures/tier1_consistency.png` |
| 6 | **Zero-shot transfer FAILS.** Applied zero-shot to a higher-aspect-ratio device (Device-C, A≈2.97), the learned `m_s` fails as an absolute model: **log-space R² = −6.4**. Weights are machine-specific and need per-device retraining (~700 solves), after which the retrained model only *ties* reduce-κ (see limitations). | MAST-U surrogate applied zero-shot to Device-C; 40 starts. | `experiments/device2_surrogate_train.py`, `experiments/device2_design_analyze.py` | `data/phase5_zeroshot_transfer_diag.json` (`log_space_R2` −6.393); `data/device2_surrogate_C_eval.json` | `figures/phase5_design_zeroshot.png` |

---

## Scope & honest limitations (each with its artifact)

Omitting these would misrepresent the work; they are load-bearing.

- **Zero-shot cross-device transfer FAILS.** log-space R² = −6.4 on Device-C; the
  weights are machine-specific (`data/phase5_zeroshot_transfer_diag.json`).
- **Surrogate-only optimization goes off-manifold; confirm-in-loop is essential.** The
  raw single-step gradient ascends the true landscape only ~40% of the time pooled (~55%
  in the design band); it is the solver **confirm-and-reject loop** that delivers the
  wins (`data/phase2_gradcheck2_80.json`; the loop still rejects ~1-in-4 marginal/mid
  steps as off-manifold, `data/phase3_summary.json` → `reject_mix`).
- **Unconstrained-κ tie.** With κ *unconstrained*, κ-geometry dominates and the learned
  `m_s` differentiator ties the reduce-κ heuristic: on the single ST manifold surrogate
  vs heuristic is within noise (`data/phase3_summary.json`), and on Device-C the
  retrained model ties reduce-κ (pooled 21/40 = 52.5%, Wilcoxon p=0.92,
  `data/device2_design_retrained_summary.json` → `ALL.POOLED`). The fixed-κ result
  (claim 2) is the load-bearing one.
- **Generality across structurally distinct devices is NOT established.** Device-C is a
  documented higher-aspect-ratio *radial transform* of MAST-U and shares its passive/coil
  topology by construction, so these results bound κ-dominance as **robust-to-aspect-ratio,
  not general** (`data/device2_design_retrained_summary.json`; `SCOPING_vertical_stability.md`).
- **Single profile family.** The surrogate is `shape → m_s` at a **fixed profile family**;
  profile-family generalization is untested (`DATASET.md`, `data/dataset_v1_80q.parquet`).
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
| 3 | Solver-confirmed differentiable **design loop** (the contribution) | `phase3_*` |
| 4 | Rigor layer + **fixed-κ learned-`m_s` differentiator** | `phase4_*` |
| 5 | **Device-C** (documented higher-aspect-ratio radial transform of MAST-U): unconstrained learned-`m_s` vs reduce-κ + zero-shot transfer test | **`device2_*` scripts/data/figures = Phase-5 Device-C**; `phase5_*` data |

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

### Provenance note (why this is a fresh-history snapshot)

This public repository is a **fresh-history snapshot**, not the private research repo made
public: the private history's early commits contain unrelated personal documents and
working notes, so publishing history would leak them. The Zenodo DOI plus this snapshot
provide the public timestamp; the private repository remains the provenance archive. All
new pre-registration freezes happen in this public repository going forward. **The Tier-1
pre-registration freeze hash (09df502) is verifiable in the private provenance archive;
publicly verifiable pre-registration begins with PREREG v2.**

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

If you use this work, please cite it (see `CITATION.cff`). A DOI will be minted at release
and this block updated:

```
Nasr, B. (2026). tokamak-ms-surrogate: a differentiable vertical-stability-margin
surrogate for MAST-U (dataset + code). Zenodo. https://doi.org/PENDING
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
