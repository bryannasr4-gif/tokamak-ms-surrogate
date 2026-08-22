# RESULTS — Living Project Ledger

**This is the single source of truth for project progress.** It holds (A) the consolidated, *verified* facts any
future session needs, and (B) a chronological ledger where **every prompt/phase appends its results plus explicit
advice for the next prompt.** Read Part A + the latest ledger entry before doing anything; append a new ledger
entry when you finish.

> **THE RULE (for every future session):** When you complete a phase, append a ledger
> entry to Part B using the template at the end. Record what you did, the numbers (with the protocol used),
> artifacts, caveats, and a concrete **"Advice for the next prompt."** Keep claims honest (see the project's global claims-discipline rules). Update Part A only for *verified* reusable facts.

Canonical companion docs: `SCOPING_vertical_stability.md` (contribution/positioning), `PHASE1_RESULTS.md`,
`AUDIT_2026-06-18.md` (verified facts + corrections), `DATASET.md`, and this `RESULTS.md`.

---

## TL;DR — current state (2026-06-26)
**PHASE 5 (Device-C = a documented higher-aspect-ratio RADIAL TRANSFORM of MAST-U, run on the Mac) DONE — outcome = honest TIE / κ-robustness scoping result (no pre-registered WIN).**
On **Device-C** (a documented higher-aspect-ratio radial transform of MAST-U, plasma A≈2.97; shares MAST-U's exact passive/coil topology + Z-extent by construction), UNCONSTRAINED: a Device-C-**retrained** learned m_s does **not** beat the reduce-κ heuristic (regime crossover — mid edge 72% p=0.06 *replicated-direction*, marginal favored reduce-κ & not replicated; pooled 52% p=0.92 is a cancellation artifact). **Zero-shot** transfer LOSES (22%, p=0.0002): the learned m_s is not a transferable absolute model (log-R²=−6.4; the loss is gradient-DIRECTION degradation, cosine 0.63, not the loop-invariant 4× miscalibration). The differentiable-loop **mechanism** (not the learned model) beats gradient-free at matched budget (surrogate 75%, *and* the non-learned reduce-κ gradient 78%, both p<1e-4). Kill-gate corr(κ,log m_s)=−0.811 (primary |corr|<0.75 NOT met; GO via the secondary rival-lever rule only). All labels/endpoints Portone-cross-checked **0.000%**; 24-agent Phase-D review (19→15 survivors) incorporated. Net: **κ-dominance is robust to aspect ratio (not a pure-ST artifact) but NOT shown general across distinct designs**; the delivered contribution remains the FIXED-κ differentiator (Phase 4) + the amortized solver-confirmed loop. Full entry in Part B; cross-platform Mac-vs-Windows m_s Δ=−0.0015%.

## TL;DR — prior state (2026-06-24)
Direction = **GO (reframed)**: a *learned, amortized, differentiable* surrogate for the Portone-2005 **stability
margin m_s** (and γ) of a **spherical tokamak** (MAST-U), whose gradient is **used** to drive **solver-confirmed**
shape design, on **open** FreeGSNKE data. Scoping + feasibility + Phases 1 + 0 + 1.5 + **2** are **done**. On
`data/dataset_v1.parquet` (3298 converged diverted MAST-U equilibria, m_s 0.001–15.4) a heteroscedastic deep-ensemble
surrogate m_s(shape) gets held-out **R²=0.954** (log 0.968) and **clearly beats a linear baseline out-of-distribution**
(corner log-R² 0.71 vs 0.36; vs a *fair* GP 0.68 the edge is within noise, p≈0.09) — though the corner aggregate is
mid-regime-carried (marginal corner ≈ mean predictor, log-R²≈0.04); calibration is honest (post-hoc s=1.86 →
cov90≈0.97; abstains at the m_s→0 boundary + sparse tail); the autodiff gradient gives a **usable net ascent direction
in the design regime** (true-m_s ascent solver-confirmed in **15/20 marginal+mid bases at 40 modes**, Wilson 0.53–0.89;
**re-verified at the converged 80 modes it WEAKENS to 11/20 = 55%, Wilson 0.34–0.74 — CI spans chance, Phase 2.5b**) though it
is **not a faithful Jacobian** (true-solver cosine ~0.6) *(SUPERSEDED 2026-07-21 — see A7/D3-U1: the ~0.6 was ≈⅔ one-sided-FD artifact; the deployed direction is validated 25/25 on the true landscape)* and rejects ~25–45% of marginal/mid steps off-manifold.
**Mode convergence RESOLVED + RE-LABELED:** the 40-mode labels were low by median −14%; m_s is **converged by 80
modes** and `dataset_v1_80.parquet` (3254 shapes) is the clean re-labeled set. **Dimensionality (honest):** in PCA
control space (**effective dim 5.5/16**) gradient-based design **beats gradient-free search and the gap GROWS with
dimension** (reaches target in ~5 solves at d=12 vs CMA/random stuck). **Phase 2.5b — κ-constrained headline FIRMED
(n=8 → n=20 stratified, 10 marginal + 10 mid, budget 18, κ held ±0.04 on the TRUE-solver κ):** the surrogate gradient
**robustly beats gradient-free search (17/20 = 85%, two-sided Wilcoxon p=1e-4; significant in both regimes)**; its edge
over the **best fair single-lever heuristic is POSITIVE BUT WITHIN NOISE (15/20 = 75%, Wilcoxon p=0.064; n.s. within
both regimes)** — the earlier n=8 "7/8 = 88%, ≈2.7×" headline did **NOT survive** honest marginal coverage + two-sided
tests (the real ratios are 1.8× vs gradient-free, 1.18× vs the heuristic). A modified-passive "Machine B" shows m_s
rescales globally with the conducting structure (true-label affine Spearman 0.994) — a calibration result, NOT
cross-device generalisation (a real 2nd device is owed). Gate stays **PARTIAL**. **Phase 2.5b also added** q95
(dataset_v1_80q; physically sane, but κ-mediated) and a TRUE high-κ extrapolation split (modest mid-regime penalty,
no collapse). **PHASE 3 DONE — the solver-confirmed differentiable design loop (the contribution).** 20 stratified
marginal+mid design tasks (d=12 PCA control, target m\*=1.0, budget 30, **80-mode** solver-confirmed every step, shape
descriptors held in range): the gradient design loop **stabilizes marginal ST plasmas (m_s≈0.3→1.1) in a median of 6
true solves and reaches m\*=1.0 in ~4× FEWER expensive solves than gradient-free search** (pooled median 4 vs 13.5;
faster 17/20, two-sided paired Wilcoxon **p=0.0009**). A **fair-CMA control** (popsize 6 + graded penalty + 2× budget)
settles the headline honestly: gradient-free *can* reach the same 9/10 marginal targets given 2× budget, but still needs
**median 26 vs 6 solves (~4×)** — so the contribution is **solve-EFFICIENCY (amortization), not unique capability**, and
the **~4× gap is robust, not a CMA-starvation artifact**. **Honest scope (all from a 38-agent adversarial panel, 24
survivors, 0 high-sev):** the efficiency win is the **GRADIENT/κ-direction, NOT the learned m_s** (the reduce-κ heuristic
also beats gradient-free 16/20 p=0.0004; surrogate-vs-heuristic within noise 6/20 p=0.23 — κ-dominated single machine);
the per-query edge is **serial-only** (within noise under parallel CMA-population eval, 5/20); final-margin is confounded
(overshoot) so solves-to-target/reach are the metrics; amortization breaks even at **~340 queries** (3254-solve training
set). Gate **PASS (honestly scoped)**; every design 80-mode solver-confirmed; 1/20 surrogate failure disclosed.
**PHASE 4 DONE — the LEARNED m_s is now load-bearing AT FIXED κ + the full rigor layer.** With κ held ±0.04 (the
reduce-κ lever DISABLED), the learned-m_s gradient (driving SECONDARY levers: squareness/gaps/l_i) beats **every
baseline** at the firmed **n=56** (marginal n=40, two disjoint top-up batches): gradient-free search **50/56=89%**,
the best realizable single FIXED physics lever **45/56=80%**, AND even the *unrealizable* per-start ORACLE best-of-8
**42/56=75% (all Wilcoxon p≤0.0007)**; the marginal cell is significant under both tests (30/40, sign p=0.002) and
replicated out-of-sample (fresh 20-marginal cohort 16/20=80%). A solver-confirmed **gallery** (κ verified locked 20/20, max drift 0.0399) decomposes the
median +0.548 gain into a residual-κ-tolerance confound (+0.181) vs a **learned-m_s secondary-lever EXCESS of +0.386
(≈70%)**; a marginal ST plasma is stabilized **m_s 0.19→1.03 at fixed κ**. Rigor: surrogate m_s+gradient **~8,100×
faster** than a full 80-mode Jacobian solve (linearisation=78% of the 22.7 s solve) at held-out **log-R²=0.971**;
the rigid **Leuer** rank ceiling (Spearman 0.92) is lifted to 0.98 (most in very-stable 0.22→0.83, a tie in marginal);
shape-parameterization ablation κ-only RMSE_log **0.442→0.122** full (3.6×, squareness the key lever); robust to 5%
input noise (grad cos 1.000); most of the accuracy is reached by ~650 shapes (small residual gains continue to the
full set: RMSE_log 0.130 at n=650 → 0.121 at n=2602). A **38-agent adversarial panel (32 raised, 3
survived, 0 high/med)** found only a labeling nit (fixed). Gate **PASS (honestly scoped)**. Standing open caveats:
the learned-m_s win is **at FIXED κ** — on the UNCONSTRAINED single-ST manifold κ-geometry still ties it (unchanged);
a **genuine different-aspect-ratio 2nd device** (FreeGS DIIID/TCV lack the FreeGSNKE passive structures the Portone
m_s needs) remains the cleanest owed differentiator. **Next = Phase 5** (reproducible open release) **or the genuine
2nd device**. (Power top-up DONE → pooled **n=56**, marginal n=40: surrogate beats fixed-lever 45/56, gradient-free
50/56, AND the oracle 42/56 — all p≤0.0007; marginal significant under both tests + replicated out-of-sample.) A
50-agent QUALITY audit found the genuine 2nd device is the thesis-closer and is **tractable** (FreeGSNKE builds
passive structures programmatically). Other standing caveats: single machine/wall; raw single-step 80-mode gradient is
weak (the LOOP, with confirm+reject, is what works).

---

# PART A — Consolidated verified foundation (reusable facts; don't re-derive)

### A1. Environment & toolchain
- venv: `./fusion-env/Scripts/python.exe` (Python 3.12.10). Installed & working: **FreeGS 0.8.2, FreeGSNKE 3.0.1**,
  torch, numpy 1.26.4, **scipy 1.15.3** (corrected — earlier "1.11.4" was wrong), sbi, zuko, matplotlib, shapely, h5py,
  **pandas 3.0.3, pyarrow 24.0.0** (installed Phase 1.5).
- Always pin BLAS threads identically: `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1` (see A4).
- Windows console is cp1252 → set `PYTHONIOENCODING=utf-8` or avoid non-ASCII in prints (Δ, κ crash `print`).

### A2. Machine & the γ / m_s API (verified)
- Machine: genuine FreeGSNKE **`MAST-U_like`** tutorial files at `machine_configs/MAST-U/*.pickle` —
  **12 active coils** (Solenoid, PX, D1, D2, D3, Dp, D5, D6, D7, P4, P5, P6 — **note: no D4**; Dp occupies
  that slot, so "D1–D7" notation is wrong) + **138 passive structures**, **aspect ratio 1.638**
  (true spherical tokamak). Open/reproducible.
- To get the vertical growth rate γ and stability margin m_s from a converged equilibrium `eq, profiles`:
  ```
  nls = freegsnke.nonlinear_solve.nl_solver(eq=eq, profiles=profiles, GSStaticSolver=solver,
                                            plasma_resistivity=1e-6, fix_n_vessel_modes=40, verbose=False)
  gamma = nls.linearised_sol.growth_rates        # [1/s], list (unstable modes)
  m_s   = nls.linearised_sol.stability_margin    # Portone-2005 largest positive eigenvalue
  leuer = nls.Leuer_metals_stab_over_active_destab
  ```
  Internally builds a finite-difference Jacobian dIy/dI (~50 perturbed GS solves) then a lumped-circuit
  eigenvalue. `m_s` implements Portone 2005 (`freegsnke/linear_solve.py:calculate_stability_margin` =
  `eig(L⁻¹S − I)`, `L=M0matrix[:n,:n]`, `S=−dMmatrix[:n,:n]`; take `max` of the positive eigenvalues).
- **m_s CONVENTION (Phase-0 verified; see `PHASE0_PROTOCOL.md` §2):** m_s = Portone-2005 *inductive* stability
  margin, dimensionless + **resistivity-independent**. **m_s>0 = stable (passively/resistively unstable,
  controllable); m_s=0 = ideal-MHD marginal; m_s<0 = Alfvénic/uncontrollable. BIGGER m_s = MORE stable; m_s→0 =
  controllability boundary.** Practical thresholds: m_s>0.15 (FUSE design), ~0.26 (TCV/C-Mod control). **The
  FreeGSNKE inline comment `# the positive (i.e. unstable) eigenvalues` is a copy-paste MISLABEL — the code is
  correct, do NOT quote that comment.** Independent recompute matches FreeGSNKE to **0.000%** (3 ways).
- **Serialized machine:** build once, reuse `phase0_lib.load_machine("machine_configs/MAST-U/serialized_tokamak.pkl")`
  (0.7 MB, ~1.1 s load; bit-for-bit reproducible across processes). Use this everywhere downstream.
- Inverse solve to a target shape: `freegsnke.inverse.Inverse_optimizer` + `GSstaticsolver.NKGSsolver.solve`.
  **NOTE:** the inverse (shape→coil-current) solve is ILL-CONDITIONED and is the source of the noise floor (A4).

### A3. Key numbers (always state the protocol they came from)
- **Baseline diverted equilibrium** (65×129 grid, single build): γ ≈ **260 /s**, m_s ≈ **0.39**, κ=2.055, δ=0.52,
  Ip=600 kA. Inverse solve 13.5 s + linearization 32.8 s (3051×53 Jacobian) ≈ **46 s/label**.
- At 65×65 (Phase-1 protocol): Jacobian 1513×53, ≈ **45–78 s/label**.
- **Signal (smooth, monotone, learnable):** elongation κ 1.65→2.05 ⇒ γ 2.3→330 /s, m_s 3.9→0.34 (toward the
  m_s→0 controllability boundary). **m_s is smoother / more reproducible than γ → prefer m_s as the target.** (elongation scan; data/12_elongation_scan.json)
- **Phase-1 (64-pt grid, elongation × wall-gap, 65×65, OMP=1):** surrogate m_s held-out **R²=0.93**; gradient
  test **cosine 0.956, 100% sign, magnitude ratio 1.43**; clean closed loop: start m_s **0.306** → gradient step
  **0.431 (+0.125, +41%)** > analytic-κ 0.424 > fd 0.414.

### A4. THE LOCKED NUMERICAL PROTOCOL + NOISE FLOOR (Phase-0 final; full detail in `PHASE0_PROTOCOL.md`)
**LOCKED PROTOCOL (use for every label):** `OMP_NUM_THREADS=OPENBLAS=MKL=NUMEXPR=VECLIB=1`; ONE serialized
machine (`serialized_tokamak.pkl`); **isolated COLD solves** (fresh eq + reset coils per shape — no warm-start);
grid **65×65**; **fix_n_vessel_modes=40**. Reference impl: `experiments/phase0_lib.solve_equilibrium`.
> **Which solve path / tolerance (read carefully):** the Phase-0/Phase-1 INVERSE pipeline (shape→coil
> currents) used `inverse target_relative_tolerance=1e-6` (~46–90 s/label) and its residual error was the
> "ill-conditioned inverse map" story below. The **canonical `dataset_v1` labels (Phase 1.5) are made by the
> FORWARD path** (`phase15_lib.forward_label`, fixed coils, no inverse solve) at **forward GS tol 1e-8,
> acceptance rel ≤ 1e-7** — this is what removes the inverse-map noise. Both tolerances are correct; just
> attach each to its path. The "inverse 1e-6 / ill-conditioned inverse" wording below describes the
> superseded inverse pipeline, not how dataset_v1 was labelled.
- **Within the locked config there is NO random label noise:** m_s is bit-reproducible to **12 digits**
  cross-process (verified). The audit's "~10%" was **cross-BLAS-thread** spread (m_s median **4.9%**, max **11.4%**;
  γ median 4.7%, max **19%**), **worst near the marginal boundary** (m_s→0) — and it is **eliminated by OMP=1**.
- **Warm-start path dependence ~2%** (leftover coil currents seed the next inverse solve) — **eliminated by cold solves**.
- **Residual SYSTEMATIC protocol-dependence of the absolute m_s** (a fixed bias, NOT label noise): retained-mode
  count is **dominant** (`modes=20` unusable; **40→80 ≈ +10%** → m_s NOT mode-converged at 40), grid ≤7%
  (129×129 lower), inverse tol ≤10% beyond 1e-6 (but 1e-6 is corroborated by the 129×129 value; tighter tol
  *chases* the ill-conditioned inverse). **Do not present any effect below the relevant floor as real.**
- **MODE CONVERGENCE — RESOLVED (Phase 2, 60-shape study at modes 40/80/120/138; 138 = all passive structures):**
  the converged count is **80 modes** (|m_s(80)−m_s(138)| median **0.4%**, max 3.4%; 120≈138 exactly). The
  **40-mode labels (dataset_v1) are systematically LOW by median −13.5%**, regime-dependent: **marginal −27%
  (×1.36), mid −15%, stable −13%, very_stable −9%** (worst near m_s→0). This is a fixed bias that **cancels in
  every relative/gradient comparison at fixed modes**; re-label dataset_v1 at 80 before publishing absolute m_s.
- **Machine-build gotcha (unchanged):** `refine_passive.py:26` module-level `LatinHypercube(seed=42)` advances per
  call → 2 builds in 1 process diverge ~20%; the serialized single build sidesteps this entirely.
- **Root-cause fix = Phase 1.5 FORWARD sampling** of coil currents + profile params (well-conditioned; no inverse
  solve). **Phase 2:** mode-convergence study (40/80/160) → finalize a converged mode count for published labels.

### A5. Novelty + verified citations (all checked against the live web — no hallucinations)
- **The ONLY defensible novelty = the conjunction:** learned + amortized + **differentiable, gradient USED for
  solver-confirmed shape design** + open synthetic data + **Portone m_s** + spherical tokamak. **Never claim**
  "first ML growth-rate surrogate" (Pertnet) or "first differentiable γ" (FGE).
- Pertnet — Wai, Boyer, Kolemen, *Nucl. Fusion* 62 (2022) 086042, arXiv:2202.13915 (learned NSTX-U γ; the direct
  precedent; derivative-free, monitored γ only).
- Portone — *Nucl. Fusion* 45 (2005) 926, doi:10.1088/0029-5515/45/8/021 (origin of m_s). **Humphreys et al. —
  *Nucl. Fusion* 49 (2009) 115003, eq. (4)** = the multi-machine inductive form FreeGSNKE actually evaluates
  (`m_s := λ[−M_mm⁻¹(M_mm + M_my ∂Iy/∂Im)]`). Corroborating (Phase-0 verified): **FUSE arXiv:2409.05894**
  (verbatim sign convention + m_s>0.15 design margin), **Olofsson OSTI-1960105** (resistivity-independence),
  VacuumFields.jl `src/mutual.jl`, Isernia & Villone *PPCF* (2023) doi:10.1088/1361-6587/acf15c. Portone 2005 full
  text is paywalled (abstract verified); anchor citations on Humphreys 2009 + FreeGSNKE example10. FreeGSNKE/Amorisco
  — *Phys. Plasmas* 31 (2024) 042517, doi:10.1063/5.0188467.
- Pentland virtual circuits — arXiv:2604.00781 / 2605.14939 (MAST-U, FreeGSNKE, NN shape control; **omits m_s** =
  the hole we fill). GSPulse — Wai…Kolemen, arXiv:2506.21760 (gradient-FREE; NSTX-U+MAST-U; open-source).
- FGE — arXiv:2512.06847 / *PPCF* ae56b7 (**first fully differentiable** free-boundary evolution *physics solver*;
  not a learned amortized surrogate). Sammuli — *Fusion Eng. Des.* 169 (2021). Liu EAST-MLP — *IEEE TPS* (SOFE)
  2023. DECAF — *PPCF* 2024, doi:10.1088/1361-6587/ad7531 (verify first author, Sabbagh group). Olofsson — *PPCF*
  64 (2022) 075016.

### A6. File map
- Docs: `SCOPING_vertical_stability.md`, `PHASE1_RESULTS.md`, `AUDIT_2026-06-18.md`, **`PHASE0_PROTOCOL.md`**,
  **`DATASET.md`** (Phase-1.5 datasheet), and this `RESULTS.md`.
- **Phase-4 (rigor + the LEARNED-m_s differentiator):** `experiments/phase4_kappa_reanalyze.py` (realizable-baseline
  reframe of the n=20 κ-constrained run), `phase4_gallery_{lib,run,worker,analyze,shapes}.py` (κ-constrained design
  loop with full recording + `kappa_nudge` confound control + LCFS extraction), `phase4_pareto.py` (cost/accuracy),
  `phase4_leuer.py` (Leuer baseline), `phase4_ablations_{light,train}.py` (ensemble/noise/mode + learning-curve/
  shape-param), `phase4_power_{run,worker}.py` (marginal power top-up), `phase4_figures.py`. Data:
  `data/phase4_{kappa_reframed,gallery_results,gallery_summary,gallery_shapes,pareto,leuer,ablations_light,
  ablations_train}.json`, `phase4_gallery_chunk_*.json`, `phase4_power_{setup,jobs,results}.json`; figures
  `figures/phase4_{kappa_differentiator,gallery,pareto,leuer,ablations}.png`.
- **Phase-3 (solver-confirmed design loop):** `experiments/phase3_lib.py` (instrumented design loop + true_eval
  in-range guard + Recorder + gradient-free baselines), `phase3_{run,worker}.py` (resume-safe atomic pool),
  `phase3_cma_fair.py` (fair-CMA control: `run_cma_fair` popsize + graded penalty), `phase3_{analyze,figures,gallery}.py`.
  Data: `data/phase3_{results,summary,setup,jobs,desc_ranges}.json`, `data/phase3_cmafair_{results,summary}.json`; figures `figures/phase3_{efficiency,gallery}.png`. Reuses `phase2_dim_lib`,
  `phase2_model`, `phase15_lib`, `data/phase2_dim_setup.json`, `data/phase2_models/{surrogate,shapemap}.pt`.
- **Phase-1.5 (forward-sampling dataset):** `experiments/phase15_lib.py` (engine: anchors + sampler +
  crash-proof forward solve + descriptors + labels), `phase15_generate.py` (chunk worker), `phase15_run.py`
  (14-worker launcher), `phase15_dataset.py` (assemble parquet + extrapolation split + coverage fig + datasheet),
  `phase15_probe.py`, `phase15_pilot.py`. Data: **`data/dataset_v1.parquet`** (3298×48), `data/phase15_chunk_*.json`,
  `data/phase15_anchors.pkl`, `data/phase15_split_meta.json`; figure `figures/phase15_coverage.png`.
  Env: pandas 3.0.3 + pyarrow 24.0.0 installed this phase.
- Phase-0 code: `experiments/phase0_lib.py` (canonical protocol engine + serialize/load), `phase0_ms_crosscheck.py`,
  `phase0_serialize_machine.py`, `phase0_verify_load.py`, `phase0_solve_one.py`, `phase0_noise_sweep.py`,
  `phase0_noise_analyze.py`. Phase-0 data: `data/phase0_ms_crosscheck.json`, `phase0_noise_sweep.json`,
  `phase0_noise_summary.json`; `machine_configs/MAST-U/serialized_tokamak.pkl`; figure `figures/phase0_noise_floor.png`.
- Code: `experiments/11_growth_rate_smoke.py` (feasibility), `12_elongation_scan.py`+`13_scan_figure.py` (signal),
  `14_gradient_quality.py`+`15_gradient_quality_figure.py` (gradient precondition), `phase1_generate.py`
  (dataset), `phase1_analyze.py` (surrogate + gradient test + closed loop), `audit_reproducibility.py`,
  `audit_determinism.py`. Reusable FreeGS forward model: `src/fusionsbi/forward.py`.
- Data: `data/11_growth_rate_smoke.json`, `12_elongation_scan.json`, `14_gradient_quality.json`,
  `phase1_chunk_*.json`, `phase1_results.json`.
- Figures: `figures/12_growth_rate_vs_elongation.png`, `14_gradient_quality.png`, `phase1_gradient_validation.png`.

### A7. Accuracy-program certifications (2026-07-19/21) — training engine + gradient, VERIFIED

- **D1-U6 — training-set force-balance certification (2026-07-19): the canonical training set's label
  engine is in the HEALTHY class.** All 3,254 rows of `dataset_v1_80q.parquet` replayed verbatim (65²,
  frozen env): **bit-exact reproduction 3,254/3,254 (all EXACT class; worst diff 6.2e-16)**; classes
  CLEAN 3,208 / GRAY 37 / SICK 9; **the science band (marginal+mid regime classes, 1,842 rows; the
  literal m_s∈[0.15,1) cut is 1,820 rows with the same result) is 100.0% CLEAN — max
  fz0 0.068 mm, min K +22,026 N/m, max |Fz0| 4.53 N** (whole-set max 11.9 N); median fz0 7.5e-5 mm; fit_r2 fails 0; K-uncertain 0.
  The GRAY/SICK tail (46 rows, 1.41%) contains NO marginal or mid state — all stable/very-stable
  (m_s ≥ 1.397, κ ≤ 1.80) — and is adjudicated (council-after,
  unanimous on substance) as fz0 = |F|/K ILL-CONDITIONING as K→0 — healthy forces (≤ 12 N), vanishing
  denominator — NOT a genuine force-imbalance class. Exclusion-sensitivity (unit-6b, 20-draw noise envelope): removing
  the flagged tail is statistically ORDINARY removal (test_extrap dR2 −0.0161 at the 55th percentile of
  the random-removal envelope [−0.0418, +0.0008]); duplication control benign (clean-set dR2 −0.0020);
  clean-val sign test PARITY (A-win 51.9%). **Verdict: banked surrogate + claims UNAFFECTED; rows KEPT
  + flagged; D1 (training-engine side) CLOSED.** Artifacts: `data/accuracy/d1u6_mastu_sweep.json` +
  `d1u6_rows.jsonl` + `d1u6b_battery.json`; adjudication records
  `data/research/council/d1u6_finding/SYNTHESIS.md` + `d1u6b_finding/SYNTHESIS.md`.

- **D3-U1 — gradient-fidelity study (2026-07-21): the surrogate gradient DIRECTION is validated on the
  TRUE landscape; the banked "not a faithful Jacobian (true-solver cosine ~0.6)" is SUPERSEDED** (it
  survives in Part-B Phase-2/2.5 entries + the 2026-06-24 TL;DR as history; this entry is the standing
  fact). n=40 d1u6-CLEAN states; verified two-sided central-FD truth at the label protocol (65², 80
  modes, tol 1e-8) + Richardson; pre-registered decision rules; council-adjudicated 4/4 + independent
  VERIFY-from-raw. **Verdict O-ROUGH:** the infinitesimal Jacobian is NOT FD-extractable at the label
  protocol on 24/30 operating-band states (Gate 0 = 0.80; deterministic — labels bit-reproducible —
  and protocol-scoped: tighter-tol/finer-grid/adjoint extractability UNTESTED, the registered D3-U2
  opener), so a pointwise cosine is not well-defined there. **What the design loop actually consumes
  IS validated, FD-independently: dir_surr+ ascends the true landscape 25/25 = 100% of operating-band
  states at the 0.15-std deployment step (median Δlog m_s +0.375; marginal 12/12, mid 13/13; stable
  7/7 besides), and the anti-direction descends 29/30.** The banked ~0.6 is EXPLAINED (n=30 same-point
  reconciliation): ≈⅔ one-sided-FD artifact (one-sided vs two-sided-Richardson truth cos 0.341) with
  the surrogate share at cos 0.633 ≈ the banked value; mode count exonerated (40-vs-80-mode cos 0.989).
  Where FD did converge (n=6, descriptive only): median cos 0.867, bootstrap CI [−0.07, 0.94]; the one
  outlier (pos 2958, cos −0.82) still ASCENDS under the surrogate direction (+0.41) — finite-step
  direction and pointwise cosine genuinely decouple (the anti-descent failure is a different state,
  pos 1114). Deployment lever (registered D3-U2): the PC-12 deployed direction captures only ~⅓ of the
  available ascent (median +0.143 vs +0.375; cos(full, PC-12) mean 0.36 / median 0.33 over the 30
  operating-band states). **Standing wording: the gradient
  is direction-operationally-sufficient at the deployment step; pointwise/magnitude fidelity is
  unresolved at the label protocol (O-ROUGH). The banked 45/56 design-loop wins rest on the validated
  finite-step direction and are UNTHREATENED.** Artifacts: `data/accuracy/d3u1_grad_fidelity.json` +
  `d3u1_FINDINGS.md` (incl. council addendum) + `d3u1_VERIFY_adjudication.md`; council record
  `data/research/council/d3u1_finding/SYNTHESIS.md`; design frozen 93bb93b.

- Some artifacts cited in this section are internal verification records not included in this snapshot; they are scheduled for a future data release. All claims are quoted verbatim from those verified records.

---

# PART B — Phase ledger (append one entry per completed phase)

### Scoping — GO-with-reframing — 2026-06-18 — DONE
- **Did:** literature-first novelty scoping (structured literature review) + empirical feasibility. Found Pertnet
  (Kolemen's own group) scoops the broad claim; identified the surviving 4-part conjunction (A5).
- **Result:** GO-with-reframing. Defensible novel sentence + positioning table in `SCOPING_vertical_stability.md`.
- **Advice for next prompt:** never pitch "first ML/differentiable ST γ"; lead with the gradient-USED, m_s,
  open-data closed loop; frame to Kolemen as an extension of Pertnet/GSPulse.

### Feasibility + signal — 2026-06-18 — DONE
- **Did:** confirmed FreeGSNKE computes γ/m_s on the ST machine (A2); benchmarked cost (A3); elongation scan
  showed smooth monotone γ(κ)/m_s(κ); fine-step test showed the true-solver gradient is well-defined.
- **Result:** feasibility proven; m_s is the smoother target.
- **Advice for next prompt:** target m_s over γ; expect ~46 s/label; the true gradient exists (greenlit Phase 1).

### Phase 1 — gradient de-risk — 2026-06-18 — PASS (audited)
- **Did:** 64-pt shape grid (elongation × wall-gap), differentiable ensemble surrogate, gradient test vs true
  solver, closed-loop design step. Then a full audit.
- **Result (robust):** gradient accurate (cosine 0.956, sign 100%, R²=0.93); gradient step raises true m_s
  +0.125 (+41%), ~3× the noise floor. **(Retracted:** "+89% / beats heuristics" — cross-config artifact; the 2-D
  margin is within the ~10% noise floor.)
- **Artifacts:** `phase1_generate.py`, `phase1_analyze.py`, `data/phase1_results.json`,
  `figures/phase1_gradient_validation.png`, `PHASE1_RESULTS.md`, `AUDIT_2026-06-18.md`.
- **Advice for next prompt (→ Phase 0/2):** (1) Lock ONE BLAS config + ONE serialized machine build before any
  new data. (2) Switch data gen to FORWARD sampling to kill the ill-conditioned-inverse noise. (3) Quantify +
  report the noise floor; never claim sub-floor effects. (4) The "gradient beats heuristics" payoff must be earned
  in higher-D (Phase 2's dimensionality experiment) — it is currently a hypothesis. (5) Cross-check m_s against an
  independent Portone eigenvalue (Phase 0) before scaling.

### Phase 0 — label trust & protocol lockdown — 2026-06-19 — PASS
- **Goal:** make the m_s/γ label trustworthy + reproducible before scaling data; lock the numerical protocol.
- **Did:** (1) Independent m_s cross-check — extracted the converged linearisation's M0/dM/M blocks and recomputed
  the Portone margin three algebraically-distinct ways (`eig(L⁻¹S−I)`, `−L⁻¹L*`, generalized `eig(S,L)−1`).
  (2) Verified the m_s definition/convention via a multi-source adversarial literature workflow. (3) Serialized the
  MAST-U machine once + verified cross-process bit-identity. (4) Quantified the noise floor: a 64-job isolated-cold-
  solve sweep over BLAS threads {1,2,4,8} × tol {1e-6,1e-8,1e-10} × grid {65²,65×129,129²} × modes {20,40,80} on 10
  shapes (m_s∈[0.37,1.82]). Protocol used: OMP=1, serialized machine, 65×65, modes=40, tol=1e-6, cold solves.
- **Result (numbers, vs the floor):** independent m_s matches FreeGSNKE to **0.000%** (all 3 methods; canonical
  m_s=0.4199404869, exactly **1** positive eigenvalue = the single n=0 mode). Convention (high-confidence, verified):
  **m_s>0 stable, =0 ideal-MHD marginal, bigger=more stable, m_s→0=controllability boundary** (Portone NF45/926;
  Humphreys NF49/115003 eq.4; FUSE; Olofsson). Noise floor: **within-config bit-reproducible (12 digits)**;
  cross-BLAS-thread spread m_s **median 4.9% / max 11.4%** (γ 4.7%/19%), worst near m_s→0, **removed by OMP=1**;
  warm-start ~2% removed by cold solves. Systematic protocol-dependence of absolute m_s: **mode count dominant**
  (modes=20 unusable; 40→80 ≈ **+10%**, NOT converged at 40), grid ≤7%, tol ≤10% beyond 1e-6 (1e-6 corroborated by
  129×129). Machine serialized (0.7 MB, 1.1 s load), two processes → m_s=0.419940486916 identical.
- **Artifacts:** `PHASE0_PROTOCOL.md`; `experiments/phase0_{lib,ms_crosscheck,serialize_machine,verify_load,solve_one,noise_sweep,noise_analyze}.py`;
  `data/phase0_{ms_crosscheck,noise_sweep,noise_summary}.json`; `machine_configs/MAST-U/serialized_tokamak.pkl`;
  `figures/phase0_noise_floor.png`; AUDIT §7 addendum; RESULTS Part A (A1/A2/A4/A5/A6) updated.
- **Caveats / surprises:** (a) FreeGSNKE's `# (i.e. unstable)` comment in `calculate_stability_margin` is a
  copy-paste mislabel — code correct, comment wrong; **never quote it**. (b) m_s is **not mode-converged** at 40
  modes (the largest systematic) — convergence study owed in Phase 2. (c) tightening inverse tol is counterproductive
  (chases the ill-conditioned solve). (d) scipy is 1.15.3, not 1.11.4. (e) Scoping refinement: GSPulse/Pertnet use γ,
  not m_s by name → sharpens the "target m_s" novelty.
- **Gate:** PASS — independent m_s <5% (got 0.000%); noise floor quantified; protocol fixed + serialized.
- **Advice for the next prompt (→ Phase 1.5):** (1) Generate data by **forward sampling** (coil currents + profile
  params) on the **serialized machine** at the locked protocol (OMP=1, cold solves) — this is the root-cause fix for
  the inverse-solve floor; reuse `phase0_lib`. (2) Run a **mode-convergence study** (40/80/160) and pick a converged
  `fix_n_vessel_modes` for the published dataset (m_s drifts ~10%/doubling at 40). (3) Report the floor per
  `PHASE0_PROTOCOL.md` §3.6; put calibrated uncertainty on m_s→0 where the floor is worst. (4) State m_s per §2 and
  never quote the FreeGSNKE code comment.

### Phase 1.5 — forward-sampling data engine — 2026-06-20 — PASS
- **Goal:** generate a clean, well-conditioned dataset that avoids the ill-conditioned inverse solve
  (the Phase-0 noise-floor root cause) by **forward sampling** control inputs.
- **Did:** Built a forward-sampling engine (`phase15_lib.py`): sample 12 active-coil currents +
  profile params {paxis, Ip, fvac, αm, αn} → **forward** free-boundary GS solve (`NKGSsolver.forward_solve`,
  constrain=None; well-conditioned) → keep iff CONVERGED + DIVERTED (`flag_limiter==False`) + not
  wall-touching → shape descriptors + labels. The sampling distribution is seeded by a one-time coarse
  set of **21 inverse solves** over a (zscale,dR) grid (anchors only CENTER the distribution; they are
  NOT labels) + anchor interpolation + log-normal current scale + per-coil jitter + independent profile
  params. Protocol: serialized machine, OMP=1, 65×65, 40 modes, forward tol 1e-8, cold solves. Ran 14
  thread-pinned workers (`phase15_run.py`/`phase15_generate.py`); deterministic per-chunk RNG
  `default_rng([20260619, chunk])`. Assembled with `phase15_dataset.py`.
- **Result (numbers, resolved + vs the floor):**
  - **Forward path cross-checked to the inverse:** a forward solve with the inverse anchor's coil
    currents reproduces the inverse m_s to **0.000%** (0.41994 both ways) — forward labels are
    bit-reproducible at the locked protocol; the inverse-solve noise is removed at the root.
  - **3298 converged diverted samples** (gate ≥3k PASS), **0 dropped in QC**, yield **~89%** (400 rejections,
    all cleanly handled, never crashing: non-convergence 246 / non-finite m_s at the m_s→0 boundary 93 /
    limited plasma 61).
  - **Coverage (resolved):** κ 1.58–2.23, δ 0.39–0.58, squareness/gaps vary, li 0.833–1.179,
    βp 0.113–0.519, Ip 489–729 kA; **m_s 0.001–15.4** (711 marginal m_s<0.4, 1427 mid, 1160 stable>1).
    **Physics cross-check:** corr(κ, m_s) = **−0.732** (strongly negative as required); the high-m_s tail
    is entirely low-κ (round, genuinely very stable), monotone across all bins — NOT artifacts.
  - **Held-out JOINT shape-corner split** (not random, but NOT extrapolation): test_extrap = the corner
    κ≥2.0049 AND δ≥0.4930 (187 samples; exact cuts in `phase15_split_meta.json`). **Honest framing:** each
    criterion alone is well-covered by train, so ~89% of the corner is INSIDE the training convex hull — a
    held-out compositional sub-region (interpolation-with-a-gap), not beyond-hull extrapolation; its median
    m_s≈0.48 (mid-regime), and the truly marginal (high-κ/**low**-δ) plasmas stay mostly in train, so it is
    **not** the marginal regime either. val 466 (random 15%); train 2645. (A true extrapolation split =
    hold out a univariate high-κ tail — a Phase-2 option.)
- **Artifacts:** `data/dataset_v1.parquet` (3298×48: 17 controls + 21 shape descriptors + 6 labels + 4
  bookkeeping), `DATASET.md` (datasheet), `figures/phase15_coverage.png`; `experiments/phase15_lib.py`,
  `phase15_generate.py`, `phase15_run.py`, `phase15_dataset.py`, `phase15_probe.py`, `phase15_pilot.py`;
  `data/phase15_chunk_*.json` (14, incremental), `data/phase15_anchors.pkl`, `data/phase15_split_meta.json`.
- **Caveats / corrections / surprises:**
  - **(Crash-proofing — key lesson)** A forward trajectory can drive the core current →0, so
    `ConstrainPaxisIp`'s `Ip/I_R` normalization produces NaN that **hard-crashes the native solver** (no
    Python traceback). Not pre-filterable (it's a mid-solve transient). Fix: run the solve under
    `np.errstate(divide/invalid/over="raise")` so it surfaces as a catchable `FloatingPointError` BEFORE
    NaN reaches native code → clean reject. The first pilot died from this; after the fix, 0 crashes.
  - **(Shutdown)** The machine powered off at **3298/3500** target. Incremental per-chunk saves preserved
    everything; 3298 already clears the gate so no resume was needed. Each chunk is a *deterministic
    prefix* of its RNG stream (per-chunk kept: 230,250,240,231,245,227,212,250,225,250,246,239,227,226),
    so the dataset is a reproducible subset (a rerun to target 3500 yields a superset).
  - **(Throughput)** 14 workers on **12 physical cores** is memory-bandwidth-bound (Jacobian builds):
    per-label cost ~40 s solo but ~110–185 CPU-s under 14-way contention (~4.6× inflation). Next bulk run:
    use ≤10–12 workers for better cache/throughput, not more.
  - m_s is still **not mode-converged at 40 modes** (Phase-0 finding; ≈+10%/doubling) — convergence study
    owed in Phase 2. γ→2.7e6/s at the m_s≈0.001 edge (expected: m_s→0 ⇒ γ Alfvénic) — valid boundary point.
  - **pandas/pyarrow were NOT actually installed** (RESULTS A1 was wrong) — installed pandas 3.0.3 +
    pyarrow 24.0.0 this phase.
- **Gate:** **PASS** — 3298 ≥ 3k converged; coverage documented (figure + datasheet); labels at the
  locked protocol; held-out shape-corner split defined; DATASET.md written.
- **Advice for the next prompt (→ Phase 2):** (0) DATA HYGIENE before training: drop the zero-variance
  columns (`I_Solenoid`≡5000, `n_unstable`≡1, `n_positive_margins`≡1) and the redundant ones (`Ip`≡`Ip_target`
  to ~1e-16; `tau_inst`≡1/γ); m_s is heavily right-skewed (0.0005–15.4, only ~200 samples >3) so train on a
  log/asinh m_s target. (1) Train the differentiable {m_s,γ}(shape) surrogate on `data/dataset_v1.parquet`
  using the shape descriptors (kappa, delta, squareness, gaps, li, betap, a, Rgeo); honor the `split` column
  (report on test_extrap separately — it is a held-out high-κ/high-δ CORNER, a generalization stress test, NOT
  true extrapolation; ~89% sits inside the train hull). (2) The m_s(shape) signal is strong (corr(κ,m_s)=−0.73)
  — but a plain linear model on log m_s already reaches R²≈0.91 (≈ the NN's 0.93), so ACCURACY IS NOT THE
  CONTRIBUTION: lead with the gradient (cosine/sign) + solver-confirmed design loop. Verify autodiff ∇m_s vs
  true-solver finite differences ON THIS dataset, resolved by m_s regime.
  (3) Do the **mode-convergence study** (40/80/160) and pick a converged `fix_n_vessel_modes` before
  finalizing published labels (current labels are at 40). (4) Put calibrated uncertainty on the m_s→0
  boundary (711 marginal samples available). (5) The very-stable tail (m_s>3, ~200 samples) is sparse —
  consider log/asinh target transform or down-weighting. (6) Reuse `phase15_lib.forward_label` for any
  fresh solver-confirmation solves; it is crash-proof and matches the dataset protocol exactly.
  (7) **Leuer baseline is STRONG by rank:** Spearman(leuer,m_s)=0.92 (Pearson only 0.13, killed by a few
  outliers) — report Spearman/AUC for the marginal-vs-stable decision, clip the leuer outliers, and show what
  the learned non-rigid m_s adds OVER it. (8) **Dimensionality headline is confounded on this data:** the
  shape descriptors are an observational cloud and strongly cross-correlated (|corr|~0.8 for several pairs;
  δ barely varies), so the effective shape dimension ≪ 6. Run the "gradient beats heuristics as D grows"
  experiment in the **independently-steerable CONTROL space** (coil currents + profile params) or on PCA-
  orthogonalized shape axes — do NOT treat each correlated descriptor as a free dimension.

### Phase 2 — surrogate, calibration, gradient verification & the dimensionality story — 2026-06-21 — PARTIAL PASS
- **Goal:** a calibrated differentiable m_s(shape) surrogate; verify its gradient against the true solver;
  finalize a converged mode count; and test whether the gradient's design value grows with shape dimensionality.
- **Did:** (protocol = the locked one: serialized machine, OMP=1, 65×65, cold solves, forward labels via
  `phase15_lib.forward_label`; all surrogate work on `dataset_v1.parquet`, split honored). New code:
  `experiments/phase2_{data,model,train,calibrate}.py` (surrogate + baselines + calibration),
  `phase2_modes_{worker,run,analyze}.py` (mode study), `phase2_grad_{worker,run,analyze}.py` +
  `phase2_gradcheck2.py` (gradient verification), `phase2_dim_{lib,worker,run,analyze}.py` (dimensionality),
  `phase2_figures.py`. Installed `cma 4.4.4`.
- **Result (numbers, resolved; vs the floor):**
  - **(0) MODE-CONVERGENCE STUDY (the owed Phase-0/1.5 action — RESOLVED).** Replayed 60 stratified shapes at
    modes {40,80,120,138} (138 = ALL passive structures = converged reference). The shipped 40-mode labels are
    **systematically LOW by median −13.5%**, regime-dependent: **marginal −27% (×1.36), mid −15%, stable −13%,
    very_stable −9%** (largest near m_s→0, as Phase 0 predicted but bigger than "+10%/doubling" — that was only
    40→80). **m_s is CONVERGED by 80 modes**: |80−138| median **0.4%**, max 3.4%; |120−138| ≈ 0.0%. **Converged
    count = 80.** Decision: dataset_v1 (40 modes) is kept for Phase-2 (all surrogate/gradient/design results are
    internally consistent at 40 modes — the absolute bias cancels in every relative comparison); the absolute m_s
    carries the stated regime-dependent systematic band. **Re-labeling dataset_v1 at 80 modes is owed before
    publication** (~10 h job; 80-mode Jacobians are ~10× a 40-mode solve — the mode study itself took 191 min).
  - **(1) SURROGATE ACCURACY** (heteroscedastic deep ensemble, smooth tanh, log-m_s target; gamma too): val
    **R²=0.954 (orig), 0.968 (log)**; held-out corner test_extrap **R²=0.876 (orig), 0.713 (log)**. **RESOLVED by
    regime — CRITICAL, the aggregate hides a collapse** (log-R²): val {marginal 0.63, mid 0.88, stable 0.91,
    very_stable 0.64}; **corner {mid 0.81, marginal 0.038}** — i.e. the held-out corner aggregate (0.713) is
    **carried entirely by its 143 mid points; on its 44 marginal points the model is ≈ a mean predictor**
    (log-R²=0.04, RMSE 0.34, only weak rank order Spearman 0.64). So "generalizes to the corner" is true for the
    mid regime, **NOT** for the marginal m_s→0 plasmas. **Baselines (corner, log-units, recomputed fairly):**
    surrogate R²=0.713/RMSE 0.181 vs **linear 0.355/0.271** vs **a FAIR GP 0.682/0.190** (GP on 1500 pts, 2
    restarts, wide bounds — the original GP was hobbled to 700 pts/0 restarts). **Honest read: the surrogate
    clearly beats LINEAR out-of-distribution, but its edge over a fair GP is small and NOT significant**
    (Wilcoxon p=0.088; median |err| 0.064 vs 0.068). Accuracy is NOT the contribution. Leuer baseline:
    Spearman(leuer,m_s) **0.90/0.87**, marginal-detection **AUC 0.97/0.94**; surrogate **AUC 0.99/0.98**.
  - **(1) GRADIENT VERIFICATION (nuanced — read carefully).** Autograd is **correctly wired** — an inline
    *self-consistency* check (autodiff vs central-FD of the *same* surrogate) gives cosine 1.0
    (`data/phase2_autograd_selfcheck.json`); this confirms no gradient-code bug but is **NOT physical
    verification**. The **physical** gradient-vs-true-solver agreement is only **moderate-to-weak**: per-axis cosine
    median **0.61** (shape-dir) / **0.64** (composed control-grad); the per-axis *pooled directional correlation is
    −0.62* (marginal −0.77, mid +0.06) and the per-base `directional_corr` is **0.17** — the surrogate's gradient is
    NOT a faithful Jacobian. What IS reliable is the **net ascent direction in the DESIGN regime**: box-clipped step
    along +∇m_s, confirmed by the true solver, raises m_s in **marginal 8/10 (Wilson 95% CI 0.49–0.94; vs coin-flip
    p≈0.06, n.s. at n=10) and mid 7/10 → pooled design regime 15/20 = 75% (Wilson 0.53–0.89, significant)**, median
    gain +0.26. (Earlier "directional sign 100%" was **tautological** — only on-manifold improving steps were
    recorded — now dropped. "beats-random" used an asymmetric setup: random=best-of-3 fixed-step vs gradient
    line-search; the gradient still wins but the comparison is not apples-to-apples.) **Off-manifold rejection is
    regime-graded, not stable-only:** the +grad step lands off the valid diverted manifold (true solve fails →
    penalty) in **marginal 2/10, mid 3/10 (~25% of design-regime steps)**, stable 7/8, very_stable 2/2 — so already-
    stable shapes fail almost always (over-stabilization off-manifold, a physical boundary) AND ~1-in-4 marginal/mid
    steps must be rejected (load-bearing for Phase-3 step budgets). The **composed**-control-gradient weakness
    (cos_B 0.64) is consistent with control→shape redundancy (effective dim 5.5/16; ShapeMap R²<0.6 for δ &
    inner-squareness, `data/phase2_shapemap_r2.json`); the **shape-space** weakness (cos_dir 0.61, pooled corr <0)
    is the surrogate's own gradient and is NOT explained by that. **Conclusion:** the surrogate gives a *usable but
    imperfect ascent direction* in the marginal/mid design regime, **NOT** a Jacobian replacement — which is exactly
    why the design loop must confirm every step and reject failures.
  - **(2) CALIBRATION.** Raw heteroscedastic ensemble is **over-confident** (RMS_z 1.86 val / 2.90 test). A single
    global post-hoc variance scale (s=1.86, fit on val) → val RMS_z **1.00**, cov90 0.97; test RMS_z 1.56, cov90
    0.98 (honestly still mildly overconfident on the harder corner). Predictive width is **widest in the marginal
    (0.100) and very_stable (0.104) regimes** vs stable (0.056): the model **abstains/widens at the m_s→0
    controllability boundary AND the sparse very-stable tail** — the trustworthiness story.
  - **(3) DIMENSIONALITY EXPERIMENT (headline — honest, partly positive).** PCA-orthogonalized CONTROL space
    (**effective dim = 5.51/16**, participation ratio); LOCAL bounded-step navigation (the only honest framing —
    see caveat); target m_s≥1.0 from marginal starts; budget 24 true solves; 5 starts; d∈{2,4,8,12}.
    **Gradient-based methods (surrogate + κ-heuristic) scale with dimension:** best true m_s 0.50→0.63→**1.07→1.13**
    (surrogate) as d=2→4→8→12, reaching the target in **~5 solves at d=12 (80% success)**. **Gradient-FREE search
    (CMA-ES, random) stays stuck** (best m_s 0.35–0.95, success 0–40%) at all d — it never navigates out of the
    marginal basin. So the differentiable surrogate's **gradient-vs-gradient-free advantage GROWS with dimension**
    (claim earned). **HONEST NULL:** the surrogate m_s-gradient does **NOT beat the simple reduce-κ heuristic**
    (comparable; heuristic hits 100% at d=12) — because **κ is the dominant stability lever on this single-machine
    ST manifold** (corr(κ,m_s)=−0.73), so the cheap κ-gradient already captures the signal. Beating the heuristic
    needs a richer/multi-lever design space (multi-machine) — future work.
- **Artifacts:** models `data/phase2_models/{surrogate,shapemap}.pt`; `data/phase2_{train_metrics,calibration,
  modes,modes_summary,modes_subset,gradcheck,gradcheck2,grad_probes,dim_results,dim_summary,dim_setup}.json`;
  `data/phase2_predictions.parquet`; figures `figures/phase2_{accuracy,gradient,calibration,modes,dimensionality}.png`;
  code listed above. Mode study 191 min (11 workers); grad probes 84 min; dim 92 min (10 workers).
- **Caveats / corrections / things that surprised you:**
  - **(BIG, methodological)** The FIRST dimensionality run was **confounded**: "random search" sampled GLOBALLY in
    the in-distribution box, which is rich in stable shapes (1160 stable samples), so it *teleported* to m_s>2 in
    one draw and trivially "won." Fixed by making ALL methods **local bounded-step** navigators (the realistic
    design setting). Lesson: a global-sampling baseline on a stable-rich manifold is not an optimizer — it is
    rejection sampling; it must be local to test navigation.
  - The per-axis FD gradient cosine (`phase2_grad_analyze`) is partly diluted by weak/redundant controls and
    sub-resolution separatrix-descriptor noise at small steps — but it is NOT only an artifact: the surrogate's
    shape-space gradient is genuinely weak-to-anti-correlated with the true solver in the design regime (pooled
    directional corr −0.62, marginal −0.77). The decision-relevant in-distribution ascent test (`phase2_gradcheck2`)
    is what actually matters for design, and it is positive (design regime 15/20) — but report it WITH n and CIs.
  - **(Adversarial-review fixes, 2026-06-21)** A 6-lens skeptic workflow (`phase2-adversarial-review`,
    36 agents) audited this entry and surfaced real overclaims in the *prose* (the raw numbers were all
    reproducible): the original "autodiff EXACT cosine 1.0000" (a self-consistency check, not physical
    verification), the corner "beats linear/GP" headline (mid-carried; marginal corner ≈ mean predictor; the GP was
    hobbled — a fair GP ties within noise), the "marginal 80% ascent" point estimate (n=10, CI 0.49–0.94, n.s.;
    pooled 15/20 is the significant statistic), the tautological "directional sign 100%", and the "fails only for
    already-stable shapes" mis-scoping (~25% off-manifold even in marginal/mid). All corrected above. New backing
    artifacts: `data/phase2_{shapemap_r2,autograd_selfcheck}.json`.
  - **DATASET.md correction:** `I_P6` is NOT near-constant — it varies (−4.17..0.15, std 0.84), small in magnitude
    but a live control. Constants are only `I_Solenoid`, `n_unstable`, `n_positive_margins`. (`Ip`≡`Ip_target` to
    6e-16, `tau_inst`≡1/γ exactly — confirmed.)
  - Per-solve cost (forward path, 40 modes) is **~23–29 s solo** (cheaper than the inverse path's 46–90 s); 80-mode
    ~same, 138-mode ~10× under contention. 11–14 workers on 12 cores is memory-bandwidth-bound (≈3–4× inflation).
  - A machine sleep/blip mid-run did NOT lose work: completed artifacts are on disk and the dim workers save
    incrementally per job (same robustness as Phase 1.5).
- **Gate:** **PARTIAL PASS.** ✅ converged mode count chosen (80) + 40-mode systematic stated; ✅ calibration honest
  (recalibrated, resolved, abstention shown); ⚠️ gradient gives a usable **net ascent direction** in the design
  regime (pooled 15/20, Wilson 0.53–0.89) but is **NOT** a faithful Jacobian (true-solver cosine ~0.6, pooled
  directional corr <0) and rejects ~25% of marginal/mid steps off-manifold — reported with n/CIs, not as "80%";
  ✅ dimensionality advantage of **gradient-vs-gradient-free** demonstrated and growing with d in a true (PCA
  control) dimension (n=5 starts — modest power). ❌ the specific **"m_s-gradient beats heuristics"** claim is
  **honestly reported as NOT holding** on this κ-dominated single-machine manifold; and ❌ the surrogate's accuracy
  edge over a *fair* GP is within noise (the OOD win is over linear only). Net: the genuine, well-supported
  contributions are the **converged-mode result**, **honest calibration/abstention**, and the **gradient-vs-
  gradient-free amortization + solver-confirmed loop**; the gradient-quality and accuracy claims are real but
  modest and must be stated with their CIs/baselines.
- **Advice for the next prompt (→ Phase 3 design loop):** (1) Pose design tasks **starting from MARGINAL shapes**
  (where the net ascent direction is reliable, pooled 15/20) — do NOT try to over-stabilize already-stable shapes
  (off-manifold); budget for **~25% off-manifold step rejections** even in the marginal/mid regime.
  (2) The design loop MUST confirm every step with a true solve and **reject off-manifold/penalty (m_s=0) steps**
  (the surrogate gradient is a direction, not a Jacobian). Reuse `phase2_dim_lib.run_gradient` (box-clipped local
  ascent) + `phase15_lib.forward_label`. (3) For baselines, gradient-free (CMA/random) must be **local** (not global
  box sampling). (4) Frame the contribution as **amortized gradient design that beats gradient-free search**, NOT as
  beating the κ heuristic — to beat the heuristic you need a design space where ≥2 shape levers matter comparably
  (a SECOND wall/machine config, or constrain κ and force the optimizer to use squareness/δ/gaps — a strong Phase-4
  experiment). (5) Absolute m_s is biased −13.5% (regime-dependent) at 40 modes; either re-label the design-loop
  confirmations at 80 modes or report the band. (6) Put the calibrated σ (s=1.86) on every design-loop m_s estimate;
  abstain where σ is wide (marginal boundary). (7) The surrogate's real accuracy edge is **out-of-distribution**
  (corner R² 0.71 vs linear 0.36) — lean on that, not on in-distribution R² (≈ linear/GP).

### Phase 2.5 — hardening sprint (fix the partial-pass shortfalls) — 2026-06-22 — IN PROGRESS
- **Goal:** turn the Phase-2 PARTIAL PASS toward a full pass by fixing the four real shortfalls the
  adversarial review surfaced: (a) the absolute m_s mode bias, (b) the unearned "gradient beats
  heuristics" headline, (c) thin statistical power, (d) single-machine scope / generalization.
- **Did + Result (numbers; all at the converged 80 modes unless noted):**
  - **RE-LABEL AT 80 MODES (bias removed).** Replayed all 3298 dataset_v1 controls at the converged
    80 modes (Machine A) → **`data/dataset_v1_80.parquet`** (3254 kept; 44 dropped at the m_s→0
    boundary). Confirms the systematic: median m_s(40)/m_s(80) = **0.859** (40-mode labels were ~14%
    low). Retrained the canonical surrogate + shapemap on the clean labels. **The real improvement is
    denominator-free:** test_extrap corner RMSE_log **0.181 → 0.072** (more than halved). The corner
    log-R² rose 0.713→**0.912**, but ~half of THAT jump is difficulty reduction, not skill: the +14%
    relabel reshuffled the corner's marginal count **44→7** and raised its min m_s 0.044→0.258, halving
    the target variance (so R² inflates mechanically) — do not headline the R² jump alone. **HONEST,
    unchanged:** resolved by regime the **marginal m_s→0 corner is still poor** (val marginal log-R²≈−0.27,
    n=57; the 7 still-marginal corner points log-R²=−1.4) — marginal-band accuracy is intrinsically hard at
    both mode counts. The surrogate's value is the gradient/design loop + calibration, not marginal point accuracy.
    (Run took 20.6 h — 80-mode + Machine-B linearisations under 11-way contention are ~6× a solo solve.)
  - **κ-CONSTRAINED design — the gradient beats gradient-free search (fair), and beats single-lever
    heuristics (with caveats).** **[⚠ SUPERSEDED 2026-06-23 by Phase 2.5b: this is the n=8 (1 marginal + 7 mid)
    run. The firmed n=20 (10 marginal + 10 mid) rerun gives vs gradient-free 17/20 = 85% (Wilcoxon p=1e-4,
    STILL significant) but vs the best fair heuristic only 15/20 = 75% (Wilcoxon p=0.064, NOT significant; n.s.
    within both regimes) and a 1.18× (not 2.7×) median-gain ratio — the "beats heuristics 7/8 / 88% / 2.7×"
    numbers below did NOT survive honest marginal coverage + two-sided tests. Read the Phase-2.5b entry as
    authoritative for this claim.]** Phase 2 found the m_s-gradient ties the reduce-κ heuristic because κ
    dominates. So we REMOVED that lever: hold κ fixed (±0.04) by projecting every step orthogonal to ∇κ
    and **rejecting any step whose TRUE-solver κ drifts beyond 0.04** (verified airtight by the review — the
    surrogate cannot cheat by drifting κ). Require m_s↑ via SECONDARY levers (squareness, gaps, l_i, …).
    8 starts (**1 marginal m_s=0.36 + 7 mid, m_s 0.44–0.90; pool floored at m_s≥0.2 so the m_s→0 regime is
    barely covered**), d=12, budget 18, 80-mode solver-confirmed, all methods at the same 18-solve budget.
      - **FAIR result (all methods budget-equalised at 18 solves; both-sign heuristics ranked by their OWN
        descriptor change, not the surrogate — the M6 fix):**
        median gain **surrogate +0.832, best fair single-lever heuristic +0.305, best gradient-free +0.270**.
        Surrogate beats **gradient-free in 100% of starts** (p=0.0039) and the **best fair heuristic in 7/8 =
        88% of starts** (paired Wilcoxon p=0.0078). So the surrogate's MULTI-LEVER knowledge genuinely beats
        both a canonical gradient-free optimizer AND any single-lever physics rule (≈2.7× the gain), fairly.
        (Honest size: n=8 with only **1 marginal m_s=0.36 start + 7 mid**; p-values at the n=8 resolution
        floor; the one start the surrogate does NOT win is where a single lever happened to suffice.)
      - **Bug caught here (own code):** the first fair-heuristic attempt had a method-name suffix bug
        (`SHAPE_FEATURES.index("sq_uo+")` threw) so every heuristic errored to 0.0, faking a "−0.616
        catastrophic failure"; a start-reproduction check (ms0=m_s_start exactly) exposed it; fixed + rerun
        → the +0.305 fair-heuristic number above.
      - **Take:** "the learned gradient beats heuristics" is now EARNED fairly, scoped to the κ-CONSTRAINED
        problem (on the unconstrained problem κ-reduction alone ties it) and to the mid/marginal start band.
  - **A MODIFIED-PASSIVE CONFIG (Machine B) — what it does and does NOT show (heavily revised post-review).**
    Built **Machine B** = MAST-U-like with the 82 PF-coil-CASE conductors removed (56 vs 138 passives;
    `machine_configs/MAST-U-B/`) — ONE subtractive perturbation of one machine. Passives carry zero static
    current, so B gives **byte-identical shapes** (all 20 descriptors max-diff 0.000) but a different m_s
    (median m_s_B/m_s_A80 = **0.654**; mean log-shift −0.50). **Spearman(κ,m_s) unchanged** (−0.875 A,
    −0.869 B). **The honest finding is a PHYSICS fact, not a surrogate win:** m_s on B is essentially a
    global affine-in-log RESCALE of m_s on A — a true-label affine (NO surrogate) maps A-labels→B with
    held-out log-R² **0.948 (val) / 0.859 (corner)**, Spearman **0.9943**. The A-surrogate transferred to B
    (held-out affine **0.913 / 0.725**, Spearman 0.989) does **NOT beat that trivial true-label ceiling** —
    it adds nothing to the transfer beyond the rescale; and the rescale is a 2-parameter affine (slope 1.29,
    not a 1-number shift; pure-shift = 0.90). Transfer also **degrades in the marginal band** (Spearman
    0.969 for m_s_B<0.5; corner OOD only 0.725) — exactly at the m_s→0 boundary that matters. **So: NOT
    "generalisation across conducting structures"** — it shows that removing these conductors rescales m_s
    while preserving the shape ordering, recoverable by a global affine; a genuine SECOND DEVICE (different
    geometry/aspect ratio) is still owed (Phase 4). The earlier circular in-sample affine (0.931) was a code
    bug, now fixed (fit-on-train/eval-held-out).
  - **DIRECT m_s(controls) SURROGATE — what it rules out (downgraded post-review).** Trained m_s end-to-end
    on controls (same architecture/hyperparameters) to test if the moderate gradient (true-solver cosine ~0.6)
    is just the lossy shape-map: direct control-gradient cosine **0.628 ≈ the composed 0.64**, so **the lossy
    ShapeMap is NOT the bottleneck**. This does NOT prove the weakness is "intrinsic / unfixable by any
    architecture" — it's one same-family alternative trained on the (biased) 40-mode labels vs a 40-mode FD
    reference; a Sobolev-trained or materially different model on 80-mode probes is untested. The robust,
    earned claim is the functional one: the surrogate gradient is a usable direction, not a Jacobian, so the
    solver-confirmed loop is necessary.
  - **DIMENSIONALITY rerun (80 modes, 10 starts) — CONFIRMS the Phase-2 headline with clean labels + double
    power.** Best true m_s at budget vs d (surrogate): 0.42→0.37→0.81→**1.10** (d=2/4/8/12), reaching the
    target in ~5 solves at d=12 (90% success); gradient-free (CMA/random) stays stuck (0–30% success, best
    m_s 0.44–0.74 even at d=12). So **gradient-based design beats gradient-free and the advantage GROWS with
    dimension** holds at the converged labels. Consistent with Phase 2, surrogate ≈ the reduce-κ heuristic in
    the UNCONSTRAINED problem (both reach ~1.1 at d=12) — which is exactly why the κ-constrained experiment
    above is the test that separates them. (40-mode version preserved as `data/phase2_dim_*_40mode.json`.)
- **Artifacts:** `data/dataset_v1_80.parquet`, `data/dataset_v2_B.parquet`, `machine_configs/MAST-U-B/`,
  `data/phase25_{crossmachine,kappa_summary,kappa_results,control_grad}.json`,
  `data/phase2_models/{surrogate,shapemap,surrogate_B,control_surrogate}.pt` (surrogate/shapemap now the
  clean 80-mode versions); code `experiments/phase25_*.py`.
- **Caveats:** absolute m_s is now at the converged 80 modes for Machine A; Machine-B labels are at all 56
  of its passives (its converged reference). The earlier 40-mode dim result is preserved as
  `data/phase2_dim_*_40mode.json`. Compute is the binding constraint (the re-label alone was 20.6 h).
- **Gate (after BOTH adversarial reviews + the fair-heuristic rerun; STILL PARTIAL):** real progress —
  ✅ converged labels (the −14% bias removed); ⚠️ **at fixed κ the gradient beats gradient-free search
  [n=8: 100%; FIRMED n=20 in Phase 2.5b: 85%, Wilcoxon p=1e-4 — still significant] AND the best fair
  single-lever heuristic [n=8: 88% / 7-of-8 p=0.0078; FIRMED n=20: 75%, Wilcoxon p=0.064 — NOT significant,
  n.s. within regimes]** — so the gradient-free win holds but the heuristic win is **within noise** (the
  ≈2.7× was an n=8 artifact; n=20 ratio is 1.18× vs heuristic); ✅ a modified-passive config shows the m_s(shape)
  ordering is recoverable by a global rescale. STILL owed before "FULL": (1) **marginal m_s→0 coverage** of
  the κ-constrained starts (only n=1); (2) the cross-machine claim is a single subtractive perturbation that
  **does not beat the trivial true-label rescale** — a genuine second DEVICE is owed (Phase 4); (3) n=8 is at
  the statistical resolution floor; (4) re-verify the gradient ascent at 80 modes. **[(1),(3),(4) all DONE in
  Phase 2.5b — see that entry; the firmed n=20 result is now authoritative.]** The two reviews + my own
  checks caught **2 code bugs** (circular cross-machine affine; fair-heuristic method-name suffix) and 8 prose
  overclaims — **all corrected**. **Net: Phase 2 stays PARTIAL; the κ-constrained "gradient beats gradient-free
  search" is the strongest, now-fair new result — BUT "beats fair single-lever heuristics" did NOT survive the
  Phase-2.5b firming (n=20: 75%, Wilcoxon p=0.064, within noise).**
- **Advice for the next prompt (→ Phase 3 / finish hardening):** (1) **[DONE in Phase 2.5b]** the fair-heuristic
  rerun + marginal-regime κ-constrained starts (n=20) — outcome: gradient-free win firm, heuristic win within
  noise. (2) Lead the design loop with the κ-constrained task vs **gradient-free** (the fair, decisive comparison).
  (3) Cross-machine: present it honestly as "m_s rescales with the conducting structure" (a calibration result),
  not surrogate transfer;
  a true second device is the real Phase-4 differentiator. (4) absolute m_s is now clean at 80 modes;
  re-verify the gradient ascent at 80 modes (the existing gradcheck2 is at 40 modes).

### Phase 2.5b — foundation-hardening (firm κ-headline + q95 + true extrapolation split + 80-mode gradient re-verify) — 2026-06-23 — PARTIAL PASS
- **Goal:** harden the Phase-2 foundation before Phase 3: (A) FIRM the κ-constrained "beats heuristics"
  headline with real marginal coverage + honest CIs; (B) add q95 + a TRUE univariate extrapolation split;
  (C) re-verify the gradient ascent at the converged 80 modes. (protocol = locked: serialized machine,
  OMP=1, 65×65, COLD forward solves, fix_n_vessel_modes=80; all on dataset_v1_80.)
- **TASK A — κ-constrained headline FIRMED (n=8 → n=20; SUPERSEDES the Phase-2.5 n=8 numbers).**
  20 STRATIFIED starts (10 marginal m_s∈[0.15,0.39] + 10 mid [0.43,0.96]), d=12, budget 18, κ held ±0.04
  **enforced on the TRUE-solver κ** (re-verified airtight: `run_constrained.accept()` and the CMA viol-path
  both gate on `forward_label`'s κ, not the surrogate's). All 11 methods budget-equalized. 220/220 runs, 0
  errors. *(Real bug fixed: the extreme marginal starts (high-κ, in the 2% PC-score tail) had x0 OUTSIDE the
  [p2,p98] box → CMA-ES crashed ("argument of inverse must be within the given bounds"); per-start box
  expansion to contain x0, applied IDENTICALLY to all methods, fixes it — the start is a valid in-distribution
  shape.)*
    - **vs gradient-free (CMA/random) — ROBUST + SIGNIFICANT:** pooled **17/20 = 85%** (Wilson 64–95%;
      two-sided Wilcoxon **p=1e-4**; sign-test p=0.003); significant in BOTH regimes (marginal 9/10 p=0.006,
      mid 8/10 p=0.014). Median gain surrogate **+0.548** vs gradient-free **+0.301** (≈1.8×).
    - **vs best fair single-lever heuristic (per-start ORACLE best-of-8 — the conservative baseline) — POSITIVE
      BUT WITHIN NOISE:** pooled **15/20 = 75%** (Wilson 53–89%; sign-test p=0.041; **two-sided Wilcoxon
      p=0.064 — NOT significant**). Resolved by regime it is **n.s. in BOTH**: marginal 7/10 (p=0.34), mid 8/10
      (p=0.11). Median gain surrogate +0.548 vs best-heuristic **+0.463** (only **1.18×**; IQRs overlap heavily).
    - **HONEST TAKE:** the earlier n=8 headline (**7/8 = 88%, p=0.0078, "≈2.7×"**) does **NOT survive** honest
      marginal coverage + two-sided tests. At fixed κ the learned gradient **robustly beats gradient-free
      search** (the real, significant result); its edge over the best single-lever physics heuristic is
      **suggestive, not established** (n=20, Wilcoxon p=0.064). The "2.7×" was an n=8 artifact (vs gradient-free
      at the old, mid-heavy start pool). Do NOT headline "beats heuristics."
- **TASK B(i) — q95 added (dataset_v1_80q).** Re-solved all 3254 shapes FORWARD-ONLY (no linearisation, ~5 s
  each; `phase2_q_lib.forward_q` replicates `phase15_lib.forward_label`'s crash-proofed solve VERBATIM → same
  equilibrium; **κ reproduces bit-exactly, Δ=0 for all 3254**). q95/qmin/q05 via `eq.q(ψ_n grid)` (length-1
  calls trip a removed `np.asscalar`; evaluated on a [0.05,0.95] grid). q95 **physically sane**: median 5.58,
  p05–p95 4.1–7.5, **100% in the ST band [3,10]**; qmin>1 for 99.3% (avoids the m=1 surface). corr(q95,κ)=+0.61,
  corr(q95,log m_s)=−0.48 — **but the m_s-correlation is almost entirely κ-MEDIATED: partial corr |κ = +0.16**
  (κ→log m_s corr −0.875). So q95 is a sane DESCRIPTIVE feature, NOT independent m_s signal (don't expect
  surrogate gains from adding it). (qmin≡q05: q is monotone-increasing on the grid.) → `data/dataset_v1_80q.parquet`.
- **TASK B(ii) — TRUE univariate high-κ extrapolation split (`split_kappa`).** The existing `test_extrap` is a
  JOINT κ×δ corner (~89% in-hull = interpolation-with-a-gap). New split: hold out the **top 7% by κ (κ≥2.078;
  228 shapes; train κ-max = 2.078 → the model NEVER sees higher κ)**; retrained (same arch/hyperparameters,
  `surrogate_kappa_extrap.pt`). **RESOLVED BY REGIME (RMSE_log, denominator-free):**
    - MID (apples-to-apples): in-dist val 0.063 → high-κ tail **0.076** (+21%, a real modest extrapolation
      penalty); corner-mid 0.059.
    - MARGINAL: in-dist val 0.242 → high-κ tail **0.133** — the tail is predicted **BETTER** than in-dist
      marginals (high-κ marginals follow the dominant smooth κ→m_s trend; low-κ-but-marginal shapes are
      intrinsically harder).
    - **The aggregate "tail RMSE_log 0.119 vs corner 0.072" is REGIME-CONFOUNDED** (corner 94% mid, tail 71%
      marginal) — do not headline it. Honest finding: the smooth surrogate **extrapolates the κ trend with only
      a modest mid-regime penalty and does NOT collapse on the held-out tail**; the κ extension is modest (the
      median tail point sits right at the boundary; +1.26σ describes only the single most-extreme point). The
      pre-registered "expect worse than corner" is only weakly borne out (mid) and reversed (marginal). The
      aggregate log-R²=0.93 is variance-inflated — lead with regime-resolved RMSE_log.
- **TASK C — gradient ascent re-verified at the converged 80 modes (the owed Phase-2.5 action).** Re-ran the
  in-distribution ascent test on the 30 held-out bases with the **A80 surrogate + 80-mode true solves**,
  **recomputing each base m_s at 80 modes** (consistency fix: the prior run compared 80-mode steps to a
  40-mode base) and re-binning regime by the 80-mode label (5 marginal migrated to mid: 40-mode 10/10 →
  80-mode 5 marginal/15 mid).
    - **Design-regime (marginal+mid) ascent 11/20 = 55% (Wilson 34–74%)** — DOWN from 75% (15/20) at 40 modes,
      and **the CI now SPANS 0.5** → at this n the raw single-step ascent is NOT distinguishable from chance.
      By regime: marginal 3/5 = 60% (n=5, uninformative), mid 8/15 = 53%. Pooled (incl. stable/very_stable)
      12/30 = 40% with NEGATIVE median gain −0.36 (stable/very_stable still go off-manifold = over-stabilization,
      expected). Where it ascends, median gains are positive (marginal +0.24, mid +0.40).
    - The directional-derivative corr "improved" 0.17→0.69 BUT this is partly artifactual: the 40-mode 0.17 was
      depressed by one near-zero-m_s base (relabeled higher at 80 modes) and the 80-mode 0.69 is computed only
      on the 13/30 on-manifold steps (a conditional statistic).
    - **HONEST TAKE:** at the converged labels the raw gradient is a **usable-but-WEAKER ascent direction
      (within noise of chance at n=20)**. This does NOT contradict Task A: the κ-constrained DESIGN LOOP confirms
      every step + rejects failures, so it is robust to imperfect single steps — the gradient is a *direction,
      not a Jacobian* (the Phase-2 framing holds). Task A (design loop) and Task C (raw single step) measure
      different things.
- **Artifacts:** `data/dataset_v1_80q.parquet` (q95/qmin/q05 + `split_kappa`); `data/phase2_q_summary.json`,
  `phase2_extrap_kappa.json`, `phase25_kappa_{summary,results}.json` (n=20), `phase2_gradcheck2_80.json`; model `data/phase2_models/surrogate_kappa_extrap.pt`; figures
  `figures/phase25b_{q95,extrap,gradient80}.png` + regenerated `phase25_kappa.png`; code
  `experiments/phase2_q_{lib,worker,run}.py`, `phase2_extrap_kappa.py`, `phase2_gradcheck2_80.py`,
  `phase25_kappa_{run,worker,lib,analyze}.py` (firmed), `phase25b_figures.py`.
- **Caveats / corrections / surprises:** (1) the Phase-2.5 n=8 κ result is **SUPERSEDED** by this n=20 run; the
  heuristic-beating is downgraded to "suggestive, not significant." (2) **Power-loss scare mid-run:** the machine
  never fully shut down, so the original jobs survived and the recovery relaunch briefly DUPLICATED them (two
  racing runs writing the same chunk files); verified the assembled output is **220 unique combos, 0 duplicates,
  0 corruption, 0 errors** (added atomic-write + per-job resume hardening to the worker). Task C had completed
  validly overnight. (3) An adversarial workflow (6 dimensions × 2 verifiers, **86 agents, 1268 tool-uses**)
  audited these claims; **29 findings survived** → all incorporated here (stale n=8/2.7×/100% prose corrected;
  the extrapolation regime-confound surfaced; the gradient-ascent drop reported plainly; one figure-code bug
  fixed). (4) Compute: q95 ~22 min (11 workers); the n=20 κ run ~5 h wall under the duplicate-process contention.
- **Gate: PARTIAL PASS.** ✅ q95 added + physically sane + reported (κ-mediated). ✅ true high-κ extrapolation
  split defined + accuracy reported regime-resolved (modest mid penalty, no collapse). ✅ **κ vs gradient-free
  firmly established** (85%, Wilcoxon p=1e-4) WITH marginal coverage. ⚠️ **κ vs best single-lever heuristic:
  positive but WITHIN NOISE** (75% pooled, Wilcoxon p=0.064; n.s. within both regimes) — the n=8 "beats
  heuristics" headline does NOT survive firming. ⚠️ **gradient ascent at converged 80 modes is WEAKER**
  (design-regime 55%, CI spans chance) — a usable direction, not a Jacobian; the design loop's value rests on
  solver-confirmation, not raw single-step ascent.
- **Advice for the next prompt (→ Phase 3 design loop):** (1) LEAD with **"gradient beats gradient-free search"**
  (the significant, robust result), NOT "beats heuristics" (within noise at n=20). (2) The raw gradient is weak
  at 80 modes — Phase 3 MUST confirm every step + reject off-manifold (~40% pooled, ~45% design-regime rejection).
  (3) The surrogate's design value comes from the multi-lever LOOP under solver-confirmation, not single-step
  accuracy. (4) Use `dataset_v1_80q.parquet` (adds q95); both splits documented in DATASET.md; q95 adds little
  independent m_s signal. (5) For more power, the κ-constrained experiment needs more marginal starts (n=10/regime
  is at the resolution floor) — but the heuristic comparison may simply be a genuine near-tie (κ-dominated manifold,
  the standing Phase-2 finding).

### Phase 3 — the differentiable, solver-confirmed design loop (the contribution) — 2026-06-23 — PASS (honestly scoped)
- **Goal:** demonstrate the money result end-to-end — gradient-based, solver-confirmed, in-range shape design that
  raises true m_s from MARGINAL ST starts, vs gradient-free baselines at the SAME true-solve budget — and report
  true-solves-to-target + final m_s resolved by regime with CIs and two-sided tests.
- **Did:** (protocol = locked: serialized machine, OMP=1, 65×65, COLD forward solves, **fix_n_vessel_modes=80**;
  surrogate/shapemap = the clean 80-mode models; all on `dataset_v1_80q.parquet`). Posed **20 stratified design
  tasks** (10 marginal m_s∈[0.15,0.39] + 10 mid [0.43,0.96]) in the **d=12 PCA control space** (the same
  independently-steerable space + setup as Phase 2/2.5b): from each start, raise true m_s toward **target m\*=1.0**
  (cross into the 'stable' regime) by **local bounded-step projected-gradient ascent** on the differentiable surrogate,
  **holding shape descriptors {κ,δ,gap_inner,gap_outer,gap_min,li} IN RANGE** (dataset [p1,p99], per-start expanded to
  contain the start), **CONFIRMING every accepted step with a fresh true 80-mode FreeGSNKE solve** and REJECTING
  off-manifold (non-converged/limited) / out-of-range / non-improving steps. Compared 5 methods at **equal budget = 30
  true solves**: `surrogate` (autodiff ∇m_s loop, FREE surrogate line-search + 1 confirm solve/iter), `heuristic`
  (reduce-κ via the differentiable ShapeMap geometry, NO learned m_s), and gradient-free `cma`/`random`/`nelder` on the
  TRUE solver. New code: `experiments/phase3_{lib,run,worker,analyze,figures,gallery}.py` (resume-safe atomic-write
  pool REUSED from phase25; 10 thread-pinned workers; **100 runs, 1651 true solves, ~4.0 h, 0 errors, 100/100 unique
  — 0 dups/corruption**). Then an adversarial-review Workflow (**38 agents, 32 findings, 24 survived, 0 high-severity**)
  + a **fair-CMA control** answering the strongest finding. ALL survivors incorporated.
- **Result (numbers, resolved by regime; vs the noise floor — within-config bit-reproducible at OMP=1/80 modes, so
  these gains of +0.7…+1.1 are >> any residual systematic floor):**
  - **HEADLINE — AMORTIZATION (true-solves-to-target = the design-relevant metric):** the gradient design loop reaches
    m\*=1.0 in a **median of 4 serial true solves vs 13.5** for the per-start **oracle best-of-3 gradient-free envelope**
    (CMA/random/Nelder; CMA alone median 15); **faster in 17/20 starts (two-sided paired Wilcoxon p=0.0009, median
    saving +9.5 solves)**. Reach-rate 18/20 vs 14/20 (**McNemar p=0.22 — within noise pooled**; the p=0.0009 is the SPEED
    test only, NOT the reach difference).
  - **MARGINAL regime (the discriminating m_s→0 boundary):** surrogate stabilizes **9/10** in median **6** solves vs the
    best-of-3 gradient-free **4/10** in median **20** (paired Wilcoxon p=0.004; McNemar 5–0 → exact **p=0.0625**,
    suggestive at n=10); random & Nelder-Mead reach **0/10**. MID regime is non-discriminating (starts near target; all
    reach).
  - **FAIR-CMA CONTROL (settles the "CMA was generation-starved" finding) — the cleanest honest result:** default CMA
    used popsize 11 → only ~2.5 generations at budget 30. Re-ran CMA with **popsize 6 + a graded out-of-range penalty**
    on the marginal starts: at **equal budget (30)** it reaches **5/10** (median 24 solves); at **2× budget (60)** it
    reaches **9/10 — matching the surrogate's reach** — but needs **median 26 solves vs the surrogate's 6 (~4×)**. So
    the *reach* gap was partly a **budget artifact**, but the **solve-EFFICIENCY gap (~4×) is robust** to a fair popsize
    AND double budget AND a graded penalty → **NOT a starvation artifact. The contribution is solve-EFFICIENCY, not
    unique capability** (gradient-free *can* reach the targets, it just costs ~4× the expensive solves).
  - **ATTRIBUTION (honest, load-bearing):** the efficiency win over gradient-free is delivered by the **GRADIENT /
    κ-direction, NOT specifically the learned m_s.** The reduce-κ heuristic (uses the differentiable ShapeMap geometry,
    **zero expensive m_s labels**) ALSO beats gradient-free on solves (**16/20, p=0.0004**), and **surrogate-vs-heuristic
    is within noise (6/20, p=0.23, median saving 0)**. Consistent with the κ-dominated single-machine manifold (the
    standing Phase-2/2.5b finding). *(Both surrogate and heuristic share the learned ShapeMap; the heuristic adds no m_s
    labels on top.)*
  - **PARALLEL-vs-SERIAL caveat (disclosed):** solves-to-target is a **SERIAL** expensive-solve count. If a CMA
    generation (popsize 11) is solved at once, the pooled per-query advantage is **within noise (5/20, p=0.54)**. The
    serial advantage holds when expensive solves are **not** parallelizable (the typical single-machine FreeGSNKE design
    setting); the **reach-rate gap is framing-invariant**.
  - **FINAL-MARGIN is NOT the metric (confound):** past the m\*=1.0 crossing, final m_s is confounded by step/population
    GRANULARITY in **all** methods (surrogate overshoots median +0.22; CMA's population overshoots more) and mid starts
    begin near target — so the pooled final-gain paired win is **within noise (60%)**. Solves-to-target / reach (cross
    the threshold at fewest expensive solves = the actual design goal) are the unconfounded, design-relevant metrics.
  - **BREAK-EVEN (amortization disclosed):** the surrogate amortizes a one-time **offline cost ≈ 3254 true 80-mode
    solves** (the training set, a **shared project asset** also used for Phase-2 accuracy/calibration/gradient + 2.5b);
    at a median saving ~9.5 solves/query it is net-positive after **~340 design queries** on this machine (the 20
    demonstrated starts recoup <6%). So "**far fewer expensive solves per query**," not "orders of magnitude fewer
    overall" until many queries.
  - **VALIDITY:** the **design gallery** (`figures/phase3_gallery.png`) shows real solver-confirmed LCFS stabilization
    (marginal start → rounder, less-elongated shape), each design **re-solved independently** at 80 modes (not the
    optimizer cache); it spans **best / typical / FAILURE** (the 1/20 surrogate failure = marginal start 6, gradient
    stalled at m_s 0.43, the loop correctly stopped rather than fake progress). The in-range guard + per-start box/guard
    expansion are computed once per start and applied **identically to all 5 methods** (refutes "secretly helps the
    surrogate"; the widening is tiny). Surrogate run used 96 of a possible 96 solves with **12 out_of_range rejections**
    (the guard binds and is load-bearing); gradient-free spent 460–513 solves each (mostly `no_improve`).
- **Artifacts:** code `experiments/phase3_{lib,run,worker,analyze,figures,gallery}.py`, `phase3_cma_fair.py`,
  `phase3_{smoke,validate}.py`; data `data/phase3_{results,summary,setup,jobs}.json`,
  `data/phase3_desc_ranges.json`, `data/phase3_cmafair_{results,summary}.json`,
  `data/phase3_chunk_*.json` (10, incremental) + `phase3_cmafair_chunk_*.json`; figures
  `figures/phase3_{efficiency,gallery}.png`. Reused `phase2_dim_lib.DesignSpace`/`_grad_x`, `phase2_model`,
  `phase15_lib.forward_label`, `data/phase2_dim_setup.json`, `data/phase2_models/{surrogate,shapemap}.pt`.
- **Caveats / corrections / things that surprised you:**
  - **(Adversarial review, the big one)** The first-pass headline ("median 4 vs 13.5, faster 17/20, p=0.0009; marginal
    9/10 vs 4/10") was REAL but **mis-scoped**. The 38-agent panel (24 survivors, 0 high-sev) forced five honest fixes,
    all now in the numbers/prose/figures: (1) the **reach** difference is NOT significant (p attaches to SPEED only);
    (2) default **CMA was generation-starved** → ran the fair-CMA control, which showed the *reach* gap was partly
    budget but the **~4× efficiency gap is robust**; (3) **serial-vs-parallel** framing (pooled edge within noise under
    parallel population eval); (4) the efficiency win is the **κ-gradient, not the learned m_s** (heuristic ties);
    (5) **break-even ~340 queries** disclosed; plus the gallery re-curated to include a FAILURE case and drop the
    inflated ×N ratio. **No finding was fatal; the design loop + the ~4× serial efficiency claim survived.**
  - The **final-margin "gain" metric is a trap** here: the stop-at-target rule + CMA's population evaluation make
    gradient-free *overshoot* m\*=1.0 on its stopping generation (mid start 15 CMA→1.80), inflating its final margin
    without improving solve-efficiency. Lead with solves-to-target/reach, never the gain ratio.
  - Per-solve cost (80-mode forward+linearisation): **~25 s solo, ~87 s under 10-worker contention** (~3.5×). The run
    was 1651 solves; the resume-safe atomic-write worker pool again held through (0 dups, 0 corruption).
  - FD-on-surrogate was NOT run as a separate baseline (the Phase-2 self-consistency cosine 1.0 shows autodiff ≡ FD of
    the same smooth net, so it would duplicate `surrogate`).
- **Gate: PASS (honestly scoped).** ✅ the differentiable, solver-confirmed design loop is demonstrated end-to-end and
  stabilizes marginal ST plasmas (m_s≈0.3→1.1) in a handful of expensive solves; ✅ it reaches targets in **~4× fewer
  serial true solves than even a FAIR, double-budget gradient-free CMA** (p=0.0009; robust to the starvation critique);
  ✅ every reported design is independently 80-mode solver-confirmed; ✅ all gains >> the noise floor. **Honestly scoped
  open items (NOT gate failures):** the efficiency win is the **gradient/κ-direction** (the LEARNED m_s ties the
  κ-heuristic on this κ-dominated single machine — the standing Phase-2/2.5b finding); the per-query edge is **serial-only**
  (within noise under parallel population eval); the marginal reach advantage is **suggestive** (n=10, McNemar p=0.0625);
  amortization needs **~340 queries** to break even.
- **Advice for the next prompt (→ Phase 4 rigor / the LEARNED-m_s differentiator):** (1) The decisive open question is
  unchanged: **make the LEARNED m_s (not just κ-geometry) the load-bearing lever.** The κ-CONSTRAINED design loop is the
  test that separates them (Phase 2.5b: learned gradient beats gradient-free 85% with κ held fixed; the heuristic
  comparison was within noise) — run a **κ-constrained design GALLERY** + the cost/accuracy Pareto there, OR a genuine
  **second device** (different geometry/aspect ratio, not Machine B's subtractive perturbation). (2) Always report BOTH
  serial and parallel-population accounting for any "fewer solves" claim. (3) For the marginal regime, n=10/regime is at
  the resolution floor (McNemar p=0.0625) — more marginal starts for power. (4) Reuse `phase3_lib` (the instrumented
  loop + Recorder + true_eval in-range guard) and `phase3_cma_fair.run_cma_fair` (fair popsize + graded penalty) — they
  survived the adversarial panel. (5) State the **~340-query break-even** wherever amortization is claimed; the training
  set is a shared asset (don't double-charge it). (6) Lead the paper's design section with **"amortized differentiable
  design ≈4× fewer expensive solves, every step solver-confirmed,"** NOT "beats gradient-free at finding higher m_s."

### Phase 4 — rigor layer + the LEARNED-m_s differentiator (κ-constrained) — 2026-06-24 — PASS (honestly scoped)
- **Goal:** the sections a referee/PPPL scientist will probe — and, leading, **make the LEARNED m_s (not just the
  κ/geometry direction) the load-bearing lever**, since on the single κ-dominated MAST-U manifold Phase 3 showed the
  learned m_s only *ties* the reduce-κ heuristic. Plus the rigor layer (cost/accuracy Pareto, Leuer baseline,
  ablations, robustness) and an honest genuine-2nd-device assessment.
- **Did:** (protocol = locked: serialized machine, OMP=1, 65×65, COLD forward solves, **fix_n_vessel_modes=80**;
  surrogate/shapemap = the clean 80-mode models; all on `dataset_v1_80q.parquet`, split honored). New code
  `experiments/phase4_*.py`. (1) **Re-framed** the Phase-2.5b n=20 κ-constrained run (κ held ±0.04 on the TRUE-solver
  κ ⇒ the reduce-κ lever is DISABLED) against *realizable* baselines (`phase4_kappa_reanalyze.py`). (2) Ran a
  **κ-constrained design GALLERY** (`phase4_gallery_{lib,run,worker}.py`, 10 thread-pinned workers, 60 runs, 136 min,
  0 errors) — the same 20 stratified starts, surrogate + a `kappa_nudge` confound control + gradient-free CMA, with
  full trajectory/descriptor/control recording; then re-solved 3 panels for LCFS shapes (`phase4_gallery_shapes.py`).
  (3) **Cost/accuracy Pareto** (`phase4_pareto.py`, 8 clean solo solves). (4) **Leuer baseline** (`phase4_leuer.py`).
  (5) **Ablations** (`phase4_ablations_light.py` ensemble-size/input-noise/mode-count; `phase4_ablations_train.py`
  learning-curve + shape-parameterization, re-run at the CANONICAL 8×2600 config — no speed compromise). (6) **Power
  top-up — TWO disjoint batches** (`phase4_power_{run,worker}.py` with `--tag`, +12 then +24 starts, 396 runs, 0
  errors) + pooled reanalysis with out-of-sample replication (`phase4_kappa_pooled.py`, **n=56, marginal n=40**).
  (7) Genuine-2nd-device feasibility check + a 50-agent QUALITY-UPGRADE audit (separate from the overclaim review).
- **Result (numbers, resolved by regime; vs the noise floor — within-config bit-reproducible at OMP=1/80 modes, so
  the +0.3…+0.8 m_s gains are >> any residual systematic floor):**
  - **LEARNED-m_s DIFFERENTIATOR at fixed κ (the lead) — re-framed honestly; FIRMED to n=56 (marginal 40 + mid 16)
    via two disjoint power top-up batches (quality directive — both prior borderline cells now resolved).** With κ held
    ±0.04 (reduce-κ DISABLED), the learned m_s gradient (via SECONDARY levers squareness/gaps/l_i) beats, POOLED (n=56):
      - **gradient-free search 50/56 = 89% (Wilcoxon p<1e-4; sign p<1e-4)** — the strongest, robust;
      - **the best REALIZABLE single FIXED lever (gap_outer−, pooled-median gain +0.391) 45/56 = 80% (Wilcoxon
        p<1e-4; sign-test p<1e-4)** — significant: no single physics rule a designer could pick in advance is
        competitive (next-best levers betap+, li−, sq_uo+; most ≈0);
      - **the per-start ORACLE best-of-8 lever (hindsight) 42/56 = 75% (Wilcoxon p=0.0007; sign p=0.0002) — now also
        SIGNIFICANT** (was borderline p=0.052 at n=32; the larger n resolved it). The oracle is still NOT realizable
        (a designer must SEARCH the 8 levers, ~8× the budget); the surrogate finds the multi-lever combination in ONE
        budget. So Phase-2.5b's "ties the best heuristic" was an artifact of comparing to an *unrealizable* baseline at
        small n — at n=56 the learned m_s beats EVERY baseline, even the oracle, significantly.
      - **OUT-OF-SAMPLE REPLICATION (pre-registered):** the FRESH disjoint 20-marginal cohort (batch 2) ALONE
        reproduces it — surrogate beats the best fixed lever **16/20 = 80%, Wilcoxon p=0.0017, sign p=0.012** (both
        tests) — so the marginal result is replicated, not just pooled.
      - **By regime (Wilcoxon two-sided): MARGINAL (n=40) vs gradient-free 37/40 = 92% (p<1e-4), vs fixed-lever
        30/40 = 75% (Wilcoxon p=0.0006, sign p=0.002 — now SIGNIFICANT under BOTH tests, up from sign-borderline at
        n=20), vs oracle 29/40 = 72% (Wilcoxon p=0.004, sign p=0.006 — significant); MID (n=16) vs gradient-free 13/16
        = 81% (p=0.002), vs fixed-lever 15/16 = 94% (p=0.0006), vs oracle 13/16 = 81% (p=0.058).** (n=20 reframe in
        `phase4_kappa_reframed.json`; pooled n=56 + replication in `phase4_kappa_pooled.json`; batches
        `phase4_power{,2}_results.json`.)
  - **DESIGN GALLERY (20 starts, budget 18) — κ-lock VERIFIED + the confound DECOMPOSED.** Every reported surrogate
    design held κ fixed: **20/20 within ±0.04 (max true-κ drift 0.0399)**. Median surrogate gain **+0.548**; the
    `kappa_nudge` control (deliberately reduce κ within the ±0.04 tolerance — the residual-κ freedom alone) buys only
    **+0.181**, so the surrogate's **EXCESS = +0.386 (≈70% of the gain) is the genuine SECONDARY-lever / learned-m_s
    contribution** (marginal excess +0.414, mid +0.366). i.e. the ±0.04 tolerance IS a real confound, but it is
    minority; subtracting it leaves a large learned-m_s effect. Solver-confirmed before→after panels (`figures/
    phase4_gallery.png`, every shape independently 80-mode re-solved): **best-marginal start m_s 0.193→1.032 (crosses
    into 'stable') at κ 2.054→2.043 (LOCKED)** via gap_outer 0.279→0.218, l_i 0.924→0.895, squareness — NOT κ;
    typical-mid 0.487→1.061; an honest weak case (start 14) only 0.640→0.770.
  - **SHAPE-PARAMETERIZATION ablation corroborates the levers are real + learnable** (held-out RMSE_log): κ-only
    **0.442** → +δ 0.262 → +gaps 0.162 → **+squareness 0.121** → +l_i,βp 0.119 → full-20 0.122 — a **3.6× error
    reduction** over κ-alone, the biggest single jump from **squareness** (mid RMSE_log 0.130→0.068). So the secondary
    descriptors carry real, learnable m_s signal beyond κ (full-20 plateaus by ~11 features).
  - **COST/ACCURACY PARETO (clean solo, locked protocol):** one full forward+80-mode-Jacobian solve = **22.7 s** (the
    **linearisation is 78% of it = 17.7 s**; forward-only 5.0 s); surrogate m_s **inference 0.35 ms**, **+ composed
    autodiff gradient 2.45 ms** ⇒ **~8,100× faster with the gradient (~65,000× inference-only)**, at held-out
    **log-R² 0.971 / RMSE_log 0.123 (n=652)**. (Amortized: the surrogate replaces specifically the expensive Jacobian.)
  - **LEUER physics baseline (held-out n=652).** Rigid Leuer is a strong RANK predictor — **Spearman 0.917** (Pearson
    only **0.134**, outlier-wrecked as predicted) — and marginal-vs-controllable **AUC 0.959**. The learned m_s lifts
    the ceiling to **Spearman 0.984 / AUC 0.972**; resolved by regime the lift is biggest where the rigid model breaks
    — **very_stable Spearman 0.219→0.834** — and is a **tie in the marginal band (0.852≈0.851)** (intrinsically hard at
    both). So the non-rigid learned m_s adds most for round, very-stable plasmas.
  - **OTHER ABLATIONS / ROBUSTNESS.** Accuracy plateaus by **k=4 ensemble members** (RMSE_log 0.123); the gradient
    DIRECTION is robust even at **k=1** (cosine 0.993 vs the full-8 ensemble). Learning curve: **most of the accuracy
    is reached by ~650 training shapes; small residual gains continue to the full set** (RMSE_log 0.130 at n=650 →
    0.121 at n=2602; 0.140 at n=325; the marginal band stays hard ~0.32 at every size). Mode count
    converged at **80** (existing 60-shape study; 40-mode labels low −9…−27%). Input-noise robustness: **5%-of-σ input
    noise → RMSE_log +0.004 and gradient-direction cosine 1.000 (p10 0.994)** — the smooth surrogate + gradient are
    robust to input perturbations (NOTE: this tests surrogate SMOOTHNESS, not true-solver gradient agreement — that is
    the separate, weaker Phase-2.5b finding).
  - **GRID SYSTEMATIC RE-VERIFIED ON THE MARGINAL BAND AT 80 MODES (quality-audit item; `phase4_grid_check.py`, 18
    shapes solved at 65² AND 129², 80 modes, cold).** PHASE0 §3.3 only checked grid at 40 modes on stable shapes; this
    re-checks where the paper headlines. **mid/stable (m_s 0.8–2.0): median |shift| 4.5%, max 4.7%** (confirms §3.3,
    129² gives slightly LOWER m_s). **But the marginal band is MORE grid-sensitive: near-marginal (m_s 0.30–0.45)
    median |shift| 10% (signed −10%); deep-marginal (m_s 0.11–0.30) median ~10% with at least one QUALITATIVE FLIP
    (a shape labelled m_s 0.26 at 65² is m_s 1.71 at 129² — the m_s→0 boundary is pathologically grid-sensitive).**
    **What this does and does NOT change:** ALL relative/gradient/design comparisons (the κ-constrained gains, the
    design-loop wins, the gallery) are computed at a FIXED 65² grid, so this systematic CANCELS exactly as the
    mode-truncation bias does — the contribution is unaffected, and the design gains (+0.5…+0.8) are ≫ the band, with
    design ENDPOINTS in the reliable mid/stable band. It DOES mean **absolute marginal m_s values carry a ~10%
    grid band (occasionally a stability-class flip at the very boundary)** — so the gallery hero start "m_s 0.193" is
    really 0.19±grid; the GAIN/crossing is robust, the absolute marginal endpoint is not. This SHARPENS (does not
    contradict) the existing story: m_s→0 absolute labels are the least trustworthy (the model already abstains there),
    so the paper leads with RELATIVE/design metrics, never absolute marginal m_s. **DIRECTLY VERIFIED — the gallery
    hero panel re-solved at 129²:** GAIN **+0.838 (65²) → +0.746 (129²)** and **κ stays locked (drift 0.011→0.008)**;
    the design endpoint (1.03→0.99) sits in the reliable band, only the absolute START shifts (0.19→0.24). So the
    design conclusion is grid-robust at the converged 80 modes.
  - **GENUINE 2nd DEVICE — SCAFFOLDED + VERIFIED, handed to the Mac for execution.** Initial assessment (FreeGS
    DIIID/TCV ship only active coils, no passive structures the Portone m_s needs) over-stated the blocker: the
    50-agent quality audit found **FreeGSNKE builds passives PROGRAMMATICALLY** (`build_machine.tokamak(passive_coils_
    data=...)`), so a credible 2nd device is scripting. Built **Device-C** (`experiments/device2_build.py`) = a
    HIGHER-ASPECT-RATIO machine by a single DOCUMENTED radial transform of the validated MAST-U build (isolates the
    aspect-ratio variable; far more defensible + tractable than an invented vessel). **Verified on Windows:** it builds
    a valid tokamak (12 active + 138 passive), raises A≈1.34→2.25 (escalatable to conventional ≈2.8), and its anchor
    inverse solves CONVERGE (κ≈2.0, ~30 iters). A **GO/NO-GO kill-gate** (`device2_killgate.py`) measures corr(κ,m_s)
    vs MAST-U's −0.875 → proceed only if κ is de-dominated, else report the honest scoping null. The κ-constrained
    differentiator above remains the DELIVERED single-machine result; the 2nd device (kill-gate-first, zero-shot
    transfer) is the in-flight thesis-closer.
- **Artifacts:** code `experiments/phase4_{kappa_reanalyze,kappa_pooled,gallery_lib,gallery_run,gallery_worker,
  gallery_analyze,gallery_shapes,pareto,leuer,ablations_light,ablations_train,power_run,power_worker,grid_check,
  figures}.py`; data `data/phase4_{kappa_reframed,kappa_pooled,gallery_results,gallery_summary,gallery_shapes,pareto,
  leuer,ablations_light,ablations_train,grid_check}.json` + `phase4_power{,2}_{results,setup,jobs}.json` +
  `phase4_{gallery,power,power2}_chunk_*.json`;
  figures `figures/phase4_{kappa_differentiator,gallery,pareto,leuer,ablations}.png`. Reused `phase25_kappa_lib`,
  `phase2_{dim_lib,model,train,data}`, `phase15_lib`, `data/phase2_dim_setup.json`, `data/phase25_kappa_{setup,results}.json`.
- **Caveats / corrections / things that surprised you:**
  - **(The κ-lock is not perfect — confound surfaced + quantified.)** A first design of the disabled-lever control
    PROJECTED the reduce-κ direction orthogonal to ∇κ and (correctly) annihilated it — but FLOATING-POINT residual +
    renormalization resurrected the κ-descent direction, so it still made progress (+0.165 on a test start). The honest
    lesson: holding κ to ±0.04 is NOT a perfect lock — because m_s is so κ-sensitive even a 0.04 κ reduction buys real
    m_s. Reframed the control as `kappa_nudge` (deliberately reduce κ within ±0.04) to QUANTIFY that confound (median
    +0.181) and SUBTRACT it; the surrogate's +0.386 excess is the clean learned-m_s contribution. The published
    gradient-free / fixed-lever baselines are κ-penalized identically, so they too have the residual freedom — the
    comparison is fair.
  - The marginal vs best-fixed-lever cell (the open weakness) was n.s. at n=10 (7/10, p=0.11). Per the quality
    directive, TWO disjoint power top-up batches (+12 then +24 starts, 396 runs, 0 errors) firmed MARGINAL to n=40:
    **30/40 = 75%, sign-test p=0.002, Wilcoxon p=0.0006 — now significant under BOTH tests** (and the fresh 20-marginal
    batch-2 cohort replicates it standalone, 16/20 = 80%, both p<0.02). The larger n ALSO promoted the pooled
    vs-oracle cell from borderline (p=0.052 at n=32) to significant (42/56, p=0.0007). Both prior caveats resolved.
  - The **learning curve implies the offline label cost could be ≪ 3254** (most of the accuracy is reached by ~650
    shapes; small residual gains continue to the full set, RMSE_log 0.130 at n=650 → 0.121 at n=2602), which would
    LOWER the Phase-3 ~340-query amortization break-even — BUT the design loop used the full-data surrogate and the
    GRADIENT/design quality at ~650 shapes is untested; do not claim the lower break-even without re-verifying.
  - Per-solve 80-mode cost solo = **22.7 s** (linearisation 78%); under 10-worker contention ~125–135 s (~5.5×,
    memory-bandwidth-bound, consistent with prior phases). The resume-safe atomic-write worker pool held (0 dups/errors).
- **Gate: PASS (honestly scoped).** ✅ the LEARNED m_s is now load-bearing at fixed κ — it beats EVERY realizable
  baseline (gradient-free 85% p=1e-4; best fixed lever 80% p=0.004), with only the unrealizable oracle tying it, and
  the gallery confound-decomposition shows ≈70% of the gain is genuine secondary-lever (not residual-κ) work; ✅ every
  headline claim has a baseline (Leuer/gradient-free/fixed-lever/oracle) + an ablation (ensemble/data/parameterization/
  modes/noise); ✅ cost/accuracy Pareto + design gallery delivered; every design 80-mode solver-confirmed. **Honestly
  scoped open items (NOT gate failures):** the learned-m_s win is at FIXED κ (on the UNCONSTRAINED single-ST manifold
  κ-geometry still ties it — the standing Phase-2/2.5b/3 finding); a genuine different-aspect-ratio 2nd device (where κ
  is not dominant) remains owed; marginal vs fixed-lever is now significant under BOTH tests at n=40 (sign p=0.002,
  Wilcoxon p=0.0006) and replicated out-of-sample. **Adversarial review:
  a 38-agent panel (6 dimensions × review + 2 independent verifiers) raised 32 findings;
  only 3 survived verification — all LOW severity and all the SAME labeling-consistency nit (report the marginal-vs-
  fixed-lever cell as Wilcoxon p=0.11, consistent with its siblings, not the sign-test p=0.34) — now FIXED. NO
  substantive overclaim survived (0 high, 0 medium): the realizable-baseline framing, the κ-lock confound
  decomposition, the gallery, and the Pareto/Leuer/ablation baselines were all judged sound.**
- **Advice for the next prompt:** (1) The cleanest remaining differentiator is the **genuine 2nd device** (different
  aspect ratio / vessel) — build FreeGSNKE passive-structure pickles, generate a forward dataset, retrain, and re-run
  the design comparison; on a machine where κ is not dominant the learned m_s should beat the κ-heuristic *unconstrained*.
  (2) Lead the paper's "learned-m_s is load-bearing" claim with the **κ-constrained realizable-baseline result + the
  gallery confound-decomposition** (≈70% secondary-lever), and now (firmed) "**beats EVERY baseline including the
  oracle**" at n=56. (3) [DONE — two disjoint top-up batches] power top-up pooled to **n=56 (marginal n=40)** via
  `phase4_kappa_pooled.py`: marginal vs-fixed-lever significant under BOTH tests (30/40, sign p=0.002, Wilcoxon
  p=0.0006) + replicated out-of-sample (16/20); pooled vs-gradient-free 50/56 (p<1e-4), vs-fixed-lever 45/56 (p<1e-4),
  vs-oracle 42/56 (p=0.0007). (4) Reuse `phase4_gallery_lib` (κ-constrained loop + GRecorder + `kappa_nudge` confound
  control) and `phase4_kappa_reanalyze` (realizable-baseline framing). (5) For Phase 5 release, the learning curve says
  a ~650-shape subset reproduces the accuracy — useful for a lightweight reproducible artifact (re-verify the gradient).
  (6) [DONE — handed to the Mac] the genuine 2nd device is now scaffolded + verified (`device2_build.py` builds a
  higher-A variant whose anchors converge; `device2_killgate.py` is the GO/NO-GO probe).

### Phase 5 — Device-C (documented higher-aspect-ratio radial transform of MAST-U): the UNCONSTRAINED learned-m_s vs reduce-κ test — 2026-06-26 — TIE (honest scoping result; **no pre-registered WIN**)
- **Where run:** the Apple-Silicon **M2 Pro Mac** (full A–D protocol; fresh venv built, pinned deps incl. freegs4e==0.13.1, **both A4 gates passed**: Leuer analysis 0.917→0.984 reproduced; a real 80-mode solve runs). All 5 BLAS vars pinned (incl. `VECLIB_MAXIMUM_THREADS` for Accelerate). **Cross-platform reproducibility datapoint:** a mid-regime MAST-U shape re-solved at 80 modes on the Mac gave m_s=**0.999924** vs the stored Windows **0.999939** (Δ=**−0.0015%**, within the floor); within-Mac m_s is bit-reproducible (re-solves Δ=0.000%, verified surviving an overnight sleep/resume).
- **Goal:** close the standing hedge — at FIXED κ the learned m_s is load-bearing (Phase 4), but UNCONSTRAINED on the single MAST-U ST κ-geometry ties it. Does the learned m_s beat the reduce-κ heuristic UNCONSTRAINED on a higher-aspect-ratio device? **Pre-registered** (`data/phase5_{killgate,design}_prereg.json`): WIN = two-sided Wilcoxon p<0.05 AND out-of-sample replication; else TIE — both honest/publishable. Metric = design-loop lever attribution by regime, NEVER prediction accuracy.
- **Did:** new code `experiments/device2_*` (kill-gate anchors/worker/run/analyze; gen worker/run; shapemap + surrogate trainers; design lib/setup/worker/run/analyze/figures; portone + endpoint cross-checks; deepdive). Protocol locked: serialized Device-C machine, OMP=1, COLD forward solves, 65×65, **expanded grid Rmax=2.8** (larger-R plasma), 80-mode confirmation every design step.
  1. **Built Device-C** = a DOCUMENTED higher-aspect-ratio radial transform of the validated MAST-U build (`--r0_old 0.9 --r0_new 1.6`, **scale=1.0**: every coil/passive/limiter/wall coord shifted +0.700 m in R; **Z-geometry, passive Z-extent ±2.22 m, and elongation envelope κ≈1.96 are BIT-IDENTICAL to MAST-U** — an aspect-ratio VARIANT, *not* an independent machine). Limiter A 1.34→2.25; plasma A_median≈**2.97** (conventional-A *plasma* on an ST machine *frame*).
  2. **KILL-GATE** (parallel thread-pinned pool, **n=330** converged diverted Device-C equilibria, 40-mode probe): **corr(κ, log m_s) = −0.811 [95% CI −0.849,−0.764]** (MAST-U −0.875). The PRIMARY de-domination criterion |corr|<0.75 was **NOT met** (κ still dominant; only a 0.06 drop); **GO fired solely via the SECONDARY rival-lever rule** (squareness sq_lo |corr|=0.654 = 0.81×κ's). Independent Portone eigenvalue cross-check (3 algebraically-distinct recomputes) = **0.000%** vs FreeGSNKE on Device-C (one positive margin each) — convention + labels verified.
  3. **UNCONSTRAINED design comparison** (κ free; 40 stratified starts = 24 main + 16 **disjoint replication**, ~50/50 marginal/mid; 3 methods × 40 = 120 resume-safe runs/framing; budget 18; **every step 80-mode solver-confirmed**; top-12 PCA Device-C control design space, box widened to contain starts; equal budget, each method ranked by its OWN objective, accept only on TRUE m_s improvement). Two pre-registered framings: **zero-shot** (MAST-U surrogate transferred + Device-C ShapeMap) and **retrained** (Device-C-native surrogate, held-out log-m_s R²=0.946; Device-C ShapeMap R²: κ 0.996, squareness 0.976, mean 0.939). Design endpoints re-validated by the independent Portone recompute (**0.000%**, both framings; each stored best_ms re-solves bit-exactly).
- **Result (regime-resolved per the pre-registration; pooling reported only as a cancellation artifact, NOT evidence):**
  - **No pre-registered WIN → TIE** (= failure to demonstrate the registered WIN; *not* proven equivalence — CIs are wide, no TOST).
  - **Retrained (the FAIR test) = a regime CROSSOVER, not a flat tie:** **MID (n=18): 13/18=72% is the POOLED (main+replication) number, and its sub-0.10 p=0.060 DEPENDS on pooling the two DISJOINT cohorts (which the pre-registration said NOT to do) — per-cohort it is NON-significant in BOTH (main mid 8/11 p=0.15; replication mid 5/7 p=0.30); median best_ms 1.42 vs 1.19** — an exploratory hint, NOT a replicated or significant result. **MARGINAL (n=22): reduce-κ favored 8/22=36% (p=0.12), median 0.41 vs 0.65; NOT replicated (main 2/13=15% vs rep 6/9=67%)** — inconclusive/noisy. **POOLED 21/40=52% (Wilson [0.375,0.671], p=0.92) is the arithmetic cancellation of the +mid edge against the −marginal deficit — represents no regime; not cited as evidence.**
  - **Zero-shot = a pre-registered NEGATIVE result** (the "learned physics, not geometry" test): the MAST-U surrogate LOSES to reduce-κ 9/40=22% (p=0.0002; replicated 3/16=19%, p=0.0017). The learned m_s preserves cross-device RANK (Spearman 0.79) but is **NOT a transferable absolute model** (log-R²=−6.4, ~4× under-prediction). The loss is driven by **local gradient-DIRECTION degradation** (zero-shot-vs-retrained gradient cosine **0.63**, marginal 0.50), NOT the 4× miscalibration (the normalized-gradient loop is provably invariant to monotone rescaling). Retraining repairs the response surface (costing **~704 Device-C 80-mode solves — the per-device budget amortization was meant to avoid**) and recovers MID (17%→72%) but NOT marginal (27%→36%).
  - **The differentiable-LOOP MECHANISM — not the learned model — beats gradient-free:** at the matched 18-solve budget BOTH the learned-m_s gradient (surrogate vs CMA 30/40=75%, p<1e-4) AND the non-learned geometry-only reduce-κ gradient (reduce-κ vs CMA 31/40=78%, p<1e-4) beat CMA. This is a **sample-efficiency** result vs a default-popsize CMA (~1.5 generations at this budget), not absolute superiority of gradients over derivative-free search. Learned-m_s credit stays confined to the FIXED-κ Phase-4 result.
  - Unconstrained, the retrained learned-m_s gradient points largely **ALONG** the reduce-κ direction (cosine 0.78; mid 0.87 > marginal 0.71) — even for the learned m_s, reducing κ is most of the ascent direction ⇒ κ-dominance.
- **Artifacts:** code `experiments/device2_{build,killgate,killgate_anchors,killgate_worker,killgate_run,killgate_analyze,gen_worker,gen_run,assemble,shapemap_train,surrogate_train,design_lib,design_setup,design_worker,design_run,design_analyze,design_figures,portone_crosscheck,endpoint_crosscheck,design_deepdive}.py`; data `data/device2_{killgate,probe.parquet,anchors.pkl,shapegen_all.parquet,ds80_all.parquet,design_setup,design_jobs,design_{zeroshot,retrained}_summary,design_deepdive,portone_crosscheck,endpoint_crosscheck_*}.json` + `phase5_{killgate_prereg,design_prereg,zeroshot_transfer_diag,A4_mac_crossplatform}.json` + models `data/phase2_models/{shapemap_C,surrogate_C}.pt`; figures `figures/phase5_design_{zeroshot,retrained}.png`.
- **Caveats / corrections (Phase-D scientific review = 24-agent panel × 5 dimensions; 19 raised → 15 survived; ALL incorporated above):**
  - Pooled win-rate is a regime cancellation → report regime-resolved only (pre-reg rule).
  - Device-C shares MAST-U's exact vertical-stability hardware (scale=1.0) → κ-dominance shown **robust to aspect ratio in isolation** (A 1.6→2.97), **NOT general** across structurally distinct designs (different vessel/passive topology untested; m_s scale is strongly set by the passive structure). "Genuine second device" → "documented aspect-ratio transform of MAST-U."
  - The kill-gate PRIMARY de-domination criterion was never met (|corr|=0.811) → the design comparison ran on a still-κ-dominated device; the rival-lever GO was an optimistic false-positive. The pre-registered escalation ladder **WAS run (2026-06-28) and hit a CONVERGENCE WALL before κ de-domination** — r0_new 1.9: 0/18 anchors converged; r0_new 1.75: first anchor stalled — i.e. a pure radial transform cannot hold a converged diverted plasma at conventional A with the MAST-U coil set (`data/phase5_escalation_ladder.json`), which reinforces that a genuine κ-de-dominated test needs structurally different hardware. A structurally distinct device was not run.
  - The marginal loss is partly an **optimizer/feasibility confound**: the surrogate's raw gradient drifts off the valid/diverted manifold more in marginal (invalid steps 11.5 vs reduce-κ 9.2) and the validity-blind one-solve-per-step line search wastes them → the marginal split is not cleanly attributable to lever quality.
  - "TIE" = failure-to-reject at n=40 (Wilson [0.375,0.671]), not equivalence; point estimate slightly favors the surrogate (median Δ +0.015) — not downgraded to a loss. The κ-robustness claim rests on the well-powered kill-gate corr (n=330), not the win-rate tie.
  - Budget is not the binding constraint (gradient methods plateau by ~solve 8; reset-on-stall is a no-op restart) → the TIE is not a small-budget artifact, but IS conditional on this deterministic local-ascent optimizer; a validity-aware/multi-start/trust-region optimizer (upgrading BOTH levers to keep lever attribution fair) is untested.
- **Gate: TIE (no pre-registered WIN).** Honest scoping result, consistent with + extending the standing MAST-U finding (Phase 3/4: unconstrained, reduce-κ ties the learned m_s). The hoped-for thesis-closing WIN did not materialize. The delivered contribution is unchanged and stated honestly: **(a)** the FIXED-κ learned-m_s differentiator (Phase 4) and **(b)** the amortized solver-confirmed differentiable design LOOP (mechanism, not model, beats gradient-free at matched budget). Every design step + endpoint 80-mode solver-confirmed and Portone-cross-checked (0.000%).
- **Advice for the next prompt:** (1) For a real cross-device test, build a **structurally distinct** machine (different passive-conductor Z-extent/topology + lower elongation envelope), not a radial transform — m_s scale is set by the passive structure, so only that probes the real question (the escalation ladder r0_new 1.9/1.75 was already tried and hit a convergence wall — `data/phase5_escalation_ladder.json` — so a radial transform won't reach |corr|<0.75). (2) The MID-regime surrogate edge (72%, p=0.06, replicated direction) is the one live thread — but resolving it needs a **validity-aware optimizer** (within-iteration fallback on invalid candidates / OOD pre-filter) to remove the marginal feasibility confound, then a Phase-4-style power top-up; only then could a regime-resolved learned-m_s win be claimed. (3) The learned m_s is NOT a transferable absolute model (zero-shot log-R²=−6.4) → cross-device use needs per-device retraining (~700 solves), undercutting the zero-shot amortization story; state this honestly. (4) Reuse `device2_design_lib` (unconstrained loop) + `device2_gen_*` (Device-C sampling) + the cross-checks. (5) Lead the paper with the FIXED-κ Phase-4 differentiator + the amortized-loop mechanism; report Phase 5 as the honest second-device **TIE / κ-robustness** scoping result.

### Phase 5 follow-on — robust/worst-case design attempt (FAILED, instructive) + QA-audit hardening — 2026-06-28
- **Goal:** test whether the surrogate's cheap differentiable sensitivity enables a capability reduce-κ + gradient-free search structurally lack — designing for WORST-CASE m_s under operational uncertainty (coil 3% / paxis 5% / Ip 3% / fvac / α drift).
- **Did:** computed `nominal` (max surrogate point m_s) and `robust` (max surrogate mean−λ·std over a K=64 perturbation ensemble) designs **SURROGATE-ONLY** (`device2_robust.py`, NO solver in the loop), then true-solver-confirmed each design + reduce-κ (Phase-5 retrained endpoint) under M=16 80-mode perturbations (`device2_robust_{worker,run,analyze}.py`; 120/120 jobs).
- **Result — FAILED / UNTESTABLE, recorded honestly:** the surrogate-only optimization drove BOTH the nominal and robust designs OFF the diverted manifold — true 80-mode center m_s = 0 for nominal in 32/40 starts and robust in 29/40; the unconditional true worst-case (failures=0) is floor-saturated at 0 for ALL three designs ⇒ robust-vs-nominal is **UNTESTABLE** (Wilcoxon p=NaN, all paired diffs 0) — **NOT a tie / no-benefit**. Even conditioned on convergence the robust design is NOT better than nominal (worst-among-converged POOLED median nominal 0.863 > robust 0.717). The only on-manifold design is reduce-κ (conv-frac 0.50 vs nominal/robust 0.28/0.25; center 0.84 vs 0.0) — but that is apples-to-oranges (reduce-κ came from the Phase-5 CONFIRM-IN-LOOP loop; nominal/robust got zero confirm-in-loop).
- **Lesson (the project's core one, now quantified):** a surrogate optimized WITHOUT the solver in the loop hallucinates off-manifold optima (~25% valid) while the confirm-and-reject loop stays on-manifold (~50%). The robust capability is **NOT refuted — it was never validly tested**; a correct demo must REDO it as a **CONFIRM-IN-LOOP robust optimizer** (propose by surrogate worst-case score, confirm validity+m_s each step, reject off-manifold), with σ recomputed at the current control and the graded worst_converged/conv_frac metrics (P(m_s>0.15) is non-binding here ≈ conv_frac). Top next-step for the other computer.
- **QA AUDIT (24-agent panel × 5 dimensions; 19 raised → 10 confirmed; ALL FIXED):** (1) robust headline reported UNTESTABLE not a tie [this entry]; (2) **filename collision fixed** — `device2_killgate.py` STARTER now writes `*_starter` paths so it can't clobber the canonical analyzer artifacts; (3) robust docstrings de-overclaimed; (4) robust σ-at-u0 vs per-design σ documented as a redo fix; (5) P(m_s>0.15)≈conv_frac double-count caveated; (6) mid-regime 13/18 relabeled a POOLED exploratory hint (per-cohort n.s.); (7) BLAS-pin guard added to standalone solve scripts (`_blas_guard.py`); (8) Device-C grid now read from the frozen anchors meta (single source of truth) in the design/robust/cross-check workers; (9) escalation ladder synced (it WAS run → convergence wall); (10) "genuine second device" → "documented radial transform of MAST-U" in headline spots. Re-verified: every ledger number matches its json; a probe sample re-solves bit-exact; all 24 device2_*.py compile.
- **Artifacts:** `experiments/device2_robust{,_worker,_run,_analyze}.py`, `_blas_guard.py`; `data/device2_robust_{designs,summary}.json`, `data/device2_robust_results/`, `data/phase5_escalation_ladder.json`.
- **Gate:** robust-capability claim NOT established (untestable); the honest delivered contribution is unchanged. The confirm-in-loop quantification + audit hardening stand.

### Tier 1 — real-data connection (original-MAST EFIT): a quantified DOMAIN GAP — 2026-06-30/07-01 — PARTIAL (headline robust; test-b heavily caveated)
- **Goal:** connect the synthetic-only project to REAL open tokamak data ("crack the synthetic ceiling"). Pre-registered + git-frozen BEFORE any re-solve (`TIER1_PREREG.md`, commit 09df502).
- **Premise correction (verified):** FAIR-MAST (open S3 Zarr, no login) hosts ONLY original **MAST** (M5–M9), **NOT MAST-U** (UKAEA-gated). So "real data" = a DIFFERENT (predecessor) ST device. Framing (5-agent grounding workflow): emulator-OOD / shape-coverage / solver-consistency **STRESS-TEST**, NOT experimental validation; any re-solved m_s is model-vs-model (MAST shape on the MAST-U vessel; m_s is passive-structure-dominated). Reference method: arXiv:2407.12432. Data via **xarray** (raw zarr misses fields): stored `lcfs_r/z` + EFIT `elongation`/`li`/`betap`/`q_95`/profiles. 357 quality flat-top slices / 93 complete-efm M9 shots; frozen deterministic selection = 60 slices / 40 shots, κ∈[1.35,2.50] (farthest-point; spans below+above training).
- **TEST (a) COVERAGE = the robust headline (artifact-free geometric κ/δ/aspect, same-code both sides, no re-solve):** real MAST shapes are **100% OOD** of the synthetic MAST-U training cloud — reproduced 4 independent ways (Mahalanobis >99th & >95th pct; convex-hull 0/357 inside; box 99.7% outside ≥1 axis; diag-covariance) ⇒ NOT a covariance artifact. **Attribution (leave-one-out):** a device-level **aspect-ratio** offset (single-axis OOD 0.98; real A bulk 1.30–1.45, median 1.37, full range to 1.85 vs synthetic 1.52–1.78 — MAST≠MAST-U, expected) **PLUS an independent triangularity mismatch (δ single-axis OOD 0.84; κ+δ WITHOUT aspect still 89% OOD; δ real 71% below training)**. κ partially overlaps (68% in-range, tails both sides). ⇒ the synthetic MAST-U distribution does NOT cover real (lower-A, lower-δ) MAST — a genuine, quantified synthetic↔real domain gap. `figures/tier1_coverage.png`, `data/tier1_coverage.json`.
- **TEST (b) EMULATOR SELF-CONSISTENCY (shape-anchored inverse re-solve on the MAST-U model; ConstrainBetapIp @ real βp/Ip/fvac; m_s at 40 & 80 modes) — HEAVILY CAVEATED:** FUNNEL = **60 selected → 59 resolved → 55 converged (93%) → 31 with a defined solver m_s → 30 enter test (b)**. The m_s-undefined attrition (empty linearised stability spectrum, npos_80=0) is **non-random in κ** (has-m_s: κ<1.6=0.25, κ1.6–2.0=0.81, κ≥2.0=0.25) ⇒ the b-set collapses to a **MID-κ subset (median 1.83; only 2/8 high-κ OOD-tail slices survive)** — the OOD tail (the whole motivation) is largely ABSENT from (b). On the 30: surrogate-vs-solver80 |rel log m_s| median **0.61** (p90 1.74); solver's OWN 40↔80 ambiguity balloons to median **0.21** on these OOD shapes (vs ~0.14 synthetic); only **23%** land within their own ambiguity. **Do NOT strongly claim "point-m_s does/doesn't transfer": (i) 0/55 slices pass the PRE-REGISTERED profile-match gate (|dβp|,|dli|≤0.1) — test (b) is OUTSIDE the pre-registered regime (completion 2026-07-15: the profile-match gate (|dβp|,|dli| ≤ 0.1) was unpassable by construction because the profile-ingestion pipeline inflated the enforced βp by ~3.4–5.3×, so 0/55 reflects that pipeline artifact, not the shapes; the VOID scoping already applied stands.); (ii) ConstrainBetapIp βp≠poloidalBeta2 so the re-solve βp is inflated (median 1.25 vs real EFIT 0.28, ~4.5×; li 1.22 vs 0.94) — the surrogate is fed OOD βp/li, an ARTIFACT; (iii) 65% of re-solves come out LIMITED (real MAST is diverted) — model-vs-model artifact geometry; (iv) an **IN-DISTRIBUTION CONTROL** (identical ConstrainBetapIp re-solve pipeline on 20 synthetic in-distribution shapes; `tier1_indist_*.py`, `data/tier1_indist_summary.json`) shows the pipeline's own floor: even in-distribution, ConstrainBetapIp inflates poloidalBeta2 ~4.85× (median re_betap 1.12 vs true 0.23) and the surrogate-vs-solver residual is **median 0.185** (vs the surrogate's NATIVE in-dist accuracy 0.032). So of the real-shape 0.61, **~0.19 is the pipeline/βp artifact (control floor) and the remaining ~0.42 IS genuine real-shape OOD degradation** — real does exceed the in-dist pipeline floor ~3.3×, so there IS real OOD degradation, but the absolute residual is artifact-inflated; report the OOD EXCESS (~0.42), not the raw 0.61. (corr(resid, geom-OOD)=+0.30.)** The ONE robust (b) signal: surrogate epistemic uncertainty median **0.71 vs an in-distribution baseline 0.028 (~25×)** — a **binary domain-shift flag** (uniformly elevated; does NOT rank OOD severity, corr≈−0.05) that supports Tier-3 abstention-by-threshold.
- ERRATUM (2026-07-15): the βp inflation is a profile-ingestion pipeline artifact (a βp-definition mismatch compounded by the profile-target handling), quantified by the in-distribution control below. Numbers unchanged.
- **TEST (c) FIDELITY (55):** re-solved vs EFIT κ **|dκ| median 0.12** (moderate; cross-machine + mostly limited); boundary **ζ median 16 cm** (paper same-machine <1 cm); |dli| 0.35, |dβp| 0.98.
- **ADVERSARIAL REVIEW (28-agent workflow, 3 phases; 22 raised → 21 survived incl. many no-defect CONFIRMATIONS, 4 tagged high):** the two positives are robustness-CONFIRMED (100% OOD real; 25× epistemic real). The 3 substantive HIGH edits — (H1) scope the transfer verdict to the mid-κ subset via the eigenvalue funnel; (H2) soften "does not transfer" (no in-dist control isolating the βp artifact); (H3) disclose the 0/55 pre-reg profile-match gate (completion 2026-07-15: the profile-match gate (|dβp|,|dli| ≤ 0.1) was unpassable by construction because the profile-ingestion pipeline inflated the enforced βp by ~3.4–5.3×, so 0/55 reflects that pipeline artifact, not the shapes; the VOID scoping already applied stands.) — plus M1 (corrected βp numbers) and M2 (δ is the substantive driver, aspect device-trivial) — **all incorporated above**. No headline falsified.
- **Artifacts:** `experiments/tier1_{lib,discover,coverage,resolve_worker,resolve_run,analyze}.py`; `data/tier1_{selection,pool,shotlist,coverage,analysis}.json` + `tier1_resolved/`; `figures/tier1_{coverage,consistency}.png`; `TIER1_PREREG.md`.
- **Gate — PARTIAL / honest:** the ROBUST positive = the **quantified synthetic↔real domain gap (100% OOD, δ+aspect)** + the surrogate's **25× uncertainty domain-flag**. The NEGATIVE (honestly owned): the surrogate's point m_s does not usefully transfer to real-derived shapes, and cross-machine re-solve reproduces real shapes only moderately (dκ 0.12, 65% limited, βp artifact) — test (b) is model-vs-model on a mid-κ subset, outside the pre-registered profile regime. Tier-1 is a **go/no-go gate + motivation** for device-specific follow-up work, not a demonstrated capability. Do NOT claim validation / MAST-U / real-m_s / transfer / "anchoring" as a win (see `TIER1_PREREG.md` must-not-claim).
- **Advice for the next prompt:** (1) *[device-specific follow-up direction withheld; reserved for a future release]*. (2) [DONE this session] the in-distribution control ran (20 synthetic shapes): pipeline floor median 0.185 vs real 0.61 ⇒ ~0.19 artifact + ~0.42 real OOD excess (test-b degradation is real above the floor, but βp-artifact-inflated). (3) Tier-3 abstention is supported by the 25× uncertainty domain-flag (threshold, not distance-calibrated).

*[entry withheld; reserved for a future release]*

*[entry withheld; reserved for a future release]*

*[entry withheld; reserved for a future release]*

### Tier-1 firm-up: in-distribution control — 2026-07-01 — DONE (resolves review H2)
*(Backfilled 2026-07-13 by the Stage-0 S0.2 ledger-integrity pass; dated by its commit `44bbc27`, 2026-07-01. Numbers from `data/tier1_indist_summary.json` — recomputed at n=24 on 2026-07-13 after the Stage-0 pass completed the 4 missing chunks; the version committed at `44bbc27` had n=20.)*
- **Goal:** adversarial-review H2 asked whether the real-shape surrogate-vs-solver residual (~0.61) is genuine shape-OOD degradation or a `ConstrainBetapIp` re-solve PIPELINE artifact. Answer it with an in-distribution control: run the IDENTICAL re-solve pipeline on synthetic in-distribution shapes and measure the pipeline's own floor.
- **Did:** `experiments/tier1_indist_{worker,run}.py` — 24 deterministic in-distribution `dataset_v1_80q` rows (mid-κ 1.58–2.10, mirroring the real b-set band; m_s defined): forward-solve each to its true LCFS, then the identical shape-anchored inverse re-solve (`tier1_resolve_worker.resolve_slice`, `ConstrainBetapIp` at the shape's own descriptors-βp), then surrogate-vs-solver-80 residual. LOCKED protocol (BLAS=1, serialized MAST-U machine, cold, 65×65; m_s at 40 & 80 modes per the Tier-1 worker).
- **Result (all numbers from `data/tier1_indist_summary.json`, n=24 recompute):** **n=24, n_ok=23**. Even fully in-distribution, the pipeline inflates poloidalBeta2: re-solve βp median **1.151** vs true **0.2387** ⇒ **betap_inflation 4.82×**. Surrogate native in-dist accuracy floor: |rel log m_s| median **0.034**; but through the re-solve pipeline the surrogate-vs-solver-80 residual floor is **0.2109** (median over the 17 chunks with both m_s defined). **Decomposition (computed, not copied):** real-shape residual median **0.614** (`data/tier1_analysis.json`, `consistency.median_resid` = 0.6136) − in-dist pipeline floor **0.211** = **OOD excess ≈ 0.40** (real ≈ 2.9× the pipeline's own floor). So the real-shape residual IS artifact-inflated (the pipeline contributes ~0.21 even in-distribution), AND there remains a genuine real-shape OOD excess (~0.40) above that floor.
- **Artifacts:** `experiments/tier1_indist_worker.py`, `experiments/tier1_indist_run.py`, `experiments/tier1_indist_summarize.py` (Stage-0 reconstruction of the summary aggregation, validated to reproduce the committed n=20 values exactly), `data/tier1_indist/000–023.json` (24/24), `data/tier1_indist_summary.json`.
- **Caveats:** (i) at the original commit the summary was n=20 (missing chunks 000/006/012/018 = worker chunk 0; completed 2026-07-13 by the Stage-0 pass — all 4 status=ok) with floor 0.1849 and inflation 4.85; the n=24 recompute (floor **0.2109**, inflation **4.82**) supersedes the "~0.19 artifact + ~0.42 OOD excess" phrasing in the Tier-1 entry above → the current decomposition is **~0.21 pipeline floor + ~0.40 OOD excess** (both bands within ±20% of the n=20 values; the Tier-1 conclusions are unchanged). (ii) This control characterizes the in-distribution re-solve pipeline only — it makes no cross-machine attribution claims. (iii) 1/24 control shapes did not converge (n_ok=23); 6 of the 23 ok chunks have no defined solver m_s (the 40/80-mode spectrum was empty), so the residual floor is a median over 17.
- **Gate:** DONE — review H2 resolved: the real-shape 0.61 residual is PARTLY pipeline artifact (in-dist floor ~0.21, βp inflation ~4.8×) and PARTLY genuine OOD degradation (~0.40 excess); report the excess, never the raw 0.61, as OOD degradation.
- **Advice for the next prompt:** a cleaner future re-solve should target `poloidalBeta2` directly (or fit li) to remove the ~4.8× βp inflation; the ~25× epistemic domain-flag headline is unaffected by this control (it is a prediction-time signal, not a re-solve product) — its in-dist baseline is persisted in `data/tier1_epistemic_baseline.json` (0.0281; real/baseline = 25×, 2 s.f., computed).
- ERRATUM (2026-07-15): the βp inflation is a profile-ingestion pipeline artifact (a βp-definition mismatch compounded by the profile-target handling), quantified by the in-distribution control below. Numbers unchanged.

### Stage-1 E2 — trivial-baseline ablation — 2026-07-21 — E2-CONFIRM
- **Goal:** does a trivial input-space OOD detector reproduce the surrogate's epistemic abstention flag on the real-MAST 60-slice shapes? (§2.2 pre-registration; PROMPT SUPPLEMENT E2-S1, bundle A8.) No solves.
- **Did:** built `experiments/e2_baselines.py`. Split from `dataset_v1_80q.parquet`'s own `split` column (train **2602** / held-out val **465**; sha256 pin `2299…6ec4` verified). All standardization/PCA/covariance/thresholds from TRAIN ONLY. D19-b solver-free feature completion: the 4 squareness features via the freegs4e `Equilibrium.squareness` (Luce-2013) code path on the raw LCFS (shim exposing `separatrix()`), the 5 gap features via `phase15_lib._seg_dist(LCFS, MAST-U limiter)` from the serialized pickle. **D19-b validation gate PASSED on all 60 slices** — geometry-derived features bit-exact vs persisted (max |Δ| = 0.0 for all 7), EFIT scalars verbatim-exact, all 9 completed features finite. Three trivial detectors (Mahalanobis full-cov `pinv`, per-feature range, convex-hull in d=6 PCA) + the ensemble epistemic comparator.
- **Result (numbers from `data/e2_baselines.json`):** all three trivial detectors **TPR = 1.00 (60/60)** on the real slices. FPR on held-out val: Mahalanobis **1.08% (5/465)**, range **1.08% (5/465)**, hull(PCA) **19.78% (92/465)**. Qualifying (TPR≥0.95 ∧ FPR≤0.05): **Mahalanobis + range** → **E2-CONFIRM** (hull fails FPR alone; the rule is ANY-detector). Comparator epistemic flag: **TPR = 1.00 (55/55)** ok slices, threshold = val p99 **0.16694** (val median 0.03059; S0.2 baseline 0.02811 in-band cross-check). Thresholds: Maha p99 = **9.8975**; PCA **d=6, 93.04%** variance; cov rank **19/20** (`pinv` required — `delta ≡ (delta_upper+delta_lower)/2`). Agreement matrix (4 flags × 55 ok slices): **all four flags agree on all 55** (every real slice flagged OOD by every flag). Hull Delaunay-vs-LP cross-check: 0 disagreements (val 92/92, real 60/60).
- **Artifacts:** `experiments/e2_baselines.py`, `data/e2_baselines.json`, `figures/e2_baselines.png`.
- **Caveats / disclosures (in the artifact):** gaps are the real MAST shape measured against the MAST-U wall (Tier-1 cross-machine framing); li/betap keep the known EFIT-vs-`descriptors()` definitional caveat. Comparator defined on 55 of 60 (5 excluded: `30380_68` never resolved; `28784_39/28796_90/28868_35/29180_42` unconverged). Trivial detectors scored on all 60. `confirmatory: false`. Bit-identical re-run verified (JSON sha256 stable).
- **Gate:** E2-CONFIRM per §2.2 decision rule applied literally — the abstention flag is reframed as a UQ consistency check agreeing with input-space OOD; unique value deferred to the pre-registered Stage-2 in-hull test. (VERIFY-S1.2 still owed, independent agent.)
- **Advice for the next prompt:** the D19-b shim (LCFSShim binding `Equilibrium.squareness`/`_separatrix_metrics` to the raw LCFS) reproduces persisted geometry bit-exact and is reusable for any future real-slice descriptor completion — no re-solve needed. The hull detector's high in-dist FPR (~20%) is generic to ≥6-D hulls, not a bug; do not read it as a detector failure. All dry-run pins in `E2_E3_TURNKEY §5–6` reproduced exactly.

<!-- ───────────────────────── TEMPLATE: copy for each new phase ─────────────────────────
### Phase <N> — <name> — <YYYY-MM-DD> — <PASS / FAIL / PARTIAL>
- **Goal:** <one line>
- **Did:** <what you ran/built; the protocol used — grid, OMP, mode count, tolerance>
- **Result (numbers, resolved not aggregated; state vs the noise floor):**
- **Artifacts:** <files in data/ figures/ experiments/ + any doc updated>
- **Caveats / corrections / things that surprised you:**
- **Gate:** <did the phase gate pass? evidence>
- **Advice for the next prompt:** <concrete, specific tips, gotchas, parameter values, what to reuse, what to avoid>
──────────────────────────────────────────────────────────────────────────────────────── -->

---

# PART C — Open questions / risks carried forward
- [x] Independent Portone m_s cross-check (label convention) — **DONE Phase 0** (0.000% match; convention verified).
- [x] Formally report the m_s noise floor — **DONE Phase 0** (within-config bit-exact; cross-thread median 4.9%/max
      11.4%, removed by OMP=1). [x] **Reduce** it at the root via forward sampling — **DONE Phase 1.5**
      (forward labels reproduce inverse m_s to 0.000% and are bit-reproducible; `data/dataset_v1.parquet`, 3298 samples).
- [x] **m_s mode-convergence** — **DONE Phase 2**: converged count = **80 modes** (|80−138|≈0.4%); the 40-mode
      labels are low by median −13.5% (regime-dependent, marginal −27%). [x] **Re-label dataset_v1 at 80 modes** —
      **DONE Phase 2.5** (`dataset_v1_80.parquet`, 3254 shapes; +q95 → `dataset_v1_80q.parquet` Phase 2.5b).
- [x] Does the gradient's advantage over heuristics emerge in higher-D? — **Phase 2 + 2.5b + 4**: gradient
      **robustly beats gradient-free search** (κ-constrained, n=20: 85%, Wilcoxon p=1e-4) and the gap GROWS with
      dimension. **Phase 4 (re-framed honestly):** at fixed κ it ALSO beats the best **realizable** single FIXED lever
      (16/20=80%, Wilcoxon p=0.004); only the *unrealizable* per-start ORACLE best-of-8 lever ties it (15/20, p=0.064).
      So the earlier "within noise vs the best heuristic" was vs an unrealizable oracle. [x] MARGINAL cell FIRMED to
      **n=40** (two disjoint top-ups): vs fixed-lever 30/40=75%, sign p=0.002 + Wilcoxon p=0.0006 (significant under
      BOTH + replicated out-of-sample); pooled n=56 the surrogate beats even the ORACLE 42/56 p=0.0007. [~] Beat the
      heuristic UNCONSTRAINED — **TESTED Phase 5 on Device-C (higher-A transform): TIE, no WIN** (retrained learned m_s
      does not beat reduce-κ unconstrained; pooled 52% p=0.92; mid-regime hint 72% p=0.06 replicated-direction only).
      κ-dominance is robust to aspect ratio. A structurally distinct device (different passive topology) is still owed.
- [x] **The solver-confirmed design loop (the contribution)** — **DONE Phase 3**: gradient design loop reaches m\*=1.0
      in **~4× fewer expensive serial solves than gradient-free** (median 4 vs 13.5, faster 17/20, Wilcoxon p=0.0009;
      robust to a FAIR-CMA control = not a starvation artifact), stabilizing marginal ST plasmas in median 6 solves,
      every step 80-mode solver-confirmed. Honest scope: the win is the **gradient/κ-direction not the learned m_s**
      (heuristic ties), **serial-only** (within noise under parallel pop-eval), break-even ~340 queries. [x] Make the
      LEARNED m_s the load-bearing lever — **DONE Phase 4 (at FIXED κ)**: κ-constrained gallery (κ locked 20/20),
      learned-m_s beats every realizable baseline (gradient-free 85% p=1e-4, best fixed lever 80% p=0.004; oracle ties),
      gain decomposed into +0.181 residual-κ confound vs **+0.386 learned-m_s secondary-lever excess (~70%)**.
      [~] UNCONSTRAINED — **Phase 5 (Device-C, A≈2.97): TIE** (retrained learned m_s ties reduce-κ, pooled 52% p=0.92;
      beats gradient-free as a *mechanism*, not specifically the learned model). κ-dominance robust to aspect ratio;
      generality across structurally distinct devices NOT established (Device-C shares MAST-U passive topology by construction).
- [x] **Rigor layer** (Pareto, Leuer, ablations, robustness, adversarial review) — **DONE Phase 4**: surrogate m_s+grad
      ~8,100× faster than a true solve at log-R² 0.971; Leuer rank ceiling 0.92→0.98; shape-param κ-only RMSE_log
      0.442→0.122 full; robust to input noise; 38-agent panel (32 raised, 3 survived, 0 high/med — a labeling nit, fixed).
- [x] **Gradient ascent at the converged 80 modes** — **DONE Phase 2.5b**: design-regime 11/20 = 55% (Wilson
      0.34–0.74, spans chance) — WEAKER than the 40-mode 75%; a usable direction, not a Jacobian.
- [x] **q95 feature + a TRUE univariate high-κ extrapolation split** — **DONE Phase 2.5b** (q95 sane but κ-mediated;
      tail shows a modest mid-regime penalty, no collapse; both splits documented in DATASET.md).
- [x] Calibrated uncertainty near m_s→0 — **DONE Phase 2** (post-hoc recalibrated; widens at the marginal boundary
      + sparse stable tail).
- *[entry withheld; reserved for a future release]*
- [ ] **Reproducible open release** (clean package, env lock, public dataset+code) — Phase 5. (Learning curve: a
      ~650-shape subset reproduces the surrogate accuracy — a candidate lightweight artifact, re-verify the gradient.)
- [ ] Anti-scoop re-scan before writing (UKAEA/FreeGSNKE/Kolemen/EPFL 2026) — Phase 6.

---

*[entry withheld; reserved for a future release]*

---

*[entry withheld; reserved for a future release]*
