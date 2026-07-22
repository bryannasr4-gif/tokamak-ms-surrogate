# Phase 0 — Label Trust & Numerical Protocol Lockdown — **GATE: PASS** (2026-06-19)

The locked numerical protocol, the measured m_s/γ reproducibility floor, and the verified
m_s definition/convention. Everything downstream (Phase 1.5 data, Phase 2 surrogate, Phase 3
design loop) MUST use this protocol and read its results against the floor reported here.

> **GATE result (all three criteria met):**
> 1. **Independent m_s cross-check ✅** — recomputed three algebraically-distinct ways, matches
>    FreeGSNKE to **0.000 %** (bar was <5 %).
> 2. **Noise floor quantified ✅** — within a fixed config m_s is **bit-reproducible** (12 digits,
>    cross-process); the prior "~10 %" was **cross-BLAS-thread** spread (median 4.9 %, max 11.4 %),
>    eliminated by locking OMP=1; residual systematic protocol-dependence characterized.
> 3. **Protocol fixed + machine serialized ✅** — single pickled tokamak; two processes reproduce
>    m_s = `0.419940486916` bit-for-bit.

Artifacts: `experiments/phase0_lib.py`, `phase0_ms_crosscheck.py`, `phase0_serialize_machine.py`,
`phase0_verify_load.py`, `phase0_solve_one.py`, `phase0_noise_sweep.py`, `phase0_noise_analyze.py`;
`data/phase0_ms_crosscheck.json`, `data/phase0_noise_sweep.json`, `data/phase0_noise_summary.json`;
`figures/phase0_noise_floor.png`; `machine_configs/MAST-U/serialized_tokamak.pkl`.

---

## 1. THE LOCKED PROTOCOL (use for every label from here on)

| Knob | Locked value | Why |
|---|---|---|
| **BLAS threads** | `OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=NUMEXPR_NUM_THREADS=VECLIB_MAXIMUM_THREADS=1` | Eliminates the dominant cross-thread floor (§3.1). Set **before** importing numpy. |
| **Machine build** | ONE serialized tokamak: `machine_configs/MAST-U/serialized_tokamak.pkl` (load via `phase0_lib.load_machine`) | One identical machine for the whole study; no rebuild, no `LatinHypercube(seed=42)` state drift (audit 4.3). |
| **Solve isolation** | **Cold** solve each shape (fresh `Equilibrium` + reset coil currents; never reuse a warm tokamak across shapes) | Removes inverse-solve warm-start path dependence (~2 %, §3.2). |
| **Grid** | `65 × 65` (Rmin=0.1, Rmax=2.0, Zmin=−2.2, Zmax=2.2) | Corroborated by the 129×129 value (§3.3); 129×129 shifts m_s ≤7 % at ~4× cost. |
| **Retained passive modes** | `fix_n_vessel_modes = 40` | Phase-1 continuity. **Caveat:** m_s is *not* mode-converged here (§3.4) — flagged, convergence study owed in Phase 2. |
| **Inverse-solve tolerance** | `target_relative_tolerance = 1e-6` (`target_relative_psit_update = 1e-3`) | Corroborated by the high-res grid (§3.5); tighter tol *chases* the ill-conditioned inverse, it does not improve accuracy. |
| **Profiles / coils / targets** | identical to `phase1_generate.evaluate` (ConstrainPaxisIp paxis=8e3, Ip=6e5, fvac=0.5, αm=1.8, αn=1.2; Solenoid fixed at 5000 A) | Continuity with the validated Phase-1 grid. |
| **Plasma resistivity** | `1e-6` (only affects γ, **not** m_s — m_s is resistivity-independent, §2) | — |

The reference implementation of all of the above is `experiments/phase0_lib.solve_equilibrium`.
Per-label cost at the locked protocol: **~46–90 s** (65×65, 40 modes); 129×129 ≈ 250–320 s; 80 modes ≈ 1.4×.

**Environment note:** Python 3.12.10, FreeGSNKE 3.0.1, FreeGS 0.8.2, numpy 1.26.4,
**scipy 1.15.3** (the figure of `1.11.4` in earlier notes is wrong — corrected here). Windows
console is cp1252 → set `PYTHONIOENCODING=utf-8`.

---

## 2. THE m_s DEFINITION & CONVENTION (verified — high confidence)

**m_s is the Portone (2005) inductive stability margin** of the axisymmetric (n=0) vertical mode:
a dimensionless, **resistivity-independent** scalar measuring the distance of the
plasma + passive-structure configuration from the **ideal-MHD (with-wall) marginal point**.

- **Sign convention (load-bearing — state it explicitly in the paper):**
  **m_s > 0 ⇒ ideally stable / only passively (resistively) unstable on the slow wall timescale ⇒ controllable.**
  **m_s = 0 ⇒ ideal-MHD marginal point** (wall provides no passive stabilization; γ → Alfvénic).
  **m_s < 0 ⇒ Alfvénically unstable, uncontrollable.**
  **Larger m_s = more stable;** m_s → 0 from above as the plasma is made more vertically unstable.
- **Empirically confirmed here:** as κ rises 1.85→2.06, γ rises 37→248 /s while m_s falls
  0.898→0.411 (and the full project scan: κ 1.65→2.06 ⇒ γ 2.3→330 /s, m_s 3.91→0.34).
- **Practical thresholds** (distinct from the m_s=0 ideal point): design margin **m_s > 0.15**
  (FUSE); device closed-loop limits ≈ 0.26 (TCV, Alcator C-Mod).
- **Formula FreeGSNKE evaluates** = Humphreys (2009) eq. (4) / FreeGSNKE `example10`:
  `m_s := λ[ −M_mm⁻¹ (M_mm + M_my ∂Iy/∂Im) ]`, λ = largest eigenvalue. This is exactly the
  `−L⁻¹L*` form the scoping flagged, and it equals `eig(L⁻¹S − I)`.

**FreeGSNKE implementation note (do NOT quote the code comment):** `calculate_stability_margin`
in `freegsnke/linear_solve.py` returns the **positive** eigenvalues of `A = L⁻¹S − I`
(`L = M0matrix[:n,:n]`, `S = −dMmatrix[:n,:n]`); we take their **max** as m_s. The inline comment
`# the positive (i.e. unstable) eigenvalues` is a **copy-paste mislabel** from the growth-rate
routine (where a positive *timescale* genuinely is unstable). Here positive eigenvalues are the
**stable-side** margins. The code is correct; the comment is wrong — verified numerically (m_s is
positive and decreases monotonically toward 0 as γ rises) and against the cited literature.

**Cite as:** A. Portone, *Nucl. Fusion* **45** (2005) 926, doi:10.1088/0029-5515/45/8/021
(origin); D.A. Humphreys et al., *Nucl. Fusion* **49** (2009) 115003 (multi-machine inductive
form, eq. 4 — the form FreeGSNKE evaluates). Corroborating: FUSE (arXiv:2409.05894, verbatim
sign convention + m_s>0.15); K. Olofsson, OSTI-1960105 (resistivity-independence); VacuumFields.jl
`src/mutual.jl`; Isernia & Villone, *PPCF* (2023) doi:10.1088/1361-6587/acf15c.
**Provenance caveat:** the Portone 2005 full text is paywalled (abstract verified on IOPscience);
the equations above are anchored on Humphreys 2009 / FreeGSNKE example10 / FUSE / Olofsson, all
mutually consistent and all citing Portone 2005.
**Scoping refinement:** Pertnet (arXiv:2202.13915) and GSPulse (arXiv:2506.21760) work with γ
directly and do **not** report Portone m_s by name — this *sharpens* the "we target m_s" novelty.

### 2.1 The independent cross-check (GATE criterion 1)
For the canonical converged equilibrium (z=1.00, dR=0, 65×65, 40 modes, OMP=1) we extracted the
linearisation's `M0matrix`/`dMmatrix`/`Mmatrix` blocks and recomputed the margin **three
algebraically-distinct ways** (none calling FreeGSNKE's own routine):

| Method | max positive eigenvalue | vs FreeGSNKE `stability_margin` (0.419940) |
|---|---|---|
| A: `eig(L⁻¹S) − 1` (FreeGSNKE's form, recomputed) | 0.419940 | **0.000 %** |
| B: `eig(−L⁻¹L*)` (the `−L⁻¹L*` form; = Humphreys eq.4) | 0.419940 | **0.000 %** |
| C: `eig(S, L) − 1` (generalized eigenproblem, no explicit inverse) | 0.419940 | **0.000 %** |

- The two forms the scoping flagged are **numerically identical** (no hidden sign/convention diff).
- Exactly **one** positive eigenvalue (the single n=0 vertical mode), matching `n_unstable = 1`;
  the rest of the spectrum clusters at −1 (decoupled stable modes).
- Eigenvalues essentially real (`max|Im| = 5.9e-3` on a decoupled mode; the margin itself is real).
- Block identities exact: `(M0+dM)−M = 0`, `L* − (L−S) = 0` to machine zero.

---

## 3. THE NUMERICAL REPRODUCIBILITY FLOOR (GATE criterion 2)

Measured on **10 representative shapes** spanning m_s ∈ [0.37, 1.82] (γ ∈ [7, 320] /s), each an
**isolated cold solve** loading the serialized machine. Full data: `data/phase0_noise_sweep.json`;
figure: `figures/phase0_noise_floor.png`.

**The key reframing:** *within a fixed configuration there is no random label noise* — m_s is
**bit-reproducible to 12 digits across processes** (Part 3). The "~10 % floor" in the audit was
**cross-configuration** spread (BLAS thread count, solve path), which is **removed by locking the
protocol**. What remains is a **systematic** dependence of the *absolute* m_s on grid/modes/tol —
a calibration bias (fixed once the protocol is fixed), not label noise. Report both, separately.

### 3.1 Cross-BLAS-thread spread (the headline floor) — eliminated by OMP=1
At {OMP=1,2,4,8} (tol=1e-6, 65×65, 40 modes): m_s spread **median 4.9 %, max 11.4 %**;
γ spread **median 4.7 %, max 19.0 %**. **The spread grows toward the marginal boundary**
(z=1.02: 11.4 %, z=1.00: 9.8 %) and is small for stable shapes (z=0.88: 2.0 %, dR variants ~1 %)
— i.e. the floor is worst exactly where m_s→0, the regime the surrogate must put honest
uncertainty on. Cause: the ill-conditioned coil **inverse** solve amplifies FP summation-order
differences set by the BLAS thread count. **Locking OMP=1 makes this 0** (bit-deterministic).

### 3.2 Warm-start path dependence — eliminated by cold solves
The canonical shape gave m_s = 0.41994 as a cold first solve but 0.4106 (κ 2.055→2.060) as the
3rd solve in a shared-tokamak sequence (~2 %): leftover coil currents seed the next inverse solve.
**Removed by isolated cold solves** (reproduces audit 4.2; also implies the Phase-1 sequential
grid carried ≤~2 % order noise, within the floor).

### 3.3 Grid resolution — systematic, ≤7 % for stable; LARGER on the marginal band
65×65 → 129×129 shifts m_s **down** by up to 7 % for high-m_s/stable shapes (z=0.88: −7.1 %) and
**negligibly** for the single marginal shape tested at 40 modes (z=1.00: −0.2 %). Cost ~4×. 65×65 retained.
**UPDATE (Phase-4 re-verification at the converged 80 modes, `phase4_grid_check.py`, 18 shapes):** at 80 modes the
stable band reproduces (~4.5 % median, 129² lower) BUT the **marginal band is MORE grid-sensitive than this 40-mode
single-point estimate implied — near-marginal (m_s 0.30–0.45) median |shift| ≈10 % (signed −10 %); deep-marginal
(m_s 0.11–0.30) median ≈10 % with at least one QUALITATIVE FLIP** (a shape at m_s=0.26 on 65² is m_s=1.71 on 129² —
the m_s→0 boundary is pathologically grid-sensitive). This is a systematic on the ABSOLUTE marginal m_s; it **cancels
in every fixed-grid relative/gradient/design comparison** (as the mode bias does), so the design/κ-constrained results
are unaffected — but absolute marginal-m_s statements carry a ~10 % grid band (occasionally a stability-class flip at
the very boundary). Report absolute m_s near m_s→0 with this band; lead with relative/design metrics + abstention.

### 3.4 Retained passive modes — the DOMINANT systematic
`fix_n_vessel_modes = 20` is **unusable** (z=1.00 → m_s=0.08, γ=1107 /s; off by ~5×).
40 → 80 shifts m_s **up by +9–13 %** (cost ~1.4×) — so **m_s is not mode-converged at 40**.
This is the largest single systematic. **Action (Phase 2):** a mode-convergence study (40/80/160)
and a final converged mode count for published labels. For Phase 0/1.5 continuity we keep 40 and
report the ~10 %/doubling sensitivity.

### 3.5 Inverse-solve tolerance — keep 1e-6 (tighter is counterproductive)
tol 1e-6 ≈ 1e-8 (identical). tol 1e-10 shifts marginal-shape m_s **down ~10 %** (z=1.00:
0.420→0.382; z=0.94: 0.690→0.622) while leaving stable shapes unchanged. **But** the loose-tol
value (0.4199) matches the **highest-resolution 129×129** value (0.4190) to <0.3 %, whereas the
1e-10 value (0.382) departs — so tightening tol beyond 1e-6 **chases the ill-conditioned inverse**
rather than improving accuracy. Keep **1e-6**. The cross-thread 1-vs-8 spread does **not** shrink
with tighter tol.

### 3.6 What the paper reports
> *Labels are generated at a single locked configuration (OMP=1; one serialized MAST-U build;
> isolated cold inverse solves; 65×65 grid; 40 retained passive modes; inverse tolerance 1e-6) and
> are bit-reproducible (12 significant figures, verified cross-process). The numerical m_s carries a
> **systematic protocol-dependence** — dominated by passive-mode truncation (≈10 % per doubling
> near 40 modes), with grid (≤7 %) and inverse-tolerance contributions — and, if the BLAS thread
> count is left free, a cross-thread spread of median 4.9 % / max 11.4 % that we remove by pinning
> OMP=1. These are systematic biases of the ill-conditioned inverse map, not random label noise;
> Phase 1.5 forward sampling (well-conditioned current/profile sampling, no inverse solve) targets
> their root cause.*

---

## 4. CARRY-FORWARD ACTIONS
- **Phase 1.5:** generate data by **forward sampling** (coil currents + profile params), not the
  ill-conditioned inverse solve — this is the root-cause fix for §3.1/§3.2/§3.5. Reuse the
  serialized machine + OMP=1 + cold solves.
- **Phase 2:** mode-convergence study (40/80/160); finalize a converged `fix_n_vessel_modes`;
  put calibrated uncertainty on the m_s→0 boundary where the floor is worst.
- **Paper:** state the m_s definition/convention per §2 (cite Portone 2005 + Humphreys 2009 eq. 4);
  never quote the FreeGSNKE `(i.e. unstable)` comment; report the floor per §3.6.
