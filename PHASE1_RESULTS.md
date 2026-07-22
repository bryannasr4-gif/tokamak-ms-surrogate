# Phase 1 — Result: **PASS** (gradient is accurate and usable) — *corrected & audited 2026-06-18*

Phase 1 is the experiment `SCOPING_vertical_stability.md` says decides the paper: *does a learned
surrogate's gradient d m_s/d(shape) match the expensive solver well enough to optimize a shape?* It does —
with honest caveats established by a full audit (see `AUDIT_2026-06-18.md`).

## Setup
- **Dataset:** 64-point grid over a 2-knob shape slice — elongation (`zscale`) × plasma–wall gap (`dR`) —
  each labeled with true FreeGSNKE m_s + γ (full linearization, `fix_n_vessel_modes=40`, 65×65).
  Generated on 8 thread-pinned workers (`OMP=1`), each a separate process building **one** machine.
  *Verified clean:* two independent processes reproduce m_s bit-for-bit, so all 8 workers built identical
  machines → the grid is internally consistent. (`experiments/phase1_generate.py`, `data/phase1_chunk_*.json`)
- **Surrogate:** 7-model ensemble of smooth (tanh) MLPs with internal input normalization, so autograd w.r.t.
  the raw physical shape gives the physical gradient. (`experiments/phase1_analyze.py`)

## Results (`data/phase1_results.json`, `figures/phase1_gradient_validation.png`)

**Accuracy.** Held-out m_s: **R² = 0.93**, RMSE = 0.071 (over a 0.91 range).

**The gradient test — PASS** (surrogate autodiff ∇m_s vs. true-solver finite-difference ∇m_s on 7 held-out
interior shapes; the FD true gradient comes from the clean `OMP=1` grid, so this test is config-consistent):
- direction **cosine = 0.956 median** (min 0.866), **100% sign agreement** on *both* shape components;
- magnitude ratio |∇sur|/|∇true| = 1.43 median (direction excellent; magnitude biased ~40% high).

**The closed-loop demo — PASS** (re-run at `OMP=1` to match the grid; start re-solved in-config):

| step from the most-unstable grid shape (start m_s = 0.306, κ = 2.07) | new shape | true m_s | Δ |
|---|---|---|---|
| **gradient ascent on the surrogate** | z 1.00→0.97, dR +0.040→+0.027 | **0.431** | **+0.125 (+41%)** |
| analytic "just reduce elongation" | z→0.964, dR +0.040 (fixed) | 0.424 | +0.118 |
| finite-difference on the surrogate | z→0.967, dR→0.031 | 0.414 | +0.108 |

The gradient step raised the *freshly re-solved* true stability margin by **+0.125 (≈ +41%)** — about **3× the
~0.04 numerical noise floor** (see below), so "the gradient step reliably increases true m_s" is a robust claim.

## What this proves (robust) — and what it does NOT (honest)
**Robust:**
1. **The central risk is dead.** m_s, despite being an eigenvalue of a finite-difference Jacobian, has a
   gradient smooth and accurate enough to *learn* (cosine 0.956, 100% sign) and to *use* (closed loop raises
   the true margin by ~3× the noise floor). The differentiable contribution is *demonstrated*, not asserted.
2. This is the capability Pertnet (derivative-free) and GSPulse (gradient-free) lack — on open ST data.

**NOT supported (do not claim):**
3. **The gradient does NOT demonstrably "beat" the analytic-κ heuristic in this 2-D slice.** Its margin
   (+0.125 vs +0.118 ≈ 0.007) is *within* the ~0.04 numerical noise floor. In 2-D, elongation dominates, so a
   κ-only heuristic is competitive. **The gradient's advantage is expected to appear only in a higher-D shape
   space where no hand-coded heuristic exists — this is the central thing Phase 2 must demonstrate.**
   (The earlier "+89%, beats baselines" figure was inflated by a cross-config confound — start measured in a
   different BLAS config than the destinations — now fixed.)

## The numerical reproducibility floor (audit finding — required for the paper)
FreeGSNKE's m_s is **bit-deterministic within a fixed configuration** (two `OMP=4` processes agreed to 10
digits) but varies **~10% across BLAS thread counts** for the same shape (canonical shape: m_s = 0.420 / 0.424 /
0.384 / 0.417 at `OMP=1/2/4/8`). Cause: the **coil inverse solve is ill-conditioned** (the SBI-era §4.2 lesson)
and amplifies floating-point summation-order differences. Also: FreeGSNKE's machine build uses a *module-level*
LatinHypercube engine (`refine_passive.py:26`, seed=42) that advances state, so **building the machine twice in
one process diverges ~20%** — but each Phase-1 worker builds **once**, so the grid is unaffected.
**Implication:** all labels must share one BLAS config + one machine build; the ~10% floor must be quantified and
reported as the noise the surrogate is read against. Phase 2 should additionally reduce it (tighter solve
tolerance; or generate data by *forward* sampling of currents/profiles — well-conditioned — instead of the
ill-conditioned inverse solve).

## Phase 2 (next)
Scale to a 4–6 parameter low-D shape space; **fix the protocol** (one BLAS config, one machine build, tighter
tolerance, or forward sampling); **report the label-noise floor**; train an ensemble for {m_s, γ}; and **show the
gradient advantage over heuristics emerges with dimensionality** — the headline figure.

## Bottom line
**Phase 1 greenlights the project.** The gradient is real, accurate, and usable; the closed loop raises the true
margin well above the noise floor. The "beats heuristics" claim is *not yet earned* (2-D, within noise) and is
deferred to Phase 2. The numerical noise floor is now characterized and must be controlled going forward.
