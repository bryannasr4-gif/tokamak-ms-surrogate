# Scoping Decision — Differentiable Vertical-Stability-Margin Surrogate for a Spherical Tokamak

**Written 2026-06-18. Read this before writing any pipeline code.** This is the literature-and-feasibility
reconciliation the project's process lessons say to do *first*. It was done first this time.

> **UPDATE — Phase 1 PASSED + AUDITED (2026-06-18).** A learned ensemble surrogate's autodiff ∇m_s matches the
> true-solver gradient (median cosine 0.956, 100% sign agreement, R²=0.93), and a gradient-ascent step raised the
> *freshly re-solved* true m_s by **+0.125 (+41%)**, ≈3× the numerical noise floor. **Audited & corrected:** the
> earlier "+89%, beats baselines" was a cross-config artifact; the gradient does NOT demonstrably beat an
> analytic-κ heuristic in 2-D (within the ~10% m_s noise floor) — that advantage is deferred to higher-D Phase 2.
> All citations verified real (FGE *is* differentiable → do not claim "first differentiable γ"). See
> `PHASE1_RESULTS.md` + `AUDIT_2026-06-18.md`. Phase-2 protocol fixes required: one BLAS config + one machine
> build + reduce/quantify the noise floor (ideally forward sampling, not the ill-conditioned inverse solve).

---

## 0. Verdict — **GO, but narrow, and reframe before you build**

The pivot is **technically real and feasible on your laptop today** (proven below, with numbers). The
contribution is **genuinely novel only in a precise, narrow form**, because a learned neural-net surrogate that
outputs the vertical growth rate of a *spherical tokamak* **already exists, from Egemen Kolemen's own group**
(Pertnet, NSTX-U, 2022). If you pitch "the first ML model for ST vertical stability," a Kolemen-group reader
recognizes their own prior work in your second sentence — the same novelty-evaporates-on-contact failure that
killed the SBI idea. **Do not make that mistake twice.**

What survives adversarial review is one demonstrable, unclaimed thing: **differentiate the inductive stability
margin m_s with respect to a clean low-dimensional plasma *shape*, validate that gradient against the expensive
solver, and *use* it to close a shape-optimization loop that FreeGSNKE confirms raises the margin — all on open,
reproducible synthetic data.** That is real, finishable solo, and directly useful to the group you want to join.

The whole paper lives or dies on **the gradient working** (not the regression). Run the Phase-1 gradient
de-risk (≈1 day) before building any dataset. If the gradient is too noisy, you have a clean fallback paper
(below) — and you'll know in week one, not month three.

---

## 1. The one defensible novel sentence

> *An open, reproducible, **differentiable** surrogate that maps a low-dimensional plasma **shape**
> (elongation, triangularity, squareness, wall gaps) to the Portone-2005 inductive **stability margin m_s**
> (and growth rate γ) of a **spherical tokamak** (MAST-U geometry), trained on open synthetic FreeGSNKE
> equilibria, whose analytic gradient d m_s/d(shape) is validated against finite differences of the expensive
> solver and then **used** as the optimization gradient to drive a shape change that FreeGSNKE independently
> confirms increases the stability margin.*

The four things that make it novel are a **conjunction** — drop any one and a published paper already covers it:
**{ m_s as target (not γ) } + { low-D shape as input (not full equilibrium state) } + { the gradient is USED for
solver-confirmed shape optimization (not merely monitored) } + { open synthetic ST data (not closed Gspert/NSTX-U) }.**

You must **not** claim "first ML growth-rate surrogate" (Pertnet 2022, Sammuli 2021, Liu 2023 all precede you)
nor "first differentiable γ" (the FGE solver, late-2025, precedes you). Claim only the conjunction above, and
demonstrate the gradient-used loop.

---

## 2. What I empirically confirmed this session (feasibility is not a question mark)

All on this laptop, with the FreeGSNKE that's already installed, real MAST-U-like machine geometry
(12 active coils + 138 passive structures, aspect ratio **1.64** → genuinely spherical).

**Feasibility — the ground truth exists and is cheap-ish** (`experiments/11_growth_rate_smoke.py`,
`data/11_growth_rate_smoke.json`):
- A diverted MAST-U equilibrium (Ip = 600 kA, κ ≈ 2.05, δ ≈ 0.52) solves in **13.5 s** (inverse) .
- `nl_solver` builds the 3051×53 finite-difference Jacobian dIy/dI and returns, in **~33 s**:
  **γ = 260 /s**, **m_s = 0.39**, Leuer ratio 1.58 — all physically correct for a MAST-U-class ST.
- **One labeled sample ≈ 46 s** (inverse solve + linearization). That is the data-generation budget — and the
  precise economic case for a surrogate: microsecond, *differentiable* inference vs. ~50 perturbed GS solves.

**Learnable signal — γ(shape) and m_s(shape) are smooth and monotonic**
(`experiments/12_elongation_scan.py`, `data/12_elongation_scan.json`, `figures/12_growth_rate_vs_elongation.png`):

| elongation κ | growth rate γ [1/s] | stability margin m_s |
|---|---|---|
| 1.65 | 2.3 | 3.91 |
| 1.82 | 22 | 1.04 |
| 1.98 | 148 | 0.52 |
| 2.05 | 330 | 0.34 |

γ rises ~2 decades and m_s falls **monotonically toward the m_s→0 controllability boundary** as the plasma is
stretched. This (a) proves a strong, smooth regression target, (b) reconciles my baseline (260/s) with the
NSTX-U literature band (~30–120/s at lower κ) — the difference is just elongation, and (c) confirms **m_s is the
smoother target** (supports predicting m_s, per §1). The scan also maps the numerically stiff regime: κ ≳ 2.1
drives the plasma into the vessel and the GS solve degrades — exactly the marginal region the surrogate must
later put honest uncertainty on.

**Gradient quality — the make-or-break precondition PASSED** (`experiments/14_gradient_quality.py`,
`data/14_gradient_quality.json`, `figures/14_gradient_quality.png`): a fine-step sweep (Δκ ≈ 0.02) around
κ ≈ 1.99 gives a smooth m_s and a **central-difference d m_s/dκ of −2.19, −1.76, −1.44** — same sign,
slowly varying, *not* noise. So the single biggest risk (m_s is an eigenvalue of a finite-difference Jacobian →
garbage gradient) is **empirically refuted** at `fix_n_vessel_modes=40` in this operating region. Caveat for
Phase 1: the inverse-solve (shape→currents) can inject mild local non-smoothness (one slope flattened where γ
also plateaued), so the surrogate-side gradient check still matters — but the *physics precondition is met*.

---

## 3. The literature reckoning — who already stands where

Verified primary sources (Pertnet checked directly this session; the very recent 2025–26 IDs are flagged
"verify before citing"):

| Work | Learned? | Spherical tokamak? | n=0 vertical mode? | Target | Input | Gradient **used** for shape opt? | Open synthetic data? |
|---|---|---|---|---|---|---|---|
| **Pertnet** (Wai/Boyer/Kolemen 2022, arXiv:2202.13915) | **yes** | **yes (NSTX-U)** | **yes (γ)** | plasma response → γ | full equilibrium state | **no** (derivative-free; γ only monitored) | no (closed Gspert) |
| FGE solver (arXiv:2512.06847 — verify) | no (physics) | tested incl. MAST-U | yes | γ | n/a | no | n/a |
| Pentland virtual circuits (MAST-U, 2026 — verify) | yes | yes | **no** (omits stability) | 7 shape params | currents, Ip, profiles | shape-control only | yes (FreeGSNKE) |
| Sammuli DIII-D RZRIG-NN (FED 169:112492, 2021) | yes | no | yes (γ) | rigid γ | equilibrium state | no (real-time) | no |
| Liu EAST-MLP (IEEE TPS 2023) | yes | no | yes (γ) | γ | 38 probes | no | no |
| GSPulse (Wai/Kolemen 2025, arXiv:2506.21760 — verify) | no | yes (NSTX-U+MAST-U) | implicit only | trajectory | n/a | **gradient-free** QP | n/a |
| **THIS WORK (reframed)** | **yes** | **yes (MAST-U)** | **yes** | **Portone m_s (+γ)** | **low-D shape** | **yes (solver-confirmed)** | **yes (FreeGSNKE)** |

**Nearest neighbor = Pertnet, and it is the killer of the broad claim** — but it fills *none* of the four
conjunction cells: it targets γ (not m_s), takes the full equilibrium state (not a clean shape), used
derivative-free `fminsearch` (never differentiated the growth rate to design a shape), on closed NSTX-U data.
The cleanest positioning is actually against **Pentland's MAST-U virtual-circuit surrogate**: same geometry, same
FreeGSNKE tooling, same "differentiate an NN trained on a GS library" recipe — but it explicitly outputs *no*
stability margin. **You fill precisely that hole.**

---

## 4. The plan — de-risk gates first, build second

**Phase 0 — label trust (≈1 day). Cheapest sanity check before anything.**
Reproduce one equilibrium's m_s by an *independent* re-derivation of the Portone eigenvalue from FreeGSNKE's own
M0/dM matrices (the scoping flagged a possible sign/convention difference between `eig(L⁻¹S − I)` and `−L⁻¹L*`).
**Pass:** independent m_s matches FreeGSNKE's to <5% and γ sits in the plausible ST band. *(Largely done: γ, m_s
already reproduce sensible values and a smooth κ-trend; this phase just nails the convention.)*

**Phase 1 — THE make-or-break gradient de-risk (≈2–4 days). Do this before building the dataset.**
On a 2–3 parameter shape slice (κ, wall gap), generate ~100–300 labels, fit a small surrogate, and verify its
**autodiff gradient d m_s/d(shape) matches a finite-difference of the *expensive FreeGSNKE* m_s** — in sign on
≥90% of held-out points and within ~30% magnitude — and that one gradient-ascent step yields a shape whose
*freshly re-solved* FreeGSNKE m_s is higher. **This single experiment greenlights or kills the paper.** If it
fails (m_s gradient too noisy near the boundary), drop to the §6 fallback — do not force a broken gradient claim.

**Phase 2 — dataset + surrogate (≈1 week).** Scale to a 4–6 parameter low-D shape space (κ, δ, squareness,
inner/outer gap), ~2–5k labels spanning both m_s>0 and marginal m_s→0 regimes; train a small MLP for {m_s, γ};
report held-out R² and calibration. **Fix and report** the retained-mode count (`fix_n_vessel_modes`) — it drives
both cost and the label value. Parallelize generation on free Colab/Kaggle (embarrassingly parallel; ~46 s each).

**Phase 3 — the contribution: the gradient-used closed loop (≈3–5 days).** From a marginally-stable shape, take
gradient-ascent steps on the surrogate m_s; confirm with a fresh FreeGSNKE solve that the true m_s rose. **Control
baselines (mandatory):** beat (i) a finite-difference loop on the surrogate and (ii) a 1–2 parameter analytic
κ-scaling heuristic, or the contribution collapses.

**Phase 4 — rigor sections (≈1 week).** Cost/accuracy Pareto vs. the full FreeGSNKE linearization (validation, not
headline); the cheap rigid **Leuer** parameter as a physics baseline/feature; **stretch:** calibrated error bars /
abstention near m_s→0.

**Phase 5 — anti-scoop + write-up.** Re-scan arXiv for 2026 UKAEA/FreeGSNKE/Kolemen preprints; cite and carve
explicitly against Pertnet and FGE. Target ML4PS (4-page) or APS-DPP poster + arXiv.

---

## 5. Risks and mitigations

| Risk | Severity | Mitigation |
|---|---|---|
| **Pertnet already is a learned ST γ surrogate from the target group.** Broad claim is dead; residual delta is thin. | High | Never claim the broad headline. Cite Pertnet in the abstract; make the **gradient-used, m_s, open-data loop** the load-bearing demonstrated contribution (the one thing Pertnet did not do). |
| **The m_s gradient is noisy** (eigenvalue of a finite-difference Jacobian) → fails Phase-1 check. | High | Front-load Phase 1. Target m_s (dimensionless, resistivity-independent, smoother) over raw γ; keep a low-D shape space; raise Jacobian FD accuracy / smooth labels if needed; fall back honestly if it still fails. |
| **Scooped mid-project** (UKAEA virtual-circuit line may add a stability margin). | Medium | Move fast; re-scan arXiv monthly; the open-data + low-D-shape + gradient-loop framing differs even from a value-only m_s output. |
| **FreeGSNKE γ/m_s itself only validated against its own solver**, not experiment. | Medium | State the ground truth honestly as "FreeGSNKE linearized m_s," not experiment; do the Phase-0 cross-check; do not overclaim physical fidelity. |
| **"So what?"** — GSPulse/FUSE already do design. | Medium | Anchor on the exact gap: both are **gradient-free**; a differentiable d m_s/d(shape) is the capability they lack — directly useful to Kolemen-style stability-aware shape/scenario design. |

---

## 6. Fallback (the guaranteed publishable result)

If the Phase-1 gradient fails, the smallest honest result is still a clean ML4PS 4-page / APS-DPP poster + arXiv,
**no domain co-author required**: an open, reproducible **benchmark + calibrated surrogate** for ST vertical
stability on open synthetic FreeGSNKE MAST-U data — a shape→{m_s, γ} regressor with held-out accuracy, the cheap
rigid **Leuer** parameter as a physics baseline, **calibrated error bars / abstention near m_s→0**, and an honest
cost/accuracy comparison vs. the full linearization. It ships an **open dataset** as a reusable artifact and is the
first to position **m_s** (not just γ) as the learned target. Not a flagship, but a real artifact with honest
numbers — the currency that matters for an internship.

---

## 7. How to pitch it to PPPL / the Kolemen group

Bring it as an **honest, well-cited extension of their own line**, not a "look what I built":
> "Your group's Pertnet showed a learned ST vertical-stability surrogate works, and GSPulse does feed-forward
> design with a gradient-free QP. I close the **gradient** loop on the inductive margin **m_s** — a differentiable
> d m_s/d(shape) that the expensive solver confirms raises the margin — on **open, reproducible** FreeGSNKE
> MAST-U data, filling the stability-margin hole the MAST-U virtual-circuit work left open."

That sentence proves you read their papers, names the precise gap, and offers a capability they lack. That is what
earns a co-authorship and an internship — far more than a novelty claim a single citation can rebut.

---

## 8. Venues
- **ML4PS** (NeurIPS workshop, non-archival 4-page) — realistic first target for a solo student.
- **APS-DPP poster** + **arXiv** (cs.LG + physics.plasm-ph).
- **PPCF / Nuclear Fusion** — only with a PPPL/Kolemen-group co-author (which is the point of the artifact).

## 9. Key references (verify each ID before citing — the 2025–26 ones especially)
- **Pertnet** — Wai, Boyer, Kolemen, "Neural net modeling of equilibria in NSTX-U," *Nucl. Fusion* 62 (2022)
  086042, arXiv:2202.13915, code github.com/PlasmaControl/nstxu-nns. **The direct precedent — cite and carve against.**
- **Portone**, "The stability margin of elongated plasmas," *Nucl. Fusion* 45 (2005) 926,
  doi:10.1088/0029-5515/45/8/021 — defines m_s; implemented in `freegsnke/linear_solve.py:calculate_stability_margin`.
- **Amorisco et al.**, "FreeGSNKE," *Phys. Plasmas* 31 (2024) 042517, doi:10.1063/5.0188467 — the toolchain.
- **Sammuli et al.**, DIII-D NN growth-rate VDE avoidance, *Fusion Eng. Des.* 169 (2021) 112492 — learned-γ precedent.
- **Liu et al.**, EAST MLP vertical growth rate, *IEEE TPS* (2023) — learned-γ precedent.
- **GSPulse** — Wai, Kolemen et al., arXiv:2506.21760 (verify) — gradient-free design tool that lacks d m_s/d(shape).
- **FGE**, fast differentiable GS evolutive solver, arXiv:2512.06847 (verify) — preempts "first differentiable γ."
- **Pentland et al.**, MAST-U virtual-circuit NN surrogate, arXiv 2026 (verify) — nearest neighbor; omits m_s.

## 10. Honest bottom line
This is a **GO** — narrow, finishable, PPPL-relevant — *conditional on the Phase-1 gradient working* and *on
reframing away from any "first ML/differentiable γ" claim*. The headline you might have instinctively written is
already Kolemen's own 2022 paper; the contribution that survives is the **gradient-used, m_s-targeted, open-data
shape-optimization loop**. It is honest, it is yours, and it is exactly the kind of self-aware, literature-anchored
work that earns a seat at PPPL. If the gradient doesn't survive Phase 1, the calibrated open-benchmark fallback is
still a clean, real, publishable artifact. Either way, you now know — *before* building — exactly what you are
claiming and against whom.
