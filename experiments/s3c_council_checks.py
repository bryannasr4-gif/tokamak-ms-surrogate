"""s3c_council_checks.py -- adjudication checks for the S3c council-after round.

EXPLORATORY. confirmatory=false. Zero new solves: every number below is computed from data
already on disk (the frozen confirmatory cohort + the A1-9 exploratory lever sweep).

Purpose: the council-after raised three checkable objections. Per CLAUDE.md section 6 a council
finding is NEVER adopted on trust -- it is independently verified. These checks settle them:

  C1 (DeepSeek Q3, Nemotron Q3): "gap_outer- ranked 1 of 8, but the 1st-vs-2nd gap is smaller
      than the 2nd-vs-8th gap, so the ranking is not robust; a paired test of gap_outer- vs the
      runner-up would likely be non-significant." -> test it directly.

  C2 (the objection that actually matters for the paper): does the surrogate still beat the
      RUNNER-UP lever, and indeed every one of the 8 levers, on this fresh cohort? If it beats
      all 8, the winner's-curse concern is materially defused: it no longer matters which lever
      the winner's curse would have picked.

  C3 (DeepSeek Q7.2): the Wilson lower bound (0.728) versus the pre-registered minimum-useful
      win rate (0.65) -- report the margin honestly.
"""
import json
import os

import numpy as np
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONF = os.path.join(ROOT, "data", "phase4_power3_results.json")      # surrogate + gap_outer-
LEV = os.path.join(ROOT, "data", "phase4_power3_levers.json")        # the other 7 levers
OUT = os.path.join(ROOT, "data", "s3c_council_checks.json")
PRE_NAMED = "heuristic:gap_outer-"
RNG_SEED = 20260802


def gains_by_method():
    """start_i -> {method: gain}, from BOTH raw files, erroring loudly on any bad record."""
    g = {}
    for path in (CONF, LEV):
        for r in json.load(open(path))["recs"]:
            assert not r.get("error"), f"unexpected error record in {path}: {r}"
            assert np.isfinite(r["gain"]), f"non-finite gain in {path}: {r}"
            g.setdefault(r["start_i"], {})[r["method"]] = r["gain"]
    return g


def paired(g, m1, m2):
    """Paired vector m1 - m2 over the starts where both exist (should be all 28)."""
    ks = sorted(k for k in g if m1 in g[k] and m2 in g[k])
    return np.array([g[k][m1] - g[k][m2] for k in ks], float), ks


def boot_ci_median(d, n_boot=20000, seed=RNG_SEED):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    meds = np.median(d[idx], axis=1)
    return [float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))]


def test(d):
    n = len(d)
    wins = int((d > 0).sum()); losses = int((d < 0).sum()); ties = int((d == 0).sum())
    nt = wins + losses
    if np.all(d == 0):
        wp = sp = 1.0
    else:
        wp = float(stats.wilcoxon(d, zero_method="pratt", alternative="two-sided").pvalue)
        sp = float(stats.binomtest(wins, nt, 0.5).pvalue) if nt else 1.0
    return dict(n=n, wins=wins, losses=losses, ties=ties,
                median=float(np.median(d)), median_ci95=boot_ci_median(d),
                wilcoxon_p=wp, sign_p=sp)


g = gains_by_method()
methods = sorted({m for v in g.values() for m in v})
levers = [m for m in methods if m.startswith("heuristic:")]
print(f"starts: {len(g)}; methods: {len(methods)} ({len(levers)} levers + surrogate)")

med_gain = {m: float(np.median([g[k][m] for k in sorted(g) if m in g[k]])) for m in methods}
order = sorted(levers, key=lambda m: -med_gain[m])
print("\nlever ranking by median gain on THIS cohort:")
for i, m in enumerate(order, 1):
    print(f"  {i}. {m:24s} {med_gain[m]:+.4f}{'   <-- PRE-NAMED' if m == PRE_NAMED else ''}")

runner_up = order[1]

# ---- C1: is the pre-named lever significantly better than the runner-up? -----------------
d1, _ = paired(g, PRE_NAMED, runner_up)
c1 = test(d1)
print(f"\nC1  {PRE_NAMED} vs runner-up {runner_up}:")
print(f"    median diff {c1['median']:+.4f}  CI95 {c1['median_ci95']}  "
      f"Wilcoxon p={c1['wilcoxon_p']:.4f}  wins {c1['wins']}/{c1['n']}")
c1_sig = c1["wilcoxon_p"] <= 0.05
print(f"    -> pre-named significantly better than runner-up? {c1_sig}")

# ---- C2: does the surrogate beat EVERY lever on this cohort? -----------------------------
print("\nC2  surrogate vs each lever (EXPLORATORY, no multiplicity control claimed):")
per_lever = {}
for m in order:
    d, _ = paired(g, "surrogate", m)
    t = test(d)
    per_lever[m] = t
    print(f"    vs {m:24s} wins {t['wins']:2d}/{t['n']}  median {t['median']:+.4f}  "
          f"Wilcoxon p={t['wilcoxon_p']:.3e}")
all_beaten = all(t["wilcoxon_p"] <= 0.05 and t["median"] > 0 for t in per_lever.values())
# Holm across the 8 exploratory comparisons, reported for transparency only
ps = sorted(((m, t["wilcoxon_p"]) for m, t in per_lever.items()), key=lambda x: x[1])
holm, k = {}, len(ps)
prev = 0.0
for i, (m, p) in enumerate(ps):
    adj = max(prev, min(1.0, (k - i) * p))
    holm[m] = adj
    prev = adj
all_beaten_holm = all(holm[m] <= 0.05 and per_lever[m]["median"] > 0 for m in per_lever)
print(f"    -> surrogate beats ALL {len(order)} levers (uncorrected)? {all_beaten}")
print(f"    -> still true after Holm across the 8?                   {all_beaten_holm}")

# ---- C3: Wilson lower bound vs the pre-registered minimum-useful rate ---------------------
conf = json.load(open(os.path.join(ROOT, "data", "s3c_third_cohort.json")))
wl = conf["secondary"]["wilson95"][0]
useful = conf["power"]["power_at_useful_0_65"]
c3 = dict(wilson_lower=wl, min_useful_rate=0.65, lower_bound_exceeds_useful=bool(wl > 0.65),
          margin=float(wl - 0.65), power_at_useful_rate=useful)
print(f"\nC3  Wilson lower bound {wl:.4f} vs minimum-useful 0.65 -> "
      f"exceeds by {wl - 0.65:+.4f}")

out = dict(
    unit="S3c-council-checks", exploratory=True, confirmatory=False,
    not_reportable_magnitudes=True, new_solves=0,
    provenance="computed from data/phase4_power3_results.json + data/phase4_power3_levers.json; "
               "no new solver runs; adjudication support for the council-after round "
               "data/research/council/s4a_s3c_result/",
    purpose="independently test three checkable council objections (CLAUDE.md section 6: a "
            "council finding is never adopted on trust)",
    bootstrap=dict(n_boot=20000, seed=RNG_SEED),
    median_gain_by_method=med_gain, lever_ranking=order, runner_up=runner_up,
    C1_prenamed_vs_runner_up=dict(comparison=f"{PRE_NAMED} - {runner_up}", **c1,
                                  significant=bool(c1_sig)),
    C2_surrogate_vs_each_lever=dict(per_lever=per_lever, holm_adjusted_p=holm,
                                    surrogate_beats_all_uncorrected=bool(all_beaten),
                                    surrogate_beats_all_holm=bool(all_beaten_holm),
                                    note="EXPLORATORY. The confirmatory family remains the single "
                                         "pre-named comparison; these 8 cells are a post-hoc "
                                         "characterisation and are not registered claims."),
    C3_wilson_vs_useful=c3)
json.dump(out, open(OUT, "w"), indent=1)
print(f"\n-> {OUT}")
