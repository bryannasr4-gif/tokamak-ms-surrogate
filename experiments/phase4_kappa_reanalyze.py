"""
phase4_kappa_reanalyze.py -- the Phase-4 LEARNED-m_s differentiator, re-framed honestly from the
EXISTING Phase-2.5b n=20 kappa-constrained results (data/phase25_kappa_results.json). NO new solves.

Phase 3 attributed the design loop's solve-efficiency win to the GRADIENT/kappa-direction, not the
learned m_s (the reduce-kappa heuristic ties the surrogate). Phase 4 asks: with kappa held FIXED
(the reduce-kappa lever DISABLED), is the learned m_s load-bearing? The Phase-2.5b run already
holds kappa fixed and forces m_s up via the secondary levers. We re-analyze it against THREE
baselines, separating what is realizable from what is an oracle:

  (1) vs the best REALIZABLE single-lever rule = the ONE secondary lever+sign with the highest
      median gain ACROSS starts, chosen once (a designer's fixed rule). Surrogate must beat this.
  (2) vs the per-start ORACLE single lever = max over 8 levers per start (hindsight selection).
      This is NOT realizable without trying all 8 levers; we report it AND charge its lever-search
      cost (it needs ~K extra design budgets to discover which lever to use).
  (3) vs the best gradient-free optimizer (CMA / random) -- the Phase-3 baseline, kappa-penalized.

Headlines are resolved by regime with Wilson CIs + two-sided sign + paired Wilcoxon tests.
Saves data/phase4_kappa_reframed.json.
"""
import json
import os
import sys

import numpy as np
from scipy.stats import wilcoxon, binomtest

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

HEURISTICS = [f"heuristic:{f}{s}" for f in ["sq_uo", "gap_outer", "li", "betap"] for s in ["+", "-"]]
GRADFREE = ["random", "cma"]


def wilson(k, n, z=1.96):
    if n == 0:
        return [float("nan"), float("nan")]
    p = k / n
    den = 1.0 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [float(c - h), float(c + h)]


def beat(sur, comp):
    diff = sur - comp
    wins = int(np.sum(diff > 0)); losses = int(np.sum(diff < 0)); ties = int(np.sum(diff == 0))
    n = len(diff); n_eff = wins + losses
    out = dict(n=n, wins=wins, losses=losses, ties=ties,
               beats_frac=float(wins / n) if n else float("nan"), wilson95=wilson(wins, n),
               median_diff=float(np.median(diff)) if n else float("nan"))
    out["sign_p"] = float(binomtest(wins, n_eff, 0.5, alternative="two-sided").pvalue) if n_eff else float("nan")
    out["wilcoxon_p"] = float(wilcoxon(sur, comp, alternative="two-sided").pvalue) if np.any(diff != 0) else float("nan")
    return out


def med(x):
    x = np.asarray(x, float)
    return float(np.median(x)) if len(x) else float("nan")


def main():
    recs = json.load(open("data/phase25_kappa_results.json"))["recs"]
    setup = json.load(open("data/phase25_kappa_setup.json"))
    starts = sorted(set(r["start_i"] for r in recs))
    regime = {r["start_i"]: r["regime"] for r in recs}

    def gain(si, m):
        g = [r["gain"] for r in recs if r["start_i"] == si and r["method"] == m]
        return max(g) if g else np.nan

    # (1) best REALIZABLE single fixed rule = the lever with the highest POOLED median gain
    lever_pooled_median = {h: med([gain(si, h) for si in starts]) for h in HEURISTICS}
    best_fixed_lever = max(lever_pooled_median, key=lever_pooled_median.get)

    rows = []
    for si in starts:
        rows.append(dict(start=si, regime=regime[si],
                         surrogate=gain(si, "surrogate"),
                         best_fixed_lever=gain(si, best_fixed_lever),
                         oracle_lever=np.nanmax([gain(si, h) for h in HEURISTICS]),
                         best_gradfree=np.nanmax([gain(si, g) for g in GRADFREE])))

    def arr(subset, key):
        return np.array([r[key] for r in subset])

    out = dict(ktol=setup["ktol"], budget=setup["budget"], n_starts=len(rows),
               best_fixed_lever=best_fixed_lever, lever_pooled_median=lever_pooled_median,
               n_levers=len(HEURISTICS), by_regime={})

    for label, subset in [("pooled", rows),
                          ("marginal", [r for r in rows if r["regime"] == "marginal"]),
                          ("mid", [r for r in rows if r["regime"] == "mid"])]:
        keys = ["surrogate", "best_fixed_lever", "oracle_lever", "best_gradfree"]
        ok = np.all([np.isfinite(arr(subset, k)) for k in keys], axis=0)
        sur = arr(subset, "surrogate")[ok]
        if len(sur) == 0:
            continue
        d = dict(n=int(len(sur)),
                 median_gain={k: med(arr(subset, k)[ok]) for k in keys},
                 vs_best_fixed_lever=beat(sur, arr(subset, "best_fixed_lever")[ok]),
                 vs_oracle_lever=beat(sur, arr(subset, "oracle_lever")[ok]),
                 vs_best_gradfree=beat(sur, arr(subset, "best_gradfree")[ok]))
        out["by_regime"][label] = d
    out["rows"] = rows
    json.dump(out, open("data/phase4_kappa_reframed.json", "w"), indent=2)

    # ----- print -----
    print(f"=== Phase-4 LEARNED-m_s differentiator: kappa held +-{out['ktol']} (reduce-kappa DISABLED), "
          f"budget {out['budget']}, n={out['n_starts']} ===\n")
    print(f"best REALIZABLE single fixed lever (highest pooled-median gain) = {best_fixed_lever} "
          f"(median {lever_pooled_median[best_fixed_lever]:+.3f})")
    print("  all single levers, pooled median gain:")
    for h, v in sorted(lever_pooled_median.items(), key=lambda kv: -kv[1]):
        print(f"    {h:20s} {v:+.3f}")
    for label in ["pooled", "marginal", "mid"]:
        r = out["by_regime"].get(label)
        if not r:
            continue
        mg = r["median_gain"]
        print(f"\n[{label.upper()}] n={r['n']}  median gain: surrogate={mg['surrogate']:+.3f}  "
              f"best_fixed_lever={mg['best_fixed_lever']:+.3f}  oracle_lever={mg['oracle_lever']:+.3f}  "
              f"gradient_free={mg['best_gradfree']:+.3f}")
        for key, nm in [("vs_best_fixed_lever", "vs best REALIZABLE fixed lever"),
                        ("vs_oracle_lever", f"vs ORACLE best-of-{out['n_levers']} lever (hindsight)"),
                        ("vs_best_gradfree", "vs best gradient-free")]:
            b = r[key]
            print(f"  {nm:42s}: {b['wins']}/{b['n']} ({b['beats_frac']*100:.0f}%) "
                  f"Wilson95 {b['wilson95'][0]*100:.0f}-{b['wilson95'][1]*100:.0f}%  "
                  f"sign_p={b['sign_p']:.4f}  wilcoxon_p={b['wilcoxon_p']:.4f}  "
                  f"med_diff={b['median_diff']:+.3f}")
    print("\nNOTE: the ORACLE lever is not a realizable strategy -- a designer must SEARCH the "
          f"{out['n_levers']} levers to discover which works per start (~{out['n_levers']}x the design "
          "budget); the surrogate finds the multi-lever combination in ONE budget.")
    print("Saved data/phase4_kappa_reframed.json")


if __name__ == "__main__":
    main()
