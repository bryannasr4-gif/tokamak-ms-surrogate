"""
phase25_kappa_analyze.py -- the kappa-constrained "beats heuristics" verdict, FIRMED (Phase-2.5b).

With kappa held fixed (+-KTOL on the TRUE-solver kappa), does the learned m_s-gradient raise m_s
(via secondary levers) more than (a) the BEST fair single-secondary-lever heuristic and (b) the
best gradient-free optimizer? All methods are budget-equalised. We now have ~20 STRATIFIED starts
(~half marginal m_s->0, ~half mid), so we report the beats-rates RESOLVED BY REGIME with:
  * Wilson 95% CIs on each beats-rate (marginal / mid / pooled),
  * a two-sided exact binomial (sign) test vs 0.5 (ties dropped),
  * a two-sided paired Wilcoxon on the per-start gain differences,
  * median gains (IQR) per method, resolved.
Headlines are NEVER a bare percentage. Saves data/phase25_kappa_summary.json.
"""
import json
import os
import sys

import numpy as np

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from scipy.stats import wilcoxon, binomtest

HEURISTICS = [f"heuristic:{f}{s}" for f in ["sq_uo", "gap_outer", "li", "betap"] for s in ["+", "-"]]
GRADFREE = ["random", "cma"]


def wilson(k, n, z=1.96):
    """Wilson score 95% CI for a binomial proportion."""
    if n == 0:
        return [float("nan"), float("nan")]
    p = k / n
    den = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [float(center - half), float(center + half)]


def med_iqr(x):
    x = np.asarray(x, float)
    if len(x) == 0:
        return dict(median=float("nan"), iqr=[float("nan"), float("nan")], n=0)
    return dict(median=float(np.median(x)),
                iqr=[float(np.percentile(x, 25)), float(np.percentile(x, 75))], n=int(len(x)))


def beat_stats(sur, comp, label):
    """Two-sided sign test + Wilson CI + paired Wilcoxon for 'surrogate > comp', strict."""
    diff = sur - comp
    wins = int(np.sum(diff > 0))
    losses = int(np.sum(diff < 0))
    ties = int(np.sum(diff == 0))
    n = len(diff)
    n_eff = wins + losses               # ties dropped for the sign test
    frac = float(wins / n) if n else float("nan")
    out = dict(label=label, n=n, wins=wins, losses=losses, ties=ties,
               beats_frac=frac, wilson95=wilson(wins, n))
    try:
        out["sign_test_p_two_sided"] = float(binomtest(wins, n_eff, 0.5, alternative="two-sided").pvalue) \
            if n_eff > 0 else float("nan")
    except Exception as e:
        out["sign_test_err"] = str(e)
    try:
        if np.any(diff != 0):
            out["wilcoxon_p_two_sided"] = float(wilcoxon(sur, comp, alternative="two-sided",
                                                         zero_method="wilcox").pvalue)
        else:
            out["wilcoxon_p_two_sided"] = float("nan")
    except Exception as e:
        out["wilcoxon_err"] = str(e)
    out["median_gain_diff"] = float(np.median(diff)) if n else float("nan")
    return out


def main():
    recs = json.load(open("data/phase25_kappa_results.json"))["recs"]
    setup = json.load(open("data/phase25_kappa_setup.json"))
    starts = sorted(set(r["start_i"] for r in recs))
    regime_of = {r["start_i"]: r["regime"] for r in recs}

    def gain(start_i, method):
        g = [r["gain"] for r in recs if r["start_i"] == start_i and r["method"] == method]
        return max(g) if g else np.nan

    rows = []
    for s in starts:
        rows.append(dict(start=s, regime=regime_of[s],
                         surrogate=gain(s, "surrogate"),
                         best_heuristic=np.nanmax([gain(s, h) for h in HEURISTICS]),
                         best_gradfree=np.nanmax([gain(s, g) for g in GRADFREE])))

    def arrays(subset):
        sur = np.array([r["surrogate"] for r in subset])
        bh = np.array([r["best_heuristic"] for r in subset])
        bg = np.array([r["best_gradfree"] for r in subset])
        ok = np.isfinite(sur) & np.isfinite(bh) & np.isfinite(bg)
        return sur[ok], bh[ok], bg[ok]

    out = dict(ktol=setup["ktol"], budget=setup["budget"], n_starts_total=len(rows),
               by_regime={}, per_method_median={})

    for label, subset in [("pooled", rows),
                          ("marginal", [r for r in rows if r["regime"] == "marginal"]),
                          ("mid", [r for r in rows if r["regime"] == "mid"])]:
        sur, bh, bg = arrays(subset)
        if len(sur) == 0:
            continue
        out["by_regime"][label] = dict(
            n=int(len(sur)),
            surrogate_gain=med_iqr(sur), best_heuristic_gain=med_iqr(bh), best_gradfree_gain=med_iqr(bg),
            vs_heuristic=beat_stats(sur, bh, "surrogate>best_heuristic"),
            vs_gradfree=beat_stats(sur, bg, "surrogate>best_gradfree"),
        )

    # per-method median gain resolved by regime
    for m in ["surrogate"] + HEURISTICS + GRADFREE:
        out["per_method_median"][m] = dict(
            pooled=float(np.nanmedian([gain(s, m) for s in starts])),
            marginal=float(np.nanmedian([gain(s, m) for s in starts if regime_of[s] == "marginal"])),
            mid=float(np.nanmedian([gain(s, m) for s in starts if regime_of[s] == "mid"])),
        )
    out["rows"] = rows
    json.dump(out, open("data/phase25_kappa_summary.json", "w"), indent=2)

    # ---- print ----
    print(f"=== KAPPA-CONSTRAINED 'beats heuristics' FIRMED (kappa held +-{out['ktol']}, "
          f"budget {out['budget']}, {out['n_starts_total']} stratified starts) ===")
    for label in ["pooled", "marginal", "mid"]:
        r = out["by_regime"].get(label)
        if not r:
            continue
        print(f"\n[{label.upper()}]  n={r['n']}")
        print(f"  median gain: surrogate={r['surrogate_gain']['median']:+.3f}  "
              f"best-heuristic={r['best_heuristic_gain']['median']:+.3f}  "
              f"best-gradient-free={r['best_gradfree_gain']['median']:+.3f}")
        for key, name in [("vs_heuristic", "vs best single-lever heuristic"),
                          ("vs_gradfree", "vs best gradient-free")]:
            b = r[key]
            ci = b["wilson95"]
            print(f"  beats {name}: {b['wins']}/{b['n']} = {b['beats_frac']*100:.0f}% "
                  f"(Wilson95 {ci[0]*100:.0f}-{ci[1]*100:.0f}%; ties={b['ties']}; "
                  f"sign-test two-sided p={b.get('sign_test_p_two_sided', float('nan')):.4f}; "
                  f"Wilcoxon two-sided p={b.get('wilcoxon_p_two_sided', float('nan')):.4f})")
    print("\n  per-method median gain (pooled / marginal / mid):")
    for m, d in out["per_method_median"].items():
        print(f"    {m:18s} {d['pooled']:+.3f} / {d['marginal']:+.3f} / {d['mid']:+.3f}")
    print("\nSaved data/phase25_kappa_summary.json")


if __name__ == "__main__":
    main()
