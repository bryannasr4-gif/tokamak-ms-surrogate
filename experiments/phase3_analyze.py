"""
phase3_analyze.py -- the Phase-3 design-loop verdict (PRE-REGISTERED before the run completed).

Question: does the AMORTIZED differentiable surrogate design loop reach stability targets with FEWER
expensive true solves (and higher final true m_s) than gradient-free search on the true solver -- and
how does it compare to the strongest single-lever physics heuristic? Resolved by m_s regime over the
stratified marginal+mid starts; headlines are NEVER a bare percentage.

PRE-REGISTERED statistics (decided before seeing results, mirroring the reviewed Phase-2.5b protocol).
  NOTE (post-adversarial-review reconciliation): the pre-registered primary was per-start GAIN, but the
  data showed final-margin GAIN is CONFOUNDED past the m*=1.0 crossing (step/population overshoot in all
  methods + mid starts near target), so the reported PRIMARY ENDPOINT is the pooled true-solves-to-target
  m*=1.0 paired Wilcoxon (p=0.000885). Gain is still reported (it is within noise) but de-headlined; this
  metric switch is disclosed, not silent. "best gradient-free" = per-start oracle min-solves / max-gain over
  {cma, random, nelder} (the CONSERVATIVE / strongest envelope; CMA alone takes median 15 solves).
  (1) surrogate vs best gradient-free  -- the DECISIVE, FAIR comparison (lead with this).
  (2) surrogate vs reduce-kappa heuristic -- reported honestly (expected near-tie on this
      kappa-dominated single-machine manifold; do NOT headline a heuristic win).
  For each: wins/n, Wilson 95% CI, two-sided sign test, two-sided paired Wilcoxon, median gain (IQR).
  (3) reach-rate of m*=1.0 per method (Wilson CI); (4) true-solves-to-target {0.3,0.5,0.7,1.0}
  median per method (the amortization/efficiency story); (5) reject-reason mix per method
  (how often the diverted-manifold / in-range guard binds). All resolved pooled / marginal / mid.
Every m_s gain is stated against the noise floor (within-config bit-reproducible at OMP=1, 80 modes;
gains here are >> the residual systematic floor).
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

GRADFREE = ["cma", "random", "nelder"]
TARGET_KEYS = ["0.3", "0.5", "0.7", "1.0"]
PRIMARY = "1.0"
# CMA default popsize at d=12 = 4 + floor(3*ln(12)) = 11 (the main run used the default). Under a
# PARALLEL deployment a whole CMA generation could be solved at once, so its wall-clock "rounds" to
# target = ceil(solves / popsize); the sequential surrogate/heuristic/random/nelder loops cannot be
# parallelized this way (each step depends on the previous confirmed solve). Used for the
# framing-honest PARALLEL accounting (adversarial fix `batch-vs-sequential-solves-to-target`).
CMA_POPSIZE = 11
SEQUENTIAL = ["surrogate", "heuristic", "random", "nelder"]


def wilson(k, n, z=1.96):
    if n == 0:
        return [float("nan"), float("nan")]
    p = k / n
    den = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [float(center - half), float(center + half)]


def med_iqr(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return dict(median=float("nan"), iqr=[float("nan"), float("nan")], n=0)
    return dict(median=float(np.median(x)),
                iqr=[float(np.percentile(x, 25)), float(np.percentile(x, 75))], n=int(len(x)))


def beat_stats(sur, comp, label):
    """Two-sided sign test + Wilson CI + paired Wilcoxon for strict 'surrogate > comp'."""
    diff = sur - comp
    wins = int(np.sum(diff > 0)); losses = int(np.sum(diff < 0)); ties = int(np.sum(diff == 0))
    n = len(diff); n_eff = wins + losses
    out = dict(label=label, n=n, wins=wins, losses=losses, ties=ties,
               beats_frac=float(wins / n) if n else float("nan"), wilson95=wilson(wins, n),
               median_gain_diff=float(np.median(diff)) if n else float("nan"))
    try:
        out["sign_test_p_two_sided"] = float(binomtest(wins, n_eff, 0.5, alternative="two-sided").pvalue) \
            if n_eff > 0 else float("nan")
    except Exception as e:
        out["sign_test_err"] = str(e)
    try:
        out["wilcoxon_p_two_sided"] = float(wilcoxon(sur, comp, alternative="two-sided",
                                                     zero_method="wilcox").pvalue) if np.any(diff != 0) else float("nan")
    except Exception as e:
        out["wilcoxon_err"] = str(e)
    return out


def paired_solves(svals, bvals):
    """Paired comparison of two censored solves-to-target vectors (lower = faster). Returns
    faster/slower/tied counts + two-sided Wilcoxon + median saving (b - s)."""
    s, b = np.array(svals, float), np.array(bvals, float)
    faster = int(np.sum(s < b)); slower = int(np.sum(s > b)); tied = int(np.sum(s == b))
    try:
        wp = float(wilcoxon(s, b, alternative="two-sided", zero_method="wilcox").pvalue) \
            if np.any(s != b) else float("nan")
    except Exception:
        wp = float("nan")
    return dict(faster=faster, slower=slower, tied=tied, wilcoxon_p=wp,
                median_saving=float(np.median(b - s)))


def main():
    recs = json.load(open("data/phase3_results.json"))["recs"]
    setup = json.load(open("data/phase3_setup.json"))
    budget = setup["budget"]
    starts = sorted(set(r["start_i"] for r in recs))
    regime_of = {r["start_i"]: r["regime"] for r in recs}
    methods = setup["methods"]

    def rec_of(si, m):
        rs = [r for r in recs if r["start_i"] == si and r["method"] == m]
        return rs[0] if rs else None

    def gain(si, m):
        r = rec_of(si, m); return r["gain"] if r else np.nan

    def final(si, m):
        r = rec_of(si, m); return r["best_ms"] if r else np.nan

    def s2t(si, m, tk):
        r = rec_of(si, m)
        if not r:
            return None
        return r["solves_to_target"].get(tk)

    # per-start table
    rows = []
    for si in starts:
        row = dict(start=si, regime=regime_of[si],
                   m_s_start=rec_of(si, methods[0])["m_s_start"])
        for m in methods:
            row[f"gain_{m}"] = gain(si, m)
            row[f"final_{m}"] = final(si, m)
        row["gain_best_gradfree"] = np.nanmax([gain(si, m) for m in GRADFREE])
        row["final_best_gradfree"] = np.nanmax([final(si, m) for m in GRADFREE])
        rows.append(row)

    out = dict(budget=budget, d=setup["d"], target=setup["target"], n_starts=len(rows),
               by_regime={}, per_method={}, solves_to_target={}, reject_mix={})

    # (1)+(2) paired beats-rates, resolved by regime
    for label, subset in [("pooled", rows),
                          ("marginal", [r for r in rows if r["regime"] == "marginal"]),
                          ("mid", [r for r in rows if r["regime"] == "mid"])]:
        if not subset:
            continue
        sur = np.array([r["gain_surrogate"] for r in subset])
        heur = np.array([r["gain_heuristic"] for r in subset])
        bg = np.array([r["gain_best_gradfree"] for r in subset])
        ok = np.isfinite(sur) & np.isfinite(heur) & np.isfinite(bg)
        sur, heur, bg = sur[ok], heur[ok], bg[ok]
        ent = dict(n=int(ok.sum()),
                   surrogate_gain=med_iqr(sur), heuristic_gain=med_iqr(heur), best_gradfree_gain=med_iqr(bg),
                   vs_gradfree=beat_stats(sur, bg, "surrogate>best_gradfree"),
                   vs_heuristic=beat_stats(sur, heur, "surrogate>heuristic"))
        # reach-rate of m*=1.0 per method
        sub_si = [r["start"] for r in subset]
        ent["reach_rate"] = {}
        for m in methods + ["best_gradfree"]:
            if m == "best_gradfree":
                reached = [int(any((rec_of(si, g) or {}).get("reached_primary") for g in GRADFREE)) for si in sub_si]
            else:
                reached = [int((rec_of(si, m) or {}).get("reached_primary", False)) for si in sub_si]
            k = int(np.sum(reached))
            ent["reach_rate"][m] = dict(k=k, n=len(reached), frac=k / len(reached) if reached else float("nan"),
                                        wilson95=wilson(k, len(reached)))
        out["by_regime"][label] = ent

    # (3) per-method median final m_s + gain, resolved
    for m in methods + ["best_gradfree"]:
        out["per_method"][m] = {}
        for label in ["pooled", "marginal", "mid"]:
            sub = starts if label == "pooled" else [s for s in starts if regime_of[s] == label]
            gm = "gain_best_gradfree" if m == "best_gradfree" else None
            gains = [next(r[gm] if gm else r[f"gain_{m}"] for r in rows if r["start"] == si) for si in sub]
            finals = [next(r["final_best_gradfree"] if m == "best_gradfree" else r[f"final_{m}"]
                           for r in rows if r["start"] == si) for si in sub]
            out["per_method"][m][label] = dict(median_gain=float(np.nanmedian(gains)),
                                               median_final=float(np.nanmedian(finals)))

    # (4) true-solves-to-target (median among reachers) + reach-rate at each target
    for tk in TARGET_KEYS:
        out["solves_to_target"][tk] = {}
        for m in methods:
            for label in ["pooled", "marginal", "mid"]:
                sub = starts if label == "pooled" else [s for s in starts if regime_of[s] == label]
                vals = [s2t(si, m, tk) for si in sub]
                reached = [v for v in vals if v is not None]
                out["solves_to_target"][tk][f"{m}:{label}"] = dict(
                    reach_k=len(reached), reach_n=len(vals),
                    median_solves_among_reachers=float(np.median(reached)) if reached else None)

    # (6) AMORTIZATION (the headline, unconfounded by CMA batch-overshoot of the stop-at-target rule):
    #     paired true-solves-to-target (right-censored at budget+1 for non-reach) and McNemar reach,
    #     surrogate vs the BEST (fewest-solve) gradient-free optimizer per start.
    CENS = budget + 1
    out["amortization"] = {}
    for label in ["pooled", "marginal", "mid"]:
        sub = starts if label == "pooled" else [s for s in starts if regime_of[s] == label]
        sur_s, bg_s, sur_reach, bg_reach = [], [], [], []
        for si in sub:
            s_sur = (rec_of(si, "surrogate") or {}).get("solves_to_target", {}).get(PRIMARY)
            gf = [(rec_of(si, m) or {}).get("solves_to_target", {}).get(PRIMARY) for m in GRADFREE]
            gf = [v for v in gf if v is not None]
            s_bg = min(gf) if gf else None
            sur_reach.append(s_sur is not None); bg_reach.append(s_bg is not None)
            sur_s.append(s_sur if s_sur is not None else CENS)
            bg_s.append(s_bg if s_bg is not None else CENS)
        sur_s, bg_s = np.array(sur_s), np.array(bg_s)
        faster = int(np.sum(sur_s < bg_s)); slower = int(np.sum(sur_s > bg_s)); tied = int(np.sum(sur_s == bg_s))
        # McNemar exact: starts where surrogate reached but best-gradfree did not, vs vice-versa
        only_sur = int(np.sum([sr and not br for sr, br in zip(sur_reach, bg_reach)]))
        only_bg = int(np.sum([br and not sr for sr, br in zip(sur_reach, bg_reach)]))
        mcn_p = float(binomtest(min(only_sur, only_bg), only_sur + only_bg, 0.5,
                                alternative="two-sided").pvalue) if (only_sur + only_bg) > 0 else float("nan")
        try:
            wp = float(wilcoxon(sur_s, bg_s, alternative="two-sided", zero_method="wilcox").pvalue) \
                if np.any(sur_s != bg_s) else float("nan")
        except Exception:
            wp = float("nan")
        k_sur, k_bg = int(np.sum(sur_reach)), int(np.sum(bg_reach))
        out["amortization"][label] = dict(
            n=len(sub),
            surrogate_reach=dict(k=k_sur, n=len(sub), wilson95=wilson(k_sur, len(sub))),
            best_gradfree_reach=dict(k=k_bg, n=len(sub), wilson95=wilson(k_bg, len(sub))),
            surrogate_median_solves=float(np.median([s for s, r in zip(sur_s, sur_reach) if r])) if k_sur else None,
            best_gradfree_median_solves=float(np.median([s for s, r in zip(bg_s, bg_reach) if r])) if k_bg else None,
            paired_solves_surrogate_faster=faster, slower=slower, tied=tied,
            paired_solves_wilcoxon_p=wp, median_solve_saving=float(np.median(bg_s - sur_s)),
            mcnemar_only_surrogate=only_sur, mcnemar_only_gradfree=only_bg, mcnemar_p_two_sided=mcn_p)

    # (6b) ATTRIBUTION + PARALLEL framing (adversarial fixes `amortization-win-is-the-free-kappa-prior`,
    #      `c2-heuristic-also-beats-gradfree-marginal`, `batch-vs-sequential-solves-to-target`):
    #   - the reduce-kappa heuristic (NO learned m_s labels, only differentiable geometry) ALSO beats
    #     gradient-free on solves -> the efficiency win is the GRADIENT DESIGN LOOP, not specifically m_s;
    #   - surrogate vs heuristic on solves is within noise (consistent with the C3 final-margin tie);
    #   - PARALLEL accounting: if a CMA generation (popsize 11) is solved at once, its cost = rounds =
    #     ceil(solves/popsize); the surrogate's per-charged-solve edge shrinks (framing-honest).
    out["attribution"] = {}
    for label in ["pooled", "marginal", "mid"]:
        sub = starts if label == "pooled" else [s for s in starts if regime_of[s] == label]

        def s2t_c(si, m):                      # censored solves-to-target (lower=faster)
            v = (rec_of(si, m) or {}).get("solves_to_target", {}).get(PRIMARY)
            return v if v is not None else CENS

        def bg_serial(si):                     # serial best-of-3 gradient-free
            return min(s2t_c(si, m) for m in GRADFREE)

        def bg_parallel(si):                   # parallel: CMA rounds vs serial random/nelder
            cma = s2t_c(si, "cma"); cma_r = CENS if cma >= CENS else int(np.ceil(cma / CMA_POPSIZE))
            return min(cma_r, s2t_c(si, "random"), s2t_c(si, "nelder"))

        sur = [s2t_c(si, "surrogate") for si in sub]
        heu = [s2t_c(si, "heuristic") for si in sub]
        bgs = [bg_serial(si) for si in sub]
        bgp = [bg_parallel(si) for si in sub]
        out["attribution"][label] = dict(
            n=len(sub),
            heuristic_vs_gradfree_serial=paired_solves(heu, bgs),
            surrogate_vs_heuristic=paired_solves(sur, heu),
            surrogate_vs_gradfree_PARALLEL=paired_solves(sur, bgp))

    # (5) reject-reason mix per method (transparency on the manifold / in-range guard)
    for m in methods:
        agg = dict(invalid=0, out_of_range=0, no_improve=0, n_runs=0, total_solves=0)
        for si in starts:
            r = rec_of(si, m)
            if not r:
                continue
            agg["n_runs"] += 1
            agg["total_solves"] += r["n_solves"]
            for kk, vv in (r.get("reject") or {}).items():
                agg[kk] = agg.get(kk, 0) + vv
        out["reject_mix"][m] = agg

    out["rows"] = rows
    json.dump(out, open("data/phase3_summary.json", "w"), indent=2)

    # ---------------- print ----------------
    print(f"=== PHASE 3 design loop: surrogate vs gradient-free vs reduce-kappa heuristic ===")
    print(f"    budget={budget} true solves, d={setup['d']}, target m*={setup['target']}, "
          f"{out['n_starts']} stratified starts\n")
    for label in ["pooled", "marginal", "mid"]:
        e = out["by_regime"].get(label)
        if not e:
            continue
        print(f"[{label.upper()}] n={e['n']}")
        print(f"  median gain: surrogate={e['surrogate_gain']['median']:+.3f}  "
              f"heuristic={e['heuristic_gain']['median']:+.3f}  "
              f"best-gradient-free={e['best_gradfree_gain']['median']:+.3f}")
        for key, name in [("vs_gradfree", "vs best gradient-free  [LEAD]"),
                          ("vs_heuristic", "vs reduce-kappa heuristic")]:
            b = e[key]; ci = b["wilson95"]
            print(f"  surrogate beats {name}: {b['wins']}/{b['n']} = {b['beats_frac'] * 100:.0f}% "
                  f"(Wilson {ci[0] * 100:.0f}-{ci[1] * 100:.0f}%; ties={b['ties']}; "
                  f"sign p={b.get('sign_test_p_two_sided', float('nan')):.4f}; "
                  f"Wilcoxon p={b.get('wilcoxon_p_two_sided', float('nan')):.4f})")
        rr = e["reach_rate"]
        print("  reach m*=1.0: " + "  ".join(
            f"{m}={rr[m]['k']}/{rr[m]['n']}" for m in ["surrogate", "heuristic", "best_gradfree"]))
        print()
    print("=== AMORTIZATION (headline): surrogate vs BEST gradient-free, true-solves-to-target m*=1.0 ===")
    for label in ["pooled", "marginal", "mid"]:
        am = out["amortization"][label]
        sr, br = am["surrogate_reach"], am["best_gradfree_reach"]
        print(f"[{label.upper()}] n={am['n']}")
        print(f"  reach m*=1.0:  surrogate {sr['k']}/{sr['n']} "
              f"(Wilson {sr['wilson95'][0]*100:.0f}-{sr['wilson95'][1]*100:.0f}%)   "
              f"best-gradient-free {br['k']}/{br['n']} "
              f"(Wilson {br['wilson95'][0]*100:.0f}-{br['wilson95'][1]*100:.0f}%)")
        print(f"  median solves-to-target (reachers): surrogate={am['surrogate_median_solves']}  "
              f"best-gradient-free={am['best_gradfree_median_solves']}")
        print(f"  paired SERIAL solves (censored@{CENS}): surrogate faster in {am['paired_solves_surrogate_faster']}, "
              f"slower {am['slower']}, tied {am['tied']} (Wilcoxon p={am['paired_solves_wilcoxon_p']:.4f}; "
              f"median saving {am['median_solve_saving']:+.1f} solves)")
        print(f"  McNemar reach (NOT the speed test): only-surrogate={am['mcnemar_only_surrogate']}, "
              f"only-gradient-free={am['mcnemar_only_gradfree']} (exact p={am['mcnemar_p_two_sided']:.4f})")
        at = out["attribution"][label]
        h = at["heuristic_vs_gradfree_serial"]; sh = at["surrogate_vs_heuristic"]; pp = at["surrogate_vs_gradfree_PARALLEL"]
        print(f"  ATTRIBUTION: heuristic(no m_s labels) vs gradient-free serial: faster {h['faster']}/{at['n']} "
              f"(p={h['wilcoxon_p']:.4f}, save {h['median_saving']:+.1f})  |  surrogate vs heuristic: "
              f"faster {sh['faster']}/{at['n']} (p={sh['wilcoxon_p']:.4f}, save {sh['median_saving']:+.1f})")
        print(f"  PARALLEL framing (CMA pop={CMA_POPSIZE} solved at once): surrogate vs best-parallel-gradient-free "
              f"faster {pp['faster']}/{at['n']} (p={pp['wilcoxon_p']:.4f}, save {pp['median_saving']:+.1f} rounds)\n")
    print("true-solves-to-target=1.0 (median among reachers / reach count):")
    for m in methods:
        d = out["solves_to_target"]["1.0"][f"{m}:pooled"]
        ms = d["median_solves_among_reachers"]
        print(f"  {m:10s} reached {d['reach_k']}/{d['reach_n']}, "
              f"median solves={ms if ms is None else f'{ms:.0f}'}")
    print("\nreject mix (per method, summed over starts):")
    for m, a in out["reject_mix"].items():
        print(f"  {m:10s} runs={a['n_runs']} solves={a['total_solves']} "
              f"invalid={a.get('invalid', 0)} out_of_range={a.get('out_of_range', 0)} "
              f"no_improve={a.get('no_improve', 0)}")
    print("\nSaved data/phase3_summary.json")


if __name__ == "__main__":
    main()
