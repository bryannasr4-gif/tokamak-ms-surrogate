"""
device2_design_analyze.py -- analyze the Device-C UNCONSTRAINED design comparison (Phase 5 C4).

Pairs the three methods PER START, classifies each start by its TRUE 80-mode start m_s regime
(marginal <0.4 / mid 0.4-1.0 / stable >=1.0), and reports the PRE-REGISTERED headline:
  surrogate (learned m_s) vs reduce_kappa (heuristic)  -- the key test
  surrogate vs cma (gradient-free)                     -- the secondary baseline
as win-rate (surrogate best_ms > comparator) with Wilson 95% CI + two-sided paired Wilcoxon +
sign test, resolved by regime, on the MAIN cohort and (separately) the DISJOINT replication cohort.

  python experiments/device2_design_analyze.py --framing zeroshot
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "experiments"))
import phase2_data as D


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, c - h), min(1.0, c + h))


def regime_of(ms):
    return D.regime_of(ms)


def compare(starts, a, b):
    """starts: list of per-start dicts each with a 'methods' {name: best_ms} map. Pair a vs b ONLY on
    starts where BOTH a and b ran (the pre-registered surrogate-vs-reduce_kappa headline must not be
    coupled to whether the unrelated cma job completed). a 'wins' if a.best_ms > b.best_ms."""
    from scipy.stats import wilcoxon, binomtest
    pairs = [s for s in starts if a in s["methods"] and b in s["methods"]]
    if not pairs:
        return dict(n=0, wins=0, losses=0, ties=0, win_rate=float("nan"), wilson95=[float("nan")] * 2,
                    wilcoxon_p=float("nan"), sign_p=float("nan"), median_a=float("nan"),
                    median_b=float("nan"), median_diff=float("nan"))
    da = np.array([p["methods"][a] for p in pairs]); db = np.array([p["methods"][b] for p in pairs])
    diff = da - db
    n = len(diff)
    wins = int(np.sum(diff > 0)); losses = int(np.sum(diff < 0)); ties = int(np.sum(diff == 0))
    lo, hi = wilson(wins, n)
    # two-sided paired Wilcoxon (drop exact zeros)
    nz = diff[diff != 0]
    try:
        wp = float(wilcoxon(nz, alternative="two-sided").pvalue) if len(nz) >= 1 else float("nan")
    except Exception:
        wp = float("nan")
    # two-sided sign test among non-tied
    nb = wins + losses
    sp = float(binomtest(wins, nb, 0.5, alternative="two-sided").pvalue) if nb > 0 else float("nan")
    return dict(n=n, wins=wins, losses=losses, ties=ties, win_rate=wins / n if n else float("nan"),
                wilson95=[lo, hi], wilcoxon_p=wp, sign_p=sp,
                median_a=float(np.median(da)), median_b=float(np.median(db)),
                median_diff=float(np.median(diff)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--framing", type=str, default="zeroshot")
    args = ap.parse_args()

    files = glob.glob(os.path.join(ROOT, "data", "device2_design_results", f"{args.framing}_job*.json"))
    recs, bad = [], 0
    for f in files:                              # robust load: a partial/corrupt file counts as missing
        try:
            recs.append(json.load(open(f)))
        except (json.JSONDecodeError, OSError):
            bad += 1
    if bad:
        print(f"WARNING: skipped {bad} unreadable/partial result file(s)")
    if not recs:
        print(f"No results for framing={args.framing}"); sys.exit(1)

    # group by start_id -> {method: best_ms} (+ start info). NO all-3 filter: each comparison pairs
    # only on the starts where BOTH compared methods ran (see compare()).
    by_start = {}
    for r in recs:
        s = by_start.setdefault(r["start_id"], dict(start_id=r["start_id"], cohort=r["cohort"],
                                                    band=r["band"], ms_start=r["ms_start"], methods={}))
        s["methods"][r["method"]] = r["best_ms"]
        s["ms_start"] = r["ms_start"]            # identical across methods (same deterministic 80-mode start)
    for s in by_start.values():
        s["regime"] = regime_of(s["ms_start"])
    all_starts = list(by_start.values())

    METHODS = ["surrogate", "reduce_kappa", "cma"]
    regimes = [name for name, _, _ in D.REGIMES] + ["POOLED"]
    setup_n = json.load(open(os.path.join(ROOT, "data", "device2_design_setup.json"))).get("n_starts")
    out = dict(framing=args.framing, n_starts_seen=len(all_starts), n_starts_setup=setup_n,
               method_counts={m: sum(1 for s in all_starts if m in s["methods"]) for m in METHODS})
    print(f"=== DEVICE-C UNCONSTRAINED DESIGN COMPARISON ({args.framing}) ===")
    print(f"starts seen: {len(all_starts)}/{setup_n}; per-method completed: {out['method_counts']}")

    for cohort in ["main", "replication", "ALL"]:
        ps = all_starts if cohort == "ALL" else [p for p in all_starts if p["cohort"] == cohort]
        if not ps:
            continue
        out[cohort] = {}
        print(f"\n----- cohort: {cohort} (n={len(ps)} starts) -----")
        for reg in regimes:
            sub = ps if reg == "POOLED" else [p for p in ps if p["regime"] == reg]
            if len(sub) < 1:
                continue
            vs_h = compare(sub, "surrogate", "reduce_kappa")
            vs_c = compare(sub, "surrogate", "cma")
            med = {m: (float(np.median([p["methods"][m] for p in sub if m in p["methods"]]))
                       if any(m in p["methods"] for p in sub) else float("nan")) for m in METHODS}
            out[cohort][reg] = dict(n_starts=len(sub), surrogate_vs_reduce_kappa=vs_h,
                                    surrogate_vs_cma=vs_c, median_best=med)
            print(f"  [{reg:11s} n={len(sub):2d}] surr vs reduce_kappa: "
                  f"{vs_h['wins']}/{vs_h['n']}={vs_h['win_rate']:.0%} "
                  f"CI[{vs_h['wilson95'][0]:.2f},{vs_h['wilson95'][1]:.2f}] "
                  f"Wilcoxon p={vs_h['wilcoxon_p']:.4f} sign p={vs_h['sign_p']:.4f} "
                  f"| med best_ms surr {med['surrogate']:.3f} vs redK {med['reduce_kappa']:.3f} "
                  f"vs cma {med['cma']:.3f}")
            print(f"  {'':15s} surr vs cma         : "
                  f"{vs_c['wins']}/{vs_c['n']}={vs_c['win_rate']:.0%} Wilcoxon p={vs_c['wilcoxon_p']:.4f}")

    json.dump(out, open(os.path.join(ROOT, "data", f"device2_design_{args.framing}_summary.json"), "w"), indent=2)
    print(f"\nSaved data/device2_design_{args.framing}_summary.json")


if __name__ == "__main__":
    main()
