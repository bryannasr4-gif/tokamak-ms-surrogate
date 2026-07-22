"""
phase4_kappa_pooled.py -- POOL the Phase-2.5b n=20 kappa-constrained run (phase25_kappa_results.json)
with the Phase-4 power top-up (phase4_power_results.json, +12 disjoint starts; 10 marginal + 2 mid)
and re-run the realizable-baseline analysis on the combined n=32 (marginal n=20, mid n=12). Same
loop / KTOL / budget / PCA setup, so the runs are poolable. NO new solves.

Reuses the helper functions + baseline definitions from phase4_kappa_reanalyze. Power start indices
are offset by +1000 to keep them distinct from the original 20. Saves data/phase4_kappa_pooled.json.
"""
import json
import os
import sys

import numpy as np

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.getcwd(), "experiments"))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from phase4_kappa_reanalyze import HEURISTICS, GRADFREE, beat, med


def main():
    import glob
    orig = json.load(open("data/phase25_kappa_results.json"))["recs"]
    recs = list(orig)
    batch_starts = {}                       # batch file -> set of pooled start ids (standalone replication)
    for bi, fn in enumerate(sorted(glob.glob("data/phase4_power*_results.json"))):
        off = 1000 * (bi + 1)               # distinct offset per batch so start ids never collide
        batch = json.load(open(fn))["recs"]
        recs += [dict(r, start_i=r["start_i"] + off) for r in batch]
        batch_starts[os.path.basename(fn)] = {r["start_i"] + off for r in batch}
    setup = json.load(open("data/phase25_kappa_setup.json"))
    starts = sorted(set(r["start_i"] for r in recs))
    regime = {r["start_i"]: r["regime"] for r in recs}

    def gain(si, m):
        g = [r["gain"] for r in recs if r["start_i"] == si and r["method"] == m]
        return max(g) if g else np.nan

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
               n_orig=len(set(r["start_i"] for r in orig)),
               n_batches={fn: len(s) for fn, s in batch_starts.items()},
               best_fixed_lever=best_fixed_lever, lever_pooled_median=lever_pooled_median,
               n_levers=len(HEURISTICS), by_regime={})

    # STANDALONE out-of-sample REPLICATION of the most-recent (largest) batch's MARGINAL cohort,
    # analyzed before pooling (pre-registered per the quality audit): does the 70% win-rate replicate
    # on fresh disjoint marginal starts, not just survive in the pool?
    if batch_starts:
        # the batch with the MOST marginal starts = the strongest standalone replication cohort
        latest = max(batch_starts, key=lambda fn: sum(1 for si in batch_starts[fn]
                                                      if regime.get(si) == "marginal"))
        repl_ids = [si for si in batch_starts[latest] if regime.get(si) == "marginal"]
        repl_rows = [r for r in rows if r["start"] in set(repl_ids)]
        rsur = arr(repl_rows, "surrogate"); rfix = arr(repl_rows, "best_fixed_lever")
        ok = np.isfinite(rsur) & np.isfinite(rfix)
        if ok.sum() > 0:
            out["marginal_replication"] = dict(batch=latest, n=int(ok.sum()),
                vs_best_fixed_lever=beat(rsur[ok], rfix[ok]),
                median_gain_surrogate=med(rsur[ok]), median_gain_fixed=med(rfix[ok]))
    for label, subset in [("pooled", rows),
                          ("marginal", [r for r in rows if r["regime"] == "marginal"]),
                          ("mid", [r for r in rows if r["regime"] == "mid"])]:
        keys = ["surrogate", "best_fixed_lever", "oracle_lever", "best_gradfree"]
        ok = np.all([np.isfinite(arr(subset, k)) for k in keys], axis=0)
        sur = arr(subset, "surrogate")[ok]
        if len(sur) == 0:
            continue
        out["by_regime"][label] = dict(
            n=int(len(sur)),
            median_gain={k: med(arr(subset, k)[ok]) for k in keys},
            vs_best_fixed_lever=beat(sur, arr(subset, "best_fixed_lever")[ok]),
            vs_oracle_lever=beat(sur, arr(subset, "oracle_lever")[ok]),
            vs_best_gradfree=beat(sur, arr(subset, "best_gradfree")[ok]))
    out["rows"] = rows
    json.dump(out, open("data/phase4_kappa_pooled.json", "w"), indent=2)

    print(f"=== POOLED kappa-constrained (n={out['n_starts']} = {out['n_orig']} orig + "
          f"{sum(out['n_batches'].values())} top-up; marginal {sum(1 for r in rows if r['regime']=='marginal')} / "
          f"mid {sum(1 for r in rows if r['regime']=='mid')}) ===")
    rep = out.get("marginal_replication")
    if rep:
        b = rep["vs_best_fixed_lever"]
        print(f"OUT-OF-SAMPLE REPLICATION (new disjoint marginal batch {rep['batch']}, n={rep['n']}): "
              f"surrogate beats best fixed lever {b['wins']}/{b['n']} ({b['beats_frac']*100:.0f}%), "
              f"Wilcoxon p={b['wilcoxon_p']:.4f}, sign p={b['sign_p']:.4f}; "
              f"median gain surrogate {rep['median_gain_surrogate']:+.3f} vs fixed {rep['median_gain_fixed']:+.3f}")
    print(f"best realizable fixed lever = {best_fixed_lever} (pooled median {lever_pooled_median[best_fixed_lever]:+.3f})")
    for label in ["pooled", "marginal", "mid"]:
        r = out["by_regime"].get(label)
        if not r:
            continue
        mg = r["median_gain"]
        print(f"\n[{label.upper()}] n={r['n']}  median gain: surrogate={mg['surrogate']:+.3f}  "
              f"fixed_lever={mg['best_fixed_lever']:+.3f}  oracle={mg['oracle_lever']:+.3f}  "
              f"gradient_free={mg['best_gradfree']:+.3f}")
        for key, nm in [("vs_best_fixed_lever", "vs best REALIZABLE fixed lever"),
                        ("vs_oracle_lever", "vs ORACLE best-of-8 (not realizable)"),
                        ("vs_best_gradfree", "vs best gradient-free")]:
            b = r[key]
            print(f"  {nm:38s}: {b['wins']}/{b['n']} ({b['beats_frac']*100:.0f}%) "
                  f"Wilson95 {b['wilson95'][0]*100:.0f}-{b['wilson95'][1]*100:.0f}%  "
                  f"sign_p={b['sign_p']:.4f}  wilcoxon_p={b['wilcoxon_p']:.4f}  med_diff={b['median_diff']:+.3f}")
    print("\nSaved data/phase4_kappa_pooled.json")


if __name__ == "__main__":
    main()
