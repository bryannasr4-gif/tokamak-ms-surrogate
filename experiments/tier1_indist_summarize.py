"""
tier1_indist_summarize.py -- Stage-0 S0.2 TASK d: recompute the Tier-1 in-distribution-control summary
(data/tier1_indist_summary.json) from the per-chunk files in data/tier1_indist/.

The original summary (committed at 44bbc27, n=20) was produced by an inline computation with NO
committed summarizer script. This script RECONSTRUCTS that aggregation and is VALIDATED to reproduce
the committed n=20 values exactly (re_betap_med 1.1235, betap_true_med 0.2315, betap_inflation 4.85,
native_resid_med 0.0319, resolve_resid_med_indist 0.1849, n_ok 19). Run with --validate to assert that
reproduction on the current directory before trusting a recompute.

Aggregation (reverse-engineered + validated against the committed artifact):
  n                        = number of chunk files
  ok                       = chunks with status == "ok"; n_ok = len(ok)
  re_betap_med             = median(re_betap over ok), round 4dp
  betap_true_med           = median(betap_true over ok), round 4dp
  betap_inflation          = re_betap_med / betap_true_med (from unrounded medians), round 2dp
  native_resid_med         = median(native_resid over ok), round 4dp
  resolve_resid_med_indist = median(resolve_resid over chunks that have it), round 4dp

  python experiments/tier1_indist_summarize.py [--validate] [--write]
"""
import os, sys, json, glob, argparse
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INDIR = os.path.join(ROOT, "data", "tier1_indist")
OUT = os.path.join(ROOT, "data", "tier1_indist_summary.json")

# committed n=20 reference values (44bbc27) for --validate
REF_N20 = dict(n=20, n_ok=19, re_betap_med=1.1235, betap_true_med=0.2315,
               betap_inflation=4.85, native_resid_med=0.0319, resolve_resid_med_indist=0.1849)


def _med(vals):
    vals = [float(v) for v in vals if v is not None and np.isfinite(v)]
    return float(np.median(vals)) if vals else None, len(vals)


def compute():
    rows = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(INDIR, "*.json")))]
    n = len(rows)
    ok = [r for r in rows if r.get("status") == "ok"]
    n_ok = len(ok)
    re_betap_raw, _ = _med([r.get("re_betap") for r in ok])
    betap_true_raw, _ = _med([r.get("betap_true") for r in ok])
    native_raw, _ = _med([r.get("native_resid") for r in ok])
    resolve_raw, n_resolve = _med([r.get("resolve_resid") for r in ok])
    inflation_raw = (re_betap_raw / betap_true_raw) if (re_betap_raw and betap_true_raw) else None
    summ = dict(
        n=n, n_ok=n_ok,
        re_betap_med=round(re_betap_raw, 4) if re_betap_raw is not None else None,
        betap_true_med=round(betap_true_raw, 4) if betap_true_raw is not None else None,
        betap_inflation=round(inflation_raw, 2) if inflation_raw is not None else None,
        native_resid_med=round(native_raw, 4) if native_raw is not None else None,
        resolve_resid_med_indist=round(resolve_raw, 4) if resolve_raw is not None else None,
        note=("in-dist pipeline floor; real-shape resolve_resid median=0.61 => OOD excess ~ "
              "(0.61 - indist_floor)"),
        n_resolve_resid=n_resolve,
    )
    return summ, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", action="store_true", help="assert reproduction of committed n=20 values")
    ap.add_argument("--write", action="store_true", help="write data/tier1_indist_summary.json")
    args = ap.parse_args()
    summ, rows = compute()
    print(json.dumps(summ, indent=2))
    if args.validate:
        for k, v in REF_N20.items():
            got = summ.get(k)
            assert got == v, f"VALIDATE FAIL: {k} got {got} expected {v}"
        print("VALIDATE OK: reproduces committed n=20 reference exactly.")
    if args.write:
        # preserve exactly the committed key order (n, n_ok, re_betap_med, betap_true_med,
        # betap_inflation, native_resid_med, resolve_resid_med_indist, note) + append n_resolve_resid
        ordered = {k: summ[k] for k in ["n", "n_ok", "re_betap_med", "betap_true_med",
                                        "betap_inflation", "native_resid_med",
                                        "resolve_resid_med_indist", "note", "n_resolve_resid"]}
        with open(OUT, "w") as f:
            json.dump(ordered, f, indent=2)
        print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
