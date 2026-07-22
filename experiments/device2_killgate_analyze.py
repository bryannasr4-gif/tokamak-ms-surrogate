"""
device2_killgate_analyze.py -- pool the Device-C kill-gate chunks and read the GO/NO-GO verdict.

Pools data/device2_kg_chunk_*.json -> corr(kappa, log m_s) (Spearman, with a bootstrap 95% CI)
and per-lever |Spearman(descriptor, log m_s)|. Applies the PRE-REGISTERED rule
(data/phase5_killgate_prereg.json): GO if |corr(kappa,log m_s)| < 0.75 OR a secondary lever's
|corr| >= 0.7*|corr(kappa)|. Saves data/device2_killgate.json + data/device2_probe.parquet.

  python experiments/device2_killgate_analyze.py [--tag <suffix>]
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LEVERS = ["kappa", "sq_uo", "sq_lo", "sq_ui", "sq_li", "gap_inner", "gap_outer",
          "gap_top", "gap_bot", "li", "betap", "delta", "delta_upper", "delta_lower"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", type=str, default=os.path.join(ROOT, "data", "device2_kg_chunk_*.json"))
    ap.add_argument("--tag", type=str, default="", help="suffix for output files (e.g. _A22 for an escalated build)")
    ap.add_argument("--prereg", type=str, default=os.path.join(ROOT, "data", "phase5_killgate_prereg.json"))
    args = ap.parse_args()

    import pandas as pd
    from scipy.stats import spearmanr

    rows = []
    files = sorted(glob.glob(args.glob))
    for fp in files:
        with open(fp) as f:
            d = json.load(f)
        rows.extend(d.get("recs", []))
    if len(rows) < 30:
        print(f"BLOCKED: only {len(rows)} pooled samples across {len(files)} chunks. "
              f"Run more workers / tune the sampler.")
        sys.exit(1)

    df = pd.DataFrame(rows)
    df = df[np.isfinite(df["m_s"]) & (df["m_s"] > 0)].reset_index(drop=True)
    logms = np.log(df["m_s"].values)
    kappa = df["kappa"].values

    corr_k = float(spearmanr(kappa, logms).statistic)
    kdom = abs(corr_k)

    # bootstrap 95% CI on the Spearman corr (rank correlation, resampled rows)
    rng = np.random.default_rng(20260626)
    n = len(df)
    boots = []
    for _ in range(2000):
        idx = rng.integers(0, n, n)
        boots.append(spearmanr(kappa[idx], logms[idx]).statistic)
    ci = [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]

    sens = {}
    for f in LEVERS:
        if f in df.columns and np.ptp(df[f].values) > 0:
            sens[f] = abs(float(spearmanr(df[f].values, logms).statistic))
    rival_items = [(f, v) for f, v in sens.items() if f != "kappa"]
    rival_lever, rival = max(rival_items, key=lambda kv: kv[1], default=(None, 0.0))

    go = (kdom < 0.75) or (rival >= 0.7 * kdom)

    out = dict(
        n=int(n), n_chunks=len(files),
        corr_kappa_logms=corr_k, corr_kappa_abs=kdom, corr_kappa_ci95=ci,
        lever_sensitivity=sens, best_rival_lever=rival_lever, best_rival_sensitivity=rival,
        rival_over_kappa=float(rival / kdom) if kdom > 0 else None,
        GO=bool(go),
        kappa_range=[float(df["kappa"].min()), float(df["kappa"].max())],
        ms_range=[float(df["m_s"].min()), float(df["m_s"].max())],
        aspect_median=float(df["aspect"].median()) if "aspect" in df else None,
        mast_u_reference_corr=-0.875,
        decision_rule="GO if |corr(kappa,log m_s)|<0.75 OR best secondary lever |corr|>=0.7*|corr(kappa)|",
    )
    jpath = os.path.join(ROOT, "data", f"device2_killgate{args.tag}.json")
    ppath = os.path.join(ROOT, "data", f"device2_probe{args.tag}.parquet")
    json.dump(out, open(jpath, "w"), indent=2)
    df.to_parquet(ppath)

    print(f"\n=== DEVICE-C KILL-GATE (n={n} over {len(files)} chunks, "
          f"A_median~{out['aspect_median']:.2f}) ===")
    print(f"kappa range [{out['kappa_range'][0]:.2f}, {out['kappa_range'][1]:.2f}]  "
          f"m_s range [{out['ms_range'][0]:.3f}, {out['ms_range'][1]:.3f}]")
    print(f"corr(kappa, log m_s) = {corr_k:+.3f}  95%CI [{ci[0]:+.3f}, {ci[1]:+.3f}]  "
          f"(MAST-U = -0.875; |corr| = {kdom:.3f})")
    print("per-lever |Spearman(feat, log m_s)| (top 8):")
    for f, v in sorted(sens.items(), key=lambda kv: -kv[1])[:8]:
        mark = " <- kappa" if f == "kappa" else (" <- best rival" if f == rival_lever else "")
        print(f"   {f:12s} {v:.3f}{mark}")
    print(f"best secondary lever = {rival_lever} ({rival:.3f}); rival/kappa = {out['rival_over_kappa']:.2f}")
    print(f"\nVERDICT: {'GO' if go else 'NO-GO'}")
    print("  GO -> kappa de-dominated (or a secondary lever rivals it): RUN the unconstrained "
          "learned-m_s vs reduce-kappa design comparison." if go else
          "  NO-GO -> kappa still dominant: escalate aspect ratio (bigger r0_new) and re-probe, "
          "or report 'kappa-dominance is general' and STOP.")
    print(f"Saved {jpath} + {ppath}")


if __name__ == "__main__":
    main()
