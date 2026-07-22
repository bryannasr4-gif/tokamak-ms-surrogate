"""
tier1_epistemic_baseline.py -- Stage-0 S0.2 TASK c: persist the in-distribution epistemic-uncertainty
baseline that the "~25x domain-shift flag" divides by (the Stage-0 epistemic-baseline gate).

Definition (G-S0.2-3): the median, over the in-dist evaluation set used by tier1_analyze.py, of the
ensemble's epistemic std (std across ensemble members' predictions), computed in the SAME space and by
the SAME code path tier1_analyze.py applies to the real slices.

Eval set: this REPRODUCES tier1_analyze.py's epi_baseline block EXACTLY (lines 79-87 there):
    tr   = pd.read_parquet(dataset_v1_80q.parquet)          # FULL parquet (NOT a held-out split)
    samp = tr.sample(min(500, len(tr)), random_state=0)[SHAPE_FEATURES]
    baseline = median(ensemble_predict(models, samp)["epi_std"][:, 0])   # column 0 = log m_s
tier1_analyze.py does not use any held-out split for the baseline, so we do not either (the frozen
prompt's "if it used the held-out split" conditional does not apply).

Read-only w.r.t. models/data; writes data/tier1_epistemic_baseline.json and adds
indist_epistemic_median (+provenance) to data/tier1_analysis.json. No solver, no training.

  python experiments/tier1_epistemic_baseline.py
"""
import os, sys, json, datetime
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")


def main():
    from phase2_model import load_ensemble, ensemble_predict
    from phase2_data import SHAPE_FEATURES

    # --- load the ensemble EXACTLY as tier1_analyze.py does
    models, _ = load_ensemble("surrogate")

    # --- SAME eval set + SAME code path as tier1_analyze.py's epi_baseline block
    tr = pd.read_parquet(os.path.join(DATA, "dataset_v1_80q.parquet"))
    samp = tr.sample(min(500, len(tr)), random_state=0)[SHAPE_FEATURES].to_numpy(float)
    epi = ensemble_predict(models, samp)["epi_std"][:, 0]   # column 0 = log m_s (same as real slices)
    baseline = float(np.median(epi))
    n = int(samp.shape[0])

    # --- the real-shape epistemic median (already computed by tier1_analyze.py) for the ratio rule
    ana = json.load(open(os.path.join(DATA, "tier1_analysis.json")))
    real_med = ana.get("consistency", {}).get("median_epi_std", None)
    ratio = (float(real_med) / baseline) if (real_med and baseline) else None

    definition = ("median over the in-dist eval set of the ensemble epistemic std (std across the 8 "
                  "ensemble members' log-m_s mean predictions), column 0 = log m_s; same space + code "
                  "path as tier1_analyze.py applies to the real slices (tier1_resolve_worker surr_epi_std).")
    eval_set = ("500-row random_state=0 sample of the FULL data/dataset_v1_80q.parquet (n_rows=%d), "
                "SHAPE_FEATURES order -- reproduces tier1_analyze.py lines 79-87 verbatim; NOT a "
                "held-out split (tier1_analyze.py samples the full parquet)." % len(tr))

    out = dict(
        value=baseline,
        n=n,
        split=eval_set,
        definition=definition,
        script="experiments/tier1_epistemic_baseline.py",
        date=datetime.date.today().isoformat(),
        real_epi_median=(float(real_med) if real_med is not None else None),
        ratio_real_over_baseline=ratio,
        ratio_2sigfig=(float(f"{ratio:.2g}") if ratio is not None else None),
        ledger_baseline_reference=0.028,
        gate_band_G_S0_2_3=[0.0224, 0.0336],
        in_band=bool(0.0224 <= baseline <= 0.0336),
        # added 2026-07-15, E-FA5-2 (mirror the FA-8 JSON flag fields so a rerun cannot regress them)
        exploratory=True,
        confirmatory=False,
        flag_note="Flags added 2026-07-15 per FA5-4 / E-FA5-2 ruling; absent at birth (d4d9ef4) because the frozen S0.2 TASK-c schema {value,n,split,definition,script,date} predated the 00 §9 cross-check. Scope: value/n/gate_band/in_band form the G-S0.2-3 feasibility-gate record (§9-gate-exempt); the exploratory flag binds real_epi_median / ratio_real_over_baseline / ratio_2sigfig — the ~25x domain-flag multiplier is descriptive/supporting (Tier-1 profile-match gate 0/55 VOIDED; abstention-flag paragraph contingent on E2), never a confirmatory statistic.",
    )
    with open(os.path.join(DATA, "tier1_epistemic_baseline.json"), "w") as f:
        json.dump(out, f, indent=2)

    # --- add the key to tier1_analysis.json (+ provenance), preserving all existing content
    ana["indist_epistemic_median"] = baseline
    ana["indist_epistemic_median_provenance"] = dict(
        script="experiments/tier1_epistemic_baseline.py",
        artifact="data/tier1_epistemic_baseline.json",
        definition=definition, eval_set=eval_set, n=n, date=out["date"],
        ratio_real_over_baseline=ratio,
    )
    with open(os.path.join(DATA, "tier1_analysis.json"), "w") as f:
        json.dump(ana, f, indent=2)

    print(f"indist_epistemic_median = {baseline:.5f} (n={n})")
    print(f"gate band [0.0224, 0.0336] -> in_band={out['in_band']}")
    print(f"real_epi_median={real_med} ratio={ratio} (2 sig figs {out['ratio_2sigfig']})")


if __name__ == "__main__":
    main()
