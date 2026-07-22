"""
device2_design_setup.py -- set up the Device-C UNCONSTRAINED design comparison (Phase 5 C4).

Builds the shared Device-C PCA design basis (mu/std/V/box over the Device-C control distribution)
ONCE, selects stratified marginal+mid START control vectors from the kill-gate probe (by 40-mode
m_s band; the TRUE 80-mode start regime is read back from each run), and emits the job list
(start x method). Methods: surrogate / reduce_kappa / cma. Main + a DISJOINT replication cohort.

  python experiments/device2_design_setup.py --d 12 --budget 18 \
        --main_marg 12 --main_mid 12 --rep_marg 8 --rep_mid 8
"""
import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "experiments"))
import phase2_data as D


def pick_band(df, lo, hi, k, used, seed):
    """k evenly-spaced (by m_s) distinct row indices with 40-mode m_s in [lo,hi), excluding used."""
    sub = df[(df["m_s"] >= lo) & (df["m_s"] < hi)].sort_values("m_s")
    sub = sub[~sub.index.isin(used)]
    if len(sub) == 0:
        return []
    idx = np.linspace(0, len(sub) - 1, min(k, len(sub))).round().astype(int)
    return list(sub.index[idx])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, default=12)
    ap.add_argument("--budget", type=int, default=18)
    ap.add_argument("--main_marg", type=int, default=12)
    ap.add_argument("--main_mid", type=int, default=12)
    ap.add_argument("--rep_marg", type=int, default=8)
    ap.add_argument("--rep_mid", type=int, default=8)
    # 40-mode m_s bands chosen so the 80-mode start (≈+14%) lands marginal (<0.4) / mid (0.4-1.0)
    ap.add_argument("--marg_band", type=float, nargs=2, default=[0.08, 0.33])
    ap.add_argument("--mid_band", type=float, nargs=2, default=[0.36, 0.82])
    ap.add_argument("--basis_parquets", nargs="+",
                    default=["data/device2_probe.parquet", "data/device2_shapegen_all.parquet"])
    ap.add_argument("--seed", type=int, default=20260626)
    args = ap.parse_args()

    import pandas as pd
    ctrl = D.CONTROL_FEATURES

    # ---- shared PCA basis over the Device-C control distribution ----
    # FAIL LOUDLY if a requested basis parquet is missing (else the basis silently collapses to the
    # probe alone -> different PCs + a narrower box). Run device2_assemble.py first.
    missing = [p for p in args.basis_parquets if not os.path.exists(os.path.join(ROOT, p))]
    assert not missing, (f"missing basis parquets {missing}; run "
                         f"`device2_assemble.py --prefix device2_shapegen --out data/device2_shapegen_all.parquet` first")
    bp = [os.path.join(ROOT, p) for p in args.basis_parquets]
    Udf = pd.concat([pd.read_parquet(p) for p in bp], ignore_index=True).dropna(subset=ctrl)
    U = Udf[ctrl].values.astype(np.float64)
    mu = U.mean(0); std = U.std(0) + 1e-8
    Z = (U - mu) / std
    _, _, Vt = np.linalg.svd(Z - Z.mean(0), full_matrices=True)
    V = Vt.T
    scores = Z @ V
    lo = np.percentile(scores, 2, axis=0); hi = np.percentile(scores, 98, axis=0)
    print(f"PCA basis from {len(U)} Device-C control samples over {len(bp)} parquet(s): "
          f"{[os.path.basename(p) for p in bp]}; d={args.d}")

    # ---- pick stratified starts from the PROBE (which has m_s labels) ----
    probe = pd.read_parquet(os.path.join(ROOT, "data", "device2_probe.parquet")).dropna(subset=ctrl)
    used = set()
    cohorts = {}
    for cohort, nm, nd in [("main", args.main_marg, args.main_mid),
                           ("replication", args.rep_marg, args.rep_mid)]:
        marg = pick_band(probe, args.marg_band[0], args.marg_band[1], nm, used, args.seed)
        used |= set(marg)
        mid = pick_band(probe, args.mid_band[0], args.mid_band[1], nd, used, args.seed)
        used |= set(mid)
        cohorts[cohort] = [("marg_band", i) for i in marg] + [("mid_band", i) for i in mid]
        print(f"  {cohort}: {len(marg)} marginal-band + {len(mid)} mid-band starts")

    starts = []
    sid = 0
    for cohort, lst in cohorts.items():
        for band, ridx in lst:
            row = probe.loc[ridx]
            u0 = [float(row[c]) for c in ctrl]
            starts.append(dict(id=sid, cohort=cohort, band=band, probe_idx=int(ridx),
                               ms40_start=float(row["m_s"]), kappa40_start=float(row["kappa"]), u0=u0))
            sid += 1

    # WIDEN the box so every selected start's x0 is feasible (starts are individual probe rows and can
    # sit in the >p98/<p2 PC tails; an unclipped x0 would make step-1 a non-gradient boundary snap and
    # hand CMA an out-of-bounds initial mean). Symmetric across all methods -> no comparison bias.
    if starts:
        U0 = np.array([s["u0"] for s in starts], dtype=np.float64)
        scores0 = ((U0 - mu) / std) @ V
        lo = np.minimum(lo, scores0.min(0))
        hi = np.maximum(hi, scores0.max(0))

    METHODS = ["surrogate", "reduce_kappa", "cma"]
    jobs = []
    jid = 0
    for s in starts:
        for m in METHODS:
            jobs.append(dict(jid=jid, start_id=s["id"], cohort=s["cohort"], band=s["band"],
                             method=m, seed=args.seed + s["id"] * 7 + METHODS.index(m)))
            jid += 1

    setup = dict(d=args.d, budget=args.budget, methods=METHODS, n_starts=len(starts), n_jobs=len(jobs),
                 mu=mu.tolist(), std=std.tolist(), V=V.tolist(), box_lo=lo.tolist(), box_hi=hi.tolist(),
                 control_features=ctrl, starts=starts, basis_n=len(U), seed=args.seed)
    json.dump(setup, open(os.path.join(ROOT, "data", "device2_design_setup.json"), "w"))
    json.dump(jobs, open(os.path.join(ROOT, "data", "device2_design_jobs.json"), "w"), indent=2)
    print(f"\n{len(starts)} starts, {len(jobs)} jobs ({len(METHODS)} methods), budget {args.budget}.")
    print("Saved data/device2_design_setup.json + data/device2_design_jobs.json")


if __name__ == "__main__":
    main()
