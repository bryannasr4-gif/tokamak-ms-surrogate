"""
device2_robust_worker.py -- TRUE-SOLVER confirmation of the robust-design capability (Phase 5 #2).

For each (start, design) where design in {nominal, robust, reduce_kappa}, draw M operational-uncertainty
perturbations of the design's controls (coil 3% / paxis 5% / Ip 3% / fvac 0.02 / alpha 0.05; common
random numbers per start so designs are compared on the SAME shocks) and solve each at 80 modes. Record
the true m_s ensemble (+ the unperturbed center) so the analyzer can compute true worst-case / CVaR /
P(m_s>0.15) per design. Resume-safe (per-(start,design) file, skip-if-exists). Thread-pinned (OMP=1).

  python experiments/device2_robust_worker.py --chunk 0 --nchunks 11 --M 16
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
sys.path.insert(0, os.path.join(ROOT, "experiments"))
import phase15_lib as L
import phase2_dim_lib as DL
import device2_killgate as KG
import device2_robust as RB


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk", type=int, default=0)
    ap.add_argument("--nchunks", type=int, default=11)
    ap.add_argument("--M", type=int, default=16, help="perturbations per design")
    ap.add_argument("--seed", type=int, default=20260628)
    ap.add_argument("--rmax", type=float, default=2.8)
    args = ap.parse_args()

    setup = json.load(open(os.path.join(ROOT, "data", "device2_design_setup.json")))
    ctrl = setup["control_features"]
    mu = np.array(setup["mu"]); std = np.array(setup["std"]); V = np.array(setup["V"])
    lo = np.array(setup["box_lo"]); hi = np.array(setup["box_hi"]); d = setup["d"]
    designs = {r["start_id"]: r for r in json.load(open(os.path.join(ROOT, "data", "device2_robust_designs.json")))["designs"]}

    # reduce_kappa control per start from the Phase-5 RETRAINED results (best_u is a PC-score x)
    redk = {}
    for f in glob.glob(os.path.join(ROOT, "data", "device2_design_results", "retrained_job*.json")):
        r = json.load(open(f))
        if r["method"] == "reduce_kappa":
            redk[r["start_id"]] = r["best_u"]

    import pickle
    with open(os.path.join(ROOT, "data", "device2_anchors.pkl"), "rb") as f:
        L.GRID = pickle.load(f)["meta"]["grid"]    # Device-C grid: single source of truth (anchors meta)
    tok = KG.load_device_c()
    outdir = os.path.join(ROOT, "data", "device2_robust_results")
    os.makedirs(outdir, exist_ok=True)

    # job list = (start_id, design_name); stripe by chunk
    jobs = []
    for sid in sorted(designs):
        for dn in ("nominal", "robust", "reduce_kappa"):
            jobs.append((sid, dn))
    mine = [j for k, j in enumerate(jobs) if k % args.nchunks == args.chunk]

    def solve(u):
        c = DL.ctrl_from_u(np.asarray(u))
        try:
            rec = L.forward_label(tok, c["active_currents"], c["paxis"], c["Ip"], c["fvac"],
                                  c["alpha_m"], c["alpha_n"], fix_n_modes=80)
            ms = rec.get("m_s", float("nan"))
            return float(ms) if (np.isfinite(ms) and ms > 0) else 0.0
        except Exception:
            return 0.0

    for sid, dn in mine:
        out = os.path.join(outdir, f"start{sid}_{dn}.json")
        if os.path.exists(out):
            continue
        rec = designs[sid]
        ds = DL.DesignSpace(mu, std, V, lo, hi, np.array(rec["u0"]), d)
        if dn == "nominal":
            u = np.array(rec["u_nominal"])
        elif dn == "robust":
            u = np.array(rec["u_robust"])
        else:
            if sid not in redk:
                continue
            u = ds.u_of_x(np.array(redk[sid]))
        sig = RB.control_sigma(u, ctrl)
        # common random numbers per start (same standard-normal draws scaled by each design's sigma)
        rng = np.random.default_rng([args.seed, sid])
        Z = rng.standard_normal((args.M, len(ctrl)))
        center = solve(u)
        pert = [solve(u + Z[k] * sig) for k in range(args.M)]
        res = dict(start_id=sid, design=dn, cohort=rec["cohort"], band=rec["band"],
                   ms40_start=rec["ms40_start"], center_ms=center, pert_ms=pert,
                   sigma_rel="coil3%/paxis5%/Ip3%/fvac.02/alpha.05", M=args.M)
        tmp = out + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(res, fh)
        os.replace(tmp, out)
        good = [m for m in pert if m > 0]
        wc = min(good) if good else 0.0
        print(f"[c{args.chunk}] start{sid} {dn:12s} center {center:.3f} "
              f"worst {wc:.3f} mean {np.mean(good) if good else 0:.3f} "
              f"({len(good)}/{args.M} converged)", flush=True)
    print(f"[c{args.chunk}] done {len(mine)} assigned (start,design) jobs", flush=True)


if __name__ == "__main__":
    main()
