"""
device2_robust.py -- ROBUST / worst-case stability-margin design on Device-C (Phase 5 follow-on #2).

THE HYPOTHESIS (NOT CONFIRMED -- this surrogate-only version FAILED; see RESULT): that the surrogate's
cheap differentiable prediction could design for the WORST-CASE m_s under operational uncertainty
(coil/profile drift) -- an objective needing many m_s evals/candidate (~free with the ~8,000x surrogate,
costly for a true-solver heuristic/CMA).

RESULT (2026-06-28; recorded honestly in RESULTS.md Part B): the SURROGATE-ONLY optimization below drives
BOTH the nominal and robust designs OFF the diverted manifold -- the true 80-mode center m_s collapses to
0 for ~75-80% of starts (nominal center==0 in 32/40 starts, robust 29/40), so the true worst-case is
floor-saturated at 0 for all designs and the robust-vs-nominal comparison is UNTESTABLE (Wilcoxon p=NaN,
all paired diffs 0). This does NOT establish any capability. It is the project's core lesson restated:
the surrogate is trustworthy ONLY with the solver IN THE LOOP. A valid robust demo must REDO this as a
CONFIRM-IN-LOOP optimizer (propose by the surrogate worst-case score, confirm validity+m_s by ONE true
solve per step, REJECT off-manifold steps). Known fixes for the redo (do not change this deprecated
artifact): (i) confirm-in-loop validity gating; (ii) recompute the perturbation sigma at the CURRENT
control u (this version fixes it at the start u0 -- see control_sigma below -- inconsistent with the
worker's per-design sigma); (iii) report worst_converged + conv_frac, not the floor-saturated worst-case.
Kept as an instructive FAILED attempt + the starting point for that redo on the other computer.

This module (surrogate-ONLY, NO true solves) computes, per start, two designs the SAME way (fair):
  nominal : maximize surrogate point estimate of log m_s        (what you'd do without robustness)
  robust  : maximize a surrogate worst-case score over an uncertainty ensemble of control perturbations
            score(x) = mean_k[log m_s(u(x)+d_k)] - LAMBDA * std_k[...]   (mean-minus-lambda-sigma)
Common random numbers (fixed perturbation ensemble per start) make the score smooth + differentiable;
gradient ascent in the d=12 PCA design space. Saves best_u (PC-score) per start/design for the true-
solver perturbation confirmation step (device2_robust_run.py).

  python experiments/device2_robust.py --lam 1.0 --K 64 --steps 60
"""
import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "experiments"))
import phase2_data as D


# per-control 1-sigma operational uncertainty (applied in raw control units):
#   coil currents: relative 3% of |current|; profile params: physically modest absolute/relative drifts.
def control_sigma(u, ctrl):
    sig = np.zeros(len(ctrl))
    for i, name in enumerate(ctrl):
        if name.startswith("I_"):
            sig[i] = 0.03 * abs(u[i])                 # 3% coil-current realization error
        elif name == "paxis":
            sig[i] = 0.05 * abs(u[i])                 # 5% pressure
        elif name == "Ip_target":
            sig[i] = 0.03 * abs(u[i])                 # 3% plasma current
        elif name == "fvac":
            sig[i] = 0.02
        elif name in ("alpha_m", "alpha_n"):
            sig[i] = 0.05                             # profile-peakedness drift
    return sig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lam", type=float, default=1.0, help="risk aversion (mean - lam*std)")
    ap.add_argument("--K", type=int, default=64, help="uncertainty-ensemble size (surrogate evals/candidate)")
    ap.add_argument("--steps", type=int, default=60, help="surrogate-only ascent steps")
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--surrogate", default="surrogate_C")
    ap.add_argument("--shapemap", default="shapemap_C")
    ap.add_argument("--seed", type=int, default=20260628)
    args = ap.parse_args()

    import torch
    import phase2_model as M, phase2_dim_lib as DL

    setup = json.load(open(os.path.join(ROOT, "data", "device2_design_setup.json")))
    ctrl = setup["control_features"]
    mu = np.array(setup["mu"]); std = np.array(setup["std"]); V = np.array(setup["V"])
    lo = np.array(setup["box_lo"]); hi = np.array(setup["box_hi"]); d = setup["d"]
    models, _ = M.load_ensemble(args.surrogate)
    smap, _ = M.load_shapemap(args.shapemap)

    def surr_logms_batch(U):                          # U: (N,16) torch -> (N,) mean-ensemble log m_s
        shp = smap(U)
        return torch.stack([m(shp)[:, 0] for m in models]).mean(0)

    rng = np.random.default_rng(args.seed)
    out = []
    for s in setup["starts"]:
        u0 = np.array(s["u0"])
        ds = DL.DesignSpace(mu, std, V, lo, hi, u0, d)
        sig = control_sigma(u0, ctrl)
        # fixed perturbation ensemble in raw control units (common random numbers)
        eps = torch.tensor(rng.standard_normal((args.K, len(ctrl))) * sig, dtype=torch.float32)
        box_lo = torch.tensor(ds.box_lo, dtype=torch.float32)
        box_hi = torch.tensor(ds.box_hi, dtype=torch.float32)

        def optimize(robust):
            x = torch.tensor(ds.x0, dtype=torch.float32, requires_grad=True)
            opt = torch.optim.Adam([x], lr=args.lr)
            for _ in range(args.steps):
                opt.zero_grad()
                u = ds.u_of_x_t(x)                     # (16,) differentiable control
                if robust:
                    Up = (u.unsqueeze(0) + eps)        # (K,16) perturbed controls
                    lm = surr_logms_batch(Up)
                    score = lm.mean() - args.lam * lm.std()
                else:
                    score = surr_logms_batch(u.unsqueeze(0))[0]
                (-score).backward()
                opt.step()
                with torch.no_grad():
                    x.clamp_(box_lo, box_hi)
            return x.detach().numpy()

        x_nom = optimize(robust=False)
        x_rob = optimize(robust=True)
        rec = dict(start_id=s["id"], cohort=s["cohort"], band=s["band"], ms40_start=s["ms40_start"],
                   u0=s["u0"],
                   x_nominal=[float(v) for v in x_nom], u_nominal=[float(v) for v in ds.u_of_x(x_nom)],
                   x_robust=[float(v) for v in x_rob], u_robust=[float(v) for v in ds.u_of_x(x_rob)],
                   sigma=[float(v) for v in sig])
        out.append(rec)

    meta = dict(lam=args.lam, K=args.K, steps=args.steps, surrogate=args.surrogate,
                shapemap=args.shapemap, uncertainty="coil 3% / paxis 5% / Ip 3% / fvac 0.02 / alpha 0.05",
                n_starts=len(out))
    json.dump(dict(meta=meta, designs=out),
              open(os.path.join(ROOT, "data", "device2_robust_designs.json"), "w"), indent=2)
    # quick surrogate-space sanity: how different is robust from nominal?
    dist = [float(np.linalg.norm(np.array(r["x_robust"]) - np.array(r["x_nominal"]))) for r in out]
    print(f"computed nominal+robust designs for {len(out)} starts (surrogate-only, no solves).")
    print(f"  median |x_robust - x_nominal| in PC space = {np.median(dist):.3f} "
          f"(MAX_STEP={DL.MAX_STEP}; >0 means robust design genuinely differs from nominal)")
    print("Saved data/device2_robust_designs.json")


if __name__ == "__main__":
    main()
