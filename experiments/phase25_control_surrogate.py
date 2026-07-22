"""
phase25_control_surrogate.py -- does a DIRECT m_s(controls) surrogate give a more faithful
control-space gradient than the composed shape-surrogate(shapemap(u))?

The composed control gradient (cos_B median 0.64 vs the true solver) is limited by the lossy
control->shape map (effective dim 5.5/16). A surrogate trained END-TO-END on controls might learn
the control->m_s sensitivities directly. We train m_s(controls) on dataset_v1 (40-mode labels, to
match the existing grad probes) and compare its autodiff control-gradient to the true-solver
finite-difference gradient at the same held-out bases (data/phase2_grad_probes.json), in
standardized control units. No new solves. Saves data/phase25_control_grad.json.
"""
import json
import os
import sys

import numpy as np
import torch

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, "experiments")
import phase2_data as D
import phase2_model as M

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

torch.manual_seed(0); np.random.seed(0)


def train_control_ensemble(Xtr, Ttr, din, n=8, warmup=800, epochs=2600, wd=1e-4):
    import phase2_train as T
    return T.train_surrogate(Xtr, Ttr, din=din, n_models=n, warmup=warmup, epochs=epochs, wd=wd)


def main():
    df = D.load()
    ctrl = D.CONTROL_FEATURES
    tr = df[df.split == "train"]; va = df[df.split == "val"]; te = df[df.split == "test_extrap"]
    Xtr = tr[ctrl].values.astype(np.float64)
    Ttr = np.column_stack([np.log(tr["m_s"].values), np.log(tr["gamma"].values)])
    print("training direct m_s(controls) ensemble...")
    models, meta = train_control_ensemble(Xtr, Ttr, din=len(ctrl))
    M.save_ensemble(models, meta, name="control_surrogate")

    def r2(y, yh):
        return float(1 - np.sum((y - yh) ** 2) / np.sum((y - y.mean()) ** 2))
    for nm, dd in [("val", va), ("test_extrap", te)]:
        pr = M.ensemble_predict(models, dd[ctrl].values.astype(np.float64))["mean"][:, 0]
        print(f"  {nm}: log-R2={r2(np.log(dd['m_s'].values), pr):.3f}")

    # --- gradient faithfulness vs true solver, on existing probes ---
    blob = json.load(open("data/phase2_grad_probes.json"))
    std_u = np.array([df[c].std() for c in ctrl])

    def direct_grad(u0):
        Xt = torch.tensor(np.atleast_2d(u0), dtype=torch.float32, requires_grad=True)
        gs = []
        for m in models:
            out = m(Xt)[:, 0]
            g, = torch.autograd.grad(out.sum(), Xt, retain_graph=True)
            gs.append(g.detach().numpy()[0])
        return np.mean(gs, 0)

    def cos(a, b):
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

    per_base = []
    for rec in blob["recs"]:
        if rec["base"] is None:
            continue
        u0 = np.array(rec["u"]); lm0 = rec["base"]["logms"]
        ks, gt = [], []
        for p in rec["probes"]:
            if p["pt"] is None:
                continue
            ks.append(p["k"]); gt.append((p["pt"]["logms"] - lm0) / p["du"])
        if len(ks) < 4:
            continue
        ks = np.array(ks); sc = std_u[ks]
        gt = np.array(gt) * sc
        gd = direct_grad(u0)[ks] * sc
        per_base.append(dict(regime=rec["regime"], cos=cos(gd, gt),
                             sign=float(np.mean(np.sign(gd) == np.sign(gt)))))
    cosv = np.array([r["cos"] for r in per_base])
    out = dict(n=len(per_base), cos_median=float(np.median(cosv)), cos_mean=float(np.mean(cosv)),
               by_regime={nm: float(np.median([r["cos"] for r in per_base if r["regime"] == nm]))
                          for nm in ["marginal", "mid", "stable", "very_stable"]
                          if sum(r["regime"] == nm for r in per_base) >= 2})
    json.dump(out, open("data/phase25_control_grad.json", "w"), indent=2)
    print(f"\nDIRECT control-gradient vs true solver: per-base cosine median={out['cos_median']:.3f} "
          f"(composed cos_B was 0.64)")
    print("  by regime:", {k: round(v, 3) for k, v in out["by_regime"].items()})
    print("Saved data/phase25_control_grad.json")


if __name__ == "__main__":
    main()
