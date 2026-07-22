"""
phase2_grad_analyze.py -- Phase-2 gradient verification (the headline rigor check).

Consumes data/phase2_grad_probes.json (true-solver finite differences at held-out base
equilibria) and the trained models, and compares the surrogate's autodiff gradient to the true
solver three ways. Everything is in STANDARDIZED control units (each control scaled by its
dataset std) so the cosine/sign/magnitude are not dominated by the raw coil-current scale.

  Test DIR  (isolates the m_s(SHAPE) surrogate): along each REALIZED shape direction
            ds_k = shape(u+du_k)-shape(u), does  grad_shape(log m_s) . ds_k  predict the true
            d(log m_s)?  -> pooled scatter (pred vs true), per-base cosine, sign, slope.
  Test A    (m_s-surrogate, control space via the FD shape Jacobian J=ds/du):
            g_A = J^T grad_shape(log m_s)  vs  g_true = d(log m_s)/du.  cosine/sign/mag per base.
  Test B    (FULL amortized pipeline used in design): g_B = d/du[ surrogate(shapemap(u)) ]
            (autodiff through the learned control->shape map) vs g_true.  cosine/sign/mag.

All metrics RESOLVED by m_s regime. Saves data/phase2_gradcheck.json.
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


def cos(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def composed_grad_u(models, smap, u0):
    """Mean ensemble autodiff gradient d(log m_s)/du through surrogate(shapemap(u)) at u0 (16)."""
    ut = torch.tensor(np.atleast_2d(u0), dtype=torch.float32, requires_grad=True)
    gs = []
    for m in models:
        logms = m(smap(ut))[:, 0]
        g, = torch.autograd.grad(logms.sum(), ut, retain_graph=False)
        gs.append(g.detach().numpy()[0])
    return np.mean(gs, 0)


def main():
    with open("data/phase2_grad_probes.json") as f:
        blob = json.load(f)
    controls = blob["controls"]
    df = D.load()
    std_u = np.array([df[c].std() for c in controls])      # standardization scale (16,)
    feat = D.SHAPE_FEATURES

    models, _ = M.load_ensemble()
    smap, _ = M.load_shapemap()

    # pooled directional samples and per-base vectors
    pooled = {"pred": [], "true": [], "regime": []}
    per_base = []      # dict per base with cosines for DIR/A/B + sign + mag, regime
    dshape_mag = []

    for rec in blob["recs"]:
        if rec["base"] is None:
            continue
        s0 = np.array([rec["base"][f] for f in feat])
        logms0 = rec["base"]["logms"]
        u0 = np.array(rec["u"])
        # surrogate shape-space gradient at the base shape
        g_shape = M.ms_grad_shape(models, s0[None, :], relative=True)[0]   # d log m_s / d shape (20)

        ks, J_cols, g_true, dir_pred, dir_true = [], [], [], [], []
        for p in rec["probes"]:
            if p["pt"] is None:
                continue
            k, du = p["k"], p["du"]
            sk = np.array([p["pt"][f] for f in feat])
            ds = sk - s0
            dlogms = p["pt"]["logms"] - logms0
            ks.append(k)
            J_cols.append(ds / du)                      # dshape/du_k (20,)
            g_true.append(dlogms / du)                  # d log m_s / du_k
            dir_pred.append(float(g_shape @ ds))        # predicted d log m_s along realized ds
            dir_true.append(float(dlogms))              # true d log m_s along realized ds
            dshape_mag.append(float(np.linalg.norm(ds)))
            pooled["pred"].append(float(g_shape @ ds)); pooled["true"].append(float(dlogms))
            pooled["regime"].append(rec["regime"])
        if len(ks) < 4:
            continue
        ks = np.array(ks)
        sc = std_u[ks]                                  # per-probe standardization scale
        g_true = np.array(g_true) * sc                  # standardized true control gradient
        # Test A: g_A[k] = g_shape . J[:,k], standardized
        g_A = np.array([g_shape @ Jc for Jc in J_cols]) * sc
        # Test B: full composed autodiff control gradient, restricted to probed dims, standardized
        gB_full = composed_grad_u(models, smap, u0)
        g_B = gB_full[ks] * sc
        dir_pred = np.array(dir_pred); dir_true = np.array(dir_true)
        per_base.append(dict(
            regime=rec["regime"], split=rec["split"], idx=rec["idx"], nprobe=int(len(ks)),
            cos_dir=cos(dir_pred, dir_true),
            sign_dir=float(np.mean(np.sign(dir_pred) == np.sign(dir_true))),
            cos_A=cos(g_A, g_true), sign_A=float(np.mean(np.sign(g_A) == np.sign(g_true))),
            mag_A=float(np.linalg.norm(g_A) / (np.linalg.norm(g_true) + 1e-12)),
            cos_B=cos(g_B, g_true), sign_B=float(np.mean(np.sign(g_B) == np.sign(g_true))),
            mag_B=float(np.linalg.norm(g_B) / (np.linalg.norm(g_true) + 1e-12)),
        ))

    pred = np.array(pooled["pred"]); true = np.array(pooled["true"])
    preg = np.array(pooled["regime"])
    # pooled directional slope + correlation
    slope = float(np.polyfit(pred, true, 1)[0])
    corr = float(np.corrcoef(pred, true)[0, 1])
    sign_pooled = float(np.mean(np.sign(pred) == np.sign(true)))

    def summ(rows, key):
        v = np.array([r[key] for r in rows])
        return dict(median=float(np.median(v)), mean=float(np.mean(v)), min=float(np.min(v)))

    res = {"n_bases": len(per_base), "n_dir_samples": int(len(pred)),
           "median_realized_dshape_norm": float(np.median(dshape_mag)),
           "pooled_directional": dict(slope=slope, corr=corr, sign_agreement=sign_pooled),
           "overall": {k: summ(per_base, k) for k in
                       ["cos_dir", "sign_dir", "cos_A", "sign_A", "mag_A", "cos_B", "sign_B", "mag_B"]},
           "by_regime": {}, "per_base": per_base}
    for name, _, _ in D.REGIMES:
        rows = [r for r in per_base if r["regime"] == name]
        if len(rows) >= 2:
            res["by_regime"][name] = dict(n=len(rows),
                **{k: summ(rows, k) for k in ["cos_dir", "sign_dir", "cos_A", "cos_B", "mag_B"]})
    # pooled directional by regime
    res["pooled_by_regime"] = {}
    for name, _, _ in D.REGIMES:
        m = preg == name
        if m.sum() >= 8:
            res["pooled_by_regime"][name] = dict(
                n=int(m.sum()), corr=float(np.corrcoef(pred[m], true[m])[0, 1]),
                sign=float(np.mean(np.sign(pred[m]) == np.sign(true[m]))))
    res["pooled_arrays"] = dict(pred=pred.tolist(), true=true.tolist(), regime=preg.tolist())

    with open("data/phase2_gradcheck.json", "w") as f:
        json.dump(res, f, indent=2)

    print(f"=== GRADIENT VERIFICATION ({res['n_bases']} bases, {res['n_dir_samples']} directional samples) ===")
    print(f" median realized |dshape| = {res['median_realized_dshape_norm']:.4g} (resolved, not noise)")
    p = res["pooled_directional"]
    print(f" POOLED directional: corr={p['corr']:.3f} sign={p['sign_agreement']*100:.0f}% slope={p['slope']:.2f}")
    o = res["overall"]
    print(f" Test DIR (m_s-surrogate, per-base): cos median={o['cos_dir']['median']:.3f} "
          f"sign={o['sign_dir']['median']*100:.0f}%")
    print(f" Test A   (FD-Jacobian control grad): cos median={o['cos_A']['median']:.3f} "
          f"sign={o['sign_A']['median']*100:.0f}% mag={o['mag_A']['median']:.2f}")
    print(f" Test B   (amortized shapemap grad):  cos median={o['cos_B']['median']:.3f} "
          f"sign={o['sign_B']['median']*100:.0f}% mag={o['mag_B']['median']:.2f}")
    print(" by regime (cos_dir / cos_A / cos_B):")
    for name, r in res["by_regime"].items():
        print(f"   {name:11s} n={r['n']:2d}  dir={r['cos_dir']['median']:.3f}  "
              f"A={r['cos_A']['median']:.3f}  B={r['cos_B']['median']:.3f}")
    print("Saved data/phase2_gradcheck.json")


if __name__ == "__main__":
    main()
