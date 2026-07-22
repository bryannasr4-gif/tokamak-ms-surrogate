"""
device2_design_deepdive.py -- supporting analyses for the Phase-5 ledger, backing the Phase-D review
findings with on-disk numbers (NO new solves). Computes:
  (1) per framing x method x regime: mean gain, accept count, invalid-step count, accept-rate
      (effective-budget / validity diagnostics -> review findings on optimizer confound + budget).
  (2) reduce_kappa vs cma paired comparison (-> "the differentiable-loop MECHANISM, not the learned
      model, beats gradient-free": the non-learned kappa gradient also beats CMA).
  (3) zero-shot vs retrained surrogate GRADIENT-DIRECTION cosine at the 40 starts (autodiff, no solves)
      -> the zero-shot loss is gradient-direction distortion, recovered by retraining; NOT the (loop-
      invariant) 4x miscalibration.

  python experiments/device2_design_deepdive.py
"""
import glob
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "experiments"))
import phase2_data as D


def regime_of(ms):
    return D.regime_of(ms)


def load(framing):
    recs = []
    for f in glob.glob(os.path.join(ROOT, "data", "device2_design_results", f"{framing}_job*.json")):
        try:
            recs.append(json.load(open(f)))
        except Exception:
            pass
    return recs


def wilcoxon_compare(starts, a, b):
    from scipy.stats import wilcoxon
    pairs = [s for s in starts if a in s["methods"] and b in s["methods"]]
    if not pairs:
        return None
    da = np.array([s["methods"][a] for s in pairs]); db = np.array([s["methods"][b] for s in pairs])
    diff = da - db; nz = diff[diff != 0]
    wins = int(np.sum(diff > 0))
    p = float(wilcoxon(nz, alternative="two-sided").pvalue) if len(nz) else float("nan")
    return dict(n=len(pairs), wins=wins, win_rate=wins / len(pairs), wilcoxon_p=p,
                median_a=float(np.median(da)), median_b=float(np.median(db)))


def main():
    out = {}

    # ---- (1) + (2) per-framing diagnostics ----
    for framing in ["zeroshot", "retrained"]:
        recs = load(framing)
        if not recs:
            continue
        budget = json.load(open(os.path.join(ROOT, "data", "device2_design_setup.json")))["budget"]
        # per method x regime: gain / accepts / invalid
        diag = {}
        by_start = {}
        for r in recs:
            reg = regime_of(r["ms_start"])
            rej = r.get("reject", {})
            invalid = rej.get("invalid", 0)
            accepts = r["n_solves"] - 1 - sum(rej.values())
            d = diag.setdefault((r["method"], reg), dict(gain=[], accepts=[], invalid=[], n=0))
            d["gain"].append(r["gain"]); d["accepts"].append(accepts); d["invalid"].append(invalid); d["n"] += 1
            s = by_start.setdefault(r["start_id"], dict(ms_start=r["ms_start"], methods={}))
            s["methods"][r["method"]] = r["best_ms"]
        diag_out = {}
        for (m, reg), d in sorted(diag.items()):
            diag_out[f"{m}|{reg}"] = dict(n=d["n"], mean_gain=float(np.mean(d["gain"])),
                                          mean_accepts=float(np.mean(d["accepts"])),
                                          mean_invalid=float(np.mean(d["invalid"])),
                                          accept_rate=float(np.mean(d["accepts"]) / budget))
        # reduce_kappa vs cma (mechanism, not model)
        for s in by_start.values():
            s["regime"] = regime_of(s["ms_start"])
        allstarts = list(by_start.values())
        rk_vs_cma = {reg: wilcoxon_compare([s for s in allstarts if reg == "POOLED" or s["regime"] == reg],
                                           "reduce_kappa", "cma")
                     for reg in ["marginal", "mid", "POOLED"]}
        sur_vs_cma = {reg: wilcoxon_compare([s for s in allstarts if reg == "POOLED" or s["regime"] == reg],
                                            "surrogate", "cma")
                      for reg in ["marginal", "mid", "POOLED"]}
        out[framing] = dict(per_method_regime=diag_out, reduce_kappa_vs_cma=rk_vs_cma,
                            surrogate_vs_cma=sur_vs_cma)

    # ---- (3) gradient-direction cosine: zero-shot vs retrained, at the 40 starts ----
    import torch
    import phase2_model as M, phase2_dim_lib as DL, phase25_kappa_lib as KL
    setup = json.load(open(os.path.join(ROOT, "data", "device2_design_setup.json")))
    mu = np.array(setup["mu"]); std = np.array(setup["std"]); V = np.array(setup["V"])
    lo = np.array(setup["box_lo"]); hi = np.array(setup["box_hi"]); d = setup["d"]
    m_zs, _ = M.load_ensemble("surrogate")
    m_rt, _ = M.load_ensemble("surrogate_C")
    smap, _ = M.load_shapemap("shapemap_C")
    KAPPA_I = D.SHAPE_FEATURES.index("kappa")
    cos_zs_rt, cos_rt_kappa, by_reg = [], [], {}
    rt_ms = {r["start_id"]: r["ms_start"] for r in load("retrained")}
    for s in setup["starts"]:
        ds = DL.DesignSpace(mu, std, V, lo, hi, np.array(s["u0"]), d)
        g_zs = KL._grad_feature(None, smap, ds, ds.x0, None, is_ms=True, models=m_zs)
        g_rt = KL._grad_feature(None, smap, ds, ds.x0, None, is_ms=True, models=m_rt)
        g_k = -KL._grad_feature(None, smap, ds, ds.x0, KAPPA_I)   # reduce_kappa direction
        def cos(a, b):
            return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
        c1 = cos(g_zs, g_rt); c2 = cos(g_rt, g_k)
        cos_zs_rt.append(c1); cos_rt_kappa.append(c2)
        reg = regime_of(rt_ms.get(s["id"], 1.0))
        by_reg.setdefault(reg, []).append((c1, c2))
    out["gradient_cosine"] = dict(
        zeroshot_vs_retrained_mean=float(np.mean(cos_zs_rt)),
        retrained_vs_reduce_kappa_mean=float(np.mean(cos_rt_kappa)),
        by_regime={k: dict(n=len(v), cos_zs_rt=float(np.mean([x[0] for x in v])),
                           cos_rt_kappa=float(np.mean([x[1] for x in v]))) for k, v in by_reg.items()})

    json.dump(out, open(os.path.join(ROOT, "data", "device2_design_deepdive.json"), "w"), indent=2)

    print("=== per-method gain / accepts / invalid by regime ===")
    for framing in ["zeroshot", "retrained"]:
        if framing not in out:
            continue
        print(f"-- {framing} --")
        for k, v in out[framing]["per_method_regime"].items():
            print(f"  {k:24s} n={v['n']:2d} gain {v['mean_gain']:+.3f} accepts {v['mean_accepts']:.1f} "
                  f"invalid {v['mean_invalid']:.1f} (accept_rate {v['accept_rate']:.0%})")
    print("\n=== reduce_kappa vs cma (mechanism, not model) ===")
    for framing in ["zeroshot", "retrained"]:
        if framing in out:
            rk = out[framing]["reduce_kappa_vs_cma"]["POOLED"]
            sc = out[framing]["surrogate_vs_cma"]["POOLED"]
            print(f"  {framing}: reduce_kappa vs cma {rk['wins']}/{rk['n']}={rk['win_rate']:.0%} p={rk['wilcoxon_p']:.4f}"
                  f"  |  surrogate vs cma {sc['wins']}/{sc['n']}={sc['win_rate']:.0%} p={sc['wilcoxon_p']:.4f}")
    print("\n=== gradient-direction cosine ===")
    gc = out["gradient_cosine"]
    print(f"  zeroshot vs retrained surrogate gradient: mean cos = {gc['zeroshot_vs_retrained_mean']:+.3f}")
    print(f"  retrained surrogate vs reduce_kappa direction: mean cos = {gc['retrained_vs_reduce_kappa_mean']:+.3f}")
    for reg, v in gc["by_regime"].items():
        print(f"    [{reg:9s} n={v['n']:2d}] cos(zs,rt)={v['cos_zs_rt']:+.3f}  cos(rt,reduceK)={v['cos_rt_kappa']:+.3f}")
    print("\nSaved data/device2_design_deepdive.json")


if __name__ == "__main__":
    main()
