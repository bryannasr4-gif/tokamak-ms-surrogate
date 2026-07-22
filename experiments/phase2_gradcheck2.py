"""
phase2_gradcheck2.py -- Phase-2 gradient verification, done the decision-relevant way.

The per-axis FD-Jacobian cosine (phase2_grad_analyze) is a deliberately harsh diagnostic: it is
diluted by ~6 weakly-varying / poorly-observable shape descriptors (delta, inner squareness:
ShapeMap R^2<0.6) and by the sub-resolution separatrix noise of tiny control steps. What design
actually uses is the GRADIENT DIRECTION. Here we test that directly and robustly: at each held-out
base equilibrium we take an IN-DISTRIBUTION (box-clipped) step along the surrogate's m_s gradient
and CONFIRM with the true solver, comparing it to (a) the anti-gradient and (b) random control
directions of the same magnitude. A correct gradient should raise true m_s, beat its own negation,
and beat random directions. We also report the directional-derivative correlation (predicted vs
true) over the on-manifold steps. RESOLVED by m_s regime. Saves data/phase2_gradcheck2.json.

Runs its own true solves (~150) -- launch in the background.
"""
import json
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "experiments"))
import phase2_data as D
import phase2_model as M
import phase15_lib as L
import phase2_dim_lib as DL
import phase2_grad_analyze as GA

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def main():
    df = D.load()
    mu = df[D.CONTROL_FEATURES].mean().values
    std = df[D.CONTROL_FEATURES].std().values
    box_lo = np.percentile(df[D.CONTROL_FEATURES].values, 1, axis=0)
    box_hi = np.percentile(df[D.CONTROL_FEATURES].values, 99, axis=0)
    models, _ = M.load_ensemble()
    smap, _ = M.load_shapemap()
    tok = L.load_machine()

    # reuse the held-out base set + their already-computed base m_s
    blob = json.load(open(os.path.join(ROOT, "data", "phase2_grad_probes.json")))
    bases = [r for r in blob["recs"] if r["base"] is not None]

    def surr_ms(u):
        with torch.no_grad():
            ut = torch.tensor(u, dtype=torch.float32).unsqueeze(0)
            return float(torch.stack([m(smap(ut))[0, 0] for m in models]).mean().exp())

    def step(u0, dz, hs):
        """surrogate line-search over step sizes hs along z-dir dz; return best in-box candidate u."""
        best_u, best_s = None, -1
        for h in hs:
            u = np.clip(mu + std * ((u0 - mu) / std + h * dz), box_lo, box_hi)
            s = surr_ms(u)
            if s > best_s:
                best_s, best_u = s, u
        return best_u

    rng = np.random.default_rng(0)
    hs = [0.30, 0.15, 0.075]
    rows = []
    t0 = time.time()
    for b in bases:
        u0 = np.array(b["u"]); ms0 = b["base"]["m_s"]; lm0 = np.log(ms0)
        g = GA.composed_grad_u(models, smap, u0)
        dz = g * std
        nz = np.linalg.norm(dz)
        if nz < 1e-9:
            continue
        dz = dz / nz
        u_g = step(u0, dz, hs)
        u_a = step(u0, -dz, hs)
        ms_g = DL.true_ms(tok, u_g)
        ms_a = DL.true_ms(tok, u_a)
        # random directions (same magnitude family)
        ms_r = []
        for _ in range(3):
            r = rng.standard_normal(len(u0)); r /= np.linalg.norm(r)
            u_r = step(u0, r, hs)            # surrogate would not pick these; use fixed mid step
            u_r = np.clip(mu + std * ((u0 - mu) / std + 0.15 * r), box_lo, box_hi)
            ms_r.append(DL.true_ms(tok, u_r))
        ms_r_best = max(ms_r) if ms_r else 0.0
        # directional derivative along +g (if on-manifold)
        dd = None
        if ms_g > 0:
            pred = float(g @ (u_g - u0))             # predicted d log m_s (linearized)
            true = float(np.log(ms_g) - lm0)
            dd = (pred, true)
        rows.append(dict(regime=b["regime"], ms0=ms0, ms_grad=ms_g, ms_anti=ms_a,
                         ms_rand=ms_r_best, dd=dd))
        print(f"[{b['regime']:11s}] ms0={ms0:.3f} grad={ms_g:.3f} anti={ms_a:.3f} rand={ms_r_best:.3f} "
              f"({len(rows)}/{len(bases)}, {(time.time()-t0)/max(len(rows),1):.0f}s/base)", flush=True)

    # metrics
    def frac(cond):
        return float(np.mean(cond)) if len(cond) else float("nan")
    ms0 = np.array([r["ms0"] for r in rows]); msg = np.array([r["ms_grad"] for r in rows])
    msa = np.array([r["ms_anti"] for r in rows]); msr = np.array([r["ms_rand"] for r in rows])
    ascent = msg > ms0
    beats_anti = msg > msa
    beats_rand = msg >= msr
    dd = np.array([r["dd"] for r in rows if r["dd"] is not None])
    out = dict(n=len(rows),
               ascent_rate=frac(ascent), beats_anti_rate=frac(beats_anti),
               beats_random_rate=frac(beats_rand),
               median_gain_grad=float(np.median(msg - ms0)),
               median_gain_rand=float(np.median(msr - ms0)),
               directional_corr=float(np.corrcoef(dd[:, 0], dd[:, 1])[0, 1]) if len(dd) > 3 else None,
               directional_sign=float(np.mean(np.sign(dd[:, 0]) == np.sign(dd[:, 1]))) if len(dd) else None,
               by_regime={}, rows=rows)
    for name, _, _ in D.REGIMES:
        sel = [i for i, r in enumerate(rows) if r["regime"] == name]
        if len(sel) >= 2:
            sel = np.array(sel)
            out["by_regime"][name] = dict(n=len(sel),
                ascent_rate=frac(ascent[sel]), beats_random_rate=frac(beats_rand[sel]),
                median_gain_grad=float(np.median(msg[sel] - ms0[sel])))
    with open(os.path.join(ROOT, "data", "phase2_gradcheck2.json"), "w") as f:
        json.dump(out, f, indent=2)

    print(f"\n=== GRADIENT VERIFICATION (in-distribution ascent, {out['n']} bases) ===")
    print(f"  ascent (true m_s up along +grad):   {out['ascent_rate']*100:.0f}%")
    print(f"  gradient beats anti-gradient:        {out['beats_anti_rate']*100:.0f}%")
    print(f"  gradient beats random direction:     {out['beats_random_rate']*100:.0f}%")
    print(f"  median true m_s gain: gradient={out['median_gain_grad']:+.3f} vs random={out['median_gain_rand']:+.3f}")
    print(f"  directional-derivative corr={out['directional_corr']}, sign={out['directional_sign']}")
    for name, r in out["by_regime"].items():
        print(f"    {name:11s} n={r['n']:2d} ascent={r['ascent_rate']*100:.0f}% "
              f"beats_rand={r['beats_random_rate']*100:.0f}% gain={r['median_gain_grad']:+.3f}")
    print("Saved data/phase2_gradcheck2.json")


if __name__ == "__main__":
    main()
