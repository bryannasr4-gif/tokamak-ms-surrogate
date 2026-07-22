"""
phase2_gradcheck2_80.py -- re-verify the gradient ascent AT THE CONVERGED 80 MODES (Phase-2.5b, C).

The original phase2_gradcheck2.py ran at 40 modes (and compared the 80-mode-default true solves
against a 40-mode STORED base m_s -- inconsistent). Here everything is at 80 modes:
  * surrogate = the A80 (80-mode-retrained) ensemble + shapemap (default load).
  * base m_s ms0 is RE-SOLVED at 80 modes (DL.true_ms, fix_n_modes=80) -- so the ascent comparison
    ms_grad vs ms0 is internally consistent (the 40-mode ms0 was ~14% low and would inflate ascent).
  * regime is RE-BINNED by the 80-mode ms0 (a 40-mode-marginal base can become mid at 80 modes).
We report ascent / beats-anti / beats-random rates RESOLVED BY REGIME with Wilson 95% CIs, plus the
design-regime (marginal+mid) pooled rate, and print a side-by-side vs the 40-mode result.
Saves data/phase2_gradcheck2_80.json. Runs ~180 true 80-mode solves -- background.
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

MODES = 80


def wilson(k, n, z=1.96):
    if n == 0:
        return [float("nan"), float("nan")]
    p = k / n
    den = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [float(center - half), float(center + half)]


def main():
    df = D.load()
    mu = df[D.CONTROL_FEATURES].mean().values
    std = df[D.CONTROL_FEATURES].std().values
    box_lo = np.percentile(df[D.CONTROL_FEATURES].values, 1, axis=0)
    box_hi = np.percentile(df[D.CONTROL_FEATURES].values, 99, axis=0)
    models, _ = M.load_ensemble()        # A80 surrogate (retrained on 80-mode labels)
    smap, _ = M.load_shapemap()
    tok = L.load_machine()

    blob = json.load(open(os.path.join(ROOT, "data", "phase2_grad_probes.json")))
    bases = [r for r in blob["recs"] if r["base"] is not None]

    def surr_ms(u):
        with torch.no_grad():
            ut = torch.tensor(u, dtype=torch.float32).unsqueeze(0)
            return float(torch.stack([m(smap(ut))[0, 0] for m in models]).mean().exp())

    def step(u0, dz, hs):
        best_u, best_s = None, -1
        for h in hs:
            u = np.clip(mu + std * ((u0 - mu) / std + h * dz), box_lo, box_hi)
            s = surr_ms(u)
            if s > best_s:
                best_s, best_u = s, u
        return best_u

    rng = np.random.default_rng(0)
    hs = [0.30, 0.15, 0.075]
    # RESUME-SAFE (power-loss hardening): incrementally persist per-base rows to a partial file and,
    # on relaunch, load it + skip bases already done. base order is fixed (grad_probes.json), so the
    # rng draws for not-yet-done bases differ from a clean run only in seeding offset -- acceptable
    # for a verification statistic (the random-direction baseline is a sanity floor, not the headline).
    partial = os.path.join(ROOT, "data", "phase2_gradcheck2_80_partial.json")
    rows = []
    done_idx = set()
    if os.path.exists(partial):
        try:
            rows = json.load(open(partial)).get("rows", [])
            done_idx = {r.get("base_idx") for r in rows}
            print(f"RESUME: {len(rows)} bases already done, skipping them", flush=True)
        except Exception:
            rows, done_idx = [], set()
    t0 = time.time()
    for b in bases:
        if b.get("idx") in done_idx:
            continue
        u0 = np.array(b["u"])
        ms0 = DL.true_ms(tok, u0, fix_n_modes=MODES)     # 80-mode base (consistency fix)
        if ms0 <= 0:
            print(f"  skip base idx={b.get('idx')} (ms0 solve failed at 80 modes)", flush=True)
            continue
        lm0 = np.log(ms0)
        regime80 = D.regime_of(ms0)
        g = GA.composed_grad_u(models, smap, u0)
        dz = g * std
        nz = np.linalg.norm(dz)
        if nz < 1e-9:
            continue
        dz = dz / nz
        u_g = step(u0, dz, hs)
        u_a = step(u0, -dz, hs)
        ms_g = DL.true_ms(tok, u_g, fix_n_modes=MODES)
        ms_a = DL.true_ms(tok, u_a, fix_n_modes=MODES)
        ms_r = []
        for _ in range(3):
            r = rng.standard_normal(len(u0)); r /= np.linalg.norm(r)
            u_r = np.clip(mu + std * ((u0 - mu) / std + 0.15 * r), box_lo, box_hi)
            ms_r.append(DL.true_ms(tok, u_r, fix_n_modes=MODES))
        ms_r_best = max(ms_r) if ms_r else 0.0
        dd = None
        if ms_g > 0:
            pred = float(g @ (u_g - u0))
            true = float(np.log(ms_g) - lm0)
            dd = (pred, true)
        rows.append(dict(base_idx=b.get("idx"), regime40=b["regime"], regime=regime80, ms0=ms0,
                         ms_grad=ms_g, ms_anti=ms_a, ms_rand=ms_r_best, dd=dd))
        print(f"[{regime80:11s}] ms0={ms0:.3f} grad={ms_g:.3f} anti={ms_a:.3f} rand={ms_r_best:.3f} "
              f"({len(rows)}/{len(bases)}, {(time.time()-t0)/max(len(rows)-len(done_idx),1):.0f}s/base)", flush=True)
        tmp = partial + ".tmp"                            # atomic incremental save
        with open(tmp, "w") as f:
            json.dump(dict(rows=rows), f)
        os.replace(tmp, partial)

    ms0 = np.array([r["ms0"] for r in rows]); msg = np.array([r["ms_grad"] for r in rows])
    msa = np.array([r["ms_anti"] for r in rows]); msr = np.array([r["ms_rand"] for r in rows])
    ascent = msg > ms0
    beats_anti = msg > msa
    beats_rand = msg >= msr
    dd = np.array([r["dd"] for r in rows if r["dd"] is not None])

    def rate(mask):
        k, n = int(np.sum(mask)), int(len(mask))
        return dict(k=k, n=n, rate=float(k / n) if n else float("nan"), wilson95=wilson(k, n))

    out = dict(modes=MODES, n=len(rows),
               ascent=rate(ascent), beats_anti=rate(beats_anti), beats_random=rate(beats_rand),
               median_gain_grad=float(np.median(msg - ms0)),
               median_gain_rand=float(np.median(msr - ms0)),
               directional_corr=float(np.corrcoef(dd[:, 0], dd[:, 1])[0, 1]) if len(dd) > 3 else None,
               directional_sign=float(np.mean(np.sign(dd[:, 0]) == np.sign(dd[:, 1]))) if len(dd) else None,
               by_regime={}, rows=rows)
    # design regime = marginal + mid pooled
    dmask = np.array([r["regime"] in ("marginal", "mid") for r in rows])
    if dmask.any():
        out["design_regime_ascent"] = rate(ascent[dmask])
    for name, _, _ in D.REGIMES:
        sel = np.array([i for i, r in enumerate(rows) if r["regime"] == name])
        if len(sel) >= 1:
            out["by_regime"][name] = dict(
                ascent=rate(ascent[sel]), beats_random=rate(beats_rand[sel]),
                beats_anti=rate(beats_anti[sel]),
                median_gain_grad=float(np.median(msg[sel] - ms0[sel])))
    with open(os.path.join(ROOT, "data", "phase2_gradcheck2_80.json"), "w") as f:
        json.dump(out, f, indent=2)
    if os.path.exists(partial):       # run completed -> drop the resume scratch file
        os.remove(partial)

    print(f"\n=== GRADIENT ASCENT @ {MODES} MODES (in-distribution, {out['n']} bases) ===")
    for key in ["ascent", "beats_anti", "beats_random"]:
        r = out[key]
        print(f"  {key:13s}: {r['k']}/{r['n']} = {r['rate']*100:.0f}% "
              f"(Wilson95 {r['wilson95'][0]*100:.0f}-{r['wilson95'][1]*100:.0f}%)")
    dr = out.get("design_regime_ascent")
    if dr:
        print(f"  DESIGN REGIME (marginal+mid) ascent: {dr['k']}/{dr['n']} = {dr['rate']*100:.0f}% "
              f"(Wilson95 {dr['wilson95'][0]*100:.0f}-{dr['wilson95'][1]*100:.0f}%)")
    print(f"  median true m_s gain: grad={out['median_gain_grad']:+.3f}  rand={out['median_gain_rand']:+.3f}")
    print(f"  directional-derivative corr={out['directional_corr']}  sign={out['directional_sign']}")
    for name, r in out["by_regime"].items():
        a = r["ascent"]
        print(f"    {name:11s} n={a['n']:2d} ascent={a['rate']*100:.0f}% "
              f"(Wilson {a['wilson95'][0]*100:.0f}-{a['wilson95'][1]*100:.0f}%) "
              f"gain={r['median_gain_grad']:+.3f}")

    # side-by-side vs the 40-mode result, if present
    old_fn = os.path.join(ROOT, "data", "phase2_gradcheck2.json")
    if os.path.exists(old_fn):
        old = json.load(open(old_fn))
        print("\n  --- vs 40-mode (phase2_gradcheck2.json) ---")
        print(f"    pooled ascent: 40-mode {old.get('ascent_rate', float('nan'))*100:.0f}%  ->  "
              f"80-mode {out['ascent']['rate']*100:.0f}%")
        for name in ("marginal", "mid"):
            o = old.get("by_regime", {}).get(name, {})
            n80 = out["by_regime"].get(name, {}).get("ascent", {})
            if o and n80:
                print(f"    {name:9s} ascent: 40-mode {o.get('ascent_rate', float('nan'))*100:.0f}% "
                      f"(n={o.get('n')})  ->  80-mode {n80['rate']*100:.0f}% (n={n80['n']})")
    print("\nSaved data/phase2_gradcheck2_80.json")


if __name__ == "__main__":
    main()
