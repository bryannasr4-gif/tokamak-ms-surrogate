"""
phase1_analyze.py -- Phase-1 make-or-break analysis.

Given the true FreeGSNKE m_s(shape) grid from phase1_generate.py, this:
  1. trains a small DIFFERENTIABLE ensemble surrogate m_s_hat(zscale, dR) (smooth tanh MLP);
  2. THE TEST: compares the surrogate's autodiff gradient d(m_s)/d(shape) against the
     true-solver finite-difference gradient (central differences on the grid) at HELD-OUT
     shapes -- direction (cosine + sign) and magnitude;
  3. THE CONTRIBUTION DEMO: from a marginally-stable shape, takes a gradient-ascent step on
     the surrogate m_s and confirms with a FRESH FreeGSNKE solve that the true m_s increased,
     versus an analytic-"reduce kappa" heuristic and a finite-difference-on-surrogate step.

Pass (greenlight Phase 2): held-out gradient direction agrees with the true solver
(median cosine > ~0.9, sign agreement > 90%) and the gradient step raises the true m_s.
"""
import glob
import json
import os
import sys

import numpy as np
import torch

try:  # Windows console defaults to cp1252; make non-ASCII prints safe
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
torch.manual_seed(0)
np.random.seed(0)


# ----------------------------------------------------------------------------- data
def load_grid():
    recs = []
    for fn in sorted(glob.glob("data/phase1_chunk_*.json")):
        with open(fn) as f:
            recs += json.load(f)
    recs = [r for r in recs if np.isfinite(r.get("m_s", np.nan))]
    zs = np.array(sorted(set(round(r["zscale"], 4) for r in recs)))
    ds = np.array(sorted(set(round(r["dR"], 4) for r in recs)))
    M = np.full((len(zs), len(ds)), np.nan)
    K = np.full((len(zs), len(ds)), np.nan)
    for r in recs:
        i = np.argmin(abs(zs - r["zscale"])); j = np.argmin(abs(ds - r["dR"]))
        M[i, j] = r["m_s"]; K[i, j] = r["kappa"]
    return recs, zs, ds, M, K


# ----------------------------------------------------------------------------- model
class Surrogate(nn.Module):
    """Smooth MLP with INTERNAL input normalization, so autograd w.r.t. the raw physical
    (zscale, dR) gives the physical gradient directly. tanh => smooth, well-defined gradient."""
    def __init__(self, xmean, xstd, ymean, ystd, h=64):
        super().__init__()
        self.register_buffer("xmean", torch.tensor(xmean, dtype=torch.float32))
        self.register_buffer("xstd", torch.tensor(xstd, dtype=torch.float32))
        self.register_buffer("ymean", torch.tensor(ymean, dtype=torch.float32))
        self.register_buffer("ystd", torch.tensor(ystd, dtype=torch.float32))
        self.net = nn.Sequential(nn.Linear(2, h), nn.Tanh(), nn.Linear(h, h), nn.Tanh(), nn.Linear(h, 1))

    def forward(self, x):
        z = (x - self.xmean) / self.xstd
        return (self.net(z) * self.ystd + self.ymean).squeeze(-1)


def train_ensemble(Xtr, Ytr, n_models=7, epochs=3000, wd=2e-3):
    xmean, xstd = Xtr.mean(0), Xtr.std(0)
    ymean, ystd = Ytr.mean(), Ytr.std()
    Xt = torch.tensor(Xtr, dtype=torch.float32); Yt = torch.tensor(Ytr, dtype=torch.float32)
    models = []
    for m in range(n_models):
        torch.manual_seed(m)
        net = Surrogate(xmean, xstd, ymean, ystd)
        opt = torch.optim.Adam(net.parameters(), lr=5e-3, weight_decay=wd)
        for ep in range(epochs):
            opt.zero_grad()
            loss = ((net(Xt) - Yt) ** 2).mean()
            loss.backward(); opt.step()
        models.append(net)
    return models


def predict(models, X):
    Xt = torch.tensor(np.atleast_2d(X), dtype=torch.float32)
    with torch.no_grad():
        ys = torch.stack([m(Xt) for m in models])
    return ys.mean(0).numpy(), ys.std(0).numpy()


def grad(models, X):
    """Mean ensemble gradient d(m_s_hat)/d(zscale,dR) at rows of X (physical units)."""
    Xt = torch.tensor(np.atleast_2d(X), dtype=torch.float32, requires_grad=True)
    gs = []
    for m in models:
        y = m(Xt).sum()
        g, = torch.autograd.grad(y, Xt, create_graph=False, retain_graph=False)
        gs.append(g.detach().numpy())
    return np.mean(gs, axis=0)


# ----------------------------------------------------------------------------- main
def main():
    recs, zs, ds, M, K = load_grid()
    n_valid = np.isfinite(M).sum()
    print(f"Loaded grid {M.shape}, {n_valid}/{M.size} valid points.")
    if n_valid < 30:
        print("Not enough data yet."); return

    # build sample list (only valid)
    pts, ys = [], []
    for i, z in enumerate(zs):
        for j, d in enumerate(ds):
            if np.isfinite(M[i, j]):
                pts.append([z, d]); ys.append(M[i, j])
    pts = np.array(pts); ys = np.array(ys)

    # train/test split
    rng = np.random.default_rng(1)
    idx = rng.permutation(len(pts))
    ntest = max(10, len(pts) // 4)
    te, tr = idx[:ntest], idx[ntest:]
    models = train_ensemble(pts[tr], ys[tr])

    # held-out value accuracy
    yhat, ystd = predict(models, pts[te])
    rmse = float(np.sqrt(np.mean((yhat - ys[te]) ** 2)))
    denom = ys[te].max() - ys[te].min()
    r2 = float(1 - np.sum((yhat - ys[te]) ** 2) / np.sum((ys[te] - ys[te].mean()) ** 2))
    print(f"\nHeld-out m_s accuracy: RMSE={rmse:.4f}  (range {denom:.3f})  R2={r2:.3f}")

    # ---- THE TEST: gradient agreement at interior held-out grid nodes ----
    # true-solver FD gradient via central differences on the full grid
    with np.errstate(invalid="ignore"):
        gZ, gD = np.gradient(M, zs, ds)  # d m_s / d zscale , d m_s / d dR
    test_set = {(round(pts[k][0], 4), round(pts[k][1], 4)) for k in te}
    rows = []
    for i in range(1, len(zs) - 1):
        for j in range(1, len(ds) - 1):
            key = (round(zs[i], 4), round(ds[j], 4))
            if key not in test_set:
                continue
            gt = np.array([gZ[i, j], gD[i, j]])
            if not np.all(np.isfinite(gt)):
                continue
            gs = grad(models, [zs[i], ds[j]])[0]
            cos = float(np.dot(gs, gt) / (np.linalg.norm(gs) * np.linalg.norm(gt) + 1e-12))
            rows.append(dict(z=float(zs[i]), dR=float(ds[j]),
                             g_true=gt.tolist(), g_sur=gs.tolist(), cosine=cos,
                             sign_z=int(np.sign(gs[0]) == np.sign(gt[0])),
                             sign_d=int(np.sign(gs[1]) == np.sign(gt[1])),
                             magratio=float(np.linalg.norm(gs) / (np.linalg.norm(gt) + 1e-12))))
    cosines = np.array([r["cosine"] for r in rows])
    signs = np.array([r["sign_z"] and r["sign_d"] for r in rows])
    signz = np.mean([r["sign_z"] for r in rows]); signd = np.mean([r["sign_d"] for r in rows])
    magr = np.array([r["magratio"] for r in rows])
    print(f"\n=== GRADIENT TEST on {len(rows)} held-out interior shapes ===")
    print(f"  cosine(sur,true): median={np.median(cosines):.3f}  min={cosines.min():.3f}  "
          f"frac>0.9={np.mean(cosines>0.9):.2f}")
    print(f"  full-vector sign agreement: {np.mean(signs)*100:.0f}%   (d/dz: {signz*100:.0f}%, d/ddR: {signd*100:.0f}%)")
    print(f"  |grad| ratio sur/true: median={np.median(magr):.2f}  (IQR {np.percentile(magr,25):.2f}-{np.percentile(magr,75):.2f})")
    grad_pass = (np.median(cosines) > 0.9) and (np.mean(signs) > 0.9)
    print(f"  -> GRADIENT {'PASS' if grad_pass else 'MARGINAL/FAIL'}")

    # ---- THE CONTRIBUTION DEMO: gradient-ascent step confirmed by a fresh solve ----
    from phase1_generate import build_tokamak, evaluate
    tok = build_tokamak()
    # start = most marginal (lowest true m_s) shape in the grid.
    # Re-solve the start with THIS build/config so start and destinations are strictly
    # comparable (the grid value may sit in a different numerical config; see the audit).
    s_i, s_j = np.unravel_index(np.nanargmin(M), M.shape)
    start = np.array([zs[s_i], ds[s_j]])
    start_ms = float(evaluate(tok, float(start[0]), float(start[1]))["m_s"])
    g0 = grad(models, start)[0]
    step_z, step_dR = (zs[-1] - zs[0]) * 0.30, (ds[-1] - ds[0]) * 0.30  # ~30% of each range

    def clamp(p):
        return np.array([np.clip(p[0], zs[0], zs[-1]), np.clip(p[1], ds[0], ds[-1])])

    gdir = g0 / (np.linalg.norm(g0) + 1e-12)
    p_grad = clamp(start + np.array([step_z, step_dR]) * gdir)          # surrogate-gradient step
    p_kappa = clamp(start + np.array([-step_z, 0.0]) * np.sign(1.0))    # analytic: just reduce elongation
    # finite-difference-on-surrogate direction (should ~equal autodiff)
    eps = np.array([(zs[1]-zs[0]), (ds[1]-ds[0])])
    fd = np.array([(predict(models, start+[eps[0],0])[0]-predict(models, start-[eps[0],0])[0])[0],
                   (predict(models, start+[0,eps[1]])[0]-predict(models, start-[0,eps[1]])[0])[0]])
    fddir = fd / (np.linalg.norm(fd) + 1e-12)
    p_fd = clamp(start + np.array([step_z, step_dR]) * fddir)

    print("\n=== CLOSED-LOOP DEMO (fresh FreeGSNKE solves) ===")
    print(f"  start shape z={start[0]:.3f} dR={start[1]:+.3f}  true m_s={start_ms:.3f}  (kappa={K[s_i,s_j]:.3f})")
    demo = {"start": {"shape": start.tolist(), "true_m_s": start_ms}}
    for name, p in [("grad_ascent", p_grad), ("analytic_reduce_kappa", p_kappa), ("fd_on_surrogate", p_fd)]:
        try:
            r = evaluate(tok, float(p[0]), float(p[1]))
            dms = r["m_s"] - start_ms
            print(f"  {name:22s}: z={p[0]:.3f} dR={p[1]:+.3f} -> true m_s={r['m_s']:.3f}  (delta={dms:+.3f}, kappa={r['kappa']:.3f})")
            demo[name] = {"shape": p.tolist(), "true_m_s": r["m_s"], "delta": dms}
        except Exception as e:
            print(f"  {name:22s}: FAILED {e}")
            demo[name] = {"shape": p.tolist(), "error": str(e)}
    loop_pass = demo.get("grad_ascent", {}).get("delta", -1) > 0

    # ---- save + figures ----
    results = dict(rmse=rmse, r2=r2, grad_rows=rows, median_cosine=float(np.median(cosines)),
                   sign_agreement=float(np.mean(signs)), grad_pass=bool(grad_pass),
                   demo=demo, loop_pass=bool(loop_pass))
    with open("data/phase1_results.json", "w") as f:
        json.dump(results, f, indent=2)

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    ZZ, DD = np.meshgrid(zs, ds, indexing="ij")
    c0 = ax[0].contourf(ZZ, DD, M, 18, cmap="viridis")
    ax[0].set_title("true $m_s$(elongation, wall-gap)"); ax[0].set_xlabel("zscale (κ↑)"); ax[0].set_ylabel("dR (gap)")
    plt.colorbar(c0, ax=ax[0])

    # gradient quiver: true (black) vs surrogate (red) at test nodes
    ax[1].contour(ZZ, DD, M, 12, cmap="viridis", alpha=0.5)
    for r in rows:
        gt = np.array(r["g_true"]); gs = np.array(r["g_sur"])
        gt = gt/ (np.linalg.norm(gt)+1e-9); gs = gs/(np.linalg.norm(gs)+1e-9)
        ax[1].arrow(r["z"], r["dR"], gt[0]*0.012, gt[1]*0.012, color="k", width=0.0004, head_width=0.003)
        ax[1].arrow(r["z"], r["dR"], gs[0]*0.012, gs[1]*0.012, color="r", width=0.0004, head_width=0.003, alpha=0.8)
    ax[1].set_title(f"∇$m_s$ direction: true(blk) vs surrogate(red)\nmedian cos={np.median(cosines):.2f}, sign {np.mean(signs)*100:.0f}%")
    ax[1].set_xlabel("zscale"); ax[1].set_ylabel("dR")

    ax[2].contourf(ZZ, DD, M, 18, cmap="viridis", alpha=0.7)
    ax[2].plot(*start, "wo", ms=10, mec="k"); ax[2].annotate("start", start, color="w")
    for name, col in [("grad_ascent", "red"), ("analytic_reduce_kappa", "orange"), ("fd_on_surrogate", "magenta")]:
        if name in demo and "shape" in demo[name] and "true_m_s" in demo[name]:
            p = demo[name]["shape"]
            ax[2].plot([start[0], p[0]], [start[1], p[1]], "-o", color=col,
                       label=f"{name} Δ={demo[name]['delta']:+.2f}")
    ax[2].legend(fontsize=7, loc="lower left"); ax[2].set_title("closed-loop steps (Δ true $m_s$)")
    ax[2].set_xlabel("zscale"); ax[2].set_ylabel("dR")
    fig.suptitle(f"Phase 1: differentiable $m_s$(shape) surrogate — gradient {'PASS' if grad_pass else 'CHECK'}, "
                 f"closed-loop {'PASS' if loop_pass else 'CHECK'}", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig("figures/phase1_gradient_validation.png", dpi=140, bbox_inches="tight")
    print("\nSaved figures/phase1_gradient_validation.png and data/phase1_results.json")
    print(f"\nVERDICT: gradient {'PASS' if grad_pass else 'CHECK'} | closed-loop {'PASS' if loop_pass else 'CHECK'}")


if __name__ == "__main__":
    main()
