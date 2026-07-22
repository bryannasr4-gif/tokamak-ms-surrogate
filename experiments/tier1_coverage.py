"""
tier1_coverage.py -- test (a): shape-realism / coverage of the synthetic MAST-U training cloud by
REAL original-MAST EFIT shapes. Artifact-free: uses ONLY definitionally-consistent GEOMETRIC
descriptors (kappa, delta, aspect) computed the SAME way (bounding-box) for real (from stored LCFS)
and training (descriptors()). No re-solve, no surrogate -> the clean primary result.

  python experiments/tier1_coverage.py
"""
import os, sys, json
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tier1_lib as T

ROOT = T.ROOT
DATA = os.path.join(ROOT, "data")
FIG = os.path.join(ROOT, "figures")
GEOM = ["kappa", "delta", "aspect"]


def maha(train, X):
    mu = train.mean(0); cov = np.atleast_2d(np.cov(train, rowvar=False)); inv = np.linalg.pinv(cov)
    d = np.atleast_2d(X) - mu
    return np.sqrt(np.einsum("ij,jk,ik->i", d, inv, d))


def main():
    os.makedirs(FIG, exist_ok=True)
    tr = pd.read_parquet(os.path.join(DATA, "dataset_v1_80q.parquet"))
    trX = tr[GEOM].to_numpy(float)
    pool = pd.DataFrame(json.load(open(os.path.join(DATA, "tier1_pool.json")))["rows"])
    reX = pool[GEOM].to_numpy(float)
    good = np.all(np.isfinite(reX), axis=1); reX = reX[good]
    n = len(reX); n_shots = pool.loc[good, "shot"].nunique()

    md = maha(trX, reX); md_self = maha(trX, trX)
    thr99, thr95 = np.percentile(md_self, 99), np.percentile(md_self, 95)
    ood99 = float(np.mean(md > thr99)); ood95 = float(np.mean(md > thr95))

    # --- covariance-free cross-checks (close the "is 100% a covariance artifact?" question) ---
    xcheck = {}
    try:
        from scipy.spatial import Delaunay
        hull = Delaunay(trX)
        xcheck["convex_hull_ood"] = float(np.mean(hull.find_simplex(reX) < 0))  # fraction outside train hull
    except Exception as e:
        xcheck["convex_hull_ood"] = None
    # axis-aligned box (outside train min/max on >=1 axis)
    lo, hi = trX.min(0), trX.max(0)
    xcheck["box_ood_any_axis"] = float(np.mean(np.any((reX < lo) | (reX > hi), axis=1)))
    # diagonal-covariance (standardized-Euclidean, zero cross-correlation)
    zself = np.sqrt(((trX - trX.mean(0)) / trX.std(0))**2 @ np.ones(len(GEOM)))
    zreal = np.sqrt(((reX - trX.mean(0)) / trX.std(0))**2 @ np.ones(len(GEOM)))
    xcheck["diagcov_ood_99"] = float(np.mean(zreal > np.percentile(zself, 99)))

    # --- leave-one-out / per-axis OOD (is the gap trivially just aspect?) ---
    loo = {}
    for drop in [None, "aspect", "delta", "kappa"]:
        idx = [i for i, f in enumerate(GEOM) if f != drop]
        sub_tr, sub_re = trX[:, idx], reX[:, idx]
        m_self = maha(sub_tr, sub_tr); m_re = maha(sub_tr, sub_re)
        key = "all" if drop is None else f"drop_{drop}"
        loo[key] = float(np.mean(m_re > np.percentile(m_self, 99)))
    # single-axis OOD
    single = {}
    for i, f in enumerate(GEOM):
        m_self = maha(trX[:, [i]], trX[:, [i]]); m_re = maha(trX[:, [i]], reX[:, [i]])
        single[f] = float(np.mean(m_re > np.percentile(m_self, 99)))

    # kNN (standardized) distance
    mu, sd = trX.mean(0), trX.std(0); sd[sd == 0] = 1
    trz = (trX - mu) / sd; rez = (reX - mu) / sd
    knn = np.array([np.sort(np.linalg.norm(trz - x, axis=1))[:10].mean() for x in rez])

    print(f"COVERAGE on {n} real quality slices ({n_shots} shots) vs {len(trX)} synthetic training")
    print(f"  Mahalanobis OOD fraction  >99pct={ood99:.2f}  >95pct={ood95:.2f}")
    print(f"  covariance-free cross-checks: convex_hull_ood={xcheck['convex_hull_ood']} "
          f"box_ood(>=1axis)={xcheck['box_ood_any_axis']:.3f} diagcov_ood99={xcheck['diagcov_ood_99']:.2f}")
    print(f"  leave-one-out OOD@99: all={loo['all']:.2f} drop_aspect(kappa+delta)={loo['drop_aspect']:.2f} "
          f"drop_delta={loo['drop_delta']:.2f} drop_kappa={loo['drop_kappa']:.2f}")
    print(f"  single-axis OOD@99: " + " ".join(f"{f}={single[f]:.2f}" for f in GEOM)
          + "  (aspect device-trivial; delta is the substantive independent driver)")
    for i, f in enumerate(GEOM):
        rr = reX[:, i]; tt = trX[:, i]
        below = float(np.mean(rr < tt.min())); above = float(np.mean(rr > tt.max()))
        print(f"  {f:7s} real[{rr.min():.3f},{rr.max():.3f}] train[{tt.min():.3f},{tt.max():.3f}] "
              f"| frac below={below:.2f} above={above:.2f}")

    # figures
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
    ax[0].scatter(tr["kappa"], tr["delta"], s=4, c="0.75", label="synthetic MAST-U train")
    ax[0].scatter(reX[:, 0], reX[:, 1], s=14, c="crimson", edgecolor="k", lw=0.3, label="real MAST")
    ax[0].set_xlabel("elongation kappa"); ax[0].set_ylabel("triangularity delta"); ax[0].legend(fontsize=8)
    ax[0].set_title("(a) shape coverage: kappa-delta")
    ax[1].scatter(tr["kappa"], tr["aspect"], s=4, c="0.75")
    ax[1].scatter(reX[:, 0], reX[:, 2], s=14, c="crimson", edgecolor="k", lw=0.3)
    ax[1].set_xlabel("kappa"); ax[1].set_ylabel("aspect ratio"); ax[1].set_title("(a) kappa-aspect")
    ax[2].hist(md_self, bins=40, density=True, alpha=0.5, color="0.6", label="train self-dist")
    ax[2].hist(md, bins=40, density=True, alpha=0.6, color="crimson", label="real MAST")
    ax[2].axvline(thr99, ls="--", c="k", label="99pct train"); ax[2].set_xlabel("Mahalanobis distance (geom)")
    ax[2].legend(fontsize=8); ax[2].set_title(f"(a) OOD: {ood99:.0%} of real > 99pct")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "tier1_coverage.png"), dpi=120)
    print("saved figures/tier1_coverage.png")

    summ = dict(n_real=n, n_shots=int(n_shots), n_train=len(trX), features=GEOM,
                ood_frac_99=ood99, ood_frac_95=ood95,
                cov_free_crosschecks=xcheck, leave_one_out_ood99=loo, single_axis_ood99=single,
                per_feature={f: dict(real=[float(reX[:, i].min()), float(reX[:, i].max())],
                                     train=[float(trX[:, i].min()), float(trX[:, i].max())],
                                     frac_below=float(np.mean(reX[:, i] < trX[:, i].min())),
                                     frac_above=float(np.mean(reX[:, i] > trX[:, i].max())))
                             for i, f in enumerate(GEOM)})
    json.dump(summ, open(os.path.join(DATA, "tier1_coverage.json"), "w"), indent=2)
    print("saved data/tier1_coverage.json")


if __name__ == "__main__":
    main()
