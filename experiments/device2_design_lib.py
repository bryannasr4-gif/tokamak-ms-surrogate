"""
device2_design_lib.py -- the UNCONSTRAINED design loop for the Device-C headline
(Phase 5 / Device-C). kappa is FREE (no projection, no kappa-drift rejection);
the question is whether the learned-m_s gradient beats the reduce-kappa heuristic AND gradient-free
search at raising TRUE m_s on Device-C, every step 80-mode solver-confirmed.

This is the unconstrained counterpart of phase4_gallery_lib (which HELD kappa fixed). Three methods,
all maximizing true m_s, all from the same start, same budget (same # true solves), same box:
  surrogate     : autodiff grad of (log m_s) via surrogate(ShapeMap_C(u(x))). RAW gradient -- NOT
                  projected off grad kappa (kappa is a free lever here). The learned-m_s lever.
  reduce_kappa  : the rigid "reduce elongation" heuristic -- step along -grad_x kappa (geometry only,
                  from the ShapeMap; no learned m_s). On the kappa-dominated MAST-U manifold this TIES
                  the surrogate; the second-device test asks if the surrogate now BEATS it.
  cma           : gradient-free CMA-ES on the true solver (the canonical derivative-free baseline).

FAIRNESS (inherited from phase25_kappa_lib post-review): each gradient method ranks candidates by its
OWN objective (surrogate->predicted m_s; reduce_kappa->predicted kappa reduction), every step is
confirmed by ONE true 80-mode solve and accepted only if TRUE m_s improves, and NO method breaks
early -- on a rejected step we shrink then reset the step so every method spends the full budget.

Reuses phase4_gallery_lib.true_full (+GRecorder) and phase25_kappa_lib gradient primitives, and
builds a Device-C DesignSpace (PCA over Device-C controls) via phase2_dim_lib.DesignSpace.
"""
import numpy as np

import phase2_data as D
import phase2_dim_lib as DL
import phase25_kappa_lib as KL
import phase4_gallery_lib as GAL

KAPPA_I = D.SHAPE_FEATURES.index("kappa")
MAX_STEP = 0.45


def build_design_space(U, u0, d=12):
    """Device-C DesignSpace: standardize controls, PCA (SVD), top-d PC-score search around u0,
    box = dataset [p2,p98] PC-score range (stay in the Device-C diverted distribution).
    U: (n,16) Device-C control matrix (CONTROL_FEATURES order). u0: (16,) start control vector."""
    U = np.asarray(U, dtype=np.float64)
    mu = U.mean(0)
    std = U.std(0) + 1e-8
    Z = (U - mu) / std
    # PCA via SVD of the centered standardized matrix; columns of V = PCs (descending variance)
    _, _, Vt = np.linalg.svd(Z - Z.mean(0), full_matrices=True)
    V = Vt.T
    scores = Z @ V
    lo = np.percentile(scores, 2, axis=0)
    hi = np.percentile(scores, 98, axis=0)
    return DL.DesignSpace(mu, std, V, lo, hi, np.asarray(u0, dtype=np.float64), d)


def _gradient_run(tok, ds, budget, kind, models, smap):
    """surrogate or reduce_kappa, UNCONSTRAINED, full recording, equal budget (reset-on-stall)."""
    ms0, k0, desc0, _ = GAL.true_full(tok, ds.u_of_x(ds.x0))
    rc = GAL.GRecorder(ms0, k0 if np.isfinite(k0) else 0.0, "?", desc0 or {"kappa": float("nan")}, ds.x0)
    best_x = ds.x0.copy()
    step0 = MAX_STEP
    while rc.n < budget:
        if kind == "surrogate":
            direction = KL._grad_feature(None, smap, ds, best_x, None, is_ms=True, models=models)
        elif kind == "reduce_kappa":
            direction = -KL._grad_feature(None, smap, ds, best_x, KAPPA_I)   # descend elongation
        else:
            raise ValueError(kind)
        nrm = np.linalg.norm(direction)
        if nrm < 1e-9:
            step0 = MAX_STEP / 2          # degenerate gradient: perturb step, keep spending budget
            direction = np.zeros_like(direction)
            direction[rc.n % len(direction)] = 1.0
            nrm = 1.0
        direction = direction / nrm
        cand, sc_best = None, -1e18
        for mag in (step0, step0 / 2, step0 / 4):
            xc = ds.clip(best_x + mag * direction)
            if kind == "surrogate":
                sc = KL._surr_ms(models, smap, ds, xc)                 # rank by predicted m_s
            else:
                sc = -KL.ds_feature(smap, ds, xc, KAPPA_I)             # rank by predicted kappa reduction
            if sc > sc_best:
                sc_best, cand = sc, xc
        ms, kappa, desc, _ = GAL.true_full(tok, ds.u_of_x(cand))
        if desc is None:
            rc.log_reject(ms, kappa, "invalid"); step0 *= 0.5
        elif ms > rc.best:
            rc.log_accept(ms, kappa, desc, cand); best_x = cand; step0 = MAX_STEP
        else:
            rc.log_reject(ms, kappa, "no_improve"); step0 *= 0.5
        if step0 < 0.03:
            step0 = MAX_STEP / 2          # local max along this lever: reset, keep spending budget
    return rc.result()


def _cma_run(tok, ds, budget, seed):
    """gradient-free CMA-ES, UNCONSTRAINED (maximize true m_s; no kappa penalty), full recording."""
    import cma
    ms0, k0, desc0, _ = GAL.true_full(tok, ds.u_of_x(ds.x0))   # gain measured from the REAL (unclipped) start
    rc = GAL.GRecorder(ms0, k0 if np.isfinite(k0) else 0.0, "?", desc0 or {"kappa": float("nan")}, ds.x0)
    es = cma.CMAEvolutionStrategy(list(ds.clip(ds.x0)), MAX_STEP, {   # feasible initial mean (box now contains x0)
        "bounds": [list(ds.box_lo), list(ds.box_hi)], "seed": int(seed) % (2 ** 31),
        "verbose": -9, "maxfevals": budget - 1})
    while rc.n < budget and not es.stop():
        sols = es.ask()
        fit = []
        for s in sols:
            if rc.n >= budget:
                fit.append(1e3); continue
            xc = ds.clip(s)
            ms, kappa, desc, _ = GAL.true_full(tok, ds.u_of_x(xc))
            if desc is None:
                rc.log_reject(ms, kappa, "invalid"); fit.append(1e3); continue
            if ms > rc.best:
                rc.log_accept(ms, kappa, desc, xc)
            else:
                rc.log_reject(ms, kappa, "no_improve")
            fit.append(-ms)
        es.tell(sols, fit)
    return rc.result()


def run_method(tok, ds, budget, method, models=None, smap=None, seed=0):
    """Dispatch one UNCONSTRAINED design run. method in {surrogate, reduce_kappa, cma}."""
    if method == "cma":
        return _cma_run(tok, ds, budget, seed)
    return _gradient_run(tok, ds, budget, method, models, smap)
