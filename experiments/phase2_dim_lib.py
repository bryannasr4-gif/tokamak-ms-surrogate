"""
phase2_dim_lib.py -- the design space + optimizers for the Phase-2 dimensionality experiment.

The contribution claim: a DIFFERENTIABLE, amortized m_s surrogate turns a many-query,
increasing-dimension stability-shaping problem (where gradient-free search is expensive) into a
cheap one -- and the advantage GROWS with the true dimensionality of the design space. We run
the comparison in PCA-orthogonalized CONTROL space (the only honest "increasing true dimension"
axis -- the shape descriptors are a correlated observational cloud, DATASET.md).

Design space (per base equilibrium u0, dimension d):
  * standardize controls z=(u-mu)/std; PCA over the dataset -> orthonormal columns V (16x16).
  * search the TOP-d PC scores x in R^d; the remaining scores are frozen at the base's values.
    u(x) = mu + std * (V @ c),  c = c0 with c[:d]=x   (c0 = V^T z0).
  * box: each x_j in the dataset [p2,p98] PC-score range (stay in-distribution / diverted).

Optimizers (all maximize true m_s; every true forward_label counts as one "true solve"):
  surrogate_grad : gradient ascent on the differentiable surrogate m_s(shapemap(u(x))); each
                   step proposes a surrogate-line-searched candidate, CONFIRMED by ONE true solve.
  heuristic      : the rigid "reduce elongation kappa" rule -- step along -grad_x kappa (geometry
                   only, from the shape map; no learned m_s), confirmed by one true solve/step.
  cma            : CMA-ES on the TRUE solver (canonical gradient-free baseline).
  random         : uniform random search on the TRUE solver (scales worst with d).
"""
import numpy as np
import torch

import phase2_data as D

IDX_MAP = {f"I_{c}": i for i, c in enumerate(D.ACTIVE_COILS)}
KAPPA_I = D.SHAPE_FEATURES.index("kappa")


def ctrl_from_u(u):
    """forward_label control dict from a 16-vector u (CONTROL_FEATURES order)."""
    I = np.zeros(12); I[0] = 5000.0
    prof = {}
    for name, val in zip(D.CONTROL_FEATURES, u):
        if name.startswith("I_"):
            I[IDX_MAP[name]] = float(val)
        else:
            prof[name] = float(val)
    return dict(active_currents=I, paxis=prof["paxis"], Ip=prof["Ip_target"],
                fvac=prof["fvac"], alpha_m=prof["alpha_m"], alpha_n=prof["alpha_n"])


def true_ms(tok, u, fix_n_modes=80):
    """True m_s at controls u, or 0.0 penalty if the solve fails / not diverted / non-finite."""
    import phase15_lib as L
    c = ctrl_from_u(u)
    try:
        rec = L.forward_label(tok, c["active_currents"], c["paxis"], c["Ip"], c["fvac"],
                              c["alpha_m"], c["alpha_n"], fix_n_modes=fix_n_modes)
        ms = rec["m_s"]
        return float(ms) if np.isfinite(ms) and ms > 0 else 0.0
    except Exception:
        return 0.0


class DesignSpace:
    """Top-d PC-score search around a base u0; torch-differentiable u(x)."""
    def __init__(self, mu, std, V, box_lo, box_hi, u0, d):
        self.mu = torch.tensor(mu, dtype=torch.float32)
        self.std = torch.tensor(std, dtype=torch.float32)
        self.V = torch.tensor(V, dtype=torch.float32)            # (16,16) columns=PCs
        self.d = d
        z0 = (np.asarray(u0) - mu) / std
        self.c0 = torch.tensor(V.T @ z0, dtype=torch.float32)    # (16,)
        self.box_lo = np.asarray(box_lo)[:d]
        self.box_hi = np.asarray(box_hi)[:d]
        self.x0 = (V.T @ z0)[:d].copy()

    def u_of_x_t(self, x):                                       # torch, differentiable
        c = self.c0.clone()
        c = torch.cat([x, c[self.d:]])
        z = self.V @ c
        return self.mu + self.std * z

    def u_of_x(self, x):                                         # numpy
        with torch.no_grad():
            return self.u_of_x_t(torch.tensor(np.asarray(x), dtype=torch.float32)).numpy()

    def clip(self, x):
        return np.clip(x, self.box_lo, self.box_hi)


def _ens_logms(models, smap, u_t):
    return torch.stack([m(smap(u_t.unsqueeze(0)))[:, 0] for m in models]).mean()


def _grad_x(models, smap, ds, x_np, kappa=False):
    """mean-ensemble grad of (log m_s) or (kappa) w.r.t. x at x_np."""
    x = torch.tensor(x_np, dtype=torch.float32, requires_grad=True)
    if kappa:
        out = smap(ds.u_of_x_t(x).unsqueeze(0))[0, KAPPA_I]
        g, = torch.autograd.grad(out, x)
        return g.numpy()
    gs = []
    for m in models:
        out = m(smap(ds.u_of_x_t(x).unsqueeze(0)))[0, 0]
        g, = torch.autograd.grad(out, x, retain_graph=True)
        gs.append(g.numpy())
    return np.mean(gs, 0)


def _surr_ms(models, smap, ds, x_np):
    with torch.no_grad():
        u = ds.u_of_x_t(torch.tensor(x_np, dtype=torch.float32))
        return float(torch.stack([m(smap(u.unsqueeze(0)))[0, 0] for m in models]).mean().exp())


# LOCAL bounded step size (standardized control / PC-score units). All methods take steps of at
# most MAX_STEP per iteration, so the experiment isolates "find the ascent DIRECTION" (local
# navigation from the marginal start) rather than global rejection-sampling of the stable-rich box.
MAX_STEP = 0.45


def run_gradient(tok, models, smap, ds, target, budget, kind="surrogate"):
    """surrogate_grad or heuristic, as LOCAL bounded-step hill-climbing. Traj of (n_solves, best)."""
    best_x = ds.x0.copy()
    best = true_ms(tok, ds.u_of_x(best_x)); n = 1
    traj = [(n, best)]
    step0 = MAX_STEP
    while n < budget and best < target:
        g = _grad_x(models, smap, ds, best_x, kappa=(kind == "heuristic"))
        direction = -g if kind == "heuristic" else g     # heuristic: descend kappa
        nrm = np.linalg.norm(direction)
        if nrm < 1e-9:
            break
        direction = direction / nrm
        # cheap surrogate line-search over a few step magnitudes <= step0; confirm best with 1 solve.
        cand, cand_score = None, -1e18
        for mag in (step0, step0 / 2, step0 / 4):
            xc = ds.clip(best_x + mag * direction)
            sc = _surr_ms(models, smap, ds, xc) if kind != "heuristic" else -ds_kappa(smap, ds, xc)
            if sc > cand_score:
                cand_score, cand = sc, xc
        ms = true_ms(tok, ds.u_of_x(cand)); n += 1
        if ms > best:
            best, best_x = ms, cand
        else:
            step0 *= 0.5
            if step0 < 0.03:
                break
        traj.append((n, best))
    return dict(traj=traj, n_solves=n, best_ms=best, reached=best >= target)


def ds_kappa(smap, ds, x_np):
    with torch.no_grad():
        u = ds.u_of_x_t(torch.tensor(x_np, dtype=torch.float32))
        return float(smap(u.unsqueeze(0))[0, KAPPA_I])


def run_random(tok, ds, target, budget, seed):
    """LOCAL random hill-climbing: propose a bounded random step from the current best, accept if
    the true m_s improves. (NOT global box sampling -- that would just rejection-sample the
    stable-rich box and trivially 'win' without navigating; see the Phase-2 ledger.)"""
    rng = np.random.default_rng(seed)
    best_x = ds.x0.copy()
    best = true_ms(tok, ds.u_of_x(best_x)); n = 1
    traj = [(n, best)]
    while n < budget and best < target:
        r = rng.standard_normal(ds.d); r /= (np.linalg.norm(r) + 1e-12)
        x = ds.clip(best_x + MAX_STEP * r)
        ms = true_ms(tok, ds.u_of_x(x)); n += 1
        if ms > best:
            best, best_x = ms, x
        traj.append((n, best))
    return dict(traj=traj, n_solves=n, best_ms=best, reached=best >= target)


def run_cma(tok, ds, target, budget, seed):
    """LOCAL CMA-ES from the marginal start (sigma0 = MAX_STEP), box-bounded."""
    import cma
    best = true_ms(tok, ds.u_of_x(ds.x0)); n = 1
    traj = [(n, best)]
    x0 = ds.x0.copy()
    sig0 = MAX_STEP
    es = cma.CMAEvolutionStrategy(list(x0), sig0, {
        "bounds": [list(ds.box_lo), list(ds.box_hi)], "seed": int(seed) % (2**31),
        "verbose": -9, "maxfevals": budget - 1})
    while n < budget and best < target and not es.stop():
        sols = es.ask()
        fit = []
        for s in sols:
            if n >= budget:
                fit.append(1e3); continue
            ms = true_ms(tok, ds.u_of_x(ds.clip(s))); n += 1
            best = max(best, ms); traj.append((n, best))
            fit.append(-ms)
        es.tell(sols, fit)
    return dict(traj=traj, n_solves=n, best_ms=best, reached=best >= target)
