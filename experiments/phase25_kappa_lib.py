"""
phase25_kappa_lib.py -- the kappa-CONSTRAINED design optimizers (the "beats heuristics" test).

Phase 2 found the m_s-gradient does not beat the reduce-kappa heuristic because kappa is the
dominant stability lever. Here we REMOVE that lever: hold kappa ~fixed (project every step
orthogonal to grad_x kappa from the ShapeMap; reject steps that move kappa beyond a tolerance)
and ask whether the learned m_s-gradient can still raise m_s using the SECONDARY shape levers
(squareness, gaps, l_i, delta) -- and whether it beats (a) the best single-secondary-lever
heuristic and (b) gradient-free search, both also kappa-constrained.

Reuses phase2_dim_lib.DesignSpace. All methods bounded-local-step; every step true-solver
confirmed and accepted only if true m_s improves AND |kappa - kappa_start| <= KTOL.
"""
import numpy as np
import torch

import phase2_data as D
import phase2_dim_lib as DL

KTOL = 0.04          # allowed kappa drift from the start (about the descriptor resolution)
MAX_STEP = 0.45
KAPPA_I = D.SHAPE_FEATURES.index("kappa")
# secondary levers a non-learned physicist might try (each is one shape descriptor to push)
SECONDARY = ["sq_uo", "sq_lo", "gap_outer", "gap_inner", "li", "delta", "betap"]


def true_ms_kappa(tok, u):
    """True (m_s, kappa) at controls u; (0,nan) on failure."""
    import phase15_lib as L
    c = DL.ctrl_from_u(u)
    try:
        r = L.forward_label(tok, c["active_currents"], c["paxis"], c["Ip"], c["fvac"],
                            c["alpha_m"], c["alpha_n"], fix_n_modes=80)
        ms = r["m_s"]
        if not np.isfinite(ms) or ms <= 0:
            return 0.0, float("nan")
        return float(ms), float(r["kappa"])
    except Exception:
        return 0.0, float("nan")


def _grad_feature(models_or_smap, smap, ds, x_np, feat_i, is_ms=False, models=None):
    """grad_x of (log m_s) [is_ms] or of shape-feature feat_i, at x."""
    x = torch.tensor(x_np, dtype=torch.float32, requires_grad=True)
    if is_ms:
        gs = []
        for m in models:
            out = m(smap(ds.u_of_x_t(x).unsqueeze(0)))[0, 0]
            g, = torch.autograd.grad(out, x, retain_graph=True)
            gs.append(g.numpy())
        return np.mean(gs, 0)
    out = smap(ds.u_of_x_t(x).unsqueeze(0))[0, feat_i]
    g, = torch.autograd.grad(out, x)
    return g.numpy()


def _grad_ms_zeroed(models, smap, ds, x_np, zero_feat_i):
    """grad_x of (log m_s) with ONE descriptor channel of d(log m_s)/d(descriptors) zeroed before
    back-propagation to x -- the S4a ablation (unit S4a, 2026-07-31; design
    data/audit/strategy/S4A_S3C_DESIGN.md).

    Implemented as an explicit vector-Jacobian product so the mask is applied in DESCRIPTOR space
    (where l_i lives), not in the 12-dim PCA control space. With zero_feat_i=None this returns the
    production gradient bit-identically (verified: max rel diff 0.00e+00 vs _grad_feature, see
    experiments/s34_preflight.py §7) -- so it is safe as a drop-in, but callers on the unablated
    path deliberately still use _grad_feature so that path is untouched byte-for-byte.
    """
    gs = []
    for m in models:
        x = torch.tensor(x_np, dtype=torch.float32, requires_grad=True)
        d = smap(ds.u_of_x_t(x).unsqueeze(0))          # (1, n_shape_features)
        out = m(d)[0, 0]                                # log m_s
        g_d, = torch.autograd.grad(out, d, retain_graph=True)
        if zero_feat_i is not None:
            g_d = g_d.clone()
            g_d[0, int(zero_feat_i)] = 0.0
        g_x, = torch.autograd.grad(d, x, grad_outputs=g_d)
        gs.append(g_x.numpy())
    return np.mean(gs, 0)


def _project_off_kappa(g, gk):
    gkn = gk / (np.linalg.norm(gk) + 1e-12)
    return g - np.dot(g, gkn) * gkn


def _surr_ms(models, smap, ds, x_np):
    with torch.no_grad():
        u = ds.u_of_x_t(torch.tensor(x_np, dtype=torch.float32))
        return float(torch.stack([m(smap(u.unsqueeze(0)))[0, 0] for m in models]).mean().exp())


def run_constrained(tok, models, smap, ds, budget, kind, k_start, rng=None):
    """kind in {'surrogate','heuristic:<feat>','random','cma'}. Returns trajectory + best."""
    best_x = ds.x0.copy()
    ms0, k0 = true_ms_kappa(tok, ds.u_of_x(best_x))
    best = ms0; n = 1
    traj = [(n, best)]
    step0 = MAX_STEP
    feat_i = None      # NOTE: feat_i (and the +/- sign) are resolved in the gradient/heuristic
                       # block below, which strips the +/- suffix; do NOT index SHAPE_FEATURES here.

    def accept(u):
        ms, k = true_ms_kappa(tok, u)
        return ms, k, (ms > best and np.isfinite(k) and abs(k - k_start) <= KTOL)

    if kind in ("random", "cma"):
        # kappa-constrained local search: penalize kappa drift in the objective
        if kind == "cma":
            import cma
            es = cma.CMAEvolutionStrategy(list(ds.x0), MAX_STEP,
                {"bounds": [list(ds.box_lo), list(ds.box_hi)], "seed": int(rng) % (2**31),
                 "verbose": -9, "maxfevals": budget - 1})
            while n < budget and not es.stop():
                sols = es.ask(); fit = []
                for s in sols:
                    if n >= budget:
                        fit.append(1e3); continue
                    ms, k = true_ms_kappa(tok, ds.u_of_x(ds.clip(s))); n += 1
                    viol = 0.0 if (np.isfinite(k) and abs(k - k_start) <= KTOL) else 10.0
                    if viol == 0.0 and ms > best:
                        best = ms
                    traj.append((n, best)); fit.append(-ms + viol)
                es.tell(sols, fit)
        else:
            g = np.random.default_rng(rng)
            while n < budget:
                r = g.standard_normal(ds.d); r /= np.linalg.norm(r) + 1e-12
                u = ds.u_of_x(ds.clip(best_x + MAX_STEP * r))
                ms, k, ok = accept(u); n += 1
                if ok:
                    best, best_x = ms, ds.clip(best_x + MAX_STEP * r)
                traj.append((n, best))
        return dict(traj=traj, n_solves=n, best_ms=best, kappa_start=k_start)

    # gradient methods (surrogate m_s, or single secondary-lever heuristic), kappa-projected.
    # FAIRNESS (post-review M6): (1) heuristics may push their descriptor in EITHER sign -- the sign is
    # given by a "+"/"-" suffix on the method ("heuristic:sq_uo+"); (2) candidates are ranked by each
    # method's OWN objective (surrogate -> predicted m_s; heuristic -> its descriptor change), NOT by the
    # surrogate for everyone; (3) NO early break -- on a rejected step we shrink/reset and keep consuming
    # the full budget, so every method spends the same number of true solves.
    sign = 1.0
    if kind.startswith("heuristic:"):
        tag = kind.split(":")[1]
        if tag.endswith("+") or tag.endswith("-"):
            sign = 1.0 if tag.endswith("+") else -1.0
            tag = tag[:-1]
        feat_i = D.SHAPE_FEATURES.index(tag)

    def feat_val(x):                       # shapemap-predicted descriptor value (free)
        return ds_feature(smap, ds, x, feat_i)

    while n < budget:
        gk = _grad_feature(None, smap, ds, best_x, KAPPA_I)
        if kind == "surrogate":
            graw = _grad_feature(None, smap, ds, best_x, None, is_ms=True, models=models)
        else:
            graw = sign * _grad_feature(None, smap, ds, best_x, feat_i)   # push descriptor in chosen sign
        direction = _project_off_kappa(graw, gk)
        nrm = np.linalg.norm(direction)
        if nrm < 1e-9:
            break
        direction /= nrm
        cand, sc_best = None, -1e18
        f0 = None if kind == "surrogate" else feat_val(best_x)
        for mag in (step0, step0 / 2, step0 / 4):
            xc = ds.clip(best_x + mag * direction)
            sc = _surr_ms(models, smap, ds, xc) if kind == "surrogate" else sign * (feat_val(xc) - f0)
            if sc > sc_best:
                sc_best, cand = sc, xc
        ms, k, ok = accept(ds.u_of_x(cand)); n += 1
        if ok:
            best, best_x = ms, cand
            step0 = MAX_STEP                # reset to full step after a successful move
        else:
            step0 *= 0.5
            if step0 < 0.03:               # local max along this lever: reset & keep spending budget
                step0 = MAX_STEP / 2
        traj.append((n, best))
    return dict(traj=traj, n_solves=n, best_ms=best, kappa_start=k_start)


def _surr_ms_pinned(models, smap, ds, x_np, feat_i, pin_value):
    """Surrogate m_s scored with ONE descriptor PINNED to a fixed value -- i.e. 'score this
    candidate as if <feat> had never moved from its start value'. Used only by the S4a arm B
    (gradient AND value-function ablation), which the council-before round required as the
    'pure' ablation. With feat_i=None this is identical to _surr_ms."""
    with torch.no_grad():
        u = ds.u_of_x_t(torch.tensor(x_np, dtype=torch.float32))
        d = smap(u.unsqueeze(0))
        if feat_i is not None:
            d = d.clone()
            d[0, int(feat_i)] = float(pin_value)
        return float(torch.stack([m(d)[0, 0] for m in models]).mean().exp())


def ds_feature(smap, ds, x_np, feat_i):
    with torch.no_grad():
        u = ds.u_of_x_t(torch.tensor(x_np, dtype=torch.float32))
        return float(smap(u.unsqueeze(0))[0, feat_i])
