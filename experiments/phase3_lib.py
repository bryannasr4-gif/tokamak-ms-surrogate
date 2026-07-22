"""
phase3_lib.py -- the Phase-3 solver-confirmed differentiable design loop + fair gradient-free
baselines (THE contribution).

DESIGN TASK (per start shape u0): RAISE the Portone-2005 inductive stability margin m_s from a
MARGINAL/MID start toward a target m* (cross into the 'stable' regime), by LOCAL bounded-step
optimization in the d-dim PCA control space, while keeping the design a VALID diverted ST
equilibrium with the key shape descriptors (kappa, delta, gaps, li) IN RANGE. Every candidate step
is CONFIRMED by a FRESH true 80-mode FreeGSNKE solve at the locked protocol; steps that fail to
converge / go limited (off the diverted manifold) / drive a descriptor out of range / do not
improve the true m_s are REJECTED (the loop is robust to an imperfect single-step gradient -- the
Phase-2.5b finding that the raw 80-mode gradient is a *direction, not a Jacobian*).

METHODS compared at the SAME true-solve budget (every forward_label = one expensive solve):
  surrogate      : autodiff gradient ascent on the differentiable surrogate m_s(shapemap(u(x))).
                   The surrogate line-search over step magnitudes is FREE; ONE true solve confirms
                   the best candidate per iteration. This is the amortized differentiable design loop.
  heuristic:kappa: the rigid "reduce elongation" physics rule (ShapeMap geometry gradient of kappa;
                   NO learned m_s). The strongest single-lever heuristic on this kappa-dominated manifold.
  cma            : CMA-ES on the TRUE solver (canonical gradient-free baseline).
  random         : local random hill-climbing on the TRUE solver.
  nelder         : Nelder-Mead simplex on the TRUE solver (classic derivative-free local optimizer).

FIGURE OF MERIT: true-solves-to-target (expensive solves) + final true m_s, resolved by m_s regime
over >=20 stratified marginal+mid starts. The amortized surrogate's value = far FEWER expensive
solves for many-query design (its training cost is paid once, offline).

Reuses phase2_dim_lib.DesignSpace / ctrl_from_u / _grad_x / _surr_ms and phase15_lib.forward_label.
All m_s at the converged fix_n_vessel_modes=80. BLAS threads pinned to 1 in the environment.
"""
import numpy as np
import torch

import phase2_data as D
import phase2_dim_lib as DL

KAPPA_I = D.SHAPE_FEATURES.index("kappa")
MAX_STEP = 0.45                       # local bounded step (PC-score units), same as Phase 2/2.5b
TARGETS = [0.3, 0.5, 0.7, 1.0]        # report solves-to-each; the PRIMARY headline target = 1.0
PRIMARY_TARGET = 1.0
# descriptors held IN RANGE (the "violate constraints" rejection). Bounds = dataset [p1,p99],
# loaded from data/phase3_desc_ranges.json by the worker and passed in (identical for all methods).
GUARD_FEATURES = ["kappa", "delta", "gap_inner", "gap_outer", "gap_min", "li"]


# ----------------------------------------------------------------- true objective
def true_eval(tok, u, ranges):
    """One expensive true 80-mode solve at controls u. Returns
        (value, info) where value = true m_s if the design is VALID + IN RANGE else 0.0 (penalty),
        info = dict(ms, ok, reason, desc) with the full true-solver descriptors (or None on failure).
    reason in {ok, invalid, out_of_range}. Identical for every method (fairness)."""
    import phase15_lib as L
    c = DL.ctrl_from_u(u)
    try:
        rec = L.forward_label(tok, c["active_currents"], c["paxis"], c["Ip"], c["fvac"],
                              c["alpha_m"], c["alpha_n"], fix_n_modes=80)
    except Exception:
        return 0.0, dict(ms=0.0, ok=False, reason="invalid", desc=None)
    ms = rec.get("m_s", float("nan"))
    if not np.isfinite(ms) or ms <= 0:
        return 0.0, dict(ms=0.0, ok=False, reason="invalid", desc=None)
    desc = {f: float(rec[f]) for f in GUARD_FEATURES if f in rec}
    desc["delta"] = float(rec.get("delta", np.nan))
    for f in ("sq_uo", "betap", "Rgeo", "a"):
        if f in rec:
            desc[f] = float(rec[f])
    for f in GUARD_FEATURES:
        lo, hi = ranges[f]
        if not (lo <= rec.get(f, lo - 1) <= hi):
            return 0.0, dict(ms=float(ms), ok=False, reason="out_of_range", desc=desc)
    return float(ms), dict(ms=float(ms), ok=True, reason="ok", desc=desc)


class Recorder:
    """Accumulates the trajectory + solves-to-target + reject reasons for one optimization run."""
    def __init__(self, m_s_start, kappa_start, regime):
        self.m_s_start = m_s_start
        self.kappa_start = kappa_start
        self.regime = regime
        self.n = 0
        self.best = 0.0
        self.best_desc = None
        self.best_x = None
        self.traj = []                 # [n, value, best]
        self.accepted = []             # accepted steps: dict(n, ms, x, descriptors) -> gallery
        self.reject = dict(invalid=0, out_of_range=0, no_improve=0)
        self.s2t = {f"{t:.1f}": None for t in TARGETS}

    def log(self, value, info, x, accepted):
        self.n += 1
        if accepted:
            self.best = value
            self.best_desc = info.get("desc")
            self.best_x = None if x is None else [float(v) for v in x]
            rec = dict(n=self.n, ms=float(value), x=self.best_x)
            if info.get("desc"):
                rec.update(info["desc"])
            self.accepted.append(rec)
        else:
            r = info.get("reason", "no_improve")
            self.reject[r if r in self.reject else "no_improve"] += 1
        for t in TARGETS:
            k = f"{t:.1f}"
            if self.s2t[k] is None and self.best >= t:
                self.s2t[k] = self.n
        self.traj.append([self.n, float(value), float(self.best)])

    def result(self):
        return dict(m_s_start=self.m_s_start, kappa_start=self.kappa_start, regime=self.regime,
                    n_solves=self.n, best_ms=float(self.best), best_desc=self.best_desc,
                    gain=float(self.best - self.m_s_start), reached_primary=bool(self.best >= PRIMARY_TARGET),
                    solves_to_target=self.s2t, reject=self.reject,
                    traj=self.traj, accepted=self.accepted)


# ----------------------------------------------------------------- start evaluation
def _eval_start(tok, ds, ranges, rc):
    """Evaluate the start shape (counts as the first true solve). The start is a real dataset row,
    so it is valid+in-range by construction; record its descriptors as the first accepted point."""
    val, info = true_eval(tok, ds.u_of_x(ds.x0), ranges)
    rc.n += 1
    rc.best = max(rc.best, val if info["ok"] else 0.0)
    if info["ok"]:
        rc.best_desc = info["desc"]
        rc.best_x = [float(v) for v in ds.x0]
        rec = dict(n=rc.n, ms=float(val), x=rc.best_x)
        rec.update(info["desc"])
        rc.accepted.append(rec)
    for t in TARGETS:
        k = f"{t:.1f}"
        if rc.s2t[k] is None and rc.best >= t:
            rc.s2t[k] = rc.n
    rc.traj.append([rc.n, float(val), float(rc.best)])
    return val


# ----------------------------------------------------------------- gradient methods
def run_gradient(tok, models, smap, ds, budget, ranges, m_s_start, kappa_start, regime,
                 kind="surrogate"):
    """surrogate autodiff ascent OR reduce-kappa heuristic, as LOCAL bounded-step hill-climbing.
    Surrogate line-search is FREE; one true solve confirms the best candidate per iteration."""
    rc = Recorder(m_s_start, kappa_start, regime)
    _eval_start(tok, ds, ranges, rc)
    best_x = ds.x0.copy()
    step0 = MAX_STEP
    while rc.n < budget and rc.best < PRIMARY_TARGET:
        g = DL._grad_x(models, smap, ds, best_x, kappa=(kind == "heuristic"))
        direction = -g if kind == "heuristic" else g       # heuristic descends kappa
        nrm = np.linalg.norm(direction)
        if nrm < 1e-9:
            break
        direction = direction / nrm
        # FREE surrogate (or geometry) line-search over a few magnitudes; confirm the best by 1 solve.
        cand, cand_score = None, -1e18
        for mag in (step0, step0 / 2, step0 / 4):
            xc = ds.clip(best_x + mag * direction)
            sc = (DL._surr_ms(models, smap, ds, xc) if kind != "heuristic"
                  else -DL.ds_kappa(smap, ds, xc))
            if sc > cand_score:
                cand_score, cand = sc, xc
        val, info = true_eval(tok, ds.u_of_x(cand), ranges)
        accept = info["ok"] and val > rc.best
        rc.log(val, info, cand if accept else None, accept)
        if accept:
            best_x = cand
            step0 = MAX_STEP
        else:
            step0 *= 0.5
            if step0 < 0.03:
                break
    return rc.result()


# ----------------------------------------------------------------- gradient-free baselines
def run_random(tok, ds, budget, ranges, m_s_start, kappa_start, regime, seed):
    """LOCAL random hill-climbing on the TRUE solver (accept iff true m_s improves + valid+in-range)."""
    rc = Recorder(m_s_start, kappa_start, regime)
    _eval_start(tok, ds, ranges, rc)
    rng = np.random.default_rng(seed)
    best_x = ds.x0.copy()
    while rc.n < budget and rc.best < PRIMARY_TARGET:
        r = rng.standard_normal(ds.d); r /= (np.linalg.norm(r) + 1e-12)
        xc = ds.clip(best_x + MAX_STEP * r)
        val, info = true_eval(tok, ds.u_of_x(xc), ranges)
        accept = info["ok"] and val > rc.best
        rc.log(val, info, xc if accept else None, accept)
        if accept:
            best_x = xc
    return rc.result()


def run_cma(tok, ds, budget, ranges, m_s_start, kappa_start, regime, seed):
    """LOCAL CMA-ES on the TRUE solver (sigma0=MAX_STEP, box-bounded)."""
    import cma
    rc = Recorder(m_s_start, kappa_start, regime)
    _eval_start(tok, ds, ranges, rc)
    es = cma.CMAEvolutionStrategy(list(ds.x0), MAX_STEP, {
        "bounds": [list(ds.box_lo), list(ds.box_hi)], "seed": int(seed) % (2 ** 31),
        "verbose": -9, "maxfevals": budget - 1})
    while rc.n < budget and rc.best < PRIMARY_TARGET and not es.stop():
        sols = es.ask()
        fit = []
        for s in sols:
            if rc.n >= budget:
                fit.append(1e3); continue
            xc = ds.clip(s)
            val, info = true_eval(tok, ds.u_of_x(xc), ranges)
            accept = info["ok"] and val > rc.best
            rc.log(val, info, xc if accept else None, accept)
            fit.append(-val if info["ok"] else 1e3)       # CMA minimizes; penalty for invalid/out-of-range
        es.tell(sols, fit)
    return rc.result()


def _range_violation(desc, ranges):
    """Normalized total amount by which the guarded descriptors fall outside their range (>=0)."""
    v = 0.0
    if not desc:
        return 1.0
    for f, (lo, hi) in ranges.items():
        x = desc.get(f)
        if x is None:
            continue
        w = (hi - lo) + 1e-9
        if x < lo:
            v += (lo - x) / w
        elif x > hi:
            v += (x - hi) / w
    return float(v)


def run_cma_fair(tok, ds, budget, ranges, m_s_start, kappa_start, regime, seed,
                 popsize=6, graded=True):
    """FAIR-CMA control (adversarial-review fix `cma-starved`): a SMALLER popsize so CMA gets many
    more generations within the budget (default popsize=4+3ln(d)=11 at d=12 gives only ~2.5
    generations at budget 30; popsize=6 gives ~5 at b=30 / ~10 at b=60 -- well into CMA's
    exploitation phase), plus a GRADED out-of-range penalty (fix `cma-flat-penalty`: -m_s + range
    violation distance, so the covariance update gets climb-back signal instead of a flat 1e3).
    Everything else identical to run_cma (sigma0=MAX_STEP, same box, stop-at-target)."""
    import cma
    rc = Recorder(m_s_start, kappa_start, regime)
    _eval_start(tok, ds, ranges, rc)
    es = cma.CMAEvolutionStrategy(list(ds.x0), MAX_STEP, {
        "bounds": [list(ds.box_lo), list(ds.box_hi)], "seed": int(seed) % (2 ** 31),
        "popsize": int(popsize), "verbose": -9, "maxfevals": budget - 1})
    while rc.n < budget and rc.best < PRIMARY_TARGET and not es.stop():
        sols = es.ask()
        fit = []
        for s in sols:
            if rc.n >= budget:
                fit.append(1e3); continue
            xc = ds.clip(s)
            val, info = true_eval(tok, ds.u_of_x(xc), ranges)
            accept = info["ok"] and val > rc.best
            rc.log(val, info, xc if accept else None, accept)
            if info["ok"]:
                fit.append(-val)
            elif graded and info.get("reason") == "out_of_range":
                fit.append(-info["ms"] + 5.0 * _range_violation(info.get("desc"), ranges))
            else:
                fit.append(1e3)
        es.tell(sols, fit)
    r = rc.result()
    r["popsize"] = int(popsize)
    return r


def run_nelder(tok, ds, budget, ranges, m_s_start, kappa_start, regime, seed):
    """Nelder-Mead simplex on the TRUE solver (classic derivative-free local optimizer).
    Initial simplex scaled to MAX_STEP so it explores at the same LOCAL scale as the others; x is
    clipped into the in-distribution box before each true solve; capped at `budget` true solves."""
    from scipy.optimize import minimize
    rc = Recorder(m_s_start, kappa_start, regime)
    _eval_start(tok, ds, ranges, rc)
    d = ds.d

    def neg_obj(x):
        if rc.n >= budget or rc.best >= PRIMARY_TARGET:
            return 1e3                                     # stop spending solves
        xc = ds.clip(x)
        val, info = true_eval(tok, ds.u_of_x(xc), ranges)
        accept = info["ok"] and val > rc.best
        rc.log(val, info, xc if accept else None, accept)
        return -val if info["ok"] else 1e3

    x0 = ds.x0.copy()
    simplex = np.vstack([x0] + [x0 + MAX_STEP * np.eye(d)[i] for i in range(d)])
    simplex = np.clip(simplex, ds.box_lo, ds.box_hi)
    try:
        minimize(neg_obj, x0, method="Nelder-Mead",
                 options=dict(initial_simplex=simplex, maxfev=budget - 1, maxiter=budget - 1,
                              xatol=1e-4, fatol=1e-4, adaptive=True))
    except Exception:
        pass
    return rc.result()


def run_one(tok, models, smap, ds, budget, ranges, start, method, seed):
    """Dispatch one (start x method) design run. `start` = dict(m_s_start, kappa_start, regime)."""
    a = (ds, budget, ranges, start["m_s_start"], start["kappa_start"], start["regime"])
    if method == "surrogate":
        return run_gradient(tok, models, smap, ds, budget, ranges, start["m_s_start"],
                            start["kappa_start"], start["regime"], kind="surrogate")
    if method == "heuristic":
        return run_gradient(tok, models, smap, ds, budget, ranges, start["m_s_start"],
                            start["kappa_start"], start["regime"], kind="heuristic")
    if method == "random":
        return run_random(tok, *a, seed=seed)
    if method == "cma":
        return run_cma(tok, *a, seed=seed)
    if method == "nelder":
        return run_nelder(tok, *a, seed=seed)
    raise ValueError(f"unknown method {method}")
