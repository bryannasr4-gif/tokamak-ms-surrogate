"""
phase4_gallery_lib.py -- the kappa-CONSTRAINED design loop with FULL recording (the Phase-4
learned-m_s differentiator + the design gallery).

Phase 3 showed the gradient design loop's solve-efficiency win over gradient-free is delivered by
the GRADIENT/kappa-direction -- NOT specifically the learned m_s (the reduce-kappa heuristic ties
the surrogate on the kappa-dominated single-machine manifold). Phase 4's lead question: make the
LEARNED m_s load-bearing. We do that by REMOVING kappa as a lever -- hold kappa fixed (+-KTOL on the
TRUE-solver kappa, projecting every step orthogonal to grad_x kappa) -- so the reduce-kappa rule is
DISABLED and the only way to raise m_s is via the SECONDARY shape levers (squareness, gaps, l_i,
delta, betap). At fixed kappa the surrogate's knowledge of how those levers map to m_s is the only
thing that can work.

This module reuses the Phase-2.5b kappa-constrained machinery (phase25_kappa_lib) but adds a
Recorder (modeled on phase3_lib) that captures, per accepted step, the FULL true-solver descriptors
+ the controls u, so we can (a) draw before->after solver-confirmed LCFS shapes at the SAME kappa,
and (b) show which secondary levers the learned m_s moved. Every step is confirmed by a fresh true
80-mode FreeGSNKE solve at the locked protocol; a step is accepted only if true m_s improves AND
|kappa - kappa_start| <= KTOL.

Methods:
  surrogate     : autodiff grad of (log m_s), projected off grad kappa. The learned-m_s lever.
  kappa_nudge   : deliberately REDUCE kappa (step along -grad kappa, UNprojected), accepting only
                  steps that keep |kappa - kappa_start| <= KTOL. This is NOT a disabled lever -- it
                  QUANTIFIES the confound that, because m_s is so kappa-sensitive, even a +-KTOL
                  kappa reduction buys some m_s. The surrogate's gain MINUS this isolates the genuine
                  SECONDARY-lever (learned-m_s) contribution; and the published gradient-free / fixed
                  -lever baselines are kappa-penalized identically, so they also have this freedom.
  cma           : gradient-free CMA-ES on the true solver, kappa-penalized (the fair baseline).
"""
import numpy as np

import phase2_data as D
import phase2_dim_lib as DL
import phase25_kappa_lib as KL

KTOL = KL.KTOL          # 0.04, identical to Phase-2.5b
MAX_STEP = KL.MAX_STEP  # 0.45
KAPPA_I = D.SHAPE_FEATURES.index("kappa")
# descriptors captured at every accepted step (for the gallery + lever attribution)
RICH = ["kappa", "delta", "sq_uo", "sq_ui", "sq_lo", "sq_li", "gap_inner", "gap_outer",
        "gap_min", "li", "betap", "a", "Rgeo"]


def true_full(tok, u):
    """One expensive true 80-mode solve at controls u. Returns (m_s, kappa, descriptors_dict, rec).
    (0.0, nan, None, None) on failure / non-diverted / non-finite."""
    import phase15_lib as L
    c = DL.ctrl_from_u(u)
    try:
        rec = L.forward_label(tok, c["active_currents"], c["paxis"], c["Ip"], c["fvac"],
                              c["alpha_m"], c["alpha_n"], fix_n_modes=80)
    except Exception:
        return 0.0, float("nan"), None, None
    ms = rec.get("m_s", float("nan"))
    if not np.isfinite(ms) or ms <= 0:
        return 0.0, float("nan"), None, None
    desc = {f: float(rec[f]) for f in RICH if f in rec}
    return float(ms), float(rec["kappa"]), desc, rec


class GRecorder:
    """Trajectory + accepted-step descriptors + best controls for one kappa-constrained run."""
    def __init__(self, m_s_start, kappa_start, regime, start_desc, start_u):
        self.m_s_start = m_s_start
        self.kappa_start = kappa_start
        self.regime = regime
        self.n = 1
        self.best = m_s_start
        self.best_u = [float(v) for v in start_u]
        self.best_desc = start_desc
        self.traj = [[1, float(m_s_start), float(kappa_start)]]   # (n_solves, m_s, kappa)
        self.accepted = [dict(n=1, ms=float(m_s_start), kappa=float(kappa_start),
                              desc=start_desc, u=[float(v) for v in start_u])]
        self.reject = dict(kappa_drift=0, no_improve=0, invalid=0)
        self.dir_cos = []      # S4a diagnostic only: cos(full, masked) per gradient evaluation
        self.pin_value = None  # S4a arm B: the pinned descriptor value (None on every other path)

    def log_accept(self, ms, kappa, desc, u):
        self.n += 1
        self.best = ms
        self.best_u = [float(v) for v in u]
        self.best_desc = desc
        self.traj.append([self.n, float(ms), float(kappa)])
        self.accepted.append(dict(n=self.n, ms=float(ms), kappa=float(kappa), desc=desc,
                                  u=[float(v) for v in u]))

    def log_reject(self, ms, kappa, reason):
        self.n += 1
        self.reject[reason] = self.reject.get(reason, 0) + 1
        self.traj.append([self.n, float(self.best), float(self.kappa_start)])

    def result(self):
        r = dict(m_s_start=float(self.m_s_start), kappa_start=float(self.kappa_start),
                 regime=self.regime, n_solves=self.n, best_ms=float(self.best),
                 gain=float(self.best - self.m_s_start), kappa_final=float(self.best_desc["kappa"]),
                 kappa_drift=float(abs(self.best_desc["kappa"] - self.kappa_start)),
                 best_u=self.best_u, best_desc=self.best_desc, traj=self.traj,
                 accepted=self.accepted, reject=self.reject)
        if self.dir_cos:                      # present only on the S4a ablated path
            r["dir_cos"] = self.dir_cos
        if self.pin_value is not None:        # present only on S4a arm B
            r["pin_value"] = self.pin_value
        return r


def run_surrogate(tok, models, smap, ds, budget, k_start, start_desc, kind="surrogate", seed=0,
                  zero_feat_i=None, mask_value=False):
    """kappa-constrained ascent with FULL recording.
    kind='surrogate'   -> learned-m_s gradient, projected orthogonal to grad kappa (secondary levers).
    kind='kappa_nudge' -> deliberately DESCEND kappa (UNprojected -grad kappa), accept only within
                          KTOL -> quantifies how much the residual +-KTOL kappa freedom alone buys.

    zero_feat_i (unit S4a, 2026-07-31): if not None, the named DESCRIPTOR channel of
    d(log m_s)/d(descriptors) is zeroed before back-propagation to x -- the l_i ablation. The
    DEFAULT (None) takes the byte-identical original code path, so every banked gallery result
    reproduces exactly (asserted by the S4a determinism control).

    mask_value (S4a ARM B, added by the council-before amendment A1): when True, the line-search
    ALSO scores candidates with that descriptor pinned to its start value, so the surrogate's
    value function cannot compensate for the masked gradient. Arm A (mask_value=False) tests the
    GRADIENT channel -- the manuscript's actual claim; Arm B (mask_value=True) is the 'pure'
    ablation three council seats independently demanded. Note the line search only ever selects a
    step MAGNITUDE along the already-fixed masked ray, so arm A's scoring cannot restore the
    unmasked DIRECTION; arm B bounds the residual concern."""
    ms0, k0, desc0, _ = true_full(tok, ds.u_of_x(ds.x0))
    rc = GRecorder(ms0, k0 if np.isfinite(k0) else k_start, "?", desc0 or start_desc, ds.x0)
    best_x = ds.x0.copy()
    step0 = MAX_STEP
    # S4a arm B: pin value = the ShapeMap-predicted descriptor at the START, frozen once.
    pin_val = (KL.ds_feature(smap, ds, ds.x0, zero_feat_i)
               if (mask_value and zero_feat_i is not None) else None)
    if pin_val is not None:
        rc.pin_value = float(pin_val)
    while rc.n < budget:
        gk = KL._grad_feature(None, smap, ds, best_x, KAPPA_I)
        if kind == "surrogate":
            if zero_feat_i is None:
                graw = KL._grad_feature(None, smap, ds, best_x, None, is_ms=True, models=models)
            else:
                gfull = KL._grad_feature(None, smap, ds, best_x, None, is_ms=True, models=models)
                graw = KL._grad_ms_zeroed(models, smap, ds, best_x, zero_feat_i)
                nf, nm = np.linalg.norm(gfull), np.linalg.norm(graw)
                rc.dir_cos.append([float(np.dot(gfull, graw) / (nf * nm + 1e-12)),
                                   float(nf), float(nm)])   # diagnostic ONLY; not used below
            direction = KL._project_off_kappa(graw, gk)   # secondary levers only
        else:  # kappa_nudge: descend kappa directly (the residual-kappa-freedom baseline)
            direction = -gk
        nrm = np.linalg.norm(direction)
        if nrm < 1e-9:
            break
        direction /= nrm
        # FREE line-search over magnitudes; surrogate scores by predicted m_s, kappa_nudge by
        # kappa reduction (push kappa down as far as KTOL will allow). Confirm the best by 1 solve.
        cand, sc_best = None, -1e18
        for mag in (step0, step0 / 2, step0 / 4):
            xc = ds.clip(best_x + mag * direction)
            if kind != "surrogate":
                sc = -KL.ds_feature(smap, ds, xc, KAPPA_I)
            elif pin_val is None:
                sc = KL._surr_ms(models, smap, ds, xc)
            else:
                sc = KL._surr_ms_pinned(models, smap, ds, xc, zero_feat_i, pin_val)
            if sc > sc_best:
                sc_best, cand = sc, xc
        ms, kappa, desc, _ = true_full(tok, ds.u_of_x(cand))
        if desc is None:
            rc.log_reject(ms, kappa, "invalid")
            step0 *= 0.5
        elif abs(kappa - rc.kappa_start) > KTOL:
            rc.log_reject(ms, kappa, "kappa_drift")
            step0 *= 0.5
        elif ms > rc.best:
            rc.log_accept(ms, kappa, desc, cand)
            best_x = cand
            step0 = MAX_STEP
        else:
            rc.log_reject(ms, kappa, "no_improve")
            step0 *= 0.5
        if step0 < 0.03:
            step0 = MAX_STEP / 2     # reset & keep spending budget (local max along this lever)
    return rc.result()


def run_cma(tok, ds, budget, k_start, start_desc, seed):
    """kappa-constrained gradient-free CMA-ES baseline with full recording (kappa-penalized)."""
    import cma
    ms0, k0, desc0, _ = true_full(tok, ds.u_of_x(ds.x0))
    rc = GRecorder(ms0, k0 if np.isfinite(k0) else k_start, "?", desc0 or start_desc, ds.x0)
    es = cma.CMAEvolutionStrategy(list(ds.x0), MAX_STEP, {
        "bounds": [list(ds.box_lo), list(ds.box_hi)], "seed": int(seed) % (2 ** 31),
        "verbose": -9, "maxfevals": budget - 1})
    while rc.n < budget and not es.stop():
        sols = es.ask()
        fit = []
        for s in sols:
            if rc.n >= budget:
                fit.append(1e3); continue
            xc = ds.clip(s)
            ms, kappa, desc, _ = true_full(tok, ds.u_of_x(xc))
            if desc is None:
                rc.log_reject(ms, kappa, "invalid"); fit.append(1e3); continue
            if abs(kappa - rc.kappa_start) > KTOL:
                rc.log_reject(ms, kappa, "kappa_drift"); fit.append(-ms + 10.0); continue
            if ms > rc.best:
                rc.log_accept(ms, kappa, desc, xc)
            else:
                rc.log_reject(ms, kappa, "no_improve")
            fit.append(-ms)
        es.tell(sols, fit)
    return rc.result()
