"""
tier1_resolve_worker.py -- one chunk of the Tier-1 shape-anchored re-solve (resume-safe).

For each assigned real MAST EFIT slice: inverse-solve its stored LCFS on the fixed MAST-U model using
the TRAINING profile family (ConstrainPaxisIp: real Ip + real fvac, nominal alpha) so SHAPE is the only
OOD axis. Then compute descriptors() -> the surrogate's predicted m_s AND FreeGSNKE's own m_s at BOTH 40
and 80 vessel modes (the intrinsic-ambiguity anchor), boundary agreement zeta, achieved-kappa error, and
the re-solved-vs-EFIT li/betap gap. One JSON per slice in data/tier1_resolved/ (resume-safe).

BLAS pinned to 1 (set here BEFORE importing numpy). Launched by tier1_resolve_run.py.
  python tier1_resolve_worker.py --chunk i --nchunks N [--source selection|pool] [--max_it 60]
"""
import os, sys, json, time, argparse
os.environ["PYTHONIOENCODING"] = "utf-8"
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tier1_lib as T
import phase0_lib as P0
import phase15_lib as P15
from phase2_data import SHAPE_FEATURES

ROOT = T.ROOT
SER = os.path.join(ROOT, "machine_configs", "MAST-U", "serialized_tokamak.pkl")
OUTDIR = os.path.join(ROOT, "data", "tier1_resolved")
GRID = dict(Rmin=0.1, Rmax=2.0, Zmin=-2.2, Zmax=2.2)
COIL_LIMS = [[5e3, 9e3, 9e3, 7e3, 7e3, 5e3, 4e3, 5e3, 0.0, 0.0, None],
             [-5e3, -9e3, -9e3, -7e3, -7e3, -5e3, -4e3, -5e3, -10e3, -10e3, None]]


def resolve_slice(tok, models, d, ti, max_it=60, tol=1e-4, nsub=40):
    from freegsnke import equilibrium_update, GSstaticsolver, nonlinear_solve
    from freegsnke.inverse import Inverse_optimizer
    from freegsnke.jtor_update import ConstrainPaxisIp, ConstrainBetapIp
    from phase2_model import ensemble_predict

    lr, lz = T.lcfs_slice(d, ti)
    g = lambda n: float(d[n][ti]) if n in d else float("nan")
    Ip = abs(g("plasma_current_x"))
    fvac = abs(g("bvac_r") * g("bvac_val"))
    rec = dict(shot=int(d["shot"][0]), slice=int(ti), t=g("time"), Ip=Ip, fvac=fvac,
               real_kappa=g("elongation"), real_li=g("li"), real_betap=g("betap"),
               real_q95=g("q_95"), chisq=g("final_chisq"), n_lcfs=int(len(lr)))

    eq = equilibrium_update.Equilibrium(tokamak=tok, nx=65, ny=65, **GRID)
    # Profile family = ConstrainBetapIp constrained to the REAL EFIT beta_p + real Ip + real fvac.
    # Rationale (empirically established): fixed-paxis nominal profiles do NOT converge to the real
    # shape; real-array (GeneralPprimeFFprime) profiles converge but are not supported by nl_solver
    # (no n_profiles_parameters); ConstrainBetapIp converges AND is nl_solver-native AND the m_s
    # eigenvalue is metal-coupling-dominated (independent of the profile-parameter Jacobian).
    real_betap = abs(g("betap"))
    try:
        profiles = ConstrainBetapIp(eq=eq, betap=real_betap, Ip=Ip, fvac=fvac, alpha_m=1.8, alpha_n=1.2)
        rec["profile"] = "betapip_real"; rec["target_betap"] = real_betap
    except Exception as e:
        profiles = ConstrainPaxisIp(eq=eq, paxis=8e3, Ip=Ip, fvac=fvac, alpha_m=1.8, alpha_n=1.2)
        rec["profile"] = "paxis_fallback"; rec["profile_err"] = f"{type(e).__name__}:{e}"
    solver = GSstaticsolver.NKGSsolver(eq=eq)
    eq.tokamak.set_coil_current("Solenoid", 5000)
    eq.tokamak["Solenoid"].control = False
    idx = np.linspace(0, len(lr) - 1, min(nsub, len(lr))).astype(int)
    isoflux_set = np.array([[list(lr[idx]), list(lz[idx])]])
    nx, nz = [], []
    for a_, b_ in ((g("xpoint1_rc"), g("xpoint1_zc")), (g("xpoint2_rc"), g("xpoint2_zc"))):
        if np.isfinite(a_) and 0.1 < a_ < 2.0 and abs(b_) < 2.2:
            nx.append(a_); nz.append(b_)
    null_points = [nx, nz] if nx else None
    constrain = Inverse_optimizer(null_points=null_points, isoflux_set=isoflux_set, coil_current_limits=COIL_LIMS)
    constrain.mu_coils = 1e5

    t0 = time.time()
    try:
        with np.errstate(divide="raise", invalid="raise", over="raise"):
            solver.solve(eq=eq, profiles=profiles, constrain=constrain, target_relative_tolerance=tol,
                         target_relative_psit_update=1e-3, verbose=False,
                         l2_reg=np.array([1e-12] * 10 + [1e-6]), max_solving_iterations=max_it)
    except Exception as e:
        rec.update(status="solve_error", error=f"{type(e).__name__}:{e}", dt=time.time() - t0)
        return rec
    rec["rel"] = float(solver.relative_change)
    rec["converged"] = bool(solver.relative_change <= 10 * tol)
    rec["limiter"] = bool(getattr(eq, "flag_limiter", True))
    try:
        rec["intersects_wall"] = float(eq.intersectsWall())
    except Exception:
        rec["intersects_wall"] = float("nan")
    rec["dt_solve"] = round(time.time() - t0, 1)
    if not rec["converged"]:
        rec["status"] = "unconverged"
        return rec
    # NOTE: keep going even if limited -- record the diverted/limited flag; the analysis decides
    # which slices count (real MAST plasmas are diverted; a limited re-solve is a fidelity failure).
    try:
        de = P15.descriptors(eq, tok)
    except Exception as e:
        rec.update(status="descriptor_error", error=str(e)); return rec
    rec["re_kappa"] = de["kappa"]; rec["re_li"] = de["li"]; rec["re_betap"] = de["betap"]
    rec["dkappa"] = de["kappa"] - rec["real_kappa"]
    rec["dli"] = de["li"] - rec["real_li"]; rec["dbetap"] = de["betap"] - rec["real_betap"]
    # boundary agreement zeta (cm): max nearest-point distance re-solved separatrix -> real LCFS
    try:
        sep = np.asarray(eq.separatrix()); sR, sZ = sep[:, 0], sep[:, 1]
        rec["zeta_cm"] = float(np.max([np.min(np.hypot(lr - r0, lz - z0)) for r0, z0 in zip(sR, sZ)])) * 100.0
    except Exception:
        pass
    # surrogate m_s (ensemble) on the re-solved descriptor vector
    try:
        x = np.array([[de[f] for f in SHAPE_FEATURES]], float)
        pred = ensemble_predict(models, x)
        rec["surr_logms"] = float(pred["mean"][0, 0])
        rec["surr_ms"] = float(np.exp(pred["mean"][0, 0]))
        rec["surr_epi_std"] = float(pred["epi_std"][0, 0])
        rec["surr_tot_std"] = float(pred["tot_std"][0, 0])
    except Exception as e:
        rec["surr_error"] = str(e)
    # FreeGSNKE m_s at 40 and 80 modes
    for fm in (40, 80):
        t1 = time.time()
        try:
            with np.errstate(divide="raise", invalid="raise", over="raise"):
                nls = nonlinear_solve.nl_solver(eq=eq, profiles=profiles, GSStaticSolver=solver,
                        plasma_resistivity=1e-6, fix_n_vessel_modes=fm, verbose=False)
            sm = np.ravel(np.asarray(nls.linearised_sol.stability_margin).real)
            rec[f"ms_solver_{fm}"] = float(sm.max()) if sm.size else float("nan")
            rec[f"npos_{fm}"] = int(sm.size)
        except Exception as e:
            rec[f"ms_solver_{fm}_err"] = str(e)
        rec[f"dt_lin{fm}"] = round(time.time() - t1, 1)
    rec["status"] = "ok"
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk", type=int, required=True)
    ap.add_argument("--nchunks", type=int, required=True)
    ap.add_argument("--source", default="selection", choices=["selection", "pool"])
    ap.add_argument("--max_it", type=int, default=60)
    args = ap.parse_args()
    os.makedirs(OUTDIR, exist_ok=True)

    with open(os.path.join(ROOT, "data", f"tier1_{args.source}.json")) as f:
        blob = json.load(f)
    items = blob["selection"] if args.source == "selection" else blob["rows"]
    mine = [it for i, it in enumerate(items) if i % args.nchunks == args.chunk]
    print(f"[chunk {args.chunk}/{args.nchunks}] {len(mine)} slices", flush=True)

    T.patch_freegs4e_profile_bug()   # enable real-EFIT-profile equilibria (fixes upstream typo)
    tok = P0.load_machine(SER)
    from phase2_model import load_ensemble
    models, _ = load_ensemble("surrogate")

    cache = {}
    for it in mine:
        shot, ti = int(it["shot"]), int(it["slice"])
        outf = os.path.join(OUTDIR, f"{shot}_{ti}.json")
        if os.path.exists(outf):
            continue
        try:
            if shot not in cache:
                cache[shot] = T.read_shot(shot)
            rec = resolve_slice(tok, models, cache[shot], ti, max_it=args.max_it)
        except Exception as e:
            rec = dict(shot=shot, slice=ti, status="worker_error", error=f"{type(e).__name__}:{e}")
        tmp = outf + ".tmp"
        with open(tmp, "w") as f:
            json.dump(rec, f)
        os.replace(tmp, outf)  # atomic
        print(f"  {shot}:{ti} status={rec.get('status')} re_k={rec.get('re_kappa','?')} "
              f"surr_ms={rec.get('surr_ms','?')} ms80={rec.get('ms_solver_80','?')}", flush=True)
    print(f"[chunk {args.chunk}] done", flush=True)


if __name__ == "__main__":
    main()
