"""
tier1_indist_worker.py -- IN-DISTRIBUTION CONTROL for Tier-1 test (b) (adversarial-review H2).

Question: is the ~0.61 surrogate-vs-solver residual on real shapes real SHAPE-OOD degradation, or a
ConstrainBetapIp re-solve PIPELINE artifact (poloidalBeta2 inflation feeding the surrogate OOD betap)?

Method: take synthetic IN-DISTRIBUTION shapes (rows of dataset_v1_80q), FORWARD-solve each to its true
LCFS, then run the IDENTICAL shape-anchored inverse re-solve pipeline (tier1_resolve_worker.resolve_slice)
on that LCFS, feeding ConstrainBetapIp the shape's own (descriptors-def) betap. If the re-solve betap
inflates OOD and the residual is ~0.6 even here, the real 0.61 is a pipeline artifact; if the re-solve
betap stays in-range and the residual is small (~native surrogate accuracy), the real 0.61 is genuine OOD.

  python tier1_indist_worker.py --chunk i --nchunks N
"""
import os, sys, json, argparse, math
os.environ["PYTHONIOENCODING"] = "utf-8"
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tier1_lib as T
import phase0_lib as P0
import phase15_lib as P15
import tier1_resolve_worker as W

ROOT = T.ROOT
SER = os.path.join(ROOT, "machine_configs", "MAST-U", "serialized_tokamak.pkl")
OUTDIR = os.path.join(ROOT, "data", "tier1_indist")
GRID = dict(Rmin=0.1, Rmax=2.0, Zmin=-2.2, Zmax=2.2)
N_TOTAL = 24  # in-distribution shapes to test


def pick_rows():
    """Deterministic in-distribution sample: mid-kappa (like the real b-set), m_s defined, spanning kappa."""
    df = pd.read_parquet(os.path.join(ROOT, "data", "dataset_v1_80q.parquet")).reset_index(drop=True)
    df = df[(df["m_s"] > 0) & np.isfinite(df["m_s"])].copy()
    # mirror the real b-set band kappa~1.5-2.1; sample evenly by kappa quantile
    df = df[(df["kappa"] >= 1.58) & (df["kappa"] <= 2.10)].sort_values("kappa").reset_index(drop=True)
    idx = np.linspace(0, len(df) - 1, N_TOTAL).astype(int)
    return df.iloc[idx].reset_index(drop=True)


def forward_eq(tok, row):
    """Forward-solve one in-distribution control row -> (eq, profiles) or None."""
    from freegsnke import equilibrium_update, GSstaticsolver
    from freegsnke.jtor_update import ConstrainPaxisIp
    cur = [5000.0] + [float(row[f"I_{c}"]) for c in P15.ACTIVE_COILS if c != "Solenoid"]
    eq = equilibrium_update.Equilibrium(tokamak=tok, nx=65, ny=65, **GRID)
    profiles = ConstrainPaxisIp(eq=eq, paxis=float(row["paxis"]), Ip=float(row["Ip_target"]),
                                fvac=float(row["fvac"]), alpha_m=float(row["alpha_m"]), alpha_n=float(row["alpha_n"]))
    solver = GSstaticsolver.NKGSsolver(eq=eq)
    eq.tokamak.set_all_coil_currents(np.asarray(cur, float))
    for step in (2.5, 1.5, 1.0):
        try:
            eq.plasma_psi = eq.create_psi_plasma_default(adaptive_centre=True)
            with np.errstate(divide="raise", invalid="raise", over="raise"):
                solver.forward_solve(eq=eq, profiles=profiles, target_relative_tolerance=1e-8,
                                     max_solving_iterations=120, step_size=step, verbose=False, suppress=True)
            if solver.relative_change <= 1e-7 and np.all(np.isfinite(eq.plasma_psi)):
                if not bool(getattr(eq, "flag_limiter", True)):
                    return eq
        except Exception:
            continue
    return None


def make_fake_shot(eq, tok, row):
    """Build a tier1_lib-format single-slice shot_data dict from a synthetic forward eq."""
    from freegs4e import critical
    sep = np.asarray(eq.separatrix()); lr, lz = sep[:, 0], sep[:, 1]
    de = P15.descriptors(eq, tok)
    # find X-point(s) of the total flux
    R = eq.R; Z = eq.Z; psi = eq.psi()
    try:
        op, xp = critical.find_critical(R, Z, psi, None, 1 if eq.plasmaCurrent() > 0 else -1)
    except Exception:
        xp = []
    x1r = float(xp[0][0]) if len(xp) else float("nan"); x1z = float(xp[0][1]) if len(xp) else float("nan")
    d = {k: np.array([v]) for k, v in dict(
        lcfs_r=None, lcfs_z=None).items()}
    d["lcfs_r"] = np.array([lr]); d["lcfs_z"] = np.array([lz])  # object-ish; resolve reads d["lcfs_r"][ti]
    d["shot"] = np.array([90000])
    d["plasma_current_x"] = np.array([eq.plasmaCurrent()])
    d["bvac_r"] = np.array([1.0]); d["bvac_val"] = np.array([float(row["fvac"])])
    d["elongation"] = np.array([de["kappa"]]); d["li"] = np.array([de["li"]]); d["betap"] = np.array([de["betap"]])
    d["q_95"] = np.array([float(row.get("q95", np.nan))]); d["final_chisq"] = np.array([0.0]); d["time"] = np.array([0.0])
    d["magnetic_axis_r"] = np.array([de["Rmag"]]); d["magnetic_axis_z"] = np.array([de["Zaxis"]])
    d["geom_axis_rc"] = np.array([de["Rgeo"]]); d["geom_axis_zc"] = np.array([0.0]); d["minor_radius"] = np.array([de["a"]])
    d["xpoint1_rc"] = np.array([x1r]); d["xpoint1_zc"] = np.array([x1z])
    d["xpoint2_rc"] = np.array([np.nan]); d["xpoint2_zc"] = np.array([np.nan])
    return d, de


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk", type=int, required=True)
    ap.add_argument("--nchunks", type=int, required=True)
    args = ap.parse_args()
    os.makedirs(OUTDIR, exist_ok=True)
    T.patch_freegs4e_profile_bug()
    tok = P0.load_machine(SER)
    from phase2_model import load_ensemble, ensemble_predict
    from phase2_data import SHAPE_FEATURES
    models, _ = load_ensemble("surrogate")
    rows = pick_rows()
    mine = [i for i in range(len(rows)) if i % args.nchunks == args.chunk]
    print(f"[chunk {args.chunk}] {len(mine)} in-dist shapes", flush=True)
    for i in mine:
        outf = os.path.join(OUTDIR, f"{i:03d}.json")
        if os.path.exists(outf):
            continue
        row = rows.iloc[i]
        rec = dict(idx=int(i), kappa_true=float(row["kappa"]), betap_true=float(row["betap"]),
                   li_true=float(row["li"]), ms_true=float(row["m_s"]))
        try:
            eq = forward_eq(tok, row)
            if eq is None:
                rec["status"] = "forward_failed"
            else:
                fake_d, de_native = make_fake_shot(eq, tok, row)
                # native surrogate prediction vs dataset m_s (in-dist accuracy baseline)
                xn = np.array([[de_native[f] for f in SHAPE_FEATURES]], float)
                surr_native = float(np.exp(ensemble_predict(models, xn)["mean"][0, 0]))
                rec["surr_native"] = surr_native
                rec["native_resid"] = abs(math.log(surr_native) - math.log(rec["ms_true"]))
                # IDENTICAL re-solve pipeline
                rr = W.resolve_slice(tok, models, fake_d, 0)
                for k in ("status", "converged", "limiter", "re_kappa", "re_betap", "re_li",
                          "surr_ms", "surr_epi_std", "ms_solver_40", "ms_solver_80", "zeta_cm"):
                    if k in rr:
                        rec[k] = rr[k]
                if rr.get("surr_ms", 0) > 0 and rr.get("ms_solver_80", 0) > 0:
                    rec["resolve_resid"] = abs(math.log(rr["surr_ms"]) - math.log(rr["ms_solver_80"]))
        except Exception as e:
            rec["status"] = "error"; rec["error"] = f"{type(e).__name__}:{e}"
        tmp = outf + ".tmp"
        with open(tmp, "w") as f:
            json.dump(rec, f)
        os.replace(tmp, outf)
        print(f"  {i}: {rec.get('status')} re_betap={rec.get('re_betap','?')} "
              f"native_resid={rec.get('native_resid','?')} resolve_resid={rec.get('resolve_resid','?')}", flush=True)
    print(f"[chunk {args.chunk}] done", flush=True)


if __name__ == "__main__":
    main()
