"""
phase0_solve_one.py -- one ISOLATED cold solve for the Phase-0 noise sweep.

Loads the serialized machine and does exactly ONE cold inverse-solve+linearisation, then
prints a single 'RESULT <json>' line. Total isolation per process avoids the warm-start /
inverse-solve path dependence (audit 4.2; reproduced in Phase 0: the canonical m_s shifts
~2% between solve-position 1 and 3 in a shared-tokamak sequence).

BLAS thread count is controlled by the PARENT via env (OMP/OPENBLAS/MKL_NUM_THREADS), set
before this process imports numpy. --threads is passed only for labelling/verification.
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import phase0_lib as P0  # imports numpy; env already set by parent

MACHINE = os.path.join(P0.MC, "serialized_tokamak.pkl")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zscale", type=float, required=True)
    ap.add_argument("--dR", type=float, default=0.0)
    ap.add_argument("--threads", type=int, required=True)
    ap.add_argument("--tol", type=float, default=P0.DEF_INV_TOL)
    ap.add_argument("--nx", type=int, default=P0.DEF_NX)
    ap.add_argument("--ny", type=int, default=P0.DEF_NY)
    ap.add_argument("--modes", type=int, default=P0.DEF_FIX_N_MODES)
    a = ap.parse_args()

    blas = {k: os.environ.get(k, "?") for k in
            ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]}
    t0 = time.time()
    tok = P0.load_machine(MACHINE)
    job = dict(zscale=a.zscale, dR=a.dR, threads=a.threads, tol=a.tol,
               nx=a.nx, ny=a.ny, modes=a.modes, blas=blas)
    try:
        res = P0.solve_equilibrium(tok, zscale=a.zscale, dR=a.dR, nx=a.nx, ny=a.ny,
                                   fix_n_modes=a.modes, inv_tol=a.tol)
        # canonical key aliases so driver/analyzer see consistent fields on ok and fail paths
        res.update(threads=a.threads, tol=res["inv_tol"], modes=res["fix_n_modes"],
                   blas=blas, t=time.time() - t0, ok=True)
        print("RESULT " + json.dumps(res))
    except Exception as e:
        import traceback
        traceback.print_exc()
        job.update(ok=False, error=f"{type(e).__name__}: {e}", t=time.time() - t0)
        print("RESULT " + json.dumps(job))


if __name__ == "__main__":
    main()
