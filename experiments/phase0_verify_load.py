"""
phase0_verify_load.py -- Phase 0, Part 3 check: a LOADED machine reproduces m_s.

Loads the serialized tokamak, solves the canonical shape at the locked protocol, and prints
m_s to 12 digits + the load time. Run this TWICE as separate processes and compare:
  - the two processes must agree bit-for-bit (cross-process determinism of the loaded machine);
  - the value must match the freshly-BUILT canonical m_s (0.4199404... from phase0_ms_crosscheck)
    to confirm serialization did not perturb the machine.

Run thread-pinned: OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python ...
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import phase0_lib as P0

MACHINE = os.path.join(P0.MC, "serialized_tokamak.pkl")


def main():
    t0 = time.time()
    tok = P0.load_machine(MACHINE)
    t_load = time.time() - t0
    res = P0.solve_equilibrium(tok, **P0.CANON)
    print(f"LOAD={t_load:.2f}s  M_S={res['m_s']:.12f}  GAMMA={res['gamma']:.8f}  "
          f"KAPPA={res['kappa']:.8f}  modes={res['n_independent_vars']}")


if __name__ == "__main__":
    main()
