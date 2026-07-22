"""
phase0_serialize_machine.py -- Phase 0, Part 3: build the MAST-U machine ONCE and pickle it.

Every downstream script then LOADS this identical machine (phase0_lib.load_machine), so:
  - no rebuild cost per run,
  - no module-level LatinHypercube(seed=42) state drift (audit 4.3: building twice in one
    process diverges ~20%),
  - every label in the study uses a bit-identical machine.

Run thread-pinned (OMP=1). Writes machine_configs/MAST-U/serialized_tokamak.pkl.
"""
import os
import pickle
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import phase0_lib as P0

OUT = os.path.join(P0.MC, "serialized_tokamak.pkl")


def main():
    t0 = time.time()
    tok = P0.build_tokamak()
    t_build = time.time() - t0

    t0 = time.time()
    with open(OUT, "wb") as f:
        pickle.dump(tok, f, protocol=pickle.HIGHEST_PROTOCOL)
    t_pickle = time.time() - t0
    size_mb = os.path.getsize(OUT) / 1e6

    # round-trip load check (same process is fine for LOAD; load uses no LHS state)
    t0 = time.time()
    with open(OUT, "rb") as f:
        tok2 = pickle.load(f)
    t_load = time.time() - t0

    print(f"build   : {t_build:6.1f} s")
    print(f"pickle  : {t_pickle:6.2f} s  ({size_mb:.1f} MB) -> {OUT}")
    print(f"load    : {t_load:6.2f} s")
    print(f"n_active={tok2.n_active_coils} n_passive={tok2.n_passive_coils}")
    print("OK: machine serialized.")


if __name__ == "__main__":
    main()
