"""
phase25_build_m2.py -- build "Machine B" (a second conducting-structure config) for the
Phase-2.5 cross-machine generalization test.

Machine B = MAST-U-like with the PF-coil-CASE passive conductors removed (the `*_case_*`
elements, ~82 of 138), keeping the main vacuum vessel + centre column + colosseum/ring/baffle/psp
plates (56 structures). This is a physically-interpretable change to the passive stabilizing
structure that alters the inductance matrix -> a DIFFERENT m_s(shape) mapping. Because passive
structures sit at zero current in the static GS solve, Machine B gives the SAME equilibria/shapes
as Machine A for the same controls (verified here) but a different stability margin -- a clean
controlled test of conducting-structure dependence.

Builds + serializes Machine B, then verifies on one dataset_v1 shape: (1) shape ~identical to
Machine A, (2) m_s genuinely differs.
"""
import os
import pickle
import sys

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "experiments"))
MC = os.path.join(ROOT, "machine_configs", "MAST-U")
MCB = os.path.join(ROOT, "machine_configs", "MAST-U-B")
SER_B = os.path.join(MCB, "serialized_tokamak_B.pkl")


def make_modified_passive():
    os.makedirs(MCB, exist_ok=True)
    d = pickle.load(open(os.path.join(MC, "MAST-U_like_passive_coils.pickle"), "rb"))
    kept = [e for e in d if "_case_" not in e.get("name", "") and "case" not in e.get("element", "")]
    out = os.path.join(MCB, "MAST-U-B_passive_coils.pickle")
    with open(out, "wb") as f:
        pickle.dump(kept, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Machine B passive: kept {len(kept)}/{len(d)} structures (removed {len(d)-len(kept)} coil cases)")
    return out


def build_and_serialize():
    from freegsnke import build_machine
    passive_b = make_modified_passive()
    tokB = build_machine.tokamak(
        active_coils_path=f"{MC}/MAST-U_like_active_coils.pickle",
        passive_coils_path=passive_b,
        limiter_path=f"{MC}/MAST-U_like_limiter.pickle",
        wall_path=f"{MC}/MAST-U_like_wall.pickle",
    )
    with open(SER_B, "wb") as f:
        pickle.dump(tokB, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"serialized Machine B -> {SER_B}")
    return tokB


def main():
    import phase0_lib as P0
    import phase15_lib as P15
    import phase2_data as D
    import pandas as pd

    tokB = build_and_serialize()
    tokA = P0.load_machine(os.path.join(MC, "serialized_tokamak.pkl"))

    df = pd.read_parquet(D.PARQUET)
    row = df[(df.split == "train") & (df.m_s > 0.4) & (df.m_s < 1.0)].iloc[0]
    rep = D.replay_kwargs(row)
    I = rep["active_currents"]

    print(f"\nstored (Machine A, 40 modes) m_s={row['m_s']:.5f} kappa={row['kappa']:.4f}")
    for name, tok, nmodes in [("A", tokA, 80), ("B", tokB, 80)]:
        try:
            r = P15.forward_label(tok, I, rep["paxis"], rep["Ip"], rep["fvac"],
                                  rep["alpha_m"], rep["alpha_n"], fix_n_modes=nmodes)
            print(f"Machine {name} ({nmodes} modes): m_s={r['m_s']:.5f} kappa={r['kappa']:.4f} "
                  f"gamma={r['gamma']:.1f} leuer={r['leuer']:.3f}")
        except Exception as e:
            print(f"Machine {name}: FAILED {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
