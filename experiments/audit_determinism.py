"""
audit_determinism.py -- is a single machine build + solve DETERMINISTIC ACROSS PROCESSES?

The module-level LatinHypercube engine (seed=42) in freegsnke/refine_passive.py advances its
state on every call, so building the machine TWICE in one process diverges. But each Phase-1
worker is a SEPARATE process that builds the machine ONCE (engine fresh from seed=42). If two
separate processes give identical m_s, the Phase-1 grid is clean (all workers built identical
machines) and the 19.5% spread seen earlier was a within-process artifact.

Run this script TWICE as separate processes and compare the printed m_s to high precision.
"""
import os
import sys
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.getcwd(), "experiments"))
from phase1_generate import build_tokamak, evaluate

tok = build_tokamak()                      # ONE build, first in this process
r = evaluate(tok, 1.00, 0.0)               # canonical shape, 65x65
print(f"M_S={r['m_s']:.10f} GAMMA={r['gamma']:.6f} KAPPA={r['kappa']:.6f}")
