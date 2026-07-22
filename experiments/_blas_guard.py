"""_blas_guard.py -- refuse to run a real-solve script unless all 5 BLAS vars are pinned to 1
(the locked numerical protocol). Call assert_pinned() at the START of main() in standalone solve
scripts. It is an assert-and-exit guard (not a setter), so the operator must prefix the command:
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 ...
"""
import os
import sys

_NEED = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
         "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")


def assert_pinned():
    bad = [k for k in _NEED if os.environ.get(k) != "1"]
    if bad:
        sys.stderr.write(
            "REFUSING TO RUN: BLAS threads are not pinned to 1 (locked protocol). "
            f"Unset/!=1: {bad}\nPrefix the command with: "
            + " ".join(f"{k}=1" for k in _NEED) + " PYTHONIOENCODING=utf-8 ...\n")
        sys.exit(2)
