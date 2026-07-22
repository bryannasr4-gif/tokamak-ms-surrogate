"""
tier1_discover.py -- deterministic, metadata-only discovery + selection of real MAST EFIT slices.

Scans a frozen candidate set of original-MAST M9 shots, keeps those with a complete efm, builds the
per-slice quality table (real EFIT descriptors), pools quality flat-top slices across shots, then
farthest-point-samples ~N slices spanning (kappa, delta, li, betap) -- deliberately including the
high-kappa tail. Publishes the full candidate list, per-shot drop reasons, yield, and the frozen
selection. NO solver, NO surrogate here -> selection cannot be tuned on results (pre-registration).

Run:  python experiments/tier1_discover.py            (default: build selection of ~60)
      python experiments/tier1_discover.py 80         (target N)
"""
import os, sys, json
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tier1_lib as T

ROOT = T.ROOT
DATA = os.path.join(ROOT, "data")

# Frozen candidate shot set: evenly spaced across the M9 efm range (28412-30473).
# Even spacing (not cherry-picked) keeps selection defensible; complete-efm + quality gates do the rest.
def candidate_shots(step=12, lo=28412, hi=30473):
    return list(range(lo, hi + 1, step))


def build(target_n=60):
    shots = candidate_shots()
    print(f"scanning {len(shots)} candidate M9 shots (28412..30473 step 12)")
    rows = []                 # one row per quality slice
    shot_report = []          # per-shot outcome
    for shot in shots:
        try:
            if not T.has_complete_efm(shot):
                shot_report.append(dict(shot=shot, status="incomplete_efm")); print(f"  {shot}: incomplete efm"); continue
            d = T.read_shot(shot)
        except Exception as e:
            shot_report.append(dict(shot=shot, status=f"read_err:{type(e).__name__}")); print(f"  {shot}: read err {e}"); continue
        qs = T.quality_slices(d)
        if not qs:
            shot_report.append(dict(shot=shot, status="no_quality_slice")); print(f"  {shot}: no quality slice"); continue
        # keep the best-chisq slice per coarse kappa bin within this shot (reduce autocorrelation)
        kbin = {}
        for ti in qs:
            r = T.real_descriptor_row(d, ti)
            b = round(r["kappa"] / 0.1)
            if b not in kbin or r["chisq"] < kbin[b][1]:
                kbin[b] = (ti, r["chisq"], r)
        kept = [v[2] | dict(shot=shot, slice=v[0]) for v in kbin.values()]
        for r in kept:
            rows.append(r)
        shot_report.append(dict(shot=shot, status="ok", n_quality=len(qs), n_kept=len(kept),
                                kappa=[round(min(r["kappa"] for r in kept), 3),
                                       round(max(r["kappa"] for r in kept), 3)]))
        print(f"  {shot}: {len(qs)} quality -> {len(kept)} kept (kappa {shot_report[-1]['kappa']})")

    print(f"\nPOOL: {len(rows)} quality slices from {sum(1 for s in shot_report if s['status']=='ok')} good shots")
    if not rows:
        print("!! no slices -- aborting"); return
    # farthest-point selection in standardized (kappa, delta, li, betap)
    feats = np.array([[r["kappa"], r["delta"], r["li"], r["betap"]] for r in rows], float)
    # seed from the highest-kappa slice so the OOD tail is guaranteed represented
    seed = int(np.argmax(feats[:, 0]))
    pick = T.farthest_point_select(feats, min(target_n, len(rows)), seed_idx=seed)
    sel = [rows[i] for i in sorted(pick, key=lambda i: (rows[i]["shot"], rows[i]["slice"]))]

    kap = np.array([r["kappa"] for r in sel])
    print(f"\nSELECTED {len(sel)} slices  kappa[{kap.min():.2f},{kap.max():.2f}] "
          f"median {np.median(kap):.2f}  (training kappa max 2.23)")
    n_ind = len(set(r["shot"] for r in sel))
    print(f"independent shots in selection: {n_ind}")

    os.makedirs(DATA, exist_ok=True)
    with open(os.path.join(DATA, "tier1_shotlist.json"), "w") as f:
        json.dump(dict(candidate_shots=shots, shot_report=shot_report,
                       n_pool=len(rows), target_n=target_n), f, indent=2)
    with open(os.path.join(DATA, "tier1_selection.json"), "w") as f:
        json.dump(dict(n=len(sel), n_independent_shots=n_ind,
                       kappa_range=[float(kap.min()), float(kap.max())],
                       selection=sel), f, indent=2)
    # also dump the FULL pool (for OOD/coverage analysis on all quality slices, not just selected)
    with open(os.path.join(DATA, "tier1_pool.json"), "w") as f:
        json.dump(dict(n=len(rows), rows=rows), f, indent=2)
    print("\nwrote data/tier1_shotlist.json, tier1_selection.json, tier1_pool.json")


if __name__ == "__main__":
    build(int(sys.argv[1]) if len(sys.argv) > 1 else 60)
