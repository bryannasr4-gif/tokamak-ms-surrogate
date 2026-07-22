"""
e2_baselines.py -- Stage-1 E2: trivial-baseline OOD detectors vs the ensemble epistemic
abstention flag, on the real-MAST 60-slice shape set.

Implements the frozen Stage-1 pre-registration (§2.2) EXACTLY. No solver runs. All
standardization / PCA / covariance / threshold
parameters are computed from TRAINING data ONLY; real slices are only projected / scored.

Question (§2.2): does a trivial input-space detector reproduce the domain flag on the real-MAST
shapes? Decision rule: E2-CONFIRM iff ANY of the three trivial detectors reaches
TPR >= 0.95 on the real slices AND FPR <= 0.05 on the held-out in-dist (val) set; else E2-SURPRISE.

D19-b feature completion (supplement item 1, solver-free): the frozen 60-slice descriptor set
persists only 11/20 SHAPE_FEATURES; the 4 squareness + 5 gap features are completed here from the
cached real LCFS boundaries WITHOUT any solve --
  * squareness := the freegs4e Luce-quadrant boundary-geometry code path (Equilibrium.squareness)
    evaluated on the raw LCFS polygon via a minimal shim exposing separatrix() (no reimplementation),
  * gaps       := phase15_lib._seg_dist(LCFS, MAST-U limiter), limiter read from the serialized
    MAST-U tokamak pickle (loading a pickle is not a solve).
VALIDATION GATE (D19-b): before any completed feature is accepted, the completion path must first
reproduce the persisted per-slice features from the same cache inputs -- bit-exact for the
geometry-derived ones, verbatim-read for the EFIT scalars. A slice that fails validation, or whose
completed features are non-finite, cannot be completed -> HARD STOP (the §2.2 missing-feature rule).

Numeric pins (supplement item 5; E2_E3_TURNKEY §5) for bit-identical VERIFY-S1.2 re-runs:
  float64 from parquet; train std ddof=0; np.cov(rowvar=False) ddof=1 + np.linalg.pinv (cov is
  RANK-DEFICIENT 19/20 because delta == (delta_upper+delta_lower)/2 exactly); PCA = SVD of the
  standardized-then-centered train matrix, d = min covering >=90% variance; np.percentile(...,99)
  default (linear) interpolation; hull = Delaunay primary + LP (linprog highs) in-script cross-check.

  ./fusion-env/Scripts/python.exe experiments/e2_baselines.py
"""
import os, sys, json, glob, hashlib, pickle
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tier1_lib as T
import phase15_lib as P15
from phase2_data import SHAPE_FEATURES
from phase2_model import load_ensemble, ensemble_predict
from freegs4e.equilibrium import Equilibrium

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
FIG = os.path.join(ROOT, "figures")

PARQUET = os.path.join(DATA, "dataset_v1_80q.parquet")
PARQUET_SHA256 = "22999307f0ab18ba676c2ec899c071373b6f1bc3fe70c803dd0d57e773ce6ec4"  # supplement item 8
SER_MACHINE = os.path.join(ROOT, "machine_configs", "MAST-U", "serialized_tokamak.pkl")
SELECTION = os.path.join(DATA, "tier1_selection.json")
RESOLVED_GLOB = os.path.join(DATA, "tier1_resolved", "*.json")
BASELINE = os.path.join(DATA, "tier1_epistemic_baseline.json")

# the 11 persisted SHAPE_FEATURES and their provenance for the D19-b validation gate
GEO_PERSISTED = ["kappa", "delta", "delta_upper", "delta_lower", "Rgeo", "a", "aspect"]  # bit-exact
EFIT_PERSISTED = ["Rmag", "Zaxis", "li", "betap"]                                        # verbatim
COMPLETED = ["sq_uo", "sq_ui", "sq_lo", "sq_li",
             "gap_min", "gap_inner", "gap_outer", "gap_top", "gap_bot"]                  # completed


def hard_stop(reason):
    """Write the §ESCALATIONS block and abort (E-2 style)."""
    block = (
        "\n\n## ESCALATION -- EXEC-S1.2 (E2) HARD STOP\n"
        f"- **when:** e2_baselines.py run\n"
        f"- **reason:** {reason}\n"
        "- **rule:** Stage-1 pre-registration §2.2 missing-feature / feasibility HARD STOP; "
        "item (1) validation gate.\n"
        "- **state:** no e2_baselines.json / figure written this run; awaiting adjudication.\n"
    )
    state = os.path.join(ROOT, "E2_ESCALATION.md")
    with open(state, "a", encoding="utf-8") as fh:
        if "§ESCALATIONS" not in open(state, encoding="utf-8").read():
            fh.write("\n\n# §ESCALATIONS\n")
        fh.write(block)
    print("HARD STOP:", reason)
    sys.exit(2)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class LCFSShim:
    """Minimal shim exposing separatrix() = the raw real LCFS polygon, so the freegs4e
    Luce-quadrant squareness code path runs on it with NO reimplementation / definitional drift."""
    def __init__(self, rz):
        self._rz = np.asarray(rz, float)

    def separatrix(self, *a, **k):
        return self._rz

    squareness = Equilibrium.squareness
    _separatrix_metrics = Equilibrium._separatrix_metrics


def complete_real_features():
    """D19-b: complete the 9 missing SHAPE_FEATURES for all 60 real slices from cached LCFS,
    validate the 11 persisted ones, and return (X_real [60,20], meta). HARD STOP on any failure."""
    sel = json.load(open(SELECTION))["selection"]
    if len(sel) != 60:
        hard_stop(f"selection has {len(sel)} slices, expected 60")
    tok = pickle.load(open(SER_MACHINE, "rb"))
    limR, limZ = P15._limiter_RZ(tok)

    cache = {}
    rows, ids, val = [], [], {"n_slices": len(sel), "geom_bit_exact": True,
                              "efit_verbatim_exact": True, "all_completed_finite": True,
                              "per_feature_max_abs_geom_diff": {}, "n_geom_mismatch": 0,
                              "n_efit_mismatch": 0, "n_nonfinite": 0}
    geo_maxdiff = {k: 0.0 for k in GEO_PERSISTED}
    for r in sel:
        shot, ti = int(r["shot"]), int(r["slice"])
        if shot not in cache:
            cache[shot] = T.read_shot(shot)
        sd = cache[shot]
        lr, lz = T.lcfs_slice(sd, ti)

        # -- validation gate: geometry-derived (bit-exact) --
        g = T.geom_descriptors(lr, lz)
        for k in GEO_PERSISTED:
            geo_maxdiff[k] = max(geo_maxdiff[k], abs(float(g[k]) - float(r[k])))
            if g[k] != r[k]:
                val["geom_bit_exact"] = False
                val["n_geom_mismatch"] += 1
        # -- validation gate: EFIT scalars (verbatim read) --
        rdr = T.real_descriptor_row(sd, ti)
        for k in EFIT_PERSISTED:
            if rdr[k] != r[k]:
                val["efit_verbatim_exact"] = False
                val["n_efit_mismatch"] += 1

        # -- complete the 9 missing features (solver-free) --
        rz = np.column_stack([lr, lz])
        sq = LCFSShim(rz).squareness()                       # (sq_uo, sq_ui, sq_lo, sq_li)
        d = P15._seg_dist(rz, limR, limZ)                    # per-LCFS-point gap to MAST-U limiter
        Rgeo = float(g["Rgeo"]); R, Z = rz[:, 0], rz[:, 1]
        inb = R < Rgeo; upp = Z > 0.0; gap_min = float(d.min())
        completed = dict(
            sq_uo=float(sq[0]), sq_ui=float(sq[1]), sq_lo=float(sq[2]), sq_li=float(sq[3]),
            gap_min=gap_min,
            gap_inner=float(d[inb].min()) if inb.any() else gap_min,
            gap_outer=float(d[~inb].min()) if (~inb).any() else gap_min,
            gap_top=float(d[upp].min()) if upp.any() else gap_min,
            gap_bot=float(d[~upp].min()) if (~upp).any() else gap_min,
        )
        if not all(np.isfinite(v) for v in completed.values()):
            val["all_completed_finite"] = False
            val["n_nonfinite"] += 1

        # assemble the full 20-dim SHAPE_FEATURES vector (persisted 11 + completed 9)
        feat = {}
        for k in SHAPE_FEATURES:
            if k in completed:
                feat[k] = completed[k]
            else:
                feat[k] = float(r[k])
        rows.append([feat[k] for k in SHAPE_FEATURES])
        ids.append(f"{shot}_{ti}")

    val["per_feature_max_abs_geom_diff"] = {k: geo_maxdiff[k] for k in GEO_PERSISTED}
    if not (val["geom_bit_exact"] and val["efit_verbatim_exact"] and val["all_completed_finite"]):
        hard_stop("D19-b validation gate FAILED: "
                  f"geom_mismatch={val['n_geom_mismatch']} efit_mismatch={val['n_efit_mismatch']} "
                  f"nonfinite_completed={val['n_nonfinite']}")
    return np.array(rows, float), ids, val


def in_hull_lp(hull_points, X):
    """LP membership cross-check: point x in conv(hull_points) iff exists w>=0, sum w = 1,
    P^T w = x. Returns boolean 'inside' per row of X. method='highs' (deterministic)."""
    from scipy.optimize import linprog
    n, d = hull_points.shape
    Pt = hull_points.T                       # (d, n)
    A_eq = np.vstack([Pt, np.ones((1, n))])  # (d+1, n)
    c = np.zeros(n)
    inside = np.zeros(len(X), bool)
    for i, x in enumerate(X):
        b_eq = np.concatenate([x, [1.0]])
        res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=(0, None), method="highs")
        inside[i] = bool(res.success)
    return inside


def main():
    os.makedirs(FIG, exist_ok=True)

    # ---- pin check (supplement item 8) ----
    got = sha256_file(PARQUET)
    if got != PARQUET_SHA256:
        hard_stop(f"parquet sha256 mismatch: got {got}, pinned {PARQUET_SHA256}")

    # ---- split (supplement item 3; A8.1): train 2602 / held-out val 465 from parquet's split ----
    df = pd.read_parquet(PARQUET)
    tr = df[df["split"] == "train"][SHAPE_FEATURES].to_numpy(float)
    va = df[df["split"] == "val"][SHAPE_FEATURES].to_numpy(float)
    n_train, n_val = len(tr), len(va)
    split_provenance = {
        "source": "dataset_v1_80q.parquet own 'split' column",
        "train": n_train, "held_out_in_dist_val": n_val,
        "held_out_definition": "split=='val' (NOT test_extrap, the deliberately-OOD kappa-delta corner)",
        "stale_meta_note": ("data/phase15_split_meta.json counts (train 2645 / val 466) are the "
                            "pre-relabel dataset_v1 counts; 44 rows (43 train + 1 val) were dropped "
                            "by the 80-mode relabel. The deployed ensemble normalization matches the "
                            "2602-row train set EXACTLY (max |dmean| = 0.0)."),
    }

    # ---- D19-b real-slice feature completion + validation gate ----
    Xreal, real_ids, d19b = complete_real_features()   # (60,20)
    if len(Xreal) != 60:
        hard_stop(f"completed {len(Xreal)} real slices, expected 60")

    # ---- standardization (TRAIN ONLY) ----
    xmean = tr.mean(0)
    xstd = tr.std(0, ddof=0)                 # ddof=0 pin
    def z(X):
        return (X - xmean) / xstd
    Ztr, Zva, Zreal = z(tr), z(va), z(Xreal)

    # ============ Detector 1: Mahalanobis (standardized train, full cov, pinv) ============
    cov = np.cov(Ztr, rowvar=False)          # ddof=1
    cov_rank = int(np.linalg.matrix_rank(cov))
    inv = np.linalg.pinv(cov)                # rank-deficient (19/20) -> pinv required
    mu_z = Ztr.mean(0)
    def maha(Z):
        dd = Z - mu_z
        return np.sqrt(np.einsum("ij,jk,ik->i", dd, inv, dd))
    md_train = maha(Ztr)
    maha_thr = float(np.percentile(md_train, 99))     # linear interpolation (default)
    maha_real = maha(Zreal); maha_val = maha(Zva)
    flag_maha_real = maha_real > maha_thr
    flag_maha_val = maha_val > maha_thr

    # ============ Detector 2: per-feature range (raw train min/max) ============
    lo = tr.min(0); hi = tr.max(0)
    viol_real = (Xreal < lo) | (Xreal > hi)
    viol_val = (va < lo) | (va > hi)
    flag_range_real = viol_real.any(1)
    flag_range_val = viol_val.any(1)
    val_viol_by_feature = {SHAPE_FEATURES[i]: int(viol_val[:, i].sum())
                           for i in range(len(SHAPE_FEATURES)) if viol_val[:, i].sum() > 0}

    # ============ Detector 3: convex hull in PCA space ============
    from scipy.spatial import Delaunay
    zmean = Ztr.mean(0)
    Zc = Ztr - zmean                          # standardized-then-centered train
    U, S, Vt = np.linalg.svd(Zc, full_matrices=False)
    evr = (S ** 2) / np.sum(S ** 2)
    cum = np.cumsum(evr)
    pca_d = int(np.searchsorted(cum, 0.90) + 1)   # min PCs covering >=90%, NO cap
    pca_var = float(cum[pca_d - 1])
    comps = Vt[:pca_d]                         # (d, 20)
    def project(X):
        return (z(X) - zmean) @ comps.T
    Ptr = project(tr); Pva = project(va); Preal = project(Xreal)
    hull = Delaunay(Ptr)
    flag_hull_real = hull.find_simplex(Preal) < 0     # outside hull -> flagged OOD
    flag_hull_val = hull.find_simplex(Pva) < 0
    # in-script LP cross-check (kills any Qhull-tolerance doubt)
    lp_in_val = in_hull_lp(Ptr, Pva)
    lp_flag_val = ~lp_in_val
    lp_in_real = in_hull_lp(Ptr, Preal)
    lp_flag_real = ~lp_in_real
    hull_xcheck = {
        "val_delaunay_flagged": int(flag_hull_val.sum()),
        "val_lp_flagged": int(lp_flag_val.sum()),
        "val_disagreements": int((flag_hull_val != lp_flag_val).sum()),
        "real_delaunay_flagged": int(flag_hull_real.sum()),
        "real_lp_flagged": int(lp_flag_real.sum()),
        "real_disagreements": int((flag_hull_real != lp_flag_real).sum()),
    }

    # ============ Comparator: ensemble epistemic flag ============
    # threshold computed INSIDE E2 on the held-out in-dist (val) set (§2.2)
    models, _ = load_ensemble("surrogate")
    epi_val = ensemble_predict(models, va)["epi_std"][:, 0]
    epi_thr = float(np.percentile(epi_val, 99))
    epi_val_median = float(np.median(epi_val))
    flag_epi_val = epi_val > epi_thr
    # median cross-check vs the S0.2 baseline artifact (different eval set by design)
    baseline_median = None; baseline_note = "artifact missing"
    if os.path.exists(BASELINE):
        b = json.load(open(BASELINE))
        baseline_median = float(b.get("value"))
        baseline_note = ("S0.2 baseline artifact present + in-band; eval set = 500-row rs=0 sample "
                         "of the FULL parquet (n=3254), so a ~9% median offset vs the val-split "
                         "median is expected -- the 'unverifiable' branch is NOT needed.")
    # real-slice epistemic flag: STORED surr_epi_std from the resolved re-solves (defined on 55 ok)
    rmap = {}
    for f in sorted(glob.glob(RESOLVED_GLOB)):
        rr = json.load(open(f))
        rmap[f"{int(rr['shot'])}_{int(rr['slice'])}"] = rr
    epi_real = {}
    for sid in real_ids:
        rr = rmap.get(sid)
        if rr is not None and rr.get("status") == "ok" and rr.get("surr_epi_std") is not None:
            epi_real[sid] = float(rr["surr_epi_std"])
    comparator_ids = [sid for sid in real_ids if sid in epi_real]          # the 55 ok slices
    missing_comparator = [sid for sid in real_ids if sid not in epi_real]  # the 5 excluded
    flag_epi_real = {sid: (epi_real[sid] > epi_thr) for sid in comparator_ids}

    # ============ per-detector rates ============
    def rates(flag_real, flag_val):
        return dict(tpr_real=float(np.mean(flag_real)), n_tpr=int(len(flag_real)),
                    n_tpr_flagged=int(np.sum(flag_real)),
                    fpr_indist=float(np.mean(flag_val)), n_fpr=int(len(flag_val)),
                    n_fpr_flagged=int(np.sum(flag_val)))
    per_detector = {
        "mahalanobis": rates(flag_maha_real, flag_maha_val),
        "range": rates(flag_range_real, flag_range_val),
        "hull_pca": rates(flag_hull_real, flag_hull_val),
    }
    comparator = {
        "tpr_real": float(np.mean([flag_epi_real[s] for s in comparator_ids])),
        "n_tpr": len(comparator_ids), "n_tpr_flagged": int(sum(flag_epi_real.values())),
        "fpr_indist": float(np.mean(flag_epi_val)), "n_fpr": int(len(flag_epi_val)),
        "n_fpr_flagged": int(np.sum(flag_epi_val)),
        "note": "comparator (not a trivial detector); FPR on val is ~1% by construction (val p99 threshold).",
    }

    # ============ agreement matrix across all 4 flags on the 55-slice intersection ============
    maha_by_id = dict(zip(real_ids, flag_maha_real))
    range_by_id = dict(zip(real_ids, flag_range_real))
    hull_by_id = dict(zip(real_ids, flag_hull_real))
    flag_names = ["mahalanobis", "range", "hull_pca", "epistemic"]
    Fmat = np.array([[bool(maha_by_id[s]), bool(range_by_id[s]), bool(hull_by_id[s]),
                      bool(flag_epi_real[s])] for s in comparator_ids])
    agree = [[int(np.sum(Fmat[:, i] == Fmat[:, j])) for j in range(4)] for i in range(4)]
    per_flag_positive = {flag_names[i]: int(Fmat[:, i].sum()) for i in range(4)}

    # ============ decision rule (§2.2), applied LITERALLY on exact values ============
    trivial = {"mahalanobis": per_detector["mahalanobis"], "range": per_detector["range"],
               "hull_pca": per_detector["hull_pca"]}
    qualifying = {k: v for k, v in trivial.items()
                  if (v["tpr_real"] >= 0.95 and v["fpr_indist"] <= 0.05)}
    decision = "E2-CONFIRM" if qualifying else "E2-SURPRISE"

    # ============ assemble artifact ============
    out = {
        "experiment": "Stage-1 E2 -- trivial-baseline OOD detectors vs epistemic abstention flag",
        "prereg": "Stage-1 pre-registration §2.2 (frozen)",
        "confirmatory": False,
        "feature_list": list(SHAPE_FEATURES),
        "standardization_provenance": "train-only",
        "standardization": {"mean": "train", "std": "train", "ddof": 0},
        "parquet_sha256": got,
        "parquet_sha256_pinned": PARQUET_SHA256,
        "split_provenance": split_provenance,
        "denominators": {
            "trivial_detectors_scored_on": 60,
            "comparator_defined_on": len(comparator_ids),
            "missing_comparator_slices": missing_comparator,
            "missing_comparator_reason": ("30380_68 selected but never produced a resolved JSON; "
                                          "28784_39, 28796_90, 28868_35, 29180_42 are unconverged."),
        },
        "d19b_feature_completion": {
            "missing_features_completed": COMPLETED,
            "squareness_source": ("freegs4e Equilibrium.squareness (Luce 2013 quadrant construction) "
                                  "via a shim exposing separatrix() = raw real LCFS polygon"),
            "gap_source": ("phase15_lib._seg_dist(LCFS, MAST-U limiter); limiter from "
                           "machine_configs/MAST-U/serialized_tokamak.pkl (pickle load, not a solve)"),
            "validation_gate": d19b,
        },
        "disclosures": {
            "gaps": "gaps are the real MAST shape measured against the MAST-U wall (the Tier-1 cross-machine framing).",
            "li_betap": "li and betap keep the known EFIT-vs-descriptors() definitional caveat.",
        },
        "detectors": {
            "mahalanobis": {"space": "standardized train features (ddof=0)",
                            "covariance": "np.cov(rowvar=False) ddof=1", "inverse": "np.linalg.pinv",
                            "covariance_rank": cov_rank, "n_features": len(SHAPE_FEATURES)},
            "range": {"space": "raw features", "rule": "flag if ANY feature outside train [min,max]",
                      "val_violations_by_feature": val_viol_by_feature},
            "hull_pca": {"space": "PCA of standardized-then-centered train",
                         "membership": "Delaunay.find_simplex (primary) + LP linprog highs (cross-check)",
                         "cross_check": hull_xcheck},
        },
        "thresholds": {
            "mahalanobis_p99_train_selfdist": maha_thr,
            "range": "per-feature train min/max",
            "epistemic_p99_val": epi_thr,
            "epistemic_median_val": epi_val_median,
            "epistemic_median_crosscheck_baseline": baseline_median,
            "epistemic_median_crosscheck_note": baseline_note,
            "percentile_interpolation": "numpy default (linear)",
        },
        "pca_dims": pca_d,
        "pca_variance_covered": pca_var,
        "pca_explained_variance_ratio": [float(x) for x in evr],
        "per_detector": per_detector,
        "comparator_epistemic": comparator,
        "agreement_matrix": {
            "flags": flag_names,
            "n_slices": len(comparator_ids),
            "matrix_pairwise_agreement_counts": agree,
            "per_flag_positive_count": per_flag_positive,
        },
        "decision_rule": ("E2-CONFIRM iff ANY trivial detector has TPR_real >= 0.95 AND "
                          "FPR_val <= 0.05; else E2-SURPRISE."),
        "qualifying_detectors": list(qualifying.keys()),
        "decision": decision,
    }
    with open(os.path.join(DATA, "e2_baselines.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print("saved data/e2_baselines.json")

    # ---- figure: operating points + per-slice flag heatmap ----
    _figure(out, real_ids, comparator_ids,
            flag_maha_real, flag_range_real, flag_hull_real, flag_epi_real)

    # ---- console summary ----
    print(f"\n=== E2 decision: {decision} ===")
    for k in ("mahalanobis", "range", "hull_pca"):
        v = per_detector[k]
        print(f"  {k:12s} TPR_real={v['tpr_real']:.4f} ({v['n_tpr_flagged']}/{v['n_tpr']})  "
              f"FPR_val={v['fpr_indist']:.4f} ({v['n_fpr_flagged']}/{v['n_fpr']})")
    print(f"  comparator   TPR_real={comparator['tpr_real']:.4f} "
          f"({comparator['n_tpr_flagged']}/{comparator['n_tpr']})  "
          f"FPR_val={comparator['fpr_indist']:.4f}")
    print(f"  thresholds: maha_p99={maha_thr:.4f}  epi_p99_val={epi_thr:.5f}  "
          f"epi_median_val={epi_val_median:.5f}  baseline={baseline_median}")
    print(f"  PCA d={pca_d} ({pca_var*100:.2f}%); cov rank={cov_rank}/20; qualifying={list(qualifying)}")
    return out


def _figure(out, real_ids, comparator_ids, fm, fr, fh, fe):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    fig, ax = plt.subplots(1, 2, figsize=(15, 5.2))

    # panel 1: operating points (TPR_real vs FPR_val), decision box
    pts = [("Mahalanobis", out["per_detector"]["mahalanobis"], "navy"),
           ("range", out["per_detector"]["range"], "darkorange"),
           ("hull(PCA)", out["per_detector"]["hull_pca"], "seagreen"),
           ("epistemic (comparator)", out["comparator_epistemic"], "purple")]
    ax[0].axvspan(0, 0.05, color="green", alpha=0.06)
    ax[0].axhspan(0.95, 1.0, color="green", alpha=0.06)
    ax[0].axvline(0.05, ls="--", c="green", lw=1)
    ax[0].axhline(0.95, ls="--", c="green", lw=1)
    for name, v, c in pts:
        ax[0].scatter(v["fpr_indist"], v["tpr_real"], s=140, c=c, edgecolor="k", zorder=5, label=name)
    ax[0].set_xlim(-0.02, 0.32); ax[0].set_ylim(0.4, 1.03)
    ax[0].set_xlabel("FPR on held-out in-dist (val)"); ax[0].set_ylabel("TPR on real MAST slices")
    ax[0].set_title(f"E2 detector operating points -> {out['decision']}\n"
                    "(green box = CONFIRM region: TPR>=0.95 & FPR<=0.05)")
    ax[0].legend(fontsize=8, loc="lower right")

    # panel 2: per-slice flag heatmap (4 flags x 60 slices; epistemic gray where undefined)
    names = ["Mahalanobis", "range", "hull(PCA)", "epistemic"]
    M = np.full((4, len(real_ids)), np.nan)
    for j, sid in enumerate(real_ids):
        M[0, j] = 1.0 if fm[j] else 0.0
        M[1, j] = 1.0 if fr[j] else 0.0
        M[2, j] = 1.0 if fh[j] else 0.0
        if sid in comparator_ids:
            M[3, j] = 1.0 if fe[sid] else 0.0
    cmap = ListedColormap(["#e9e9e9", "#c0392b"])   # 0=not flagged (light), 1=flagged (red)
    Mmask = np.ma.masked_invalid(M)
    cmap.set_bad("#7f8c8d")                          # gray = comparator undefined
    ax[1].imshow(Mmask, aspect="auto", cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
    ax[1].set_yticks(range(4)); ax[1].set_yticklabels(names)
    ax[1].set_xlabel(f"real slice index (n={len(real_ids)}; gray = comparator undefined, "
                     f"{len(real_ids)-len(comparator_ids)} slices)")
    ax[1].set_title("per-slice OOD flags (red = flagged OOD)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "e2_baselines.png"), dpi=130)
    print("saved figures/e2_baselines.png")


if __name__ == "__main__":
    main()
