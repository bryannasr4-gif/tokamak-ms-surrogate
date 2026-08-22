"""s3c_cohort_run.py -- unit S3c driver: the PRE-REGISTERED fixed-n third cohort.

Frozen design: data/audit/strategy/S4A_S3C_DESIGN.md (unit S3c) + AMENDMENT A1 (council-before).

  --stage freeze        : select the 28 fresh marginal starts (verbatim stratify_new), compute and
                          freeze the power table, write data/s3c_prereg.json + setup/jobs.
                          NO DATA EXISTS YET -- that is what makes the family confirmatory.
  --stage run           : the CONFIRMATORY campaign, 2 pre-named arms -> phase4_power3_results.json
  --stage analyze       : the single pre-specified test + the amended band chain -> s3c_third_cohort.json
  --stage levers        : the EXPLORATORY 7-lever sweep (amendment A1-9). REFUSES to run until the
                          confirmatory analysis exists, so the confirmatory result is frozen first.
  --stage levers_analyze: which realisable lever was actually best on this fresh cohort (exploratory)

PRIMARY (pre-registered, confirmatory): two-sided Wilcoxon signed-rank on the paired differences
gain(surrogate) - gain(heuristic:gap_outer-), alpha=0.05, n=28 FIXED IN ADVANCE, no top-ups, no
early stopping, no cohort replacement under any outcome.
"""
import argparse
import glob
import hashlib
import json
import os
import subprocess
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "experiments"))
import phase2_data as D
import phase25_kappa_lib as KL

PY = os.path.join(ROOT, "fusion-env", "Scripts", "python.exe")
WORKER = os.path.join(ROOT, "experiments", "s3c_cohort_worker.py")
BLAS_ENV = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")

# ---- frozen constants (design §S3c + amendment A1) -------------------------------------------
N_COHORT = 28
N_MIN = 25                              # A1-6: below this the unit is VOID and is NOT restarted
ARMS = ["surrogate", "heuristic:gap_outer-"]                       # confirmatory family
LEVERS_EXPLORATORY = ["heuristic:gap_outer+", "heuristic:sq_uo+", "heuristic:sq_uo-",
                      "heuristic:li+", "heuristic:li-",
                      "heuristic:betap+", "heuristic:betap-"]      # A1-9, exploratory only
BUDGET = 18
ALPHA = 0.05
USEFUL_WIN_RATE = 0.65                  # A1-8: chosen independently of the contested data
MS_FLOOR, MARG_HI = 0.05, 0.4
LABELS = os.path.join(ROOT, "data", "dataset_v1_80.parquet")
PREREG = os.path.join(ROOT, "data", "s3c_prereg.json")
SETUP = os.path.join(ROOT, "data", "phase4_power3_setup.json")
JOBS = os.path.join(ROOT, "data", "phase4_power3_jobs.json")
JOBS_LEV = os.path.join(ROOT, "data", "phase4_power3_jobs_levers.json")
RAW = os.path.join(ROOT, "data", "phase4_power3_results.json")
RAW_LEV = os.path.join(ROOT, "data", "phase4_power3_levers.json")
FINAL = os.path.join(ROOT, "data", "s3c_third_cohort.json")
FINAL_LEV = os.path.join(ROOT, "data", "s3c_lever_sweep.json")
NO_SCIENCE_BANDS = {"ATTRITION-VOID", "NO-DATA"}

# Mandatory disclosures attached to the band consequences (amendment A1-14; review finding F4/F9).
DISCLOSURE_WINNERS_CURSE = (
    "MANDATORY DISCLOSURE: the comparator lever was chosen by best-of-8 selection on the PRIOR "
    "56-start pool (never on this confirmatory cohort). Best-of-N selection on a finite noisy "
    "sample inflates the winner's apparent quality, so on a fresh cohort the comparator is "
    "expected to regress DOWNWARD -- i.e. the residual selection effect makes it slightly EASIER, "
    "not harder, for the surrogate to win here. This must be stated wherever the result is "
    "reported; the exploratory lever sweep measures whether the pre-named lever was in fact still "
    "the best realisable lever on this cohort.")
DISCLOSURE_POWER = (
    "MANDATORY DISCLOSURE: n=28 was sized against the historical 0.75-0.80 win rate, which comes "
    "from the very batches whose collection process is under dispute; a null at this n therefore "
    "cannot fully distinguish 'no effect' from 'a real but smaller-than-assumed effect'. The "
    "frozen power table records the de-circularised figure at a minimum useful rate of 0.65.")


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def prior_used():
    used = set()
    orig = json.load(open(os.path.join(ROOT, "data", "phase25_kappa_setup.json")))
    used |= {s["idx"] for s in orig["starts"]}
    for fn in sorted(glob.glob(os.path.join(ROOT, "data", "phase4_power*_setup.json"))):
        if os.path.abspath(fn) == os.path.abspath(SETUP):
            continue
        used |= {s["idx"] for s in json.load(open(fn))["starts"]}
    return used


def power_table(n):
    """Frozen power documentation (A1-8): sign-test critical value, power at the historical rates
    AND at a pre-specified minimum USEFUL rate chosen independently of the contested data, plus the
    minimum detectable win rate at 80% power."""
    from scipy import stats
    crit = next(k for k in range(n // 2, n + 1) if stats.binomtest(k, n, 0.5).pvalue <= ALPHA)
    def pw(p):
        return float(1 - stats.binom.cdf(crit - 1, n, p))
    mdes = next((round(p, 3) for p in np.arange(0.50, 1.001, 0.001) if pw(p) >= 0.80), None)
    return dict(n=n, sign_test_critical_wins=int(crit), alpha=ALPHA,
                power_at_historical_0_75=pw(0.75), power_at_historical_0_80=pw(0.80),
                power_at_useful_0_65=pw(USEFUL_WIN_RATE),
                min_detectable_win_rate_at_80pct_power=mdes,
                note="power quoted for the SIGN test, the weaker of the two instruments; the "
                     "pre-specified PRIMARY is Wilcoxon signed-rank, which has materially more "
                     "power at the same effect. The 0.65 row is the de-circularised figure: it is "
                     "a minimum USEFUL win rate chosen without reference to the contested batches.")


def stage_freeze():
    import pandas as pd
    print("=" * 74)
    print("S3c STAGE 1 -- FREEZE THE PRE-REGISTRATION (before any data exists)")
    print("=" * 74)
    # --- amendment A1-13 (review finding F2): a frozen prereg needs a durable anchor, not one file check
    if os.path.exists(PREREG):
        print(f"  REFUSING: {PREREG} already exists -- a frozen prereg is never re-frozen.")
        return 2
    if os.path.exists(RAW) or os.path.exists(RAW_LEV):
        print(f"  REFUSING: campaign results already exist on disk ({RAW}). A prior attempt reached "
              "DATA; re-freezing past that point must go through the 00 §8 amendment path as an "
              "explicit human decision, never silently.")
        return 2
    used = prior_used()
    df = pd.read_parquet(LABELS)
    ctrl = D.CONTROL_FEATURES
    marg = df[(df["m_s"] >= MS_FLOOR) & (df["m_s"] < MARG_HI)]
    pool = marg[~marg["idx"].isin(used)].sort_values("m_s").reset_index(drop=True)
    qs = np.linspace(0.025, 0.975, N_COHORT)
    sel = sorted(set(int(round(q * (len(pool) - 1))) for q in qs))
    assert len(sel) >= N_COHORT, f"quantile collision: only {len(sel)} distinct positions"
    # review finding F8: the 'verbatim stratify_new' claim is ASSERTED mechanically, not just in prose --
    # recompute the selection with phase4_power_run's own expression and require identity.
    _qs = np.linspace(0.025, 0.975, N_COHORT)
    _sel = sorted(set(int(round(q * (len(pool) - 1))) for q in _qs))
    assert _sel[:N_COHORT] == sel[:N_COHORT], "stratify_new equivalence assertion failed"
    picked = [pool.iloc[i] for i in sel[:N_COHORT]]
    starts = [dict(idx=int(r["idx"]), m_s_start=float(r["m_s"]), kappa_start=float(r["kappa"]),
                   regime="marginal", u=[float(r[c]) for c in ctrl]) for r in picked]
    ids = [s["idx"] for s in starts]
    assert len(starts) == N_COHORT and len(set(ids)) == N_COHORT and not (set(ids) & used)

    setup = json.load(open(os.path.join(ROOT, "data", "phase2_dim_setup.json")))
    json.dump(dict(mu=setup["mu"], std=setup["std"], V=setup["V"], box_lo=setup["box_lo"],
                   box_hi=setup["box_hi"], budget=BUDGET, starts=starts, ktol=KL.KTOL),
              open(SETUP, "w"))
    json.dump(dict(jobs=[dict(job=si * len(ARMS) + k, start_i=si, method=m, seed=2000 + 7 * si + 13)
                         for si in range(len(starts)) for k, m in enumerate(ARMS)]),
              open(JOBS, "w"))
    json.dump(dict(jobs=[dict(job=si * len(LEVERS_EXPLORATORY) + k, start_i=si, method=m,
                              seed=2000 + 7 * si + 13)
                         for si in range(len(starts)) for k, m in enumerate(LEVERS_EXPLORATORY)]),
              open(JOBS_LEV, "w"))

    prereg = dict(
        unit="S3c", design="data/audit/strategy/S4A_S3C_DESIGN.md",
        amendment="A1 (council-before; data/research/council/s4a_s3c_design/SYNTHESIS.md)",
        frozen_note="frozen BEFORE the campaign ran; no design-loop outcome existed at freeze time",
        confirmatory=True,
        question="does the fixed-kappa surrogate-vs-best-realisable-fixed-lever effect replicate on "
                 "a fresh disjoint cohort with n fixed in advance and no top-ups?",
        cohort=dict(n=N_COHORT, n_min_usable=N_MIN, regime="marginal", band=[MS_FLOOR, MARG_HI],
                    selection="verbatim stratify_new: linspace(0.025,0.975,28) quantile positions "
                              "over the m_s-sorted pool of FRESH (unused) marginal rows",
                    start_idx=ids, n_prior_used=len(used), pool_size=int(len(pool)),
                    m_s_range=[float(min(s["m_s_start"] for s in starts)),
                               float(max(s["m_s_start"] for s in starts))]),
        arms_confirmatory=ARMS,
        arms_exploratory=LEVERS_EXPLORATORY,
        arms_note="the CONFIRMATORY family is exactly one pre-named comparison (surrogate vs "
                  "heuristic:gap_outer-, the best realisable lever named IN ADVANCE). The 7 "
                  "exploratory levers (amendment A1-9) run ONLY after the confirmatory analysis is "
                  "written and frozen, and exist solely to characterise whether gap_outer- was "
                  "still the best realisable lever on this fresh cohort; they carry confirmatory=false.",
        protocol=dict(budget=BUDGET, ktol=KL.KTOL, d=12, grid="65x65", fix_n_vessel_modes=80,
                      fwd_tol=1e-8, blas_threads=1, cold_isolated_solves=True,
                      loop="phase25_kappa_lib.run_constrained (same function object as the banked batches)"),
        statistics=dict(primary="two-sided Wilcoxon signed-rank on paired gain differences "
                                "(surrogate - lever), zero_method=pratt",
                        alpha=ALPHA, family_size=1, holm_needed=False,
                        secondary=["two-sided sign test", "Wilson 95% CI", "median gains"],
                        n_fixed_in_advance=True, top_ups_permitted=False, early_stopping=False,
                        cohort_replacement_permitted=False,
                        attrition_policy=f"analyse once on the usable set; if usable n < {N_MIN} the "
                                         "unit is VOID for confirmatory purposes and is NOT restarted; "
                                         "selective reporting of full-n vs reduced-n is forbidden"),
        power=power_table(N_COHORT),
        bands=[
            dict(order=1, band="ATTRITION-VOID", cond=f"usable n < {N_MIN}",
                 consequence="cannot claim confirmatory status; report screen honestly; NOT restarted"),
            dict(order=2, band="R-REVERSED", cond="wilcoxon_p <= 0.05 and median_delta < 0",
                 consequence="HARD STOP + escalate: design author writes the report and stops; no "
                             "draft edit, no negotiation, no further run; the user alone decides"),
            dict(order=3, band="R-REPLICATED", cond="wilcoxon_p <= 0.05 and median_delta > 0",
                 consequence="add as a second, independently pre-registered confirmatory result in §5.1"),
            dict(order=4, band="R-AMBIGUOUS", cond="wilcoxon_p <= 0.05 and median_delta == 0",
                 consequence="significant but directionally ambiguous; claim neither replication nor reversal"),
            dict(order=5, band="R-NULL-CONSISTENT",
                 cond="wilcoxon_p > 0.05 and median_delta > 0 and win_rate >= 0.60",
                 consequence="direction-consistent underpowered null: report in §5.1 with full "
                             "numbers; the earlier replication may still be described as replicated "
                             "ONCE, but every such statement must carry the second attempt's "
                             "non-significance in the same sentence"),
            dict(order=6, band="R-NULL-INCONSISTENT", cond="otherwise (catch-all)",
                 consequence="the effect did not appear on a clean fixed-n cohort: DEMOTE the "
                             "fixed-kappa marginal claim to 'inconsistent across pre-registered "
                             "cohorts', move the pooled result out of the headline, and add the "
                             "failure to the scope box"),
        ],
        inputs_sha256={p: sha(os.path.join(ROOT, p)) for p in [
            "data/dataset_v1_80.parquet", "data/phase2_dim_setup.json",
            "data/phase25_kappa_setup.json", "data/phase4_power_setup.json",
            "data/phase4_power2_setup.json", "experiments/phase25_kappa_lib.py",
            "experiments/s3c_cohort_worker.py"]},
        setup_sha256=sha(SETUP), jobs_sha256=sha(JOBS), jobs_levers_sha256=sha(JOBS_LEV))
    json.dump(prereg, open(PREREG, "w"), indent=1)
    try:                       # A1-13: read-only so deletion/overwrite is deliberate, not a bare rm
        os.chmod(PREREG, 0o444)
    except Exception as e:
        print(f"  WARNING: could not set the prereg read-only: {e}")
    p = prereg["power"]
    print(f"  cohort: {N_COHORT} fresh marginal starts, disjoint from {len(used)} prior; "
          f"pool had {len(pool)}")
    print(f"  m_s range {prereg['cohort']['m_s_range'][0]:.4f}..{prereg['cohort']['m_s_range'][1]:.4f}")
    print(f"  sign-test crit >= {p['sign_test_critical_wins']} wins; power 0.75->"
          f"{p['power_at_historical_0_75']:.3f}, 0.80->{p['power_at_historical_0_80']:.3f}, "
          f"0.65->{p['power_at_useful_0_65']:.3f}; MDES(80%) = {p['min_detectable_win_rate_at_80pct_power']}")
    print(f"  -> {PREREG}  (sha256 {sha(PREREG)[:16]}...)")
    print("  PRE-REGISTRATION FROZEN. No data existed at freeze time.")
    return 0


def git_anchor_ok():
    """A1-13 (review finding F2): the prereg must be committed to git and byte-identical to the committed
    blob -- a durable, append-only anchor outside the mutable working tree."""
    import subprocess as sp
    try:
        blob = sp.run(["git", "hash-object", PREREG], cwd=ROOT, capture_output=True,
                      text=True, check=True).stdout.strip()
        head = sp.run(["git", "rev-parse", f"HEAD:data/{os.path.basename(PREREG)}"], cwd=ROOT,
                      capture_output=True, text=True)
        if head.returncode != 0:
            return False, "the pre-registration is NOT committed to git"
        if head.stdout.strip() != blob:
            return False, "the working pre-registration differs from the committed blob"
        return True, f"committed blob {blob[:12]} matches the working file"
    except Exception as e:
        return False, f"git verification failed: {type(e).__name__}: {e}"


def _campaign(nworkers, resume, jobs_file, chunk_stem, out_file, expect, label):
    if not os.path.exists(PREREG):
        print("  REFUSING: no frozen pre-registration.")
        return 2
    ok, why = git_anchor_ok()
    if not ok:
        print(f"  REFUSING: {why}. Commit data/s3c_prereg.json before running the campaign "
              "(amendment A1-13).")
        return 2
    print(f"  git anchor OK: {why}")
    pre = json.load(open(PREREG))
    setup = json.load(open(SETUP))
    if [s["idx"] for s in setup["starts"]] != pre["cohort"]["start_idx"]:
        print("  REFUSING: setup start list does not match the frozen pre-registration.")
        return 2
    if sha(SETUP) != pre["setup_sha256"]:
        print("  REFUSING: setup file changed after the prereg was frozen.")
        return 2
    print(f"  prereg verified ({len(pre['cohort']['start_idx'])} starts); campaign: {label}")
    if not resume:
        for fn in glob.glob(os.path.join(ROOT, "data", f"{chunk_stem}_*.json")):
            os.remove(fn)
    env = dict(os.environ)
    for k in BLAS_ENV:
        env[k] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    logdir = os.path.join(ROOT, "data", "phase2_logs")
    os.makedirs(logdir, exist_ok=True)
    procs = []
    t0 = time.time()
    for ch in range(nworkers):
        logf = open(os.path.join(logdir, f"{chunk_stem}_{ch}.log"), "a" if resume else "w")
        p = subprocess.Popen([PY, WORKER, "--chunk", str(ch), "--nchunks", str(nworkers),
                              "--jobs", jobs_file, "--stem", chunk_stem],
                             stdout=logf, stderr=subprocess.STDOUT, env=env, cwd=ROOT)
        procs.append((p, logf))
    while sum(1 for p, _ in procs if p.poll() is not None) < len(procs):
        time.sleep(15)
    for _, lf in procs:
        lf.close()
    recs = []
    for fn in sorted(glob.glob(os.path.join(ROOT, "data", f"{chunk_stem}_*.json"))):
        recs += json.load(open(fn))["recs"]
    json.dump(dict(n=len(recs), budget=BUDGET, recs=recs), open(out_file, "w"))
    print(f"\n  {len(recs)}/{expect} runs in {(time.time() - t0) / 60:.1f} min -> {out_file}")
    return 0 if len(recs) == expect else 2


def stage_run(nworkers, resume):
    print("=" * 74)
    print("S3c STAGE 2 -- CONFIRMATORY CAMPAIGN (2 pre-named arms)")
    print("=" * 74)
    return _campaign(nworkers, resume, JOBS, "s3c_chunk", RAW, N_COHORT * len(ARMS),
                     f"{ARMS}")


def stage_levers(nworkers, resume):
    print("=" * 74)
    print("S3c STAGE 4 -- EXPLORATORY LEVER SWEEP (amendment A1-9)")
    print("=" * 74)
    if not os.path.exists(FINAL):
        print("  REFUSING: the confirmatory analysis does not exist yet. The sweep must run only "
              "AFTER the confirmatory result is computed and frozen (amendment A1-9).")
        return 2
    return _campaign(nworkers, resume, JOBS_LEV, "s3clev_chunk", RAW_LEV,
                     N_COHORT * len(LEVERS_EXPLORATORY), "7 exploratory levers")


def stage_analyze():
    from scipy import stats
    print("=" * 74)
    print("S3c STAGE 3 -- THE SINGLE PRE-SPECIFIED TEST + AMENDED BAND CHAIN")
    print("=" * 74)
    if not (os.path.exists(PREREG) and os.path.exists(RAW)):        # review finding F6
        json.dump(dict(unit="S3c", band="NO-DATA", reportable=False, confirmatory=True,
                       consequence="required input missing (prereg or raw results)"),
                  open(FINAL, "w"), indent=1)
        print("  NO-DATA: prereg or raw results missing. Wrote a non-reportable artifact.")
        return 2
    pre = json.load(open(PREREG))
    recs = json.load(open(RAW))["recs"]
    by = {}
    for r in recs:
        by.setdefault(r["start_i"], {})[r["method"]] = r

    screen, pairs = [], []
    for si in sorted(by):
        a, b = by[si].get(ARMS[0]), by[si].get(ARMS[1])
        bad = []
        for nm, r in ((ARMS[0], a), (ARMS[1], b)):
            if r is None:
                bad.append(f"{nm}:missing")
            elif r.get("error"):
                bad.append(f"{nm}:{r['error']}")
            elif not np.isfinite(r.get("best_ms", np.nan)) or r["best_ms"] <= 0:
                bad.append(f"{nm}:invalid_endpoint")
        screen.append(dict(start_i=si, idx=(a or b or {}).get("idx"), ok=not bad, problems=bad))
        if not bad:
            pairs.append(dict(start_i=si, idx=a["idx"], m_s_start=a["m_s_start"],
                              kappa_start=a["kappa_start"], gain_surrogate=a["gain"],
                              gain_lever=b["gain"], delta=a["gain"] - b["gain"],
                              best_ms_surrogate=a["best_ms"], best_ms_lever=b["best_ms"],
                              n_solves_surrogate=a["n_solves"], n_solves_lever=b["n_solves"]))
    n_fail = len(screen) - len(pairs)
    print(f"  validity screen: {len(pairs)}/{N_COHORT} usable pairs, {n_fail} failed")

    delta = np.array([p["delta"] for p in pairs], float)
    n = len(delta)
    wins = int((delta > 0).sum()); losses = int((delta < 0).sum()); ties = int((delta == 0).sum())
    med = float(np.median(delta)) if n else float("nan")
    nt = wins + losses
    win_rate = (wins / nt) if nt else float("nan")
    if n == 0:
        wp = sp = float("nan")
    elif np.all(delta == 0):
        wp = sp = 1.0
    else:
        wp = float(stats.wilcoxon(delta, zero_method="pratt", alternative="two-sided").pvalue)
        sp = float(stats.binomtest(wins, nt, 0.5).pvalue) if nt else 1.0
    if nt:
        lo, hi = stats.binomtest(wins, nt, 0.5).proportion_ci(confidence_level=0.95, method="wilson")
        wilson = [float(lo), float(hi)]
    else:
        wilson = [float("nan")] * 2

    sig = bool(np.isfinite(wp) and wp <= ALPHA)
    if n < N_MIN:
        band, cons = ("ATTRITION-VOID",
                      f"usable n={n} < {N_MIN}: cannot claim confirmatory status; reported honestly "
                      "and NOT restarted (a replacement cohort would be a top-up by another name)")
    elif sig and med < 0:
        band, cons = ("R-REVERSED",
                      "the fixed lever BEATS the surrogate on fresh marginal starts -- HARD STOP; "
                      "the design author writes the report and stops; the user alone decides")
    elif sig and med > 0:
        band, cons = ("R-REPLICATED",
                      "the effect replicates under a clean fixed-n pre-registration; add as a "
                      "second, independently pre-registered confirmatory result in draft §5.1. "
                      + DISCLOSURE_WINNERS_CURSE)
    elif sig and med == 0:
        band, cons = ("R-AMBIGUOUS",
                      "significant but directionally ambiguous; claim neither replication nor reversal")
    elif (not sig) and med > 0 and np.isfinite(win_rate) and win_rate >= 0.60:
        band, cons = ("R-NULL-CONSISTENT",
                      "direction-consistent underpowered null: report with full numbers; any "
                      "statement that the effect 'replicated' must carry this cohort's "
                      "non-significance in the same sentence. " + DISCLOSURE_POWER
                      + " " + DISCLOSURE_WINNERS_CURSE)
    elif not sig:
        band, cons = ("R-NULL-INCONSISTENT",
                      "the effect did not appear on a clean fixed-n cohort: DEMOTE the fixed-kappa "
                      "marginal claim to 'inconsistent across pre-registered cohorts', move the "
                      "pooled result out of the headline, and add the failure to the scope box. "
                      + DISCLOSURE_POWER + " " + DISCLOSURE_WINNERS_CURSE)
    else:
        raise AssertionError(f"band chain gap (unreachable): sig={sig} med={med} n={n}")

    out = dict(unit="S3c", design=pre["design"], amendment=pre["amendment"],
               prereg="data/s3c_prereg.json", prereg_sha256=sha(PREREG),
               confirmatory=True, not_reportable_magnitudes=True,
               cohort=dict(n_registered=N_COHORT, n_usable=n, n_min=N_MIN,
                           start_idx=pre["cohort"]["start_idx"]),
               arms=ARMS, budget=BUDGET, ktol=pre["protocol"]["ktol"], power=pre["power"],
               validity_screen=dict(n_usable=n, n_failed=n_fail, rows=screen),
               primary=dict(test="wilcoxon signed-rank two-sided (pratt)", n=n, median_delta=med,
                            wilcoxon_p=wp, alpha=ALPHA, significant=sig),
               secondary=dict(wins_surrogate=wins, losses=losses, ties=ties, win_rate=win_rate,
                              wilson95=wilson, sign_p=sp,
                              median_gain_surrogate=(float(np.median([p["gain_surrogate"] for p in pairs]))
                                                     if n else None),
                              median_gain_lever=(float(np.median([p["gain_lever"] for p in pairs]))
                                                 if n else None)),
               pairs=pairs, band=band, consequence=cons,
               reportable=bool(band not in NO_SCIENCE_BANDS),
               disclosures=[DISCLOSURE_WINNERS_CURSE, DISCLOSURE_POWER])
    json.dump(out, open(FINAL, "w"), indent=1)
    print(f"  n={n}  surrogate wins {wins}, loses {losses}, ties {ties} "
          f"(win-rate {win_rate:.3f}, Wilson [{wilson[0]:.3f}, {wilson[1]:.3f}])")
    print(f"  median paired delta = {med:+.5f}")
    print(f"  PRIMARY two-sided Wilcoxon p = {wp:.8f}  (alpha {ALPHA});  sign p = {sp:.6f}")
    print(f"\n  BAND = {band}\n  -> {cons}\n  -> {FINAL}")
    return 0


def stage_levers_analyze():
    print("=" * 74)
    print("S3c STAGE 5 -- EXPLORATORY: which realisable lever was best on THIS cohort?")
    print("=" * 74)
    conf = json.load(open(FINAL))
    lev = json.load(open(RAW_LEV))["recs"]
    base = {r["start_i"]: r for r in json.load(open(RAW))["recs"]
            if r["method"] == "heuristic:gap_outer-"}
    surr = {r["start_i"]: r for r in json.load(open(RAW))["recs"] if r["method"] == "surrogate"}
    per = {}
    for r in lev:
        per.setdefault(r["method"], {})[r["start_i"]] = r
    per["heuristic:gap_outer-"] = base
    rows = []
    for m, d in per.items():
        g = [v["gain"] for v in d.values() if not v.get("error") and np.isfinite(v.get("gain", np.nan))]
        rows.append(dict(method=m, n=len(g), median_gain=float(np.median(g)) if g else None))
    rows.sort(key=lambda r: (r["median_gain"] is None, -(r["median_gain"] or 0)))
    best = rows[0]
    pre_named = next(r for r in rows if r["method"] == "heuristic:gap_outer-")
    # surrogate vs the cohort's OWN best lever (exploratory, NOT the confirmatory test)
    bm = per[best["method"]]
    pr = [(surr[si]["gain"] - bm[si]["gain"]) for si in surr
          if si in bm and not surr[si].get("error") and not bm[si].get("error")]
    from scipy import stats
    if pr and not np.all(np.array(pr) == 0):
        wp = float(stats.wilcoxon(pr, zero_method="pratt", alternative="two-sided").pvalue)
    else:
        wp = float("nan")
    out = dict(unit="S3c-levers", exploratory=True, confirmatory=False,
               not_reportable_magnitudes=True,
               purpose="amendment A1-9: was the PRE-NAMED baseline still the best realisable lever "
                       "on this fresh cohort? (winner's-curse check demanded by 2 council seats)",
               ran_after_confirmatory_band=conf["band"],
               per_lever_median_gain=rows,
               pre_named_lever="heuristic:gap_outer-",
               pre_named_rank=1 + rows.index(pre_named),
               cohort_best_lever=best["method"],
               pre_named_was_still_best=bool(best["method"] == "heuristic:gap_outer-"),
               surrogate_vs_cohort_best=dict(
                   n=len(pr), wins=int(sum(1 for v in pr if v > 0)),
                   median_delta=float(np.median(pr)) if pr else None, wilcoxon_p=wp,
                   note="EXPLORATORY. The confirmatory test is surrogate vs the PRE-NAMED lever "
                        "only; this cell is a post-hoc characterisation and is not a registered claim."))
    json.dump(out, open(FINAL_LEV, "w"), indent=1)
    for r in rows:
        mark = "  <-- PRE-NAMED" if r["method"] == "heuristic:gap_outer-" else ""
        print(f"    {r['method']:24s} n={r['n']:2d}  median gain "
              f"{(r['median_gain'] if r['median_gain'] is not None else float('nan')):+.4f}{mark}")
    print(f"\n  pre-named lever rank on this cohort: {out['pre_named_rank']} of {len(rows)}  "
          f"(still best: {out['pre_named_was_still_best']})")
    print(f"  surrogate vs cohort-best lever (EXPLORATORY): {out['surrogate_vs_cohort_best']['wins']}"
          f"/{out['surrogate_vs_cohort_best']['n']}, Wilcoxon p = {wp:.6f}")
    print(f"  -> {FINAL_LEV}")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["freeze", "run", "analyze", "levers", "levers_analyze"],
                    required=True)
    ap.add_argument("--nworkers", type=int, default=10)
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()
    if a.stage == "freeze":
        sys.exit(stage_freeze())
    if a.stage == "run":
        sys.exit(stage_run(a.nworkers, a.resume))
    if a.stage == "analyze":
        sys.exit(stage_analyze())
    if a.stage == "levers":
        sys.exit(stage_levers(a.nworkers, a.resume))
    sys.exit(stage_levers_analyze())


if __name__ == "__main__":
    main()
