"""s4a_li_ablation.py -- unit S4a driver: zero-l_i ablation of the fixed-kappa design loop.

Frozen design: data/audit/strategy/S4A_S3C_DESIGN.md (unit S4a) + AMENDMENT A1 (council-before,
data/research/council/s4a_s3c_design/SYNTHESIS.md). Stages are explicit and ordered; the campaign
REFUSES to run until the determinism control has passed on ALL 20 starts.

  --stage smoke    : re-run the UNMODIFIED arm on all 20 starts; require bit-exact reproduction of
                     the banked gallery -> data/s4a_li_smoke.json
  --stage run      : ARM A (gradient mask) and ARM B (gradient + value mask) -> data/s4a_li_raw.json
  --stage analyze  : paired stats + the amended band chain -> data/s4a_li_ablation.json

Frozen statistics: two-sided Wilcoxon signed-rank (zero_method='pratt', the E1 convention) on the
paired differences gain_ablated - gain_full; two-sided sign test secondary; percentile bootstrap
(10k resamples, seed 20260731) for the median-difference CI used by the equivalence band.
Materiality delta* = 0.05 in internal-simulator delta-m_s -- justified as BELOW the measured ~10%
marginal-band grid systematic (draft §5.2), i.e. below the numerical resolution of the target.
"""
import argparse
import glob
import json
import os
import subprocess
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "experiments"))
PY = os.path.join(ROOT, "fusion-env", "Scripts", "python.exe")
WORKER = os.path.join(ROOT, "experiments", "s4a_li_worker.py")
BLAS_ENV = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
GALLERY = os.path.join(ROOT, "data", "phase4_gallery_results.json")
SMOKE_OUT = os.path.join(ROOT, "data", "s4a_li_smoke.json")
RAW_OUT = os.path.join(ROOT, "data", "s4a_li_raw.json")
FINAL_OUT = os.path.join(ROOT, "data", "s4a_li_ablation.json")
MATERIALITY = 0.05
ALPHA = 0.05
KTOL = 0.04
BOOT_N, BOOT_SEED = 10000, 20260731
N_STARTS = 20
EXACT_FIELDS = ["best_ms", "gain", "n_solves", "kappa_final", "kappa_drift", "m_s_start"]
SHAM_SEL = os.path.join(ROOT, "data", "s4a_sham_selection.json")
ARMS = {"A": ("ablate", "s4a_li_chunk_*.json"),
        "B": ("ablate_full", "s4a_lifull_chunk_*.json"),
        "S": ("sham", "s4a_sham_chunk_*.json")}
NO_SCIENCE_BANDS = {"E-HARNESS-FAIL", "F-ATTRITION"}


def stage_sham_select():
    """Amendment A1-12 (review finding F1): freeze the SHAM control channel from GRADIENT GEOMETRY ALONE --
    no solver runs, no outcome data. Pick the non-l_i descriptor whose masking perturbs the ascent
    direction by the amount closest to l_i's, so the negative control is perturbation-matched."""
    import numpy as _np
    import phase2_data as D
    import phase2_dim_lib as DL
    import phase25_kappa_lib as KL
    import phase2_model as M
    print("=" * 74)
    print("S4a STAGE 0 -- FREEZE THE SHAM CONTROL CHANNEL (no solves, no outcome data)")
    print("=" * 74)
    if os.path.exists(SHAM_SEL):
        print(f"  REFUSING: {SHAM_SEL} already frozen.")
        return 2
    S = json.load(open(os.path.join(ROOT, "data", "phase25_kappa_setup.json")))
    mu = _np.array(S["mu"]); std = _np.array(S["std"]); V = _np.array(S["V"])
    lo = _np.array(S["box_lo"]); hi = _np.array(S["box_hi"])
    models, _ = M.load_ensemble()
    smap, _ = M.load_shapemap()
    li_i = D.SHAPE_FEATURES.index("li")
    per_feat = {i: [] for i in range(len(D.SHAPE_FEATURES))}
    for s in S["starts"]:
        ds = DL.DesignSpace(mu, std, V, lo, hi, _np.array(s["u"]), 12)
        m = 0.05 * (ds.box_hi - ds.box_lo)
        ds.box_lo = _np.minimum(ds.box_lo, ds.x0 - m)
        ds.box_hi = _np.maximum(ds.box_hi, ds.x0 + m)
        gf = KL._grad_feature(None, smap, ds, ds.x0, None, is_ms=True, models=models)
        nf = _np.linalg.norm(gf)
        for i in per_feat:
            gm = KL._grad_ms_zeroed(models, smap, ds, ds.x0, i)
            per_feat[i].append(float(_np.dot(gf, gm) / (nf * _np.linalg.norm(gm) + 1e-12)))
    med = {i: float(_np.median(v)) for i, v in per_feat.items()}
    li_med = med[li_i]
    cand = sorted((abs(med[i] - li_med), i) for i in med if i != li_i)
    sham_i = cand[0][1]
    out = dict(unit="S4a", stage="sham_selection", amendment="A1-12 (council-before, review finding F1)",
               exploratory=True, confirmatory=False,
               rule="the non-l_i descriptor whose median cos(full, masked) over the 20 starts is "
                    "closest to l_i's -- i.e. perturbation-matched to the l_i ablation",
               computed_from="gradient geometry only; NO solver runs and NO outcome data existed",
               li_feat="li", li_feat_i=li_i, li_median_cos=li_med,
               sham_feat=D.SHAPE_FEATURES[sham_i], sham_feat_i=int(sham_i),
               sham_median_cos=med[sham_i], cos_gap=float(abs(med[sham_i] - li_med)),
               all_median_cos={D.SHAPE_FEATURES[i]: med[i] for i in sorted(med)})
    json.dump(out, open(SHAM_SEL, "w"), indent=1)
    print(f"  l_i  median cos(full, masked) = {li_med:.6f}")
    print(f"  SHAM = '{out['sham_feat']}' (index {sham_i}), median cos {med[sham_i]:.6f}, "
          f"gap {out['cos_gap']:.2e}")
    print(f"  -> {SHAM_SEL}   FROZEN before any ablation arm runs.")
    return 0


def spawn(nworkers, mode, resume):
    env = dict(os.environ)
    for k in BLAS_ENV:
        env[k] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    logdir = os.path.join(ROOT, "data", "phase2_logs")
    os.makedirs(logdir, exist_ok=True)
    procs = []
    for ch in range(nworkers):
        logf = open(os.path.join(logdir, f"s4a_{mode}_chunk_{ch}.log"), "a" if resume else "w")
        p = subprocess.Popen([PY, WORKER, "--chunk", str(ch), "--nchunks", str(nworkers),
                              "--mode", mode], stdout=logf, stderr=subprocess.STDOUT,
                             env=env, cwd=ROOT)
        procs.append((p, logf))
    t0 = time.time()
    while sum(1 for p, _ in procs if p.poll() is not None) < len(procs):
        time.sleep(10)
    for _, lf in procs:
        lf.close()
    return (time.time() - t0) / 60.0


def collect(pattern):
    recs = []
    for fn in sorted(glob.glob(os.path.join(ROOT, "data", pattern))):
        recs += json.load(open(fn))["recs"]
    return recs


def banked_surrogate():
    return {r["start_i"]: r for r in json.load(open(GALLERY))["recs"]
            if r["method"] == "surrogate"}


def stage_smoke(nworkers, resume):
    print("=" * 74)
    print("S4a STAGE 1 -- DETERMINISM CONTROL on ALL 20 starts (amendment A1 / DeepSeek F6)")
    print("=" * 74)
    mins = spawn(nworkers, "smoke", resume)
    got = {r["start_i"]: r for r in collect("s4a_smoke_chunk_*.json")}
    bank = banked_surrogate()
    rows, ok_all = [], True
    for si in sorted(bank):
        g, b = got.get(si), bank.get(si)
        if g is None:
            rows.append(dict(start_i=si, ok=False, reason="rerun missing"))
            ok_all = False
            continue
        diffs = {f: [b.get(f), g.get(f), bool(b.get(f) == g.get(f))] for f in EXACT_FIELDS}
        dsc = all(b["best_desc"].get(k) == g["best_desc"].get(k) for k in b["best_desc"])
        ok = all(v[2] for v in diffs.values()) and dsc
        ok_all &= ok
        rows.append(dict(start_i=si, ok=bool(ok), fields=diffs, best_desc_exact=bool(dsc)))
        print(f"  start {si:2d}: {'EXACT' if ok else 'MISMATCH'}  "
              f"banked gain {b['gain']:+.10f} | rerun {g['gain']:+.10f}")
    out = dict(unit="S4a", stage="smoke", exploratory=True, confirmatory=False,
               purpose="determinism control on all 20 starts: zero_feat_i=None must take the "
                       "byte-identical production path and reproduce the banked gallery exactly",
               n_checked=len(rows), rows=rows, all_exact=bool(ok_all), minutes=round(mins, 2))
    json.dump(out, open(SMOKE_OUT, "w"), indent=1)
    print(f"\n  -> {SMOKE_OUT}   all_exact={ok_all}  ({len(rows)} starts, {mins:.1f} min)")
    if not ok_all:
        print("  BAND E-HARNESS-FAIL -- HARD STOP. Pairing invalid; no science read.")
        return 2
    print("  determinism control PASSED on all 20 -- the campaign may run.")
    return 0


def stage_run(nworkers, resume):
    print("=" * 74)
    print("S4a STAGE 2 -- ABLATION CAMPAIGN (arm A: gradient mask; arm B: gradient + value mask)")
    print("=" * 74)
    sm = json.load(open(SMOKE_OUT)) if os.path.exists(SMOKE_OUT) else None
    if not (sm and sm.get("all_exact") and sm.get("n_checked") == N_STARTS):
        print("  REFUSING TO RUN: determinism control absent, partial, or failed.")
        return 2
    if not os.path.exists(SHAM_SEL):
        print("  REFUSING TO RUN: the sham control channel is not frozen (amendment A1-12). "
              "Run --stage sham_select first.")
        return 2
    total = {}
    for arm, (mode, pat) in ARMS.items():
        print(f"\n  --- arm {arm} ({mode}) ---")
        mins = spawn(nworkers, mode, resume)
        total[arm] = len(collect(pat))
        print(f"  arm {arm}: {total[arm]} runs in {mins:.1f} min")
    recs = {arm: collect(pat) for arm, (mode, pat) in ARMS.items()}
    json.dump(dict(budget=18, zero_feat="li", n=total, arms=list(ARMS),
                   sham=json.load(open(SHAM_SEL)), recs=recs), open(RAW_OUT, "w"))
    print(f"\n  -> {RAW_OUT}")
    return 0 if all(v == N_STARTS for v in total.values()) else 2


def boot_ci_median(x, n=BOOT_N, seed=BOOT_SEED):
    if len(x) == 0:
        return [float("nan")] * 2
    rng = np.random.default_rng(seed)
    meds = np.median(rng.choice(x, size=(n, len(x)), replace=True), axis=1)
    return [float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))]


def analyse_arm(abl, bank, smoke_ok, sham_material=None):
    """Screen -> paired stats -> the AMENDED ordered band chain.

    Every branch states its FULL condition explicitly (review finding F3): no band is reached by
    elimination, and an unreachable-else canary fails loudly if a future edit reopens a gap.
    sham_material: True/False for the l_i arms (the A1-12 negative-control gate); None for the
    sham arm itself, which is not gated on itself."""
    from scipy import stats
    screen, usable, invariant_violation = [], [], []
    for si in sorted(bank):
        a = abl.get(si)
        hard = []
        if a is None:
            hard.append("missing")
        elif a.get("error"):
            hard.append(f"error:{a['error']}")
        elif not np.isfinite(a.get("best_ms", np.nan)) or a.get("best_ms", 0) <= 0:
            hard.append("invalid_endpoint")
        # kappa drift is a HARNESS INVARIANT (impossible by construction), never an exclusion
        if a is not None and np.isfinite(a.get("kappa_drift", np.nan)) and a["kappa_drift"] > KTOL:
            invariant_violation.append(dict(start_i=si, kappa_drift=a["kappa_drift"]))
        screen.append(dict(start_i=si, ok=not hard, hard_failures=hard))
        if not hard:
            usable.append(si)
    n_fail = len(screen) - len(usable)

    pairs = []
    for si in usable:
        a, b = abl[si], bank[si]
        dc = a.get("dir_cos") or []
        pairs.append(dict(start_i=si, idx=a.get("idx"), regime=a.get("regime"),
                          m_s_start=b["m_s_start"], gain_full=b["gain"], gain_ablated=a["gain"],
                          delta=a["gain"] - b["gain"],
                          kappa_drift_full=b["kappa_drift"], kappa_drift_ablated=a["kappa_drift"],
                          dir_cos_median=(float(np.median([c[0] for c in dc])) if dc else None),
                          n_grad_evals=len(dc)))
    delta = np.array([p["delta"] for p in pairs], float)
    n = len(delta)
    wins = int((delta > 0).sum()); losses = int((delta < 0).sum()); ties = int((delta == 0).sum())
    med = float(np.median(delta)) if n else float("nan")
    ci = boot_ci_median(delta)
    if n == 0:
        wp = sp = float("nan")
    elif np.all(delta == 0):
        wp = sp = 1.0
    else:
        wp = float(stats.wilcoxon(delta, zero_method="pratt", alternative="two-sided").pvalue)
        sp = float(stats.binomtest(wins, wins + losses, 0.5).pvalue) if (wins + losses) else 1.0

    sig = bool(np.isfinite(wp) and wp <= ALPHA)
    equiv = bool((not sig) and np.isfinite(ci[0]) and ci[0] > -MATERIALITY and ci[1] < MATERIALITY)
    if (not smoke_ok) or invariant_violation:
        band, cons = ("E-HARNESS-FAIL",
                      "determinism control failed or a kappa-drift invariant was violated "
                      "(impossible by construction) -- no science read")
    elif n_fail >= 3 or n == 0:
        band, cons = ("F-ATTRITION", f"{n_fail}/{N_STARTS} hard failures -- no science read")
    elif sham_material is True:
        band, cons = ("N-PATH-CHAOS",
                      "the SHAM control (an equal-magnitude perturbation of an arbitrary channel) "
                      "is itself material, so the 18-step loop's path sensitivity dominates: NO "
                      "l_i-specific attribution is possible from this design, in either direction")
    elif sig and med == 0:
        band, cons = ("AMBIGUOUS-SIGN",
                      "significant test with a non-directional point estimate (median exactly 0): "
                      "escalate to the design author; do not auto-label weightless or load-bearing")
    elif sig and abs(med) < MATERIALITY:
        band, cons = ("T-SIGNIFICANT-TRIVIAL",
                      "a statistically detectable but IMMATERIAL effect: the 'nearly weightless' "
                      "claim SURVIVES and is restated with the measured effect size; do not retract")
    elif sig and med < 0:
        band, cons = ("B-LOAD-BEARING",
                      "the l_i gradient component IS materially used; RETRACT the 'nearly "
                      "weightless' sentence in draft §5.5.2 and rewrite around the measured effect")
    elif sig and med > 0:
        band, cons = ("C-HARMFUL",
                      "removing the l_i component materially IMPROVES the loop; report as a "
                      "measured method improvement AND as a contradiction of 'weightless'")
    elif (not sig) and equiv:
        band, cons = ("A-WEIGHTLESS",
                      "equivalence: not significant AND the 95% CI of the median paired difference "
                      "lies wholly inside +-delta*; UPGRADE draft §5.5.2 from argued to demonstrated")
    elif not sig:
        band, cons = ("D-INCONCLUSIVE",
                      "not significant but the effect is not bounded inside +-delta*; do NOT "
                      "upgrade the claim; report the ablation as inconclusive at this n")
    else:
        raise AssertionError(f"band chain gap (unreachable): sig={sig} med={med} ci={ci}")

    allc = [c[0] for si in usable for c in (abl[si].get("dir_cos") or [])]
    return dict(
        validity_screen=dict(n_usable=n, n_hard_failures=n_fail, rows=screen,
                             kappa_invariant_violations=invariant_violation),
        stats=dict(n=n, wins_ablated_better=wins, losses=losses, ties=ties, median_delta=med,
                   median_delta_ci95=ci, mean_delta=float(np.mean(delta)) if n else float("nan"),
                   wilcoxon_p=wp, sign_p=sp,
                   median_gain_full=float(np.median([p["gain_full"] for p in pairs])) if n else None,
                   median_gain_ablated=float(np.median([p["gain_ablated"] for p in pairs])) if n else None),
        direction_diagnostic=dict(n_gradient_evals=len(allc),
                                  cos_median=float(np.median(allc)) if allc else None,
                                  cos_min=float(np.min(allc)) if allc else None,
                                  cos_max=float(np.max(allc)) if allc else None),
        pairs=pairs, band=band, consequence=cons,
        reportable=bool(band not in NO_SCIENCE_BANDS))


def stage_analyze():
    print("=" * 74)
    print("S4a STAGE 3 -- PAIRED ANALYSIS + AMENDED BAND CHAIN")
    print("=" * 74)
    # review finding F6: never analyse missing or stale data by accident.
    for f in (RAW_OUT, SMOKE_OUT, SHAM_SEL):
        if not os.path.exists(f):
            json.dump(dict(unit="S4a", band="NO-DATA", reportable=False,
                           consequence=f"required input missing: {f}"),
                      open(FINAL_OUT, "w"), indent=1)
            print(f"  NO-DATA: required input missing ({f}). Wrote a non-reportable artifact.")
            return 2
    raw = json.load(open(RAW_OUT))
    smoke = json.load(open(SMOKE_OUT))
    if smoke.get("n_checked") != N_STARTS or not smoke.get("all_exact"):
        json.dump(dict(unit="S4a", band="E-HARNESS-FAIL", reportable=False,
                       consequence="determinism control absent, partial or failed"),
                  open(FINAL_OUT, "w"), indent=1)
        print("  E-HARNESS-FAIL: determinism control not clean. No science read.")
        return 2
    bank = banked_surrogate()
    # the SHAM arm is analysed FIRST and gates the two l_i arms (amendment A1-12)
    res = {"S": analyse_arm({r["start_i"]: r for r in raw["recs"]["S"]}, bank,
                            bool(smoke.get("all_exact")), sham_material=None)}
    ss = res["S"]["stats"]
    sham_material = bool((np.isfinite(ss["wilcoxon_p"]) and ss["wilcoxon_p"] <= ALPHA)
                         or abs(ss["median_delta"]) >= MATERIALITY)
    print(f"\n  SHAM control ('{raw['sham']['sham_feat']}', perturbation-matched): "
          f"median delta {ss['median_delta']:+.5f}, Wilcoxon p {ss['wilcoxon_p']:.6f} "
          f"-> material={sham_material}")
    for arm in ("A", "B"):
        abl = {r["start_i"]: r for r in raw["recs"][arm]}
        res[arm] = analyse_arm(abl, bank, bool(smoke.get("all_exact")),
                               sham_material=sham_material)
    for arm in ("A", "B", "S"):
        s = res[arm]["stats"]
        print(f"\n  ARM {arm}: n={s['n']}  ablated-better {s['wins_ablated_better']}, "
              f"worse {s['losses']}, tied {s['ties']}")
        print(f"    median delta = {s['median_delta']:+.5f}  CI95 "
              f"[{s['median_delta_ci95'][0]:+.5f}, {s['median_delta_ci95'][1]:+.5f}]  "
              f"(materiality {MATERIALITY})")
        print(f"    Wilcoxon p = {s['wilcoxon_p']:.6f}   sign p = {s['sign_p']:.6f}")
        d = res[arm]["direction_diagnostic"]
        if d["cos_median"] is not None:
            print(f"    cos(full, masked): median {d['cos_median']:.6f} "
                  f"[{d['cos_min']:.6f}, {d['cos_max']:.6f}] over {d['n_gradient_evals']} evals")
        print(f"    BAND = {res[arm]['band']}")

    out = dict(unit="S4a", design="data/audit/strategy/S4A_S3C_DESIGN.md",
               amendment="A1 (council-before, data/research/council/s4a_s3c_design/SYNTHESIS.md)",
               exploratory=True, confirmatory=False, not_reportable_magnitudes=True,
               question="does zeroing the l_i channel of the surrogate's descriptor-space gradient "
                        "change the fixed-kappa design-loop outcome?",
               arm_definitions=dict(
                   A="gradient l_i channel zeroed; line search scores with the FULL surrogate "
                     "(tests the GRADIENT channel = the manuscript's actual claim) -- PRIMARY",
                   B="gradient l_i channel zeroed AND line search scores with l_i pinned at its "
                     "start value (the 'pure' ablation required by 3 council seats) -- SECONDARY",
                   S="SHAM negative control (amendment A1-12): the same masking applied to a "
                     "perturbation-matched NON-l_i channel, to measure the loop's path-sensitivity "
                     "noise floor. Frozen from gradient geometry before any arm ran."),
               sham_selection=raw.get("sham"),
               sham_material=sham_material,
               scope_note="S4a arms A/B scope: l_i's contribution to gradient DIRECTION selection. "
                          "The accept/reject gate always sees the true unmasked l_i, so no verdict "
                          "here speaks to l_i's importance to the model in general (review finding F11).",
               zero_feat="li", zero_feat_i=18, budget=18, ktol=KTOL, n_starts=N_STARTS,
               frozen=dict(alpha=ALPHA, materiality=MATERIALITY,
                           materiality_justification="below the measured ~10% marginal-band grid "
                           "systematic (draft §5.2): a difference this small is below the numerical "
                           "resolution of the target itself",
                           primary_test="wilcoxon signed-rank two-sided (pratt)",
                           secondary_test="two-sided sign test",
                           equivalence="A-WEIGHTLESS requires the bootstrap 95% CI of the median "
                                       "paired difference to lie wholly inside +-materiality",
                           bootstrap=dict(n=BOOT_N, seed=BOOT_SEED, method="percentile"),
                           n_fixed_in_advance=True, top_ups_permitted=False),
               determinism_control=dict(all_exact=bool(smoke.get("all_exact")),
                                        n_checked=smoke.get("n_checked")),
               arms=res, band=res["A"]["band"], consequence=res["A"]["consequence"],
               band_arm_B=res["B"]["band"], band_sham=res["S"]["band"],
               reportable=bool(res["A"]["band"] not in NO_SCIENCE_BANDS))
    json.dump(out, open(FINAL_OUT, "w"), indent=1)
    print(f"\n  UNIT BAND (arm A, primary) = {res['A']['band']}")
    print(f"  -> {res['A']['consequence']}")
    print(f"  arm B (pure ablation) band = {res['B']['band']}")
    print(f"  sham control band          = {res['S']['band']}")
    print(f"  -> {FINAL_OUT}")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["sham_select", "smoke", "run", "analyze"], required=True)
    ap.add_argument("--nworkers", type=int, default=10)
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()
    if a.stage == "sham_select":
        sys.exit(stage_sham_select())
    if a.stage == "smoke":
        sys.exit(stage_smoke(a.nworkers, a.resume))
    if a.stage == "run":
        sys.exit(stage_run(a.nworkers, a.resume))
    sys.exit(stage_analyze())


if __name__ == "__main__":
    main()
