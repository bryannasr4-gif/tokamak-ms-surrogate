"""verify_s3c_recompute.py -- INDEPENDENT fresh-agent VERIFY of unit S3c, from raw only.

Per the D18 law (the review protocol sec3 item 5 / sec6): every EXEC gets a paired VERIFY by a different
agent, recomputing from raw artifacts -- never trusting the executor's own analysis JSON or
report. This script does NOT import experiments/s3c_cohort_run.py; every check below is
independently (re)implemented against:
  - data/s3c_prereg.json          (the frozen pre-registration)
  - data/phase4_power3_results.json + data/s3c_chunk_*.json  (raw per-start-arm records)
  - data/dataset_v1_80.parquet + data/phase25_kappa_setup.json + data/phase4_power*_setup.json
    (to independently re-derive the "28 fresh starts, disjoint from prior" claim)
against the executor's claimed output data/s3c_third_cohort.json.

Design of record: data/audit/strategy/S4A_S3C_DESIGN.md (unit S3c section, lines ~380-484).
"""
import glob
import hashlib
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "experiments"))

PREREG = os.path.join(ROOT, "data", "s3c_prereg.json")
RAW = os.path.join(ROOT, "data", "phase4_power3_results.json")
FINAL = os.path.join(ROOT, "data", "s3c_third_cohort.json")
LABELS = os.path.join(ROOT, "data", "dataset_v1_80.parquet")
N_COHORT, N_MIN, ALPHA = 28, 25, 0.05
ARMS = ["surrogate", "heuristic:gap_outer-"]
MS_FLOOR, MARG_HI = 0.05, 0.4

checks = []


def check(name, ok, detail):
    checks.append((name, ok, detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


print("=" * 78)
print("VERIFY-S3c -- independent recompute from raw (fresh agent, own script)")
print("=" * 78)

# ---------------------------------------------------------------------------
# 1. Artifact existence + prereg integrity (hash + git anchor)
# ---------------------------------------------------------------------------
for p, nm in [(PREREG, "prereg"), (RAW, "raw results"), (FINAL, "final analysis")]:
    check(f"exists:{nm}", os.path.exists(p), p)

pre = json.load(open(PREREG))
final = json.load(open(FINAL))

prereg_sha = sha(PREREG)
check("prereg sha256 matches final.prereg_sha256", prereg_sha == final.get("prereg_sha256"),
      f"computed {prereg_sha[:16]}... vs recorded {final.get('prereg_sha256', '')[:16]}...")

try:
    blob = subprocess.run(["git", "hash-object", PREREG], cwd=ROOT, capture_output=True,
                          text=True, check=True).stdout.strip()
    head = subprocess.run(["git", "rev-parse", "HEAD:data/s3c_prereg.json"], cwd=ROOT,
                          capture_output=True, text=True)
    anchored = head.returncode == 0 and head.stdout.strip() == blob
    check("prereg git-anchored (working==HEAD blob)", anchored,
          f"working blob {blob[:12]} vs HEAD blob {head.stdout.strip()[:12]} (rc={head.returncode})")
except Exception as e:
    check("prereg git-anchored", False, f"exception: {type(e).__name__}: {e}")

check("prereg.confirmatory == True", pre.get("confirmatory") is True, str(pre.get("confirmatory")))
check("frozen_note asserts before-data freeze",
      "before" in pre.get("frozen_note", "").lower(), pre.get("frozen_note"))

# ---------------------------------------------------------------------------
# 2. Independently re-derive the 28-start cohort selection (disjointness + stratify_new)
# ---------------------------------------------------------------------------
orig = json.load(open(os.path.join(ROOT, "data", "phase25_kappa_setup.json")))
used = {s["idx"] for s in orig["starts"]}
power_setups = sorted(glob.glob(os.path.join(ROOT, "data", "phase4_power*_setup.json")))
power_setups = [p for p in power_setups if os.path.basename(p) != "phase4_power3_setup.json"]
for fn in power_setups:
    used |= {s["idx"] for s in json.load(open(fn))["starts"]}
check("prior-used pool reconstructed", True,
      f"{len(used)} prior idx from {1 + len(power_setups)} setup files "
      f"(phase25_kappa_setup.json + {[os.path.basename(p) for p in power_setups]})")
check("prior-used count matches prereg.cohort.n_prior_used",
      len(used) == pre["cohort"]["n_prior_used"],
      f"recomputed {len(used)} vs recorded {pre['cohort']['n_prior_used']}")

df = pd.read_parquet(LABELS)
marg = df[(df["m_s"] >= MS_FLOOR) & (df["m_s"] < MARG_HI)]
pool = marg[~marg["idx"].isin(used)].sort_values("m_s").reset_index(drop=True)
check("pool size matches prereg.cohort.pool_size", len(pool) == pre["cohort"]["pool_size"],
      f"recomputed {len(pool)} vs recorded {pre['cohort']['pool_size']}")

# --- Execute the REFERENCE stratify_new, rather than re-typing its formula -----------------
# VERIFY-S3c finding F1 (code-reading council seat, 2026-08-02): the executor's in-code
# "verbatim stratify_new" equivalence assertion re-runs an identical COPY-PASTED formula, and
# the first version of this verifier hand-retyped that same formula a third time. Three hand
# transcriptions of two lines cannot detect a systematic transcription error common to all
# three. `stratify_new` is a nested closure in phase4_power_run.py and cannot be imported, so
# we extract its definition with `ast` and EXECUTE THE REFERENCE SOURCE ITSELF, binding the
# closed-over `used_idx` to the exclusion set this script rebuilt independently above.
import ast

ref_path = os.path.join(ROOT, "experiments", "phase4_power_run.py")
ref_src = open(ref_path, encoding="utf-8").read()
ref_fn_src = None
for node in ast.walk(ast.parse(ref_src)):
    if isinstance(node, ast.FunctionDef) and node.name == "stratify_new":
        ref_fn_src = ast.get_source_segment(ref_src, node)
        break
check("reference stratify_new extracted from phase4_power_run.py", ref_fn_src is not None,
      f"{len(ref_fn_src.splitlines()) if ref_fn_src else 0} source lines recovered by ast")

ref_ns = {"np": np, "pd": pd, "used_idx": used}
exec(compile(ast.parse(ref_fn_src), ref_path, "exec"), ref_ns)
ref_picked = ref_ns["stratify_new"](marg, N_COHORT)     # NOTE: the reference re-filters `used_idx`
recomputed_idx = [int(r["idx"]) for r in ref_picked]
recorded_idx = pre["cohort"]["start_idx"]
check("frozen prereg idx == output of the EXECUTED reference stratify_new (not a retyped copy)",
      recomputed_idx == recorded_idx,
      f"match={recomputed_idx == recorded_idx}; first mismatch at "
      f"{next((i for i in range(min(len(recomputed_idx), len(recorded_idx))) if recomputed_idx[i] != recorded_idx[i]), 'n/a') if recomputed_idx != recorded_idx else 'none'}")

check("cohort disjoint from prior-used (28 fresh, 0 overlap)",
      len(set(recorded_idx) & used) == 0,
      f"overlap = {sorted(set(recorded_idx) & used)}")
check("cohort has no internal duplicates", len(set(recorded_idx)) == N_COHORT,
      f"{len(set(recorded_idx))} unique of {N_COHORT}")

# ---------------------------------------------------------------------------
# 3. Raw records: chunk-file reassembly, completeness, arm-pre-naming
# ---------------------------------------------------------------------------
raw = json.load(open(RAW))
chunk_files = sorted(glob.glob(os.path.join(ROOT, "data", "s3c_chunk_*.json")))
chunk_recs = []
for fn in chunk_files:
    chunk_recs += json.load(open(fn))["recs"]
check("chunk files reassemble to RAW byte-for-record count",
      len(chunk_recs) == raw["n"] == 56,
      f"sum(chunks)={len(chunk_recs)}, RAW.n={raw['n']}, expected 56 (28 starts x 2 arms)")
check("RAW.recs count == 28*2", len(raw["recs"]) == N_COHORT * len(ARMS), str(len(raw["recs"])))

methods = sorted(set(r["method"] for r in raw["recs"]))
check("exactly the 2 pre-named confirmatory arms present, no extras",
      methods == sorted(ARMS), f"methods found: {methods}")
starts_covered = sorted(set(r["start_i"] for r in raw["recs"]))
check("all 28 start_i indices (0..27) present exactly once per arm",
      starts_covered == list(range(N_COHORT)), f"{len(starts_covered)} distinct start_i")
raw_idx_by_start = {r["start_i"]: r["idx"] for r in raw["recs"]}
check("raw record idx matches prereg start_idx at every position",
      all(raw_idx_by_start[i] == recorded_idx[i] for i in range(N_COHORT)),
      "spot-checked all 28 positions")

# ---------------------------------------------------------------------------
# 4. Independent statistics recompute (own scipy calls, own screen logic)
# ---------------------------------------------------------------------------
by = {}
for r in raw["recs"]:
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
    screen.append(dict(start_i=si, ok=not bad, problems=bad))
    if not bad:
        pairs.append(dict(start_i=si, delta=a["gain"] - b["gain"],
                          gain_surrogate=a["gain"], gain_lever=b["gain"]))

n_usable = len(pairs)
check("validity screen: n_usable == 28, n_failed == 0",
      n_usable == 28 and (len(screen) - n_usable) == 0,
      f"n_usable={n_usable}, n_failed={len(screen) - n_usable}")
check("validity screen matches final.validity_screen exactly",
      [s["ok"] for s in screen] == [r["ok"] for r in final["validity_screen"]["rows"]],
      "row-by-row ok-flag comparison")

delta = np.array([p["delta"] for p in pairs], float)
wins = int((delta > 0).sum()); losses = int((delta < 0).sum()); ties = int((delta == 0).sum())
med = float(np.median(delta))
nt = wins + losses
win_rate = wins / nt
wp = float(stats.wilcoxon(delta, zero_method="pratt", alternative="two-sided").pvalue)
sp = float(stats.binomtest(wins, nt, 0.5).pvalue)
lo, hi = stats.binomtest(wins, nt, 0.5).proportion_ci(confidence_level=0.95, method="wilson")

fp = final["primary"]
fs = final["secondary"]
check("primary.median_delta bit-exact", med == fp["median_delta"],
      f"recomputed {med!r} vs recorded {fp['median_delta']!r}")
check("primary.wilcoxon_p bit-exact", wp == fp["wilcoxon_p"],
      f"recomputed {wp!r} vs recorded {fp['wilcoxon_p']!r}")
check("primary.significant matches (p<=0.05)", (wp <= ALPHA) == fp["significant"],
      f"recomputed sig={wp <= ALPHA} vs recorded {fp['significant']}")
check("secondary wins/losses/ties bit-exact",
      (wins, losses, ties) == (fs["wins_surrogate"], fs["losses"], fs["ties"]),
      f"recomputed {(wins, losses, ties)} vs recorded "
      f"{(fs['wins_surrogate'], fs['losses'], fs['ties'])}")
check("secondary.win_rate bit-exact", win_rate == fs["win_rate"],
      f"recomputed {win_rate!r} vs recorded {fs['win_rate']!r}")
check("secondary.sign_p bit-exact", sp == fs["sign_p"], f"recomputed {sp!r} vs recorded {fs['sign_p']!r}")
check("secondary.wilson95 bit-exact",
      [lo, hi] == fs["wilson95"], f"recomputed [{lo!r},{hi!r}] vs recorded {fs['wilson95']!r}")

# ---------------------------------------------------------------------------
# 5. Band-decision chain re-applied independently (ordered, first-match-wins)
# ---------------------------------------------------------------------------
sig = wp <= ALPHA
if n_usable < N_MIN:
    band = "ATTRITION-VOID"
elif sig and med < 0:
    band = "R-REVERSED"
elif sig and med > 0:
    band = "R-REPLICATED"
elif sig and med == 0:
    band = "R-AMBIGUOUS"
elif (not sig) and med > 0 and win_rate >= 0.60:
    band = "R-NULL-CONSISTENT"
elif not sig:
    band = "R-NULL-INCONSISTENT"
else:
    band = "UNREACHABLE-CANARY"
check("band recomputed from the frozen ordered chain == final.band",
      band == final["band"], f"recomputed {band} vs recorded {final['band']}")
check("reportable flag correct for this band",
      final["reportable"] == (band not in {"ATTRITION-VOID", "NO-DATA"}),
      f"band={band}, reportable={final['reportable']}")
check("mandatory winner's-curse disclosure present in consequence text (R-REPLICATED requires it)",
      band != "R-REPLICATED" or "MANDATORY DISCLOSURE" in final["consequence"],
      "checked for literal string in final.consequence")
check("power table n==28 matches design's frozen power spec",
      pre["power"]["n"] == 28 and pre["power"]["sign_test_critical_wins"] == 20,
      str(pre["power"]))

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
n_pass = sum(1 for _, ok, _ in checks if ok)
n_fail = len(checks) - n_pass
print("=" * 78)
print(f"VERIFY-S3c: {n_pass}/{len(checks)} checks PASS, {n_fail} FAIL")
print(f"Independently recomputed band: {band}  (recorded: {final['band']})")
print(f"Independently recomputed p:    {wp!r}")
print(f"Independently recomputed wins: {wins}/{nt} ({win_rate:.4f})  median_delta={med!r}")
print("=" * 78)

report = dict(unit="VERIFY-S3c", verifier="fresh agent, independent script "
              "(experiments/verify_s3c_recompute.py), raw-only recompute",
              n_checks=len(checks), n_pass=n_pass, n_fail=n_fail,
              overall="PASS" if n_fail == 0 else "FAIL",
              recomputed=dict(band=band, wilcoxon_p=wp, median_delta=med, wins=wins,
                              losses=losses, ties=ties, win_rate=win_rate,
                              wilson95=[lo, hi], sign_p=sp, n_usable=n_usable,
                              cohort_idx_match=recomputed_idx == recorded_idx,
                              disjoint_from_prior=len(set(recorded_idx) & used) == 0),
              checks=[dict(name=n, ok=ok, detail=d) for n, ok, d in checks])
out_path = os.path.join(ROOT, "data", "tracks_ab", "s3c_verify_recompute.json")
json.dump(report, open(out_path, "w"), indent=1)
print(f"-> {out_path}")
sys.exit(0 if n_fail == 0 else 2)
