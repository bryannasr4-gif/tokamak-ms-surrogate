"""s34_preflight.py -- §3 verify-before-run preflight for units S4a (zero-l_i ablation) and
S3c (fixed-n third cohort). Proves every referenced input exists and matches, and that every
gate criterion is SATISFIABLE and arithmetically well-defined against the REAL data shapes,
BEFORE either design is frozen. Runs no solver and writes no experiment artifact.

  fusion-env/Scripts/python.exe experiments/s34_preflight.py
"""
import glob
import hashlib
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "experiments"))

FAIL = []
def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{(' -- ' + detail) if detail else ''}")
    if not ok:
        FAIL.append(name)
    return ok


def sha(path, n=16):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()[:n]


print("=" * 78)
print("S3c/S4a PREFLIGHT -- §3 verify-before-run")
print("=" * 78)

# ---------------------------------------------------------------- 1. inputs exist
print("\n1. REFERENCED INPUTS EXIST + HASHES")
need = [
    "data/dataset_v1_80.parquet",          # labels file used by the prior power batches
    "data/dataset_v1_80q.parquet",         # canonical (analysis)
    "data/phase2_dim_setup.json",          # PCA control space (mu/std/V/box)
    "data/phase25_kappa_setup.json",       # the ORIGINAL 20 starts
    "data/phase4_power_setup.json",        # top-up batch 1 (12)
    "data/phase4_power2_setup.json",       # top-up batch 2 (24)
    "data/phase4_gallery_results.json",    # the 20-start gallery (S4a pairing partner)
    "data/phase4_kappa_pooled.json",       # pooled stats + best realisable lever
    "experiments/phase25_kappa_lib.py",
    "experiments/phase4_gallery_lib.py",
    "experiments/phase4_power_run.py",
    "experiments/phase4_power_worker.py",
]
for p in need:
    fp = os.path.join(ROOT, p)
    ok = os.path.exists(fp)
    check(p, ok, (f"sha256[:16]={sha(fp)} size={os.path.getsize(fp)}" if ok else "MISSING"))

# ---------------------------------------------------------------- 2. descriptor indices
print("\n2. DESCRIPTOR / FEATURE INDICES (the S4a ablation target)")
import phase2_data as D
SF = list(D.SHAPE_FEATURES)
check("SHAPE_FEATURES has 20 entries", len(SF) == 20, f"n={len(SF)}")
check("'li' present in SHAPE_FEATURES", "li" in SF,
      f"index={SF.index('li') if 'li' in SF else 'NA'}")
check("'kappa' present", "kappa" in SF, f"index={SF.index('kappa') if 'kappa' in SF else 'NA'}")
check("'gap_outer' present (S3c lever)", "gap_outer" in SF,
      f"index={SF.index('gap_outer') if 'gap_outer' in SF else 'NA'}")
print(f"  SHAPE_FEATURES = {SF}")

# ---------------------------------------------------------------- 3. S3c: fresh-start pool
print("\n3. S3c GATE SATISFIABILITY -- are there >=28 FRESH marginal starts?")
import pandas as pd
MS_FLOOR, MARG_HI, MID_HI = 0.05, 0.4, 1.0        # verbatim from phase4_power_run.py
labels = os.path.join(ROOT, "data", "dataset_v1_80.parquet")
df = pd.read_parquet(labels)
check("labels parquet has an 'm_s' column", "m_s" in df.columns, f"cols={len(df.columns)}")
check("labels parquet has an 'idx' column", "idx" in df.columns, f"rows={len(df)}")

used = set()
orig = json.load(open(os.path.join(ROOT, "data", "phase25_kappa_setup.json")))
used |= {s["idx"] for s in orig["starts"]}
n_orig = len(orig["starts"])
per_batch = {}
for fn in sorted(glob.glob(os.path.join(ROOT, "data", "phase4_power*_setup.json"))):
    b = json.load(open(fn))
    ids = {s["idx"] for s in b["starts"]}
    per_batch[os.path.basename(fn)] = len(ids)
    used |= ids
print(f"  original setup starts: {n_orig}; prior power batches: {per_batch}")
check("total prior starts == 56 (20 + 12 + 24, the pooled n)", len(used) == 56, f"len(used)={len(used)}")

marg = df[(df["m_s"] >= MS_FLOOR) & (df["m_s"] < MARG_HI)]
mid = df[(df["m_s"] >= MARG_HI) & (df["m_s"] < MID_HI)]
fresh_marg = marg[~marg["idx"].isin(used)]
print(f"  marginal band [0.05,0.4): {len(marg)} rows total; {len(fresh_marg)} FRESH (unused)")
print(f"  mid band      [0.4,1.0):  {len(mid)} rows total")
check("S3c needs >=28 fresh marginal starts", len(fresh_marg) >= 28,
      f"available={len(fresh_marg)} (headroom {len(fresh_marg)-28})")

# reproduce the exact stratify_new selection the builder will make (n=28), and prove
# the picks are disjoint, unique and inside the band -- the G-S0.5-3 lesson.
pool = fresh_marg.sort_values("m_s").reset_index(drop=True)
qs = np.linspace(0.025, 0.975, 28)
sel = sorted(set(int(round(q * (len(pool) - 1))) for q in qs))
check("stratified selection yields 28 DISTINCT rows (no quantile collision)", len(sel) >= 28,
      f"distinct positions={len(sel)} (collisions would silently shrink n)")
picked = pool.iloc[sel[:28]]
check("all 28 picks disjoint from prior starts", not set(picked["idx"]) & used,
      f"overlap={len(set(picked['idx']) & used)}")
check("all 28 picks inside the marginal band", bool(((picked["m_s"] >= MS_FLOOR) &
      (picked["m_s"] < MARG_HI)).all()), f"m_s range [{picked['m_s'].min():.4f}, {picked['m_s'].max():.4f}]")
check("all 28 picks have finite kappa", bool(np.isfinite(picked["kappa"]).all()),
      f"kappa range [{picked['kappa'].min():.3f}, {picked['kappa'].max():.3f}]")

# ---------------------------------------------------------------- 4. S3c: the named lever
print("\n4. S3c ARM CHOICE -- is 'gap_outer-' the pre-registered best REALISABLE fixed lever?")
pooled = json.load(open(os.path.join(ROOT, "data", "phase4_kappa_pooled.json")))
bfl = pooled.get("best_fixed_lever")
lev = pooled.get("lever_pooled_median", {})
top = sorted(lev.items(), key=lambda kv: -kv[1])[:4]
print(f"  best_fixed_lever (banked) = {bfl}; top levers by pooled median = {top}")
check("banked best realisable lever is gap_outer-", bfl == "heuristic:gap_outer-" or bfl == "gap_outer-",
      f"value={bfl!r}")
check("n_levers == 8 (the realisable menu)", pooled.get("n_levers") == 8, f"n={pooled.get('n_levers')}")

# ---------------------------------------------------------------- 5. S4a: pairing partner
print("\n5. S4a PAIRING -- the 20 recorded gallery surrogate runs")
gal = json.load(open(os.path.join(ROOT, "data", "phase4_gallery_results.json")))
recs = gal["recs"]
surr = [r for r in recs if r["method"] == "surrogate"]
check("gallery has exactly 20 surrogate runs", len(surr) == 20, f"n={len(surr)}")
check("every surrogate run records best_u (endpoint controls)", all("best_u" in r for r in surr))
check("every surrogate run records idx + start", all(("idx" in r and "start_i" in r) for r in surr))
check("gallery budget is 18", gal.get("budget", 18) == 18, f"budget={gal.get('budget')}")
gs = [r["gain"] for r in surr]
print(f"  banked surrogate gains: median {np.median(gs):+.4f}, range [{min(gs):+.4f}, {max(gs):+.4f}]")
kd = [r["kappa_drift"] for r in surr]
check("all 20 gallery kappa drifts <= KTOL 0.04", max(kd) <= 0.04, f"max drift={max(kd):.6f}")

# ---------------------------------------------------------------- 6. models load
print("\n6. MODELS + MACHINE LOAD (no solve)")
try:
    import phase2_model as M
    models, _ = M.load_ensemble()
    smap, _ = M.load_shapemap()
    check("ensemble loads", len(models) == 8, f"n_members={len(models)}")
    check("shapemap loads", smap is not None)
except Exception as e:
    check("models load", False, f"{type(e).__name__}: {e}")

# ---------------------------------------------------------------- 7. masked-gradient mechanics
print("\n7. S4a MECHANISM -- masked chain rule is well-defined and CHANGES the direction")
try:
    import torch
    import phase2_dim_lib as DL
    import phase25_kappa_lib as KL
    setup = json.load(open(os.path.join(ROOT, "data", "phase2_dim_setup.json")))
    s0 = orig["starts"][0]
    ds = DL.DesignSpace(np.array(setup["mu"]), np.array(setup["std"]), np.array(setup["V"]),
                        np.array(setup["box_lo"]), np.array(setup["box_hi"]),
                        np.array(s0["u"]), 12)
    x = torch.tensor(ds.x0, dtype=torch.float32, requires_grad=True)
    LI_I = SF.index("li")
    # full gradient (identical to KL._grad_feature is_ms=True)
    g_full = KL._grad_feature(None, smap, ds, ds.x0, None, is_ms=True, models=models)
    # masked: zero the l_i CHANNEL of d(log m_s)/d(descriptors), then VJP back to x
    gs = []
    for m in models:
        xt = torch.tensor(ds.x0, dtype=torch.float32, requires_grad=True)
        d = smap(ds.u_of_x_t(xt).unsqueeze(0))
        out = m(d)[0, 0]
        g_d, = torch.autograd.grad(out, d, retain_graph=True, create_graph=False)
        g_d = g_d.clone()
        g_d[0, LI_I] = 0.0
        g_x, = torch.autograd.grad(d, xt, grad_outputs=g_d, retain_graph=False)
        gs.append(g_x.numpy())
    g_mask = np.mean(gs, 0)
    cos = float(np.dot(g_full, g_mask) / (np.linalg.norm(g_full) * np.linalg.norm(g_mask) + 1e-12))
    check("masked gradient is finite + non-degenerate", bool(np.all(np.isfinite(g_mask))
          and np.linalg.norm(g_mask) > 1e-9), f"||g_mask||={np.linalg.norm(g_mask):.5f}")
    check("masked != full (the ablation actually does something)", cos < 0.999999,
          f"cos(full, masked) = {cos:.8f}")
    print(f"  ||g_full||={np.linalg.norm(g_full):.5f}  ||g_mask||={np.linalg.norm(g_mask):.5f}")
    # sanity: an UNMASKED VJP must reproduce KL._grad_feature exactly (validates the VJP path)
    gs2 = []
    for m in models:
        xt = torch.tensor(ds.x0, dtype=torch.float32, requires_grad=True)
        d = smap(ds.u_of_x_t(xt).unsqueeze(0))
        out = m(d)[0, 0]
        g_d, = torch.autograd.grad(out, d, retain_graph=True)
        g_x, = torch.autograd.grad(d, xt, grad_outputs=g_d)
        gs2.append(g_x.numpy())
    g_vjp = np.mean(gs2, 0)
    rel = float(np.max(np.abs(g_vjp - g_full)) / (np.max(np.abs(g_full)) + 1e-12))
    check("UNMASKED VJP reproduces KL._grad_feature (validates the implementation path)",
          rel < 1e-5, f"max rel diff = {rel:.2e}")
except Exception as e:
    check("masked-gradient mechanics", False, f"{type(e).__name__}: {e}")

# ---------------------------------------------------------------- 8. compute budget
print("\n8. COMPUTE BUDGET")
ncpu = os.cpu_count()
print(f"  os.cpu_count() = {ncpu}")
check("enough cores for 10 thread-pinned workers", (ncpu or 0) >= 10, f"ncpu={ncpu}")
T = 22.73
for name, nsolve in (("S4a  20 starts x 1 arm x 18", 20 * 1 * 18),
                     ("S3c  28 starts x 2 arms x 18", 28 * 2 * 18)):
    ser = nsolve * T / 3600
    par = ser / 10
    print(f"  {name} = {nsolve} solves ~ {ser:.1f} h serial / ~{par:.1f} h at 10 workers")

print("\n" + "=" * 78)
print(f"PREFLIGHT {'PASS -- both designs may freeze' if not FAIL else 'FAIL: ' + ', '.join(FAIL)}")
print("=" * 78)
sys.exit(1 if FAIL else 0)
