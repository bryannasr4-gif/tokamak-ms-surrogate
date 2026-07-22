"""
phase0_noise_analyze.py -- tabulate + plot the Phase-0 m_s/gamma reproducibility floor.

Consumes data/phase0_noise_sweep.json and produces:
  - printed tables for each axis (threads / tol / grid / modes),
  - the HEADLINE floor numbers (cross-BLAS-thread m_s & gamma spread across shapes),
  - the recommended fixed protocol,
  - figures/phase0_noise_floor.png (4-panel).
Robust to failed/missing jobs.
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IN = os.path.join(ROOT, "data", "phase0_noise_sweep.json")
FIG = os.path.join(ROOT, "figures", "phase0_noise_floor.png")


def load():
    with open(IN) as f:
        d = json.load(f)
    return d, [r for r in d["results"] if r.get("ok")]


def find(res, z, d, th, tol, nx, ny, modes):
    for r in res:
        if (abs(r["zscale"] - z) < 1e-9 and abs(r.get("dR", 0) - d) < 1e-9 and
                r["threads"] == th and abs(r["tol"] - tol) < 1e-15 and
                r["nx"] == nx and r["ny"] == ny and r["modes"] == modes):
            return r
    return None


def spread_pct(vals):
    vals = np.array([v for v in vals if v is not None and np.isfinite(v)])
    if vals.size < 2:
        return float("nan"), vals.size
    return float(100 * (vals.max() - vals.min()) / np.mean(vals)), vals.size


def main():
    d, res = load()
    base = d["base"]; REP = [tuple(x) for x in d["rep"]]; SHAPES = [tuple(x) for x in d["shapes"]]
    THREADS = (1, 2, 4, 8)
    print(f"{len(res)} ok results (of {len(d['results'])})\n")

    # --- AXIS 1: BLAS threads (headline) --------------------------------------
    print("=" * 78)
    print("AXIS 1 -- BLAS THREADS {1,2,4,8}  (baseline tol=1e-6, 65x65, modes=40)")
    print(f"{'shape (z,dR)':>16} | {'m_s @1':>8} {'@2':>8} {'@4':>8} {'@8':>8} | "
          f"{'m_s spr%':>8} | {'gamma spr%':>10}")
    ms_spreads, gm_spreads = [], []
    thread_rows = []
    for (z, dd) in SHAPES:
        msv = [find(res, z, dd, th, 1e-6, 65, 65, 40) for th in THREADS]
        ms = [r["m_s"] if r else None for r in msv]
        gm = [r["gamma"] if r else None for r in msv]
        sp_m, n_m = spread_pct(ms); sp_g, _ = spread_pct(gm)
        if np.isfinite(sp_m):
            ms_spreads.append(sp_m)
        if np.isfinite(sp_g):
            gm_spreads.append(sp_g)
        thread_rows.append(((z, dd), ms, gm))
        fmt = lambda x: f"{x:8.4f}" if x is not None else f"{'--':>8}"
        print(f"({z:.2f},{dd:+.2f})".rjust(16) + " | " + " ".join(fmt(x) for x in ms) +
              f" | {sp_m:8.2f} | {sp_g:10.2f}")
    ms_spreads = np.array(ms_spreads); gm_spreads = np.array(gm_spreads)
    print(f"\n  HEADLINE m_s cross-thread spread: median {np.median(ms_spreads):.1f}%  "
          f"max {ms_spreads.max():.1f}%  (n={ms_spreads.size} shapes)")
    print(f"  HEADLINE gamma cross-thread spread: median {np.median(gm_spreads):.1f}%  "
          f"max {gm_spreads.max():.1f}%")
    print(f"  NOTE: within a FIXED thread count m_s is bit-deterministic (verified Part 3) "
          f"=> fixing OMP=1 makes the reproducibility floor ~0; the spread above is the "
          f"cross-config systematic uncertainty in the absolute value.")

    # --- AXIS 2: inverse-solve tolerance --------------------------------------
    print("\n" + "=" * 78)
    print("AXIS 2 -- INVERSE TOL {1e-6,1e-8,1e-10}: does tighter tol shrink the 1-vs-8 spread?")
    print(f"{'shape':>12} {'tol':>8} | {'m_s @1':>9} {'m_s @8':>9} | {'1v8 spr%':>9}")
    tol_data = {}
    for (z, dd) in REP:
        for tol in (1e-6, 1e-8, 1e-10):
            r1 = find(res, z, dd, 1, tol, 65, 65, 40)
            r8 = find(res, z, dd, 8, tol, 65, 65, 40)
            m1 = r1["m_s"] if r1 else None; m8 = r8["m_s"] if r8 else None
            sp, _ = spread_pct([m1, m8])
            tol_data[(z, dd, tol)] = (m1, m8, sp)
            f = lambda x: f"{x:9.4f}" if x is not None else f"{'--':>9}"
            print(f"({z:.2f},{dd:+.2f})".rjust(12) + f" {tol:8.0e} | {f(m1)} {f(m8)} | {sp:9.2f}")

    # --- AXIS 3: grid resolution ----------------------------------------------
    print("\n" + "=" * 78)
    print("AXIS 3 -- GRID {65x65,65x129,129x129}  (threads=1, tol=1e-6, modes=40)")
    print(f"{'shape':>12} | {'65x65':>9} {'65x129':>9} {'129x129':>9} | shift% vs 65x65 | cost x")
    grid_data = {}
    for (z, dd) in REP:
        vals, costs = [], []
        for (nx, ny) in ((65, 65), (65, 129), (129, 129)):
            r = find(res, z, dd, 1, 1e-6, nx, ny, 40)
            vals.append(r["m_s"] if r else None); costs.append(r["wall_s"] if r else None)
        grid_data[(z, dd)] = (vals, costs)
        base_v = vals[0]
        shift = [(100 * (v - base_v) / base_v if (v is not None and base_v) else None) for v in vals]
        f = lambda x: f"{x:9.4f}" if x is not None else f"{'--':>9}"
        cx = (costs[2] / costs[0]) if (costs[0] and costs[2]) else float("nan")
        print(f"({z:.2f},{dd:+.2f})".rjust(12) + f" | {f(vals[0])} {f(vals[1])} {f(vals[2])} | "
              f"{str([None if s is None else round(s,1) for s in shift]):>16} | {cx:.1f}x")

    # --- AXIS 4: mode count ---------------------------------------------------
    print("\n" + "=" * 78)
    print("AXIS 4 -- MODES {20,40,80}  (threads=1, tol=1e-6, 65x65)")
    print(f"{'shape':>12} | {'m20':>9} {'m40':>9} {'m80':>9} | 40->80 shift% | cost x")
    mode_data = {}
    for (z, dd) in REP:
        vals, costs = [], []
        for m in (20, 40, 80):
            r = find(res, z, dd, 1, 1e-6, 65, 65, m)
            vals.append(r["m_s"] if r else None); costs.append(r["wall_s"] if r else None)
        mode_data[(z, dd)] = (vals, costs)
        sh = (100 * (vals[2] - vals[1]) / vals[1]) if (vals[1] and vals[2]) else float("nan")
        f = lambda x: f"{x:9.4f}" if x is not None else f"{'--':>9}"
        cx = (costs[2] / costs[1]) if (costs[1] and costs[2]) else float("nan")
        print(f"({z:.2f},{dd:+.2f})".rjust(12) + f" | {f(vals[0])} {f(vals[1])} {f(vals[2])} | "
              f"{sh:13.1f} | {cx:.1f}x")

    # --- FIGURE ---------------------------------------------------------------
    fig, ax = plt.subplots(2, 2, figsize=(13, 9))
    # Panel A: m_s vs threads per shape
    for (sh, ms, gm) in thread_rows:
        ys = [m for m in ms]
        ax[0, 0].plot(THREADS, ys, "o-", ms=4, lw=1, label=f"z={sh[0]:.2f},dR={sh[1]:+.2f}")
    ax[0, 0].set_xscale("log", base=2); ax[0, 0].set_xticks(THREADS); ax[0, 0].set_xticklabels(THREADS)
    ax[0, 0].set_xlabel("BLAS threads (OMP_NUM_THREADS)"); ax[0, 0].set_ylabel("m_s")
    ax[0, 0].set_title(f"A. m_s vs BLAS threads (per shape)\nmedian cross-thread spread "
                       f"= {np.median(ms_spreads):.1f}%, max {ms_spreads.max():.1f}%")
    ax[0, 0].legend(fontsize=6, ncol=2)
    # Panel B: 1v8 spread vs tol
    for (z, dd) in REP:
        sps = [tol_data[(z, dd, tol)][2] for tol in (1e-6, 1e-8, 1e-10)]
        ax[0, 1].plot([6, 8, 10], sps, "s-", label=f"z={z:.2f},dR={dd:+.2f}")
    ax[0, 1].set_xlabel("inverse-solve tolerance (1e-X)"); ax[0, 1].set_ylabel("OMP 1-vs-8 m_s spread [%]")
    ax[0, 1].set_title("B. Does tighter tol shrink the cross-thread floor?")
    ax[0, 1].legend(fontsize=7)
    # Panel C: grid
    gl = ["65x65", "65x129", "129x129"]
    for (z, dd) in REP:
        vals = grid_data[(z, dd)][0]
        ax[1, 0].plot(range(3), vals, "^-", label=f"z={z:.2f},dR={dd:+.2f}")
    ax[1, 0].set_xticks(range(3)); ax[1, 0].set_xticklabels(gl)
    ax[1, 0].set_xlabel("grid resolution"); ax[1, 0].set_ylabel("m_s")
    ax[1, 0].set_title("C. m_s vs grid resolution"); ax[1, 0].legend(fontsize=7)
    # Panel D: modes
    for (z, dd) in REP:
        vals = mode_data[(z, dd)][0]
        ax[1, 1].plot([20, 40, 80], vals, "D-", label=f"z={z:.2f},dR={dd:+.2f}")
    ax[1, 1].set_xlabel("fix_n_vessel_modes"); ax[1, 1].set_ylabel("m_s")
    ax[1, 1].set_title("D. m_s vs retained passive modes"); ax[1, 1].legend(fontsize=7)
    fig.suptitle("Phase 0 -- m_s numerical reproducibility floor (MAST-U, isolated cold solves)",
                 fontsize=13)
    fig.tight_layout()
    os.makedirs(os.path.dirname(FIG), exist_ok=True)
    fig.savefig(FIG, dpi=130)
    print(f"\nSaved figure -> {FIG}")

    # --- machine-readable summary ---------------------------------------------
    summary = dict(
        ms_cross_thread_spread_median=float(np.median(ms_spreads)),
        ms_cross_thread_spread_max=float(ms_spreads.max()),
        gamma_cross_thread_spread_median=float(np.median(gm_spreads)),
        gamma_cross_thread_spread_max=float(gm_spreads.max()),
        n_shapes=int(ms_spreads.size),
    )
    with open(os.path.join(ROOT, "data", "phase0_noise_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("Summary:", json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
