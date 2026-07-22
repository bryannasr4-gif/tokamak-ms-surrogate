"""
tier1_analyze.py -- Tier-1 tests (b) emulator self-consistency + (c) fidelity + yield, on the
shape-anchored re-solves. (a) coverage lives in tier1_coverage.py (artifact-free, on the full pool).

(b) On real-shape re-solves: surrogate m_s vs FreeGSNKE-80 m_s, judged against the solver's OWN
    40<->80-mode ambiguity; plus the surrogate's epistemic uncertainty vs the geometric OOD distance
    (the robust, artifact-free signal). GEOMETRIC OOD only (kappa,delta,aspect) -- li/betap have EFIT-
    vs-descriptors() definitional mismatches and the ConstrainBetapIp re-solve inflates poloidalBeta2.
(c) shape fidelity: |dkappa| and boundary zeta.
Yield + dropped-set characterization by kappa bin (survivorship). Shot-clustered (report n shots).

  python experiments/tier1_analyze.py
"""
import os, sys, json, glob
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tier1_lib as T

ROOT = T.ROOT
DATA = os.path.join(ROOT, "data")
FIG = os.path.join(ROOT, "figures")
GEOM = ["kappa", "delta", "aspect"]


def maha_fit(train):
    mu = train.mean(0); inv = np.linalg.pinv(np.cov(train, rowvar=False))
    return mu, inv


def maha(mu, inv, X):
    d = X - mu
    return np.sqrt(np.einsum("ij,jk,ik->i", d, inv, d))


def main():
    os.makedirs(FIG, exist_ok=True)
    rows = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(DATA, "tier1_resolved", "*.json")))]
    if not rows:
        print("no resolved rows"); return
    df = pd.DataFrame(rows)
    n = len(df)
    n_shots = df["shot"].nunique()
    print(f"=== Tier-1 re-solve: {n} slices / {n_shots} shots ===")
    from collections import Counter
    print("status:", dict(Counter(df.get("status", pd.Series(dtype=str)))))

    # ---- YIELD + survivorship by kappa bin
    df["converged"] = df.get("converged", False).fillna(False)
    conv = df["converged"].mean()
    div = df.apply(lambda r: bool(r.get("converged")) and (r.get("limiter") is False), axis=1)
    print(f"convergence yield: {conv:.2f} ({int(df['converged'].sum())}/{n}); diverted: {div.mean():.2f}")
    if "real_kappa" in df:
        for lo, hi in [(0, 1.6), (1.6, 2.0), (2.0, 3.0)]:
            m = (df["real_kappa"] >= lo) & (df["real_kappa"] < hi)
            if m.any():
                print(f"  kappa[{lo},{hi}): n={m.sum()} conv={df.loc[m,'converged'].mean():.2f}")

    ok = df[df.get("status") == "ok"].copy()
    print(f"\nstatus=ok: {len(ok)} ({ok['shot'].nunique()} shots)")
    if ok.empty:
        return

    # ---- geometric OOD of the REAL shape (from selection) merged onto resolved
    sel = pd.DataFrame(json.load(open(os.path.join(DATA, "tier1_selection.json")))["selection"])
    ok = ok.merge(sel[["shot", "slice", "kappa", "delta", "aspect"]], on=["shot", "slice"], how="left")
    tr = pd.read_parquet(os.path.join(DATA, "dataset_v1_80q.parquet"))
    mu, inv = maha_fit(tr[GEOM].to_numpy(float))
    md_self = maha(mu, inv, tr[GEOM].to_numpy(float)); thr99 = np.percentile(md_self, 99)
    okg = ok.dropna(subset=GEOM).copy()
    okg["maha"] = maha(mu, inv, okg[GEOM].to_numpy(float))

    # ---- (b) emulator self-consistency
    b = okg.dropna(subset=["surr_ms", "ms_solver_80", "ms_solver_40"]).copy()
    b = b[(b["ms_solver_80"] > 0) & (b["surr_ms"] > 0) & (b["ms_solver_40"] > 0)]
    b["resid"] = np.abs(np.log(b["surr_ms"]) - np.log(b["ms_solver_80"]))
    b["amb"] = np.abs(np.log(b["ms_solver_80"]) - np.log(b["ms_solver_40"]))
    # in-distribution epistemic-uncertainty baseline (surrogate on a training-shape sample)
    epi_baseline = None
    try:
        from phase2_model import load_ensemble, ensemble_predict
        from phase2_data import SHAPE_FEATURES
        models, _ = load_ensemble("surrogate")
        samp = tr.sample(min(500, len(tr)), random_state=0)[SHAPE_FEATURES].to_numpy(float)
        epi_baseline = float(np.median(ensemble_predict(models, samp)["epi_std"][:, 0]))
        print(f"    in-distribution epistemic std baseline (train sample): median={epi_baseline:.3f}")
    except Exception as e:
        print("    (epi baseline failed:", e, ")")
    print(f"\n(b) EMULATOR CONSISTENCY on {len(b)} slices ({b['shot'].nunique()} shots)")
    if len(b):
        print(f"    surrogate-vs-solver80 |rel log m_s|: median={b['resid'].median():.3f} p90={b['resid'].quantile(0.9):.3f}")
        print(f"    intrinsic 40<->80 ambiguity:        median={b['amb'].median():.3f} p90={b['amb'].quantile(0.9):.3f} (synthetic ref ~0.14)")
        print(f"    surrogate epistemic std:            median={b['surr_epi_std'].median():.3f}")
        if len(b) > 4:
            print(f"    corr(resid, maha_OOD)={np.corrcoef(b['resid'], b['maha'])[0,1]:+.2f}   "
                  f"corr(epi_std, maha_OOD)={np.corrcoef(b['surr_epi_std'], b['maha'])[0,1]:+.2f}")
        frac_within = float((b["resid"] <= b["amb"]).mean())
        print(f"    fraction with resid <= own 40-80 ambiguity: {frac_within:.2f}")

    # ---- HONESTY DISCLOSURES (from adversarial review) ----
    n_sel = len(sel)
    n_conv = int(df["converged"].sum())
    n_defms = int(((df.get("ms_solver_80") > 0)).sum())
    print(f"\n[FUNNEL] {n_sel} selected -> {n} resolved -> {n_conv} converged -> {n_defms} with defined solver m_s "
          f"-> {len(b)} enter test(b). Yield (kept/selected) = {n}/{n_sel} = {n/n_sel:.2f}.")
    # attrition non-random in kappa: m_s-undefined vs defined by kappa
    okd = ok.copy(); okd["has_ms"] = okd["ms_solver_80"].notna() & (okd["ms_solver_80"] > 0)
    for lo, hi in [(0, 1.6), (1.6, 2.0), (2.0, 3.0)]:
        m = (okd["real_kappa"] >= lo) & (okd["real_kappa"] < hi)
        if m.any():
            print(f"    kappa[{lo},{hi}): has-defined-m_s = {okd.loc[m,'has_ms'].mean():.2f} (n={int(m.sum())})")
    if len(b):
        print(f"    b-set kappa: median={b['real_kappa'].median():.2f} range[{b['real_kappa'].min():.2f},{b['real_kappa'].max():.2f}]; "
              f"high-kappa tail (>=2.0) in b-set = {int((b['real_kappa']>=2.0).sum())}/{int((sel['kappa']>=2.0).sum())} selected "
              f"=> test(b) is a MID-kappa subset (OOD tail largely absent).")
    # pre-registered profile-match gate (0/N expected)
    if "dli" in ok and "dbetap" in ok:
        gate = ((ok["dli"].abs() <= 0.1) & (ok["dbetap"].abs() <= 0.1))
        print(f"    PRE-REG profile-match gate (|dli|,|dbetap|<=0.1): {int(gate.sum())}/{len(ok)} pass "
              f"=> test(b) is OUTSIDE the pre-registered profile-match regime (ConstrainBetapIp betap=poloidalBeta2).")
        print(f"    re-solve vs EFIT (CORRECTED): betap median re={ok['re_betap'].median():.2f} vs real={ok['real_betap'].median():.2f} "
              f"(~{ok['re_betap'].median()/max(ok['real_betap'].median(),1e-9):.1f}x); li median re={ok['re_li'].median():.2f} vs real={ok['real_li'].median():.2f}")
    # limited fraction among converged (cross-machine fidelity limitation)
    convrows = df[df["converged"] == True]
    if len(convrows):
        lim_frac = float((convrows["limiter"] == True).mean())
        print(f"    of converged re-solves, {lim_frac:.0%} come out LIMITED (not diverted) — real MAST discharges are diverted "
              f"=> cross-machine fidelity limitation; test(b) bounds consistency on re-solved ARTIFACT geometry.")

    # ---- (c) fidelity
    c = ok.dropna(subset=["re_kappa", "real_kappa"])
    dk = (c["re_kappa"] - c["real_kappa"]).abs()
    print(f"\n(c) FIDELITY on {len(c)}: |dkappa| median={dk.median():.3f} p90={dk.quantile(0.9):.3f} (target<=0.03)")
    if "zeta_cm" in ok:
        z = ok["zeta_cm"].dropna()
        if len(z):
            print(f"    boundary zeta_cm: median={z.median():.2f} p90={z.quantile(0.9):.2f} (cross-machine)")
    if "dli" in ok:
        print(f"    re-solve vs EFIT: |dli| median={ok['dli'].abs().median():.3f}  |dbetap| median={ok['dbetap'].abs().median():.3f} "
              f"(profile-family gap; re-solve betap=poloidalBeta2 != EFIT betap)")

    # ---- figure
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
    if len(b):
        ax[0].scatter(b["maha"], b["resid"], s=26, c="navy", label="surr vs solver-80")
        ax[0].axhline(b["amb"].median(), ls="--", c="green", label=f"40-80 ambiguity (med {b['amb'].median():.2f})")
        ax[0].set_xlabel("geometric OOD (Mahalanobis)"); ax[0].set_ylabel("|rel log m_s error|")
        ax[0].legend(fontsize=8); ax[0].set_title("(b) surrogate-vs-solver vs OOD")
        ax[1].scatter(b["maha"], b["surr_epi_std"], s=26, c="purple", label="real MAST")
        if epi_baseline is not None:
            ax[1].axhline(epi_baseline, ls="--", c="gray", label=f"in-dist baseline ({epi_baseline:.2f})")
        ax[1].set_xlabel("geometric OOD (Mahalanobis)"); ax[1].set_ylabel("surrogate epistemic std")
        ax[1].legend(fontsize=8)
        ax[1].set_title("(b) epistemic uncertainty: uniformly elevated on real (OOD)")
    if len(c):
        ax[2].scatter(c["real_kappa"], c["re_kappa"], s=26, c="teal")
        lim = [min(c["real_kappa"].min(), c["re_kappa"].min()), max(c["real_kappa"].max(), c["re_kappa"].max())]
        ax[2].plot(lim, lim, "k--"); ax[2].set_xlabel("EFIT kappa"); ax[2].set_ylabel("re-solved kappa")
        ax[2].set_title("(c) shape fidelity")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "tier1_consistency.png"), dpi=120)
    print("\nsaved figures/tier1_consistency.png")

    summ = dict(
        n_slices=n, n_shots=int(n_shots), convergence_yield=float(conv), diverted_frac=float(div.mean()),
        status=dict(Counter(df.get("status", pd.Series(dtype=str)))),
        consistency=dict(n=int(len(b)), n_shots=int(b["shot"].nunique()) if len(b) else 0,
                         median_resid=float(b["resid"].median()) if len(b) else None,
                         median_ambiguity_40_80=float(b["amb"].median()) if len(b) else None,
                         median_epi_std=float(b["surr_epi_std"].median()) if len(b) else None,
                         corr_resid_ood=float(np.corrcoef(b["resid"], b["maha"])[0, 1]) if len(b) > 4 else None,
                         corr_epi_ood=float(np.corrcoef(b["surr_epi_std"], b["maha"])[0, 1]) if len(b) > 4 else None),
        fidelity=dict(median_dkappa=float(dk.median()) if len(c) else None,
                      median_zeta_cm=float(ok["zeta_cm"].median()) if "zeta_cm" in ok and ok["zeta_cm"].notna().any() else None))
    json.dump(summ, open(os.path.join(DATA, "tier1_analysis.json"), "w"), indent=2)
    print("saved data/tier1_analysis.json")


if __name__ == "__main__":
    main()
