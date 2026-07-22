"""
phase4_figures.py -- all Phase-4 figures. Each panel guards on its data file, so this can be run
incrementally as results land. Saves into figures/.
  phase4_kappa_differentiator.png : median gains by regime + beats-rates (the learned-m_s lead)
  phase4_gallery.png              : solver-confirmed before->after LCFS at FIXED kappa (panels)
  phase4_pareto.png               : cost/accuracy (surrogate inference+grad vs full 80-mode solve)
  phase4_leuer.png                : Leuer rank ceiling vs the learned m_s, by regime
  phase4_ablations.png            : learning curve + ensemble size + shape param + input noise
"""
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FIG = "figures"
os.makedirs(FIG, exist_ok=True)


def load(p):
    return json.load(open(p)) if os.path.exists(p) else None


def fig_differentiator():
    # prefer the POOLED (n=32) result if the power top-up has landed; else the n=20 reframe
    d = load("data/phase4_kappa_pooled.json") or load("data/phase4_kappa_reframed.json")
    if not d:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    regimes = [r for r in ["pooled", "marginal", "mid"] if r in d["by_regime"]]
    methods = ["surrogate", "best_fixed_lever", "oracle_lever", "best_gradfree"]
    labels = ["learned m_s\n(surrogate)", "best realizable\nfixed lever",
              "oracle best-of-8\nlever (hindsight)", "best gradient-\nfree search"]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    ax = axes[0]
    w = 0.2
    x = np.arange(len(regimes))
    for j, m in enumerate(methods):
        vals = [d["by_regime"][r]["median_gain"][m] for r in regimes]
        ax.bar(x + (j - 1.5) * w, vals, w, label=labels[j], color=colors[j])
    ax.set_xticks(x); ax.set_xticklabels([f"{r}\n(n={d['by_regime'][r]['n']})" for r in regimes])
    ax.set_ylabel("median true-m_s gain at FIXED kappa")
    ax.set_title("(a) kappa held +-%.2f: learned m_s raises m_s\nvia secondary levers" % d["ktol"])
    ax.legend(fontsize=7, loc="upper left", ncol=2)
    ax.grid(axis="y", alpha=0.3)

    # beats-rates vs each baseline (pooled) with Wilson CIs
    ax = axes[1]
    p = d["by_regime"]["pooled"]
    comps = [("vs_best_fixed_lever", "vs best realizable\nfixed lever"),
             ("vs_oracle_lever", "vs oracle best-of-8\n(not realizable)"),
             ("vs_best_gradfree", "vs gradient-free\nsearch")]
    y = np.arange(len(comps))
    for i, (k, nm) in enumerate(comps):
        b = p[k]
        frac = b["beats_frac"]; ci = b["wilson95"]
        ax.barh(i, frac, color="#1f77b4", alpha=0.8)
        ax.errorbar(frac, i, xerr=[[frac - ci[0]], [ci[1] - frac]], fmt="none", ecolor="k", capsize=4)
        ax.text(0.02, i, f"{b['wins']}/{b['n']} = {frac*100:.0f}%  (Wilcoxon p={b['wilcoxon_p']:.3f})",
                va="center", fontsize=8, color="white" if frac > 0.3 else "black")
    ax.axvline(0.5, color="grey", ls="--", lw=1)
    ax.set_yticks(y); ax.set_yticklabels([nm for _, nm in comps], fontsize=8)
    ax.set_xlim(0, 1); ax.set_xlabel("fraction of starts surrogate beats baseline (pooled)")
    ax.set_title("(b) surrogate beats every REALIZABLE baseline\n(only the unrealizable oracle ties)")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{FIG}/phase4_kappa_differentiator.png", dpi=140)
    plt.close(fig)
    print("wrote phase4_kappa_differentiator.png")


def fig_gallery():
    sh = load("data/phase4_gallery_shapes.json")
    summ = load("data/phase4_gallery_summary.json")
    if not sh or not sh.get("panels"):
        return
    res = None
    rr = load("data/phase4_gallery_results.json")
    if rr:
        res = {r["start_i"]: r for r in rr["recs"] if r["method"] == "surrogate"}
    panels = sh["panels"]
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 5.2))
    if n == 1:
        axes = [axes]
    limR, limZ = sh["limiter"]["R"], sh["limiter"]["Z"]
    for ax, p in zip(axes, panels):
        ax.plot(limR, limZ, color="0.6", lw=1.0, label="limiter")
        st, be = p["start"], p["best"]
        ax.plot(st["R"], st["Z"], color="#d62728", lw=2, label=f"start  m_s={st['m_s']:.2f}")
        ax.plot(be["R"], be["Z"], color="#1f77b4", lw=2, label=f"design m_s={be['m_s']:.2f}")
        ax.set_aspect("equal")
        ax.set_title(f"{p['label']} ({p['regime']})\nkappa {st['kappa']:.3f}->{be['kappa']:.3f} "
                     f"(LOCKED), m_s +{be['m_s']-st['m_s']:.2f}", fontsize=9)
        ax.legend(fontsize=7, loc="upper right")
        ax.set_xlabel("R [m]")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Z [m]")
    fig.suptitle("Phase-4 kappa-CONSTRAINED design gallery: learned-m_s gradient raises true m_s "
                 "at FIXED kappa (every shape 80-mode solver-confirmed)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(f"{FIG}/phase4_gallery.png", dpi=140)
    plt.close(fig)
    print("wrote phase4_gallery.png")


def fig_pareto():
    d = load("data/phase4_pareto.json") or load("data/phase4_pareto_partial.json")
    if not d:
        return
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    t_true = d.get("t_true_solve_s", 24.0)
    items = [("full 80-mode\nFreeGSNKE solve", t_true, "#d62728"),
             ("surrogate m_s\n+ gradient", d["t_surr_infer_s"] + d["t_grad_s"], "#1f77b4"),
             ("surrogate m_s\ninference only", d["t_surr_infer_s"], "#2ca02c")]
    names = [i[0] for i in items]
    times = [i[1] for i in items]
    ax.barh(range(len(items)), times, color=[i[2] for i in items], log=True)
    for i, t in enumerate(times):
        ax.text(t * 1.3, i, f"{t*1e3:.2f} ms" if t < 1 else f"{t:.1f} s", va="center", fontsize=9)
    ax.set_yticks(range(len(items))); ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("wall-clock per evaluation (s, log scale)")
    sp = (t_true / (d["t_surr_infer_s"] + d["t_grad_s"]))
    ax.set_title(f"Cost/accuracy: surrogate m_s+grad is ~{sp:,.0f}x faster than a true solve\n"
                 f"at held-out log-R2={d.get('r2_log',0):.3f} (RMSE_log={d.get('rmse_log',0):.3f})",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(f"{FIG}/phase4_pareto.png", dpi=140)
    plt.close(fig)
    print("wrote phase4_pareto.png")


def fig_leuer():
    d = load("data/phase4_leuer.json")
    if not d:
        return
    fig, ax = plt.subplots(figsize=(7, 4.4))
    regs = list(d["by_regime"].keys())
    x = np.arange(len(regs))
    leu = [d["by_regime"][r]["spearman_leuer"] for r in regs]
    sur = [d["by_regime"][r]["spearman_surrogate"] for r in regs]
    ax.bar(x - 0.2, leu, 0.4, label="rigid Leuer parameter", color="#ff7f0e")
    ax.bar(x + 0.2, sur, 0.4, label="learned m_s(shape)", color="#1f77b4")
    ax.set_xticks(x); ax.set_xticklabels([f"{r}\n(n={d['by_regime'][r]['n']})" for r in regs])
    ax.set_ylabel("Spearman(prediction, true m_s)")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Leuer rank ceiling vs the learned m_s (held-out)\n"
                 f"overall Spearman: Leuer {d['spearman_leuer_clipped']:.3f} -> surrogate "
                 f"{d['spearman_surrogate']:.3f}; marginal AUC {d['auc_marginal_leuer']:.3f}->"
                 f"{d['auc_marginal_surrogate']:.3f}", fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{FIG}/phase4_leuer.png", dpi=140)
    plt.close(fig)
    print("wrote phase4_leuer.png")


def fig_ablations():
    light = load("data/phase4_ablations_light.json")
    train = load("data/phase4_ablations_train.json")
    if not light:
        return
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    # ensemble size
    ax = axes[0]
    e = light["ensemble_size"]
    ks = [x["k"] for x in e]
    ax.plot(ks, [x["rmse_log"] for x in e], "o-", label="all held-out")
    ax.plot(ks, [x["rmse_log_marginal"] for x in e], "s--", label="marginal")
    ax.set_xlabel("ensemble size k"); ax.set_ylabel("RMSE_log")
    ax.set_title("(a) ensemble size"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    # input noise
    ax = axes[1]
    nz = light["input_noise"]["rows"]
    sig = [r["sigma_frac"] * 100 for r in nz]
    ax.plot(sig, [r["rmse_log_mean"] for r in nz], "o-", color="#1f77b4", label="RMSE_log")
    ax.axhline(light["input_noise"]["baseline_rmse_log"], color="grey", ls="--", lw=1, label="baseline")
    ax2 = ax.twinx()
    ax2.plot(sig, [r["grad_cos_median"] for r in nz], "s--", color="#2ca02c", label="grad-dir cos")
    ax2.set_ylabel("gradient-direction cosine", color="#2ca02c"); ax2.set_ylim(0.9, 1.005)
    ax.set_xlabel("input noise (% of feature std)"); ax.set_ylabel("RMSE_log", color="#1f77b4")
    ax.set_title("(b) input-noise robustness"); ax.grid(alpha=0.3)
    # learning curve / shape param (if available)
    ax = axes[2]
    if train and train.get("learning_curve"):
        lc = train["learning_curve"]
        ax.plot([x["n"] for x in lc], [x["rmse_log"] for x in lc], "o-", label="all held-out")
        ax.plot([x["n"] for x in lc], [x["rmse_log_marginal"] for x in lc], "s--", label="marginal")
        ax.set_xlabel("# training shapes"); ax.set_ylabel("RMSE_log")
        ax.set_title("(c) dataset-size learning curve"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    else:
        ax.text(0.5, 0.5, "learning curve\n(pending retrain)", ha="center", va="center")
        ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(f"{FIG}/phase4_ablations.png", dpi=140)
    plt.close(fig)
    print("wrote phase4_ablations.png")


def main():
    which = sys.argv[1:] if len(sys.argv) > 1 else ["all"]
    fns = dict(differentiator=fig_differentiator, gallery=fig_gallery, pareto=fig_pareto,
               leuer=fig_leuer, ablations=fig_ablations)
    if "all" in which:
        for f in fns.values():
            f()
    else:
        for w in which:
            fns[w]()


if __name__ == "__main__":
    main()
