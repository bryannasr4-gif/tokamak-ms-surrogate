"""
13_scan_figure.py -- Figure + dataset from the elongation scan (run 12 / task bykizoh44+bwzw8e1fm).

These are the CONVERGED data points computed by FreeGSNKE in experiments/12_elongation_scan.py,
restricted to the numerically well-behaved, controllability-crossing range kappa in [1.65, 2.05].
(Taller targets, zscale>1.0, drive the plasma into the vessel and the GS solve degrades --
exactly the marginal regime where the surrogate must later quantify uncertainty.)

Each row is one full FreeGSNKE evaluation: inverse solve to a target shape -> 3051x53
finite-difference Jacobian dIy/dI -> lumped-circuit eigenvalue -> {gamma, m_s, Leuer}.
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Real converged outputs (MAST-U-like, 12 active + 138 passive, fix_n_vessel_modes=40, Ip=600kA)
records = [
    {"zscale": 0.80, "kappa": 1.650, "gamma": 2.27,   "stability_margin": 3.914, "leuer": -36.23},
    {"zscale": 0.88, "kappa": 1.822, "gamma": 22.32,  "stability_margin": 1.044, "leuer": 2.808},
    {"zscale": 0.96, "kappa": 1.975, "gamma": 147.92, "stability_margin": 0.522, "leuer": 1.774},
    {"zscale": 1.00, "kappa": 2.055, "gamma": 329.84, "stability_margin": 0.336, "leuer": 1.541},
]

os.makedirs("data", exist_ok=True)
with open("data/12_elongation_scan.json", "w") as f:
    json.dump(records, f, indent=2)

k = np.array([r["kappa"] for r in records])
g = np.array([r["gamma"] for r in records])
ms = np.array([r["stability_margin"] for r in records])

fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
ax[0].semilogy(k, g, "o-", color="crimson", lw=2, ms=8)
ax[0].set_xlabel("elongation  $\\kappa$")
ax[0].set_ylabel("vertical growth rate  $\\gamma$  [1/s]")
ax[0].set_title("$\\gamma$ rises ~2 decades with elongation")
ax[0].grid(alpha=0.3, which="both")
for kk, gg in zip(k, g):
    ax[0].annotate(f"{gg:.0f}", (kk, gg), textcoords="offset points", xytext=(6, -10), fontsize=8)

ax[1].plot(k, ms, "s-", color="navy", lw=2, ms=8)
ax[1].axhline(0.0, color="grey", ls=":", lw=1)
ax[1].axhspan(-0.05, 0.3, color="orange", alpha=0.12)
ax[1].text(1.66, 0.13, "controllability\nboundary region", fontsize=8, color="darkorange")
ax[1].set_xlabel("elongation  $\\kappa$")
ax[1].set_ylabel("inductive stability margin  $m_s$")
ax[1].set_title("$m_s$ falls monotonically toward the boundary")
ax[1].grid(alpha=0.3)

fig.suptitle("FreeGSNKE MAST-U-like ST:  $\\gamma(\\mathrm{shape})$ and $m_s(\\mathrm{shape})$ are smooth, "
             "monotonic, learnable\n(each point = one ~tens-of-GS-solves evaluation; "
             "the surrogate amortizes this to microseconds + a gradient)", fontsize=10)
fig.tight_layout(rect=(0, 0, 1, 0.93))
os.makedirs("figures", exist_ok=True)
fig.savefig("figures/12_growth_rate_vs_elongation.png", dpi=140, bbox_inches="tight")
print("Saved figures/12_growth_rate_vs_elongation.png and data/12_elongation_scan.json")
print(f"kappa {k.min():.2f}->{k.max():.2f}  |  gamma {g.min():.1f}->{g.max():.0f}/s  |  m_s {ms.max():.2f}->{ms.min():.2f}")
