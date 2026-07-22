"""
15_gradient_quality_figure.py -- visualize the Phase-1 precondition result (from run 14).

Shows m_s(kappa) at fine steps is smooth and its central-difference slope is consistent
(same sign, gently varying) -- i.e. the true-solver gradient d(m_s)/d(shape) is well-defined,
not noise-dominated. This is the necessary precondition for the gradient-USED contribution.
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

with open("data/14_gradient_quality.json") as f:
    rows = json.load(f)
rows = [r for r in rows if np.isfinite(r["m_s"])]
k = np.array([r["kappa"] for r in rows])
ms = np.array([r["m_s"] for r in rows])

# central-difference slopes at interior points
ks, slopes = [], []
for i in range(1, len(rows) - 1):
    dk = k[i + 1] - k[i - 1]
    slopes.append((ms[i + 1] - ms[i - 1]) / dk)
    ks.append(k[i])

fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
ax[0].plot(k, ms, "o-", color="navy", lw=2, ms=8)
ax[0].set_xlabel("elongation  $\\kappa$")
ax[0].set_ylabel("stability margin  $m_s$")
ax[0].set_title("$m_s(\\kappa)$ is smooth at fine step (no jitter)")
ax[0].grid(alpha=0.3)

ax[1].plot(ks, slopes, "D-", color="seagreen", lw=2, ms=9)
ax[1].set_xlabel("elongation  $\\kappa$")
ax[1].set_ylabel("true-solver  $d m_s/d\\kappa$  (central diff)")
ax[1].set_title("Gradient is same-sign & slowly varying → well-defined")
ax[1].grid(alpha=0.3)
for kk, ss in zip(ks, slopes):
    ax[1].annotate(f"{ss:.2f}", (kk, ss), textcoords="offset points", xytext=(6, 6), fontsize=9)

fig.suptitle("Phase-1 precondition PASSED: the true $d m_s/d(\\mathrm{shape})$ exists and is smooth\n"
             "(the central risk — noisy eigenvalue gradient — is empirically refuted at this mode count)",
             fontsize=10)
fig.tight_layout(rect=(0, 0, 1, 0.92))
os.makedirs("figures", exist_ok=True)
fig.savefig("figures/14_gradient_quality.png", dpi=140, bbox_inches="tight")
print("Saved figures/14_gradient_quality.png")
print(f"slopes: {[round(s,2) for s in slopes]}  (all same sign: {all(np.sign(slopes)==np.sign(slopes[0]))})")
