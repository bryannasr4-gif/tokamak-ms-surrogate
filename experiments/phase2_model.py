"""
phase2_model.py -- the differentiable surrogates used across Phase 2 (shared by training,
gradient verification, calibration, and the dimensionality experiment).

Two models:
  * Surrogate     : SHAPE (20 descriptors) -> [log m_s, log gamma] with heteroscedastic
                    (per-target log-variance) heads. Smooth (tanh) + INTERNAL input/target
                    normalization so torch.autograd w.r.t. the RAW physical shape gives the
                    physical gradient d(log m_s)/d(shape) directly. This is the headline model.
  * ShapeMap      : CONTROL (16 inputs) -> SHAPE (20 descriptors). The well-conditioned forward
                    map; composing Surrogate(ShapeMap(u)) gives a fully differentiable, amortized
                    m_s(controls) for the solver-confirmed design / dimensionality experiments.

A deep ensemble of Surrogates gives epistemic uncertainty; the log-variance heads give
aleatoric uncertainty; total predictive variance = mean(aleatoric) + var(ensemble means).
"""
import os
import numpy as np
import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELDIR = os.path.join(ROOT, "data", "phase2_models")


class Surrogate(nn.Module):
    """SHAPE -> [log m_s, log gamma] mean + log-variance. Internal (de)normalization."""
    def __init__(self, xmean, xstd, ymean, ystd, din, h=96):
        super().__init__()
        self.din = din
        self.register_buffer("xmean", torch.as_tensor(xmean, dtype=torch.float32))
        self.register_buffer("xstd", torch.as_tensor(xstd, dtype=torch.float32))
        self.register_buffer("ymean", torch.as_tensor(ymean, dtype=torch.float32))   # (2,)
        self.register_buffer("ystd", torch.as_tensor(ystd, dtype=torch.float32))     # (2,)
        self.trunk = nn.Sequential(
            nn.Linear(din, h), nn.Tanh(),
            nn.Linear(h, h), nn.Tanh(),
            nn.Linear(h, h), nn.Tanh(),
        )
        self.mean_head = nn.Linear(h, 2)
        self.logvar_head = nn.Linear(h, 2)

    def _z(self, x):
        return (x - self.xmean) / self.xstd

    def forward(self, x):
        """Return physical mean [log m_s, log gamma] (N,2)."""
        z = self.trunk(self._z(x))
        return self.mean_head(z) * self.ystd + self.ymean

    def mean_logvar(self, x):
        """Physical mean (N,2) and physical-space log-variance (N,2) of [log m_s, log gamma]."""
        z = self.trunk(self._z(x))
        mean = self.mean_head(z) * self.ystd + self.ymean
        # logvar head is in standardized space; convert variance to physical: *ystd^2
        logvar = self.logvar_head(z) + 2.0 * torch.log(self.ystd)
        return mean, logvar


class ShapeMap(nn.Module):
    """CONTROL -> SHAPE (20 descriptors). Internal (de)normalization; smooth."""
    def __init__(self, xmean, xstd, ymean, ystd, din, dout, h=128):
        super().__init__()
        self.register_buffer("xmean", torch.as_tensor(xmean, dtype=torch.float32))
        self.register_buffer("xstd", torch.as_tensor(xstd, dtype=torch.float32))
        self.register_buffer("ymean", torch.as_tensor(ymean, dtype=torch.float32))
        self.register_buffer("ystd", torch.as_tensor(ystd, dtype=torch.float32))
        self.net = nn.Sequential(
            nn.Linear(din, h), nn.Tanh(),
            nn.Linear(h, h), nn.Tanh(),
            nn.Linear(h, h), nn.Tanh(),
            nn.Linear(h, dout),
        )

    def forward(self, x):
        z = (x - self.xmean) / self.xstd
        return self.net(z) * self.ystd + self.ymean


# ----------------------------------------------------------------- ensemble helpers
def ensemble_predict(models, X):
    """X: (N,din) np array. Returns dict with per-target mean, epistemic std, aleatoric std,
    total std -- all in LOG space (log m_s, log gamma), shape (N,2) each."""
    Xt = torch.as_tensor(np.atleast_2d(X), dtype=torch.float32)
    means, vars = [], []
    with torch.no_grad():
        for m in models:
            mu, lv = m.mean_logvar(Xt)
            means.append(mu.numpy())
            vars.append(np.exp(lv.numpy()))
    means = np.stack(means)          # (M,N,2)
    vars = np.stack(vars)            # (M,N,2)
    mean = means.mean(0)
    epi = means.var(0)               # epistemic variance (disagreement of means)
    ale = vars.mean(0)               # aleatoric variance (avg predicted)
    tot = epi + ale
    return dict(mean=mean, epi_std=np.sqrt(epi), ale_std=np.sqrt(ale), tot_std=np.sqrt(tot))


def ms_grad_shape(models, X, relative=True):
    """Mean ensemble gradient of (log m_s) w.r.t. physical SHAPE at rows of X. (N,din).
    relative=True -> d log m_s / d shape (relative); else d m_s / d shape (absolute)."""
    Xt = torch.tensor(np.atleast_2d(X), dtype=torch.float32, requires_grad=True)
    gs = []
    for m in models:
        out = m(Xt)[:, 0]                      # log m_s
        if not relative:
            out = torch.exp(out)               # m_s
        g, = torch.autograd.grad(out.sum(), Xt, retain_graph=False, create_graph=False)
        gs.append(g.detach().numpy())
    return np.mean(gs, axis=0)


def save_ensemble(models, meta, name="surrogate"):
    os.makedirs(MODELDIR, exist_ok=True)
    torch.save({"states": [m.state_dict() for m in models], "meta": meta},
               os.path.join(MODELDIR, f"{name}.pt"))


def load_ensemble(name="surrogate"):
    blob = torch.load(os.path.join(MODELDIR, f"{name}.pt"), weights_only=False)
    meta = blob["meta"]
    models = []
    for st in blob["states"]:
        m = Surrogate(meta["xmean"], meta["xstd"], meta["ymean"], meta["ystd"], meta["din"])
        m.load_state_dict(st)
        m.eval()
        models.append(m)
    return models, meta


def save_shapemap(model, meta, name="shapemap"):
    os.makedirs(MODELDIR, exist_ok=True)
    torch.save({"state": model.state_dict(), "meta": meta}, os.path.join(MODELDIR, f"{name}.pt"))


def load_shapemap(name="shapemap"):
    blob = torch.load(os.path.join(MODELDIR, f"{name}.pt"), weights_only=False)
    meta = blob["meta"]
    m = ShapeMap(meta["xmean"], meta["xstd"], meta["ymean"], meta["ystd"], meta["din"], meta["dout"])
    m.load_state_dict(blob["state"])
    m.eval()
    return m, meta
