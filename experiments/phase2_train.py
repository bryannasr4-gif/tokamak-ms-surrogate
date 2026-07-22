"""
phase2_train.py -- Phase-2 components 1 & 2 (no solver needed; pure dataset_v1).

Trains the headline DIFFERENTIABLE surrogate m_s(shape) (+ gamma) as a heteroscedastic deep
ensemble, plus the control->shape map for the design/dimensionality experiments. Then:
  * accuracy RESOLVED by m_s regime and on the held-out test_extrap corner (split honored);
  * baselines: linear, GP, and the rigid Leuer parameter (Spearman + marginal-vs-stable AUC);
  * calibration: ensemble(epistemic)+heteroscedastic(aleatoric) intervals, coverage + reliability
    resolved in the marginal m_s<0.4 bin; predictive width vs m_s (abstention near m_s->0).

Saves: data/phase2_models/{surrogate,shapemap}.pt, data/phase2_predictions.parquet,
data/phase2_train_metrics.json. Figures are made by phase2_figures.py.

NOTE: accuracy is NOT the contribution (a linear fit on log m_s already ~0.9). The headline is
the GRADIENT (verified in phase2_gradcheck) and the solver-confirmed design loop. R2 here is a
sanity check + the calibration/abstention story.
"""
import json
import os
import sys

import numpy as np
import torch

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, "experiments")
import phase2_data as D
import phase2_model as M

torch.manual_seed(0)
np.random.seed(0)


# ----------------------------------------------------------------- training
def _standardized_out(model, Xt):
    z = model.trunk(model._z(Xt))
    return model.mean_head(z), model.logvar_head(z)


def train_surrogate(Xtr, Ttr, din, n_models=8, warmup=800, epochs=2600, wd=1e-4, seed0=0):
    """Heteroscedastic deep ensemble on standardized targets Ttr=[log m_s, log gamma]."""
    xmean, xstd = Xtr.mean(0), Xtr.std(0) + 1e-8
    ymean, ystd = Ttr.mean(0), Ttr.std(0) + 1e-8
    Xt = torch.tensor(Xtr, dtype=torch.float32)
    Tt = torch.tensor(Ttr, dtype=torch.float32)
    Ys = (Tt - torch.tensor(ymean, dtype=torch.float32)) / torch.tensor(ystd, dtype=torch.float32)
    N = len(Xtr)
    models = []
    for mi in range(n_models):
        torch.manual_seed(seed0 + mi)
        g = torch.Generator().manual_seed(1000 + mi)
        boot = torch.randint(0, N, (N,), generator=g)        # bootstrap for epistemic diversity
        net = M.Surrogate(xmean, xstd, ymean, ystd, din)
        opt = torch.optim.Adam(net.parameters(), lr=3e-3, weight_decay=wd)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
        xb, yb = Xt[boot], Ys[boot]
        for ep in range(epochs):
            opt.zero_grad()
            mu, logvar = _standardized_out(net, xb)
            if ep < warmup:
                loss = ((mu - yb) ** 2).mean()               # stabilize means first
            else:
                lv = logvar.clamp(-8, 8)
                loss = 0.5 * (torch.exp(-lv) * (mu - yb) ** 2 + lv).mean()
            loss.backward()
            opt.step()
            sched.step()
        net.eval()
        models.append(net)
        print(f"  surrogate member {mi} trained (final loss {loss.item():.4f})", flush=True)
    return models, dict(xmean=xmean, xstd=xstd, ymean=ymean, ystd=ystd, din=din)


def train_shapemap(Xtr, Ytr, din, dout, epochs=3000, wd=1e-5):
    xmean, xstd = Xtr.mean(0), Xtr.std(0) + 1e-8
    ymean, ystd = Ytr.mean(0), Ytr.std(0) + 1e-8
    Xt = torch.tensor(Xtr, dtype=torch.float32)
    Yt = torch.tensor(Ytr, dtype=torch.float32)
    Ys = (Yt - torch.tensor(ymean, dtype=torch.float32)) / torch.tensor(ystd, dtype=torch.float32)
    torch.manual_seed(0)
    net = M.ShapeMap(xmean, xstd, ymean, ystd, din, dout)
    opt = torch.optim.Adam(net.parameters(), lr=2e-3, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    for ep in range(epochs):
        opt.zero_grad()
        z = (Xt - net.xmean) / net.xstd
        pred = net.net(z)
        loss = ((pred - Ys) ** 2).mean()
        loss.backward(); opt.step(); sched.step()
    net.eval()
    print(f"  shapemap trained (final standardized MSE {loss.item():.4f})", flush=True)
    return net, dict(xmean=xmean, xstd=xstd, ymean=ymean, ystd=ystd, din=din, dout=dout)


# ----------------------------------------------------------------- metrics
def r2(y, yhat):
    return float(1 - np.sum((y - yhat) ** 2) / np.sum((y - y.mean()) ** 2))


def rmse(y, yhat):
    return float(np.sqrt(np.mean((y - yhat) ** 2)))


def resolved(df_split, ms_true, ms_pred):
    """R2/RMSE in ORIGINAL m_s units, resolved by regime, within one split."""
    out = {}
    out["all"] = dict(n=len(ms_true), r2=r2(ms_true, ms_pred), rmse=rmse(ms_true, ms_pred))
    reg = df_split["regime"].values
    for name, _, _ in D.REGIMES:
        m = reg == name
        if m.sum() >= 5:
            out[name] = dict(n=int(m.sum()), r2=r2(ms_true[m], ms_pred[m]),
                             rmse=rmse(ms_true[m], ms_pred[m]))
    return out


def coverage_metrics(z, levels=(0.5, 0.9, 0.95)):
    """z = standardized residual (y-mean)/sigma. Empirical central-interval coverage."""
    from scipy.stats import norm
    out = dict(rms_z=float(np.sqrt(np.mean(z ** 2))), mean_abs_z=float(np.mean(np.abs(z))))
    for lv in levels:
        k = norm.ppf(0.5 + lv / 2)
        out[f"cov_{int(lv*100)}"] = float(np.mean(np.abs(z) <= k))
    return out


def main():
    df = D.load()
    feat = D.SHAPE_FEATURES
    ctrl = D.CONTROL_FEATURES
    tr = df[df.split == "train"]; va = df[df.split == "val"]; te = df[df.split == "test_extrap"]
    print(f"train {len(tr)} val {len(va)} test_extrap {len(te)}; "
          f"{len(feat)} shape feats, {len(ctrl)} control feats")

    Xtr = tr[feat].values.astype(np.float64)
    Ttr = np.column_stack([np.log(tr["m_s"].values), np.log(tr["gamma"].values)])

    # ---- train surrogate ensemble + shape map ----
    print("Training surrogate ensemble (heteroscedastic deep ensemble)...")
    models, meta = train_surrogate(Xtr, Ttr, din=len(feat))
    M.save_ensemble(models, meta)
    print("Training control->shape map...")
    smap, smeta = train_shapemap(tr[ctrl].values.astype(np.float64),
                                 tr[feat].values.astype(np.float64), din=len(ctrl), dout=len(feat))
    M.save_shapemap(smap, smeta)

    # ---- accuracy + calibration on val & test ----
    metrics = {"n": dict(train=len(tr), val=len(va), test_extrap=len(te)),
               "features": feat, "controls": ctrl, "target": "log(m_s), log(gamma)"}
    preds_rows = []
    for split_name, dd in [("val", va), ("test_extrap", te)]:
        X = dd[feat].values.astype(np.float64)
        pred = M.ensemble_predict(models, X)
        logms_mean = pred["mean"][:, 0]
        ms_true = dd["m_s"].values
        ms_pred = np.exp(logms_mean)
        logms_true = np.log(ms_true)
        # accuracy (orig units + log units), resolved by regime
        acc = resolved(dd, ms_true, ms_pred)
        acc_log = dict(all_r2_log=r2(logms_true, logms_mean), all_rmse_log=rmse(logms_true, logms_mean))
        # calibration on log m_s
        z = (logms_true - logms_mean) / pred["tot_std"][:, 0]
        cal_all = coverage_metrics(z)
        marg = dd["regime"].values == "marginal"
        cal_marg = coverage_metrics(z[marg]) if marg.sum() >= 5 else None
        metrics[split_name] = dict(accuracy=acc, accuracy_log=acc_log,
                                   calibration_all=cal_all, calibration_marginal=cal_marg)
        # gamma accuracy (log)
        g_pred = pred["mean"][:, 1]; g_true = np.log(dd["gamma"].values)
        metrics[split_name]["gamma_r2_log"] = r2(g_true, g_pred)
        for i in range(len(dd)):
            preds_rows.append(dict(split=split_name, regime=dd["regime"].values[i],
                                   m_s=float(ms_true[i]), m_s_pred=float(ms_pred[i]),
                                   logms_mean=float(logms_mean[i]),
                                   epi_std=float(pred["epi_std"][i, 0]),
                                   ale_std=float(pred["ale_std"][i, 0]),
                                   tot_std=float(pred["tot_std"][i, 0]),
                                   gamma=float(dd["gamma"].values[i]),
                                   leuer=float(dd["leuer"].values[i]),
                                   kappa=float(dd["kappa"].values[i])))

    # ---- baselines ----
    from sklearn.linear_model import LinearRegression
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
    from sklearn.preprocessing import StandardScaler
    from scipy.stats import spearmanr
    from sklearn.metrics import roc_auc_score

    base = {}
    lin = LinearRegression().fit(Xtr, Ttr[:, 0])
    gp_n = min(700, len(tr))
    gidx = np.random.default_rng(0).choice(len(tr), gp_n, replace=False)
    scaler = StandardScaler().fit(Xtr[gidx])
    kern = ConstantKernel(1.0) * RBF(np.ones(len(feat))) + WhiteKernel(0.1)
    gp = GaussianProcessRegressor(kernel=kern, normalize_y=True, n_restarts_optimizer=0,
                                  random_state=0).fit(scaler.transform(Xtr[gidx]), Ttr[gidx, 0])
    for split_name, dd in [("val", va), ("test_extrap", te)]:
        X = dd[feat].values.astype(np.float64)
        logms_true = np.log(dd["m_s"].values)
        ms_true = dd["m_s"].values
        lin_log = lin.predict(X)
        gp_log = gp.predict(scaler.transform(X))
        # Leuer rank baseline: clip robustly, marginal(m_s<0.4)-vs-rest AUC, score = -leuer
        leu = np.clip(dd["leuer"].values, -50, 50)
        is_marg = (ms_true < 0.4).astype(int)
        auc_leuer = float(roc_auc_score(is_marg, -leu)) if 0 < is_marg.sum() < len(is_marg) else None
        nn_logms = np.array([r["logms_mean"] for r in preds_rows if r["split"] == split_name])
        auc_nn = float(roc_auc_score(is_marg, -nn_logms)) if 0 < is_marg.sum() < len(is_marg) else None
        base[split_name] = dict(
            linear_r2_log=r2(logms_true, lin_log), linear_r2_orig=r2(ms_true, np.exp(lin_log)),
            gp_r2_log=r2(logms_true, gp_log), gp_r2_orig=r2(ms_true, np.exp(gp_log)),
            leuer_spearman=float(spearmanr(dd["leuer"].values, ms_true).statistic),
            leuer_marginal_auc=auc_leuer, surrogate_marginal_auc=auc_nn,
            surrogate_r2_log=metrics[split_name]["accuracy_log"]["all_r2_log"],
        )
    metrics["baselines"] = base

    import pandas as pd
    pd.DataFrame(preds_rows).to_parquet("data/phase2_predictions.parquet")
    with open("data/phase2_train_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # ---- console summary ----
    print("\n=== ACCURACY (m_s, original units) ===")
    for s in ("val", "test_extrap"):
        a = metrics[s]["accuracy"]
        print(f" {s}: all R2={a['all']['r2']:.3f} RMSE={a['all']['rmse']:.3f} | "
              f"log R2={metrics[s]['accuracy_log']['all_r2_log']:.3f}")
        for name, _, _ in D.REGIMES:
            if name in a:
                print(f"    {name:11s} n={a[name]['n']:3d} R2={a[name]['r2']:+.3f} RMSE={a[name]['rmse']:.3f}")
    print("\n=== BASELINES (R2 on log m_s) ===")
    for s in ("val", "test_extrap"):
        b = base[s]
        print(f" {s}: surrogate={b['surrogate_r2_log']:.3f} linear={b['linear_r2_log']:.3f} "
              f"gp={b['gp_r2_log']:.3f} | Leuer Spearman={b['leuer_spearman']:.3f} "
              f"AUC(leuer)={b['leuer_marginal_auc']:.3f} AUC(surrogate)={b['surrogate_marginal_auc']:.3f}")
    print("\n=== CALIBRATION (log m_s; coverage of central intervals) ===")
    for s in ("val", "test_extrap"):
        c = metrics[s]["calibration_all"]
        print(f" {s}: RMS_z={c['rms_z']:.2f} cov50={c['cov_50']:.2f} cov90={c['cov_90']:.2f} cov95={c['cov_95']:.2f}")
        cm = metrics[s]["calibration_marginal"]
        if cm:
            print(f"    marginal: RMS_z={cm['rms_z']:.2f} cov90={cm['cov_90']:.2f} (n={int(cm['mean_abs_z']*0+0)})")
    print("\nSaved data/phase2_train_metrics.json, data/phase2_predictions.parquet, models.")


if __name__ == "__main__":
    main()
