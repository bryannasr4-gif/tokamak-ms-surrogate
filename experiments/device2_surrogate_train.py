"""
device2_surrogate_train.py -- train a Device-C-retrained shape->m_s surrogate ensemble (the CLEAN
ABLATION for Phase 5 C4; SECOND_DEVICE.md Step 3b). Same heteroscedastic deep-ensemble recipe as
phase2_train.train_surrogate (8 members, bootstrap, warmup 800, 2600 epochs, Adam 3e-3, wd 1e-4),
trained on an 80-mode Device-C dataset (shape descriptors -> [log m_s, log gamma]). Saves
data/phase2_models/surrogate_C.pt and reports held-out log-R^2 by regime.

The retrained framing reuses the SAME Device-C ShapeMap (shapemap_C); only the surrogate differs from
zero-shot -- so if zero-shot (MAST-U surrogate) wins, the retrained run is the control showing the win
is not an artifact of the surrogate's provenance.

  python experiments/device2_surrogate_train.py --parquet data/device2_ds80_all.parquet --name surrogate_C
"""
import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "experiments"))
import phase2_data as D
import phase2_model as M


def _std_out(model, Xt):
    z = model.trunk(model._z(Xt))
    return model.mean_head(z), model.logvar_head(z)


def r2(y, yhat):
    return float(1 - np.sum((y - yhat) ** 2) / np.sum((y - y.mean()) ** 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", default="data/device2_ds80_all.parquet")
    ap.add_argument("--name", default="surrogate_C")
    ap.add_argument("--val_frac", type=float, default=0.15)
    ap.add_argument("--n_models", type=int, default=8)
    ap.add_argument("--warmup", type=int, default=800)
    ap.add_argument("--epochs", type=int, default=2600)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import pandas as pd
    import torch

    feat = D.SHAPE_FEATURES
    df = pd.read_parquet(os.path.join(ROOT, args.parquet) if not os.path.isabs(args.parquet) else args.parquet)
    df = df[np.isfinite(df["m_s"]) & (df["m_s"] > 0) & np.isfinite(df["gamma"]) & (df["gamma"] > 0)]
    df = df.dropna(subset=feat).reset_index(drop=True)
    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(len(df))
    nval = int(len(df) * args.val_frac)
    va_i, tr_i = idx[:nval], idx[nval:]
    tr, va = df.iloc[tr_i], df.iloc[va_i]
    print(f"Device-C surrogate: {len(tr)} train / {len(va)} val (80-mode labels)")

    Xtr = tr[feat].values.astype(np.float64)
    Ttr = np.column_stack([np.log(tr["m_s"].values), np.log(tr["gamma"].values)])
    xmean, xstd = Xtr.mean(0), Xtr.std(0) + 1e-8
    ymean, ystd = Ttr.mean(0), Ttr.std(0) + 1e-8
    Xt = torch.tensor(Xtr, dtype=torch.float32)
    Ys = (torch.tensor(Ttr, dtype=torch.float32) - torch.tensor(ymean, dtype=torch.float32)) / torch.tensor(ystd, dtype=torch.float32)
    N = len(Xtr)
    models = []
    for mi in range(args.n_models):
        torch.manual_seed(args.seed + mi)
        g = torch.Generator().manual_seed(1000 + mi)
        boot = torch.randint(0, N, (N,), generator=g)
        net = M.Surrogate(xmean, xstd, ymean, ystd, len(feat))
        opt = torch.optim.Adam(net.parameters(), lr=3e-3, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
        xb, yb = Xt[boot], Ys[boot]
        for ep in range(args.epochs):
            opt.zero_grad()
            mu, logvar = _std_out(net, xb)
            if ep < args.warmup:
                loss = ((mu - yb) ** 2).mean()
            else:
                lv = logvar.clamp(-8, 8)
                loss = 0.5 * (torch.exp(-lv) * (mu - yb) ** 2 + lv).mean()
            loss.backward(); opt.step(); sched.step()
        net.eval(); models.append(net)
        print(f"  member {mi} (final loss {loss.item():.4f})", flush=True)
    M.save_ensemble(models, dict(xmean=xmean, xstd=xstd, ymean=ymean, ystd=ystd, din=len(feat)), name=args.name)

    # held-out log-m_s R^2 by regime
    pred = M.ensemble_predict(models, va[feat].values.astype(np.float64))
    logms_pred = pred["mean"][:, 0]
    logms_true = np.log(va["m_s"].values)
    out = dict(name=args.name, n_train=len(tr), n_val=len(va),
               logR2_all=r2(logms_true, logms_pred))
    print(f"\nheld-out log-m_s R^2 (all) = {out['logR2_all']:+.3f}")
    reg = np.array([D.regime_of(m) for m in va["m_s"].values])
    for name, _, _ in D.REGIMES:
        m = reg == name
        if m.sum() >= 5:
            out[f"logR2_{name}"] = r2(logms_true[m], logms_pred[m])
            print(f"   {name:12s} n={int(m.sum()):3d}  logR2 {out[f'logR2_{name}']:+.3f}")
    json.dump(out, open(os.path.join(ROOT, "data", f"device2_{args.name}_eval.json"), "w"), indent=2)
    print(f"Saved surrogate -> {args.name}.pt + eval json")


if __name__ == "__main__":
    main()
