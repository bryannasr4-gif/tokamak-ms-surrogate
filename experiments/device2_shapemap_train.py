"""
device2_shapemap_train.py -- train a Device-C ShapeMap (CONTROL -> SHAPE descriptors) for the
zero-shot transfer framing (Phase 5 / Device-C). The shape->m_s surrogate transfers from
MAST-U UNCHANGED (m_s is shape-determined; Portone cross-check confirms the convention holds on
Device-C); but the controls->shape map is machine-specific, so we train a Device-C ShapeMap on the
kill-gate probe samples (controls->descriptors; cheap, NO new solves -- shape descriptors are
mode-independent so the 40-mode probe is fine for this). Composing surrogate(ShapeMap_C(u)) gives a
differentiable, amortized m_s(controls) for Device-C.

Same recipe as phase2_train.train_shapemap (standardized MSE, Adam lr=2e-3, 3000 epochs, wd=1e-5,
seed 0). Reports held-out per-descriptor R^2. Run thread-pinned.

  python experiments/device2_shapemap_train.py [--parquets a.parquet b.parquet] [--name shapemap_C]
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


def r2(y, yhat):
    return float(1 - np.sum((y - yhat) ** 2) / np.sum((y - y.mean()) ** 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquets", nargs="+",
                    default=[os.path.join(ROOT, "data", "device2_probe.parquet")])
    ap.add_argument("--name", default="shapemap_C")
    ap.add_argument("--val_frac", type=float, default=0.2)
    ap.add_argument("--epochs", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import pandas as pd
    import torch

    df = pd.concat([pd.read_parquet(p) for p in args.parquets], ignore_index=True)
    feat = D.SHAPE_FEATURES
    ctrl = D.CONTROL_FEATURES
    df = df.dropna(subset=feat + ctrl).reset_index(drop=True)
    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(len(df))
    nval = int(len(df) * args.val_frac)
    va_i, tr_i = idx[:nval], idx[nval:]
    tr, va = df.iloc[tr_i], df.iloc[va_i]
    print(f"Device-C ShapeMap: {len(tr)} train / {len(va)} val; {len(ctrl)} controls -> {len(feat)} descriptors")

    Xtr = tr[ctrl].values.astype(np.float64)
    Ytr = tr[feat].values.astype(np.float64)
    xmean, xstd = Xtr.mean(0), Xtr.std(0) + 1e-8
    ymean, ystd = Ytr.mean(0), Ytr.std(0) + 1e-8
    Xt = torch.tensor(Xtr, dtype=torch.float32)
    Ys = (torch.tensor(Ytr, dtype=torch.float32) - torch.tensor(ymean, dtype=torch.float32)) / torch.tensor(ystd, dtype=torch.float32)
    torch.manual_seed(args.seed)
    net = M.ShapeMap(xmean, xstd, ymean, ystd, len(ctrl), len(feat))
    opt = torch.optim.Adam(net.parameters(), lr=2e-3, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    for ep in range(args.epochs):
        opt.zero_grad()
        z = (Xt - net.xmean) / net.xstd
        loss = ((net.net(z) - Ys) ** 2).mean()
        loss.backward(); opt.step(); sched.step()
    net.eval()
    smeta = dict(xmean=xmean, xstd=xstd, ymean=ymean, ystd=ystd, din=len(ctrl), dout=len(feat))
    M.save_shapemap(net, smeta, name=args.name)
    print(f"  trained (final standardized MSE {loss.item():.4f}) -> {args.name}.pt")

    # held-out per-descriptor R^2
    with torch.no_grad():
        pred = net(torch.tensor(va[ctrl].values, dtype=torch.float32)).numpy()
    Yva = va[feat].values.astype(np.float64)
    per = {f: r2(Yva[:, i], pred[:, i]) for i, f in enumerate(feat)}
    key = ["kappa", "sq_uo", "sq_lo", "delta", "gap_outer", "gap_inner", "li", "betap"]
    print("\nheld-out R^2 (key levers):")
    for f in key:
        print(f"   {f:10s} {per[f]:+.3f}")
    print(f"\n   mean R^2 over all {len(feat)} descriptors = {np.mean(list(per.values())):+.3f}")
    json.dump(dict(name=args.name, n_train=len(tr), n_val=len(va), per_descriptor_r2=per,
                   mean_r2=float(np.mean(list(per.values()))), final_mse=float(loss.item())),
              open(os.path.join(ROOT, "data", f"device2_{args.name}_eval.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
