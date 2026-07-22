"""
device2_assemble.py -- concatenate device2_{prefix}_chunk_*.json record lists into one parquet.

  python experiments/device2_assemble.py --prefix device2_shapegen --out data/device2_shapegen_all.parquet
"""
import argparse
import glob
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    import pandas as pd
    rows = []
    files = sorted(glob.glob(os.path.join(ROOT, "data", f"{args.prefix}_chunk_*.json")))
    for fp in files:
        rows.extend(json.load(open(fp)).get("recs", []))
    df = pd.DataFrame(rows)
    out = args.out if os.path.isabs(args.out) else os.path.join(ROOT, args.out)
    df.to_parquet(out)
    print(f"assembled {len(df)} rows from {len(files)} chunks -> {out}")
    if "m_s" in df.columns:
        print(f"  m_s range [{df['m_s'].min():.4f}, {df['m_s'].max():.3f}]  "
              f"kappa range [{df['kappa'].min():.3f}, {df['kappa'].max():.3f}]")
    else:
        print(f"  (forward-only; kappa range [{df['kappa'].min():.3f}, {df['kappa'].max():.3f}])")


if __name__ == "__main__":
    main()
