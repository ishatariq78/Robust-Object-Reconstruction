"""
Experiment 1: Point Cloud Quality Degradation
Generates 5 degraded variants of each input pkl for evaluation.

Conditions:
  baseline   - original point cloud (100% points, no noise)
  sparse_50  - 50% of points, randomly subsampled
  sparse_25  - 25% of points, randomly subsampled
  sparse_10  - 10% of points, randomly subsampled
  noisy      - 100% points + Gaussian noise (sigma = 5% of object metric extent)

Usage:
    python experiments/exp1_pointcloud_degradation/prepare_data.py \
        --input_pkls data/ADT1292__stool.pkl data/BNB2909__pitcher.pkl \
        --output_dir data/exp1_pointcloud
"""

import argparse
import pickle
from pathlib import Path

import numpy as np
import torch


def subsample(points, inv_dist_std, dist_std, fraction, rng):
    n = len(points)
    k = max(3, int(n * fraction))
    idx = rng.choice(n, size=k, replace=False)
    idx.sort()
    return points[idx], inv_dist_std[idx], dist_std[idx]


def add_noise(points, bounds, sigma_fraction=0.05):
    sigma = sigma_fraction * float(np.max(bounds))
    noise = rng_global.standard_normal(points.shape).astype(np.float32) * sigma
    return points + noise


rng_global = np.random.default_rng(42)


def make_variants(input_path, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(input_path).stem

    with open(input_path, "rb") as f:
        pkl = pickle.load(f)

    pts = pkl["points_model"].numpy().copy().astype(np.float32)
    ids = pkl["inv_dist_std"].numpy().copy().astype(np.float32)
    ds = pkl["dist_std"].numpy().copy().astype(np.float32)
    bounds = pkl["bounds"].numpy()

    rng = np.random.default_rng(42)

    conditions = {
        "baseline":  (pts.copy(), ids.copy(), ds.copy()),
        "sparse_50": subsample(pts, ids, ds, 0.50, rng),
        "sparse_25": subsample(pts, ids, ds, 0.25, rng),
        "sparse_10": subsample(pts, ids, ds, 0.10, rng),
        "noisy":     (add_noise(pts, bounds, sigma_fraction=0.05), ids.copy(), ds.copy()),
    }

    saved = []
    for cond, (p, i, d) in conditions.items():
        variant = dict(pkl)
        variant["points_model"] = torch.from_numpy(p)
        variant["inv_dist_std"] = torch.from_numpy(i)
        variant["dist_std"] = torch.from_numpy(d)

        out_path = output_dir / f"{stem}__{cond}.pkl"
        with open(out_path, "wb") as f:
            pickle.dump(variant, f)
        print(f"  [{cond}] {len(p)} points -> {out_path}")
        saved.append(str(out_path))

    return saved


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_pkls", nargs="+", required=True)
    parser.add_argument("--output_dir", default="data/exp1_pointcloud")
    args = parser.parse_args()

    all_saved = []
    for path in args.input_pkls:
        print(f"\nProcessing: {path}")
        all_saved.extend(make_variants(path, args.output_dir))

    print(f"\nGenerated {len(all_saved)} pkl files in {args.output_dir}")


if __name__ == "__main__":
    main()
