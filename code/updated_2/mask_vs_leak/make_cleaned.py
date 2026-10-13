# Copyright (c) Meta Platforms, Inc. and affiliates.
"""
Build cluttered pickles with the intruder's points removed — by the SEGMENTER, not by an oracle.

WHY. The mask fix alone cannot win, and the reason is structural rather than a tuning failure:

  * it locates the intruder by splitting the POINT cloud, so it needs the intruder to be in the
    points; but
  * when the intruder is in the points, the point branch builds it regardless of what the mask
    says. Four experiments now agree that image-side signals barely move ShapeR's geometry.
  * in image-only clutter, where the mask would decide the outcome, there are no intruder points
    for the segmenter to find, so there is nothing to make negative clicks from.

So the mask is either outvoted or blind. The combination is the only version that can work:
take the intruder out of the points AND tell the mask it is not the target, so neither branch is
instructed to build it.

The removal uses `segment_by_support_v2` on the cloud alone — no label, nothing an inference-time
system would not have. The label is loaded only to REPORT how accurate the removal was.

    python updated_2/mask_vs_leak/make_cleaned.py
"""

import pickle
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "updated_2"))
from segment_v2 import segment_by_support_v2                     # noqa: E402

ALIGNED = ("points_model", "inv_dist_std", "dist_std")
SOURCES = ["updated_2/outputs/stage1/clutter/ADT1292__woodentable__both__L025.pkl",
           "updated_2/outputs/stage1/clutter/HWD0045__desktable__both__L025.pkl",
           "updated_2/outputs/stage1/clutter/ADT1292__coffeetable__both__L025.pkl",
           "updated_2/outputs/stage1/clutter/ADT1292__nightstand__both__L025.pkl",
           "updated_2/confirm/outputs/clutter/HWD0045__woodentable__both__L025.pkl",
           "updated_2/confirm/outputs/clutter/BNB2909__coffeetable__both__L025.pkl",
           "updated_2/confirm/outputs/clutter/ADT1292__tvstand__both__L025.pkl",
           "updated_2/confirm/outputs/clutter/BNB1182__woodenbox__both__L025.pkl"]


def main():
    out = HERE / "cleaned_pkls"
    out.mkdir(parents=True, exist_ok=True)
    print(f"{'object':30}{'points':>9}{'removed':>9}{'recall':>9}{'false-pos':>11}")
    for src in SOURCES:
        p = REPO / src
        if not p.exists():
            print(f"  missing {src}")
            continue
        d = pickle.load(open(p, "rb"))
        pts = d["points_model"].numpy().astype(np.float64)[:, :3]
        n = len(pts)
        for k in ALIGNED:
            assert len(d[k]) == n, f"{p.name}: {k} not aligned with points_model"

        keep = segment_by_support_v2(pts)                # True = target. No label used.
        if keep.all():
            print(f"{p.stem:30}{n:>9}{0:>9}{'--':>9}{'--':>11}   segmenter abstained")
            continue

        e = dict(d)
        idx = np.where(keep)[0]
        for k in ALIGNED:
            v = d[k]
            e[k] = v[torch.from_numpy(idx)] if isinstance(v, torch.Tensor) else v[idx]
        dst = out / f"{p.stem}__cleaned.pkl"
        pickle.dump(e, open(dst, "wb"))

        lab = p.with_name(p.stem + "__label.npz")
        rec = fp = float("nan")
        if lab.exists():
            z = np.load(lab)
            ntar = int(z["n_target_points"])
            truth = np.zeros(n, bool)
            truth[:ntar] = True
            rec = float((~keep & ~truth).sum()) / max((~truth).sum(), 1) * 100
            fp = float((~keep & truth).sum()) / max(truth.sum(), 1) * 100
        print(f"{p.stem:30}{n:>9}{int((~keep).sum()):>9}{rec:>8.0f}%{fp:>10.2f}%")
    print(f"\nwrote cleaned pickles to {out}")


if __name__ == "__main__":
    main()
