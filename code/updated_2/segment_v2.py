# Copyright (c) Meta Platforms, Inc. and affiliates.
"""
A support-surface segmenter that does not assume the target is table-shaped.

WHAT WAS WRONG WITH THE OLD RULE. `segment_by_support` scans upward for the first height where
the horizontal spread of everything above it falls below 50% of the object's full width, and
calls everything above that the intruder. That encodes one silhouette: wide base, narrow top.

Measured on 19 cluttered cells, the cut height mechanically caps recall — only points above the
cut can ever be labelled intruder — and actual recall tracks that ceiling at r = +0.98. On open
frames and large flat tops (kitchen island, bar cart, utility cart, shelving unit) the spread
never collapses, so the cut lands at the 97th-98th percentile or is never found at all, and
recall collapses to 0-13%. `kitchenisland` scanned its whole range, found nothing, and silently
returned "everything is target".

THE REPLACEMENT. Find the support surface directly instead of inferring it from a width ratio.
A support surface is a height where many points sit AND they spread out horizontally — that is
what a tabletop, a shelf or a cabinet lid looks like in a point cloud, whatever the object's
overall silhouette. Take the HIGHEST such surface (a shelving unit has several; the intruder
rests on the top one) and cut just above it.

This drops the "narrow top" assumption entirely. It keeps the property that actually mattered
about the old rule: it errs towards leaving intruder points in rather than removing target
points, because removing real surface damages the object and under-suppression merely fails to
help.

    python updated_2/segment_v2.py          # benchmark old vs new on every cluttered cell
"""

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree


def segment_by_support_v2(pts, nbins=48, min_pts=12, min_above=8,
                          margin_frac=0.02, spread_floor=0.35, peak_floor=0.5,
                          max_above_extent=0.5, max_above_frac=0.35,
                          comp_frac=0.08, min_comp_pts=6, min_aspect=0.15):
    """
    Split points into target and intruder. Returns a boolean mask, True = target.

    A z-slice is a support-surface candidate when it holds a decent share of the points AND
    those points spread horizontally. `peak_floor` and `spread_floor` are fractions of the
    best slice's count and of the object's full xy-extent respectively, so both are relative
    to the object and neither carries an absolute scale.
    """
    pts = np.asarray(pts, np.float64)
    n = len(pts)
    if n < 2 * min_pts:
        return np.ones(n, bool)

    z, xy = pts[:, 2], pts[:, :2]
    full = float(np.linalg.norm(np.ptp(xy, axis=0)))
    zlo, zhi = float(z.min()), float(z.max())
    if full < 1e-9 or zhi - zlo < 1e-9:
        return np.ones(n, bool)

    edges = np.linspace(zlo, zhi, nbins + 1)
    idx = np.clip(np.digitize(z, edges) - 1, 0, nbins - 1)

    counts = np.zeros(nbins)
    spreads = np.zeros(nbins)
    for b in range(nbins):
        m = idx == b
        c = int(m.sum())
        counts[b] = c
        if c >= 3:
            spreads[b] = float(np.linalg.norm(np.ptp(xy[m], axis=0))) / full

    if counts.max() < 3:
        return np.ones(n, bool)

    # A support surface: populated AND horizontally extended. Both thresholds are relative.
    is_support = (counts >= peak_floor * counts.max()) & (spreads >= spread_floor)
    if not is_support.any():
        return np.ones(n, bool)

    # The highest one — a shelving unit has several and the intruder is on the top shelf.
    b = int(np.max(np.where(is_support)[0]))
    cut = edges[b + 1] + margin_frac * (zhi - zlo)

    above = z > cut
    if above.sum() < min_above:
        return np.ones(n, bool)          # nothing credible above it; claim no intruder

    # SAFETY. Cutting above the wrong surface eats the target: on `kitchenisland` the raw rule
    # gave 10% false positives with 0% recall, strictly worse than the old segmenter. A blanket
    # size guard was tried first and was too blunt — it rejected two genuine detections
    # (`woodenbox` 80% -> 0%) while still missing `barcart`'s 11% false positives.
    #
    # The discriminator that actually separates them is SHAPE. An intruder resting on a surface
    # is ONE compact blob. A cut through an open frame (a cart, a shelving unit) leaves scattered
    # rim and upright fragments instead. So keep only the largest connected component above the
    # cut, and require it to look like a blob: a clear majority of the above-points, covering
    # part of the surface rather than all of it, and a minority of the object.
    #
    # Under-suppression merely fails to help; removing real surface damages the object. The two
    # errors are not equally bad, so every check below abstains rather than guesses.
    ai = np.where(above)[0]
    P = pts[ai]
    rad = comp_frac * float(np.linalg.norm(np.ptp(pts, axis=0)))
    pairs = cKDTree(P).query_pairs(rad, output_type="ndarray")
    if len(pairs) == 0:
        return np.ones(n, bool)
    g = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(len(P), len(P)))
    _ncomp, lab = connected_components(g, directed=False)

    # EVERY compact cluster above the surface is an intruder, not just the biggest one.
    #
    # The previous version kept only the largest component and abstained when it was under 60%
    # of the above-points. That was written when the test data had exactly one intruder, and it
    # fails the realistic case badly: with three things on a table the largest cluster is about a
    # third of the above-points, so the rule abstained and removed NOTHING. Even when it did
    # fire it could only ever remove one object.
    #
    # Judging each cluster on its own also keeps the safety property that mattered. A cut through
    # an open frame leaves rim and upright fragments that are individually wide, so they fail the
    # extent test one by one; a real object resting on the surface is individually compact and
    # passes. Nothing here needs to know WHAT the objects are, or how many there are.
    keep = np.ones(n, bool)
    removed = 0
    for c in np.unique(lab):
        sel = ai[lab == c]
        if len(sel) < min_comp_pts:
            continue                                    # too small to trust; leave it alone
        ext_c = float(np.linalg.norm(np.ptp(xy[sel], axis=0))) / full
        if ext_c > max_above_extent:
            continue                                    # spans the surface -> frame, not an object
        # A thing STANDING on a surface has height. A sliver of the surface's own rim caught
        # above the cut is a pancake: wide but flat. Judging that per component is what stops
        # `kitchenisland` from having its own tabletop edge shaved off (10% false positives when
        # this check was missing), without touching any real object.
        zc = float(np.ptp(pts[sel, 2]))
        xyc = max(float(np.linalg.norm(np.ptp(xy[sel], axis=0))), 1e-9)
        if zc / xyc < min_aspect:
            continue
        keep[sel] = False
        removed += len(sel)
    if removed < min_above or removed / n > max_above_frac:
        return np.ones(n, bool)                         # nothing credible, or implausibly much
    return keep


if __name__ == "__main__":
    import glob
    import os
    import pickle
    import sys

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from latent_seg import segment_by_support

    rows = []
    for pat, tag in (("updated_2/outputs/stage1/clutter/*__both__L025.pkl", "orig"),
                     ("updated_2/confirm/outputs/clutter/*__both__L025.pkl", "fresh")):
        for p in sorted(glob.glob(pat)):
            lab = p.replace(".pkl", "__label.npz")
            if not os.path.exists(lab):
                continue
            d = pickle.load(open(p, "rb"))
            z = np.load(lab)
            pts = d["points_model"].numpy().astype(np.float64)[:, :3]
            ntar = int(z["n_target_points"])
            is_i = np.zeros(len(pts), bool)
            is_i[ntar:] = True
            if is_i.sum() < 5:
                continue
            r = {}
            for nm, fn in (("old", segment_by_support), ("new", segment_by_support_v2)):
                mask = fn(pts)
                r[nm] = (float((~mask & is_i).sum()) / is_i.sum() * 100,
                         float((~mask & ~is_i).sum()) / max((~is_i).sum(), 1) * 100)
            rows.append((os.path.basename(p).replace("__both__L025.pkl", ""), tag, r))

    print(f"{'object':28}{'set':6}{'OLD recall':>12}{'OLD fp':>9}{'NEW recall':>12}{'NEW fp':>9}"
          f"{'change':>9}")
    for o, t, r in sorted(rows, key=lambda x: x[2]["old"][0]):
        d = r["new"][0] - r["old"][0]
        print(f"{o:28}{t:6}{r['old'][0]:>11.1f}%{r['old'][1]:>8.2f}%"
              f"{r['new'][0]:>11.1f}%{r['new'][1]:>8.2f}%{d:>+9.1f}")

    for nm in ("old", "new"):
        rec = np.array([r[nm][0] for _, _, r in rows])
        fp = np.array([r[nm][1] for _, _, r in rows])
        print(f"\n{nm.upper():4} mean recall {rec.mean():5.1f}%   mean false-pos {fp.mean():.2f}%   "
              f"cells below 50% recall: {int((rec < 50).sum())}/{len(rows)}")
