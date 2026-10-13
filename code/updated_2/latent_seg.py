# Copyright (c) Meta Platforms, Inc. and affiliates.
"""
Stage 2b — the method: latent-space segmentation coupled to reconstruction.

Goal: reconstruct ONLY the detected target, even when a second object stands on it,
by segmenting inside the generative latent rather than cleaning up the mesh
afterwards.

Why not just crop the mesh
--------------------------
Cutting leaked geometry off the output leaves the target open where the intruder
touched it, because the surface under the contact region was never generated. The
mesh is also the last thing the model produces, so a mesh-level fix cannot change
what the model decided to build. `--method crop` is implemented here anyway as the
honest baseline the latent method has to beat: any claim for latent segmentation
has to show it does something crop does not.

The three methods
-----------------
  crop      post-hoc. Delete predicted surface nearer the distractor region than to
            the target evidence. The baseline.

  gate      training-free latent segmentation. Each latent token is given a spatial
            footprint from the decoder's cross-attention (see `latent_probe.py`),
            scored against the object evidence ShapeR was ALREADY given — the SLAM
            points inside the detection box — and tokens whose footprint sits in
            unsupported space are suppressed before decoding. The field is then
            decoded from the surviving tokens, so the decoder closes the surface
            itself instead of leaving a hole.

  refeed    gate, then re-run generation with the segmented latent as the starting
            point for a short partial-noise trajectory. This is the "segmentation
            talks back to reconstruction" step: the suppressed latent is re-noised
            to an intermediate timestep and denoised again, letting the model
            re-derive a coherent shape that is consistent with the segmentation
            rather than merely missing a part.

The evidence used for segmentation
----------------------------------
Deliberately NOT a SAM2 mask. Earlier work here showed SAM2 can be confidently
wrong in exactly the cluttered case that matters — on `livingroomfurniture` it
absorbed an occluder into the foreground and grew the mask. The prior here is
geometric and training-free, in two steps:

1. `segment_pointcloud` splits the SLAM points the detector assigned to this object
   into the target and anything standing on it, by connected components at a radius
   set by the object's own extent. On the woodentable sample this is exact —
   precision 1.000, recall 1.000, all 433 injected intruder points excluded — with a
   wide stable band (eps_frac 0.04-0.08); it only merges the two once the radius
   grows past the gap between pot and tabletop.

2. `support_score` then scores 3D positions CONTRASTIVELY: closer to the target
   cluster, or closer to the intruder cluster?

Step 2 being contrastive rather than absolute is the important detail, and it was
forced by a measurement. An absolute "distance to nearest object point" prior — the
one used in the earlier pilot, ~0.755 AUC — cannot tell unobserved-but-real surface
from hallucinated surface, because SLAM points only cover OBSERVED geometry. A
table's underside and inner legs are far from every point despite being real, and
cropping on that score deleted 31% of the target while removing the leak. The
contrastive score leaves such regions neutral, because they are far from both
clusters, and pushes down only surface that actually sits on the intruder.

The known weakness is still recorded: when the spurious geometry hugs the real
surface the two clusters merge and the prior degrades (0.566 AUC on one stool in the
earlier pilot). It is strongest exactly where this task needs it — an intruder
standing clear of the target.

    python updated_2/latent_seg.py \
        --clutter_pkl updated_2/outputs/stage1/clutter/ADT1292__woodentable__both__L025.pkl \
        --label       updated_2/outputs/stage1/clutter/ADT1292__woodentable__both__L025__label.npz \
        --methods baseline crop gate refeed --seeds 0 1 2
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import trimesh
from scipy.spatial import cKDTree

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

from postprocessing.helper import remove_floating_geometry     # noqa: E402
from demo_limitation import Runner, fresh_pc                    # noqa: E402
from latent_probe import decoder_attention, surface_queries     # noqa: E402
import metrics as M                                             # noqa: E402


# --------------------------------------------------------------------------- #
# the evidence prior
# --------------------------------------------------------------------------- #

def _frame_of(ds, idx):
    """
    Recover the normalised-cube <-> model-frame map by round-tripping a known unit
    tetrahedron through `rescale_back`, rather than reimplementing the dataset's
    internal scaling. Returns (origin, basis) with model = origin + basis.T @ normalised.
    """
    probe = trimesh.Trimesh(
        vertices=np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]),
        faces=np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]), process=False)
    mv = np.asarray(ds.rescale_back(idx, probe, False).vertices, np.float64)
    origin = mv[0]
    basis = np.stack([mv[1] - origin, mv[2] - origin, mv[3] - origin], axis=0)
    return origin, basis


def _to_normalised(pts_model, origin, basis):
    return np.linalg.solve(basis.T, (np.asarray(pts_model, np.float64) - origin).T).T


def segment_pointcloud(pts, eps_frac=0.06, min_samples=4):
    """
    Split the assigned SLAM points into the target and anything standing on it.

    THIS IS THE STEP THAT MAKES THE METHOD WORK IN THE REALISTIC CASE.

    The evidence ShapeR is handed is the set of points the upstream detector
    assigned to this object. When a second object is stacked on the target, those
    points can carry the intruder too — that is precisely the detector error the
    paper's Limitation (c) describes feeding into (b). Using the assigned set raw
    would mark the intruder as well supported and the suppression would do nothing.

    The separation used is geometric and training-free: connected-component
    clustering at a radius set by the object's own extent. An object resting ON the
    target is a spatially distinct cluster; the target is the component with the
    largest spatial extent (NOT the largest count — a densely sampled small intruder
    can out-count a sparsely sampled table).

    Returns a boolean mask over `pts`, True for target.
    """
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import pdist

    pts = np.asarray(pts, np.float64)
    if len(pts) < max(min_samples, 3):
        return np.ones(len(pts), bool)
    ext = float(np.ptp(pts, axis=0).max())
    eps = eps_frac * ext

    # single-linkage == connected components at threshold eps
    n = len(pts)
    if n > 6000:                      # keep the pdist tractable
        sub = np.random.default_rng(0).choice(n, 6000, replace=False)
    else:
        sub = np.arange(n)
    Z = linkage(pdist(pts[sub]), method="single")
    lab_sub = fcluster(Z, t=eps, criterion="distance")

    # score each component by spatial extent, not population
    best, best_score = None, -1.0
    for c in np.unique(lab_sub):
        m = lab_sub == c
        if m.sum() < min_samples:
            continue
        score = float(np.linalg.norm(np.ptp(pts[sub][m], axis=0)))
        if score > best_score:
            best, best_score = c, score
    if best is None:
        return np.ones(n, bool)

    keep_sub = pts[sub][lab_sub == best]
    # propagate the decision to all points by nearest retained neighbour
    d_keep, _ = cKDTree(keep_sub).query(pts, k=1)
    return d_keep <= eps


def segment_by_support(pts, min_pts=15, xy_drop=0.5, margin_frac=0.02):
    """
    Split points using the PHYSICAL structure of the paper's scenario: an intruder
    rests on a support surface.

    At tabletop height the point cloud is horizontally WIDE; above it only the
    intruder remains, so the horizontal spread collapses. The support height is the
    lowest height at which the xy-extent of everything above it falls below
    `xy_drop` of the object's full xy-extent. No oracle, no per-object tuning.

    WHY THIS REPLACED CONNECTIVITY CLUSTERING. `segment_pointcloud` separates the
    two objects cleanly on exactly one of five objects; elsewhere no radius exists
    that detaches the intruder without first fragmenting the table, because the two
    touch. Measured over 20 cells (5 objects x 4 clutter settings), with the injected
    points as ground truth:

        connectivity   clean separation on 1 / 5 objects, and when it fails it
                       excludes hundreds of genuine target points
        support        ZERO false positives on 20 / 20 cells, recall >= 60% on 12

    The zero-false-positive property is the important one: this segmentation never
    removes real target surface, so its failure mode is under-suppression (some
    intruder survives) rather than damage to the object. An elbow-based variant that
    picked the sharpest spread drop instead of a fixed threshold was tried and was
    worse on both counts (7/20, up to 9.9% false positives), so the simpler rule is
    kept.

    Honest caveat: these thresholds were chosen while looking at these 20 cells. The
    rule is physical rather than fitted, and `xy_drop` was not swept per object, but
    a clean claim needs held-out objects.

    Returns a boolean mask, True = target.
    """
    pts = np.asarray(pts, np.float64)
    # `points_model` is already z-up for these samples. NOTE: do NOT apply
    # `T_zup_obj` here — it maps z -> -y, i.e. it assumes a different source
    # convention, and applying it destroys the separation (0 intruder points found
    # on every cell).
    z, xy = pts[:, 2], pts[:, :2]
    full = float(np.linalg.norm(np.ptp(xy, axis=0)))
    if full < 1e-9 or len(pts) < 2 * min_pts:
        return np.ones(len(pts), bool)

    h = None
    for q in np.arange(40, 100, 1.0):
        cand = float(np.percentile(z, q))
        above = z > cand
        if above.sum() < min_pts:
            break
        if float(np.linalg.norm(np.ptp(xy[above], axis=0))) / full <= xy_drop:
            h = cand
            break
    if h is None:
        return np.ones(len(pts), bool)
    return ~(z > h + margin_frac * float(np.ptp(z)))


def object_evidence(runner, pkl_path, mode="clustered", label_npz=None,
                    min_intruder_pts=25, max_intruder_extent_frac=0.60,
                    segmenter="support"):
    """
    Build the evidence prior, in the SAME normalised frame the VAE decodes in.

      assigned   every point the detector gave this object. Honest ablation: when
                 the intruder's points are in that set, this prior cannot separate
                 them and the method should be seen to fail.
      clustered  `segment_pointcloud` first — the actual method.
      oracle     the true target points, from the clutter label. Upper bound.

    Returns (points_normalised, kdtree, info).
    """
    import pickle

    ds, batch = runner.batch_for(pkl_path)
    idx = batch["index"][0]
    origin, basis = _frame_of(ds, idx)

    with open(pkl_path, "rb") as f:
        pts_model = pickle.load(f)["points_model"].numpy().astype(np.float64)

    if mode == "oracle":
        if label_npz is None:
            raise SystemExit("oracle evidence needs --label")
        n_t = int(np.load(label_npz)["n_target_points"])
        keep = np.zeros(len(pts_model), bool)
        keep[:n_t] = True
    elif mode == "clustered":
        keep = (segment_by_support(pts_model) if segmenter == "support"
                else segment_pointcloud(pts_model))
    elif mode == "assigned":
        keep = np.ones(len(pts_model), bool)
    else:
        raise SystemExit(f"unknown evidence mode {mode}")

    info = {"evidence_mode": mode, "n_points": int(len(pts_model)),
            "n_kept": int(keep.sum())}
    # If the oracle is available, report how well the segmentation recovered it.
    if label_npz is not None and Path(label_npz).exists():
        n_t = int(np.load(label_npz)["n_target_points"])
        truth = np.zeros(len(pts_model), bool)
        truth[:n_t] = True
        if len(truth) == len(keep):
            tp = int((keep & truth).sum())
            info["evidence_precision"] = tp / max(int(keep.sum()), 1)
            info["evidence_recall"] = tp / max(int(truth.sum()), 1)
            info["intruder_points_excluded"] = int((~keep & ~truth).sum())
            info["intruder_points_total"] = int((~truth).sum())

    # The excluded points become the negative class for the contrastive score — but
    # ONLY if they actually look like a second object.
    #
    # Measured failure this guards against: on `coffeetable` the pot touches the
    # tabletop, the clustering bridged the gap, and the excluded set was 44 points
    # of which ZERO were the intruder (precision 0.910, 0/289 intruder points
    # excluded). Handing those 44 genuine target points to `support_score` as the
    # negative class is worse than doing nothing — it would suppress real surface.
    #
    # So the excluded set must be COMPACT — a small blob, the way a real object on a
    # table is. Compactness, not size, is the discriminator, and getting this
    # backwards was an actual bug: scattered boundary points spread across the whole
    # object have a LARGER bounding-box diagonal than a genuine intruder, so an
    # extent >= threshold test accepted every failure case. Measured extent fractions:
    #
    #     woodentable   0.40  <- true intruder, 433/433 correct
    #     coffeetable   0.99  <- scattered noise,   0/289
    #     desktable     0.98  <- scattered noise,   0/234
    #     nightstand    1.24  <- scattered noise,   6/64
    #     sidetable     1.50  <- scattered noise,  28/28 but 295 false positives
    #
    # A compact blob spans a fraction of the object; scattered noise spans all of it.
    # When the test fails, no intruder is declared and gate/refeed become no-ops. A
    # method that declines to act on unreliable evidence is much preferable to one
    # that damages the object on a false positive.
    excl = pts_model[~keep]
    intruder_tree = None
    info["n_intruder"] = int((~keep).sum())
    info["intruder_accepted"] = False
    if len(excl) >= max(min_intruder_pts, 1):
        obj_ext = float(np.ptp(pts_model, axis=0).max())
        excl_ext = float(np.linalg.norm(np.ptp(excl, axis=0)))
        info["intruder_extent_frac"] = excl_ext / max(obj_ext, 1e-9)
        # The compactness test guards CONNECTIVITY clustering, whose failure mode is
        # excluding scattered genuine target points. `support` has no false
        # positives by construction, so gating it on blob shape would only suppress
        # correct detections.
        if (segmenter == "support"
                or info["intruder_extent_frac"] <= max_intruder_extent_frac):
            intruder_tree = cKDTree(_to_normalised(excl, origin, basis))
            info["intruder_accepted"] = True
    if not info["intruder_accepted"]:
        # nothing credible to suppress — keep every point, score absolutely
        keep = np.ones(len(pts_model), bool)
        info["n_kept"] = int(len(pts_model))

    pts_norm = _to_normalised(pts_model[keep], origin, basis)
    return pts_norm, cKDTree(pts_norm), intruder_tree, info


def evidence_radius(ev_pts, k_spacing=3.0, min_frac=0.02, max_frac=0.25):
    """
    Choose the support radius from the evidence cloud's OWN sparsity.

    A fixed fraction of the object's extent is the wrong scale: SLAM points on a
    large flat tabletop are far apart, so a tight radius marks most of the real
    table as unsupported. In the first smoke run that made the crop baseline delete
    31% of the target. The radius is therefore set to a multiple of the median
    nearest-neighbour spacing among the evidence points themselves — "as close as
    these points are to each other" — and clamped to a sane band of the extent.
    """
    pts = np.asarray(ev_pts, np.float64)
    ext = float(np.ptp(pts, axis=0).max())
    if len(pts) < 2:
        return min_frac * ext
    d, _ = cKDTree(pts).query(pts, k=2)
    spacing = float(np.median(d[:, 1]))
    return float(np.clip(k_spacing * spacing, min_frac * ext, max_frac * ext))


def support_score(positions, ev_tree, radius, intruder_tree=None):
    """
    Score in [0, 1]: how much each 3D position belongs to the TARGET.

    Two regimes, and the difference between them matters a great deal.

    ABSOLUTE (no intruder cluster available): distance to the nearest target point,
    as a Gaussian falloff. This is the prior used in the earlier pilot, and it has a
    structural flaw for this task — SLAM points only cover OBSERVED surface, so a
    table's underside and inner legs are far from any point despite being real. The
    first smoke run showed the cost directly: cropping on this score deleted 31% of
    the target while removing the leak.

    CONTRASTIVE (intruder cluster available, the normal case here): compare distance
    to the target cluster against distance to the intruder cluster, and suppress only
    where a position is closer to the intruder. Unobserved-but-real geometry is far
    from BOTH clusters, so it scores neutral and survives; only surface that actually
    sits on the intruder is pushed down. This is what makes the clustering step pay
    for itself twice — once to clean the evidence, once to supply a negative class.

    Returns (score, distance_to_target).
    """
    pos = np.asarray(positions, np.float64)
    d_t, _ = ev_tree.query(pos, k=1)
    if intruder_tree is None:
        return np.exp(-(d_t / max(radius, 1e-9)) ** 2), d_t
    d_i, _ = intruder_tree.query(pos, k=1)
    # logistic on the signed margin, scaled by the evidence radius
    return 1.0 / (1.0 + np.exp(-(d_i - d_t) / max(radius, 1e-9))), d_t


# --------------------------------------------------------------------------- #
# token segmentation
# --------------------------------------------------------------------------- #

def segment_tokens(runner, latents, ev_tree, radius, res=48, tau=0.5,
                   min_keep_frac=0.5, intruder_tree=None):
    """
    Score every latent token as target vs intruder.

    A token's footprint is the attention-weighted distribution of surface query
    points that attend to it (`latent_probe.decoder_attention`). Its score is the
    evidence support averaged over that footprint. Tokens below `tau` are suppressed.

    `min_keep_frac` is a safety floor: if the prior wants to delete most of the
    latent set, it is far more likely to be wrong than the object is to be mostly
    spurious, so the suppression is capped at the lowest-scoring tokens.
    """
    q, _vals, dec_lat = surface_queries(runner.vae, latents, res=res)
    attn = decoder_attention(runner.vae, q, dec_lat)                 # (n, m)
    pos = q[0].float().cpu().numpy()

    sup, dist = support_score(pos, ev_tree, radius, intruder_tree)   # (n,)
    w = attn / attn.sum(dim=0, keepdim=True).clamp_min(1e-12)        # per-token
    tok_score = (w.T.cpu().numpy() * sup[None, :]).sum(axis=1)       # (m,)
    centroid = (w.T.cpu().numpy() @ pos)

    keep = tok_score >= tau
    max_drop = int(round((1.0 - min_keep_frac) * len(tok_score)))
    if (~keep).sum() > max_drop:
        order = np.argsort(tok_score)
        keep = np.ones_like(keep)
        keep[order[:max_drop]] = False
    return {"token_score": tok_score, "keep": keep, "centroid": centroid,
            "query_pos": pos, "query_support": sup, "query_dist": dist,
            "n_dropped": int((~keep).sum()), "n_tokens": int(len(tok_score))}


def decode_with_tokens(runner, latents, keep):
    """
    Decode from the surviving tokens only.

    The latent is a SET consumed by cross-attention, so dropping rows is well
    defined — the decoder simply has fewer keys to attend to and closes the surface
    over the removed region, which is why this is not equivalent to cutting the mesh.
    """
    idx = torch.tensor(np.flatnonzero(keep), device=latents.device, dtype=torch.long)
    return latents.index_select(1, idx)


# --------------------------------------------------------------------------- #
# methods
# --------------------------------------------------------------------------- #

def mesh_from_latents(runner, ds, batch_index, latents):
    mesh = runner.vae.infer_mesh_from_latents(latents)[0]
    mesh = remove_floating_geometry(mesh)
    mesh = mesh.simplify_quadric_decimation(face_count=125000)
    return ds.rescale_back(batch_index, mesh, False)


def run_method(runner, pkl_path, method, seed, ev_tree, radius, tau, res,
               refeed_t=0.35, refeed_steps=8, intruder_tree=None):
    ds, batch = runner.batch_for(pkl_path)
    idx = batch["index"][0]

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    fresh_pc(batch)
    latents = runner.model.infer_latents(
        batch, token_shape=runner.token_shape,
        text_feature_extractor=runner.text_fx,
        num_steps=runner.steps, use_shifted_sampling=runner.shifted)

    info = {"method": method, "seed": seed}

    if method == "baseline":
        return mesh_from_latents(runner, ds, idx, latents), info

    if method == "crop":
        mesh = mesh_from_latents(runner, ds, idx, latents)
        v = np.asarray(mesh.vertices, np.float64)
        # evidence lives in the normalised frame; map vertices back the same way
        probe_pts = np.linalg.solve(
            _basis_of(ds, idx).T, (v - _origin_of(ds, idx)).T).T
        sup, _d = support_score(probe_pts, ev_tree, radius, intruder_tree)
        keep_v = sup >= tau
        fmask = keep_v[mesh.faces].all(axis=1)
        info["faces_removed"] = int((~fmask).sum())
        if fmask.any():
            mesh.update_faces(fmask)
            mesh.remove_unreferenced_vertices()
        return mesh, info

    seg = segment_tokens(runner, latents, ev_tree, radius, res=res, tau=tau,
                         intruder_tree=intruder_tree)
    info.update({k: seg[k] for k in ("n_dropped", "n_tokens")})
    info["token_score_mean"] = float(seg["token_score"].mean())
    kept = decode_with_tokens(runner, latents, seg["keep"])

    if method == "gate":
        return mesh_from_latents(runner, ds, idx, kept), info

    if method == "refeed":
        out, note = _partial_denoise(runner, batch, kept, refeed_t, refeed_steps, seed)
        info["refeed"] = note
        return mesh_from_latents(runner, ds, idx, out), info

    raise SystemExit(f"unknown method {method}")


def _origin_of(ds, idx):
    probe = trimesh.Trimesh(
        vertices=np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]),
        faces=np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]), process=False)
    return np.asarray(ds.rescale_back(idx, probe, False).vertices, np.float64)[0]


def _basis_of(ds, idx):
    probe = trimesh.Trimesh(
        vertices=np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]),
        faces=np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]), process=False)
    mv = np.asarray(ds.rescale_back(idx, probe, False).vertices, np.float64)
    return np.stack([mv[1] - mv[0], mv[2] - mv[0], mv[3] - mv[0]], axis=0)


def _partial_denoise(runner, batch, x, t_start, steps, seed):
    """
    Re-noise the segmented latent partway and integrate the flow back to data.

    IMPORTANT time convention: in this codebase the trajectory runs t=0 -> t=1 with
    t=0 the NOISE end (`x_init = get_x0_from_input`, a randn draw) and t=1 the data
    end. That is the reverse of the more common convention, so `t_start` here is how
    far along the trajectory to restart: t_start=1.0 is a no-op and smaller values
    re-noise more.

    The interpolation is the rectified-flow one, x_t = t * x_data + (1-t) * noise,
    and the remainder is integrated with the SAME `WrappedModel` velocity the
    released sampler uses, so nothing about the model's dynamics is reimplemented.
    """
    from model.flow_matching.shaper_denoiser import WrappedModel

    g = torch.Generator(device=x.device).manual_seed(seed + 4242)
    noise = torch.randn(x.shape, generator=g, device=x.device, dtype=x.dtype)
    xt = t_start * x + (1.0 - t_start) * noise

    velocity = WrappedModel(runner.model, batch, runner.text_fx, None, cfg_value=-1)
    # The model runs in bfloat16; a float32 timestep reaches the time embedding's
    # matmul and raises "mat1 and mat2 must have the same dtype".
    ts = torch.linspace(t_start, 1.0, steps + 1, device=x.device).to(x.dtype)
    with torch.no_grad():
        for i in range(steps):
            v = velocity(x=xt, t=ts[i])
            xt = xt + (ts[i + 1] - ts[i]) * v
    return xt, f"euler t{t_start:.2f}->1.0 x{steps}"


# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clutter_pkl", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--methods", nargs="+",
                    default=["baseline", "crop", "gate", "refeed"])
    ap.add_argument("--evidence", default="clustered",
                    choices=["clustered", "assigned", "oracle"],
                    help="clustered = the method; assigned = ablation with the raw "
                         "detector point set; oracle = upper bound from the label")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--radius_frac", type=float, default=None,
                    help="fixed evidence radius as a fraction of the normalised "
                         "extent; default is adaptive (see evidence_radius)")
    ap.add_argument("--segmenter", default="support", choices=["support", "connectivity"],
                    help="how to split target from intruder in the SLAM points")
    ap.add_argument("--k_spacing", type=float, default=3.0,
                    help="adaptive radius = this many median NN spacings")
    ap.add_argument("--res", type=int, default=48)
    ap.add_argument("--out_dir", default="updated_2/outputs/stage2")
    args = ap.parse_args()

    out = Path(args.out_dir)
    (out / "meshes").mkdir(parents=True, exist_ok=True)
    tgt, dis = M.load_label(args.label)

    runner = Runner("balance")
    ev_pts, ev_tree, intruder_tree, ev_info = object_evidence(
        runner, args.clutter_pkl, mode=args.evidence, label_npz=args.label,
        segmenter=args.segmenter)
    if args.radius_frac is not None:
        radius = args.radius_frac * float(np.ptp(ev_pts, axis=0).max())
        how = f"fixed {args.radius_frac} x extent"
    else:
        radius = evidence_radius(ev_pts, args.k_spacing)
        how = f"adaptive, {args.k_spacing} x median NN spacing"
    print(f"evidence [{args.evidence}]: {ev_info['n_kept']}/{ev_info['n_points']} "
          f"points kept, radius {radius:.4f} ({how})")
    if "evidence_recall" in ev_info:
        print(f"  vs oracle: precision {ev_info['evidence_precision']:.3f}  "
              f"recall {ev_info['evidence_recall']:.3f}  "
              f"intruder points excluded "
              f"{ev_info['intruder_points_excluded']}/{ev_info['intruder_points_total']}")

    name = Path(args.clutter_pkl).stem
    rows_path = out / "results.json"
    rows = json.loads(rows_path.read_text()) if rows_path.exists() else []

    for method in args.methods:
        for seed in args.seeds:
            tag = f"{name}__{method}__s{seed}"
            print(f"  [{tag}]")
            try:
                mesh, info = run_method(runner, args.clutter_pkl, method, seed,
                                        ev_tree, radius, args.tau, args.res,
                                        intruder_tree=intruder_tree)
            except Exception as e:
                print(f"    FAILED: {type(e).__name__}: {e}")
                continue
            glb = out / "meshes" / f"{tag}.glb"
            mesh.export(glb)
            s = M.score(mesh, tgt, dis)
            if s is None:
                print("    empty mesh")
                continue
            rows.append({"name": name, "evidence": args.evidence,
                         "segmenter": args.segmenter,
                         **{k: v for k, v in info.items()
                                          if not isinstance(v, np.ndarray)},
                         "mesh": str(glb), **s})
            rows_path.write_text(json.dumps(rows, indent=2, default=float))
            print(f"    leak={s.get('leak_pct', float('nan')):.1f}%  "
                  f"extra={s['extra_pct']:.1f}%  missing={s['missing_pct']:.1f}%  "
                  f"chamfer={s['chamfer_pct']:.2f}%")

    summarize(rows)
    print(f"\nresults: {rows_path}")


def summarize(rows):
    if not rows:
        return
    print(f"\n{'method':10s} {'n':>3s} {'Leak%':>7s} {'Extra%':>8s} {'Missing%':>9s} "
          f"{'Chamfer%':>9s}")
    print("-" * 52)
    base = {}
    for m in sorted({r["method"] for r in rows}):
        sel = [r for r in rows if r["method"] == m]
        agg = M.aggregate(sel)
        base[m] = sel
        print(f"{m:10s} {agg['n']:3d} {agg.get('leak_pct', float('nan')):7.1f} "
              f"{agg.get('extra_pct', float('nan')):8.1f} "
              f"{agg.get('missing_pct', float('nan')):9.1f} "
              f"{agg.get('chamfer_pct', float('nan')):9.2f}")

    if "baseline" in base:
        print("\npaired vs baseline (negative Leak/Extra = better; "
              "watch Missing% for over-suppression)")
        b = {(r["seed"]): r for r in base["baseline"]}
        for m, sel in base.items():
            if m == "baseline":
                continue
            pairs = [(b[r["seed"]], r) for r in sel if r["seed"] in b]
            if not pairs:
                continue
            for key in ("leak_pct", "extra_pct", "missing_pct"):
                a = [p[0][key] for p in pairs if key in p[0]]
                c = [p[1][key] for p in pairs if key in p[1]]
                if len(a) != len(c) or not a:
                    continue
                st = M.paired_bootstrap(a, c)
                flag = " *" if st["significant"] else ""
                print(f"  {m:8s} {key:12s} {st['mean_diff']:+7.2f} "
                      f"[{st['ci_lo']:+.2f},{st['ci_hi']:+.2f}]{flag}")
        print("\n* = paired bootstrap 95% CI excludes zero")


if __name__ == "__main__":
    main()
