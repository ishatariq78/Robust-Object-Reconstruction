# Copyright (c) Meta Platforms, Inc. and affiliates.
"""
Re-score the occlusion bench with the RIGHT instrument.

This bench was originally judged on Extra%, which cannot answer the question it was built for.
Extra% lumps leaked clutter together with ordinary fidelity error: on `woodentable` an image-only
distractor moved Extra% by -0.1 while Leak% rose +7.8. So "no difference" in Extra% was never
evidence that the intruder was absent.

Leak% needs the intruder's placed mesh, which the bench did not save. It is recoverable exactly:
`put_on_target` places the pot as `vertices * scale + support_point(gt)`, `support_point` is
deterministic, and the scale is in the manifest. So the pot is rebuilt here, not re-guessed.

Three questions, one table:
  1. does occlusion degrade the reconstruction?      -> Missing% / Chamfer, clean vs occluded
  2. does the pot leak into the mesh at all?         -> Leak%, clean vs onobject
  3. does the mask model remove it without damage?   -> original vs trained, on both at once

Leak% is also computed for `clean` and `occluded`, where no pot exists in the input. That is the
control: it measures surface that merely happens to sit where the pot would be. Only the RISE
from clean to onobject is evidence of leakage.

    python updated_shapeR/testing/score_occlusion.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import trimesh

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "updated_2"))

from augment import load_distractor, support_point   # noqa: E402
from metrics import score                            # noqa: E402

OUT = HERE / "outputs"
CONDS = ("clean", "occluded", "onobject")
MODELS = ("original", "trained")


def main():
    manifest = json.loads((OUT / "manifest.json").read_text())
    rows = []
    for name in sorted(manifest):
        rec = manifest[name]
        d = __import__("pickle").load(open(REPO / f"data/{name}.pkl", "rb"))
        gt = trimesh.Trimesh(vertices=d["mesh_vertices"].numpy().astype(np.float64),
                             faces=d["mesh_faces"].numpy(), process=False)

        st = rec.get("stats", {}).get("onobject", {})
        pot = None
        if "scale" in st:
            t, _span, _how = support_point(gt)
            pot = load_distractor(st.get("distractor", "plantpot")).copy()
            pot.vertices = pot.vertices * float(st["scale"]) + t

        for cond in CONDS:
            for model in MODELS:
                p = rec.get("meshes", {}).get(f"{cond}__{model}")
                if not p or not Path(p).exists():
                    continue
                pred = trimesh.load(p, force="mesh")
                s = score(pred, gt, distractor=pot)
                rows.append(dict(name=name, cond=cond, model=model,
                                 **{k: float(v) for k, v in s.items()
                                    if isinstance(v, (int, float))}))
    if not rows:
        raise SystemExit("nothing scored")
    json.dump(rows, open(OUT / "occlusion_leak_scores.json", "w"), indent=2)

    def sel(cond, model):
        return [r for r in rows if r["cond"] == cond and r["model"] == model]

    print("=== Q1+Q2: what each condition does, per model (mean over 5 objects) ===\n")
    print(f"{'model':10}{'condition':11}{'Leak%':>8}{'Extra%':>9}{'Missing%':>10}{'Chamfer%':>10}")
    for model in MODELS:
        for cond in CONDS:
            rs = sel(cond, model)
            if not rs:
                continue
            print(f"{model:10}{cond:11}" + "".join(
                f"{np.mean([r[k] for r in rs]):>{w}.2f}"
                for k, w in (("leak_pct", 8), ("extra_pct", 9), ("missing_pct", 10),
                             ("chamfer_pct", 10))))
        print()

    print("=== Q2: did the pot actually leak? (rise from clean -> onobject, paired) ===\n")
    print(f"{'model':10}{'dLeak':>9}{'dMissing':>11}{'dChamfer':>11}")
    for model in MODELS:
        a = {r["name"]: r for r in sel("onobject", model)}
        b = {r["name"]: r for r in sel("clean", model)}
        common = sorted(set(a) & set(b))
        if not common:
            continue
        print(f"{model:10}" + "".join(
            f"{np.mean([a[n][k] - b[n][k] for n in common]):>{w}.2f}"
            for k, w in (("leak_pct", 9), ("missing_pct", 11), ("chamfer_pct", 11))))

    print("\n=== Q3: trained minus original, paired per object and condition ===\n")
    print(f"{'condition':11}{'dLeak':>9}{'dExtra':>9}{'dMissing':>11}{'dChamfer':>11}")
    for cond in CONDS:
        a = {r["name"]: r for r in sel(cond, "trained")}
        b = {r["name"]: r for r in sel(cond, "original")}
        common = sorted(set(a) & set(b))
        if not common:
            continue
        print(f"{cond:11}" + "".join(
            f"{np.mean([a[n][k] - b[n][k] for n in common]):>{w}.2f}"
            for k, w in (("leak_pct", 9), ("extra_pct", 9), ("missing_pct", 11),
                         ("chamfer_pct", 11))))

    print("\n=== per object, onobject condition ===\n")
    print(f"{'object':26}{'Leak orig':>11}{'Leak trained':>14}{'Miss orig':>11}{'Miss trained':>14}")
    for name in sorted({r["name"] for r in rows}):
        o = [r for r in rows if r["name"] == name and r["cond"] == "onobject" and r["model"] == "original"]
        t = [r for r in rows if r["name"] == name and r["cond"] == "onobject" and r["model"] == "trained"]
        if not o or not t:
            continue
        print(f"{name.split('__')[-1]:26}{o[0]['leak_pct']:>11.2f}{t[0]['leak_pct']:>14.2f}"
              f"{o[0]['missing_pct']:>11.2f}{t[0]['missing_pct']:>14.2f}")


if __name__ == "__main__":
    main()
