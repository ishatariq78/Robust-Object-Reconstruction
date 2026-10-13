# Copyright (c) Meta Platforms, Inc. and affiliates.
"""
Does the mask-guided model actually suppress the intruder?

The question the mask fine-tune was built to answer, finally asked with the right instrument.
Earlier it was judged on Extra% over clean and occluded inputs and looked like a null. But
Extra% is the wrong tool here — on `woodentable` an image-only distractor moved Extra% by -0.1
while Leak% rose +7.8, because leaked geometry displaces other error inside the same number.

Here the model is run on the SAME cluttered pickles Stage 1 used, and scored with **Leak%**:
predicted surface sitting on the injected intruder rather than on the target.

The control is `--disable_guidance`: the identical code path with fg_bias_scale=0 and a
zero-init mask conv, so guidance is the ONLY difference. Same seeds in both arms, so every
comparison is paired.

Prediction worth stating before looking: SAM2 masks are image-space, so guidance can only act on
the image branch. It should help most in `image` mode and least in `points` mode, where the
intruder never appears in a picture. If it helps equally in `points` mode, something other than
the masks is doing the work.

    python updated_2/mask_vs_leak/score.py
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "updated_2"))
from metrics import paired_bootstrap, score_paths  # noqa: E402

OUT = REPO / "updated_2/mask_vs_leak/outputs"
CLUT = REPO / "updated_2/outputs/stage1/clutter"


def main():
    rows = []
    for arm in ("guided", "baseline"):
        for g in sorted((OUT / arm).glob("*.glb")):
            stem = g.stem                       # <obj>__<mode>__L025[__s<k>]
            seed = 0
            if "__s" in stem:
                stem, s = stem.rsplit("__s", 1)
                seed = int(s)
            label = CLUT / f"{stem}__label.npz"
            if not label.exists():
                print(f"  no label for {stem} — skipped")
                continue
            sc = score_paths(str(g), label_npz=str(label))
            parts = stem.split("__")
            rows.append(dict(arm=arm, obj=f"{parts[0]}__{parts[1]}", mode=parts[2],
                             seed=seed, **{k: float(v) for k, v in sc.items()
                                           if isinstance(v, (int, float))}))
    if not rows:
        raise SystemExit("no meshes scored — has the job run?")
    json.dump(rows, open(OUT / "scores.json", "w"), indent=2)

    idx = {(r["arm"], r["obj"], r["mode"], r["seed"]): r for r in rows}
    pairs = defaultdict(list)
    for k, r in idx.items():
        if k[0] != "guided":
            continue
        b = idx.get(("baseline",) + k[1:])
        if b:
            pairs[k[2]].append((r, b))
            pairs["ALL"].append((r, b))

    print(f"\n{'mode':8}{'n':>4}{'Leak base':>11}{'Leak guided':>13}{'dLeak':>9}{'95% CI':>18}"
          f"{'dMissing':>10}")
    for mode in [m for m in ("image", "both", "points") if m in pairs] + ["ALL"]:
        pr = pairs[mode]
        if not pr:
            continue
        gl = np.mean([g["leak_pct"] for g, _ in pr])
        bl = np.mean([b["leak_pct"] for _, b in pr])
        dl = [g["leak_pct"] - b["leak_pct"] for g, b in pr]
        dm = [g["missing_pct"] - b["missing_pct"] for g, b in pr]
        # paired_bootstrap(a, b) returns a DICT for (b - a). Order matters: baseline first,
        # guided second, so a negative mean_diff means guidance REDUCED leakage.
        pb = paired_bootstrap([b["leak_pct"] for _, b in pr],
                              [g["leak_pct"] for g, _ in pr])
        lo, hi, sig = pb["ci_lo"], pb["ci_hi"], pb["significant"]
        star = " *" if sig else ""
        print(f"{mode:8}{len(pr):>4}{bl:>11.2f}{gl:>13.2f}{np.mean(dl):>9.2f}"
              f"   [{lo:+.2f},{hi:+.2f}]{star:>3}{np.mean(dm):>10.2f}")

    print(f"\n{'object':26}{'mode':8}{'Leak base':>11}{'Leak guided':>13}{'dLeak':>9}")
    for obj in sorted({r["obj"] for r in rows}):
        for mode in sorted({r["mode"] for r in rows}):
            pr = [(g, b) for g, b in pairs["ALL"] if g["obj"] == obj and g["mode"] == mode]
            if not pr:
                continue
            bl = np.mean([b["leak_pct"] for _, b in pr])
            gl = np.mean([g["leak_pct"] for g, _ in pr])
            print(f"{obj:26}{mode:8}{bl:>11.2f}{gl:>13.2f}{gl-bl:>9.2f}")


if __name__ == "__main__":
    main()
