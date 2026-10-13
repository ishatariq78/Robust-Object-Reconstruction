"""
Experiment 2: Caption / Text Conditioning Ablation
Generates 3 caption variants of each input pkl for evaluation.

Conditions:
  original       - unchanged caption from the pkl (e.g. "stool, a round stool...")
  wrong_category - replaced with "a red car" (maximally wrong category)
  vague          - replaced with "a 3D object" (the model's own fallback default)

This tests whether the text conditioning actually influences the output shape.
If F-Score is similar across all three, the text signal is weak.

Usage:
    python experiments/exp2_caption_ablation/prepare_data.py \
        --input_pkls data/ADT1292__stool.pkl data/BNB2909__pitcher.pkl \
        --output_dir data/exp2_caption
"""

import argparse
import pickle
from pathlib import Path


CAPTION_CONDITIONS = {
    "original":       None,          # keep existing caption
    "wrong_category": "a red car",   # maximally wrong domain
    "vague":          "a 3D object", # model's own fallback in shaper_dataset.py
    "no_caption":     "",            # empty string — bypasses all fallbacks, pure unconditioned
}


def make_variants(input_path, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(input_path).stem

    with open(input_path, "rb") as f:
        pkl = pickle.load(f)

    original_caption = pkl.get("caption") or pkl.get("category", "a 3D object")
    print(f"  Original caption: '{original_caption}'")

    saved = []
    for cond, caption in CAPTION_CONDITIONS.items():
        variant = dict(pkl)
        variant["caption"] = caption if caption is not None else original_caption

        out_path = output_dir / f"{stem}__{cond}.pkl"
        with open(out_path, "wb") as f:
            pickle.dump(variant, f)
        print(f"  [{cond}] caption='{variant['caption']}' -> {out_path}")
        saved.append(str(out_path))

    return saved


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_pkls", nargs="+", required=True)
    parser.add_argument("--output_dir", default="data/exp2_caption")
    args = parser.parse_args()

    all_saved = []
    for path in args.input_pkls:
        print(f"\nProcessing: {path}")
        all_saved.extend(make_variants(path, args.output_dir))

    print(f"\nGenerated {len(all_saved)} pkl files in {args.output_dir}")


if __name__ == "__main__":
    main()
