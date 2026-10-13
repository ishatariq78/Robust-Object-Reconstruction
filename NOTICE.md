# Notice and attribution

This repository documents and evaluates work built **on top of** two pretrained models from
Meta's FAIR, neither of which is vendored here.

## ShapeR

- Upstream: `https://github.com/facebookresearch/ShapeR`
- License: **Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)** — see
  [`LICENSE`](LICENSE), copied verbatim from the upstream repository.
- This repository's own additions (`code/`, `docs/`, `assets/`, `results/`) are licensed under
  the same CC BY-NC 4.0 terms, consistent with being derivative, non-commercial, academic
  (thesis) research built directly on ShapeR's released checkpoint.
- **Not vendored here**: ShapeR's model code, checkpoints, or dataset. This repository assumes
  you have your own working clone of upstream ShapeR with checkpoints set up via its own
  `setup_checkpoints()` — see [docs/setup-and-reproduction.md](docs/setup-and-reproduction.md).

## SAM2 (Segment Anything 2)

- Used unmodified, as a frozen, pretrained tool: `facebook/sam2.1-hiera-base-plus`, loaded at
  run time via Hugging Face. No SAM2 weights or source are vendored in this repository, and SAM2
  itself is not fine-tuned anywhere in this work — only used to produce the foreground masks
  described in [docs/B-mask-conditioning.md](docs/B-mask-conditioning.md) and
  [docs/D-depth-verified-masks.md](docs/D-depth-verified-masks.md).
- License: see Meta's own `segment-anything-2` repository for current terms; not re-stated here
  to avoid misquoting it.

## What is original to this repository

The four experimental studies (A–D), their code (`code/`), documentation (`docs/`), result
summaries (`results/`), and generated/curated figures (`assets/`) are the author's own additions,
built on — and clearly distinguished from — the upstream models above throughout the
documentation. No claim is made of having developed ShapeR or SAM2.
