# Limitations and future work

## Cross-cutting limitations

1. **Ground truth is pseudo-GT.** Every Chamfer/F-score/Extra%/Missing% number in A, B, and C1
   is scored against the ShapeR dataset's own shipped mesh, which is itself a reconstruction, not
   a physically captured reference. Only the clutter-specific Leak% metric (C3, C4, D) is scored
   against a geometrically known quantity — the exact mesh a script placed. See
   [provenance.md](provenance.md) for the full breakdown of which numbers are which.
2. **ShapeR's sampler is unseeded by default**, and a direct measurement found run-to-run Chamfer
   spread of roughly 23% of the mean on a single object. Every single-draw number in this
   repository (most of A, parts of C2) should be read with that floor in mind; the sections that
   used paired multi-seed bootstrap CIs (B's verdict, C3/C4) are the ones to trust for
   significance claims.
3. **One consistent story across A, B, and D**: the image and text conditioning channels move
   reconstruction quality far less than the point-cloud channel does. A2 (captions), B (mask
   guidance), and D (depth-verified masks) all independently land on the same conclusion from
   different angles — this is not a single experiment's artifact.
4. **C4's mitigation does not generalise** past its original 5-object demonstration set at a
   fixed abstention threshold (confirmed negative on 14 fresh objects, pre-registered). It is
   reported as a cautionary result, not a solved method.
5. **Two sub-results could not be independently re-verified in this documentation pass**
   (B's multi-seed bootstrap CIs, C2's null leakage result) because their source folders are
   reported gone from the working VM. Both are flagged explicitly in
   [provenance.md](provenance.md) rather than silently presented as freshly confirmed.

## Future work

- **A reliable abstention test** for C4's mitigation — "is there actually an intruder here?" —
  rather than acting unconditionally; the health-ordering signal is real (every harmful object in
  the 14-object confirmation scored below threshold) but needs a threshold rule that doesn't
  require retuning per dataset, e.g. a relative/within-batch rule instead of an absolute constant.
- **A better selector for identifying which latent tokens correspond to clutter** without relying
  on SAM2 or a hand-built support segmenter — the support segmenter is explicitly named in
  `updated_2/README.md` as "the bottleneck of the whole method, not the latent machinery."
- Compounding degradations in A (sparse *and* noisy *and* vague, together) were not tested.
- Two related studies exist on the same working VM but are **out of scope for this repository**
  by design (not part of the A–D brief) and are not documented here: a LoRA fine-tune that
  attempted to rebalance image/point reliance directly (did not converge, regressed quality) and
  a non-architectural seed-consensus method (draws K reconstructions, keeps the one best agreeing
  with the point cloud) that did beat vanilla ShapeR on a pre-registered test. Both are natural
  next steps given this repository's finding that the image/text channels are under-used, but are
  left for a separate write-up rather than folded in here.
- **Novel-view generation and generated-view reconstruction are deliberately excluded** from this
  repository — that work lives on a different VM and will have its own repository
  (`generative-views-for-3d-reconstruction`, companion to this one; link to be added once that
  repo exists).

## Companion project

This is one of two connected studies for the thesis *"Towards Object-Centric 3D Scene
Understanding for Robotic Manipulation."* The companion project,
**`generative-views-for-3d-reconstruction`**, covers novel-view generation and
generated-view reconstruction on a separate VM. Its repository had not been created at the time
this repository was written — link to be added once available, rather than guessed.
