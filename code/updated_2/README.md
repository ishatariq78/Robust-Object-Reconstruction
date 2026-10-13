# updated_2 — reconstructing the target, not the clutter on it

Work on ShapeR's Limitation (b). Nothing outside this folder is modified.

## The limitation, in the paper's own words

> Second, when objects have other items stacked or closely attached (for example,
> tables supporting other objects), the reconstructed meshes sometimes include
> remnants of these adjacent structures instead of cleanly isolating the target
> object.
> — ShapeR, Appendix D (Limitations)

and Fig. 15(b):

> Closely stacked or attached objects can cause meshes to include parts of adjacent
> structures, **even when the points associated with these structures are not in the
> input**.

## What was already known before this folder existed

An earlier bench (`updated_shapeR/testing/`) stood a real 3D object on the target and
painted it into the images, leaving the SLAM point cloud untouched. Re-scoring those
saved meshes:

| object | Extra% clean | Extra% object-on-it | Δ |
|---|---|---|---|
| ADT1292\_\_stool | 15.3 | 15.7 | +0.4 |
| ADT1292\_\_graycouch | 3.1 | 3.5 | +0.4 |
| BNB1045\_\_sofa | 18.9 | 20.3 | +1.4 |
| BNB1045\_\_wovenstool | 67.0 | 35.1 | −31.9 |
| BNB1723\_\_livingroomfurniture | 47.7 | 37.2 | −10.5 |

No leakage. The two large negative swings are seed noise, not an improvement — this
model's run-to-run Chamfer spread is roughly a quarter of its mean, because
`get_x0_from_input` draws the initial latent from the global RNG and the released
`infer_shape.py` never seeds it.

**Conclusion carried into this folder:** ShapeR's geometry is carried mainly by the
semi-dense point branch, so a demonstration built only on image compositing risks
showing nothing. The clutter has to be able to enter the point cloud too — hence the
three modes below.

That conclusion needed one correction once Leak% existed. Extra% is the wrong
instrument: on `woodentable` an image-only distractor moves Extra% by −0.1 while
Leak% rises +7.8, because leaked geometry displaces other error inside the same
number. So the table above understates the case — part of what looked like "no
leakage" was the metric, not the model. The image branch does leak, on some objects.

## Design

### Stage 1 — demonstrate the limitation (`clutter.py`, `demo_limitation.py`)

`clutter.py` stands a real object on the target's own support surface and writes it
into the pickle so that both conditioning branches see it. Three modes separate the
channels cleanly:

| mode | distractor in images | distractor in SLAM points |
|---|---|---|
| `image` | yes | no |
| `points` | no | yes |
| `both` | yes | yes |

so the experiment reports *where* the leakage enters, not just that it happens.

Care taken:

- **Same view pool for both conditions.** The pool is pruned once, then clean and
  cluttered are built from it, so ShapeR's own view selection sees identical
  candidates and the only difference is the clutter.
- **Point count matched to the target's own density.** A denser-than-real injection
  would overstate the effect.
- **Real visibility.** A distractor point is added to a view only if that view
  actually sees it — exact ray casting against target *and* distractor, so the far
  side of the intruder is self-occluded rather than injected as see-through.
- **Index-aligned schema.** `inv_dist_std` / `dist_std` stay aligned with
  `points_model`; added rows take the median of the originals, the one explicit
  imputation. No view is left with an empty projection list, which would collapse
  ShapeR's crop box to a default corner box.
- **Multi-seed.** Every cell runs at three seeds and differences are reported with a
  paired bootstrap CI. Single-seed deltas on this model have not survived replication
  before.

The headline metric is **Leak%** (`metrics.py`): predicted surface sitting on the
injected distractor rather than on the target. `clutter.py` saves the distractor's
exact placed mesh, so this is measured directly. Extra% cannot do this job on its
own — it lumps leaked clutter together with ordinary fidelity error, and on the woven
stool most of a 55% Extra score turned out to be shape-proportion error with no
second object present at all.

### Stage 2a — is the latent even segmentable? (`latent_probe.py`)

The method suppresses latent *tokens*, which is only meaningful if a token's
influence is spatially localised. There is reason to expect it: the VAE encoder picks
its latent queries by farthest-point sampling **on the input surface**
(`fps(pos, batch, ratio=...)` in `SharpCoarseCrossAttentionEncoder`), so at encode
time token *k* is anchored to a 3D location by construction. Whether latents produced
by the *denoiser* inherit that organisation is measured, not assumed, two ways:
decoder cross-attention footprints, and actually dropping a token and seeing where
the field moves.

This gate is why the probe exists as its own script: if tokens are not localised, the
gating method cannot work and that is the finding.

**Result (woodentable, clean, seed 0) — the gate passes.**

| measure | value |
|---|---|
| token localisation score, mean | 0.565 |
| uniform (non-localised) reference | 0.051 |
| ratio | **11.1x** |
| tokens scoring >3x uniform | 86% |
| ablated tokens with a real effect on the field | 8 / 8 |

So latents produced by the denoiser do inherit the encoder's spatial organisation,
and token-level segmentation is meaningful.

**Caveat, recorded because it bounds the method:** the attention-footprint centroid
sits a median 0.589 from where ablating the token actually moves the field, on an
extent of 2.0 — a 29% offset. Attention footprints are therefore a usable guide to
*where* a token writes, not an exact localisation. If `gate` underperforms, this is
the first thing to suspect, and the fix would be to score tokens by ablation
displacement directly rather than by attention (accurate, but 8192 decodes per
object instead of one).

### Stage 2b — the method (`latent_seg.py`)

| method | what it does |
|---|---|
| `baseline` | untouched ShapeR |
| `crop` | delete unsupported surface from the finished mesh — the honest baseline |
| `gate` | score each latent token by evidence support, drop the intruder's tokens, decode from the rest |
| `refeed` | gate, then re-noise and re-denoise so the model re-derives a coherent shape consistent with the segmentation |

`gate` is the "segmentation talks to reconstruction in latent space" step. It is not
equivalent to `crop`: the latent is a *set* consumed by cross-attention, so removing
rows leaves the decoder to close the surface over the removed region, whereas cutting
the mesh leaves the target open where the intruder touched it. `crop` is implemented
precisely so that this claim has to be earned — if `gate` does not beat `crop`, the
latent machinery is not paying for itself.

**The evidence used is deliberately not a SAM2 mask.** SAM2 was measured absorbing an
occluder into the foreground on exactly the cluttered object where it mattered most.
The prior here is geometric and training-free, in two steps.

*Step 1 — segment the conditioning points.* When a second object is stacked on the
target, the points the detector assigned to the target can carry the intruder too.

Two segmenters are implemented, and **this step is the bottleneck of the whole
method**, not the latent machinery.

`segment_pointcloud` (`--segmenter connectivity`) splits by connected components at a
radius set by the object's extent. On `woodentable` it is exact — precision 1.000,
recall 1.000, 433/433 intruder points excluded, stable for eps_frac 0.04–0.08. **It
does not generalise.** Swept against ground truth on all five objects, no radius
separates the intruder on four of them: shrinking eps fragments the *table* before it
detaches the pot.

| object | connectivity result |
|---|---|
| woodentable | clean: 433/433 caught, 0 false positives |
| coffeetable | catches 289/289 only with 584–2048 false positives |
| desktable | no radius works |
| nightstand | 64/64 only with 200+ false positives |
| sidetable | excludes two thirds of the target |

`segment_by_support` (`--segmenter support`, the default) uses the physical structure
of the paper's scenario instead: an intruder rests ON a support surface, so above the
tabletop the point cloud's horizontal spread collapses. The support height is the
lowest height where the xy-extent above it drops below half the object's full
xy-extent. Measured over 20 cells (5 objects x 4 clutter settings):

| segmenter | recall >= 60% | false positives |
|---|---|---|
| connectivity | 1 / 5 objects | hundreds of real points when it fails |
| **support** | **12 / 20 cells** | **zero on 20 / 20** |

The zero-false-positive property is the one that matters: `support` never removes
real target surface, so its failure mode is under-suppression rather than damage. An
elbow variant (pick the sharpest spread drop rather than a fixed threshold) was tried
and was worse on both counts — 7/20 and up to 9.9% false positives — so the simpler
rule is kept.

Two traps recorded, both of which silently produced wrong results before being found:

- **Do not apply `T_zup_obj` here.** It maps z -> −y, i.e. it assumes a different
  source convention; `points_model` is already z-up. Applying it drops detection to
  zero intruder points found on every single cell.
- **The connectivity safety guard tests compactness, not size.** Requiring the
  excluded blob to be *large* accepts every failure case, because scattered noise
  spread over the whole object has a bigger bounding-box diagonal (0.98–1.50) than a
  genuine compact intruder (0.40).

*Step 2 — score contrastively.* Positions are scored by whether they are closer to
the target cluster or to the intruder cluster, **not** by absolute distance to object
points. This was forced by a measurement rather than chosen on taste: an absolute
prior cannot tell unobserved-but-real surface from hallucinated surface, because SLAM
points only cover *observed* geometry. A table's underside and inner legs are far
from every point despite being real, and cropping on that score removed the leak but
**deleted 31% of the target**. The contrastive score leaves those regions neutral —
they are far from both clusters — and pushes down only surface sitting on the
intruder.

`--evidence {clustered, assigned, oracle}` exposes this as an ablation: `assigned`
uses the raw detector point set and *should* be seen to fail when the intruder's
points are in it; `oracle` is the upper bound from the label.

`missing_pct` is reported alongside Leak% throughout, because the failure mode of any
suppression method is cutting into the object itself — as the crop baseline showed.

## Layout

```
updated_2/
├── clutter.py           build cluttered pkls (images and/or SLAM points) + GT label
├── metrics.py           Extra% / Leak% / Missing% / Chamfer% + paired bootstrap
├── demo_limitation.py   Stage 1: original ShapeR, clean vs cluttered, multi-seed
├── latent_probe.py      Stage 2a: is the latent set spatially segmentable?
├── latent_seg.py        Stage 2b: crop / gate / refeed
├── run_smoke.slurm      one object, all paths, fast
├── run_stage1.slurm     the Stage 1 sweep
└── outputs/
```

## Run

```bash
sbatch updated_2/run_smoke.slurm     # validates every path end to end
sbatch updated_2/run_stage1.slurm    # the demonstration sweep
```

## Results

### Stage 1 — the limitation, 5 objects x 3 seeds, level 0.25

Paired clean-vs-cluttered, bootstrap 95% CI, 15 pairs per cell. ΔLeak is the rise
over the clean mesh scored against the same intruder.

| mode | level | ΔLeak | 95% CI | ΔExtra | 95% CI |
|---|---|---|---|---|---|
| both | 0.15 | **+1.2** | [+0.7, +1.9] * | −0.9 | [−2.3, +0.5] |
| both | 0.25 | **+4.2** | [+2.5, +5.8] * | +3.0 | [+0.7, +6.6] * |
| both | 0.40 | **+11.8** | [+6.8, +17.1] * | +12.1 | [+6.0, +19.1] * |
| points | 0.25 | **+3.0** | [+1.4, +4.8] * | +2.3 | [+0.0, +4.6] * |
| image | 0.25 | +1.5 | [−0.1, +3.3] | +1.2 | [−0.3, +2.6] |

**A clean monotonic dose-response: +1.2 -> +4.2 -> +11.8 as the intruder grows, every
level significant.** That is much harder to explain away than a single contrast, and
it is the strongest single piece of evidence that the limitation is real here.

Note the 0.15 row. ΔLeak is significant (+1.2) while ΔExtra is *negative* (−0.9) and
not significant — at small intruder sizes the standard metric points the wrong way
entirely. This is the clearest case for why Leak% had to be built.

Two things the aggregate hides, both of which matter more than the pooled number:

**The effect is strongly object-dependent.** ΔLeak per object at level 0.25:

| object | both | points | image |
|---|---|---|---|
| ADT1292\_\_woodentable | +8.3 | +9.2 | **+7.8** |
| HWD0045\_\_desktable | +7.3 | +2.5 | −0.1 |
| ADT1292\_\_coffeetable | +4.2 | +2.2 | −0.1 |
| ADT1292\_\_nightstand | +1.0 | +1.1 | +0.0 |
| BNB1182\_\_sidetable | +0.1 | +0.1 | −0.3 |

On `woodentable` the IMAGE branch alone leaks strongly (+7.8), so the paper's "even
when the points are not in the input" does hold — just not on every object.

**Why the split, and it is not arbitrary.** Ranking the objects by how well ShapeR
reconstructs them CLEAN reproduces the leakage ranking exactly:

| object | clean Missing% | clean Chamfer% | baseline quality | ΔLeak (both) |
|---|---|---|---|---|
| ADT1292\_\_woodentable | 0.0 | 1.19 | good | +8.3 |
| HWD0045\_\_desktable | 0.7 | 2.81 | good | +7.3 |
| ADT1292\_\_coffeetable | 8.7 | 1.63 | good | +4.2 |
| ADT1292\_\_nightstand | 31.1 | 3.24 | poor | +1.0 |
| BNB1182\_\_sidetable | 53.6 | 4.63 | broken | +0.1 |

**Leakage is a failure mode of otherwise-successful reconstructions.** Where ShapeR
already fails to build the object — `sidetable` is missing over half its surface
before any clutter is added — it does not build the intruder either. Those two
objects are not "cases where there is nothing to fix"; they are cases where the model
is failing for a different reason entirely, the paper's Limitation (a), low image
fidelity or too few views. They are out of scope for this work and are reported
separately rather than pooled, because averaging them in would dilute any real effect
toward zero in both directions.

`analyze.py` selects on Stage-1 ΔLeak >= 2.0, which recovers exactly the three
well-reconstructed objects — the rule was fixed before Stage 2 ran and depends only
on the baseline model.

**Leak% is far more seed-stable than Extra%.** `woodentable`/`both` across its three
seeds: ΔLeak +8, +8, +9 — ΔExtra +0, +1, +24. And on `woodentable`/`image`, Extra%
moved −0.1 while Leak% rose +7.8: Extra% was hiding a real leak, because leaked
geometry displaced other error. That is why the earlier image-only bench read as a
clean null.

### Stage 2/3 — the method, 3 seeds, support segmenter

Selection rule fixed BEFORE the method ran (`analyze.py`): a cell is "affected" if
its Stage-1 delta-Leak >= 2.0. That depends only on the baseline model.

**AFFECTED - 18 paired runs (woodentable, coffeetable, desktable x both/points x 3 seeds)**

| method | dLeak | dExtra | dMissing (damage) | Chamfer |
|---|---|---|---|---|
| crop | -5.82 [-7.3,-4.3] * | -5.97 [-7.6,-4.4] * | **+0.97 [+0.4,+1.7] *** | 2.13->1.86 |
| gate | -3.97 [-5.3,-2.7] * | -4.05 [-5.5,-2.7] * | +0.02 [-0.0,+0.0] | 2.13->1.91 |
| **refeed** | **-5.16 [-6.9,-3.5] *** | **-5.32 [-7.2,-3.6] *** | **-0.01 [-0.1,+0.0]** | 2.13->1.86 |

`refeed` removes as much leakage as cropping with damage statistically
indistinguishable from zero, while cropping's +0.97 is small but significant.

**IMPORTANT correction, recorded so it is not overclaimed.** An earlier run with the
connectivity segmenter showed cropping costing +29.72 missing against `refeed`'s
+6.20, which looked like a decisive win for latent segmentation. It was not. With the
support segmenter cropping's damage collapses to +0.97. **Cropping amplifies
segmentation error; latent re-generation absorbs it.** Given a clean segmentation the
two perform similarly, so the honest claim for the latent method is *robustness to
imperfect segmentation*, not raw superiority.

**UNAFFECTED - 12 paired runs (nightstand, sidetable). The harm check, and it FAILS.**

| method | dLeak | dExtra | dMissing |
|---|---|---|---|
| crop | +0.08 | +1.15 | **+10.06 *** |
| gate | -0.12 * | +4.75 * | +0.91 * |
| refeed | +0.13 | +3.46 * | +2.91 * |

On objects with no leakage every method still degrades the reconstruction. Much
better than the connectivity segmenter (`refeed` was +15.5 missing) but **not
solved**, and it is the main obstacle to the method being usable. The next piece of
work is a reliable abstention test - "is there actually an intruder here?" - rather
than always acting.

**Evidence ablation (woodentable/points, 3 seeds) - where the value actually sits**

| evidence | Leak% | Extra% | Missing% | Chamfer |
|---|---|---|---|---|
| baseline (no method) | 12.1 | 17.7 | 0.0 | 1.79 |
| `assigned` (no segmentation) | 13.2 | 33.0 | 5.8 | 3.52 |
| `support` (the method) | 6.8 | 11.6 | 0.0 | 1.48 |
| `oracle` (perfect) | 3.5 | 9.9 | 0.2 | 1.42 |

With unsegmented evidence `gate` is WORSE than doing nothing on every metric. The
support segmenter captures 62% of the leak reduction perfect segmentation would give
(-5.3 of a possible -8.6) at zero damage; the remaining 38% is headroom from its 85%
recall.

### Earlier single-seed figures, superseded

Kept for the record. `ADT1292__woodentable`, `points` clutter, connectivity segmenter:

| method | Leak% | Extra% | Missing% | Chamfer% |
|---|---|---|---|---|
| baseline | 12.3 | 20.7 | 0.0 | 1.86 |
| crop | **0.0** | 4.2 | **13.0** | 1.50 |
| **gate** | 5.3 | 11.2 | **0.0** | **1.40** |

`gate` more than halves both leak and extra geometry **without losing any of the
target**, while `crop` reaches zero leak only by deleting 13% of the table. That
difference is the entire argument for segmenting in the latent rather than on the
mesh, and it is why `crop` is in the comparison.

Evidence ablation on the same cell — `assigned` uses the raw detector point set with
no clustering:

| method (evidence=assigned) | Leak% | Extra% | Missing% |
|---|---|---|---|
| baseline | 12.3 | 20.7 | 0.0 |
| crop | 22.2 | 20.6 | 40.2 |
| gate | 12.9 | 32.5 | 3.2 |

The method collapses. `gate` gives no leak reduction over baseline and makes Extra%
worse. So the win above comes from the point-cloud segmentation supplying a clean
negative class, not from latent gating on its own — worth stating plainly rather than
claiming the latent machinery does the work by itself.

## Status

Code written and validated end to end. Stage 1 sweep (5 objects x 3 modes x 3 levels
x 3 seeds) and Stage 2 sweep are queued. Nothing here should be quoted at one seed.
