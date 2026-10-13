# Abstention — deciding WHEN to clean, not just how

The blocker this addresses, in the Stage-2/3 README's own words:

> On objects with no leakage every method still degrades the reconstruction. **Not solved**, and
> it is the main obstacle to the method being usable. The next piece of work is a reliable
> abstention test — "is there actually an intruder here?" — rather than always acting.

Everything below costs zero GPU: it reuses the 150 Stage-1 and 120 Stage-3 meshes already on disk.

## The detector

May use only what inference has — the reconstructed mesh and the SLAM cloud. No ground truth.

| feature | idea | AUC |
|---|---|---|
| `unsupported` | share of predicted surface with no observed point nearby | 0.43 |
| `blob_frac` | largest connected lump of unsupported surface | 0.42 |
| **`top_unsup`** | **unsupported surface in the UPPER region of the object** | **0.82** |

Only the third works, and it should: things get stacked *on top of* tables. A table's unmeasured
underside is unsupported surface too, which is why plain `unsupported` sits at chance.

The upper region is cut at 72% of the **cloud's** vertical range, not the mesh's — otherwise a
bloated mesh moves its own goalposts.

## The framing mistake that hid the signal

Labelling runs by their INPUT ("was clutter added?") gives AUC **0.53** — a coin flip, and the
first version of this probe concluded the detector failed.

That is the wrong question. Clutter in the input does not mean the mesh leaked: on `sidetable`
clutter raises Leak% by +0.07. The decision the system actually faces is about the OUTPUT — *did
this mesh leak?* Relabelling that way gives AUC **0.82**, rising to **0.88** for the worst leaks.

Controlling for object identity (AUC computed within each object, so "some objects just leak
more" cannot inflate it):

| object | positives | within-object AUC |
|---|---|---|
| ADT1292__woodentable | 11/30 | **1.000** |
| HWD0045__desktable | 6/30 | 0.799 |
| ADT1292__coffeetable | 3/30 | 0.778 |
| ADT1292__nightstand | 0/30 | nothing to detect (correct) |
| BNB1182__sidetable | 0/30 | nothing to detect (correct) |

## End-to-end: detect first, then act

Simulated on the Stage-3 runs — the detector reads each baseline mesh and chooses `refeed` or
nothing; the stored result for that choice is then used.

**Affected objects** (leakage is real) — Leak%, lower is better.
**Unaffected objects** — Missing%, the harm check. Baseline is 43.87.

| policy | fires (aff) | fires (unaff) | Leak% aff | Missing% unaff |
|---|---|---|---|---|
| do nothing (ShapeR) | 0/18 | 0/12 | 8.65 | 43.87 |
| **gated @ 0.35** | 14/18 | 6/12 | **3.80** | **43.91** |
| gated @ 0.63 | 6/18 | 0/12 | 5.55 | 43.87 |
| always refeed | 18/18 | 12/12 | 3.50 | 46.78 |

At threshold 0.35 the gate captures **94%** of the leak reduction that always-acting gives
(4.85 of 5.15) for **1.4%** of its damage (+0.04 vs +2.91). That is the result: it turns a method
that helps and harms into one that helps and does not harm.

## WHY it damaged unaffected objects — the mechanism

Not sparsity, and not bad luck. The damage is predicted by **how badly ShapeR already failed on
that object before any clutter existed**:

    corr(baseline Missing%, damage) = +0.80
    corr(cloud point count, damage) = -0.53   (weaker — density is not the cause)

`sidetable` has a *denser* cloud than `woodentable` (2257 vs 1041 pts/m2) yet is 52.8% missing
before anything is added. Its reconstruction is already carried by generative in-filling rather
than by the measured points. `refeed` then deletes points it believes belong to the intruder, and
a reconstruction that was already starving loses the little support it had: 52.4% -> 58.5%.

So the rule is not "don't act on sparse clouds". It is **don't act on reconstructions that are
already failing.**

## The fix: gate on reconstruction health

`Missing%` needs ground truth, but a close proxy does not — **cloud recall**, the share of
observed SLAM points the mesh actually explains. It is computable at inference and tracks the
thing that matters:

    corr(cloud recall, true Missing%) = -0.89   (n=30 runs)

| object | cloud recall | true Missing% | decision at 0.95 |
|---|---|---|---|
| ADT1292__woodentable | 0.999 | 0.0 | act |
| ADT1292__coffeetable | 0.992 | 10.5 | act |
| HWD0045__desktable | 0.988 | 0.3 | act |
| ADT1292__nightstand | 0.920 | 35.0 | **abstain** |
| BNB1182__sidetable | 0.790 | 52.8 | **abstain** |

| policy | fires (aff) | fires (unaff) | Leak% aff | Missing% unaff |
|---|---|---|---|---|
| do nothing (ShapeR) | 0/18 | 0/12 | 8.65 | 43.87 |
| always refeed | 18/18 | 12/12 | 3.50 | **46.78** |
| leak gate `top_unsup` >= 0.35 | 14/18 | 6/12 | 3.80 | 43.91 |
| **health gate, recall >= 0.95** | **18/18** | **2/12** | **3.50** | **43.84** |
| both gates | 14/18 | 2/12 | 3.80 | 43.84 |

**The health gate gives the full benefit of always acting with none of its damage**, and it beats
the leak detector, which loses 4 of 18 true positives for nothing. Adding the leak gate on top
only subtracts.

### What the health gate is actually doing — state this honestly

It does **not** detect the intruder. It answers a different question: *is it safe to act here?*
It works because Stage 1 established that **leakage only happens on well-reconstructed objects** —
so "safe to act" and "worth acting" happen to coincide. That is a real mechanism, not a
coincidence, but the claim must be phrased as safety gating, not intruder detection. A dataset
where a badly-reconstructed object leaked heavily would break it.

## What is NOT established — read before quoting any of this

1. **Both thresholds were read off the data they are reported on** — 0.35 for the leak gate and
   0.95 for the health gate. They must be fixed on held-out objects before any number is quoted.
   The health gate's margin is wide (0.920 vs 0.988, nothing in between), which is encouraging,
   but five objects cannot establish where the boundary sits.
2. **Three objects carry it.** Only `woodentable`, `desktable`, `coffeetable` have any positives.
   `woodentable` alone contributes 11 of 20. The health gate's split is 3 objects vs 2.
3. **The classes overlap**: affected scores span 0.32-0.68, unaffected 0.24-0.58. This is a useful
   signal, not a clean separation.
4. **`top_unsup` was one of three features tried.** Mild selection effect.
5. All of it is at one clutter level (L=0.25) with one intruder type.

## CONFIRMATION RUN — DONE, AND IT FAILED

Everything below this line was written before the confirmation ran. It did not hold up. See
`confirm/RESULT.md`: on 14 fresh objects the frozen 0.95 threshold let through three badly
harmful cases (health 0.963-0.982, damage +9 to +18), and `refeed` itself was five times weaker
than on the original five (ΔLeak -1.03 vs -5.16) while doing real damage (+5.95 Missing).

The health signal still ORDERS risk correctly. The threshold VALUE does not transfer. Read the
sections below as the hypothesis that was tested, not as findings.

## The confirmation that would settle it

Generate clutter on **fresh table-like objects** not used above (the pool has ~100 unused), fix
the threshold from the existing five, and report the gated system on the new ones with nothing
tuned on them — the same protocol that made the v2 seed-consensus result defensible.

## Files

`abstain_probe.py` — the detector and its AUCs. `outputs/abstain_probe.json`,
`outputs/gated_sim.json` — per-run scores.
