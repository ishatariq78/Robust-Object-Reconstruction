# Copyright (c) Meta Platforms, Inc. and affiliates.
"""
FROZEN PROTOCOL for the abstention confirmation. Written before any of these objects was run.

Why this file exists as code rather than a note: the two weaknesses of the abstention result are
(1) both thresholds were read off the same data they were reported on, and (2) five objects, split
three against two. Neither is fixable by re-analysing the old runs. The fix is fresh objects with
every choice fixed in advance, which is only credible if the choices are committed before the data
exists — hence a file, in git, imported by the scorer.

NOTHING IN THIS FILE MAY BE EDITED ONCE THE JOBS ARE SUBMITTED.
"""

# ---------------------------------------------------------------------------------------------
# The objects. Chosen by CATEGORY AND NAME ONLY — every one is something a second object can
# stand on, which is the paper's own example ("tables supporting other objects"). Their
# reconstruction quality was NOT consulted; that is the very thing the gate has to decide, so
# selecting on it would guarantee the answer. None appears in the original five.
#
# The cloud sizes happen to span 287-7628 points, so both well- and badly-reconstructed objects
# should be present. That is what makes the test able to fail: a set of uniformly good objects
# would let a gate that always fires look perfect.
# ---------------------------------------------------------------------------------------------

OBJECTS = [
    "HWD0045__woodentable", "HWD0045__coffeetable", "BNB2909__coffeetable",
    "ADT1292__kitchenisland", "ADT1292__tvstand", "BNB1182__nightstand_2",
    "BNB1045__nightstand_2", "BNB1723__nightstand", "BNB1182__barcart",
    "ADT1292__utilitycart", "BNB2909__nightstand", "ADT1292__shelvingunit",
    "HWD0045__storagecabinet", "BNB1182__woodenbox",
]

ALREADY_USED = {"ADT1292__woodentable", "ADT1292__coffeetable", "ADT1292__nightstand",
                "BNB1182__sidetable", "HWD0045__desktable"}
assert not (set(OBJECTS) & ALREADY_USED), "an object from the original five leaked into the set"
assert len(OBJECTS) == len(set(OBJECTS)), "duplicate object"

MODE = "both"          # the realistic case: intruder visible in images AND present in the points
LEVEL = 0.25           # same clutter size as the original Stage 2/3 runs
SEEDS = [0, 1, 2]

# ---------------------------------------------------------------------------------------------
# The gate. Fixed at the values read off the original five — deliberately NOT re-tuned here.
# If 0.95 is the wrong place for the boundary, this run is what reveals it.
# ---------------------------------------------------------------------------------------------

HEALTH_THRESHOLD = 0.95     # cloud recall: share of observed SLAM points the mesh explains
HEALTH_TAU_FRAC = 0.03      # a point counts as explained within 3% of the cloud's extent
LEAK_THRESHOLD = 0.35       # top_unsup, the secondary gate, reported but not primary


def should_act(health, leakscore=None):
    """The decision, in one place so the scorer cannot drift from what was registered."""
    return health >= HEALTH_THRESHOLD


# ---------------------------------------------------------------------------------------------
# SUCCESS CRITERION, fixed in advance.
#
# The claim is NOT "gating reduces leakage" — `refeed` already does that. The claim is that
# gating keeps refeed's benefit while removing its damage. So both halves are required:
#
#   PRIMARY (the damage claim). On objects the gate ABSTAINS from, damage must be zero by
#     construction, and always-refeed must be shown to do real damage on those same objects —
#     paired ΔMissing > 0 with a bootstrap 95% CI excluding zero. If always-refeed turns out to
#     be harmless on them, the gate solves a problem that does not exist here and the result is
#     negative, however good the leak numbers look.
#
#   SECONDARY (the benefit claim). On objects the gate ACTS on, ΔLeak must fall with a 95% CI
#     excluding zero, and the gated policy must retain >= 70% of always-refeed's total leak
#     reduction across all objects.
#
# A gate that fires on everything passes the secondary and fails the primary. A gate that fires
# on nothing does the reverse. Both are required precisely so neither degenerate answer counts.
# ---------------------------------------------------------------------------------------------

MIN_BENEFIT_RETAINED = 0.70


def verdict(abstain_damage_ci, act_leak_ci, benefit_retained):
    lo, hi = abstain_damage_ci
    llo, lhi = act_leak_ci
    primary = lo > 0                       # always-refeed genuinely harms what the gate skips
    secondary = lhi < 0 and benefit_retained >= MIN_BENEFIT_RETAINED
    if primary and secondary:
        return "CONFIRMED — the gate keeps the benefit and removes the damage"
    if not primary:
        return "NEGATIVE — refeed did not measurably harm the abstained objects; gating is unmotivated here"
    return "PARTIAL — the gate avoids damage but gives up too much of the benefit"
