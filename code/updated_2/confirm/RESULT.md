# Abstention confirmation — NEGATIVE at the frozen threshold

14 fresh table-like objects, `both` clutter at L=0.25, 3 seeds, thresholds frozen in
`protocol.py` before any of them was generated. 42 baseline/refeed pairs.

## Verdict

**NEGATIVE.** The pre-registered primary criterion required that always-refeed be shown to do
real damage to the objects the gate skipped: ΔMissing **+11.32**, 95% CI **[-0.48, +23.79]** —
the interval includes zero, so the criterion is not met.

But the interval is not the important part. Two things this run revealed matter more.

## 1. The method itself is far weaker on fresh objects

`refeed`, independent of any gating:

| | original five | these 14 |
|---|---|---|
| ΔLeak | **-5.16** | **-1.03** |
| ΔMissing (damage) | -0.01 | **+5.95** |

On the original five, refeed removed leakage at no cost. Here it removes a fifth as much and
does substantial damage. **The Stage 2/3 headline does not generalise as stated**, and that
finding is more consequential than anything about the gate. Any claim built on the -5.16 figure
needs re-stating against this.

## 2. The mechanism holds, but the threshold value does not transfer

Ordering damage by cloud recall still works — the harmful objects are the low-health ones:

| object | health | ΔMissing | gate said |
|---|---|---|---|
| BNB1045__nightstand_2 | 0.820 | **+43.48** | abstain (correct) |
| BNB1182__woodenbox | 0.963 | **+17.95** | ACT (wrong) |
| ADT1292__kitchenisland | 0.982 | **+14.65** | ACT (wrong) |
| BNB1182__barcart | 0.963 | **+9.08** | ACT (wrong) |
| ADT1292__shelvingunit | 0.999 | +0.10 | ACT (correct) |
| HWD0045__woodentable | 0.999 | -0.07 | ACT (correct) |
| ADT1292__tvstand | 1.000 | -0.11 | ACT (correct) |

Every harmful object scores <= 0.982; every object at >= 0.985 is safe. The signal is real. But
**0.95 is in the wrong place** — it was set where the original five happened to split (0.920 vs
0.988, a wide empty gap) and the true boundary is far higher, near 0.984.

This is exactly the failure the confirmation was built to detect, and it detected it. A gate
tuned and reported on the same five objects would have looked perfect.

## What must NOT be done with this

A threshold of ~0.984 separates these 14 almost perfectly. **That number may not be quoted.** It
was read off the confirmation set, which is the precise error this run existed to avoid. Adopting
it would put us back where we started, one dataset later, with the appearance of validation.

The correct next step is a third set of objects with 0.984 frozen in advance. If the boundary
moves again, the conclusion is that cloud recall orders risk but has no transferable threshold,
and the gate should be replaced by a relative rule (for example: act only on the top decile of
health *within the batch being processed*) which has no absolute constant to transfer.

## Reproduce

    python updated_2/confirm/score.py
