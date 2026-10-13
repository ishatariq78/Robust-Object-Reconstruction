# Copyright (c) Meta Platforms, Inc. and affiliates.
"""
The spoken script, written into the deck's own speaker-notes field.

Notes live inside the .pptx so they appear in Presenter View during the talk, rather than on a
separate sheet nobody can look at while speaking. A printable copy is written alongside.

Each note is roughly 40-60 seconds at a normal pace, and says what to POINT AT as well as what
to say — a slide full of numbers is useless if the audience does not know which one matters.

    python updated_2/deck_script.py
"""

from pathlib import Path

from pptx import Presentation

REPO = Path(__file__).resolve().parents[1]
DECK = REPO / "ShapeR_Slides.pptx"

SCRIPT = [
("Opening",
 "This is everything we tested on ShapeR and what survived testing. Three things to take away. "
 "One: we found a way to cut the surface the model invents by about a fifth, with no retraining "
 "at all. Two: my mask fine-tune works, but not for the reason we designed it — it makes the "
 "model build MORE of the object rather than removing clutter. Three: we proved that a better "
 "mask cannot remove clutter, and I'll show you how we measured that ceiling. I'll also show two "
 "results I reported and then had to take back, because I think those matter as much as the wins."),

("The data",
 "Quick orientation on the data. Someone walks through a real home wearing camera glasses. "
 "179 objects across seven homes. Each object gets one file with photos, 3D scan dots measured on "
 "the real object, and the true 3D shape for scoring. The important line is the bottom box: some "
 "objects have 68 scan dots, others have ten thousand. When there are few dots, the model is "
 "INVENTING rather than measuring — and that single fact explains most of what follows."),

("Two inputs",
 "One idea to hold onto for the whole talk. The model gets two inputs: the photos, and the 3D scan "
 "dots. My mask fine-tune only ever changes what the model sees in the PHOTOS. It never touches "
 "the dots. So for every experiment coming up, the question is: did we change the photos, or did "
 "we change the dots? Changing the photos moves the reconstruction very little. Changing the dots "
 "moves it a lot."),

("The contradiction",
 "I want to address something head-on, because it looked like a contradiction in my earlier draft. "
 "Some slides show the pot NOT being reconstructed, and others show it clearly being reconstructed. "
 "Both are correct — they are different experiments. On the left, the pot was painted into the "
 "photos only, and leakage actually went DOWN by 3.25. On the right, the pot was also given scan "
 "dots, and leakage went UP by 4.18. Same pot, same size, same generator. The only difference is "
 "which input carried it. This is genuinely the most useful thing we found."),

("Occlusion data",
 "First synthetic dataset. We hid 60% of the object in every photo, using the same generator and "
 "the same seeding rule for all five objects, so the comparison is fair. Left is before, right is "
 "after. This was meant to be a hard test for the mask — if the object is half hidden, surely a "
 "mask saying 'the target is here' should help."),

("Occlusion result",
 "It did not. With 60% of the object hidden, the original and the trained model produce almost "
 "identical meshes. Extra% moves by about one point and missing surface does not rise at all. "
 "In hindsight this is exactly what the two-input slide predicts: we made the photos much harder "
 "and left the dots untouched, and the dots are what carry the geometry."),

("Pot, photos only",
 "Second dataset. A plant pot standing on the object, painted into the photos with a proper depth "
 "test so it correctly hides what is behind it. Scan dots untouched. Neither model puts the pot "
 "into the 3D shape — leakage actually falls by 3.25 points. So on this version of the experiment, "
 "there is simply nothing to remove."),

("Pot, photos and dots",
 "Now the same pot, but this time it also gets scan dots — which is what the ShapeR paper describes "
 "as its own limitation. Green is the true table, red is surface the model built ON the pot. The "
 "wooden table goes from 4% contaminated to 25%. Look at rows four and five though: those leak "
 "almost nothing, because the model was already failing to build them properly. Leaking is a "
 "failure mode of otherwise-good reconstructions."),

("Dose response",
 "We ran it at three intruder sizes and got a clean staircase — 1.2, 4.2, 11.8 — with every level "
 "statistically significant. A staircase is much harder to explain away than a single contrast. "
 "The box on the right is a methods point worth making: at the smallest intruder, leakage rises "
 "significantly while the standard Extra% metric moves the WRONG WAY. Leaked geometry displaces "
 "other error inside that number, so it hides the problem. That is why we built a separate measure."),

("Seed consensus idea",
 "Now the strongest result. ShapeR is generative — it rolls dice. Same photos, same dots, change "
 "only the random seed, and you get a different object. Across 60 objects the gap between the best "
 "and worst of eight rolls is a median of 20 points of wrong surface. One object went from 28% "
 "wrong to 86% wrong on identical input. ShapeR rolls once and ships whatever comes out. Our fix: "
 "roll eight times, and keep the one that best matches the object's own scan dots. Those are real "
 "measurements the model already receives, so this needs no retraining and no ground truth."),

("Seed consensus visuals",
 "Here is what that looks like. Red is surface the model invented. Left to right: the photos, the "
 "scan dots, ShapeR's single roll, our chosen roll, and the truth. The laptop goes from 34% "
 "invented down to 5%. I should be straight with you: these are the six largest gains. The average "
 "across all objects is smaller, and that average is the number I quote."),

("Seed consensus result",
 "The headline: invented surface drops from 30.3% to 24.4%. And the important part is HOW it was "
 "tested — we fixed the method first, then generated 40 fresh objects, then measured. Nothing was "
 "tuned to make those objects look good. The chart shows more rolls keep helping, but three rolls "
 "already gets you most of the benefit for a third of the compute, so that's what I'd recommend "
 "in practice."),

("How the mask is made",
 "Now to my mask model. First, how the mask is actually built, because this turned out to matter. "
 "The pipeline uses SAM2, which is a 'click the object and I'll trace it' tool. It picks the clicks "
 "automatically from the scan dots. Columns here are: the photo, the true table, the true pot, "
 "what SAM2 produced in blue, and a verified clean mask in purple."),

("The bug",
 "And here is a real bug we found. In a cluttered scene, the scan dots sit on the table AND the "
 "pot. The code picks sixteen of them at random, so some clicks land on the pot, and the bounding "
 "box stretches to include it. So SAM2 traces both, and the mask tells the model the pot is part "
 "of the target — and my model was trained to build what the mask marks. We measured SAM2's pot "
 "coverage: it ranged from 0.6% to 75% across objects. Nothing in the pipeline ever tells SAM2 what "
 "to EXCLUDE — it never used negative clicks at all."),

("The check",
 "This slide is a mistake I caught, and I think it's worth showing. To find the ceiling, I built a "
 "'perfect' mask from the table's true 3D shape. Before trusting it, I measured it — and it covered "
 "76 to 100% of the pot. The reason is simple: the pot stands in FRONT of the table, so drawing the "
 "table's outline fills in the pixels the pot is hiding. If I had run the experiment on that mask "
 "and seen no improvement, I would have concluded 'even a perfect mask cannot help' — from a mask "
 "pointing straight at the pot. A depth test fixed it."),

("The decisive test",
 "So here is the clean test. Fourteen tables the fine-tune has never seen — I checked that in code. "
 "The pot goes into the photos only, never the scan dots, so the mask is the only thing that could "
 "possibly act. Three conditions on identical input and identical seeds: no mask, today's mask, and "
 "the verified clean mask. This is an upper bound — if a perfect mask can't suppress the pot, no "
 "better segmentation ever will."),

("Outputs",
 "And this is the answer. These are the five tables out of fourteen where the clean mask helped "
 "MOST. Follow any row left to right: the red shrinks a little, and it never disappears. Each panel "
 "carries its own number so you can check the picture against the measurement."),

("Numbers",
 "The numbers confirm it. Leakage moves from 2.34 to 1.91 — but both intervals cross zero, so we "
 "cannot claim it. And note the second row: clean mask versus today's mask is minus 0.11 with a "
 "TIGHT interval, plus or minus 0.2. That's important — this isn't 'not enough objects to tell'. "
 "The ceiling has been measured and it is roughly zero. But look at the last row: missing surface "
 "drops 2.87 and that interval does NOT cross zero."),

("What it does",
 "So this is what my fine-tune actually achieves. Here the TRUE shape is drawn, coloured red "
 "wherever the model failed to build it. Less red is better. The improvement is real but undramatic "
 "to the eye, which is exactly why it needed measuring rather than eyeballing: 2.9 points less "
 "missing surface, on tables the model never saw. That's how I'd report the model — it improves "
 "completeness."),

("The combination",
 "One promising thing, clearly flagged as unconfirmed. If we block clutter in BOTH inputs — delete "
 "the pot's scan dots AND correct the mask — leakage drops from 8.3 to 3.8. On the wooden table the "
 "pot nearly vanishes, 11.8 down to 0.9. But read across honestly: removing the dots does almost "
 "all the work, and on the desk table the mask arm is actually worse than dots alone. And I chose "
 "these eight objects because the segmenter works well on them. It needs a fresh unselected set."),

("Real data",
 "This is the slide that reframed the whole clutter question, and we found it late. The dataset "
 "already contains real tables with real objects standing on them — a kitchen island with seven "
 "things on it. Real leakage averages 0.73%. Our synthetic pot produced 8 to 12%. And the "
 "'could have leaked' column shows this isn't because the real objects are small: the model had the "
 "opportunity to leak 13 to 37% and mostly didn't. On the desk table there's a laptop worth 20% and "
 "ShapeR builds none of it. So our fake pot overstated this limitation by roughly 5 to 10 times."),

("All experiments",
 "Everything on one page. Each dot is an experiment, each bar is the uncertainty. The rule to read "
 "it: if a bar crosses the zero line, the effect can't be told apart from luck. Green passed the "
 "test we set beforehand, red didn't, grey is measuring the problem rather than fixing it."),

("Failures",
 "Four things that didn't work. The one I'd point to is the third: reducing photos from 16 to 4. "
 "That was MY test design at fault, not the model — changing the photo count leaves the scan dots "
 "untouched, so the strongest input never weakened. We corrected it by thinning the dots instead. "
 "And the fourth: deleting the pot's dots looked great on five tables and then damaged "
 "reconstructions on fourteen fresh ones. The held-out result is the one to trust."),

("Withdrawn",
 "Two findings I reported and then took back. The first looked strong — a correlation of 0.72 "
 "saying the cleaner the mask, the more the guidance helped. It came from five objects, and two of "
 "them were in my own training set. On fourteen unseen tables the pattern vanishes completely. "
 "The second is the fake 'perfect' mask I showed earlier. I'm including these because both would "
 "have produced confident, wrong conclusions, and showing them is what makes the rest credible."),

("Conclusions",
 "To close. The strongest result is seed consensus: invented surface from 30.3 down to 24.4 on 40 "
 "fresh objects, with no retraining. My mask fine-tune improves completeness by about 2.9 points. "
 "And a better mask alone does not remove clutter — we measured that ceiling directly and it's near "
 "zero. One sentence: ShapeR's biggest controllable error is randomly invented surface, and seed "
 "consensus reduces it. Removing clutter requires acting on the scan dots, not the photos — and "
 "that combined method still needs a fresh test. Happy to take questions."),
]

QA = [
 ("Why build synthetic clutter when the dataset already had real cases?",
  "Fair — we found that late. And it paid off: it showed the synthetic pot overstated the problem "
  "5-10 times, which is itself a result worth reporting."),
 ("Is seed consensus just best-of-N sampling?",
  "Yes, that is the family. What's new is measuring that sampling variance is the DOMINANT error "
  "source here, and that a training-free scorer using the SLAM points recovers about half of it."),
 ("Why does it cost 8 forward passes?",
  "It doesn't have to. Three draws gets most of the benefit at a third of the cost — that's the "
  "recommendation on the results slide."),
 ("Is the mask fine-tune worth keeping if it does not remove clutter?",
  "Yes, but reported honestly: it reduces missing surface by 2.9 points on unseen objects. That is "
  "a real, replicated effect, just not the one it was designed for."),
 ("How confident are you in the 0.73% real-leakage number?",
  "It is 8 of 14 real cluttered objects; the other 6 need about an hour of GPU time to finish. "
  "The direction is clear but I would call it provisional."),
]


def main():
    prs = Presentation(str(DECK))
    assert len(prs.slides) == len(SCRIPT), f"{len(prs.slides)} slides vs {len(SCRIPT)} notes"
    for slide, (_tag, note) in zip(prs.slides, SCRIPT):
        slide.notes_slide.notes_text_frame.text = note
    prs.save(str(DECK))
    print(f"speaker notes written into {DECK.name} ({len(SCRIPT)} slides)")

    lines = ["# Speaking script — ShapeR findings deck", "",
             "One block per slide. Roughly 40–60 seconds each at a normal pace; "
             "the whole deck runs about 18–20 minutes with pauses.", ""]
    for i, (tag, note) in enumerate(SCRIPT, 1):
        lines += [f"## Slide {i} — {tag}", "", note, ""]
    lines += ["", "## Likely questions", ""]
    for q, a in QA:
        lines += [f"**{q}**", "", a, ""]
    out = REPO / "ShapeR_Script.md"
    out.write_text("\n".join(lines))
    print(f"printable script: {out.name}")


if __name__ == "__main__":
    main()
