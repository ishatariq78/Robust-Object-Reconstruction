# Copyright (c) Meta Platforms, Inc. and affiliates.
"""
Build the supervisor deck as a real, editable .pptx.

Native PowerPoint objects throughout — text frames, tables, coloured panels — rather than
pictures of slides, so every word can be edited before the talk. Figures are embedded as images
because they are renders, but nothing else is flattened.

Layout mirrors the reference deck: orange eyebrow, page number top-right, title, blue subtitle,
body, and a rule-topped footer carrying the one-line takeaway.

    python updated_2/build_deck_pptx.py
"""

from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.util import Emu, Inches, Pt

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "ShapeR_Slides.pptx"

INK = RGBColor(0x1A, 0x1A, 0x1A)
INK2 = RGBColor(0x5A, 0x60, 0x68)
INK3 = RGBColor(0x8B, 0x92, 0x9B)
ACCENT = RGBColor(0xC7, 0x41, 0x1F)
BLUE = RGBColor(0x1F, 0x5F, 0xA8)
GREEN = RGBColor(0x1E, 0x7A, 0x4D)
WARN = RGBColor(0x8A, 0x6A, 0x16)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
LINE2 = RGBColor(0xCB, 0xD2, 0xD9)
SUNKEN = RGBColor(0xF5, 0xF7, 0xF9)
BAD_S = RGBColor(0xFD, 0xF0, 0xEC)
GOOD_S = RGBColor(0xEA, 0xF5, 0xEF)
BLUE_S = RGBColor(0xED, 0xF3, 0xFA)
WARN_S = RGBColor(0xFB, 0xF3, 0xE0)

W, H = Inches(13.333), Inches(7.5)
M = Inches(0.62)
TONE = {"r": (BAD_S, ACCENT), "g": (GOOD_S, GREEN), "b": (BLUE_S, BLUE),
        "y": (WARN_S, WARN), "n": (SUNKEN, INK2)}

FIGS = {
    "wins": "updated_shapeR/testing/v2/outputs/fig_top.jpg",
    "ksweep": "updated_shapeR/testing/v2/outputs/fig3_ksweep.png",
    "summary": "updated_shapeR/testing/v2/outputs/fig4_summary.png",
    "leakdemo": "updated_2/outputs/stage1/comparison/both_L040.png",
    "verify": "updated_2/mask_vs_leak/outputs/fig_mask_verify.jpg",
    "cleanmask": "updated_2/mask_vs_leak/outputs/clean_test/fig_cleanmask.jpg",
    "combo": "updated_2/mask_vs_leak/outputs/fig_leak_improved.jpg",
    "complete": "updated_2/mask_vs_leak/outputs/fig_complete.jpg",
    "occl": "updated_shapeR/testing/outputs/comparison/occluded.jpg",
    "onobj": "updated_shapeR/testing/outputs/comparison/onobject.jpg",
}


def txt(slide, l, t, w, h, size=13, bold=False, color=INK, align=PP_ALIGN.LEFT,
        anchor=MSO_ANCHOR.TOP, space=1.22):
    tb = slide.shapes.add_textbox(l, t, w, h)
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    p = tf.paragraphs[0]
    p.alignment = align
    p.line_spacing = space
    r = p.add_run()
    r.font.size, r.font.bold, r.font.color.rgb, r.font.name = Pt(size), bold, color, "Lato"
    return tb, tf, r


def panel(slide, l, t, w, h, tone="n"):
    from pptx.enum.shapes import MSO_SHAPE
    sh = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, l, t, w, h)
    fill, border = TONE[tone]
    sh.fill.solid()
    sh.fill.fore_color.rgb = fill
    sh.line.color.rgb = border if tone != "n" else LINE2
    sh.line.width = Pt(1.1)
    sh.shadow.inherit = False
    sh.adjustments[0] = 0.05
    sh.text_frame.text = ""
    return sh


def card(slide, l, t, w, h, title, body, tone="n"):
    panel(slide, l, t, w, h, tone)
    _c, tf, r = txt(slide, l + Inches(0.17), t + Inches(0.13), w - Inches(0.34), Inches(0.3),
                    size=13, bold=True, color=TONE[tone][1] if tone != "n" else INK)
    r.text = title
    _c2, tf2, r2 = txt(slide, l + Inches(0.17), t + Inches(0.48), w - Inches(0.34),
                       h - Inches(0.6), size=11.5, color=INK2, space=1.3)
    r2.text = body


def table(slide, l, t, w, rows, widths=None, hi=None, fs=11):
    nr, nc = len(rows), len(rows[0])
    gt = slide.shapes.add_table(nr, nc, l, t, w, Inches(0.34) * nr).table
    if widths:
        for i, frac in enumerate(widths):
            gt.columns[i].width = Emu(int(w * frac))
    for ri, row in enumerate(rows):
        gt.rows[ri].height = Inches(0.3)
        for ci, val in enumerate(row):
            cell = gt.cell(ri, ci)
            cell.text = str(val)
            cell.margin_left = cell.margin_right = Inches(0.08)
            cell.margin_top = cell.margin_bottom = Inches(0.02)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            cell.fill.solid()
            cell.fill.fore_color.rgb = (SUNKEN if ri == 0 else
                                        (BLUE_S if hi is not None and ri == hi else WHITE))
            p = cell.text_frame.paragraphs[0]
            p.alignment = PP_ALIGN.LEFT if ci == 0 else PP_ALIGN.RIGHT
            for r_ in p.runs:
                r_.font.size = Pt(9.5 if ri == 0 else fs)
                r_.font.bold = ri == 0 or (hi is not None and ri == hi)
                r_.font.color.rgb = INK3 if ri == 0 else INK
                r_.font.name = "IBM Plex Mono" if ri > 0 and ci > 0 else "Lato"
    return gt


def new_slide(prs, eyebrow, title, sub, page, total):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    _a, _b, r = txt(s, M, Inches(0.34), Inches(9), Inches(0.24), size=10.5, bold=True, color=ACCENT)
    r.text = eyebrow.upper()
    _c, _d, r2 = txt(s, W - M - Inches(1.4), Inches(0.34), Inches(1.4), Inches(0.24),
                     size=10.5, color=INK3, align=PP_ALIGN.RIGHT)
    r2.text = f"{page} / {total}"
    _e, _f, r3 = txt(s, M, Inches(0.62), W - 2 * M, Inches(0.62), size=25, bold=True, color=INK)
    r3.text = title
    _g, _h, r4 = txt(s, M, Inches(1.28), W - 2 * M, Inches(0.28), size=12, bold=True,
                     color=BLUE, align=PP_ALIGN.CENTER)
    r4.text = sub
    return s


def footer(slide, text, warn=False):
    ln = slide.shapes.add_shape(1, M, H - Inches(0.72), W - 2 * M, Pt(2))
    ln.fill.solid()
    ln.fill.fore_color.rgb = ACCENT if warn else GREEN
    ln.line.fill.background()
    ln.shadow.inherit = False
    _a, _b, r = txt(slide, M, H - Inches(0.58), W - 2 * M, Inches(0.4), size=11.5, bold=True)
    r.text = text


def picture(slide, key, top=Inches(1.72), bottom=Inches(0.92)):
    """Fit the figure inside the body area without distorting it."""
    from PIL import Image
    p = REPO / FIGS[key]
    iw, ih = Image.open(p).size
    avail_w, avail_h = int(W - 2 * M), int(H - top - bottom)
    scale = min(avail_w / iw, avail_h / ih)
    w, h = int(iw * scale), int(ih * scale)
    slide.shapes.add_picture(str(p), int((W - w) // 2), int(top + (avail_h - h) // 2),
                             width=w, height=h)


def build():
    prs = Presentation()
    prs.slide_width, prs.slide_height = W, H
    N = 25
    n = [0]

    def S(eb, ti, sb):
        n[0] += 1
        return new_slide(prs, eb, ti, sb, n[0], N)

    # EMU must be an INTEGER: a divided Inches() value is a float and PowerPoint
    # refuses to open the file. Round once, here.
    CW = int((W - 2 * M - Inches(0.36)) / 3)     # three-card width
    CW2 = int((W - 2 * M - Inches(0.28)) / 2)    # two-card width
    BT = Inches(1.78)                            # body top

    # 1 ------------------------------------------------------------------
    s = S("ShapeR · findings", "What we tested, what held up, and what did not",
          "179 objects · 7 real homes · roughly 1,100 reconstructions")
    card(s, M, BT, CW, Inches(1.75), "Confirmed",
         "Seed consensus cuts invented surface from 30.3% to 24.4% on 40 objects the method never saw.", "g")
    card(s, M + CW + Inches(0.18), BT, CW, Inches(1.75), "Confirmed",
         "The mask fine-tune reduces MISSING target surface by 2.9 points on 14 unseen tables.", "b")
    card(s, M + 2 * (CW + Inches(0.18)), BT, CW, Inches(1.75), "Not supported",
         "A better mask does not remove clutter. Measured with a perfect mask: ceiling near zero.", "r")
    card(s, M, BT + Inches(1.95), W - 2 * M, Inches(1.5), "How to read this deck",
         "Every claim carries the evidence behind it. Where a result was NOT statistically confirmed, "
         "the slide says so rather than rounding it up. Two earlier findings were withdrawn after "
         "re-testing, and both withdrawals are shown.", "b")
    footer(s, "Numbers quoted here come from held-out objects unless the slide states otherwise.")

    # 2 ------------------------------------------------------------------
    s = S("The data", "Each object is one file from a real walk-through",
          "Recorded with Aria camera glasses in seven homes")
    _a, _b, r = txt(s, M, BT + Inches(0.1), Inches(2.4), Inches(0.9), size=40, bold=True,
                    color=BLUE, align=PP_ALIGN.CENTER)
    r.text = "179"
    _a, _b, r = txt(s, M, BT + Inches(0.85), Inches(2.4), Inches(0.3), size=11, color=INK2,
                    align=PP_ALIGN.CENTER)
    r.text = "objects"
    _a, _b, r = txt(s, M, BT + Inches(1.35), Inches(2.4), Inches(0.9), size=40, bold=True,
                    color=BLUE, align=PP_ALIGN.CENTER)
    r.text = "7"
    _a, _b, r = txt(s, M, BT + Inches(2.1), Inches(2.4), Inches(0.3), size=11, color=INK2,
                    align=PP_ALIGN.CENTER)
    r.text = "real homes"
    table(s, M + Inches(2.75), BT, W - M - Inches(2.75) - M,
          [["What is in the file", "Plain meaning", "Range"],
           ["points_model", "3D scan dots on the object", "68 – 10,893"],
           ["image_data", "grey camera frames", "28 – 1,725"],
           ["rgb_image_data", "colour photos", "27 – 1,254"],
           ["mesh_vertices", "the true 3D shape, for scoring", "—"],
           ["T_model_world", "where it sits in the room", "—"]], widths=[.28, .46, .26])
    card(s, M + Inches(2.75), BT + Inches(2.15), W - M - Inches(2.75) - M, Inches(1.3),
         "Why the range matters",
         "Objects run from a 7 cm cup to a 2.7 m bed. One object is described by 68 measured dots, "
         "another by ten thousand. Where dots are few the model is INVENTING rather than measuring "
         "— that single fact explains most of what follows.", "b")
    footer(s, "Typical object: 702 scan dots, 0.60 m across.")

    # 3 ------------------------------------------------------------------
    s = S("The input", "The same photos go to both models — nothing is changed here",
          "Plain, unmodified data: no occlusion, no added objects")
    bullets = [("Two inputs reach the model:", " the photos, and the 3D scan dots."),
               ("The mask fine-tune only ever changes", " what the model sees in the PHOTOS."),
               ("The scan dots are untouched by the mask", " — they go straight through."),
               ("This distinction decides", " almost every result in this deck.")]
    y = BT + Inches(0.15)
    for b1, b2 in bullets:
        _a, tf, r = txt(s, M, y, CW2 - Inches(0.2), Inches(0.6), size=13, bold=True)
        r.text = "•  " + b1
        r2 = tf.paragraphs[0].add_run()
        r2.text = b2
        r2.font.size, r2.font.bold, r2.font.color.rgb, r2.font.name = Pt(13), False, INK2, "Lato"
        y += Inches(0.72)
    card(s, M + CW2 + Inches(0.28), BT + Inches(0.15), CW2, Inches(2.2), "Set this up early",
         "Every later slide can be read as: did we change the photos, or did we change the dots?\n\n"
         "Changing the photos moves the reconstruction very little. Changing the dots moves it a lot.", "b")
    footer(s, "Both models receive identical photos and identical dots in every comparison.")

    # 4 ------------------------------------------------------------------
    s = S("Resolving an apparent contradiction",
          "A pot can be added in two ways — and they give opposite results",
          "This is why some slides show no pot in the output and others show a large one")
    card(s, M, BT, CW2, Inches(1.9), "Way 1 · pot painted into the PHOTOS only",
         "Scan dots untouched. Objects: sofa, couch, stool, side table, woven stool.\n\n"
         "Leak change: −3.25 — the pot does NOT enter the 3D shape. It goes slightly down.", "b")
    card(s, M + CW2 + Inches(0.28), BT, CW2, Inches(1.9), "Way 2 · pot in the photos AND the scan dots",
         "Objects: wooden table, coffee table, desk table, nightstand, side table.\n\n"
         "Leak change: +4.18 — the pot DOES become part of the table.", "r")
    card(s, M, BT + Inches(2.1), W - 2 * M, Inches(1.4),
         "There is no contradiction — the two experiments changed different inputs",
         "Same pot, same size, same generator. The only difference is WHICH INPUT CARRIED IT. When the "
         "pot exists only in the photos, ShapeR ignores it. When the pot also has scan dots, ShapeR "
         "builds it. This is the single most useful finding in the project.", "b")
    footer(s, "Any slide showing pot leakage is a 'Way 2' experiment. Any slide showing none is 'Way 1'.")

    # 5-8 figure slides --------------------------------------------------
    s = S("Dataset 1 · occlusion", "Hiding 60% of the object in every photo",
          "Same generator and same seeding rule for every object")
    picture(s, "occl")
    footer(s, "Five objects at a 59–60% hidden level. Ground truth, original model, trained model.")

    s = S("Dataset 1 · result", "Hiding most of the object barely changes the 3D shape",
          "Both models produce near-identical meshes")
    card(s, M, BT, CW2, Inches(1.9), "The mask does not help here",
         "With 60% of the object hidden, the original and trained meshes are visually and numerically "
         "almost the same.\n\nThe geometry comes from the dots, and the dots were untouched.", "r")
    table(s, M + CW2 + Inches(0.28), BT + Inches(0.2), CW2,
          [["Condition", "Extra%", "Missing%"], ["clean photos", "30.1", "27.6"],
           ["60% hidden", "31.0", "25.7"]], widths=[.5, .25, .25])
    footer(s, "Consistent with everything that followed: image-side changes barely move ShapeR's geometry.")

    s = S("Dataset 2 · Way 1", "A pot standing on the object, painted into the photos only",
          "Ground truth · original model · trained model")
    picture(s, "onobj")
    footer(s, "Neither model puts the pot into the 3D shape — leak actually falls by 3.25 points.")

    s = S("Dataset 2 · Way 2", "With scan dots too, the pot becomes part of the table",
          "Green = true table · red = surface the model built on the pot")
    picture(s, "leakdemo")
    footer(s, "Wooden table goes 4.0% → 25.4%. Rows 4–5 leak nothing — already badly reconstructed.")

    # 9 ------------------------------------------------------------------
    s = S("Dataset 2 · dose response", "The bigger the intruder, the more leaks in",
          "A monotonic response is much harder to explain away than a single contrast")
    table(s, M, BT + Inches(0.25), CW2,
          [["Intruder size", "Leak change", "95% interval"],
           ["15% of target area", "+1.24", "[+0.67, +1.86]"],
           ["25%", "+4.18", "[+2.53, +5.86]"],
           ["40%", "+11.80", "[+6.78, +17.01]"]], widths=[.4, .26, .34])
    card(s, M + CW2 + Inches(0.28), BT, CW2, Inches(2.2), "Why a new measure was needed",
         "At the smallest intruder, leakage rises a significant +1.24 while the standard Extra% "
         "measure moves −0.9 — the wrong way entirely.\n\nLeaked geometry displaces other error "
         "inside the same number. Every leakage claim here uses Leak%.", "b")
    footer(s, "Every level is statistically significant; the effect scales with intruder size.")

    # 10 -----------------------------------------------------------------
    s = S("Seed consensus", "ShapeR rolls dice — so roll several times and keep the best",
          "Same photos, same dots, different random seed")
    card(s, M, BT, CW2, Inches(2.1), "The problem",
         "Identical input, different seed, different object. Across 60 objects the gap between best "
         "and worst of eight draws is a median of 20 points of wrong surface, up to 51.\n\n"
         "One object went from 28% wrong to 86% wrong. ShapeR rolls once and ships it.", "r")
    card(s, M + CW2 + Inches(0.28), BT, CW2, Inches(2.1), "The fix",
         "Draw eight. Keep the one that best agrees with the object's own scan dots — real "
         "measurements the model already receives.\n\nNo retraining, no new parameters, no ground "
         "truth at inference.", "g")
    footer(s, "The selector uses only information available at inference time.")

    s = S("Seed consensus · visuals", "Same model, same input, better draw",
          "Red = surface the model invented. Photos · dots · ShapeR's draw · our draw · truth")
    picture(s, "wins")
    footer(s, "The laptop goes from 34% invented to 5%; the footstool 36% to 9%. Six largest gains.")

    # 12 -----------------------------------------------------------------
    s = S("Seed consensus · result", "Three draws already capture most of the gain",
          "Method fixed in advance, then confirmed on 40 objects it had never seen")
    _a, _b, r = txt(s, M, BT + Inches(0.1), Inches(2.5), Inches(0.8), size=36, bold=True,
                    color=ACCENT, align=PP_ALIGN.CENTER)
    r.text = "30.3%"
    _a, _b, r = txt(s, M, BT + Inches(0.78), Inches(2.5), Inches(0.4), size=11, color=INK2,
                    align=PP_ALIGN.CENTER)
    r.text = "invented surface · plain ShapeR"
    _a, _b, r = txt(s, M, BT + Inches(1.2), Inches(2.5), Inches(0.4), size=20, color=BLUE,
                    align=PP_ALIGN.CENTER)
    r.text = "↓"
    _a, _b, r = txt(s, M, BT + Inches(1.6), Inches(2.5), Inches(0.8), size=36, bold=True,
                    color=GREEN, align=PP_ALIGN.CENTER)
    r.text = "24.4%"
    _a, _b, r = txt(s, M, BT + Inches(2.28), Inches(2.5), Inches(0.4), size=11, color=INK2,
                    align=PP_ALIGN.CENTER)
    r.text = "invented surface · seed consensus"
    card(s, M, BT + Inches(2.75), Inches(2.5), Inches(0.62), "40 fresh objects",
         "nothing tuned on them", "g")
    from PIL import Image
    p = REPO / FIGS["ksweep"]
    iw, ih = Image.open(p).size
    availw, availh = W - M - Inches(3.4) - M, Inches(3.5)
    sc = min(availw / iw, availh / ih)
    s.shapes.add_picture(str(p), int(M + Inches(3.4)), int(BT), width=int(iw * sc), height=int(ih * sc))
    footer(s, "Recommendation: use 3 draws. Eight is slightly better but costs far more compute.")

    # 13-14 --------------------------------------------------------------
    s = S("The mask · how it is built", "The mask is traced by SAM2, and the clicks come from the scan dots",
          "Positive clicks are sampled automatically; the pipeline never used negative 'not this' clicks")
    picture(s, "verify")
    footer(s, "Photo · true table (green) · true pot (red) · SAM2's mask (blue) · verified clean mask (purple).")

    s = S("The mask · the bug", "A scan dot on the pot tells SAM2 that the pot is the target",
          "SAM2 traces what it is pointed at; nothing tells it what to exclude")
    card(s, M, BT, CW, Inches(1.5), "1 · Clicks land on the pot",
         "16 dots are picked automatically. In a cluttered scene, some sit on the pot.", "r")
    card(s, M + CW + Inches(0.18), BT, CW, Inches(1.5), "2 · The box stretches",
         "The bounding box is drawn around all the dots, so it grows to include the pot.", "r")
    card(s, M + 2 * (CW + Inches(0.18)), BT, CW, Inches(1.5), "3 · The model obeys",
         "The mask marks the pot as target, and the fine-tune was trained to build what the mask marks.", "r")
    card(s, M, BT + Inches(1.7), W - 2 * M, Inches(1.3), "Measured, not assumed",
         "SAM2's pot coverage varied from 0.6% to 75% across objects. Nothing in the pipeline ever "
         "tells SAM2 the pot is not part of the object — when it gets it right, that is SAM2's own "
         "visual judgement.", "b")
    footer(s, "The model was being told to build the pot. It was doing what it was told.")

    # 15 -----------------------------------------------------------------
    s = S("The mask · a check that changed the answer", "Our first 'perfect' mask was covering the pot",
          "The pot stands in front of the table, so the table's outline fills the pixels it hides")
    card(s, M, BT, CW2, Inches(2.2), "Before checking",
         "The naive perfect mask covered 76–100% of the pot — the exact opposite of its purpose.\n\n"
         "Had we run the experiment on it and seen no improvement, we would have concluded 'even a "
         "perfect mask cannot help' — from a mask pointing straight at the pot.", "r")
    card(s, M + CW2 + Inches(0.28), BT, CW2, Inches(2.2), "After a depth test",
         "Keeping only pixels where the table is the nearer surface brings pot coverage down to "
         "0.7–15.6%, while still covering about 90% of the visible table.\n\n"
         "Only then is it fair to call it a clean mask.", "g")
    footer(s, "Verify the instrument before trusting the experiment.")

    # 16 -----------------------------------------------------------------
    s = S("The mask · the decisive test", "Pot in the photos, but not in the mask",
          "14 tables the fine-tune never saw · pot never added to the scan dots · only the mask changes")
    sw = int((W - 2 * M - Inches(0.4)) / 5)
    for i, (lab, tone) in enumerate([("14 unseen tables", "n"), ("pot in photos only", "r"),
                                     ("scan dots unchanged", "b"), ("one mask condition", "g"),
                                     ("mesh + 2 metrics", "n")]):
        panel(s, M + i * (sw + Inches(0.1)), BT, sw, Inches(0.72), tone)
        _a, _b, r = txt(s, M + i * (sw + Inches(0.1)) + Inches(0.08), BT + Inches(0.2),
                        sw - Inches(0.16), Inches(0.4), size=11, bold=True,
                        color=TONE[tone][1], align=PP_ALIGN.CENTER)
        r.text = lab
    card(s, M, BT + Inches(1.0), CW, Inches(1.5), "1 · Guidance off", "No target mask at all.", "n")
    card(s, M + CW + Inches(0.18), BT + Inches(1.0), CW, Inches(1.5), "2 · Current SAM2 mask",
         "What the pipeline builds today.", "r")
    card(s, M + 2 * (CW + Inches(0.18)), BT + Inches(1.0), CW, Inches(1.5), "3 · Verified clean mask",
         "Depth-corrected, pot excluded by construction.", "g")
    footer(s, "This is the upper bound: if a perfect mask cannot suppress the pot, no better segmentation will.")

    s = S("The mask · outputs", "The pot stays in the reconstruction under every mask",
          "The five tables (of 14) where the clean mask helped MOST")
    picture(s, "cleanmask")
    footer(s, "Red shrinks a little across each row and never disappears. Every panel carries its own score.")

    # 18 -----------------------------------------------------------------
    s = S("The mask · numbers", "Completeness improves; leakage does not",
          "Averages over 14 unseen tables · lower is better for both")
    table(s, M, BT + Inches(0.25), CW2,
          [["What the model got", "Pot in result", "Missing table"],
           ["no guidance", "2.34", "14.36"], ["mask as built today", "2.02", "11.78"],
           ["verified clean mask", "1.91", "11.49"]], widths=[.46, .27, .27], hi=3)
    table(s, M + CW2 + Inches(0.28), BT + Inches(0.25), CW2,
          [["Paired comparison", "Change", "Interval"],
           ["clean vs off (leak)", "−0.43", "[−1.00, +0.09]"],
           ["clean vs today (leak)", "−0.11", "[−0.31, +0.05]"],
           ["clean vs off (missing)", "−2.87", "[−4.63, −1.36]"]], widths=[.42, .22, .36], hi=3)
    _a, _b, r = txt(s, M, BT + Inches(1.9), W - 2 * M, Inches(0.6), size=11.5, color=INK2)
    r.text = ("An interval crossing zero means the effect could be noise. Both leak intervals cross "
              "zero; the missing-surface interval does not.")
    footer(s, "The clean-vs-today interval is tight (±0.2) — a measured ceiling, not a small-sample shrug.")

    s = S("What the fine-tune achieves", "The supported claim is better completeness — not clutter removal",
          "The TRUE shape drawn in red wherever the model failed to build it")
    picture(s, "complete")
    footer(s, "Report the model as reducing missing target surface by about 2.9 points on unseen tables.")

    s = S("Promising · not yet confirmed", "Blocking clutter in BOTH inputs cuts leakage by more than half",
          "Delete the pot's scan dots and correct the mask, so neither input asks for it")
    picture(s, "combo")
    footer(s, "Leak 8.3 → 3.8 on 8 objects chosen for good segmentation. Needs a fresh unselected set.", True)

    # 21 -----------------------------------------------------------------
    s = S("Real-data check", "Real clutter leaks far less than our synthetic pot",
          "The dataset already contains tables with real objects standing on them")
    table(s, M, BT, Inches(7.2),
          [["Real object", "Things on it", "Could have leaked", "Actually leaked"],
           ["kitchen island", "7", "13.1%", "2.26%"], ["cutting board", "1", "9.3%", "1.88%"],
           ["TV stand", "2", "36.6%", "0.85%"], ["desk table", "1", "20.4%", "0.00%"],
           ["cabinet", "1", "23.8%", "0.00%"]], widths=[.32, .2, .24, .24], hi=1)
    x = M + Inches(7.5)
    _a, _b, r = txt(s, x, BT, W - x - M, Inches(0.7), size=34, bold=True, color=GREEN,
                    align=PP_ALIGN.CENTER)
    r.text = "0.73%"
    _a, _b, r = txt(s, x, BT + Inches(0.62), W - x - M, Inches(0.3), size=11, color=INK2,
                    align=PP_ALIGN.CENTER)
    r.text = "average real leakage"
    _a, _b, r = txt(s, x, BT + Inches(1.0), W - x - M, Inches(0.7), size=34, bold=True,
                    color=ACCENT, align=PP_ALIGN.CENTER)
    r.text = "8–12%"
    _a, _b, r = txt(s, x, BT + Inches(1.62), W - x - M, Inches(0.3), size=11, color=INK2,
                    align=PP_ALIGN.CENTER)
    r.text = "our synthetic pot"
    card(s, x, BT + Inches(2.0), W - x - M, Inches(1.3), "",
         "The 'could have leaked' column shows the model had the OPPORTUNITY to leak 13–37% and "
         "mostly did not. On the desk table there is a laptop worth 20% and ShapeR builds none of it.", "y")
    footer(s, "The synthetic pot made this limitation look roughly 5–10 times larger than it is on real data.")

    s = S("Every experiment", "All results with their uncertainty, on one page",
          "A bar crossing zero means the effect cannot be distinguished from luck")
    picture(s, "summary")
    footer(s, "Green met its pre-registered criterion; red did not; grey measures the problem rather than fixing it.")

    # 23 -----------------------------------------------------------------
    s = S("What failed", "Four attempts that did not work, and what each one taught",
          "Kept because several taught us more than the successes")
    fails = [("Smarter chooser", "30 scoring rules. The practice-set winner was SIGNIFICANTLY WORSE on "
              "held-out objects — textbook overfitting, caught only because objects were held back."),
             ("Merge all eight meshes", "Removed invented surface but also deleted REAL surface. "
              "Unseen-but-real surface is inconsistent across draws too, so agreement cannot separate them."),
             ("Reduce 16 photos to 4", "OUR TEST WAS AT FAULT, not the model. Photo count leaves the scan "
              "dots untouched, so the dominant input never weakened. Corrected by thinning the dots."),
             ("Delete the pot's scan dots alone", "Worked on 5 initial tables, then DAMAGED reconstructions "
              "on 14 fresh ones. The held-out result is the one to trust.")]
    for i, (t_, b_) in enumerate(fails):
        card(s, M + (i % 2) * (CW2 + Inches(0.28)), BT + (i // 2) * Inches(1.72),
             CW2, Inches(1.55), t_, b_, "r")
    footer(s, "Every failure above was caught by testing on objects the method had not been tuned on.")

    # 24 -----------------------------------------------------------------
    s = S("Withdrawn findings", "Two results we reported and then took back",
          "Both were withdrawn after re-testing, before they reached any conclusion")
    card(s, M, BT, CW2, Inches(2.0), "1 · Mask quality predicts suppression",
         "We found a strong pattern (correlation +0.72): the cleaner SAM2's mask, the more the "
         "guidance helped.\n\nIt came from 5 objects, 2 of which were in the fine-tune's TRAINING "
         "set. On 14 unseen tables the pattern disappears entirely.", "y")
    card(s, M + CW2 + Inches(0.28), BT, CW2, Inches(2.0), "2 · A 'perfect' mask that was not",
         "Our first oracle mask covered 76–100% of the pot, because a table's outline fills the "
         "pixels the pot hides.\n\nCaught by checking the mask against ground truth BEFORE running "
         "the experiment on it.", "y")
    card(s, M, BT + Inches(2.2), W - 2 * M, Inches(1.1), "Why this belongs in the deck",
         "Both would have produced confident, wrong conclusions. Showing them is what makes the "
         "surviving results credible.", "b")
    footer(s, "The earlier −0.55 image-clutter result is also training-contaminated and is not quoted as confirmed.")

    # 25 -----------------------------------------------------------------
    s = S("Conclusions", "What you can confidently say from these experiments",
          "Each claim below is backed by held-out objects and a fixed success criterion")
    card(s, M, BT, CW, Inches(1.8), "1 · Strongest result",
         "Seed consensus cuts invented surface 30.3% → 24.4% on 40 fresh objects. No retraining.", "g")
    card(s, M + CW + Inches(0.18), BT, CW, Inches(1.8), "2 · What the mask improves",
         "The fine-tune reduces missing target surface by about 2.9 points on 14 unseen tables.", "b")
    card(s, M + 2 * (CW + Inches(0.18)), BT, CW, Inches(1.8), "3 · What is not supported",
         "A better mask alone does not remove clutter. The ceiling was measured with a perfect mask "
         "and is near zero.", "r")
    card(s, M, BT + Inches(2.0), W - 2 * M, Inches(1.35), "One sentence",
         "ShapeR's largest controllable error is RANDOM INVENTED SURFACE, and seed consensus reduces "
         "it. The mask fine-tune improves COMPLETENESS. Clutter removal requires acting on the scan "
         "dots, not the photos — and that combined method still needs a fresh test.", "b")
    footer(s, "Open: the selector captures only about half the available gain; the combined method needs confirming.")

    assert n[0] == N, f"slide count {n[0]} != {N}"
    prs.save(str(OUT))
    print(f"wrote {OUT}  ({OUT.stat().st_size/1e6:.2f} MB, {n[0]} slides)")


if __name__ == "__main__":
    build()
