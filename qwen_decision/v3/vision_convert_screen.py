"""Screen, receipt and form converters (2026-11-01): web pages and app screens in the board's Mind2Web
set-of-marks form, receipts and forms in its KIE form (CORD + FUNSD). See vision_mix for the item fields.

  mind2web       web  Mind2Web train: the kit's m2w_convert (viewport crop around the gold, gold plus 3 negatives as
                      boxes A-D, task, last 5 actions, operation), marks drawn with the board build's font. Human
                      action logs.
  cord           kie  CORD train shards: full_cord's field values (rank-picked distractors from the receipt's own
                      amounts) and field-present Yes/No, balanced per field like the kit's rows.balance.
  funsd          kie  FUNSD train: full_funsd's field values and field-present Yes/No (balanced). Images are
                      nielsr/funsd's; that copy has no question-answer links, so the entities and links come from
                      the official zip's training_data (sha256 = the kit's FUNSD_SHA256), read in place.
  rico_screenqa  web  ScreenQA train: human questions on RICO screens, one per screen. The answer is the text of the
                      one UI element the annotators selected, kept only when each human full answer contains it.
                      on/off and yes/no answers become 2 balanced options; numbers get numeric_distractors; other
                      answers get same-kind answers from the screen, then answers to the same question elsewhere.
  osatlas        web  OS-Atlas grounding, one item per screenshot from each part in turn (linux, macOS,
                      AndroidWorld, UIBert, widget captioning, RICOSCA): the element's box (relative ltrb) and 3
                      non-overlapping element boxes of the same screenshot as A-D. The gold's text must name one box
                      of the screen. Names from accessibility trees and RICOSCA's template commands are program
                      labels (RICOSCA scrolls and relative phrasings dropped); UIBert expressions and widget
                      captions are human.
  aitw           web  Android in the Wild general, standard train split, one tap per episode: the human's tap
                      (touch and lift within AitW's 0.04) must fall inside exactly one of AitW's element boxes, which
                      come from its OCR and icon detector; the previous actions name their element the same way.
  seeclick_ground web SeeClick's Common Crawl web pages (OS-Atlas web_domain, element texts and boxes from the
                      DOM, program labels), in osatlas's grounding form. Added 11-02: without Mind2Web the web
                      skill had no web page, and held-out VisualWebBench fell on every subtask.
  seeclick_name   web The same pages, one element in a red box (VisualWebBench's mark): which of 4 texts of the
                      page, all visible, is the boxed one. The two sources take disjoint pages. Only the pages used
                      are fetched, by range requests into the split zip (seeclick_fetch).
android_control is not built: smolagents/android-control has screenshots and actions but no element boxes.
"""
from __future__ import annotations

import gzip
import hashlib
import io
import json
import random
import re
import struct
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq
from decision_index.vision import convert as kit
from decision_index.vision.build import VH, m2w_box
from decision_index.vision.images import MAX_PIXELS, fit

from qwen_decision.v3.vision_mix import ROOT, item, norm

RAW = ROOT / "data/raw"
FONT = "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"  # the board build's mark font
M2W = RAW / "osunlp__Multimodal-Mind2Web@1b4c6a8cf9f7/data"
CORD = RAW / "naver-clova-ix__cord-v2@7f0115a4b758/data"
FUNSD = RAW / "nielsr__funsd@7e7eeeedd84c/data"
FUNSD_ZIP = RAW / "extracted/funsd/dataset.zip"  # https://guillaumejaume.github.io/FUNSD/dataset.zip
SCREENQA = RAW / "bevaya__RICO-ScreenQA@44c8508ea528/data"
ATLAS = RAW / "OS-Copilot__OS-Atlas-data@e3a4c90c5f61"
AITW = RAW / "leosltl__Android-in-the-Wild@89110e13fa41"
BOXES = [f"box {kit.LETTERS[i]}" for i in range(4)]
MAX_TEXT = 2500


def sampled_rows(files: list[Path], columns: list[str], rng: random.Random, n: int):
    """Rows at seeded random positions over all files, n positions per pass, each row group read once a pass."""
    groups = [(f, g, pf.metadata.row_group(g).num_rows) for f in files
              for pf in [pq.ParquetFile(f)] for g in range(pf.num_row_groups)]
    pos = [(i, r) for i, (_, _, k) in enumerate(groups) for r in range(k)]
    rng.shuffle(pos)
    for start in range(0, len(pos), n):
        batch = defaultdict(list)
        for i, r in pos[start:start + n]:
            batch[i].append(r)
        order = sorted(batch)
        rng.shuffle(order)
        for i in order:
            f, g, _ = groups[i]
            rows = sorted(batch[i])
            got = pq.ParquetFile(f).read_row_group(g, columns=columns).take(rows).to_pylist()
            yield from (((f.name, g, r), row) for r, row in zip(rows, got))


def open_image(data: bytes):
    from PIL import Image

    Image.MAX_IMAGE_PIXELS = None
    return Image.open(io.BytesIO(data)).convert("RGB")


def capped(im):
    """The image at the kit's engine cap, so marks are drawn at the resolution the model sees."""
    from PIL import Image

    size = fit(im.width, im.height, MAX_PIXELS)
    return im if size == im.size else im.resize(size, Image.LANCZOS)


def pixels(b, W: int, H: int) -> list[float]:
    """A relative [left, top, right, bottom] box in pixels."""
    return [b[0] * W, b[1] * H, b[2] * W, b[3] * H]


def too_long(question: str, options: list) -> bool:
    return len(question) + sum(len(str(o)) for o in options) > MAX_TEXT


# --- Mind2Web ---------------------------------------------------------------------------------------------------

def m2w_item(saver, row: dict, rng: random.Random):
    """The kit's m2w_convert, writing the marked crop through the Saver."""
    if not row["pos_candidates"]:
        return None
    g = m2w_box(row["pos_candidates"][0])
    if not g or g[2] - g[0] < 4 or g[3] - g[1] < 4:
        return None
    im = open_image(row["screenshot"]["bytes"])
    W, H = im.size
    vh = min(H, VH)
    cy = (g[1] + g[3]) / 2
    top = max(0, int(min(max(0, cy - vh / 2 + rng.uniform(-200, 200)), H - vh)))
    if not (g[1] >= top and g[3] <= top + vh and g[2] <= W):
        return None
    negs = []
    for cand in row["neg_candidates"]:
        b = m2w_box(cand)
        if not b:
            continue
        w, h = b[2] - b[0], b[3] - b[1]
        if w < 8 or h < 8 or w > 0.6 * W or h > 0.4 * vh:
            continue
        if b[1] < top or b[3] > top + vh or b[2] > W:
            continue
        if kit.overlaps(b, g) or any(kit.overlaps(b, x) for x in negs):
            continue
        negs.append(b)
    if len(negs) < 3:
        return None
    boxes = rng.sample(negs, 3) + [g]
    rng.shuffle(boxes)
    crop = im.crop((0, top, W, top + vh))
    px = [[b[0], b[1] - top, b[2], b[3] - top] for b in boxes]
    img = saver.pil(kit.draw_marks(crop, px, FONT))
    op = json.loads(row["operation"])
    prev = row["action_reprs"][:int(row["target_action_index"])]
    hist = "\n".join(f"- {a}" for a in prev[-5:]) or "- (none)"
    q = (f"Task: {row['confirmed_task']}\nPrevious actions:\n{hist}\n"
         f"Which marked element should the next action ({op['op']}) be performed on?")
    return item("mind2web", row["action_uid"], [img], q, BOXES, boxes.index(g), row["domain"], "human",
                ordered=True, op=op["op"])


def mind2web(saver, cap: int, rng: random.Random):
    cols = ["action_uid", "operation", "pos_candidates", "neg_candidates", "domain", "confirmed_task", "screenshot",
            "action_reprs", "target_action_index"]
    for _, row in sampled_rows(sorted(M2W.glob("train-*.parquet")), cols, rng, round(cap * 1.2)):
        yield m2w_item(saver, row, rng)


# --- CORD and FUNSD ---------------------------------------------------------------------------------------------

def balanced_present(items: list[dict], key, rng: random.Random) -> list[dict]:
    """The kit's rows.balance: per key, as many Yes as No field_present items (a seeded pick of the larger side)."""
    groups = defaultdict(lambda: ([], []))
    for it in items:
        groups[key(it)][it["gold"]].append(it)
    out = []
    for yes, no in groups.values():
        m = min(len(yes), len(no))
        out += rng.sample(yes, m) + rng.sample(no, m)
    return out


def cord(saver, cap: int, rng: random.Random):
    """full_cord on the train shards."""
    files = sorted(CORD.glob("train-*.parquet"))
    recs = []
    for fi, f in enumerate(files):
        for i, gt in enumerate(pq.read_table(f, columns=["ground_truth"]).column("ground_truth").to_pylist()):
            recs.append((f"train-{fi}-{i}", fi, i, json.loads(gt)["gt_parse"]))
    rng.shuffle(recs)
    plan = []
    for key, fi, i, gp in recs:
        f, amounts, menu = kit.cord_fields(gp)
        vals = {}
        for a, k in amounts:
            vals.setdefault(a, set()).add(k)
        if len(vals) < 3:
            continue
        for fld in kit.CORD_VALUE_FIELDS:
            if fld == "item_price":
                names = Counter(n.lower() for n, _ in menu)
                for j, (n, pr) in enumerate(menu):
                    if names[n.lower()] == 1 and 2 <= len(n) <= 40 and re.search(r"[A-Za-z]{2}", n):
                        plan.append((key, fi, i, fld, j, n, pr, vals))
            else:
                g = kit.parse_amount(f.get(fld))
                if g:
                    plan.append((key, fi, i, fld, None, None, g, vals))
    n_elig = Counter(pl[3] for pl in plan)
    rank_cap = {fld: -(-n_elig[fld] // 2) for fld in kit.CORD_VALUE_FIELDS}
    rank_counts = {fld: Counter() for fld in kit.CORD_VALUE_FIELDS}
    found = []  # (key, file, row, sid, question, options, gold, subtask, extra)
    for key, fi, i, fld, j, nm, gold_v, vals in plan:
        q = kit.CORD_VALUE_Q[fld].format(nm=nm) if fld == "item_price" else kit.CORD_VALUE_Q[fld]
        pool = [(v, None) for v in vals if abs(v - gold_v) > 1e-9]
        dec = any(isinstance(v, float) and v != int(v) for v in list(vals) + [gold_v])
        fm = kit.fmt_dec if dec else kit.fmt_int
        ch = kit.pick_by_rank(gold_v, pool, 3, rng, rank_counts[fld], rank_cap[fld])
        if ch is None:
            continue
        opts = [fm(gold_v)] + [fm(x[0]) for x in ch]
        sid = f"{key}:value:{fld}" + (f":{j}" if j is not None else "")
        found.append((key, fi, i, sid, q, opts, 0, "field_value", {"field": fld}))
    present = []
    for key, fi, i, gp in recs:
        f = kit.cord_fields(gp)[0]
        for fld, what in kit.CORD_PRESENT_Q.items():
            has = fld in f and kit.parse_amount(f[fld] if not isinstance(f[fld], list) else f[fld][0]) is not None
            if not has and fld in f:
                continue
            present.append({"key": key, "fi": fi, "i": i, "field": fld, "gold": 0 if has else 1,
                            "q": f"Does this receipt show {what}?"})
    for p in balanced_present(present, lambda p: p["field"], rng):
        found.append((p["key"], p["fi"], p["i"], f"{p['key']}:present:{p['field']}", p["q"], ["Yes", "No"], p["gold"],
                      "field_present", {"field": p["field"]}))
    rng.shuffle(found)
    found = found[:cap]
    need = defaultdict(set)
    for key, fi, i, *_ in found:
        need[fi].add(i)
    saved = {}
    for fi in sorted(need):
        col = pq.read_table(files[fi], columns=["image"]).column("image")
        for i in sorted(need[fi]):
            saved[fi, i] = saver.bytes(col[i].as_py()["bytes"])
    for key, fi, i, sid, q, opts, gold, sub, extra in found:
        yield item("cord", sid, [saved[fi, i]], q, opts, gold, sub, "human", ordered=sub == "field_present", **extra)


def funsd_forms() -> dict[str, list[dict]]:
    """Entity lists (with question-answer links) of the official FUNSD training_data, by form id."""
    from decision_index.vision.spec import FUNSD_SHA256

    if hashlib.sha256(FUNSD_ZIP.read_bytes()).hexdigest() != FUNSD_SHA256:
        raise ValueError(f"{FUNSD_ZIP} is not the kit's pinned FUNSD zip")
    zf = zipfile.ZipFile(FUNSD_ZIP)
    base = "dataset/training_data/annotations/"
    return {Path(n).stem: json.loads(zf.read(n))["form"] for n in zf.namelist()
            if n.startswith(base) and n.endswith(".json")}


def funsd(saver, cap: int, rng: random.Random):
    """full_funsd on the train forms: nielsr/funsd images, official training annotations."""
    forms = funsd_forms()
    rows = pq.read_table(FUNSD / "train-00000-of-00001.parquet", columns=["image"]).column("image").to_pylist()
    images = {Path(r["path"]).stem: r["bytes"] for r in rows}
    if set(images) != set(forms):
        raise ValueError("nielsr/funsd train forms differ from the official training_data")
    value_c, present_c = [], []
    for fid in sorted(forms):
        ents = forms[fid]
        byid = {e["id"]: e for e in ents}
        answers = [e for e in ents if e["label"] == "answer" and 1 <= len(e["text"].strip()) <= 60]
        qtexts = Counter(e["text"].strip().lower() for e in ents if e["label"] == "question")
        for e in ents:
            if e["label"] != "question":
                continue
            qt = e["text"].strip()
            if qtexts[qt.lower()] != 1 or not re.search(r"[A-Za-z]{2}", qt) or len(qt) > 50:
                continue
            linked = [byid[b] for a, b in e["linking"] if a == e["id"] and b in byid and byid[b]["label"] == "answer"]
            linked = [x for x in linked if x["text"].strip()]
            name = qt.strip(" :\"'").strip()
            if len(linked) == 1 and 1 <= len(linked[0]["text"].strip()) <= 60:
                gold = linked[0]["text"].strip()
                gt = kit.funsd_type(gold)
                others = sorted({a["text"].strip() for a in answers
                                 if a["id"] != linked[0]["id"] and a["text"].strip().lower() != gold.lower()})
                same = [o for o in others if kit.funsd_type(o) == gt]
                if len(same) >= 3:
                    value_c.append((fid, e, name, gold, same, gt))
                present_c.append({"fid": fid, "e": e, "name": name, "gold": 0})
            elif not linked and not any(byid.get(b, {}).get("label") == "answer" for _, b in e["linking"]):
                x0, y0, x1, y1 = e["box"]
                near = [a for a in answers
                        if (abs((a["box"][1] + a["box"][3]) / 2 - (y0 + y1) / 2) < 12 and 0 <= a["box"][0] - x1 < 250)
                        or (0 <= a["box"][1] - y1 < 25 and abs(a["box"][0] - x0) < 60)]
                if not near:
                    present_c.append({"fid": fid, "e": e, "name": name, "gold": 1})
    found = []
    for fid, e, name, gold, same, gt in value_c:
        found.append((fid, f"{fid}:value:{e['id']}", f"On this form, what is filled in for the field \"{name}\"?",
                      [gold] + rng.sample(same, 3), 0, "field_value", {"field": name, "answer_type": gt}))
    for p in balanced_present(present_c, lambda p: "all", rng):
        found.append((p["fid"], f"{p['fid']}:present:{p['e']['id']}",
                      f"On this form, is the field \"{p['name']}\" filled in?", ["Yes", "No"], p["gold"],
                      "field_present", {"field": p["name"]}))
    rng.shuffle(found)
    saved = {}
    for fid, sid, q, opts, gold, sub, extra in found[:cap]:
        if fid not in saved:
            saved[fid] = saver.bytes(images[fid])
        yield item("funsd", sid, [saved[fid]], q, opts, gold, sub, "human", ordered=sub == "field_present", **extra)


# --- RICO-ScreenQA ----------------------------------------------------------------------------------------------

def loose(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def screenqa_answer(r: dict) -> str | None:
    """The text of the one UI element every annotator selected, when each human full answer contains it."""
    texts = {}
    for g in r["ground_truth"]:
        if len(g["ui_elements"]) != 1:
            return None
        t = re.sub(r"\s+", " ", g["ui_elements"][0]["text"] or "").strip().rstrip(",;")
        if not loose(t) or loose(t) not in loose(g["full_answer"]) or len(t) > 80:
            return None
        texts.setdefault(loose(t), t)
    return next(iter(texts.values())) if len(texts) == 1 else None


def answer_kind(a: str) -> str:
    if loose(a) in ("on", "off", "yes", "no"):
        return loose(a)
    if kit.NUM.match(a):
        return "number"
    t = kit.funsd_type(a)
    return "code" if t == "number" else t  # mostly digits: phone numbers, prices, times


def distinct(gold: str, question: str, cands: list[str]) -> list[str]:
    """Candidates that are not the gold, part of it or of the question (nor it part of them), deduplicated."""
    g, q, out, seen = loose(gold), f" {loose(question)} ", [], set()
    for c in cands:
        n = loose(c)
        if n and n not in seen and n != g and n not in g and g not in n and f" {n} " not in q:
            seen.add(n)
            out.append(c)
    return out


def rico_screenqa(saver, cap: int, rng: random.Random):
    files = sorted(SCREENQA.glob("train-*.parquet"))
    qa = []  # (file, row group, row, screen, question, answer)
    for f in files:
        pf = pq.ParquetFile(f)
        for g in range(pf.num_row_groups):
            for r, row in enumerate(pf.read_row_group(g, columns=["screen_id", "question", "ground_truth"]).to_pylist()):
                a = screenqa_answer(row)
                if a:
                    qa.append((f, g, r, row["screen_id"], row["question"].strip(), a))
    # fallback pools hold each answer once, so frequent answers ("Facebook", crawler accounts) are not mostly wrong
    by_screen, by_question, by_kind = defaultdict(list), defaultdict(dict), defaultdict(dict)
    for x in qa:
        by_screen[x[3]].append(x[5])
        by_question[loose(x[4])][x[5]] = None
        by_kind[answer_kind(x[5])][x[5]] = None
    by_question = {k: list(v) for k, v in by_question.items()}
    by_kind = {k: list(v) for k, v in by_kind.items()}
    rng.shuffle(qa)
    kinds, chosen, used = Counter(), [], set()
    binary_cap = cap // 10  # on/off and yes/no items: each answer at most this many, and about as many as its pair
    partner = {"on": "off", "off": "on", "yes": "no", "no": "yes"}
    for f, g, r, screen, q, a in qa:
        if screen in used:
            continue
        kind = answer_kind(a)
        if kind in partner:
            if kinds[kind] >= binary_cap or kinds[kind] > kinds[partner[kind]] + 20:
                continue
            opts = ["On", "Off"] if kind in ("on", "off") else ["Yes", "No"]
            gold, ordered = opts.index(kind.capitalize()), True
        else:
            # other answers on this screen, then answers to the same question elsewhere, then same-kind answers
            same = [o for o in distinct(a, q, by_screen[screen]) if answer_kind(o) == kind]
            if kind == "number":
                ds = kit.numeric_distractors(a, [o for o in same if o.endswith("%") == a.endswith("%")], rng)
            else:
                rng.shuffle(same)
                ds = same[:3]
                for pool in (by_question[loose(q)], by_kind[kind]):
                    if len(ds) < 3:
                        more = [o for o in pool if answer_kind(o) == kind]
                        more = distinct(a, q, rng.sample(more, min(len(more), 40)))
                        ds += [o for o in more if loose(o) not in {loose(d) for d in ds}][:3 - len(ds)]
            opts, gold, ordered = [a] + ds, 0, False
        if len(opts) not in (2, 4) or too_long(q, opts):
            continue
        used.add(screen)
        kinds[kind] += 1
        chosen.append((f, g, r, screen, q, opts, gold, ordered, kind))
        if len(chosen) >= cap:
            break
    need = defaultdict(list)
    for c in chosen:
        need[c[0], c[1]].append(c[2])
    images = {}
    for (f, g), rows in need.items():
        rows = sorted(set(rows))
        col = pq.ParquetFile(f).read_row_group(g, columns=["image"]).column("image").take(rows).to_pylist()
        images.update({(f, g, r): saver.bytes(v["bytes"]) for r, v in zip(rows, col)})
    for f, g, r, screen, q, opts, gold, ordered, kind in chosen:
        sub = "status" if kind in ("on", "off") else "yes/no" if kind in ("yes", "no") else kind
        yield item("rico_screenqa", f"{screen}:{f.stem[6:11]}-{g}-{r}", [images[f, g, r]], q, opts, gold, sub, "human",
                   ordered=ordered)


# --- set-of-marks grounding (OS-Atlas) ----------------------------------------------------------------------------

def marks_item(saver, source: str, sid: str, im, gold: list, others: list, question: str, subtask: str, label: str,
               rng: random.Random, **extra):
    """Gold box plus 3 of `others` (pixel xyxy on `im`) that are usable, inside the image and overlap neither the
    gold nor each other, drawn as boxes A-D on the image at the engine cap."""
    W0, H0 = im.size
    im = capped(im)
    W, H = im.size
    sx, sy = W / W0, H / H0

    def scaled(b):
        return [b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy]

    def usable(b, lo):  # Mind2Web's limits, except that full-width rows (common on phones) are allowed
        w, h = b[2] - b[0], b[3] - b[1]
        return (b[0] >= 0 and b[1] >= 0 and b[2] <= W and b[3] <= H and w >= lo and lo <= h <= 0.4 * H
                and w * h <= 0.25 * W * H)

    g = scaled(gold)
    if not usable(g, 6):
        return None
    pool = [scaled(b) for b in others]
    rng.shuffle(pool)
    negs = []
    for b in pool:
        if usable(b, 8) and not kit.overlaps(b, g) and not any(kit.overlaps(b, x) for x in negs):
            negs.append(b)
            if len(negs) == 3:
                break
    if len(negs) < 3 or too_long(question, BOXES):
        return None
    boxes = negs + [g]
    rng.shuffle(boxes)
    img = saver.pil(kit.draw_marks(im, [[round(v) for v in b] for b in boxes], FONT))
    return item(source, sid, [img], question, BOXES, boxes.index(g), subtask, label, ordered=True, **extra)


# OS-Atlas parts: name -> (annotation json, image zip, member prefix, label, phrasing)
ATLAS_PARTS = {
    "linux": ("desktop_domain/linux_splited.json", "desktop_domain/linux_images.zip", "", "program", "name"),
    "macos": ("desktop_domain/macos_splited.json", "desktop_domain/macos_images.zip", "", "program", "name"),
    "android_world": ("mobile_domain/aw_mobile.json", "mobile_domain/mobile_images.zip", "", "program", "name"),
    "uibert": ("mobile_domain/uibert_raw.json", "mobile_domain/UIBert.zip", "", "human", "instruction"),
    "widget_captioning": ("mobile_domain/widget_captioning.json", "mobile_domain/rico_imgs.zip", "combined/", "human",
                          "name"),
    "ricosca": ("mobile_domain/ricosca.json", "mobile_domain/rico_imgs.zip", "combined/", "program", "instruction"),
}

# RICOSCA's synthetic commands: scrolls are not element choices, and its relative phrasings ("the icon below",
# "item next to x") are often ambiguous among the other marked elements
RICOSCA_SKIP = re.compile(r"^(scroll|swipe)\b")
RICOSCA_RELATIVE = re.compile(r"\b(next to|above|below|beside|left|right|top|bottom|corner|center|middle)\b")


def atlas_screens(part: str) -> dict[str, list[tuple[str, tuple]]]:
    """(instruction, relative ltrb box) elements per screenshot, deduplicated."""
    recs = json.loads((ATLAS / ATLAS_PARTS[part][0]).read_text())
    out = defaultdict(dict)
    for r in recs:
        for e in r.get("elements") or [r]:
            text = re.sub(r"\s+", " ", e["instruction"] or "").strip()
            box = tuple(float(v) for v in e["bbox"])
            if not re.search(r"[A-Za-z0-9]", text) or len(box) != 4 or box[0] >= box[2] or box[1] >= box[3]:
                continue
            if part == "ricosca" and (RICOSCA_SKIP.match(text) or RICOSCA_RELATIVE.search(text)):
                continue
            out[r["img_filename"]][text, box] = None
    return {k: list(v) for k, v in out.items()}


def atlas_item(saver, part: str, zf, name: str, els: list, rng: random.Random):
    _, _, prefix, label, phrasing = ATLAS_PARTS[part]
    boxes_of = defaultdict(set)
    for t, b in els:
        boxes_of[norm(t)].add(b)
    # the gold's text names one element (one box) of the screen
    cands = [(t, b) for t, b in els if len(boxes_of[norm(t)]) == 1 and len(t) <= 200]
    if not cands:
        return None
    text, gold = rng.choice(cands)
    im = open_image(zf.read(prefix + name))
    W, H = im.size
    others = [pixels(b, W, H) for t, b in els if norm(t) != norm(text) and b != gold]
    if phrasing == "instruction":
        q = f"Instruction: {text}\nWhich marked element should this instruction be performed on?"
    else:
        q = f"Which marked element matches the description \"{text}\"?"
    sid = f"{part}:{name}:{hashlib.sha256(text.encode()).hexdigest()[:8]}"
    return marks_item(saver, "osatlas", sid, im, pixels(gold, W, H), others, q, part, label, rng)


def osatlas(saver, cap: int, rng: random.Random):
    """One item per screenshot; the parts give an item in turn, so each gives about the same number."""
    queues = []
    for part, (_, zname, prefix, _, _) in ATLAS_PARTS.items():
        zf = zipfile.ZipFile(ATLAS / zname)
        have = set(zf.namelist())
        screens = [(k, v) for k, v in sorted(atlas_screens(part).items()) if prefix + k in have and len(v) >= 4]
        rng.shuffle(screens)
        queues.append((part, zf, iter(screens)))
    while queues:
        for q in list(queues):
            part, zf, screens = q
            for name, els in screens:
                it = atlas_item(saver, part, zf, name, els, rng)
                if it:
                    yield it
                    break
            else:
                queues.remove(q)


# --- SeeClick web pages (OS-Atlas web_domain) -------------------------------------------------------------------

SEECLICK = ATLAS / "web_domain/seeclick_web.json"
SEECLICK_IMG = RAW / "extracted/seeclick_web"
SEECLICK_PARTS = [f"web_domain/seeclick_web_image_a{c}" for c in "abcd"]  # one zip split in byte parts
SEECLICK_PAGES = 1400  # pages fetched per source; the two sources take disjoint pages
NAME_Q = "Which text is the element in the red box?"


class Joined(io.RawIOBase):
    """Byte parts of one split file, read as a single seekable file."""

    def __init__(self, files: list):
        self.files, self.sizes = files, [f.size for f in files]
        self.size, self.pos = sum(self.sizes), 0

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = {0: off, 1: self.pos + off, 2: self.size + off}[whence]
        return self.pos

    def readinto(self, b):
        n, out, start = len(b), 0, 0
        for f, s in zip(self.files, self.sizes):
            if start <= self.pos < start + s and out < n:
                f.seek(self.pos - start)
                chunk = f.read(min(n - out, start + s - self.pos))
                b[out:out + len(chunk)] = chunk
                out += len(chunk)
                self.pos += len(chunk)
            start += s
        return out


def seeclick_fetch(names: list[str]) -> None:
    """Pages missing from SEECLICK_IMG, read from the remote split zip by range requests (not the 127 GB whole)."""
    from concurrent.futures import ThreadPoolExecutor

    from huggingface_hub import HfFileSystem

    SEECLICK_IMG.mkdir(parents=True, exist_ok=True)
    missing = [n for n in names if not (SEECLICK_IMG / n).exists()]
    if not missing:
        return
    repo = f"datasets/OS-Copilot/OS-Atlas-data@{ATLAS.name.split('@')[1]}/"

    def zip_file():
        fs = HfFileSystem()
        return zipfile.ZipFile(io.BufferedReader(Joined([fs.open(repo + p, "rb", block_size=1 << 20)
                                                         for p in SEECLICK_PARTS]), 1 << 20))

    def fetch(chunk):
        zf = zip_file()  # one reader per thread
        for n in chunk:
            tmp = SEECLICK_IMG / f"{n}.part"
            tmp.write_bytes(zf.read(f"seeclick_web_imgs/{n}"))
            tmp.rename(SEECLICK_IMG / n)

    with ThreadPoolExecutor(16) as ex:
        list(ex.map(fetch, [missing[i::16] for i in range(16)]))


def seeclick_pages(source: str, rng: random.Random) -> list[tuple[str, list]]:
    """SEECLICK_PAGES pages of this source (the sources split one seeded order in turn), fetched if needed."""
    recs = json.loads(SEECLICK.read_text())
    recs.sort(key=lambda r: r["img_filename"])
    random.Random("seeclick").shuffle(recs)  # the same order for both sources
    turn = ["seeclick_ground", "seeclick_name"].index(source)
    pages = []
    for r in recs[turn::2]:
        els = {}
        for e in r.get("elements") or []:
            text = re.sub(r"\s+", " ", e["instruction"] or "").strip()
            box = tuple(float(v) for v in e["bbox"])
            if re.search(r"[A-Za-z0-9]", text) and len(box) == 4 and box[0] < box[2] and box[1] < box[3]:
                els[text, box] = None
        if len(els) >= 4:
            pages.append((r["img_filename"], list(els)))
        if len(pages) == SEECLICK_PAGES:
            break
    seeclick_fetch([n for n, _ in pages])
    rng.shuffle(pages)
    return pages


def seeclick_ground(saver, cap: int, rng: random.Random):
    """OS-Atlas's grounding form on web pages: the element whose text is given, among 4 marked boxes."""
    for name, els in seeclick_pages("seeclick_ground", rng):
        boxes_of = defaultdict(set)
        for t, b in els:
            boxes_of[norm(t)].add(b)
        cands = [(t, b) for t, b in els if len(boxes_of[norm(t)]) == 1 and len(t) <= 200]
        if not cands:
            continue
        text, gold = rng.choice(cands)
        im = open_image((SEECLICK_IMG / name).read_bytes())
        W, H = im.size
        others = [pixels(b, W, H) for t, b in els if norm(t) != norm(text) and b != gold]
        q = f"Which marked element matches the description \"{text}\"?"
        yield marks_item(saver, "seeclick_ground", f"{name}:{hashlib.sha256(text.encode()).hexdigest()[:8]}", im,
                         pixels(gold, W, H), others, q, "ground", "program", rng)


def seeclick_name(saver, cap: int, rng: random.Random):
    """One element in a red box (VisualWebBench's mark); the options are its text and the texts of 3 other elements
    of the page, all visible, so the box has to be read."""
    from PIL import ImageDraw

    for name, els in seeclick_pages("seeclick_name", rng):
        boxes_of = defaultdict(set)
        for t, b in els:
            boxes_of[norm(t)].add(b)
        cands = [(t, b) for t, b in els if len(boxes_of[norm(t)]) == 1 and len(t) <= 80]
        if len(cands) < 4:
            continue
        im = capped(open_image((SEECLICK_IMG / name).read_bytes()))
        W, H = im.size
        rng.shuffle(cands)
        text, gold = cands[0]
        g = pixels(gold, W, H)
        w, h = g[2] - g[0], g[3] - g[1]
        if w < 6 or h < 6 or h > 0.4 * H or w * h > 0.25 * W * H:
            continue
        negs = [t for t, b in cands[1:] if not kit.overlaps(pixels(b, W, H), g)][:3]
        if len(negs) < 3:
            continue
        ImageDraw.Draw(im).rectangle([round(v) for v in g], outline=(255, 0, 0), width=max(2, round(min(W, H) / 300)))
        opts = [text, *negs]
        rng.shuffle(opts)
        yield item("seeclick_name", f"{name}:{hashlib.sha256(text.encode()).hexdigest()[:8]}", [saver.pil(im)],
                   NAME_Q, opts, opts.index(text), "name", "program")


# --- Android in the Wild ----------------------------------------------------------------------------------------

def _varint(b, i: int) -> tuple[int, int]:
    x = shift = 0
    while True:
        c = b[i]
        i += 1
        x |= (c & 0x7F) << shift
        shift += 7
        if c < 0x80:
            return x, i


def _fields(b):
    """(field number, wire type, value) of a protobuf message."""
    i = 0
    while i < len(b):
        key, i = _varint(b, i)
        f, wt = key >> 3, key & 7
        if wt == 2:
            n, i = _varint(b, i)
            yield f, wt, b[i:i + n]
            i += n
        elif wt == 0:
            v, i = _varint(b, i)
            yield f, wt, v
        elif wt in (1, 5):
            n = 8 if wt == 1 else 4
            yield f, wt, b[i:i + n]
            i += n
        else:
            raise ValueError(f"wire type {wt}")


def _feature(b) -> list:
    for f, _, v in _fields(b):
        out = []
        for _, wt, x in _fields(v):
            if f == 1:
                out.append(bytes(x))
            elif f == 2:
                out += struct.unpack(f"<{len(x) // 4}f", x)
            elif wt == 2:
                j = 0
                while j < len(x):
                    y, j = _varint(x, j)
                    out.append(y)
            else:
                out.append(x)
        return out
    return []


def tf_examples(path: Path):
    """tf.train.Example features of a gzipped TFRecord file, as {name: list}, without TensorFlow."""
    with gzip.open(path) as f:
        data = memoryview(f.read())
    i = 0
    while i + 12 <= len(data):
        (n,) = struct.unpack("<Q", data[i:i + 8])
        rec, i = data[i + 12:i + 12 + n], i + 16 + n
        ex = {}
        for _, _, feats in _fields(rec):
            for _, _, entry in _fields(feats):
                k = v = None
                for f, _, x in _fields(entry):
                    if f == 1:
                        k = bytes(x).decode()
                    else:
                        v = _feature(x)
                ex[k] = v
        yield ex


AITW_TAP, AITW_TYPE, AITW_BACK, AITW_HOME, AITW_ENTER = 4, 3, 5, 6, 7  # AitW action_type ids (DUAL_POINT = tap/swipe)


def aitw_boxes(ex: dict) -> list[tuple[list, str]]:
    """Element boxes as (normalized [left, top, right, bottom], description); AitW stores (y, x, height, width)."""
    pos = ex["image/ui_annotations_positions"]
    out = []
    for k, (t, kind) in enumerate(zip(ex["image/ui_annotations_text"], ex["image/ui_annotations_ui_types"])):
        y, x, h, w = pos[4 * k:4 * k + 4]
        kind = kind.decode()
        name = f"[text] {t.decode().strip()}" if kind == "TEXT" else f"[icon] {kind[5:].lower().replace('_', ' ')}"
        out.append(([x, y, x + w, y + h], name))
    return out


def aitw_step(ex: dict) -> tuple[str, list | None]:
    """A step as Mind2Web's action repr, and the tap point (x, y) when it is a tap."""
    a = ex["results/action_type"][0]
    if a == AITW_TAP:
        (ty, tx), (ly, lx) = ex["results/yx_touch"], ex["results/yx_lift"]
        if ((ty - ly) ** 2 + (tx - lx) ** 2) ** 0.5 <= 0.04:  # AitW's tap/swipe threshold
            hit = [n for b, n in aitw_boxes(ex) if b[0] <= tx <= b[2] and b[1] <= ty <= b[3]]
            return f"{hit[0] if len(hit) == 1 else '[unlabeled]'} -> TAP", [tx, ty]
        d = ("DOWN" if ly > ty else "UP") if abs(ly - ty) >= abs(lx - tx) else ("RIGHT" if lx > tx else "LEFT")
        return f"SWIPE {d}", None
    if a == AITW_TYPE:
        return f"TYPE: {ex['results/type_action'][0].decode()}", None
    return {AITW_BACK: "PRESS BACK", AITW_HOME: "PRESS HOME", AITW_ENTER: "PRESS ENTER"}.get(a, "STOP"), None


def aitw_item(saver, ep: list[dict], s: int, rng: random.Random):
    ex = ep[s]
    _, tap = aitw_step(ex)
    W, H = ex["image/width"][0], ex["image/height"][0]
    boxes = aitw_boxes(ex)
    hit = [b for b, _ in boxes if b[0] <= tap[0] <= b[2] and b[1] <= tap[1] <= b[3]]
    if len(hit) != 1:
        return None
    from PIL import Image

    im = Image.frombytes("RGB", (W, H), ex["image/encoded"][0])  # AitW stores raw RGB pixels
    hist = "\n".join(f"- {aitw_step(e)[0]}" for e in ep[max(0, s - 5):s]) or "- (none)"
    q = (f"Task: {ex['goal_info'][0].decode().strip()}\nPrevious actions:\n{hist}\n"
         "Which marked element should the next action (TAP) be performed on?")
    others = [pixels(b, W, H) for b, _ in boxes if b is not hit[0]]
    sid = f"{ex['episode_id'][0].decode()}:{ex['step_id'][0]}"
    kind = "icon" if next(n for b, n in boxes if b is hit[0]).startswith("[icon]") else "text"
    return marks_item(saver, "aitw", sid, im, pixels(hit[0], W, H), others, q, kind, "human", rng)


def aitw(saver, cap: int, rng: random.Random):
    """At most one tap per episode, at most `quota` episodes per shard so the sample spans the shards."""
    train = set(json.loads((AITW / "splits/standard.json").read_text())["train"])
    shards = sorted(p for p in (AITW / "general").iterdir() if p.stat().st_size > 1000)
    rng.shuffle(shards)
    quota = -(-2 * cap // len(shards))
    for path in shards:
        eps = defaultdict(list)
        for ex in tf_examples(path):
            if int(ex["episode_id"][0]) in train:
                eps[ex["episode_id"][0]].append(ex)
        order = sorted(eps)
        rng.shuffle(order)
        taken = 0
        for e in order:
            ep = sorted(eps[e], key=lambda x: x["step_id"][0])
            if [x["step_id"][0] for x in ep] != list(range(len(ep))):
                continue
            taps = [s for s, x in enumerate(ep) if aitw_step(x)[1] is not None]
            rng.shuffle(taps)
            for s in taps:
                it = aitw_item(saver, ep, s, rng)
                if it:
                    taken += 1
                    yield it
                    break
            if taken >= quota:
                break


CONVERTERS = {"mind2web": ("web", mind2web), "cord": ("kie", cord), "funsd": ("kie", funsd),
              "rico_screenqa": ("web", rico_screenqa), "osatlas": ("web", osatlas),
              "aitw": ("web", aitw), "seeclick_ground": ("web", seeclick_ground),
              "seeclick_name": ("web", seeclick_name)}
