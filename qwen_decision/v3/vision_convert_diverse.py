"""Image training converters for more fields (2026-11-02): new image domains and new task shapes, for task
diversity (see vision_mix for the item fields). Gold comes from the sources' human labels or from a program over
them; never from a model.

New domains, one small train part each (vision_sources):
  dtd           classification  DTD textures: "What texture is shown in this image?", 3 other textures
  gtsrb         classification  GTSRB traffic signs; distractors share the class name's first 3 words first
                                ("red and white circle ...")
  aircraft      classification  FGVC-Aircraft variants (2 of 6 shards, 48 classes); distractors share the first 3
                                characters ("737-", "A32") first
  eurosat       remote-sensing  EuroSAT land use / land cover, 10 classes
  resisc45      remote-sensing  NWPU-RESISC45 aerial scenes, 45 classes
  country211    geolocation     Country211 (YFCC photos): "In which country was this photo taken?" (1 of 7
                                shards, so the countries early in its list)
  pcam          medical         PatchCamelyon: does the patch contain metastatic tumour tissue (Yes/No, balanced)
  retinopathy   medical         EyePACS fundus photos: the 5 grades as ordered options, grades spread evenly
  vqa_rad       medical         VQA-RAD train (clinicians' QA): yes/no balanced; open answers get 3 other answers
                                to questions with the same first two words, then any open answer
  path_vqa      medical         PathVQA train, the same way; its QA were made by rules from textbook captions,
                                so the label is program
  svhn          reading         SVHN (VTAB): the centre digit, gold plus 3 other digits ascending
  rendered_sst2 reading         SST-2 sentences rendered as images: positive or negative
  iam_line      reading         IAM handwritten lines: the transcription and 3 copies with one or two words swapped
                                for corpus words of about the same length
  dmlab         depth           DeepMind Lab frames (VTAB): how far the nearest object is (3 ordered bins), or
                                whether it is an apple/melon or a lemon
  smallnorb     3d-pose         smallNORB (VTAB elevation labels, 30-70 degrees): two toys, which is seen from
                                higher; elevations at least 20 degrees apart
  clevr_distance depth          CLEVR (VTAB closest-object distance bins): two scenes, in which is the nearest
                                object closer to the camera; bins at least 3 apart

New task shapes over sources already built:
  cls_pick      multi-image     4 images of one classification source (cars, flowers, food101, pets, places365,
                                dtd, gtsrb, aircraft, eurosat, resisc45; their deduplicated pools): which shows the
                                named class
  cls_odd       multi-image     4 images, 3 of one class: which shows a different class
  cls_same      multi-image     2 images: do they show the same class (Yes/No, balanced)
  fsc_compare   counting        two fsc147_crop crops of one object class from different photos, counts apart by
                                at least 2 and a quarter: which has more
  android_next  web             AndroidControl (human trajectories): a screen, the step's instruction, and 3
                                screens of the same episode: which comes right after the step. Screens are
                                downscaled to 1024 px on the long side; steps that change little are dropped

Two-image and four-image items name their images "Image 1".."Image 4" in order, options are those names (ordered),
and the gold position is balanced. The new multi-image tasks reuse pool images, so a source's dev images can be
another task's training images; dev only tracks training, so this was accepted.
"""
from __future__ import annotations

import base64
import io
import random
import re
import tarfile
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq

from qwen_decision.v3.vision_convert_misc import RAW, classify, label_names, parquets
from qwen_decision.v3.vision_mix import POOL, item, norm


def raw_dir(repo: str) -> Path:
    return next(RAW.glob(repo.replace("/", "__") + "@*"))


def classnames(d: Path) -> list[str]:
    return [c for c in (d / "classnames.txt").read_text(encoding="utf-8").split("\n") if c]


def wds_index(d: Path) -> list[tuple[Path, str, int]]:
    """(tar, image member, class) for every sample in the downloaded train shards."""
    out = []
    for t in sorted((d / "train").glob("*.tar")):
        with tarfile.open(t) as tf:
            members = tf.getmembers()
            cls = {m.name.rsplit(".", 1)[0]: int(tf.extractfile(m).read()) for m in members if m.name.endswith(".cls")}
            out += [(t, m.name, cls[m.name.rsplit(".", 1)[0]]) for m in members
                    if m.name.rsplit(".", 1)[-1] in ("jpg", "jpeg", "png", "webp") and m.name.rsplit(".", 1)[0] in cls]
    return out


def spread(samples: list, cap: int, rng: random.Random, key=lambda s: s[2]) -> list:
    """Up to cap samples, classes taken in turn so rare classes are not drowned."""
    by = defaultdict(list)
    for s in samples:
        by[key(s)].append(s)
    for v in by.values():
        rng.shuffle(v)
    labels, picks, depth = sorted(by), [], 0
    while len(picks) < cap and any(depth < len(by[c]) for c in labels):
        layer = [by[c][depth] for c in labels if depth < len(by[c])]
        rng.shuffle(layer)
        picks += layer[:cap - len(picks)]
        depth += 1
    return picks


def wds_images(saver, samples) -> dict[tuple[Path, str], str]:
    """Saved image path per (tar, member), reading each tar once."""
    by_tar, out = defaultdict(set), {}
    for t, name, *_ in samples:
        by_tar[t].add(name)
    for t, names in sorted(by_tar.items()):
        with tarfile.open(t) as tf:
            for m in tf:
                if m.name in names:
                    out[(t, m.name)] = saver.bytes(tf.extractfile(m).read())
    return out


def sid(t: Path, name: str) -> str:
    return f"{t.stem}-{name.rsplit('.', 1)[0]}"


def balanced(rng: random.Random, counts: Counter, n: int) -> int:
    """The least used gold position so far (ties at random)."""
    low = min(counts[k] for k in range(n))
    k = rng.choice([k for k in range(n) if counts[k] == low])
    counts[k] += 1
    return k


def wds_name(saver, cap, rng, source, repo, question, others, subtask=lambda lab: "class", names=None,
             ordered=False, label="human"):
    """One image, its class name among options; others(lab) -> candidate pools tried in order for 3 distractors."""
    d = raw_dir(repo)
    names = names or classnames(d)
    picks = spread([s for s in wds_index(d) if s[2] < len(names)], cap, rng)
    paths = wds_images(saver, picks)
    for t, name, lab in picks:
        picked = []
        for pool in others(lab):
            pool = [c for c in pool if c != lab and c not in picked]
            rng.shuffle(pool)
            picked += pool[:3 - len(picked)]
        if len(picked) < 3:
            continue
        labs = sorted([lab] + picked) if ordered else [lab] + picked
        yield item(source, sid(t, name), [paths[(t, name)]], question, [names[c] for c in labs], labs.index(lab),
                   subtask(lab), label, ordered=ordered, label_name=names[lab])


# ---------------------------------------------------------------- new domains, one image

def dtd(saver, cap, rng):
    files = parquets(raw_dir("tanganke/dtd") / "data")
    names = label_names(files[0])
    every = list(range(len(names)))
    yield from classify(saver, cap, rng, "dtd", files, names, lambda lab, _: "What texture is shown in this image?",
                        lambda lab, _: [every], lambda lab, _: "texture")


def gtsrb(saver, cap, rng):
    files = parquets(raw_dir("tanganke/gtsrb") / "data")
    names = label_names(files[0])
    head = [" ".join(n.split()[:3]) for n in names]
    every = list(range(len(names)))
    yield from classify(saver, cap, rng, "gtsrb", files, names, lambda lab, _: "Which traffic sign is this?",
                        lambda lab, _: [[c for c in every if head[c] == head[lab]], every],
                        lambda lab, _: "sign")


def eurosat(saver, cap, rng):
    files = parquets(raw_dir("tanganke/eurosat") / "data")
    names = label_names(files[0])
    every = list(range(len(names)))
    yield from classify(saver, cap, rng, "eurosat", files, names,
                        lambda lab, _: "What land use or land cover does this satellite image show?",
                        lambda lab, _: [every], lambda lab, _: "land cover")


def resisc45(saver, cap, rng):
    files = parquets(raw_dir("tanganke/resisc45") / "data")
    names = label_names(files[0])
    every = list(range(len(names)))
    yield from classify(saver, cap, rng, "resisc45", files, names, lambda lab, _: "What does this aerial image show?",
                        lambda lab, _: [every], lambda lab, _: "scene")


def aircraft(saver, cap, rng):
    names = classnames(raw_dir("clip-benchmark/wds_fgvc_aircraft"))
    every = list(range(len(names)))
    yield from wds_name(saver, cap, rng, "aircraft", "clip-benchmark/wds_fgvc_aircraft",
                        "Which aircraft variant is this?",
                        lambda lab: [[c for c in every if names[c][:3] == names[lab][:3]], every],
                        lambda lab: names[lab][:3])


def country211(saver, cap, rng):
    names = classnames(raw_dir("clip-benchmark/wds_country211"))
    every = list(range(len(names)))
    yield from wds_name(saver, cap, rng, "country211", "clip-benchmark/wds_country211",
                        "In which country was this photo taken?", lambda lab: [every], lambda lab: "country")


def svhn(saver, cap, rng):
    digits = [str(k) for k in range(10)]
    yield from wds_name(saver, cap, rng, "svhn", "clip-benchmark/wds_vtab-svhn",
                        "What is the digit in the centre of the image?", lambda lab: [list(range(10))],
                        lambda lab: "digit", names=digits, ordered=True)


def retinopathy(saver, cap, rng):
    d = raw_dir("clip-benchmark/wds_vtab-diabetic_retinopathy")
    names = [n[0].upper() + n[1:] for n in classnames(d)]
    picks = spread(wds_index(d), cap, rng)
    paths = wds_images(saver, picks)
    for t, name, lab in picks:
        yield item("retinopathy", sid(t, name), [paths[(t, name)]],
                   "What grade of diabetic retinopathy does this retina photo show?", names, lab, "grade", "human",
                   ordered=True)


def yes_no_wds(saver, cap, rng, source, repo, question, yes_class, subtask):
    d = raw_dir(repo)
    picks = spread(wds_index(d), cap, rng)
    paths = wds_images(saver, picks)
    for t, name, lab in picks:
        yield item(source, sid(t, name), [paths[(t, name)]], question, ["Yes", "No"], 0 if lab == yes_class else 1,
                   subtask, "human", ordered=True)


def pcam(saver, cap, rng):
    yield from yes_no_wds(saver, cap, rng, "pcam", "clip-benchmark/wds_vtab-pcam",
                          "Does this lymph node tissue patch contain metastatic tumour tissue?", 1, "tumour")


def rendered_sst2(saver, cap, rng):
    d = raw_dir("clip-benchmark/wds_renderedsst2")
    picks = spread(wds_index(d), cap, rng)
    paths = wds_images(saver, picks)
    for t, name, lab in picks:
        yield item("rendered_sst2", sid(t, name), [paths[(t, name)]],
                   "Is the sentence in the image positive or negative?", ["Positive", "Negative"], 0 if lab else 1,
                   "sentiment", "human", ordered=True)


DMLAB_DISTANCE = ["Nearby", "Far", "Very far"]


def dmlab(saver, cap, rng):
    d = raw_dir("clip-benchmark/wds_vtab-dmlab")
    picks = spread(wds_index(d), cap, rng)
    paths = wds_images(saver, picks)
    for k, (t, name, lab) in enumerate(picks):
        if k % 2 == 0:
            yield item("dmlab", sid(t, name), [paths[(t, name)]],
                       "In this 3D game scene, how far away is the nearest object (an apple, a melon or a lemon)?",
                       DMLAB_DISTANCE, lab % 3, "distance", "program", ordered=True)
        else:
            yield item("dmlab", sid(t, name), [paths[(t, name)]],
                       "In this 3D game scene, is the nearest object an apple or melon, or a lemon?",
                       ["An apple or melon", "A lemon"], lab // 3, "kind", "program", ordered=True)


def pair_compare(saver, cap, rng, source, repo, question, min_gap, subtask, higher_wins=True):
    """Two samples whose ordinal classes differ by at least min_gap: which image has the larger (or smaller) one."""
    d = raw_dir(repo)
    samples = wds_index(d)
    rng.shuffle(samples)
    by = defaultdict(list)
    for s in samples:
        by[s[2]].append(s)
    labels = sorted(by)
    pairs, used, gold_at = [], set(), Counter()
    for a in samples:
        if len(pairs) >= cap:
            break
        if a in used:
            continue
        far = [c for c in labels if abs(c - a[2]) >= min_gap]
        if not far:
            continue
        pool = [s for s in by[rng.choice(far)] if s not in used]
        if not pool:
            continue
        b = pool[0]
        used |= {a, b}
        first_wins = (a[2] > b[2]) == higher_wins
        k = balanced(rng, gold_at, 2)
        pair = (a, b) if (k == 0) == first_wins else (b, a)
        pairs.append(pair)
    paths = wds_images(saver, [s for p in pairs for s in p])
    for a, b in pairs:
        gold = 0 if (a[2] > b[2]) == higher_wins else 1
        yield item(source, f"{sid(a[0], a[1])}+{sid(b[0], b[1])}", [paths[(a[0], a[1])], paths[(b[0], b[1])]],
                   "Two images are shown: Image 1 and Image 2. " + question, ["Image 1", "Image 2"], gold, subtask,
                   "program", ordered=True)


def smallnorb(saver, cap, rng):
    yield from pair_compare(saver, cap, rng, "smallnorb", "clip-benchmark/wds_vtab-smallnorb_label_elevation",
                            "In which image is the toy seen from higher above (the camera looking down more "
                            "steeply)?", 4, "elevation")


def clevr_distance(saver, cap, rng):
    yield from pair_compare(saver, cap, rng, "clevr_distance", "clip-benchmark/wds_vtab-clevr_closest_object_distance",
                            "In which image is the object nearest to the camera closer to the camera?", 3,
                            "closest_object", higher_wins=False)


# ---------------------------------------------------------------- medical VQA, handwriting

def capital_q(q: str) -> str:
    q = q.strip()
    q = q[0].upper() + q[1:]
    return q if q.endswith("?") else q + "?"


def qhead(q: str) -> str:
    return " ".join(q.lower().split()[:2])


def vqa(saver, cap, rng, source, repo, label):
    files = parquets(raw_dir(repo) / "data")
    rows = []
    for f in files:
        t = pq.read_table(f, columns=["question", "answer"]).to_pydict()
        rows += [(f, k, q, a.strip()) for k, (q, a) in enumerate(zip(t["question"], t["answer"])) if a.strip()]
    open_by_key, open_all = defaultdict(set), set()
    for _, _, q, a in rows:
        if a.lower() not in ("yes", "no"):
            open_by_key[qhead(q)].add(a)
            open_all.add(a)
    open_all = sorted(open_all)
    rng.shuffle(rows)
    yn, chosen = Counter(), []
    for f, k, q, a in rows:
        if len(chosen) >= cap:
            break
        if a.lower() in ("yes", "no"):
            gold = 0 if a.lower() == "yes" else 1
            if yn[gold] > yn[1 - gold]:
                continue
            yn[gold] += 1
            chosen.append((f, k, capital_q(q), ["Yes", "No"], gold, "yes/no", True))
            continue
        ds, seen = [], {norm(a)}
        for pool in (sorted(open_by_key[qhead(q)]), rng.sample(open_all, min(60, len(open_all)))):
            pool = list(pool)
            rng.shuffle(pool)
            for d in pool:
                if norm(d) in seen or norm(d) in norm(a) or norm(a) in norm(d):
                    continue
                ds.append(d)
                seen.add(norm(d))
                if len(ds) == 3:
                    break
            if len(ds) == 3:
                break
        if len(ds) < 3:
            continue
        opts = [a] + ds
        order = rng.sample(range(4), 4)
        chosen.append((f, k, capital_q(q), [opts[i] for i in order], order.index(0), "open", False))
    by_file = defaultdict(list)
    for c in chosen:
        by_file[c[0]].append(c)
    for f, cs in sorted(by_file.items()):
        col = pq.read_table(f, columns=["image"]).column("image")
        for _, k, q, opts, gold, sub, ordered in cs:
            path = saver.bytes(col[k].as_py()["bytes"])
            yield item(source, f"{Path(f).stem.split('-')[1]}-{k}", [path], q, opts, gold, sub, label,
                       ordered=ordered)


def vqa_rad(saver, cap, rng):
    yield from vqa(saver, cap, rng, "vqa_rad", "flaviagiammarino/vqa-rad", "human")


def path_vqa(saver, cap, rng):
    yield from vqa(saver, cap, rng, "path_vqa", "flaviagiammarino/path-vqa", "program")


WORD = re.compile(r"^[A-Za-z]{3,}$")


def iam_line(saver, cap, rng):
    f = raw_dir("Teklia/IAM-line") / "data/train.parquet"
    texts = pq.read_table(f, columns=["text"]).column("text").to_pylist()
    by_len = defaultdict(set)
    for t in texts:
        for w in t.split():
            if WORD.match(w):
                by_len[len(w)].add(w.lower())
    by_len = {n: sorted(v) for n, v in by_len.items()}
    order = list(range(len(texts)))
    rng.shuffle(order)
    chosen = []
    for k in order:
        if len(chosen) >= cap:
            break
        words = texts[k].split()
        slots = [i for i, w in enumerate(words) if WORD.match(w)]
        if len(slots) < 2:
            continue
        variants, seen = [], {texts[k]}
        for _ in range(20):
            new = list(words)
            for i in rng.sample(slots, rng.choice((1, 2))):
                w = words[i]
                pool = by_len.get(len(w), []) + by_len.get(len(w) + 1, [])
                r = rng.choice(pool) if pool else w.lower()
                if r == w.lower():
                    continue
                new[i] = r.capitalize() if w[0].isupper() else r
            v = " ".join(new)
            if v not in seen:
                variants.append(v)
                seen.add(v)
            if len(variants) == 3:
                break
        if len(variants) < 3:
            continue
        opts = [texts[k]] + variants
        o = rng.sample(range(4), 4)
        chosen.append((k, [opts[i] for i in o], o.index(0)))
    col = pq.read_table(f, columns=["image"]).column("image")
    for k, opts, gold in sorted(chosen):
        yield item("iam_line", str(k), [saver.bytes(col[k].as_py()["bytes"])], "Which text is written in this image?",
                   opts, gold, "line", "human")


# ---------------------------------------------------------------- new task shapes over built pools

KINDS = {"cars": "car model", "flowers": "kind of flower", "food101": "dish", "pets": "breed",
         "places365": "kind of place", "dtd": "texture", "gtsrb": "traffic sign", "aircraft": "aircraft variant",
         "eurosat": "land cover", "resisc45": "kind of scene"}
FOUR = "Four images are shown in order: Image 1, Image 2, Image 3 and Image 4. "
IMAGES4 = ["Image 1", "Image 2", "Image 3", "Image 4"]


def pool_items(name: str) -> list[dict]:
    import json

    path = POOL / f"{name}.dedup.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").split("\n") if line]


def class_pools():
    out = {}
    for source in KINDS:
        by = defaultdict(list)
        for it in pool_items(source):
            by[it["label_name"]].append(it["images"][0])
        out[source] = {k: v for k, v in by.items()}
    return out


def cls_task(saver, cap, rng, task):
    pools = class_pools()
    sources = sorted(pools)
    gold_at, yn = Counter(), Counter()
    made = 0
    for k in range(cap * 3):
        if made >= cap:
            break
        source = sources[k % len(sources)]
        by = pools[source]
        kind = KINDS[source]
        labels = sorted(by)
        if task == "cls_pick":
            lab = rng.choice(labels)
            others = rng.sample([c for c in labels if c != lab], 3)
            imgs = [rng.choice(by[lab])] + [rng.choice(by[c]) for c in others]
            g = balanced(rng, gold_at, 4)
            imgs.insert(g, imgs.pop(0))
            it = item(task, f"{source}:{k}", imgs, FOUR + f'Which image shows the {kind} "{lab}"?', IMAGES4, g,
                      source, "human", ordered=True)
        elif task == "cls_odd":
            many = [c for c in labels if len(set(by[c])) >= 3]
            if not many:
                continue
            lab = rng.choice(many)
            odd = rng.choice([c for c in labels if c != lab])
            imgs = rng.sample(sorted(set(by[lab])), 3)
            g = balanced(rng, gold_at, 4)
            imgs.insert(g, rng.choice(by[odd]))
            it = item(task, f"{source}:{k}", imgs, FOUR + f"Three of them show the same {kind}. Which image shows a "
                      f"different {kind}?", IMAGES4, g, source, "human", ordered=True)
        else:
            same = yn[0] <= yn[1]
            if same:
                pool = [c for c in labels if len(set(by[c])) >= 2]
                lab = rng.choice(pool)
                imgs = rng.sample(sorted(set(by[lab])), 2)
            else:
                a, b = rng.sample(labels, 2)
                imgs = [rng.choice(by[a]), rng.choice(by[b])]
            yn[0 if same else 1] += 1
            it = item(task, f"{source}:{k}", imgs, f"Two images are shown: Image 1 and Image 2. Do they show the "
                      f"same {kind}?", ["Yes", "No"], 0 if same else 1, source, "human", ordered=True)
        if it:
            made += 1
            yield it


def cls_pick(saver, cap, rng):
    yield from cls_task(saver, cap, rng, "cls_pick")


def cls_odd(saver, cap, rng):
    yield from cls_task(saver, cap, rng, "cls_odd")


def cls_same(saver, cap, rng):
    yield from cls_task(saver, cap, rng, "cls_same")


HOW_MANY = re.compile(r"^How many (.+) are in the image\?$")


def fsc_compare(saver, cap, rng):
    by = defaultdict(list)
    for it in pool_items("fsc147_crop"):
        m = HOW_MANY.match(it["question"])
        if m:
            by[m.group(1)].append((it["id"].split(":")[1], int(it["options"][it["gold"]]), it["images"][0]))
    classes = sorted(c for c, v in by.items() if len({p for p, _, _ in v}) >= 2)
    gold_at, made, seen = Counter(), 0, set()
    for k in range(cap * 4):
        if made >= cap:
            break
        cls = classes[k % len(classes)]
        a, b = sorted(rng.sample(by[cls], 2))
        if a[0] == b[0] or abs(a[1] - b[1]) < max(2, 0.25 * max(a[1], b[1])) or (a, b) in seen:
            continue
        seen.add((a, b))
        g = balanced(rng, gold_at, 2)
        more, less = (a, b) if a[1] > b[1] else (b, a)
        imgs = [more[2], less[2]] if g == 0 else [less[2], more[2]]
        it = item("fsc_compare", f"{a[0]}:{a[1]}+{b[0]}:{b[1]}", imgs,
                  f"Two images are shown: Image 1 and Image 2. Which image has more {cls}?", ["Image 1", "Image 2"],
                  g, "compare", "human", ordered=True)
        if it:
            made += 1
            yield it


ANDROID = "smolagents/android-control"
ANDROID_SIDE = 1024
SCREEN_DIFF = 6.0  # mean absolute difference (0-255) of 32x16 grey thumbnails below which two screens count as same


def thumb(im):
    return list(im.convert("L").resize((16, 32)).getdata())


def differ(a, b) -> bool:
    return sum(abs(x - y) for x, y in zip(a, b)) / len(a) > SCREEN_DIFF


def screen(shots: list[str], j: int, ims: dict, th: dict):
    """Thumbnail of screenshot j, decoding it once into ims."""
    from PIL import Image

    if j not in ims:
        ims[j] = Image.open(io.BytesIO(base64.b64decode(shots[j])))
        th[j] = thumb(ims[j])
    return th[j]


def android_next(saver, cap, rng):
    from PIL import Image

    files = sorted((raw_dir(ANDROID) / "data").glob("train-*.parquet"))
    rng.shuffle(files)
    gold_at, made = Counter(), 0
    for f in files:
        if made >= cap:
            break
        pf = pq.ParquetFile(f)
        for g in range(pf.num_row_groups):
            t = pf.read_row_group(g, columns=["episode_id", "screenshots_b64", "actions", "step_instructions"])
            for ep, shots, actions, steps in zip(*(t.column(c).to_pylist() for c in t.column_names)):
                if made >= cap or len(shots) < 5 or not steps:
                    continue
                cands = [i for i in range(min(len(steps), len(shots) - 1))
                         if actions[i]["action_type"] not in ("wait",) and steps[i]]
                if not cands:
                    continue
                i = rng.choice(cands)
                ims, th = {}, {}
                if not differ(screen(shots, i, ims, th), screen(shots, i + 1, ims, th)):
                    continue
                rest = [j for j in range(len(shots)) if j not in (i, i + 1)]
                rng.shuffle(rest)
                picks = []
                for j in rest:
                    if differ(screen(shots, j, ims, th), th[i + 1]) and differ(th[j], th[i]):
                        picks.append(j)
                        if len(picks) == 2:
                            break
                if len(picks) < 2:
                    continue
                k = balanced(rng, gold_at, 3)
                picks.insert(k, i + 1)
                paths = []
                for j in [i] + picks:
                    im = ims[j].convert("RGB")
                    s = ANDROID_SIDE / max(im.size)
                    if s < 1:
                        im = im.resize((round(im.width * s), round(im.height * s)), Image.LANCZOS)
                    paths.append(saver.pil(im))
                q = (f'Image 1 is a phone screen. On it, the user does this step: "{steps[i].strip()}" Which image shows '
                     "the screen right after this step: Image 2, Image 3 or Image 4?")
                it = item("android_next", f"{ep}:{i}", paths, q, ["Image 2", "Image 3", "Image 4"], k, "next_screen",
                          "human", ordered=True)
                if it:
                    made += 1
                    yield it


CONVERTERS = {
    "dtd": ("classification", dtd),
    "gtsrb": ("classification", gtsrb),
    "aircraft": ("classification", aircraft),
    "eurosat": ("remote-sensing", eurosat),
    "resisc45": ("remote-sensing", resisc45),
    "country211": ("geolocation", country211),
    "pcam": ("medical", pcam),
    "retinopathy": ("medical", retinopathy),
    "vqa_rad": ("medical", vqa_rad),
    "path_vqa": ("medical", path_vqa),
    "svhn": ("reading", svhn),
    "rendered_sst2": ("reading", rendered_sst2),
    "iam_line": ("reading", iam_line),
    "dmlab": ("depth", dmlab),
    "smallnorb": ("3d-pose", smallnorb),
    "clevr_distance": ("depth", clevr_distance),
    "cls_pick": ("multi-image", cls_pick),
    "cls_odd": ("multi-image", cls_odd),
    "cls_same": ("multi-image", cls_same),
    "fsc_compare": ("counting", fsc_compare),
    "android_next": ("web", android_next),
}
