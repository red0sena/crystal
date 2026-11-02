"""Image training converters, misc sources (2026-11-01): spatial QA, charts, classification, geometry and
image-text matching turned into the board's multiple-choice items (see vision_mix for the item fields).

  gqa          spatial   GQA balanced train (program QA from Visual Genome scene graphs). One question per image,
                         about 40% relation, 20% position, 20% compare, 20% other types; subtask = GQA's detailed
                         type. Yes/no -> ordered Yes/No, balanced; "X or Y" -> the two alternatives; left/right and
                         top/bottom queries -> the answer and its opposite. Other open answers get 3 distractors
                         from the alternatives GQA itself paired with that answer in its "X or Y" questions on the
                         same attribute (its generator avoids synonyms), at most one of them an object or answer of
                         the same image. Answers with no such alternatives (people: man/player/boy...) are skipped.
  chartqa      chart     ChartQA train, human questions only (the machine ones are T5-made). Yes/no -> Yes/No;
                         numbers -> 4 ascending numbers (other answers on the same chart, then kit
                         numeric_distractors variants; years step by one), gold rank balanced; text -> other text
                         answers of the same chart, then of questions with the same opening words, then any of the
                         same kind (colour or not). List answers ("[63, 26]") are skipped.
  pets, flowers, food101, cars, places365
               classification  Train splits, classes spread evenly. 4 options: the gold class and 3 others,
                         from the same species (pets) or make, then body type (cars); the rest have no
                         super-category in the data, so distractors are uniform over the other classes.
  geometry3k   math      Geometry3K train. This mirror kept only the gold choice, so the original 4 choices are
                         taken from the Cauldron's InterGPS copy (same diagram by dhash, choices containing the
                         gold); otherwise numbers get ascending numeric distractors and \\sqrt, \\frac, \\pi
                         answers get their numbers perturbed (value checked to differ, forms kept simplified).
  geoqa_plus   math      GeoQA+ train (Chinese). The original A-D choices; gold letter from "故选X" in the
                         human solution, else the solution's final value when exactly one choice equals it.
  coco_karpathy image-text  Karpathy train + restval (never val/test). "Which caption describes this image?"
                         (the board's Winoground wording): a true caption, 1-2 programmatic variants of it that
                         swap two nouns of the same kind (both animate or both not; symmetric "X and Y" swaps
                         excluded) or two adjectives of one kind (colours, sizes...) bound to different nouns, and
                         a caption of another image sharing at most one content word with this image's captions.
                         All options share one surface form (no final period, capital first letter).
  imagecode    image-text  ImageCoDe train only: the gold frame plus the 3 nearest frames of its 10-image set in
                         time order (random 3 for Open Images sets), gold position balanced; options "Image 1"..
                         "Image 4". Descriptions naming a position in the full set ("the last image") are skipped.

ScienceQA is not written here; the Cauldron's scienceqa covers it.
"""
from __future__ import annotations

import ast
import json
import math
import random
import re
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq
from decision_index.vision.convert import NUM, numeric_distractors, pick_by_rank

from qwen_decision.v3.vision_mix import ROOT, item

RAW = ROOT / "data/raw"
GQA = RAW / "lmms-lab-encoder__GQA@a6e72d6e1b91"
CHARTQA = RAW / "HuggingFaceM4__ChartQA@b605b6e08b57/data"
IMAGECODE = RAW / "BennoKrojer__ImageCoDe@b7b7a73dbc74"
COCO = RAW / "yerevann__coco-karpathy@448fdb1bc7b2/data"
COCO_ZIPS = {"train2014": RAW / "url/coco_images/train2014.zip", "val2014": RAW / "url/coco_images/val2014.zip"}
GEO3K = RAW / "hiyouga__geometry3k@fd21e533e1e5/data/train-00000-of-00001.parquet"
INTERGPS = RAW / "HuggingFaceM4__the_cauldron@847a98a779b1/intergps"
GEOQA = RAW / "Yuting6__geoqa_plus@5bfcaebb62f2/data/train-00000-of-00001.parquet"
MAX_TEXT = 2500


def parquets(path: Path, pattern: str = "train-*.parquet") -> list[Path]:
    return sorted(p for p in path.glob(pattern) if ".cache" not in p.parts)


def too_long(question: str, options: list[str]) -> bool:
    return len(question) + sum(len(o) for o in options) > MAX_TEXT


def stem(w: str) -> str:
    return re.sub(r"(es|s)$", "", w) if len(w) > 3 else w


def words(s: str) -> set[str]:
    return {stem(w) for w in re.findall(r"[a-z0-9]+", s.lower())}


def num_value(s: str) -> float | None:
    s = s.strip()
    return float(s.rstrip("%")) if NUM.match(s) else None


def number_variants(gold: str, rng: random.Random) -> list[str]:
    g = float(gold.rstrip("%"))
    if NUM.match(gold) and "." not in gold and not gold.endswith("%") and 1900 <= g <= 2100:
        return [str(int(g) + d) for d in (-5, -4, -3, -2, -1, 1, 2, 3, 4, 5)]  # years
    out = []
    for _ in range(3):
        out += numeric_distractors(gold, [], rng)
    if "." in gold:  # drop near-copies such as 1.813888887 for 1.813888889
        out = [v for v in out if abs(float(v.rstrip("%")) - g) >= 0.05 * abs(g)]
    return out


def ranked_numbers(gold: str, pool: list[str], rng: random.Random, counts: Counter):
    """Gold plus 3 numbers ascending, gold rank balanced by `counts`; pool numbers first, then kit variants."""
    g = num_value(gold)
    pct = gold.endswith("%")

    def year(v: float) -> bool:
        return v == int(v) and 1900 <= v <= 2100

    def fits(v: float) -> bool:  # pool numbers of another kind (a count beside years, x10 off) are not used
        return year(v) if year(g) else not year(v) and (g == 0 or v == 0 or 0.1 <= abs(v / g) <= 10)

    cands = {}
    pool = [p.strip() for p in pool if num_value(p.strip()) is not None and fits(num_value(p.strip()))]
    for p in pool + number_variants(gold, rng):
        v = num_value(p)
        if v is None or p.endswith("%") != pct or abs(v - g) < 1e-9 or v in cands:
            continue
        cands[v] = p
    chosen = pick_by_rank(g, list(cands.items()), 3, rng, counts)
    if chosen is None or len(chosen) < 3:
        return None
    ranked = sorted(chosen + [(g, gold)])
    return [s for _, s in ranked], [s for _, s in ranked].index(gold)


# ---------------------------------------------------------------- GQA

GQA_SHARE = {"relation": 0.4, "position": 0.2, "compare": 0.2, "other": 0.2}
GQA_PER_GROUP = 24  # images used per parquet row group (~98 images each), so a cap spreads over ~170 of 735
GQA_PAIR_TYPES = {"objThisChoose", "categoryThisChoose", "categoryRelOChoose", "chooseAttr", "materialChoose",
                  "activityChoose", "placeChoose", "weatherChoose", "stateChoose", "companyChoose", "typeChoose"}
REL_ANSWER = {"to the left of": "left", "to the right of": "right", "in front of": "front"}
REL_OPTION = {"to the left of": "left", "to the right of": "right", "in front of": "in front"}
OPPOSITE = {"left": "right", "right": "left", "top": "bottom", "bottom": "top"}
LAST_OP = re.compile(r"^([^:]+):\s?(.*?)\s*(\[[0-9, ]*\])?$")
NAMES = re.compile(r"(?:select|relate): ([^,(]+?)\s*[,(]")
ALTS = re.compile(r",\s*(?:the |a |an )?([^,]+?)\s+or\s+(?:the |a |an )?([^,]+?)\?$")


def gqa_bucket(structural: str, semantic: str, detailed: str) -> str:
    if detailed.startswith("position"):
        return "position"
    if structural == "compare":
        return "compare"
    return "relation" if semantic == "rel" else "other"


def gqa_options(q: dict, pairs: dict, scene: set[str], generic: set[str], rng: random.Random):
    """(options, gold, ordered) for one GQA question, or None."""
    ans, s, d, op, arg = q["answer"], q["structural"], q["detailed"], q["op"], q["arg"]
    if ans in ("yes", "no"):
        return (["Yes", "No"], 0 if ans == "yes" else 1, True) if s in ("verify", "logical", "compare") else None
    if s == "choose" and d == "relChooser":
        alts = arg.split(",")[1].split("|") if arg.count(",") >= 2 else []
        hits = [i for i, a in enumerate(alts) if REL_ANSWER.get(a, a) == ans]
        return ([REL_OPTION.get(a, a) for a in alts], hits[0], False) if len(alts) == 2 and len(hits) == 1 else None
    if s == "choose" and op.startswith("choose") and arg:
        alts = arg.split("|")
        return (alts, alts.index(ans), False) if len(alts) == 2 and ans in alts else None
    if d == "comparativeChoose":
        m = ALTS.search(q["question"])
        if not m:
            return None
        alts = list(m.groups())
        hits = [i for i, a in enumerate(alts) if a == ans] or [i for i, a in enumerate(alts) if stem(ans) in words(a)]
        return (alts, hits[0], False) if len(hits) == 1 else None
    if d == "twoCommon":
        kinds = ["color", "material"]
        return (kinds, kinds.index(ans), False) if ans in kinds else None
    if s != "query":
        return None
    if arg in ("hposition", "vposition"):
        return ([ans, OPPOSITE[ans]], 0, False) if ans in OPPOSITE else None
    gw, qw = words(ans), words(q["question"])
    alts = [(a, n) for a, n in pairs.get((arg, ans), {}).items()
            if not words(a) & (gw | qw) and stem(a) not in generic]
    hard = [a for a, _ in alts if a in scene]
    easy = [(a, n) for a, n in alts if a not in scene]
    picked = rng.sample(hard, 1) if hard else []
    while len(picked) < 3 and easy:
        easy = [(b, n) for b, n in easy if not any(words(b) & words(p) for p in picked)]  # no "cat"/"cats" pairs
        if not easy:
            break
        a = rng.choices([a for a, _ in easy], weights=[n for _, n in easy])[0]
        picked.append(a)
    if len(picked) < 3:
        return None
    return [ans] + picked, 0, False


def gqa(saver, cap, rng):
    cols = ["id", "imageId", "question", "answer", "types", "groups", "semanticStr"]
    t = pq.read_table(GQA / "train_balanced_instructions/train-00000-of-00001.parquet", columns=cols)
    types = t.column("types").combine_chunks()
    cols = {"id": t.column("id").to_pylist(), "image": t.column("imageId").to_pylist(),
            "question": t.column("question").to_pylist(), "answer": t.column("answer").to_pylist(),
            "structural": types.field("structural").to_pylist(), "semantic": types.field("semantic").to_pylist(),
            "detailed": types.field("detailed").to_pylist(),
            "group": t.column("groups").combine_chunks().field("global").to_pylist(),
            "prog": t.column("semanticStr").to_pylist()}
    del t
    pairs = defaultdict(Counter)
    scene = defaultdict(set)
    by_image = defaultdict(list)
    for i, prog in enumerate(cols["prog"]):
        img, ans, d = cols["image"][i], cols["answer"][i], cols["detailed"][i]
        scene[img].add(ans)
        scene[img].update(n for n in NAMES.findall(prog) if n != "_")
        op, arg = LAST_OP.match(prog.split("->")[-1]).group(1, 2)
        if d in GQA_PAIR_TYPES:
            alts = arg.split("|")
            if len(alts) == 2 and ans in alts:
                attr = op.partition(" ")[2] or "type"  # "choose color" -> color, "choose name" -> name
                other = alts[1 - alts.index(ans)]
                pairs[(attr, ans)][other] += 1
                pairs[(attr, other)][ans] += 1
        by_image[img].append((i, op, arg))
    generic = {stem(g) for g in set(cols["group"]) if g}
    files = sorted((GQA / "train_balanced_images").glob("*.parquet"))
    groups = [(f, g) for f in files for g in range(pq.ParquetFile(f).num_row_groups)]
    rng.shuffle(groups)
    counts, yesno = Counter(), Counter()
    made = 0
    for f, g in groups:
        pf = pq.ParquetFile(f)
        ids = pf.read_row_group(g, columns=["id"]).column("id").to_pylist()
        order = list(range(len(ids)))
        rng.shuffle(order)
        chosen = []
        for r in order:
            if len(chosen) == GQA_PER_GROUP:
                break
            qs = []
            for i, op, arg in by_image[ids[r]]:
                q = {k: cols[k][i] for k in ("id", "question", "answer", "structural", "semantic", "detailed")}
                q.update(op=op, arg=arg, bucket=gqa_bucket(q["structural"], q["semantic"], q["detailed"]))
                qs.append(q)
            total = sum(counts.values()) + 1
            buckets = sorted({q["bucket"] for q in qs}, key=lambda b: (counts[b] - GQA_SHARE[b] * total, b))
            for b in buckets:
                cand = [q for q in qs if q["bucket"] == b]
                rng.shuffle(cand)
                # yes/no answers: prefer whichever this bucket has fewer of
                cand.sort(key=lambda q: yesno[(b, q["answer"])] if q["answer"] in ("yes", "no") else -1)
                got = None
                for q in cand:
                    got = gqa_options(q, pairs, scene[ids[r]], generic, rng)
                    if got:
                        break
                if got:
                    counts[b] += 1
                    if q["answer"] in ("yes", "no"):
                        yesno[(b, q["answer"])] += 1
                    chosen.append((r, q, got))
                    break
        if not chosen:
            continue
        images = pf.read_row_group(g, columns=["image"]).column("image")
        for r, q, (options, gold, ordered) in chosen:
            path = saver.bytes(images[r].as_py()["bytes"])
            yield item("gqa", q["id"], [path], q["question"], options, gold, q["detailed"], "program", ordered=ordered,
                       gqa_type=f"{q['structural']}/{q['semantic']}")
            made += 1
        if made >= cap:
            return


# ---------------------------------------------------------------- ChartQA

COLORS = {"red", "blue", "green", "yellow", "orange", "purple", "pink", "brown", "black", "white", "gray", "grey",
          "violet", "navy", "maroon", "teal", "cyan", "magenta", "beige", "gold", "silver", "turquoise"}
GENDERS = ({"male", "men", "man", "boy", "masculine"}, {"female", "women", "woman", "girl", "feminine"})


def chartqa(saver, cap, rng):
    import hashlib

    rows = []
    for f in parquets(CHARTQA):
        pf = pq.ParquetFile(f)
        for g in range(pf.num_row_groups):
            t = pf.read_row_group(g, columns=["query", "label", "human_or_machine"]).to_pylist()
            keep = [k for k, r in enumerate(t) if r["human_or_machine"] == 0 and len(r["label"]) == 1]
            if not keep:
                continue
            images = pf.read_row_group(g, columns=["image"]).column("image")
            for k in keep:
                chart = hashlib.sha1(images[k]["bytes"].as_py()).hexdigest()
                rows.append({"f": f, "g": g, "k": k, "chart": chart, "q": t[k]["query"].strip(),
                             "a": t[k]["label"][0].strip()})
    by_chart = defaultdict(list)
    by_open = defaultdict(list)
    for r in rows:
        r["kind"] = "yesno" if r["a"].lower() in ("yes", "no") else "number" if NUM.match(r["a"]) else \
            "list" if r["a"].startswith("[") else "text"  # "[63, 26]": several values, skipped
        r["open"] = " ".join(r["q"].lower().split()[:2])
        by_chart[r["chart"]].append(r)
        if r["kind"] == "text":
            by_open[r["open"]].append(r["a"])
    texts = [r["a"] for r in rows if r["kind"] == "text"]
    rng.shuffle(rows)
    counts = Counter()
    yesno = Counter()
    plan = []
    for r in rows:
        a = r["a"]
        same = [o["a"] for o in by_chart[r["chart"]] if o is not r and o["kind"] != "list"]
        if r["kind"] == "list":
            continue
        if r["kind"] == "yesno":
            gold = 0 if a.lower() == "yes" else 1
            if yesno[gold] > yesno[1 - gold] + 20:  # keep Yes/No gold roughly balanced
                continue
            yesno[gold] += 1
            options, ordered = ["Yes", "No"], True
        elif r["kind"] == "number":
            got = ranked_numbers(a, same, rng, counts)
            if not got:
                continue
            (options, gold), ordered = got, True
        else:
            gw = re.sub(r"\W", "", a.lower())

            color = bool(words(a) & COLORS)
            gw_set = words(a) | next((g for g in GENDERS if words(a) & g), set())
            picked, used = [], {gw}
            # the same chart, then answers to questions opening the same way, then any answer of the same kind
            # (colour or not) and about the same length
            for tier, src in enumerate((same, by_open[r["open"]], texts)):
                src = src[:]
                rng.shuffle(src)
                for d in src[:2000]:
                    dw = re.sub(r"\W", "", d.lower())
                    if len(picked) < 3 and dw and dw not in used and dw not in gw and gw not in dw \
                            and not words(d) & gw_set and not NUM.match(d) and d.lower() not in ("yes", "no") \
                            and not d.startswith("[") and (tier == 0 or (bool(words(d) & COLORS) == color and
                                                                         abs(len(d.split()) - len(a.split())) <= 2)):
                        picked.append(d)
                        used.add(dw)
            if len(picked) < 3:
                continue
            options, gold, ordered = [a] + picked, 0, False
        if too_long(r["q"], options):
            continue
        plan.append((r, options, gold, ordered))
        if len(plan) >= cap:
            break
    tables = {}
    for r, options, gold, ordered in sorted(plan, key=lambda p: (str(p[0]["f"]), p[0]["g"], p[0]["k"])):
        key = (r["f"], r["g"])
        if key not in tables:
            tables.clear()
            tables[key] = pq.ParquetFile(r["f"]).read_row_group(r["g"], columns=["image"]).column("image")
        path = saver.bytes(tables[key][r["k"]]["bytes"].as_py())
        sid = f"{Path(r['f']).name.split('-')[1]}-{r['g']}-{r['k']}"
        yield item("chartqa", sid, [path], r["q"], options, gold, r["kind"], "human", ordered=ordered)


# ---------------------------------------------------------------- classification

PET_NAMES = {"german_shorthaired": "German Shorthaired Pointer", "wheaten_terrier": "Soft-coated Wheaten Terrier",
             "great_pyrenees": "Great Pyrenees", "english_cocker_spaniel": "English Cocker Spaniel"}
# Oxford 102 flowers, class ids 1..102 (the dataset's label names are these ids)
FLOWERS = [
    "pink primrose", "hard-leaved pocket orchid", "canterbury bells", "sweet pea", "english marigold", "tiger lily",
    "moon orchid", "bird of paradise", "monkshood", "globe thistle", "snapdragon", "colt's foot", "king protea",
    "spear thistle", "yellow iris", "globe-flower", "purple coneflower", "peruvian lily", "balloon flower",
    "giant white arum lily", "fire lily", "pincushion flower", "fritillary", "red ginger", "grape hyacinth",
    "corn poppy", "prince of wales feathers", "stemless gentian", "artichoke", "sweet william", "carnation",
    "garden phlox", "love in the mist", "mexican aster", "alpine sea holly", "ruby-lipped cattleya", "cape flower",
    "great masterwort", "siam tulip", "lenten rose", "barbeton daisy", "daffodil", "sword lily", "poinsettia",
    "bolero deep blue", "wallflower", "marigold", "buttercup", "oxeye daisy", "common dandelion", "petunia",
    "wild pansy", "primula", "sunflower", "pelargonium", "bishop of llandaff", "gaura", "geranium", "orange dahlia",
    "pink-yellow dahlia", "cautleya spicata", "japanese anemone", "black-eyed susan", "silverbush",
    "californian poppy", "osteospermum", "spring crocus", "bearded iris", "windflower", "tree poppy", "gazania",
    "azalea", "water lily", "rose", "thorn apple", "morning glory", "passion flower", "lotus", "toad lily",
    "anthurium", "frangipani", "clematis", "hibiscus", "columbine", "desert-rose", "tree mallow", "magnolia",
    "cyclamen", "watercress", "canna lily", "hippeastrum", "bee balm", "ball moss", "foxglove", "bougainvillea",
    "camellia", "mallow", "mexican petunia", "bromelia", "blanket flower", "trumpet creeper", "blackberry lily"]
CAR_MAKES = ("AM General", "Aston Martin", "Land Rover")
CAR_BODIES = ("Sedan", "SUV", "Coupe", "Convertible", "Hatchback", "Wagon", "Van", "Minivan", "Cab")


def label_names(f: Path, column: str = "label") -> list[str]:
    """Class names from the parquet's datasets metadata, else from the README's class_label block."""
    meta = pq.ParquetFile(f).schema_arrow.metadata or {}
    if b"huggingface" in meta:
        return json.loads(meta[b"huggingface"])["info"]["features"][column]["names"]
    readme = next(d / "README.md" for d in f.parents if (d / "README.md").exists())
    found = re.findall(r"^\s+'(\d+)': '?([^'\n]+?)'?$", readme.read_text(encoding="utf-8"), re.MULTILINE)
    return [v for _, v in sorted(((int(k), v) for k, v in found))]


def car_parts(name: str) -> tuple[str, str, str]:
    """(make, model without year, body type)."""
    make = next((m for m in CAR_MAKES if name.startswith(m + " ")), name.split()[0])
    model = re.sub(r"\s+\d{4}$", "", name)
    body = next((b for b in CAR_BODIES if f" {b}" in f" {model}"), "other")
    return make, model, body


def classify(saver, cap, rng, source, files, names, question, groups, subtask, extra_col=None):
    """Classes spread evenly; groups(label, extra) -> candidate pools tried in order for the 3 distractors."""
    rows = defaultdict(list)
    for f in files:
        pf = pq.ParquetFile(f)
        for g in range(pf.num_row_groups):
            cols = ["label"] + ([extra_col] if extra_col else [])
            t = pf.read_row_group(g, columns=cols).to_pydict()
            for k, lab in enumerate(t["label"]):
                rows[lab].append((f, g, k, t[extra_col][k] if extra_col else None))
    for v in rows.values():
        rng.shuffle(v)
    labels = sorted(rows)
    picks, depth = [], 0
    while len(picks) < cap and any(depth < len(rows[c]) for c in labels):
        layer = [rows[c][depth] + (c,) for c in labels if depth < len(rows[c])]
        rng.shuffle(layer)
        picks += layer[:cap - len(picks)]
        depth += 1
    plan = []
    for f, g, k, extra, lab in picks:
        picked = []
        for pool in groups(lab, extra):
            pool = [c for c in pool if c != lab and c not in picked and names[c] != names[lab]]
            rng.shuffle(pool)
            picked += pool[:3 - len(picked)]
        if len(picked) < 3:
            continue
        options = [names[lab]] + [names[c] for c in picked]
        plan.append((f, g, k, lab, extra, options))
    plan.sort(key=lambda p: (str(p[0]), p[1], p[2]))
    current, images = None, None
    for f, g, k, lab, extra, options in plan:
        if current != (f, g):
            current = (f, g)
            images = pq.ParquetFile(f).read_row_group(g, columns=["image"]).column("image")
        path = saver.bytes(images[k]["bytes"].as_py())
        sid = f"{Path(f).stem.split('-')[1]}-{g}-{k}"
        yield item(source, sid, [path], question(lab, extra), options, 0, subtask(lab, extra), "human",
                   label_name=names[lab])


def pets(saver, cap, rng):
    files = parquets(RAW / "timm__oxford-iiit-pet@089695c834a7/data")
    raw = label_names(files[0])
    names = [PET_NAMES.get(n, " ".join(w.capitalize() for w in n.split("_"))) for n in raw]
    species = {}
    for f in files:
        t = pq.read_table(f, columns=["label", "label_cat_dog"]).to_pydict()
        species.update(zip(t["label"], t["label_cat_dog"]))
    kind = {0: "cat", 1: "dog"}
    same = {s: [c for c, v in species.items() if v == s] for s in (0, 1)}
    yield from classify(saver, cap, rng, "pets", files, names,
                        lambda lab, sp: f"What breed is this {kind[sp]}?",
                        lambda lab, sp: [same[sp]], lambda lab, sp: kind[sp], extra_col="label_cat_dog")


def flowers(saver, cap, rng):
    files = parquets(RAW / "nkirschi__oxford-flowers@a37b1891609c/data")
    names = [FLOWERS[int(n) - 1] for n in label_names(files[0])]
    every = list(range(len(names)))
    yield from classify(saver, cap, rng, "flowers", files, names, lambda lab, _: "What kind of flower is this?",
                        lambda lab, _: [every], lambda lab, _: "flower")


def food101(saver, cap, rng):
    files = parquets(RAW / "ethz__food101@83488de741c1/data")
    names = [n.replace("_", " ") for n in label_names(files[0])]
    every = list(range(len(names)))
    yield from classify(saver, cap, rng, "food101", files, names, lambda lab, _: "What dish is shown in this image?",
                        lambda lab, _: [every], lambda lab, _: "dish")


def cars(saver, cap, rng):
    files = parquets(RAW / "tanganke__stanford_cars@9abf6cf7d6df/data")
    names = label_names(files[0])
    parts = [car_parts(n) for n in names]
    every = list(range(len(names)))

    def groups(lab, _):
        make, model, body = parts[lab]
        other = [c for c in every if parts[c][1] != model]  # never the same model in another year
        same_make = [c for c in other if parts[c][0] == make]
        rng.shuffle(same_make)
        return [same_make[:2], [c for c in other if parts[c][2] == body and parts[c][0] != make], other]

    yield from classify(saver, cap, rng, "cars", files, names, lambda lab, _: "Which car model is this?", groups,
                        lambda lab, _: parts[lab][2])


def places365(saver, cap, rng):
    files = parquets(RAW / "ljnlonoljpiljm__places365-256px@2bcb4953a4cd/data")
    names = [re.sub(r"/(.+)$", lambda m: f" ({m.group(1).replace('/', ', ')})", n).replace("_", " ")
             for n in label_names(files[0])]
    every = list(range(len(names)))
    yield from classify(saver, cap, rng, "places365", files, names, lambda lab, _: "What kind of place is this?",
                        lambda lab, _: [every], lambda lab, _: "indoor" if "indoor" in names[lab] else "scene")


# ---------------------------------------------------------------- geometry

LATEX_NUM = re.compile(r"\d+(?:\.\d+)?")
RADICAND = re.compile(r"\\sqrt\s*\{?\s*(\d+(?:\.\d+)?)\s*\}?")


def latex_value(s: str) -> float | None:
    """Numeric value of a simple LaTeX answer (\\sqrt, \\frac, \\pi, + - * /), or None."""
    e = s.replace(" ", "")
    for _ in range(4):
        e = re.sub(r"\\frac\{([^{}]*)\}\{([^{}]*)\}", r"((\1)/(\2))", e)
        e = re.sub(r"\\sqrt\{([^{}]*)\}", r"sqrt(\1)", e)
    e = re.sub(r"\\sqrt(\d+(?:\.\d+)?)", r"sqrt(\1)", e).replace("\\pi", "pi").replace("\\times", "*")
    e = re.sub(r"(\d|\))(sqrt|pi|\()", r"\1*\2", e)
    e = re.sub(r"(pi)(\d|sqrt|\()", r"\1*\2", e)
    try:
        return arith(ast.parse(e, mode="eval").body)
    except (SyntaxError, ValueError, ZeroDivisionError, OverflowError, RecursionError):
        return None


def arith(node) -> float:
    """Evaluates numbers, + - * /, sqrt(x) and pi from a parsed expression; anything else raises ValueError."""
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.Name) and node.id == "pi":
        return math.pi
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        v = arith(node.operand)
        return -v if isinstance(node.op, ast.USub) else v
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
        a, b = arith(node.left), arith(node.right)
        return {ast.Add: a + b, ast.Sub: a - b, ast.Mult: a * b}.get(type(node.op)) if not isinstance(
            node.op, ast.Div) else a / b
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "sqrt" \
            and len(node.args) == 1 and not node.keywords:
        return math.sqrt(arith(node.args[0]))
    raise ValueError("unsupported expression")


def braced(s: str, start: int) -> tuple[str, int] | None:
    """The {...} group opening at s[start] (spaces skipped) and the index after it."""
    while start < len(s) and s[start] == " ":
        start += 1
    if start >= len(s) or s[start] != "{":
        return None
    depth = 0
    for k in range(start, len(s)):
        depth += {"{": 1, "}": -1}.get(s[k], 0)
        if depth == 0:
            return s[start + 1:k], k + 1
    return None


def simplified(s: str) -> bool:
    """No square factor under a root, no root of 0 or 1, no fraction whose coefficient and denominator share a
    factor or whose denominator is 1."""
    if re.search(r"(?<![\d.])1\s*\\(sqrt|pi)", s):  # "1 \sqrt{6}"
        return False
    for x in RADICAND.findall(s):
        r = float(x)
        if r != int(r):
            continue
        r = int(r)
        if r < 2 or any(r % (k * k) == 0 for k in range(2, math.isqrt(r) + 1)):
            return False
    for m in re.finditer(r"\\frac", s):
        num = braced(s, m.end())
        den = num and braced(s, num[1])
        if not den:
            continue
        lead = re.match(r"\s*(\d+)", num[0])
        coef = int(lead.group(1)) if lead else 1
        if not re.fullmatch(r"\s*\d+\s*", den[0]):
            continue
        d = int(den[0])
        if d <= 1 or math.gcd(coef, d) > 1:
            return False
    return True


def latex_variants(gold: str, rng: random.Random) -> list[str] | None:
    """3 distractors made by changing one or two numbers of a LaTeX answer; values must differ from the gold."""
    g = latex_value(gold)
    spans = [m.span() for m in LATEX_NUM.finditer(gold)]
    if g is None or not spans:
        return None
    out, values = [], [g]
    for _ in range(40):
        s = gold
        for a, b in sorted(rng.sample(spans, rng.choice([1, min(2, len(spans))])), reverse=True):
            n = float(s[a:b])
            steps = [n + 1, n + 2, n - 1, n * 2, n + 3, n + 4, n * 3] if n > 1 else [n + 1, n + 2, n + 3]
            m = rng.choice(steps)
            s = s[:a] + (str(int(m)) if m == int(m) else f"{m:g}") + s[b:]
        v = latex_value(s)
        if not simplified(s):
            continue  # an unsimplified \sqrt{8} or \frac{6}{2} would give the simplified gold away
        if v is not None and v > 0 and all(abs(v - w) > 1e-6 for w in values) and s not in out:
            out.append(s)
            values.append(v)
        if len(out) == 3:
            return out
    return None


def option_key(s: str) -> str:
    return re.sub(r"\s+", "", s)


def intergps_choices() -> dict[str, list[list[str]]]:
    """Cauldron InterGPS (Geometry3K with its original choices): image dhash -> choice lists."""
    from qwen_decision.v3.vision_heldout import dhash

    out = defaultdict(list)
    for f in parquets(INTERGPS, "*.parquet"):
        for r in pq.read_table(f).to_pylist():
            h = dhash(r["images"][0]["bytes"])
            for turn in r["texts"]:
                m = re.search(r"\nChoices:\n(.*?)\nAnswer with", turn["user"], re.DOTALL)
                if m:
                    out[h].append([re.sub(r"^[A-D]\.\s*", "", c).strip() for c in m.group(1).split("\n")])
    return out


def geometry3k(saver, cap, rng):
    from qwen_decision.v3.vision_heldout import dhash

    orig = intergps_choices()
    rows = pq.read_table(GEO3K).to_pylist()
    order = list(range(len(rows)))
    rng.shuffle(order)
    counts = Counter()
    for i in order:
        r = rows[i]
        data = r["images"][0]["bytes"]
        gold = r["answer"].strip()
        question = r["problem"].replace("<image>", "").strip()
        lists = {tuple(c) for c in orig.get(dhash(data), []) if option_key(gold) in map(option_key, c)}
        if len(lists) == 1:
            options = list(lists.pop())
            g = [option_key(o) for o in options].index(option_key(gold))
            values = [num_value(o) for o in options]
            if all(v is not None for v in values):  # numbers ascending, like the board's numeric choices
                ranked = sorted(range(len(options)), key=lambda k: values[k])
                options, g = [options[k] for k in ranked], ranked.index(g)
            got, sub = (options, g, all(v is not None for v in values)), "original_choices"
        elif NUM.match(gold):
            nums = ranked_numbers(gold, [], rng, counts)
            got, sub = (nums + (True,) if nums else None), "numeric"
        else:
            ds = latex_variants(gold, rng)
            got, sub = (([gold] + ds, 0, False) if ds else None), "latex"
        if not got or too_long(question, got[0]):
            continue
        yield item("geometry3k", str(i), [saver.bytes(data)], question, got[0], got[1], sub, "human", ordered=got[2])


GEOQA_UNITS = re.compile(r"(cm|m|°|度|米|厘米|\s)+$")


def geoqa_plus(saver, cap, rng):
    rows = pq.read_table(GEOQA).to_pylist()
    order = list(range(len(rows)))
    rng.shuffle(order)
    for i in order:
        r = rows[i]
        m = re.match(r"(.*?)\nChoices:\n(.*)$", r["problem"].replace("<image>", ""), re.DOTALL)
        if not m:
            continue
        question = m.group(1).strip()
        parts = re.findall(r"^([A-D])\.\s*(.*)$", m.group(2), re.MULTILINE)
        if [p[0] for p in parts] != list("ABCD"[:len(parts)]) or len(parts) < 2:
            continue
        options = [p[1].strip() for p in parts]
        hit = re.findall(r"选[:：择]?\s*([A-D])", r["answer"])
        if hit:
            gold, sub = "ABCD".index(hit[-1]), "letter"
        else:  # the solution's final value, when exactly one choice equals it
            last = re.split(r"[=＝]", r["answer"].rstrip("．。.,， "))[-1].strip()
            same = [k for k, o in enumerate(options) if GEOQA_UNITS.sub("", o) == GEOQA_UNITS.sub("", last) != ""]
            if len(same) != 1:
                continue
            gold, sub = same[0], "final_value"
        if gold >= len(options) or too_long(question, options):
            continue
        img = r["images"][0]["bytes"]
        yield item("geoqa_plus", str(i), [saver.bytes(img)], question, options, gold, sub, "human")


# ---------------------------------------------------------------- COCO captions

NOUNS = {
    "man", "woman", "boy", "girl", "child", "kid", "person", "dog", "cat", "horse", "cow",
    "sheep", "elephant", "bear", "zebra", "giraffe", "bird", "duck", "pigeon", "table", "chair", "couch", "sofa",
    "bed", "desk", "bench", "car", "truck", "bus", "train", "bike", "bicycle", "motorcycle", "boat", "plane",
    "airplane", "kite", "umbrella", "frisbee", "ball", "skateboard", "surfboard", "board", "laptop", "computer",
    "phone", "television", "tv", "keyboard", "book", "clock", "vase", "bottle", "cup", "glass", "plate", "bowl",
    "pizza", "sandwich", "cake", "donut", "banana", "apple", "orange", "broccoli", "carrot", "sink", "toilet",
    "oven", "microwave", "refrigerator", "fridge", "window", "door", "wall", "building", "tree", "fence", "street",
    "road", "sign", "pole", "hydrant", "field", "beach", "ocean", "water", "wave", "snow", "grass", "mountain",
    "hill", "river", "lake", "player", "tennis", "racket", "bat", "baseball", "skier", "surfer", "bag", "backpack",
    "suitcase", "hat", "shirt", "jacket", "teddy", "tie", "wine", "fork", "knife", "spoon", "counter", "shelf",
    "cabinet", "lamp", "mirror", "pillow", "blanket", "rug", "floor", "sidewalk", "tower", "bridge", "sky", "cloud",
    "flower", "plant", "box", "basket", "tray"}
ADJECTIVE_KIND = {
    **dict.fromkeys(["red", "blue", "green", "yellow", "white", "black", "brown", "orange", "pink", "purple", "gray",
                     "grey", "silver"], "color"),
    **dict.fromkeys(["large", "small", "big", "little", "tall", "short"], "size"),
    **dict.fromkeys(["young", "old"], "age"), **dict.fromkeys(["wooden", "metal", "plastic"], "material"),
    **dict.fromkeys(["empty", "full"], "fill"), **dict.fromkeys(["dark", "wet"], "state")}
ADJECTIVES = set(ADJECTIVE_KIND)
MODIFIERS = {"fire", "stop", "picnic", "cell", "traffic", "parking", "police", "coffee", "surf", "dining", "living",
             "passenger", "soccer", "rubber", "toy", "stuffed"}  # "fire hydrant": not a noun to swap alone
SYMMETRIC = {"and", "or", "next to", "beside", "near", "with", "by", "and a", "and the", "and an", ",", "&"}
ANIMATE = {"man", "woman", "boy", "girl", "child", "kid", "person", "dog", "cat", "horse", "cow", "sheep", "elephant",
           "bear", "zebra", "giraffe", "bird", "duck", "pigeon", "player", "skier", "surfer"}
ACTIVITIES = {"skiing", "surfing", "boarding", "skating", "sailing", "riding", "fishing"}
DETERMINERS = {"a", "an", "the", "two", "three", "some", "his", "her", "their", "its", "one"}
STOP = {
    "a", "an", "the", "of", "on", "in", "at", "with", "and", "is", "are", "to", "its", "his", "her", "their", "two",
    "three", "some", "near", "next", "by", "for", "from"}


def fix_articles(tokens: list[str]) -> list[str]:
    out = tokens[:]
    for k in range(len(out) - 1):
        if out[k].lower() in ("a", "an") and re.match(r"[A-Za-z]", out[k + 1]):
            want = "an" if out[k + 1][0].lower() in "aeiou" else "a"
            out[k] = want if out[k].islower() else want.capitalize()
    return out


def tidy(caption: str) -> str:
    """One surface form for every option (no trailing period, first letter upper), so form gives nothing away."""
    c = re.sub(r"\s+", " ", caption.strip()).rstrip(" .")
    return c[:1].upper() + c[1:]


def caption_variants(caption: str, rng: random.Random, avoid: set[str]) -> list[tuple[str, str]]:
    """Meaning-changing swaps of a tidied caption: two nouns, or two adjectives bound to different nouns."""
    tokens = re.findall(r"[A-Za-z']+|[^\sA-Za-z']", caption)
    low = [t.lower() for t in tokens]
    out = []
    nouns = [k for k, w in enumerate(low) if w in NOUNS and (k + 1 == len(low) or low[k + 1] not in NOUNS)
             and (k == 0 or (low[k - 1] not in NOUNS and low[k - 1] not in MODIFIERS))
             and (k + 1 == len(low) or low[k + 1] not in ACTIVITIES)]
    for a in range(len(nouns)):
        for b in range(a + 1, len(nouns)):
            i, j = nouns[a], nouns[b]
            if low[i] == low[j] or stem(low[i]) == stem(low[j]) or low[i].endswith("s") != low[j].endswith("s") \
                    or (low[i] in ANIMATE) != (low[j] in ANIMATE):  # "a shirt wearing a man" is too easy to reject
                continue
            between = " ".join(w for w in low[i + 1:j] if w not in DETERMINERS and w not in ADJECTIVES)
            if between in SYMMETRIC or not between:
                continue
            t = tokens[:]
            t[i], t[j] = tokens[j].lower(), tokens[i].lower()
            out.append(("noun_swap", t))
    adjs = [k for k, w in enumerate(low[:-1]) if w in ADJECTIVES and re.fullmatch(r"[a-z]+", low[k + 1])
            and low[k + 1] not in ADJECTIVES and low[k + 1] not in STOP]
    for a in range(len(adjs)):
        for b in range(a + 1, len(adjs)):
            i, j = adjs[a], adjs[b]
            if low[i] != low[j] and low[i + 1] != low[j + 1] and ADJECTIVE_KIND[low[i]] == ADJECTIVE_KIND[low[j]]:
                t = tokens[:]
                t[i], t[j] = tokens[j].lower(), tokens[i].lower()
                out.append(("adjective_swap", t))
    rng.shuffle(out)
    out.sort(key=lambda kt: kt[0] != "adjective_swap")  # the rarer binding swaps first
    res, seen = [], {a.lower() for a in avoid} | {caption.lower()}
    for kind, t in out:
        s = tidy(re.sub(r"\s+([,;:!?'])", r"\1", " ".join(fix_articles(t))))
        if s.lower() not in seen:
            seen.add(s.lower())
            res.append((kind, s))
    return res


def content(caption: str) -> set[str]:
    return {w for w in words(caption) if w not in STOP}


def coco_karpathy(saver, cap, rng):
    rows = []
    for split in ("train", "restval"):
        for f in parquets(COCO, f"{split}-*.parquet"):
            rows += pq.read_table(f, columns=["filepath", "filename", "sentences", "cocoid", "split"]).to_pylist()
    rows = [r for r in rows if r["split"] in ("train", "restval")]
    for r in rows:
        r["sentences"] = [s.strip() for s in r["sentences"] if s.strip()]
    zips = {k: zipfile.ZipFile(p) for k, p in COCO_ZIPS.items()}
    order = list(range(len(rows)))
    rng.shuffle(order)
    made = 0
    for n, i in enumerate(order):
        r = rows[i]
        caps = [tidy(c) for c in r["sentences"]]
        rng.shuffle(caps)
        for true in caps:
            variants = caption_variants(true, rng, set(caps))
            if variants:
                break
        else:
            continue
        mine = set().union(*(content(c) for c in r["sentences"]))
        other = None
        for _ in range(50):
            o = rows[order[rng.randrange(len(order))]]
            if o is r or not o["sentences"]:
                continue
            c = rng.choice(o["sentences"])
            if len(content(c) & mine) <= 1:
                other = tidy(c)
                break
        if other is None:
            continue
        picked = variants[:rng.choice([1, 2])]
        options = [true] + [v for _, v in picked] + [other]
        name = f"{r['filepath']}/{r['filename']}"
        try:
            data = zips[r["filepath"]].read(name)
        except KeyError:
            continue
        sub = "+".join(sorted({k for k, _ in picked}))
        yield item("coco_karpathy", str(r["cocoid"]), [saver.bytes(data)], "Which caption describes this image?",
                   options, 0, sub, "program")
        made += 1
        if made >= cap:
            return


# ---------------------------------------------------------------- ImageCoDe


# a description pointing at a position in the full 10-image set no longer fits the 4 images shown
ORDINAL = re.compile(r"\b(first|last|second|third|fourth|previous|next|final) (image|photo|picture|frame|one|pic)\b",
                     re.IGNORECASE)


def imagecode(saver, cap, rng):
    rows = [json.loads(line) for line in (IMAGECODE / "train_data.jsonl").read_text(encoding="utf-8").splitlines()]
    z = zipfile.ZipFile(IMAGECODE / "image-sets.zip")
    members = defaultdict(list)
    for info in z.infolist():
        parts = info.filename.split("/")
        if len(parts) == 3 and parts[2].endswith(".jpg"):
            members[parts[1]].append(info.filename)
    for v in members.values():  # ImageCoDe's image_index counts frames by the number in imgN.jpg
        v.sort(key=lambda p: int(re.search(r"img(\d+)\.jpg$", p).group(1)))
    order = list(range(len(rows)))
    rng.shuffle(order)
    slots = Counter()
    made = 0
    for i in order:
        r = rows[i]
        frames = members.get(r["image_set"], [])
        g = int(r["image_index"])
        description = r["description"].strip()
        if len(frames) != 10 or not 0 <= g < 10 or not description or ORDINAL.search(description):
            continue
        video = not r["image_set"].startswith("open-images")
        # video sets: the gold and its nearest frames, in time order, with the gold at the least used feasible slot;
        # Open Images sets are retrieved neighbours with no order, so 3 random others
        feasible = range(max(0, g - 6), min(3, g) + 1) if video else range(4)
        lo = min(slots[s] for s in feasible)
        pos = rng.choice([s for s in feasible if slots[s] == lo])
        slots[pos] += 1
        if video:
            shown = list(range(g - pos, g + 4 - pos))
        else:
            shown = rng.sample([k for k in range(10) if k != g], 3)
            shown.insert(pos, g)
        question = ("Four similar images are shown in order: Image 1, Image 2, Image 3 and Image 4. "
                    f"Which image matches this description: \"{description}\"?")
        options = [f"Image {k + 1}" for k in range(4)]
        if too_long(question, options):
            continue
        images = [saver.bytes(z.read(frames[k])) for k in shown]
        yield item("imagecode", f"{r['image_set']}:{g}", images, question, options, pos,
                   "video" if video else "open-images", "human", ordered=True)
        made += 1
        if made >= cap:
            return


CONVERTERS = {
    "gqa": ("spatial", gqa),
    "chartqa": ("chart", chartqa),
    "pets": ("classification", pets),
    "flowers": ("classification", flowers),
    "food101": ("classification", food101),
    "cars": ("classification", cars),
    "places365": ("classification", places365),
    "geometry3k": ("math", geometry3k),
    "geoqa_plus": ("math", geoqa_plus),
    "coco_karpathy": ("image-text", coco_karpathy),
    "imagecode": ("image-text", imagecode),
}

