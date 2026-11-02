"""Image training mix (2026-11-01): every image training source turned into the board's multiple-choice form,
deduplicated against the evaluation images, then sampled with text decision rows into training files.

  PYTHONPATH=.:external/decision-index .venv/bin/python -m qwen_decision.v3.vision_mix build [--only NAME ...]
  PYTHONPATH=.:external/decision-index .venv/bin/python -m qwen_decision.v3.vision_mix dedupe [--only NAME ...]
  PYTHONPATH=.:external/decision-index .venv/bin/python -m qwen_decision.v3.vision_mix mix --rows 16384 --name screen-16k [--clean] [--balance skill|source]
  PYTHONPATH=.:external/decision-index .venv/bin/python -m qwen_decision.v3.vision_mix list

An item is the kit's item shape (decision_index.vision.convert.mc_item) plus our fields:
  id        "<source>:<id in the source>", unique
  source    the converter name (a vision_sources name, or "cauldron/<subset>")
  subtask   short label, for reports
  images    1..MAX_IMAGES paths relative to data/vision-mix
  question  all of the row's text; the state stays empty, as on the board
  options   2..MAX_OPTIONS distinct strings
  gold      index of the correct option
  ordered   True when the options must keep their order (yes/no, numbers ascending, letters drawn in the image)
  label     human | program (who made the answer); never a model
Training renders an item as VisionEngine's decide prompt does: the images as the state, the options under the kit's
letters (decision_index.vision.rows.request), shuffled first unless ordered.

Converters live in vision_convert_*.py modules; each has CONVERTERS = {name: (skill, fn)} with
fn(saver: Saver, cap: int, rng: random.Random) -> iterator of items (None entries are skipped). A converter draws
a seeded sample spread over its whole source, at most `cap` items. Pools land in data/vision-mix/pool/<name>.jsonl.
`dedupe` drops every item with an image whose sha256 matches, or whose 64-bit difference hash is within
DHASH_DISTANCE bits of, a board vision image (data/vision-work) or a held-out test image (data/vision-heldout).
`mix` samples equal shares per skill, then per source within a skill (or, with --balance source, equal shares per
source), and adds text rows from v3-gold6.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import io
import json
import random
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MIX = ROOT / "data/vision-mix"
IMG = MIX / "img"
POOL = MIX / "pool"
BOARD = ROOT / "data/vision-work/vision/full"
HELDOUT = ROOT / "data/vision-heldout"
TEXT = ROOT / "data/mix/v3-gold6.jsonl"
MODULES = ("vision_convert_cauldron", "vision_convert_spatial", "vision_convert_screen", "vision_convert_misc",
           "vision_convert_diverse")
SEED = 20261101
MAX_OPTIONS = 10   # the board's most (MMMU-Pro); the kit letters stop at J for items
MAX_IMAGES = 10    # ImageCoDe's sets; the board has at most 2
DEFAULT_CAP = 4000
DHASH_DISTANCE = 2
DEV_PER_SOURCE = 64
# mix shares are equal per skill, then per source within a skill
SKILLS = ("spatial", "counting", "depth", "correspondence", "multi-image", "general", "chart", "document", "kie",
          "web", "image-text", "meme", "science", "math", "classification",
          "remote-sensing", "medical", "reading", "3d-pose", "geolocation")  # the last five added 11-02
BOARD_TRAIN = ("cauldron/infographic_vqa", "cauldron/hateful_memes", "mind2web", "cord", "funsd")
# built but left out of every training file
LEFT_OUT = {"hpatches": "BLINK Visual_Correspondence is built from HPatches; the sequences not matching a public "
                        "board image may still be in BLINK's test split (the board's private half)"}
# sources an evaluated set is built from: its own train split, or the dataset its items, annotations or scenes come
# from. `mix --clean` leaves them out too. Sources that only share an image pool (COCO or VG photos under other
# annotations) stay, since the dedupe already drops their images that match an evaluation image.
EVAL_PARENTS = {
    "cauldron/infographic_vqa": "board InfographicVQA (its train split)",
    "cauldron/hateful_memes": "board Hateful Memes (its train split)",
    "mind2web": "board Mind2Web (its train split)",
    "cord": "board KIE (CORD train split)",
    "funsd": "board KIE (FUNSD train split)",
    "spair71k": "board BLINK Semantic_Correspondence is built from SPair-71k",
    "cauldron/vsr": "board BLINK Spatial_Relation is built from VSR",
    "cauldron/tallyqa": "board BLINK Counting is built from TallyQA",
    "hypersim": "board CV-Bench Distance comes from Omni3D, which includes Hypersim",
    "nyu_depth": "board CV-Bench Distance comes from Omni3D, whose SUN RGB-D part includes NYU Depth v2 scenes",
    "coco_karpathy": "held-out SugarCrepe is built from COCO captions",
    "cauldron/cocoqa": "COCO-QA is generated from COCO captions, held-out SugarCrepe's parent",
    "vqa_rad": "held-out MathVista testmini keeps VQA-RAD items (2 images matched by sha256, 11-02)",
}
EXT = {"JPEG": "jpg", "MPO": "jpg", "PNG": "png", "WEBP": "webp", "GIF": "gif", "BMP": "bmp", "TIFF": "tif"}


def dumps(x) -> str:
    return json.dumps(x, ensure_ascii=False, separators=(",", ":"))


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


class Saver:
    """Writes a source's images under data/vision-mix/img/<source>/, named by content hash (shared images once).

    Images over the kit's 1.6 MP cap are stored capped exactly as engines load them (images.capped); smaller ones
    keep their bytes. Drawn images (marks, points) go through `pil` and are stored as JPEG quality 92 (the kit's
    Mind2Web marks) unless png=True.
    """

    def __init__(self, source: str):
        self.source = source
        self.dir = IMG / source.replace("/", "_")
        self.dir.mkdir(parents=True, exist_ok=True)

    def _write(self, raw: bytes, ext: str) -> str:
        path = self.dir / f"{hashlib.sha256(raw).hexdigest()[:24]}.{ext}"
        if not path.exists():
            tmp = path.with_suffix(".part")
            tmp.write_bytes(raw)
            tmp.rename(path)
        return str(path.relative_to(MIX))

    def bytes(self, data: bytes) -> str:
        from decision_index.vision import images as img
        from PIL import Image

        raw = img.capped(bytes(data), None, img.MAX_PIXELS)
        with Image.open(io.BytesIO(raw)) as im:
            fmt = im.format
            if fmt not in ("JPEG", "PNG", "WEBP", "GIF", "MPO"):  # BMP, TIFF, ...: store as PNG
                buf = io.BytesIO()
                im.convert("RGBA" if "A" in im.mode else "RGB").save(buf, format="PNG")
                raw, fmt = buf.getvalue(), "PNG"
        return self._write(raw, EXT[fmt])

    def pil(self, im, png: bool = False) -> str:
        buf = io.BytesIO()
        if png:
            im.save(buf, format="PNG")
        else:
            im.convert("RGB").save(buf, format="JPEG", quality=92)
        return self.bytes(buf.getvalue())

    def file(self, path: Path | str) -> str:
        return self.bytes(Path(path).read_bytes())


def item(source: str, sid: str, images: list[str], question: str, options: list, gold: int, subtask: str,
         label: str, ordered: bool = False, **extra) -> dict | None:
    """A validated item, or None when it cannot be a fair single-answer question (duplicate options and so on)."""
    options = [str(o).strip() for o in options]
    question = question.strip()
    if not question or not 1 <= len(images) <= MAX_IMAGES or not 2 <= len(options) <= MAX_OPTIONS:
        return None
    if not 0 <= gold < len(options) or any(not o for o in options):
        return None
    if len({norm(o) for o in options}) != len(options):
        return None
    if label not in ("human", "program"):
        raise ValueError(f"label must be human or program, not {label!r}")
    return {"id": f"{source}:{sid}", "source": source, "subtask": subtask, "images": images, "question": question,
            "options": options, "gold": gold, "ordered": ordered, "label": label, **extra}


def converters() -> dict:
    out = {}
    for m in MODULES:
        try:
            module = importlib.import_module(f"qwen_decision.v3.{m}")
        except ModuleNotFoundError as e:
            if e.name != f"qwen_decision.v3.{m}":
                raise
            continue
        for name, entry in module.CONVERTERS.items():
            if name in out:
                raise ValueError(f"converter {name} defined twice")
            out[name] = entry
    return out


def cap_for(name: str) -> int:
    """Pool size per converter: DEFAULT_CAP, less for each Cauldron subset (36 of them) and classification set."""
    if name in BOARD_TRAIN:
        return DEFAULT_CAP
    if name.startswith("cauldron/"):
        return 2000
    if name == "android_next":  # four phone screens an item, slow to decode; a source's mix share is ~530 rows
        return 600
    small = ("pets", "flowers", "food101", "cars", "places365")
    return 1500 if name in small or importlib.import_module("qwen_decision.v3.vision_convert_diverse").CONVERTERS.get(
        name) else DEFAULT_CAP


def build(only: list[str] | None) -> None:
    POOL.mkdir(parents=True, exist_ok=True)
    table = converters()
    for name, (skill, fn) in table.items():
        if only and name not in only:
            continue
        if skill not in SKILLS:
            raise ValueError(f"{name}: unknown skill {skill!r}")
        cap = cap_for(name)
        started = time.time()
        rng = random.Random(f"{SEED}:{name}")
        items, seen = [], set()
        for it in fn(Saver(name), cap, rng):
            if it is None:
                continue
            if it["id"] in seen:
                raise ValueError(f"{name}: duplicate id {it['id']}")
            seen.add(it["id"])
            items.append(it | {"skill": skill})
            if len(items) >= cap:
                break
        path = POOL / f"{name.replace('/', '_')}.jsonl"
        path.write_text("".join(dumps(it) + "\n" for it in items), encoding="utf-8")
        meta = {"name": name, "skill": skill, "items": len(items), "cap": cap,
                "images": len({p for it in items for p in it["images"]}),
                "subtasks": dict(Counter(it["subtask"] for it in items).most_common()),
                "options": dict(sorted(Counter(len(it["options"]) for it in items).items())),
                "gold_position": dict(sorted(Counter(it["gold"] for it in items).items())),
                "labels": dict(Counter(it["label"] for it in items)), "seconds": round(time.time() - started, 1)}
        path.with_suffix(".meta.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False) + "\n")
        print(json.dumps({k: meta[k] for k in ("name", "items", "images", "seconds")}), flush=True)


def dhash(data: bytes) -> int:
    from qwen_decision.v3.vision_heldout import (
        dhash as kit_dhash,  # the held-out index uses this hash
    )

    return int(kit_dhash(data), 16)


def _hash_file(path: str):
    data = Path(path).read_bytes()
    return path, hashlib.sha256(data).hexdigest(), dhash(data)


def hash_files(paths: list[str]) -> dict:
    from concurrent.futures import ProcessPoolExecutor

    with ProcessPoolExecutor(12) as ex:
        return {p: (s, d) for p, s, d in ex.map(_hash_file, paths, chunksize=256)}


def evaluation_hashes() -> tuple[set[str], list[int]]:
    """sha256 and dhash of every board vision image and every held-out test image."""
    cache = MIX / "eval-image-hashes.json"
    if cache.exists():
        got = json.loads(cache.read_text())
        return set(got["sha256"]), got["dhash"]
    board = [str(p) for p in sorted(BOARD.glob("*/img/*")) if p.is_file()]
    hashed = hash_files(board)
    shas = {s for s, _ in hashed.values()}
    hashes = {d for _, d in hashed.values()}
    for v in json.loads((HELDOUT / "images.json").read_text()).values():
        shas.add(v["sha256"])
        hashes.add(int(v["dhash"], 16))
    cache.write_text(json.dumps({"board_images": len(board), "sha256": sorted(shas), "dhash": sorted(hashes)}) + "\n")
    return shas, sorted(hashes)


def near(train_hashes: list[int], eval_hashes: list[int], distance: int) -> set[int]:
    """Indices of train_hashes within `distance` bits of any eval hash."""
    import numpy as np

    table = np.array([i.bit_count() for i in range(256)], dtype=np.uint8)
    ev = np.array(eval_hashes, dtype=np.uint64)
    hits = set()
    for start in range(0, len(train_hashes), 512):
        chunk = np.array(train_hashes[start:start + 512], dtype=np.uint64)
        x = (chunk[:, None] ^ ev[None, :]).view(np.uint8).reshape(len(chunk), len(ev), 8)
        bits = table[x].sum(-1, dtype=np.uint16)
        hits.update((np.nonzero((bits <= distance).any(1))[0] + start).tolist())
    return hits


def dedupe(only: list[str] | None = None) -> None:
    shas, hashes = evaluation_hashes()
    old = MIX / "dedupe.json"
    report = json.loads(old.read_text()) if only and old.exists() else {}
    for path in sorted(POOL.glob("*.jsonl")):
        if path.name.endswith(".dedup.jsonl"):
            continue
        if only and path.stem not in {n.replace("/", "_") for n in only}:
            continue
        items = [json.loads(line) for line in path.read_text(encoding="utf-8").split("\n") if line]
        files = sorted({p for it in items for p in it["images"]})
        hashed = hash_files([str(MIX / p) for p in files])
        order = list(hashed)
        close = {order[i] for i in near([hashed[p][1] for p in order], hashes, DHASH_DISTANCE)}
        bad = {p for p, (s, _) in hashed.items() if s in shas} | close
        kept = [it for it in items if not any(str(MIX / p) in bad for p in it["images"])]
        path.with_suffix(".dedup.jsonl").write_text("".join(dumps(it) + "\n" for it in kept), encoding="utf-8")
        report[path.stem] = {"items": len(items), "kept": len(kept), "images": len(files),
                             "sha256_hits": sum(s in shas for s, _ in hashed.values()), "dhash_hits": len(close)}
        print(json.dumps({path.stem: report[path.stem]}), flush=True)
    (MIX / "dedupe.json").write_text(json.dumps(report, indent=1) + "\n")


def pools() -> dict[str, list[dict]]:
    out = {}
    for path in sorted(POOL.glob("*.dedup.jsonl")):
        items = [json.loads(line) for line in path.read_text(encoding="utf-8").split("\n") if line]
        if items and items[0]["source"] not in LEFT_OUT:
            out[items[0]["source"]] = items
    return out


def split_dev(items: list[dict]) -> tuple[list[dict], list[dict]]:
    """A fixed DEV_PER_SOURCE items per source (by id hash) held out of every training file."""
    ranked = sorted(items, key=lambda it: hashlib.sha256(f"dev:{it['id']}".encode()).hexdigest())
    return ranked[DEV_PER_SOURCE:], ranked[:DEV_PER_SOURCE]


def shares(sizes: dict[str, int], total: int) -> dict[str, int]:
    """Equal shares capped by size; what a small group cannot use goes to the others (water filling)."""
    out, left, open_ = {k: 0 for k in sizes}, total, set(sizes)
    while left > 0 and open_:
        each = max(1, left // len(open_))
        for k in sorted(open_):
            take = min(each, sizes[k] - out[k], left)
            out[k] += take
            left -= take
            if out[k] == sizes[k]:
                open_.discard(k)
            if left == 0:
                break
    return out


def text_rows(n: int, rng: random.Random) -> list[dict]:
    """n rows drawn uniformly from v3-gold6 (the base's own training data), reservoir-sampled in one pass."""
    reservoir = []
    with TEXT.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            if len(reservoir) < n:
                reservoir.append(line)
            else:
                j = rng.randrange(i + 1)
                if j < n:
                    reservoir[j] = line
    return [json.loads(line) for line in reservoir]


def mix(rows: int, text_share: float, name: str, seed: int, clean: bool = False, balance: str = "skill") -> None:
    """balance: "skill" = equal shares per skill, then per source; "source" = equal shares per source (11-02)."""
    rng = random.Random(f"{SEED}:mix:{seed}")
    by_source = pools()
    train_pool, dev = {}, []
    for source, items in by_source.items():
        train, d = split_dev(items)
        dev += d  # every source, so dev.jsonl stays the same file with or without --clean
        if not (clean and source in EVAL_PARENTS):
            train_pool[source] = train
    skills = defaultdict(list)
    for source, items in train_pool.items():
        skills[items[0]["skill"]].append(source)
    n_text = round(rows * text_share)
    n_image = rows - n_text
    if balance == "source":
        flat = shares({s: len(v) for s, v in train_pool.items()}, n_image)
        per_skill = {k: sum(flat[s] for s in v) for k, v in skills.items()}
    else:
        per_skill = shares({k: sum(len(train_pool[s]) for s in v) for k, v in skills.items()}, n_image)
    chosen, plan = [], {}
    for skill, sources in sorted(skills.items()):
        per_source = ({s: flat[s] for s in sources} if balance == "source"
                      else shares({s: len(train_pool[s]) for s in sources}, per_skill[skill]))
        for s, k in per_source.items():
            chosen += [{"modality": "image", **it} for it in rng.sample(train_pool[s], k)]
            plan[s] = {"skill": skill, "rows": k, "pool": len(train_pool[s])}
    chosen += [{"modality": "text", **r} for r in text_rows(n_text, rng)]
    rng.shuffle(chosen)
    MIX.mkdir(parents=True, exist_ok=True)
    (MIX / f"{name}.jsonl").write_text("".join(dumps(r) + "\n" for r in chosen), encoding="utf-8")
    (MIX / "dev.jsonl").write_text("".join(dumps({"modality": "image", **it}) + "\n" for it in dev), encoding="utf-8")
    report = {"rows": len(chosen), "image_rows": n_image, "text_rows": n_text, "text_share": text_share, "seed": seed,
              "clean": clean, "balance": balance, "left_out": sorted(EVAL_PARENTS) if clean else [],
              "skills": {k: per_skill[k] for k in sorted(per_skill)}, "sources": plan, "dev_rows": len(dev)}
    (MIX / f"{name}.report.json").write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps({k: report[k] for k in ("rows", "image_rows", "text_rows", "skills", "dev_rows")}, indent=1))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["build", "dedupe", "mix", "list"])
    p.add_argument("--only", nargs="*")
    p.add_argument("--rows", type=int, default=16384)
    p.add_argument("--text-share", type=float, default=1 / 3)
    p.add_argument("--name", default="screen-16k")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--clean", action="store_true", help="also leave out EVAL_PARENTS")
    p.add_argument("--balance", choices=["skill", "source"], default="skill")
    a = p.parse_args()
    if a.command == "build":
        build(a.only)
    elif a.command == "dedupe":
        dedupe(a.only)
    elif a.command == "mix":
        mix(a.rows, a.text_share, a.name, a.seed, a.clean, a.balance)
    else:
        for name, (skill, _) in converters().items():
            print(f"{name:28s} {skill}")


if __name__ == "__main__":
    main()
