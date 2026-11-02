"""Held-out vision test set (2026-10-31): multiple-choice rows from benchmarks that are neither image training
sources nor board sets, so a score here measures transfer to other benchmarks.

  PYTHONPATH=.:external/decision-index .venv/bin/python -m qwen_decision.v3.vision_heldout build [--only NAME ...]
  PYTHONPATH=.:external/decision-index .venv/bin/python -m qwen_decision.v3.vision_heldout run <run name> [--limit N]
  PYTHONPATH=.:external/decision-index .venv/bin/python -m qwen_decision.v3.vision_heldout score

Sources are the `heldout` entries of vision_sources (fetched by `vision_sources download --role heldout`).
Each benchmark gives at most CAP rows, a seeded sample stratified by subtask. Options are shuffled with the same
seed except yes/no and numbers, which keep their natural order. Items drawn from benchmarks we train on or from
the board's parents are dropped (MMStar keeps SEED-Bench and MMBench items; MathVista drops items whose source is
a training benchmark). Rows follow the kit's row format (decision_index.vision.rows.to_row), so the kit runner and
our VisionEngine run them unchanged; the score is each benchmark's chance-corrected skill and their plain mean.
images.json lists every image's sha256 and 64-bit difference hash, for removing these images from training.
"""
from __future__ import annotations

import argparse
import ast
import base64
import csv
import hashlib
import io
import json
import random
import re
import tarfile
import zipfile
from pathlib import Path

import pyarrow.parquet as pq

from qwen_decision.v3.vision_sources import MANIFEST, URL_ROOT

ROOT = Path("data/vision-heldout")
ROWS = ROOT / "rows.jsonl"
OUT = Path("runs/vision-heldout")
CAP = 300
SEED = 20261031
LETTERS = "ABCDEFGHIJKL"
ENGINE = "qwen_decision.v3.vision_engine:VisionEngine"
RUNS = {"base-chat": ("models/qwen3.5-0.8b", "chat"), "wise80-decide": ("runs/v3/wise80-g6-g7e2", "decide")}
# MathVista sources that are image training benchmarks (vision_sources) or their close variants
TRAINED = {"FigureQA", "DVQA", "PlotQA", "MapQA", "ChartQA", "IconQA", "TQA", "AI2D", "ScienceQA", "GeoQA+",
           "Geometry3K", "CLEVR-Math", "TabMWP", "VQA2.0", "A-OKVQA", "TextVQA", "DocVQA", "ST-VQA", "OCR-VQA",
           "Visual7W", "TallyQA", "NLVR2", "RAVEN", "GQA", "VSR", "COCO-QA", "OK-VQA", "VisualMRC", "Inter-GPS",
           "UniGeo"}  # UniGeo reuses Geometry3K and GeoQA problems


def snapshot(name: str) -> Path:
    rec = [r for r in map(json.loads, MANIFEST.read_text().splitlines()) if r["name"] == name and r["status"] == "ok"]
    if not rec:
        raise FileNotFoundError(f"{name} is not fetched; run vision_sources download --role heldout")
    r = rec[-1]
    return Path(r["path"]) if "path" in r else Path(r["repos"][0]["path"])


def parquets(path: Path, pattern: str = "*.parquet") -> list[Path]:
    return sorted(p for p in path.rglob(pattern) if ".cache" not in p.parts)


def table(path: Path, pattern: str = "*.parquet", columns=None) -> list[dict]:
    return [r for f in parquets(path, pattern) for r in pq.read_table(f, columns=columns).to_pylist()]


def image_bytes(x) -> bytes:
    if isinstance(x, dict):
        return x["bytes"]
    if isinstance(x, (bytes, bytearray)):
        return bytes(x)
    if isinstance(x, list):
        return bytes(x)
    raise TypeError(type(x))


def ext(data: bytes) -> str:
    return ".png" if data[:8] == b"\x89PNG\r\n\x1a\n" else ".webp" if data[8:12] == b"WEBP" else ".jpg"


class Bench:
    def __init__(self, name: str, skill: str):
        self.name, self.skill, self.items = name, skill, []
        self.rng = random.Random(f"{SEED}:{name}")

    def save(self, key: str, data: bytes) -> str:
        rel = Path(self.name) / "img" / f"{key}{ext(data)}"
        (ROOT / rel).parent.mkdir(parents=True, exist_ok=True)
        (ROOT / rel).write_bytes(data)
        return str(rel)

    def add(self, key, images, question, options, gold: int, subtask="all", shuffle=True):
        options = [str(o).strip() for o in options]
        if len(set(options)) != len(options) or not 2 <= len(options) <= len(LETTERS) or not 0 <= gold < len(options):
            return
        order = list(range(len(options)))
        if shuffle:
            self.rng.shuffle(order)
        self.items.append({"id": f"heldout_{self.name}:{key}", "bench": self.name, "skill": self.skill,
                           "subtask": subtask, "images": images, "question": question.strip(),
                           "options": [options[i] for i in order], "gold": LETTERS[order.index(gold)]})


def stratified(rows: list, cap: int, key, seed: str) -> list:
    """Seeded sample of at most cap rows, spread evenly over key groups."""
    rng = random.Random(f"{SEED}:{seed}")
    groups = {}
    for r in rows:
        groups.setdefault(key(r), []).append(r)
    for g in groups.values():
        rng.shuffle(g)
    out, i = [], 0
    while len(out) < cap and any(i < len(g) for g in groups.values()):
        out += [g[i] for g in groups.values() if i < len(g)][: cap - len(out)]
        i += 1
    return out


def letter_options(text: str, pattern: str) -> tuple[str, list[str]]:
    """Split "question ... (A) x (B) y" style text into the question and its options."""
    marks = list(re.finditer(pattern, text))
    if len(marks) < 2:
        return text, []
    opts = [text[m.end(): (marks[i + 1].start() if i + 1 < len(marks) else len(text))].strip().rstrip(",").strip()
            for i, m in enumerate(marks)]
    return text[: marks[0].start()].strip(), opts


# --- builders ---------------------------------------------------------------------------------------------------

def mmstar(b: Bench):
    rows = [r for r in table(snapshot("mmstar"), "mmstar.parquet")
            if r["meta_info"]["source"] in ("SEEDBench_IMG", "MMBench")]
    for r in stratified(rows, CAP, lambda r: r["category"], b.name):
        q, opts = letter_options(r["question"].replace("\nOptions:", " Options:"), r"(?:Options: |, )([A-F]): ")
        q = q.replace("Options:", "").strip()
        b.add(r["index"], [b.save(str(r["index"]), image_bytes(r["image"]))], q, opts, "ABCDEF".index(r["answer"]),
              r["category"])


def mmvp(b: Bench):
    root = snapshot("mmvp")
    with open(root / "Questions.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        _, opts = letter_options(r["Options"], r"\(([a-z])\)\s*")
        gold = "abcdefgh".index(r["Correct Answer"].strip("() "))
        img = (root / "MMVP Images" / f"{r['Index']}.jpg").read_bytes()
        b.add(r["Index"], [b.save(r["Index"], img)], r["Question"], opts, gold)


def whatsup(b: Bench):
    root = snapshot("whatsup")
    rows = []
    for split in ("Controlled_Images_A", "Controlled_Images_B"):
        rows += [dict(r, split=split) for r in table(root, f"{split}-*.parquet")]
    for i, r in enumerate(stratified(rows, CAP, lambda r: r["split"], b.name)):
        imgs = r["image_options"] if isinstance(r["image_options"], list) else [r["image_options"]]
        caps = r["caption_options"] if isinstance(r["caption_options"], list) else ast.literal_eval(r["caption_options"])
        b.add(f"{r['split']}:{i}", [b.save(f"{r['split']}_{i}", image_bytes(imgs[0]))],
              "Which caption describes the image?", caps, 0, r["split"])


def countbench(b: Bench):
    rows = table(snapshot("countbench"))
    for i, r in enumerate(stratified(rows, CAP, lambda r: r["number"], b.name)):
        n = int(r["number"])
        near = [k for k in range(max(0, n - 3), n + 4) if k != n]
        b.rng.shuffle(near)
        opts = sorted([n] + near[:3])
        b.add(i, [b.save(str(i), image_bytes(r["image"]))], r["question"], opts, opts.index(n), str(n), shuffle=False)


def embspatial(b: Bench):
    rows = table(snapshot("embspatial"), columns=["question_id", "question", "relation", "image", "answer_options",
                                                  "answer"])
    for r in stratified(rows, CAP, lambda r: r["relation"], b.name):
        opts = r["answer_options"] if isinstance(r["answer_options"], list) else ast.literal_eval(r["answer_options"])
        b.add(r["question_id"], [b.save(r["question_id"], image_bytes(r["image"]))], r["question"], opts,
              int(r["answer"]), r["relation"])


def threedsr(b: Bench):
    root = snapshot("3dsrbench")
    zf = zipfile.ZipFile(root / "coco_images.zip")
    names = {n.rsplit("/", 1)[-1]: n for n in zf.namelist() if not n.endswith("/")}
    # "-flip" rows expect the horizontally flipped image, which is not shipped; keep the originals only
    rows = [r for r in table(root) if r["image_url"].rsplit("/", 1)[-1] in names and "-flip" not in r["index"]]
    for r in stratified(rows, CAP, lambda r: r["category"], b.name):
        opts = [r[k] for k in "ABCD" if r[k] not in (None, "None", "")]
        data = zf.read(names[r["image_url"].rsplit("/", 1)[-1]])
        b.add(r["index"], [b.save(r["index"], data)], r["question"], opts, "ABCD".index(r["answer"]), r["category"])


def naturalbench(b: Bench):
    rows = table(snapshot("naturalbench"))
    items = []
    for r in rows:
        for qi in (0, 1):
            for ii in (0, 1):
                items.append((r, qi, ii))
    for r, qi, ii in stratified(items, CAP, lambda x: x[0]["Question_Type"], b.name):
        q, ans = r[f"Question_{qi}"], r[f"Image_{ii}_Question_{qi}"].strip()
        key = f"{r['Index']}_{qi}_{ii}"
        img = [b.save(key, image_bytes(r[f"Image_{ii}"]))]
        if r["Question_Type"] == "yes_no":
            b.add(key, img, q, ["Yes", "No"], 0 if ans.lower() == "yes" else 1, "yes_no", shuffle=False)
        else:
            q2, _, rest = q.partition("\nOption:")
            opts = [t.strip() for _, t in re.findall(r"([A-D]):\s*(.*?);", rest)]
            if opts and ans[:1] in "ABCD":
                b.add(key, img, q2, opts, "ABCD".index(ans[:1]), r["Question_Type"])


def vstar(b: Bench):
    root = snapshot("vstar")
    rows = [json.loads(l) for l in (root / "test_questions.jsonl").read_text().splitlines()]
    for r in rows:
        text = r["text"].split("\nAnswer with")[0]
        q, opts = letter_options(text, r"\(([A-D])\)\s*")
        b.add(r["question_id"], [b.save(str(r["question_id"]), (root / r["image"]).read_bytes())], q, opts,
              "ABCD".index(r["label"].strip()), r["category"])


def mme_realworld(b: Bench):
    rows = table(snapshot("mme_realworld"), columns=["bytes", "index", "question", "multi-choice options", "answer",
                                                     "category", "l2-category"])
    for r in stratified(rows, CAP, lambda r: r["category"], b.name):
        opts = [re.sub(r"^\([A-E]\)\s*", "", o) for o in r["multi-choice options"]]
        b.add(r["index"], [b.save(str(r["index"]), base64.b64decode(r["bytes"]))], r["question"], opts,
              "ABCDE".index(r["answer"]),
              r["category"])


def chartqapro(b: Bench):
    rows = [r for r in table(snapshot("chartqapro")) if r["Question Type"] in ("Multi Choice", "Fact Checking")]
    for i, r in enumerate(stratified(rows, CAP, lambda r: r["Question Type"], b.name)):
        q, ans = r["Question"][0], r["Answer"][0].strip()
        if r["Paragraph"]:
            q = f"{r['Paragraph'].strip()}\n\n{q}"
        img = [b.save(str(i), image_bytes(r["image"]))]
        if r["Question Type"] == "Fact Checking":
            if ans in ("True", "False"):
                b.add(i, img, f"True or false: {q}", ["True", "False"], 0 if ans == "True" else 1, "fact", shuffle=False)
        else:
            q2, opts = letter_options(q, r"\n([a-e])\)\s*")
            if opts and ans[:1].upper() in "ABCDE":
                b.add(i, img, q2, opts, "ABCDE".index(ans[:1].upper()), "multi_choice")


def seed2plus(b: Bench):
    root = snapshot("seed2plus")
    qs = json.loads((root / "SEED-Bench-2-plus-text-rich.json").read_text())
    zf = zipfile.ZipFile(root / "text_rich.zip")
    for q in stratified(qs, CAP, lambda q: q["question_image_type"], b.name):
        opts = [q[f"choice_{k}"] for k in "ABCD"]
        b.add(q["question_id"], [b.save(q["question_id"], zf.read(q["data_id"]))], q["question"], opts,
              "ABCD".index(q["answer"]), q["question_image_type"])


def visualwebbench(b: Bench):
    root = snapshot("visualwebbench")
    rows = []
    for task in ("element_ground", "action_ground", "action_prediction"):
        pf = pq.ParquetFile(next(iter(parquets(root, f"{task}/*.parquet"))))
        cols = [c for c in ("id", "task_type", "image", "options", "answer", "elem_desc", "instruction", "bbox")
                if c in pf.schema_arrow.names]
        rows += pf.read(columns=cols).to_pylist()
    for r in stratified(rows, CAP, lambda r: r["task_type"], b.name):
        img = [b.save(r["id"], image_bytes(r["image"]))]
        boxes = [f"box {LETTERS[i]}" for i in range(len(r["options"]))]
        if r["task_type"] == "element_ground":
            b.add(r["id"], img, f'Which marked element is "{r["elem_desc"]}"?', boxes, r["answer"], "element_ground",
                  shuffle=False)
        elif r["task_type"] == "action_ground":
            b.add(r["id"], img, f'Task: {r["instruction"]}. Which marked element should be clicked?', boxes,
                  r["answer"], "action_ground", shuffle=False)
        else:
            b.add(r["id"], img, f'Which page opens after clicking the element in the red box ("{r["elem_desc"]}")?',
                  r["options"], r["answer"], "action_prediction")


def sugarcrepe(b: Bench):
    rec = [r for r in map(json.loads, MANIFEST.read_text().splitlines()) if r["name"] == "sugarcrepe"][-1]
    rows = []
    for repo in rec["repos"]:
        subset = repo["repo"].split("SugarCrepe_")[1]
        for f in parquets(Path(repo["path"])):
            meta = json.loads(pq.ParquetFile(f).schema_arrow.metadata[b"huggingface"])
            names = meta["info"]["features"]["true_label"]["names"]
            rows += [dict(r, subset=subset, truth=names[r["true_label"]]) for r in pq.read_table(f).to_pylist()]
    for i, r in enumerate(stratified(rows, CAP, lambda r: r["subset"], b.name)):
        caps = list(r["tested_labels"])
        if r["truth"] in caps:
            b.add(f"{r['subset']}:{i}", [b.save(f"{r['subset']}_{i}", image_bytes(r["image"]))],
                  "Which caption describes the image?", caps, caps.index(r["truth"]), r["subset"])


FIELDS = {5: "telephone number", 7: "date", 9: "time", 17: "subtotal", 19: "tax amount", 21: "tip amount",
          23: "total amount"}
AMOUNTS = {13, 15, 17, 19, 21, 23}


def wildreceipt(b: Bench):
    """Board-KIE-style items from WildReceipt test receipts: a field's value among the receipt's other values,
    and whether a field is shown at all (Yes and No balanced per field)."""
    with tarfile.open(URL_ROOT / "wildreceipt" / "wildreceipt.tar") as tf:
        receipts = [json.loads(l) for l in tf.extractfile("wildreceipt/test.txt").read().decode().splitlines() if l.strip()]
        value, present = [], {}
        for n, rc in enumerate(receipts):
            boxes = [(a["label"], a["text"].strip()) for a in rc["annotations"] if a["text"].strip()]
            labels = [lab for lab, _ in boxes]
            for lab in FIELDS:
                present.setdefault((lab, lab in labels), []).append(n)
                if labels.count(lab) != 1:
                    continue
                gold = next(t for l_, t in boxes if l_ == lab)
                group = AMOUNTS if lab in AMOUNTS else set(FIELDS) - AMOUNTS
                others = sorted({t for l_, t in boxes if l_ in group and l_ != lab and t != gold})
                if len(others) >= 2:
                    value.append((n, lab, gold, others))
        cap = CAP // 2
        picks = [("value", x) for x in stratified(value, cap, lambda x: x[1], f"{b.name}:value")]
        pres = [(lab, truth, n) for (lab, truth), ns in present.items() for n in ns]
        picks += [("present", x) for x in stratified(pres, CAP - len(picks), lambda x: (x[0], x[1]), f"{b.name}:present")]
        for i, (kind, x) in enumerate(picks):
            n = x[0] if kind == "value" else x[2]
            data = tf.extractfile("wildreceipt/" + receipts[n]["file_name"]).read()
            img = [b.save(f"{i}", data)]
            if kind == "value":
                _, lab, gold, others = x
                b.rng.shuffle(others)
                b.add(i, img, f"What is the {FIELDS[lab]} on this receipt?", [gold] + others[:3], 0, "field_value")
            else:
                lab, truth, _ = x
                b.add(i, img, f"Does this receipt show a {FIELDS[lab]}?", ["Yes", "No"], 0 if truth else 1,
                      "field_present", shuffle=False)


def memotion(b: Bench):
    rows = [r for r in table(snapshot("memotion"), "test-*.parquet") if r["offensive"] != "slight"]
    for i, r in enumerate(stratified(rows, CAP, lambda r: r["offensive"] == "not_offensive", b.name)):
        b.add(i, [b.save(str(i), image_bytes(r["image"]))], "Is this meme offensive?", ["Yes", "No"],
              1 if r["offensive"] == "not_offensive" else 0, r["offensive"], shuffle=False)


def mathvista(b: Bench):
    rows = [r for r in table(snapshot("mathvista"), "testmini-*.parquet")
            if r["question_type"] == "multi_choice" and r["metadata"]["source"] not in TRAINED]
    for r in stratified(rows, CAP, lambda r: r["metadata"]["source"], b.name):
        opts = list(r["choices"])
        if r["answer"] not in opts:
            continue
        b.add(r["pid"], [b.save(r["pid"], image_bytes(r["decoded_image"]))], r["question"], opts,
              opts.index(r["answer"]), r["metadata"]["source"])


def pope(b: Bench):
    rows = table(snapshot("pope"), "test-*.parquet")
    for r in stratified(rows, CAP, lambda r: (r["category"], r["answer"]), b.name):
        key = f"{r['category']}_{r['question_id']}"  # question ids restart in each category
        b.add(key, [b.save(key, image_bytes(r["image"]))], r["question"],
              ["Yes", "No"], 0 if r["answer"].strip().lower() == "yes" else 1, r["category"], shuffle=False)


BUILDERS = {"mmstar": ("general", mmstar), "mmvp": ("fine perception", mmvp), "whatsup": ("spatial", whatsup),
            "countbench": ("counting", countbench), "embspatial": ("spatial", embspatial),
            "3dsrbench": ("spatial", threedsr), "naturalbench": ("fine perception", naturalbench),
            "vstar": ("fine perception", vstar), "mme_realworld": ("real-world", mme_realworld),
            "chartqapro": ("chart", chartqapro), "seed2plus": ("text-rich", seed2plus),
            "visualwebbench": ("web", visualwebbench), "sugarcrepe": ("image-text", sugarcrepe),
            "memotion": ("moderation", memotion), "wildreceipt": ("kie", wildreceipt),
            "mathvista": ("math", mathvista), "pope": ("general", pope)}


# --- rows, image index, run, score ------------------------------------------------------------------------------

def dumps(x) -> str:
    return json.dumps(x, ensure_ascii=False, separators=(",", ":"))


def to_row(it: dict) -> dict:
    crit = {LETTERS[i]: o for i, o in enumerate(it["options"])}
    questions = {"answer": {"type": "choice", "instructions": it["question"], "criteria": crit}}
    prompt = {"options_in_image": False, "readout": None}
    image_sha = [hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in it["images"]]
    payload = {"state": {}, "questions": questions, "images": image_sha, "prompt": prompt}
    return {"id": it["id"], "benchmark": it["bench"], "set": it["bench"], "subtask": it["subtask"], "skill": it["skill"],
            "images": it["images"], "state": {}, "questions": questions, "prompt": prompt,
            "expected": {"answer": it["gold"]},
            "_evaluation": {"run_id": it["id"], "benchmark": it["bench"], "set": it["bench"],
                            "payload_sha256": hashlib.sha256(dumps(payload).encode()).hexdigest(),
                            "image_sha256": image_sha}}


def dhash(data: bytes) -> str:
    from PIL import Image
    im = Image.open(io.BytesIO(data)).convert("L").resize((9, 8), Image.LANCZOS)
    px = list(im.tobytes())
    bits = "".join("1" if px[r * 9 + c] > px[r * 9 + c + 1] else "0" for r in range(8) for c in range(8))
    return f"{int(bits, 2):016x}"


def build(only: list[str] | None) -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    for name, (skill, fn) in BUILDERS.items():
        if only and name not in only:
            continue
        b = Bench(name, skill)
        fn(b)
        with open(ROOT / f"{name}.items.jsonl", "w", encoding="utf-8") as f:
            f.writelines(dumps(it) + "\n" for it in b.items)
        print(json.dumps({"bench": name, "items": len(b.items)}), flush=True)
    items = [json.loads(l) for f in sorted(ROOT.glob("*.items.jsonl")) for l in f.read_text().splitlines()]
    with open(ROWS, "w", encoding="utf-8") as f:
        f.writelines(dumps(to_row(it)) + "\n" for it in items)
    index = {}
    for it in items:
        for p in it["images"]:
            data = (ROOT / p).read_bytes()
            index[p] = {"sha256": hashlib.sha256(data).hexdigest(), "dhash": dhash(data)}
    (ROOT / "images.json").write_text(json.dumps(index) + "\n")
    print(json.dumps({"rows": len(items), "images": len(index)}))


def run(name: str, limit: int | None) -> None:
    from decision_index.vision.run import run as kit_run
    model, prompt = RUNS[name]
    kit_run(ENGINE, {"model": model, "prompt": prompt}, ROOT, OUT / name, rows_path=ROWS, workers=1, limit=limit)


def score() -> None:
    rows = {r["id"]: r for r in map(json.loads, ROWS.read_text().splitlines())}
    table_ = {}
    for name in RUNS:
        path = OUT / name / "results.jsonl"
        if not path.exists():
            continue
        pred = {}
        for line in path.read_text().splitlines():
            r = json.loads(line)
            if r["status"] == "ok":
                pred[r["run_id"]] = r["response"]["answers"]["answer"]["choice"]
        per = {}
        for rid, r in rows.items():
            k = len(r["questions"]["answer"]["criteria"])
            per.setdefault(r["benchmark"], []).append((pred.get(rid) == r["expected"]["answer"], 1 / k))
        skills = {}
        for bench, v in per.items():
            acc, ch = sum(c for c, _ in v) / len(v), sum(x for _, x in v) / len(v)
            skills[bench] = {"rows": len(v), "accuracy": round(acc, 4), "chance": round(ch, 4),
                             "skill": round((acc - ch) / (1 - ch), 4)}
        mean = sum(max(0.0, min(1.0, s["skill"])) for s in skills.values()) / len(skills)
        table_[name] = {"index": round(100 * mean, 2), "missing": len(rows) - len(pred), "benchmarks": skills}
    (OUT / "summary.json").write_text(json.dumps(table_, indent=1) + "\n")
    print(json.dumps({n: {"index": t["index"], **{b: s["skill"] for b, s in t["benchmarks"].items()}}
                      for n, t in table_.items()}, indent=1))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=["build", "run", "score"])
    p.add_argument("name", nargs="?", choices=list(RUNS))
    p.add_argument("--only", nargs="*")
    p.add_argument("--limit", type=int)
    a = p.parse_args()
    if a.command == "build":
        build(a.only)
    elif a.command == "run":
        run(a.name, a.limit)
    else:
        score()


if __name__ == "__main__":
    main()
