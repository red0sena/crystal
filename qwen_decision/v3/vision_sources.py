"""Image training sources and the held-out vision test benchmarks (2026-10-31).

  .venv/bin/python -m qwen_decision.v3.vision_sources download [--role train|heldout] [--only NAME ...]
  .venv/bin/python -m qwen_decision.v3.vision_sources list

Hub sources go through sources.common.fetch (data/raw/<repo>@<revision>, the revision pinned at first fetch);
URL sources land in data/raw/url/<name>/. Every fetch appends one line to data/raw/vision-manifest.jsonl
(name, repo or urls, revision, patterns, files, bytes, seconds, status), and a name already fetched is skipped.

Roles. `train` sources feed the image training mix; labels are human or programmatic only (no model labels).
`heldout` benchmarks are evaluation only: never trained on, and none of them is a training source or a board
set, so they test transfer to other benchmarks. The board's public vision suite stays the last check.
Flags: board_train = the train split of a benchmark on the board (disclose at submission); nc = non-commercial
or unverified licence, comparison experiments only. Catalogue and reasons: wiki/designs/vision-data-sources.md.
Very large repos are taken in part (patterns below); what was left out is in the wiki page.
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

from qwen_decision.v3.sources.common import RAW, fetch

MANIFEST = RAW / "vision-manifest.jsonl"
URL_ROOT = RAW / "url"

CAULDRON = ["ai2d", "aokvqa", "clevr", "clevr_math", "cocoqa", "docvqa", "dvqa", "figureqa", "finqa", "geomverse",
            "hateful_memes", "hitab", "iconqa", "infographic_vqa", "intergps", "mapqa", "multihiertt", "nlvr2",
            "ocrvqa", "okvqa", "plotqa", "raven", "robut_sqa", "scienceqa", "screen2words", "spot_the_diff",
            "st_vqa", "tabmwp", "tallyqa", "tat_qa", "textvqa", "tqa", "visual7w", "visualmrc", "vqav2", "vsr"]


@dataclass
class Source:
    name: str
    role: str                       # train | heldout
    skill: str
    label: str                      # human | program
    licence: str
    repo: str | None = None
    patterns: list[str] | None = None
    urls: list[str] = field(default_factory=list)
    board_train: bool = False
    nc: bool = False
    note: str = ""


def hub(name, repo, patterns, skill, label, licence, role="train", **kw) -> Source:
    return Source(name, role, skill, label, licence, repo=repo, patterns=patterns, **kw)


def url(name, urls, skill, label, licence, role="train", **kw) -> Source:
    return Source(name, role, skill, label, licence, urls=urls, **kw)


SOURCES = [
    # --- held-out test benchmarks (evaluation only) ---
    hub("mmstar", "Lin-Chen/MMStar", None, "general", "human", "unstated", "heldout",
        note="items drawn from training benchmarks (AI2D, ScienceQA) or the board's parents (MMMU) are dropped"),
    hub("mmvp", "MMVP/MMVP", None, "fine perception", "human", "MIT", "heldout"),
    hub("whatsup", "ServiceNow/whatsup_all", ["data/Controlled_Images_*", "README.md"], "spatial", "human", "unstated",
        "heldout", note="Controlled_Images_A/B only (the first fetch took the whole 10.8 GB repo)"),
    hub("countbench", "vikhyatk/CountBenchQA", None, "counting", "human", "unstated", "heldout"),
    hub("embspatial", "FlagEval/EmbSpatial-Bench", None, "spatial", "program", "unstated", "heldout"),
    hub("3dsrbench", "ccvl/3DSRBench", None, "spatial", "human", "CC BY 4.0", "heldout"),
    hub("naturalbench", "BaiqiL/NaturalBench", None, "fine perception", "human", "Apache-2.0", "heldout"),
    hub("vstar", "craigwu/vstar_bench", None, "fine perception", "human", "unstated", "heldout"),
    hub("mme_realworld", "yifanzhang114/MME-RealWorld-Lmms-eval", [f"data/train-000{i}0-*" for i in range(7)],
        "real-world", "human", "unstated", "heldout", note="7 shards spread over the 70 (about 14 of 140 GB)"),
    hub("chartqapro", "ahmed-masry/ChartQAPro", None, "chart", "human", "MIT", "heldout"),
    hub("seed2plus", "AILab-CVC/SEED-Bench-2-plus", None, "text-rich", "human", "CC BY-NC 4.0", "heldout"),
    hub("visualwebbench", "visualwebbench/VisualWebBench", None, "web", "human", "Apache-2.0", "heldout"),
    hub("sugarcrepe", None, None, "image-text", "program", "unstated", "heldout"),  # repos filled below
    hub("memotion", "Ahren09/MMSoc_Memotion", None, "moderation", "human", "unstated", "heldout"),
    hub("mathvista", "AI4Math/MathVista", ["*testmini*", "README.md"], "math", "human", "CC BY-SA 4.0", "heldout",
        note="items from sources that are training benchmarks are dropped"),
    hub("pope", "lmms-lab-encoder/POPE", None, "general", "program", "unstated", "heldout"),
    url("wildreceipt", ["https://download.openmmlab.com/mmocr/data/wildreceipt.tar"], "kie", "human", "unstated",
        "heldout", note="test split only"),

    # --- training sources ---
    hub("cauldron", "HuggingFaceM4/the_cauldron", [f"{c}/*" for c in CAULDRON] + ["README.md"], "mixed",
        "human/program", "per subset", note="hateful_memes and infographic_vqa are board train splits"),
    hub("gqa", "lmms-lab-encoder/GQA", ["train_balanced_images/*", "train_balanced_instructions/*", "README.md"],
        "spatial", "program", "MIT tag; VG CC BY 4.0"),
    hub("chartqa", "HuggingFaceM4/ChartQA", ["data/train-*", "README.md"], "chart", "human", "GPL-3.0",
        note="human_or_machine = human only"),
    hub("scienceqa", "derek-thomas/ScienceQA", ["data/train-*", "README.md"], "science", "human", "CC BY-NC-SA",
        nc=True),
    hub("sat", "array/SAT", ["SAT_train.parquet", "SAT_static.parquet", "README.md"], "spatial", "program", "MIT"),
    hub("mind2web", "osunlp/Multimodal-Mind2Web", ["data/train-*", "README.md"], "web", "human", "CC BY 4.0",
        board_train=True),
    hub("cord", "naver-clova-ix/cord-v2", ["data/train-*", "README.md"], "kie", "human", "CC BY 4.0",
        board_train=True),
    hub("funsd", "nielsr/funsd", ["data/train-*", "README.md"], "kie", "human", "non-commercial", board_train=True,
        nc=True),
    hub("rico_screenqa", "bevaya/RICO-ScreenQA", ["data/train-*", "README.md"], "web", "human", "CC BY 4.0"),
    hub("imagecode", "BennoKrojer/ImageCoDe", ["image-sets.zip", "train_data.jsonl", "valid_data.jsonl", "README.md"],
        "image-text", "human", "AFL-3.0"),
    hub("coco_karpathy", "yerevann/coco-karpathy", None, "image-text", "human", "COCO terms"),
    url("coco_images", ["http://images.cocodataset.org/zips/train2014.zip",
                        "http://images.cocodataset.org/zips/val2014.zip"], "image-text", "human", "COCO terms"),
    hub("fsc147", "isentropic/FSC147", ["images_384_VarV2/*", "annotation_FSC147_384.json",
                                        "Train_Test_Val_FSC_147.json", "ImageClasses_FSC147.txt"],
        "counting", "human", "MIT tag"),
    hub("nyu_depth", "sayakpaul/nyu_depth_v2", ["data/train-000000.tar", "data/train-000001.tar", "README.md"], "depth",
        "program", "Apache tag", note="2 train shards of the 35 GB repo"),
    hub("megadepth", "vctvct123/Megadepth", None, "depth", "program", "unstated", nc=True),
    hub("hypersim", "ritianyu/Hypersim", None, "depth", "program", "CC BY-SA 3.0 (Hypersim)"),
    hub("hpatches", "vbalnt/hpatches", ["hpatches-sequences-release.zip", "README.md"], "correspondence", "program",
        "MIT tag"),
    url("spair71k", ["https://cvlab.postech.ac.kr/research/SPair-71k/data/SPair-71k.tar.gz"], "correspondence",
        "human", "research", nc=True),
    hub("co3d_tv", "mutou0308/co3dv2", ["tv.zip"], "multi-view", "program", "CC BY-NC (CO3D)", nc=True,
        note="one category of the 5.3 TB repo"),
    hub("nuscenes_qa", "KevinNotSmile/nuscenes-qa-mini", ["day-train/*"], "real-world", "program", "CC BY-NC-SA",
        nc=True),
    hub("android_control", "smolagents/android-control", [f"data/train-000{i}*" for i in range(7)], "web", "human",
        "unstated", note="70 of 275 train shards"),
    hub("aitw", "leosltl/Android-in-the-Wild", ["general/*", "splits/*", "README.md"], "web", "human", "CC BY 4.0",
        note="general subset only (13.6 of 494 GB)"),
    hub("osatlas", "OS-Copilot/OS-Atlas-data",
        ["desktop_domain/linux_*", "desktop_domain/macos_*", "mobile_domain/rico_imgs.zip",
         "mobile_domain/ricosca.json", "mobile_domain/widget_captioning.json", "mobile_domain/mobile_images.zip",
         "mobile_domain/aw_mobile.json", "mobile_domain/UIBert.zip", "mobile_domain/uibert_raw.json",
         "prompts.json", "README.md"], "web", "program", "Apache tag",
        note="linux/macOS desktop and RICO/mobile parts; web (598 GB) and IG_data (holds Mind2Web images) left out"),
    hub("geometry3k", "hiyouga/geometry3k", ["*train*", "README.md"], "math", "human", "MIT tag"),
    hub("geoqa_plus", "Yuting6/geoqa_plus", ["*train*", "README.md"], "math", "human", "unstated", nc=True),
    hub("pets", "timm/oxford-iiit-pet", ["*train*", "README.md"], "classification", "human", "CC BY-SA 4.0"),
    hub("flowers", "nkirschi/oxford-flowers", ["*train*", "README.md"], "classification", "human", "unstated",
        nc=True),
    hub("food101", "ethz/food101", ["*train*", "README.md"], "classification", "human", "unstated", nc=True),
    hub("cars", "tanganke/stanford_cars", ["data/train-*", "README.md"], "classification", "human", "research",
        nc=True),
    hub("places365", "ljnlonoljpiljm/places365-256px", [f"data/train-0000{i}-*" for i in range(8)],
        "classification", "human", "research", nc=True, note="8 of 39 shards"),

    # --- more fields (11-02): one small train part each, for task diversity ---
    hub("dtd", "tanganke/dtd", ["data/train-*", "README.md"], "classification", "human", "research", nc=True,
        note="textures"),
    hub("gtsrb", "tanganke/gtsrb", ["data/train-*", "README.md"], "classification", "human", "CC0 (GTSRB)",
        note="traffic signs"),
    hub("eurosat", "tanganke/eurosat", ["data/train-*", "README.md"], "remote-sensing", "human", "MIT (EuroSAT)"),
    hub("resisc45", "tanganke/resisc45", ["data/train-*", "README.md"], "remote-sensing", "human", "research",
        nc=True),
    hub("aircraft", "clip-benchmark/wds_fgvc_aircraft", ["train/0.tar", "train/1.tar", "classnames.txt"],
        "classification", "human", "research", nc=True, note="2 of 6 train shards"),
    hub("pcam", "clip-benchmark/wds_vtab-pcam", ["train/0.tar", "classnames.txt"], "medical", "human", "CC0",
        note="1 of 28 train shards"),
    hub("retinopathy", "clip-benchmark/wds_vtab-diabetic_retinopathy", ["train/0.tar", "classnames.txt"], "medical",
        "human", "Kaggle EyePACS terms", nc=True, note="1 of 12 train shards"),
    hub("vqa_rad", "flaviagiammarino/vqa-rad", ["data/train-*", "README.md"], "medical", "human", "CC0"),
    hub("path_vqa", "flaviagiammarino/path-vqa", ["data/train-*", "README.md"], "medical", "human", "MIT tag"),
    hub("rendered_sst2", "clip-benchmark/wds_renderedsst2", ["train/*", "classnames.txt"], "reading", "human",
        "MIT (CLIP release)", note="sentiment of a rendered sentence"),
    hub("svhn", "clip-benchmark/wds_vtab-svhn", ["train/0.tar", "train/1.tar", "classnames.txt"], "reading", "human",
        "non-commercial", nc=True, note="2 of 8 train shards"),
    hub("iam_line", "Teklia/IAM-line", ["data/train.parquet", "README.md"], "reading", "human", "IAM terms", nc=True,
        note="handwritten lines"),
    hub("smallnorb_elevation", "clip-benchmark/wds_vtab-smallnorb_label_elevation",
        ["train/0.tar", "train/1.tar", "classnames.txt"], "3d-pose", "program", "unstated", nc=True),
    hub("smallnorb_azimuth", "clip-benchmark/wds_vtab-smallnorb_label_azimuth",
        ["train/0.tar", "train/1.tar", "classnames.txt"], "3d-pose", "program", "unstated", nc=True),
    hub("dsprites", "clip-benchmark/wds_vtab-dsprites_label_orientation", ["train/0.tar", "train/1.tar",
                                                                           "classnames.txt"],
        "3d-pose", "program", "Apache-2.0"),
    hub("dmlab", "clip-benchmark/wds_vtab-dmlab", ["train/0.tar", "classnames.txt"], "depth", "program",
        "unstated", nc=True, note="1 of 8 train shards"),
    hub("clevr_distance", "clip-benchmark/wds_vtab-clevr_closest_object_distance", ["train/0.tar",
                                                                                    "classnames.txt"],
        "depth", "program", "CC BY 4.0 (CLEVR)", note="1 of 8 train shards"),
    hub("country211", "clip-benchmark/wds_country211", ["train/0.tar", "classnames.txt"], "geolocation", "human",
        "YFCC100M CC subset", note="1 of 7 train shards"),
]
SUGARCREPE = [f"HuggingFaceM4/SugarCrepe_{k}" for k in
              ("replace_obj", "replace_att", "replace_rel", "swap_obj", "swap_att", "add_obj", "add_att")]
BY_NAME = {s.name: s for s in SOURCES}


def done() -> set[str]:
    if not MANIFEST.exists():
        return set()
    return {r["name"] for r in map(json.loads, MANIFEST.read_text().splitlines()) if r["status"] == "ok"}


def size(path: Path) -> tuple[int, int]:
    files = [p for p in path.rglob("*") if p.is_file() and ".cache" not in p.parts]
    return len(files), sum(p.stat().st_size for p in files)


def download(s: Source) -> dict:
    started = time.time()
    rec = {"name": s.name, "role": s.role, "skill": s.skill, "label": s.label, "licence": s.licence,
           "board_train": s.board_train, "nc": s.nc, "note": s.note}
    try:
        if s.urls:
            local = URL_ROOT / s.name
            local.mkdir(parents=True, exist_ok=True)
            for u in s.urls:
                target = local / u.rsplit("/", 1)[1]
                if not target.exists():
                    urllib.request.urlretrieve(u, target.with_suffix(target.suffix + ".part"))
                    target.with_suffix(target.suffix + ".part").rename(target)
            rec.update(urls=s.urls, path=str(local))
        else:
            repos = SUGARCREPE if s.name == "sugarcrepe" else [s.repo]
            rec["repos"] = []
            for repo in repos:
                local, revision = fetch(repo, s.patterns)
                rec["repos"].append({"repo": repo, "revision": revision, "path": str(local), "patterns": s.patterns})
        paths = [Path(rec["path"])] if s.urls else [Path(r["path"]) for r in rec["repos"]]
        counts = [size(p) for p in paths]
        rec.update(files=sum(c[0] for c in counts), bytes=sum(c[1] for c in counts), status="ok")
    except Exception as e:  # noqa: BLE001 - recorded and retried on the next run
        rec.update(status="failed", error=f"{type(e).__name__}: {e}"[:500])
    rec["seconds"] = round(time.time() - started, 1)
    with open(MANIFEST, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(json.dumps({k: rec.get(k) for k in ("name", "status", "files", "bytes", "seconds", "error")}), flush=True)
    return rec


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=["download", "list"])
    p.add_argument("--role", choices=["train", "heldout"])
    p.add_argument("--only", nargs="*")
    a = p.parse_args()
    chosen = [s for s in SOURCES if (not a.role or s.role == a.role) and (not a.only or s.name in a.only)]
    if a.command == "list":
        for s in chosen:
            print(f"{s.role:8s} {s.name:16s} {s.skill:16s} {s.label:14s} board_train={s.board_train!s:5s} "
                  f"nc={s.nc!s:5s} {s.urls[0] if s.urls else s.repo or 'SugarCrepe x7'}")
        return
    skip = done()
    for s in chosen:
        if s.name not in skip:
            download(s)


if __name__ == "__main__":
    main()
