"""Historical URL annotations to choice/noul, using only URL-derived inputs.

The source labels come from phishing feeds and legitimate-site collections, not
a teacher model. Do not copy the dataset's webpage/WHOIS/reputation features into
inputs: they are unavailable to our native inference request and may reveal gold.
"""
from collections import Counter
import hashlib
from pathlib import Path
import random
import traceback
from urllib.parse import urlsplit

import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download

from qwen_decision.v3 import champion_audit as audit, decontam, train
from qwen_decision.v3 import champion_joint_data as old
from qwen_decision.v3.sources import common, phishing

ROOT = Path("runs/champion-v1/full-g7-v3")
RAW = ROOT / "raw/url-evidence"
REPO = "pirocheto/phishing-url"
REVISION = "1375d450d4540205ac11781b807249c9509739cd"
SOURCE = "phishing-url-train"


def views(url, label):
    assert label in ("phishing", "legitimate")
    host = (urlsplit(url).hostname or "").lower()
    assert host
    key = hashlib.sha256(url.encode()).hexdigest()
    state = {"url": url, "hostname": host}
    options = ["phishing", "legitimate"]
    random.Random(key).shuffle(options)
    q, t = common.choice(state, "Classify the website URL as phishing or legitimate.", options, options.index(label))
    variants = [("choice", q, t),
                ("positive", *common.noul("The link points to a phishing website.", label == "phishing")),
                ("negative", *common.noul("The link points to a legitimate website.", label == "legitimate"))]
    for name, question, target in variants:
        yield {"id": f"{SOURCE}:{key}:{name}", "original_id": f"{SOURCE}:{key}", "source": SOURCE,
               "family": "phishing-url", "modality": "text", "state": state,
               "question": question, "target": target, "kind": question["type"], "ordered": False,
               "item": f"{SOURCE}:{key}", "documents": [f"{SOURCE}:host:{host}"],
               "split": "train", "label_origin": "programmatic", "target_kind": "single",
               "annotation_basis": "Historical Phishtank/Openphish and Alexa/Yandex corpus labels; not model-generated",
               "upstream": REPO, "upstream_revision": REVISION, "upstream_split": "train",
               "upstream_label": label, "training_component": "url-supplement"}


def prepare():
    done = ROOT / "enriched-urls-prepared.json"
    if done.exists():
        proof = audit.read(done)
        assert proof["source_sha256"] == audit.sha(__file__)
        for f, sha in proof["files"].items():
            assert audit.sha(f) == sha
        return
    info = HfApi().dataset_info(REPO, revision=REVISION, files_metadata=True)
    entry = next(f for f in info.siblings if f.rfilename == "data/train.parquet")
    p = Path(hf_hub_download(REPO, entry.rfilename, repo_type="dataset", revision=REVISION, local_dir=RAW))
    assert p.stat().st_size == entry.size and audit.sha(p) == entry.lfs.sha256
    # Block all observed evaluation and previous dev URL hosts without reading their golds.
    urls, hosts = set(), set()
    sources = [Path("runs/champion-v1/public-eval-v1/text-counted-rows.jsonl"),
               old.ROOT / "restored/data/mix/v3-gold7.dev.jsonl", Path("data/fresh/fresh-v1.jsonl")]
    for source in sources:
        for r in audit.rows(source):
            for url in phishing.links(" ".join(decontam.strings(r["state"]))):
                urls.add(url.rstrip("/").lower())
                try:
                    host = (urlsplit(url).hostname or "").lower()
                except ValueError:
                    continue
                if host:
                    hosts.add(host)
    flt = decontam.Filter(old.ROOT / "restored/data/decontam")
    dropped, counts, training, dev = Counter(), Counter(), [], []
    seen = set()
    raw = pq.read_table(p, columns=["url", "status"]).to_pylist()
    for r in raw:
        url, label = r["url"].strip(), r["status"]
        parsed = urlsplit(url)
        host = (parsed.hostname or "").lower()
        assert label in ("phishing", "legitimate")
        if not host or parsed.scheme not in ("http", "https"):
            dropped["invalid_url"] += 1
            continue
        if url.rstrip("/").lower() in urls or host in hosts:
            dropped["evaluation_or_existing_dev_url_host"] += 1
            continue
        if url in seen:
            dropped["duplicate_url"] += 1
            continue
        seen.add(url)
        variants = list(views(url, label))
        if any(flt.flagged(decontam.row_texts(r["state"], {"q": r["question"]})) for r in variants):
            dropped["public_jeval_text_overlap"] += 1
            continue
        # Every URL and every view sharing a hostname uses one split.
        is_dev = int(hashlib.sha256(host.encode()).hexdigest()[:8], 16) % 5 == 0
        (dev if is_dev else training).extend(variants)
        counts[("dev" if is_dev else "train", label)] += 1
    assert training and dev
    assert not {r["state"]["hostname"] for r in training} & {r["state"]["hostname"] for r in dev}
    files = []
    for name, rows in [("phishing-url-train", training), ("phishing-url-dev", dev)]:
        dest = ROOT / "mix" / (name + ".jsonl")
        tmp = dest.with_suffix(".partial")
        if tmp.exists():
            tmp.unlink()
        old.write(tmp, rows)
        tmp.replace(dest)
        files.append(dest)
    proof = {"complete": True, "training_started": False, "repo": REPO, "revision": REVISION,
             "licence": "CC-BY-4.0", "upstream_split": "train", "raw_urls": len(raw),
             "train_rows": len(training), "dev_rows": len(dev), "dropped_urls": dict(dropped),
             "distinct_urls_by_label": {str(k): v for k, v in counts.items()},
             "raw_file": str(p), "raw_sha256": audit.sha(p), "source_sha256": audit.sha(__file__),
             "files": {str(f): audit.sha(f) for f in files},
             "provenance": "https://arxiv.org/html/2010.12847v1#S4.SS1",
             "limitations": "Historical reputation-derived website annotations. Not current URL safety, and not email-level labels. No page content, external reputation features or original source test split used. Hostname holdout is not registrable-domain holdout."}
    train.atomic_json(done, proof)
    print("URLS_PREPARED", len(training), len(dev), proof["distinct_urls_by_label"], flush=True)


if __name__ == "__main__":
    try:
        prepare()
    except Exception:
        train.atomic_json(ROOT / "enriched-urls-failure.json", {"traceback": traceback.format_exc()})
        raise
