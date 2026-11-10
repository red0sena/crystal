"""Fixed-weight Crystal: question header x short-input repetition, text and vision.

Native prompt/readout/attention are controls. No KV approximation, training,
option reordering, test-based selection, truncation, or default serving changes.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import shutil
import time
import traceback

import numpy as np
import torch
from transformers import AutoProcessor, AutoTokenizer

from qwen_decision.v3 import format as fmt, modeling
from qwen_decision.v3 import vision_heldout as vision
from qwen_decision.v3.engine import Scorer
from qwen_decision.v3.jepa_decision import sha

ROOT = Path("runs/inference-policy-v1")
MODEL = Path("runs/v3/wise80-g6-g7e2")
BASE = Path("models/qwen3.5-0.8b")
TEXT = Path("data/fresh/fresh-v1.jsonl")
POLICIES = ("base", "header", "repeat", "header_repeat")
BENCHMARKS = ("mmstar", "mmvp", "countbench", "embspatial", "chartqapro", "seed2plus", "mathvista", "memotion")
MAX_ONCE, MAX_INPUT, BOOTSTRAPS = 3000, 8192, 5000
READ_AGAIN = "\n\nRead the same input again before answering:\n"
IMAGE = "<|vision_start|><|image_pad|><|vision_end|>"


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def read(path):
    return json.loads(path.read_text())


def rows(path):
    return [json.loads(line) for line in path.open()]


def render(tokenizer, state, question, codes, policy, repeat_enabled, questions=None):
    assert policy in POLICIES
    user = fmt.user_prompt(state, question, codes, "d3")
    if policy in ("header", "header_repeat"):
        asked = list((questions or {"q": question}).values())
        header = "Questions that will be asked about the context:\n" + "\n".join(
            "- " + fmt.describe(q.get("instructions") or fmt.DEFAULT_INSTRUCTIONS) for q in asked)
        user = header + "\n\n" + user
    repeated = repeat_enabled and policy in ("repeat", "header_repeat")
    if repeated:
        user = user + READ_AGAIN + user
    text = tokenizer.apply_chat_template(
        [{"role": "system", "content": fmt.SYSTEM_PROMPT}, {"role": "user", "content": user}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False)
    return text, repeated


def group_assign(records):
    # Group repeated text states / shared image files, with MMVP's paired images
    # kept together even though their pixel hashes are different.
    parent = list(range(len(records))); first = {}
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i
    for i, row in enumerate(records):
        if row["modality"] == "text":
            value = row["state"]
            state = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, sort_keys=True)
            tokens = ["text:" + " ".join(state.lower().split())]
        else:
            tokens = ["image:" + x for x in row["image_sha256"]]
            if row["source"] == "mmvp":
                tokens.append("mmvp-pair:" + str((int(row["upstream_id"].rsplit(":", 1)[1]) - 1) // 2))
        for token in tokens:
            if token in first: parent[find(i)] = find(first[token])
            else: first[token] = i
    keys = defaultdict(list)
    for i, row in enumerate(records): keys[find(i)].append(row["id"])
    group_keys = {k: hashlib.sha256(min(v).encode()).hexdigest() for k, v in keys.items()}
    for i, row in enumerate(records):
        row["group"] = group_keys[find(i)]
        row["split"] = "calibration" if int(row["group"][:8], 16) % 5 == 0 else "evaluation"
    return records


def prepare(root):
    assert not (root / "plan.json").exists(), "Keep existing experiment unchanged"
    root.mkdir(parents=True, exist_ok=True)
    config = modeling.read_config(MODEL)
    assert config["layout"] == "d3" and config["attention"] == "noncausal" and config["deltanet"] == "causal"
    assert (MODEL / "model.safetensors").exists()
    # Build only the eight predeclared benchmarks, in an isolated directory.
    vision.ROOT = root / "vision-data"; vision.ROWS = vision.ROOT / "rows.jsonl"
    vision.build(list(BENCHMARKS))
    examples = []
    for r in rows(TEXT):
        q = r["question"]; keys = fmt.options(q)[0]
        gold = int(bool(r["label"])) if q["type"] == "noul" else keys.index(r["label"])
        examples.append({"id": "text:" + r["id"], "upstream_id": r["id"], "modality": "text",
                         "source": r["source"], "state": r["state"], "question": q, "gold": gold,
                         "images": [], "image_sha256": []})
    for r in rows(vision.ROWS):
        q = r["questions"]["answer"]; keys = fmt.options(q)[0]
        examples.append({"id": "vision:" + r["id"], "upstream_id": r["id"], "modality": "vision",
                         "source": r["benchmark"], "state": {}, "question": q,
                         "gold": keys.index(r["expected"]["answer"]),
                         "images": [str(vision.ROOT / p) for p in r["images"]],
                         "image_sha256": r["_evaluation"]["image_sha256"]})
    assert len({r["id"] for r in examples}) == len(examples)
    group_assign(examples)
    with (root / "examples.jsonl").open("x") as f:
        for r in examples: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    model_files = {str(p): sha(p) for p in sorted(MODEL.iterdir()) if p.is_file()}
    provenance = {str(p): sha(p) for p in [Path(__file__), Path(fmt.__file__), Path(modeling.__file__), Path(vision.__file__)]}
    plan = {"model": str(MODEL), "config": {k: config[k] for k in ("attention", "layout", "deltanet")},
            "model_sha256": model_files, "source_sha256": provenance, "examples_sha256": sha(root / "examples.jsonl"),
            "text_input_sha256": sha(TEXT), "policies": list(POLICIES), "vision_benchmarks": list(BENCHMARKS),
            "counts": {m: dict(Counter(r["source"] for r in examples if r["modality"] == m)) for m in ("text", "vision")},
            "split_counts": dict(Counter(r["modality"] + ":" + r["split"] for r in examples)),
            "split_rule": "SHA256(min id in connected group) first32bits modulo5=0 calibration, else evaluation; text state/shared image/MMVP pairs never split",
            "repeat": "repeat full native user turn including images only when baseline expanded input <=3000 tokens, same eligibility mask for both repetition conditions; one logical prediction",
            "header": "prepend question text list before native state, keep native system/chat template/code readout; single-question evaluation requests",
            "limits": {"baseline_repeat_threshold": MAX_ONCE, "max_expanded_tokens": MAX_INPUT,
                       "image_pixels": 1600 * 32 * 32, "no_truncation": True},
            "temperature": "per modality x policy x question type, source-balanced calibration NLL only; bounded log-T search [.05,20]; fewer than20 calibration cases => T1; all original logits retained",
            "metrics": "evaluation split source macro accuracy/NLL/Brier; all-rows descriptive means; source per-group paired bootstrap5000, fixed sources; ECE15bins descriptive",
            "latency": "one row, synchronized render+tokenization/processor+transfer+forward+logit readout; initial image disk loading excluded and reported separately; no KV cache; round-robin policy order; warmup excluded",
            "stop": "all frozen examples x four policies; no winner selection, no optional stopping; interruption may resume same plan",
            "scope": "previously observed evaluation datasets, fixed one Crystal checkpoint, eight of17 heldout vision benchmarks; not blind benchmark or PRH theorem test",
            "neural_training": False,
            "refs": ["https://github.com/1000luca/FlexDM/tree/v1.0", "https://github.com/wcole3/jev-brain/tree/di-0.3-gemma-4-31b-r2"]}
    dump(root / "plan.json", plan); shutil.copyfile(__file__, root / "experiment-source.py")
    dump(root / "status.json", {"state": "prepared", "requests": len(examples), "predictions": len(examples) * 4})
    print("PREPARED", plan["counts"], plan["split_counts"], flush=True)


class Runner:
    def __init__(self):
        self.scorer = Scorer(MODEL, max_batch=1)
        self.model = self.scorer.model.eval().requires_grad_(False)
        self.text_tok = self.scorer.tokenizer
        self.processor = AutoProcessor.from_pretrained(BASE, local_files_only=True)
        self.vision_tok = self.processor.tokenizer
        self.codes = {m: fmt.answer_codes(t) for m, t in (("text", self.text_tok), ("vision", self.vision_tok))}
        self.model.model.visual.to("cuda")

    def inputs(self, row, pictures, policy, repeat_enabled):
        modality = row["modality"]; tok = self.text_tok if modality == "text" else self.vision_tok
        codes, code_ids = self.codes[modality]; count = len(fmt.options(row["question"])[0])
        state = row["state"] if modality == "text" else IMAGE * len(pictures)
        text, repeated = render(tok, state, row["question"], codes[:count], policy, repeat_enabled)
        if modality == "text":
            encoded = tok(text, add_special_tokens=False, return_tensors="pt")
        else:
            encoded = self.processor(text=[text], images=(pictures * (2 if repeated else 1)) or None, return_tensors="pt")
            # Each expanded image token sequence must have one corresponding grid.
            assert len(encoded["image_grid_thw"]) == len(pictures) * (2 if repeated else 1)
        return encoded, code_ids[:count], repeated, hashlib.sha256(text.encode()).hexdigest()

    @torch.inference_mode()
    def predict(self, row, pictures, policy, repeat_enabled):
        torch.cuda.synchronize(); start = time.perf_counter()
        encoded, ids, repeated, prompt_sha = self.inputs(row, pictures, policy, repeat_enabled)
        length = int(encoded["input_ids"].shape[1]); assert length <= MAX_INPUT, (row["id"], policy, length)
        encoded = encoded.to("cuda"); torch.cuda.synchronize(); gpu_start = time.perf_counter()
        if row["modality"] == "text":
            codes = torch.tensor([ids], device="cuda")
            logits = modeling.code_logits(self.model, encoded["input_ids"], encoded["attention_mask"], codes,
                                           torch.ones_like(codes, dtype=torch.bool), readout=self.scorer.readout)[0]
        else:
            hidden = self.model.model(**encoded, use_cache=False).last_hidden_state[0, -1]
            assert self.scorer.readout is None, "Native vision reference uses tied LM rows"
            weight = self.model.get_output_embeddings().weight[torch.tensor(ids, device="cuda")]
            logits = weight.float() @ hidden.float()
        torch.cuda.synchronize(); gpu_ms = (time.perf_counter() - gpu_start) * 1000
        z = logits.float().cpu().tolist(); elapsed_ms = (time.perf_counter() - start) * 1000
        assert np.isfinite(z).all(), (row["id"], policy, "nonfinite logits")
        return {"logits": z, "prediction": int(np.argmax(z)), "input_tokens": length, "repeat_applied": repeated,
                "prompt_sha256": prompt_sha, "milliseconds": elapsed_ms, "forward_ms": gpu_ms,
                "image_tokens": int((encoded["input_ids"] == self.model.config.image_token_id).sum()) if row["modality"] == "vision" else 0}


def logprob(z, temperature=1.):
    z = np.asarray(z, dtype=float) / temperature
    return z - np.max(z) - math.log(float(np.exp(z - np.max(z)).sum()))


def fit_temperature(records):
    if len(records) < 20: return 1.
    counts = Counter(r["source"] for r in records)
    def objective(t):
        return sum(-logprob(r["logits"], math.exp(t))[r["gold"]] / counts[r["source"]] for r in records) / len(counts)
    # Golden-section search avoids a new dependency; only calibration labels enter.
    lo, hi = math.log(.05), math.log(20.); ratio = (math.sqrt(5) - 1) / 2
    a, b = hi - ratio * (hi - lo), lo + ratio * (hi - lo)
    fa, fb = objective(a), objective(b)
    for _ in range(70):
        if fa < fb:
            hi, b, fb = b, a, fa; a = hi - ratio * (hi - lo); fa = objective(a)
        else:
            lo, a, fa = a, b, fb; b = lo + ratio * (hi - lo); fb = objective(b)
    best = min((0., (lo + hi) / 2), key=objective)
    return math.exp(best)


def metric_rows(records, temperatures=None):
    result = []
    for r in records:
        temperature = (temperatures or {}).get(r["question_type"], 1.)
        lp = logprob(r["logits"], temperature); p = np.exp(lp); gold = r["gold"]
        target = np.zeros(len(p)); target[gold] = 1.
        result.append({**r, "correct": int(np.argmax(lp) == gold), "nll": -float(lp[gold]),
                       "brier": float(np.square(p - target).sum()), "confidence": float(p.max())})
    return result


def summary(records):
    sources = {}
    for source in sorted({r["source"] for r in records}):
        selected = [r for r in records if r["source"] == source]
        sources[source] = {"n": len(selected), **{m: float(np.mean([r[m] for r in selected])) for m in ("correct", "nll", "brier")}}
    times = np.array([r["milliseconds"] for r in records])
    ece = 0.
    for i in range(15):
        group = [r for r in records if i / 15 <= r["confidence"] <= (i + 1) / 15 and (i == 0 or r["confidence"] > i / 15)]
        if group: ece += len(group) / len(records) * abs(np.mean([r["confidence"] - r["correct"] for r in group]))
    return {"rows": len(records), "sources": sources,
            "macro_accuracy": float(np.mean([s["correct"] for s in sources.values()])),
            "macro_nll": float(np.mean([s["nll"] for s in sources.values()])),
            "macro_brier": float(np.mean([s["brier"] for s in sources.values()])), "ece15": float(ece),
            "milliseconds": {"median": float(np.median(times)), "mean": float(times.mean()), "p80": float(np.quantile(times, .8)), "p95": float(np.quantile(times, .95))},
            "mean_tokens": float(np.mean([r["input_tokens"] for r in records])),
            "repeat_applied": sum(r["repeat_applied"] for r in records)}


def group_bootstrap(records, rng, draws=BOOTSTRAPS):
    # Same deterministic call order gives the identical paired draw in each policy.
    by_source = defaultdict(lambda: defaultdict(list))
    for r in records: by_source[r["source"]][r["group"]].append(r)
    result = np.zeros((draws, 3))
    for source in sorted(by_source):
        groups = by_source[source]
        values = np.array([[sum(r[m] for r in groups[g]) for m in ("correct", "nll", "brier")] + [len(groups[g])] for g in sorted(groups)])
        weights = rng.multinomial(len(values), np.full(len(values), 1 / len(values)), size=draws)
        sums = weights @ values
        result += sums[:, :3] / sums[:, 3:4] / len(by_source)
    return result


def analyze(root):
    records = rows(root / "predictions.jsonl"); reports = {}; comparisons = {}; temperatures = {}
    for modality in ("text", "vision"):
        reports[modality] = {}; boot = {}
        for policy in POLICIES:
            subset = [r for r in records if r["modality"] == modality and r["policy"] == policy]
            calibration = [r for r in subset if r["split"] == "calibration"]
            ts = {kind: fit_temperature([r for r in calibration if r["question_type"] == kind]) for kind in sorted({r["question_type"] for r in subset})}
            temperatures[modality + ":" + policy] = ts
            raw = metric_rows(subset); calibrated = metric_rows(subset, ts)
            assert [r["prediction"] for r in raw] == [r["prediction"] for r in calibrated]
            reports[modality][policy] = {}
            for name, data in (("raw", raw), ("calibrated", calibrated)):
                evaluation = [r for r in data if r["split"] == "evaluation"]
                reports[modality][policy][name] = summary(evaluation)
                reports[modality][policy][name + "_all_rows_descriptive"] = summary(data)
                boot[policy, name] = group_bootstrap(evaluation, np.random.default_rng(202611042))
        comparisons[modality] = {}
        for name in ("raw", "calibrated"):
            for left, right in (("header", "base"), ("repeat", "base"), ("header_repeat", "base"), ("header_repeat", "header"), ("header_repeat", "repeat")):
                a, b = reports[modality][left][name], reports[modality][right][name]
                comparisons[modality][f"{left}-minus-{right}:{name}"] = {
                    "point": [a[m] - b[m] for m in ("macro_accuracy", "macro_nll", "macro_brier")],
                    "ci95": np.quantile(boot[left, name] - boot[right, name], [.025, .975], axis=0).tolist()}
            delta = boot["header_repeat", name] - boot["header", name] - boot["repeat", name] + boot["base", name]
            comparisons[modality]["interaction:" + name] = {"ci95": np.quantile(delta, [.025, .975], axis=0).tolist()}
    dump(root / "temperatures.json", temperatures)
    dump(root / "report.json", {"complete": True, "evaluation": reports, "comparisons": comparisons,
                               "temperatures": temperatures, "bootstrap_draws": BOOTSTRAPS, "plan_sha256": sha(root / "plan.json")})


def verify(root):
    plan = read(root / "plan.json")
    for k in ("source_sha256", "model_sha256"):
        for p, digest in plan[k].items(): assert sha(Path(p)) == digest, p
    assert sha(root / "examples.jsonl") == plan["examples_sha256"]
    examples = {r["id"]: r for r in rows(root / "examples.jsonl")}; records = rows(root / "predictions.jsonl")
    assert len(records) == len(examples) * 4 and len({(r["id"], r["policy"]) for r in records}) == len(records)
    image_checks = {}
    for row in examples.values():
        for path, digest in zip(row["images"], row["image_sha256"]):
            if path not in image_checks: image_checks[path] = sha(Path(path))
            assert image_checks[path] == digest
    runner = Runner(); sample = []; checked = 0
    # Replay one fixed ID per source/modality under all policies. Full prediction
    # algebra/labels/coverage are audited for all records, model replay is sampled.
    picked = {}
    for row in examples.values(): picked.setdefault((row["modality"], row["source"]), row["id"])
    from decision_index.vision import images as vimages
    for r in records:
        e = examples[r["id"]]
        assert r["gold"] == e["gold"] and r["source"] == e["source"] and r["split"] == e["split"] and r["group"] == e["group"]
        assert r["prediction"] == int(np.argmax(r["logits"]))
        assert len(r["logits"]) == len(fmt.options(e["question"])[0])
        assert r["repeat_applied"] == (r["eligible"] and r["policy"] in ("repeat", "header_repeat"))
        if r["id"] in picked.values():
            pictures = [vimages.pil(p, max_pixels=vimages.MAX_PIXELS).convert("RGB") for p in e["images"]]
            new = runner.predict(e, pictures, r["policy"], r["eligible"])
            assert new["prompt_sha256"] == r["prompt_sha256"] and new["input_tokens"] == r["input_tokens"]
            assert new["prediction"] == r["prediction"]
            assert max(abs(a - b) for a, b in zip(new["logits"], r["logits"])) < 1e-4
            sample.append((r["id"], r["policy"]))
        checked += 1
    del runner
    # Independent NumPy aggregate from original stored logits, without metric_rows.
    report = read(root / "report.json")
    for modality in ("text", "vision"):
        for policy in POLICIES:
            for version in ("raw", "calibrated"):
                sums = defaultdict(lambda: [0., 0., 0])
                for r in records:
                    if (r["modality"], r["policy"], r["split"]) != (modality, policy, "evaluation"): continue
                    t = 1. if version == "raw" else report["temperatures"][modality + ":" + policy][r["question_type"]]
                    x = np.array(r["logits"], dtype=float) / t; maximum = x.max()
                    nll = maximum + np.log(np.exp(x - maximum).sum()) - x[r["gold"]]
                    s = sums[r["source"]]; s[0] += int(np.argmax(x) == r["gold"]); s[1] += nll; s[2] += 1
                accuracy = np.mean([s[0] / s[2] for s in sums.values()]); nll = np.mean([s[1] / s[2] for s in sums.values()])
                expected = report["evaluation"][modality][policy][version]
                assert abs(accuracy - expected["macro_accuracy"]) < 1e-12 and abs(nll - expected["macro_nll"]) < 1e-10
    dump(root / "verification.json", {"complete": True, "predictions_algebra_checked": checked,
            "model_replay_samples": sample, "image_hashes_checked": len(image_checks), "report_sha256": sha(root / "report.json"),
            "scope": "all source/model/input/image hashes, coverage, gold, repeat policy, independent accuracy/NLL aggregates; model replay one fixed example/source x4",
            "limits": "model replay is sampled; bootstrap/temperature optimization not independently reimplemented; no independent hardware reproduction"})


def publish(root):
    project = Path("wiki/project"); report = read(root / "report.json")
    for src, dst in (("report.json", "inference-policy-results.json"), ("verification.json", "inference-policy-verification.json")):
        shutil.copyfile(root / src, project / dst)
    lines = ["## 결과", "", "최종 비교용 evaluation 그룹만의 출처 macro 평균이다. 정확도는%, NLL은 낮을수록 좋다.", "",
             "| 입력 | 정책 | 정확도 | NLL | 보정 후 NLL | median ms | 평균 토큰 | 반복 적용 |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for modality in ("text", "vision"):
        for policy in POLICIES:
            a = report["evaluation"][modality][policy]["raw"]; b = report["evaluation"][modality][policy]["calibrated"]
            lines.append(f"| {modality} | {policy} | {a['macro_accuracy']*100:.2f} | {a['macro_nll']:.3f} | {b['macro_nll']:.3f} | {a['milliseconds']['median']:.1f} | {a['mean_tokens']:.1f} | {a['repeat_applied']}/{a['rows']} |")
    lines += ["", "### 기본 대비 차이", "", "구간은 출처 안의 입력 그룹을 함께 재추출한 paired bootstrap5000회다. 고정된 출처와 모델의 불확실성이며 seed/새 벤치마크의 불확실성을 포함하지 않는다.", "",
              "| 입력 | 비교 | 정확도 차이%p [95% 구간] | NLL 차이 [95% 구간] |", "|---|---|---:|---:|"]
    for modality in ("text", "vision"):
        for policy in POLICIES[1:]:
            value = report["comparisons"][modality][policy + "-minus-base:raw"]; p = value["point"]; lo, hi = value["ci95"]
            lines.append(f"| {modality} | {policy}−base | {p[0]*100:+.2f} [{lo[0]*100:+.2f},{hi[0]*100:+.2f}] | {p[1]:+.3f} [{lo[1]:+.3f},{hi[1]:+.3f}] |")
    page = project / "inference-policy.md"; content = page.read_text(); assert "<!-- POLICY_RESULTS -->" in content
    page.write_text(content.replace("**실행 중.**", "**평가·검산 완료.**").replace("<!-- POLICY_RESULTS -->", "\n".join(lines)))


def run(root):
    plan = read(root / "plan.json")
    for kind in ("source_sha256", "model_sha256"):
        for p, digest in plan[kind].items(): assert sha(Path(p)) == digest
    examples = rows(root / "examples.jsonl"); path = root / "predictions.jsonl"
    existing = rows(path) if path.exists() else []; completed = {(r["id"], r["policy"]) for r in existing}
    assert len(completed) == len(existing)
    runner = Runner(); dump(root / "environment.json", {"torch": torch.__version__, "device": torch.cuda.get_device_name(), "matmul_precision": "highest", "plan_sha256": sha(root / "plan.json")})
    from decision_index.vision import images as vimages
    start = time.monotonic(); warmed = set()
    with path.open("a", buffering=1) as handle:
        for index, row in enumerate(examples):
            if all((row["id"], p) in completed for p in POLICIES): continue
            io_start = time.perf_counter()
            pictures = [vimages.pil(p, max_pixels=vimages.MAX_PIXELS).convert("RGB") for p in row["images"]]
            io_ms = (time.perf_counter() - io_start) * 1000
            once, _, _, _ = runner.inputs(row, pictures, "base", False)
            baseline_tokens = int(once["input_ids"].shape[1]); eligible = baseline_tokens <= MAX_ONCE
            assert baseline_tokens <= MAX_INPUT, (row["id"], baseline_tokens)
            if row["modality"] not in warmed:
                for policy in POLICIES: runner.predict(row, pictures, policy, eligible)
                warmed.add(row["modality"])
            order = POLICIES[index % 4:] + POLICIES[:index % 4]
            for policy in order:
                if (row["id"], policy) in completed: continue
                value = runner.predict(row, pictures, policy, eligible)
                record = {k: row[k] for k in ("id", "modality", "source", "gold", "group", "split")}
                record.update({"policy": policy, "question_type": row["question"]["type"], "baseline_tokens": baseline_tokens,
                               "eligible": eligible, "image_disk_ms": io_ms, **value})
                handle.write(json.dumps(record, ensure_ascii=False) + "\n"); completed.add((row["id"], policy))
            if index % 100 == 0 or index == len(examples) - 1:
                status = {"state": "inference", "predictions_complete": len(completed), "predictions_total": len(examples) * 4,
                          "last_modality": row["modality"], "seconds_this_attempt": time.monotonic() - start}
                dump(root / "status.json", status); print("PROGRESS", status, flush=True)
    del runner; torch.cuda.empty_cache()
    for name, action in (("analyzing", analyze), ("verifying", verify), ("saving_wiki", publish)):
        dump(root / "status.json", {"state": name, "predictions_complete": len(completed)}); action(root)
    manifest = {str(p.relative_to(root)): sha(p) for p in sorted(root.rglob("*")) if p.is_file() and p.name not in ("checksums.json", "status.json") and p.suffix != ".log"}
    dump(root / "checksums.json", manifest)
    dump(root / "status.json", {"state": "complete", "verified": True, "predictions_complete": len(completed),
                              "seconds_this_attempt": time.monotonic() - start, "wiki": "wiki/project/inference-policy.md"})
    print("COMPLETE", root, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("command", choices=("prepare", "run", "analyze", "verify", "publish"))
    parser.add_argument("--out", type=Path, default=ROOT); args = parser.parse_args()
    torch.set_num_threads(4); torch.set_float32_matmul_precision("highest")
    try: globals()[args.command](args.out)
    except Exception:
        if args.out.exists(): dump(args.out / "status.json", {"state": "failed", "traceback": traceback.format_exc()})
        raise


if __name__ == "__main__": main()
