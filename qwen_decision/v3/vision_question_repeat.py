"""Single-image question/option repetition control on frozen Crystal inputs.

Preserves inference-policy-v1. Reruns the baseline and two one-image policies;
the old full-image repetition is a frozen reference, with replay and drift checks.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
from pathlib import Path
import shutil
import time
import traceback

import numpy as np
import torch

from qwen_decision.v3 import format as fmt, inference_policy as previous

PARENT = previous.ROOT
ROOT = Path("runs/vision-question-repeat-v1")
POLICIES = ("base", "q_repeat", "q_echo")
ALL_POLICIES = (*POLICIES, "image_repeat")


def question_block(question, codes):
    lines = ["Question:", fmt.describe(question.get("instructions") or fmt.DEFAULT_INSTRUCTIONS), "", "Options:"]
    lines += [f"{code}: {text}" for code, text in zip(codes, fmt.options(question)[1])]
    return "\n".join(lines + ["", fmt.CLOSING_LINE])


def render(tokenizer, state, question, codes, policy, enabled):
    assert policy in POLICIES
    applied = enabled and policy != "base"
    user = fmt.user_prompt(state, question, codes, "echo" if applied and policy == "q_echo" else "d3")
    if applied and policy == "q_repeat":
        block = question_block(question, codes)
        assert user.endswith(block)
        user += previous.READ_AGAIN + block
    text = tokenizer.apply_chat_template(
        [{"role": "system", "content": fmt.SYSTEM_PROMPT}, {"role": "user", "content": user}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False)
    return text, applied


class Runner(previous.Runner):
    def inputs(self, row, pictures, policy, repeat_enabled):
        assert row["modality"] == "vision"
        codes, ids = self.codes["vision"]; count = len(fmt.options(row["question"])[0])
        text, applied = render(self.vision_tok, previous.IMAGE * len(pictures), row["question"], codes[:count], policy, repeat_enabled)
        # No image is copied in any new policy, including when question repetition is on.
        encoded = self.processor(text=[text], images=pictures, return_tensors="pt")
        assert len(encoded["image_grid_thw"]) == len(pictures)
        return encoded, ids[:count], applied, hashlib.sha256(text.encode()).hexdigest()


def parent_records(policy):
    return [r for r in previous.rows(PARENT / "predictions.jsonl") if r["modality"] == "vision" and r["policy"] == policy]


def check_plan(root):
    plan = previous.read(root / "plan.json")
    for field in ("source_sha256", "model_sha256", "parent_sha256"):
        for path, digest in plan[field].items(): assert previous.sha(Path(path)) == digest, path
    assert previous.sha(root / "examples.jsonl") == plan["examples_sha256"]
    return plan


def prepare(root):
    assert not root.exists(), "Preserve the old run"
    parent = previous.read(PARENT / "plan.json")
    assert previous.read(PARENT / "status.json")["state"] == "complete"
    for field in ("source_sha256", "model_sha256"):
        for path, digest in parent[field].items(): assert previous.sha(Path(path)) == digest
    selected = [r for r in previous.rows(PARENT / "examples.jsonl") if r["modality"] == "vision"]
    assert len(selected) == 2184
    root.mkdir(parents=True)
    with (root / "examples.jsonl").open("x") as handle:
        import json
        for row in selected: handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    shutil.copyfile(__file__, root / "experiment-source.py")
    plan = {"source_sha256": {str(Path(__file__)): previous.sha(Path(__file__)), **parent["source_sha256"]},
            "model_sha256": parent["model_sha256"], "parent_sha256": {str(PARENT / n): previous.sha(PARENT / n)
                  for n in ("plan.json", "report.json", "verification.json", "predictions.jsonl", "examples.jsonl")},
            "examples_sha256": previous.sha(root / "examples.jsonl"), "policies": list(POLICIES),
            "frozen_reference": "image_repeat is the original repeat prediction and timing from parent; baseline is rerun for current timing and drift",
            "input_order": {"base": "image,QO", "q_repeat": "image,QO,READ_AGAIN,QO", "q_echo": "QO,image,QO",
                            "image_repeat": "image,QO,READ_AGAIN,image,QO (frozen old run)"},
            "primary": "q_repeat vs image_repeat removes only the second State/image block, preserving other text; also compare against freshly rerun base",
            "secondary": "q_echo tests QO before and after the one image; its native echo template has one closing instruction, unlike the two in q_repeat",
            "eligibility": "reuse parent baseline <=3000 expanded token mask; all new conditions retain exactly the original image inputs once",
            "counts": dict(Counter(r["source"] for r in selected)), "splits": dict(Counter(r["split"] for r in selected)),
            "temperature": parent["temperature"], "metrics": parent["metrics"], "latency": parent["latency"],
            "bootstrap_draws": previous.BOOTSTRAPS, "neural_training": False,
            "stop": "2184 fixed vision rows x three new predictions; no exclusions, no tuning on evaluation labels",
            "scope": "post-hoc control after observing original repetition failure; same already-examined split and one model; no new text inference; no generalization or double-counting mechanism proof"}
    previous.dump(root / "plan.json", plan)
    previous.dump(root / "status.json", {"state": "prepared", "predictions_total": len(selected) * len(POLICIES)})
    print("PREPARED", plan["counts"], plan["splits"], flush=True)


def combined_records(root):
    result = previous.rows(root / "predictions.jsonl")
    return result + [{**r, "policy": "image_repeat", "cached_reference": True} for r in parent_records("repeat")]


def analyze(root):
    records = combined_records(root); reports = {}; temperatures = {}; boot = {}
    for policy in ALL_POLICIES:
        selected = [r for r in records if r["policy"] == policy]
        calibration = [r for r in selected if r["split"] == "calibration"]
        ts = {kind: previous.fit_temperature([r for r in calibration if r["question_type"] == kind])
              for kind in sorted({r["question_type"] for r in selected})}
        temperatures[policy] = ts
        reports[policy] = {}
        for name, data in (("raw", previous.metric_rows(selected)), ("calibrated", previous.metric_rows(selected, ts))):
            evaluation = [r for r in data if r["split"] == "evaluation"]
            reports[policy][name] = previous.summary(evaluation)
            reports[policy][name + "_all_rows_descriptive"] = previous.summary(data)
            boot[policy, name] = previous.group_bootstrap(evaluation, np.random.default_rng(202611042))
    comparisons = {}
    for left, right in (("q_repeat", "base"), ("q_repeat", "image_repeat"), ("q_echo", "base"),
                        ("q_echo", "image_repeat"), ("q_echo", "q_repeat")):
        for version in ("raw", "calibrated"):
            a, b = reports[left][version], reports[right][version]
            comparisons[f"{left}-minus-{right}:{version}"] = {
                "point": [a[m] - b[m] for m in ("macro_accuracy", "macro_nll", "macro_brier")],
                "ci95": np.quantile(boot[left, version] - boot[right, version], [.025, .975], axis=0).tolist()}
    original = {r["id"]: r for r in parent_records("base")}
    base = [r for r in records if r["policy"] == "base"]
    drift = {"prediction_differences": sum(r["prediction"] != original[r["id"]]["prediction"] for r in base),
             "max_logit_difference": max(max(abs(a - b) for a, b in zip(r["logits"], original[r["id"]]["logits"])) for r in base)}
    previous.dump(root / "report.json", {"complete": True, "evaluation": reports, "comparisons": comparisons,
                 "temperatures": temperatures, "baseline_drift": drift, "plan_sha256": previous.sha(root / "plan.json")})


def verify(root):
    plan = check_plan(root); examples = {r["id"]: r for r in previous.rows(root / "examples.jsonl")}
    records = previous.rows(root / "predictions.jsonl"); original = {r["id"]: r for r in parent_records("base")}
    assert len(records) == 3 * len(examples) and len({(r["id"], r["policy"]) for r in records}) == len(records)
    image_hashes = {}
    for row in examples.values():
        for path, expected in zip(row["images"], row["image_sha256"]):
            if path not in image_hashes: image_hashes[path] = previous.sha(Path(path))
            assert image_hashes[path] == expected
    for r in records:
        row, base = examples[r["id"]], original[r["id"]]
        for key in ("gold", "group", "split", "source"): assert r[key] == row[key]
        assert r["policy"] in POLICIES and r["prediction"] == int(np.argmax(r["logits"]))
        assert r["image_copies"] == 1 and r["image_tokens"] == base["image_tokens"]
        assert r["eligible"] == base["eligible"]
        assert r["repeat_applied"] == (r["eligible"] and r["policy"] != "base")
        assert r["baseline_tokens"] == base["baseline_tokens"]
        if r["policy"] == "base": assert r["prompt_sha256"] == base["prompt_sha256"]
    # Every aggregate is independently recomputed from the stored logits.
    report = previous.read(root / "report.json")
    for policy in ALL_POLICIES:
        for version in ("raw", "calibrated"):
            by = defaultdict(lambda: [0., 0., 0])
            for r in combined_records(root):
                if r["policy"] != policy or r["split"] != "evaluation": continue
                temperature = 1. if version == "raw" else report["temperatures"][policy][r["question_type"]]
                z = np.array(r["logits"], dtype=float) / temperature
                nll = z.max() + np.log(np.exp(z - z.max()).sum()) - z[r["gold"]]
                values = by[r["source"]]; values[0] += int(z.argmax() == r["gold"]); values[1] += nll; values[2] += 1
            expected = report["evaluation"][policy][version]
            assert abs(np.mean([x[0] / x[2] for x in by.values()]) - expected["macro_accuracy"]) < 1e-12
            assert abs(np.mean([x[1] / x[2] for x in by.values()]) - expected["macro_nll"]) < 1e-10
    from decision_index.vision import images as vimages
    runner = Runner(); old_runner = previous.Runner.__new__(previous.Runner); old_runner.__dict__.update(runner.__dict__)
    picked = {}
    for r in examples.values(): picked.setdefault(r["source"], r["id"])
    replayed = []
    for r in combined_records(root):
        if r["id"] not in picked.values(): continue
        row = examples[r["id"]]
        pictures = [vimages.pil(p, max_pixels=vimages.MAX_PIXELS).convert("RGB") for p in row["images"]]
        active, policy = (old_runner, "repeat") if r["policy"] == "image_repeat" else (runner, r["policy"])
        current = active.predict(row, pictures, policy, r["eligible"])
        assert current["prompt_sha256"] == r["prompt_sha256"]
        assert current["prediction"] == r["prediction"]
        assert max(abs(a - b) for a, b in zip(current["logits"], r["logits"])) < 1e-4
        replayed.append((r["id"], r["policy"]))
    previous.dump(root / "verification.json", {"complete": True, "predictions_algebra_checked": len(records),
                  "frozen_reference_predictions": len(examples), "image_hashes_checked": len(image_hashes),
                  "model_replay_samples": replayed, "baseline_drift": report["baseline_drift"],
                  "report_sha256": previous.sha(root / "report.json"),
                  "scope": "all source/model/parent/example/image hashes; all new image-token counts unchanged; all gold, coverage, raw/calibrated aggregate accuracy/NLL; sampled native model replay including cached full repetition",
                  "limits": "model replay sampled; bootstrap/temperature solver not independently repeated; cached full-image timing is from earlier run"})


def publish(root):
    report = previous.read(root / "report.json"); project = Path("wiki/project")
    for source, target in (("report.json", "vision-question-repeat-results.json"), ("verification.json", "vision-question-repeat-verification.json")):
        shutil.copyfile(root / source, project / target)
    lines = ["## 결과", "", "평가용1737문항의8출처 macro 평균이다. calibration447문항은 이 표에 포함하지 않는다.", "",
             "| 정책 | 정확도% | NLL | 보정 후 NLL | median ms | mean ms | 평균 입력 토큰 |", "|---|---:|---:|---:|---:|---:|---:|"]
    for policy in ALL_POLICIES:
        a, b = report["evaluation"][policy]["raw"], report["evaluation"][policy]["calibrated"]
        lines.append(f"| {policy} | {a['macro_accuracy']*100:.2f} | {a['macro_nll']:.3f} | {b['macro_nll']:.3f} | {a['milliseconds']['median']:.1f} | {a['milliseconds']['mean']:.1f} | {a['mean_tokens']:.1f} |")
    lines += ["", "`image_repeat`는 보존한 이전 실행의 점수·시간이다. 나머지는 이번 실행이다.",
              f"새 기본 입력과 이전 기본 입력의 예측 차이는{report['baseline_drift']['prediction_differences']}개, 최대logit 차이는{report['baseline_drift']['max_logit_difference']:.3g}다.",
              "", "| 비교 | 정확도 차이%p [95% 구간] | NLL 차이 [95% 구간] |", "|---|---:|---:|"]
    for key, value in report["comparisons"].items():
        if not key.endswith(":raw"): continue
        point, (lo, hi) = value["point"], value["ci95"]
        lines.append(f"| {key[:-4]} | {point[0]*100:+.2f} [{lo[0]*100:+.2f},{hi[0]*100:+.2f}] | {point[1]:+.3f} [{lo[1]:+.3f},{hi[1]:+.3f}] |")
    page = project / "vision-question-repeat.md"; content = page.read_text(); assert "<!-- Q_REPEAT_RESULTS -->" in content
    page.write_text(content.replace("**실행 중.**", "**평가·검산 완료.**").replace("<!-- Q_REPEAT_RESULTS -->", "\n".join(lines)))


def run(root):
    check_plan(root); examples = previous.rows(root / "examples.jsonl")
    old = {r["id"]: r for r in parent_records("base")}; path = root / "predictions.jsonl"
    existing = previous.rows(path) if path.exists() else []; complete = {(r["id"], r["policy"]) for r in existing}
    assert len(complete) == len(existing)
    runner = Runner()
    previous.dump(root / "environment.json", {"torch": torch.__version__, "device": torch.cuda.get_device_name(), "matmul_precision": "highest"})
    from decision_index.vision import images as vimages
    import json
    started = time.monotonic(); warm = False
    with path.open("a", buffering=1) as handle:
        for i, row in enumerate(examples):
            if all((row["id"], policy) in complete for policy in POLICIES): continue
            io_start = time.perf_counter()
            pictures = [vimages.pil(p, max_pixels=vimages.MAX_PIXELS).convert("RGB") for p in row["images"]]
            io_ms = (time.perf_counter() - io_start) * 1000
            reference = old[row["id"]]; eligible = reference["eligible"]
            if not warm:
                for policy in POLICIES: runner.predict(row, pictures, policy, eligible)
                warm = True
            for policy in POLICIES[i % 3:] + POLICIES[:i % 3]:
                if (row["id"], policy) in complete: continue
                value = runner.predict(row, pictures, policy, eligible)
                assert value["image_tokens"] == reference["image_tokens"]
                record = {k: row[k] for k in ("id", "modality", "source", "gold", "group", "split")}
                record.update({"policy": policy, "question_type": row["question"]["type"], "eligible": eligible,
                               "baseline_tokens": reference["baseline_tokens"], "image_copies": 1, "image_disk_ms": io_ms, **value})
                handle.write(json.dumps(record, ensure_ascii=False) + "\n"); complete.add((row["id"], policy))
            if i % 100 == 0 or i == len(examples) - 1:
                status = {"state": "inference", "predictions_complete": len(complete), "predictions_total": len(examples) * 3,
                          "seconds": time.monotonic() - started}
                previous.dump(root / "status.json", status); print("PROGRESS", status, flush=True)
    del runner; torch.cuda.empty_cache()
    for name, action in (("analyzing", analyze), ("verifying", verify), ("saving_wiki", publish)):
        previous.dump(root / "status.json", {"state": name, "predictions_complete": len(complete)}); action(root)
    manifest = {str(p.relative_to(root)): previous.sha(p) for p in sorted(root.rglob("*")) if p.is_file() and p.suffix != ".log" and p.name not in ("checksums.json", "status.json")}
    previous.dump(root / "checksums.json", manifest)
    previous.dump(root / "status.json", {"state": "complete", "verified": True, "predictions_complete": len(complete),
                                        "seconds": time.monotonic() - started, "wiki": "wiki/project/vision-question-repeat.md"})
    print("COMPLETE", root, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run", "analyze", "verify", "publish")); parser.add_argument("--out", type=Path, default=ROOT)
    args = parser.parse_args(); torch.set_num_threads(4); torch.set_float32_matmul_precision("highest")
    try: globals()[args.command](args.out)
    except Exception:
        if args.out.exists(): previous.dump(args.out / "status.json", {"state": "failed", "traceback": traceback.format_exc()})
        raise


if __name__ == "__main__": main()
