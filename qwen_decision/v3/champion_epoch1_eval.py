"""Evaluate the immutable first-epoch adapter and original base on identical internal dev rows.

Run in the separate epoch1 systemd unit. Its ExecStopPost resumes the untouched full-training pipeline.
This does not change checkpoint selection, training inputs, or any public-evaluation outputs.
"""
from __future__ import annotations

import gc
import json
import math
from pathlib import Path
import statistics
import time
import traceback

import torch
from safetensors.torch import load_file

from qwen_decision.v3 import champion_full_train as fit, champion_audit as audit
from qwen_decision.v3 import train, vision_train as vt

ROOT = fit.ROOT / "epoch1-eval"
CHECKPOINT_STEP = 70659


def state(stage, **extra):
    train.atomic_json(ROOT / "status.json", {"stage": stage, "updated_unix": time.time(), **extra})


def worker_init(_):
    fit.bind()
    torch.set_num_threads(1)


class DevChunks(vt.EvalChunks):
    def __init__(self, entries, token_budget, processor_path):
        self.processor_path = str(processor_path)
        vt.PROCESSOR = Path(self.processor_path)
        super().__init__(entries, token_budget)

    def __getitem__(self, i):
        if self.encoder is None:
            vt.PROCESSOR = Path(self.processor_path)
            self.encoder = vt.Encoder(max_pixels=fit.CFG.max_pixels)
        return super().__getitem__(i)


def dev_rows(path):
    rows = vt.gold_text(list(audit.rows(path)))
    return [{**r, "images": [str((fit.data.MIX / p).resolve()) for p in r["images"]]}
            if r["modality"] == "image" else r for r in rows]


def load_arm(arm, request):
    if arm == "crystal":
        manifest = audit.read(ROOT / "crystal-model.json")
        for path, sha in manifest["files"].items():
            assert audit.sha(path) == sha
        from qwen_decision.v3 import modeling
        modeling.compile_elementwise()
        vt.use_vision_attention("varlen")
        model = modeling.load(Path(manifest["path"]), attention="noncausal", deltanet="causal",
                              dtype=torch.bfloat16, kernel="varlen")
        model.model.visual.to("cuda")
        train.set_checkpointing(model, False)
        model.requires_grad_(False)
        model.eval()
        return model
    model = fit.legacy.load_model()
    params = {n: p for n, p in model.named_parameters() if ".lora_" in n}
    assert len(params) == 372
    if arm == "epoch1":
        checkpoint = request["epoch1_checkpoint"]
        folder = Path(checkpoint["directory"])
        for name, sha in checkpoint["files"].items():
            assert audit.sha(folder / name) == sha
        stored = load_file(str(folder / "adapter.safetensors"), device="cpu")
        adapter = {n: t for n, t in stored.items() if not n.startswith("master.")}
        assert set(adapter) == set(params)
        assert all(torch.isfinite(t).all() for t in stored.values())
        assert sum(bool(torch.count_nonzero(t)) for n, t in adapter.items() if n.endswith(".lora_b")) == 186
        # BF16 adapter values are the actual forward weights, not unrounded FP32 optimizer masters.
        with torch.no_grad():
            for name, param in params.items():
                assert param.shape == adapter[name].shape and param.dtype == adapter[name].dtype
                param.copy_(adapter[name])
        assert all(torch.equal(p.detach().cpu(), adapter[n]) for n, p in params.items())
    else:
        assert arm == "original"
        assert all(not torch.count_nonzero(p) for n, p in params.items() if n.endswith(".lora_b"))
    train.set_checkpointing(model, False)
    model.requires_grad_(False)
    model.eval()
    return model


def source_summary(predictions):
    by = {}
    for p in predictions:
        by.setdefault(p["source"], []).append(p)
    sources = {}
    for name, ps in sorted(by.items()):
        accuracy = statistics.mean(p["correct"] for p in ps)
        chance = statistics.mean(1 / p["k"] for p in ps)
        sources[name] = {"rows": len(ps), "accuracy": accuracy,
                         "ce": statistics.mean(p["ce"] for p in ps),
                         "kl": statistics.mean(p["kl"] for p in ps),
                         "chance_accuracy": chance, "skill": (accuracy-chance)/(1-chance)}
    return {"rows": len(predictions), "accuracy": statistics.mean(p["correct"] for p in predictions),
            "ce": statistics.mean(p["ce"] for p in predictions),
            "kl": statistics.mean(p["kl"] for p in predictions),
            "macro_accuracy": statistics.mean(s["accuracy"] for s in sources.values()),
            "sources": sources}


def validate_predictions(rows, predictions):
    """Identity is the position in the immutable dataset; upstream IDs can collide across views."""
    assert len(predictions) == len(rows)
    seen = set()
    for prediction in predictions:
        index = prediction["index"]
        assert isinstance(index, int) and 0 <= index < len(rows) and index not in seen
        seen.add(index)
        row = rows[index]
        assert (prediction["id"], prediction["source"]) == (row["id"], row["source"])
        targets = row["target"] if row["modality"] == "text" else [float(j == row["gold"]) for j in range(len(row["options"]))]
        assert prediction["k"] == len(targets) == len(prediction["logits"])
        logits = torch.tensor(prediction["logits"], dtype=torch.float32)
        target = torch.tensor(targets, dtype=torch.float32)
        assert torch.isfinite(logits).all()
        pred = int(logits.argmax())
        assert pred == prediction["pred"]
        assert prediction["correct"] == bool(target[pred] >= target.max()-1e-6)
        ce = float(train.soft_ce(logits, target))
        kl = ce-float(train.entropy(target))
        assert math.isclose(prediction["ce"], ce, abs_tol=2e-5, rel_tol=2e-5)
        assert math.isclose(prediction["kl"], kl, abs_tol=2e-5, rel_tol=2e-5)
    assert seen == set(range(len(rows)))


@torch.inference_mode()
def evaluate(model, arm, name, rows):
    target = ROOT / arm
    target.mkdir(exist_ok=True)
    done = target / (name + ".summary.json")
    if done.exists():
        summary = audit.read(done)
        assert summary["rows"] == len(rows)
        return summary
    prediction_path = target / (name + ".jsonl")
    repair_path = ROOT / "row-index-repair.json"
    inherited = audit.read(repair_path)["inherited_completed_predictions"] if repair_path.exists() else {}
    if str(prediction_path) in inherited:
        assert audit.sha(prediction_path) == inherited[str(prediction_path)]
        predictions = list(audit.rows(prediction_path))
        validate_predictions(rows, predictions)
        summary = source_summary(predictions) | {"seconds": None, "reused_verified_complete_predictions": True,
                    "prediction_file_sha256": inherited[str(prediction_path)]}
        train.atomic_json(done, summary)
        return summary
    started = time.monotonic()
    entries = [("mix", r) for r in rows]
    processor = Path(audit.read(ROOT / "crystal-model.json")["path"]) if arm == "crystal" else fit.data.BASE
    loader = torch.utils.data.DataLoader(DevChunks(entries, fit.CFG.token_budget, processor), batch_size=None,
        num_workers=fit.CFG.workers, pin_memory=True, prefetch_factor=2, worker_init_fn=worker_init,
        multiprocessing_context="spawn", generator=torch.Generator().manual_seed(fit.data.SEED))
    output = [None] * len(rows)
    visited = 0
    with (target / (name + ".jsonl")).open("w", buffering=1) as log:
        for chunk, groups in enumerate(loader):
            for cpu in groups:
                z = vt.packed_logits(model, vt.to_device(cpu)).float().cpu()
                assert torch.isfinite(z[cpu["code_mask"]]).all()
                targets = cpu["targets"]
                ce = train.soft_ce(z, targets)
                entropy = train.entropy(targets)
                probabilities = z.softmax(-1)
                pred = z.argmax(-1)
                for j, index in enumerate(cpu["index"]):
                    assert output[index] is None
                    k = int(cpu["code_mask"][j].sum())
                    t = targets[j, :k]
                    record = {"id": rows[index]["id"], "source": rows[index]["source"],
                        "index": index, "k": k, "pred": int(pred[j]),
                        "correct": bool(t[pred[j]] >= t.max()-1e-6),
                        "ce": float(ce[j]), "kl": float(ce[j]-entropy[j]),
                        "confidence": float(probabilities[j, pred[j]]),
                        "logits": z[j, :k].tolist()}
                    assert math.isfinite(record["ce"]) and record["kl"] >= -1e-5
                    output[index] = record
                    log.write(json.dumps(record) + "\n")
                    visited += 1
            if chunk % 8 == 0 or visited == len(rows):
                seconds = time.monotonic()-started
                state("evaluating", arm=arm, dataset=name, visited=visited, rows=len(rows),
                      seconds=seconds, estimated_remaining_seconds=seconds*(len(rows)-visited)/max(1, visited))
                print("EVAL", arm, name, visited, len(rows), round(seconds, 1), flush=True)
    assert visited == len(rows) and all(p is not None for p in output)
    assert {p["index"] for p in output} == set(range(len(rows)))
    summary = source_summary(output) | {"seconds": time.monotonic()-started}
    train.atomic_json(done, summary)
    print("COMPLETE", arm, name, {k: v for k, v in summary.items() if k != "sources"}, flush=True)
    return summary


def main():
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("highest")
    fit.bind()
    state("verifying")
    plan = fit.verify()
    request = audit.read(ROOT / "request.json")
    assert plan["steps_per_epoch"] == CHECKPOINT_STEP == request["epoch1_checkpoint"]["step"]
    prepared = audit.read(fit.ROOT / "prepared.json")
    sets = {name: dev_rows(path) for name, path in prepared["evaluation"].items()}
    crystal = audit.read(ROOT / "crystal-model.json")
    from transformers import AutoTokenizer
    from qwen_decision.v3 import format as fmt
    native = AutoTokenizer.from_pretrained(crystal["path"], local_files_only=True)
    codes, ids = fmt.answer_codes(native)
    crystal_config = audit.read(Path(crystal["path"]) / "crystal_config.json")
    assert codes == crystal_config["codes"] and ids == crystal_config["token_ids"]
    assert (crystal_config["attention"], crystal_config["layout"], crystal_config["deltanet"]) == ("noncausal", "d3", "causal")
    eval_plan = {"training_plan_sha256": audit.sha(fit.PLAN), "checkpoint": request["epoch1_checkpoint"],
        "implementation_sha256": audit.sha(__file__),
        "datasets": {name: {"path": path, "sha256": audit.sha(path), "rows": len(sets[name])}
                     for name, path in prepared["evaluation"].items()},
        "arms": ["epoch1", "crystal", "original"], "crystal_model": crystal,
        "scope": "internal dev only, no public checkpoint selection; Crystal historical training may overlap these dev rows",
        "attention": "noncausal full-attention, causal DeltaNet, packed varlen",
        "prompt": "d3, stored option order and answer codes, augmentation disabled, dropout disabled; native tokenizer/processor per model (Crystal tokenizer differs from current original Qwen)",
        "metrics": "row/source accuracy and CE/KL; not official benchmark-specific F1/index",
        "row_identity": "immutable dataset row index; upstream IDs may repeat, and no rows are dropped",
        "repair": audit.read(ROOT / "row-index-repair.json") if (ROOT / "row-index-repair.json").exists() else None,
        "max_pixels": fit.CFG.max_pixels, "token_budget": fit.CFG.token_budget}
    if (ROOT / "plan.json").exists():
        assert audit.read(ROOT / "plan.json") == eval_plan
    else:
        train.atomic_json(ROOT / "plan.json", eval_plan)
    result = {}
    for arm in eval_plan["arms"]:
        state("loading", arm=arm)
        model = load_arm(arm, request)
        result[arm] = {}
        for name, rows in sets.items():
            result[arm][name] = evaluate(model, arm, name, rows)
            train.atomic_json(ROOT / "results.json", result)
        del model
        gc.collect()
        torch.cuda.empty_cache()
    differences = {baseline: {name: {metric: result["epoch1"][name][metric]-result[baseline][name][metric]
                         for metric in ("accuracy", "macro_accuracy", "ce", "kl")}
                   for name in sets} for baseline in ("original", "crystal")}
    assert audit.sha(__file__) == eval_plan["implementation_sha256"]
    for info in eval_plan["datasets"].values():
        assert audit.sha(info["path"]) == info["sha256"]
    train.atomic_json(ROOT / "completion.json", {"complete": True, "checkpoint_step": CHECKPOINT_STEP,
        "results": result, "epoch1_minus_original": differences,
        "official_rank": None, "training_resume": "systemd ExecStopPost starts original full-training service"})
    state("complete", checkpoint_step=CHECKPOINT_STEP)


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        state("failed", traceback=traceback.format_exc())
        raise
