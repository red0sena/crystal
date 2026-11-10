"""Measure real untruncated rows with the intended LoRA/packing/checkpointing path.

Probe updates are discarded; this command never launches or resumes the full fit.
Extremely long, unreviewed raw email payloads are catalogued, not blindly run.
"""
from collections import Counter
import gc
from pathlib import Path
import time

import numpy as np
import torch

from qwen_decision.v3 import champion_full_inputs as data, champion_full_train as fit
from qwen_decision.v3 import champion_audit as audit, train, vision_train as vt

ROOT = data.ROOT


def run():
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("highest")
    fit.bind()
    files = [data.MIX/"text-train.jsonl", data.MIX/"phishing-url-train.jsonl"]
    indices = [ROOT/"indices"/f"text-train-{i}.npy" for i in range(2)]
    texts = data.IndexedRows(files, indices)
    images = list(audit.rows(data.MIX/"image-train.jsonl"))
    lengths = np.concatenate([np.load(p, mmap_mode="r")[:, 2] for p in indices])
    schedule = fit.Schedule(texts, images)
    trials = [("joint16text4image", schedule[0])]
    for low, high in [(2048, 4096), (4096, 8192), (8192, 16384), (16384, 32768), (32768, 65536)]:
        found = np.flatnonzero((lengths > low) & (lengths <= high))
        if len(found):
            i = int(found[np.argmax(lengths[found])])
            trials.append((f"text_at_most_{high}", [(texts[i], f"memory:{i}")]))
    extremes = []
    for i in np.flatnonzero(lengths > 65536):
        r = texts[int(i)]
        extremes.append({"index": int(i), "source": r["source"], "id": r["original_id"], "tokens": int(lengths[i]),
                         "exceeds_native_context": int(lengths[i]) > 262144})
    train.seed_all(data.SEED)
    free, total = torch.cuda.mem_get_info()
    model = fit.legacy.load_model()
    model.train()
    opt = train.MasterAdamW(model, 2e-4)
    report = {"gpu": torch.cuda.get_device_name(), "device_total_gib": total/2**30,
              "device_free_before_model_gib": free/2**30,
              "model_and_fp32_masters_allocated_gib": torch.cuda.memory_allocated()/2**30,
              "parameters": sum(p.numel() for p in model.parameters()),
              "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
              "settings": vars(fit.CFG), "trials": [], "extreme_rows_not_tested": extremes,
              "training_started": False, "probe_updates_discarded": True,
              "input_files": {str(p): audit.sha(p) for p in files+indices},
              "implementation": fit.implementation(),
              "limitation": "Successful trials cover actual complete rows up to64k and one real joint batch; not every possible batch. Extreme unreviewed email payloads still prevent an unconditional full-run clearance."}
    for name, rows in trials:
        opt.zero_grad()
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        start = time.monotonic()
        groups = vt.MixSteps([rows], fit.CFG, [])[0]
        lengths_now = [int(b["longest"]) for b in groups]
        record = {"name": name, "rows": len(rows), "tokens": sum(len(b["ids"]) for b in groups),
                  "longest_sequence": max(lengths_now),
                  "sources": dict(Counter(r["source"] for r, _ in rows))}
        try:
            loss_sum = 0.
            for cpu in groups:
                batch = vt.to_device(cpu)
                train.set_checkpointing(model, len(cpu["ids"]) > fit.CFG.checkpoint_above)
                loss = train.soft_ce(vt.packed_logits(model, batch), batch["targets"]).sum()/len(rows)
                assert torch.isfinite(loss)
                loss.backward()
                loss_sum += float(loss.detach())
                del batch, loss
            norm = opt.clip_grad_norm(1.)
            assert np.isfinite(norm)
            opt.step()
            torch.cuda.synchronize()
            record.update(passed=True, loss=loss_sum, grad_norm=norm)
        except torch.OutOfMemoryError as e:
            record.update(passed=False, error="CUDA out of memory", detail=str(e)[:1200])
            opt.zero_grad()
        record.update(seconds=time.monotonic()-start,
                      peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                      peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,
                      device_free_after_gib=torch.cuda.mem_get_info()[0]/2**30)
        report["trials"].append(record)
        train.atomic_json(ROOT/"memory-probe.json", report)
        print("MEMORY_PROBE", record, flush=True)
    report["complete"] = True
    train.atomic_json(ROOT/"memory-probe.json", report)


if __name__ == "__main__":
    import traceback
    try:
        run()
    except Exception:
        train.atomic_json(ROOT/"memory-probe-failure.json", {"traceback": traceback.format_exc()})
        raise
