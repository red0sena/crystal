"""One from-base joint CE pilot, both-modality evaluations, and frozen real-pair rotation diagnosis.

This is an exploratory reference, not a seed-confirmed or official-board result.
The alignment step changes no deployed predictions; its held-out geometry decides
whether a later controlled decision-transfer experiment is warranted.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import gc
import hashlib
import json
import math
from pathlib import Path
import random
import shutil
import statistics
import time
import traceback
from types import SimpleNamespace

import numpy as np
import torch
from safetensors.torch import load_file, save_file
from transformers import AutoProcessor

from qwen_decision.v3 import champion_audit as audit, champion_joint_data as data
from qwen_decision.v3 import format as fmt, lora_experiment as lora, lora_placement_experiment as place
from qwen_decision.v3 import modeling, train, vision_train as vt

ROOT = data.ROOT
PLAN = ROOT / "plan.v2.json"
SEED = data.SEED
EVALS = ("text-dev", "image-dev", "text-diagnostic", "vision-diagnostic")
CFG = SimpleNamespace(**(vt.PRESETS["fast"] | {"workers": 4, "token_budget": 4096,
                                             "checkpoint_above": 4096, "max_pixels": 1638400}))


def configure():
    vt.MIX = data.MIX; vt.PROCESSOR = data.BASE
    vt.LM_TARGETS = place.ARMS["all"][0]
    vt.TEXT_ARGS = SimpleNamespace(**(vars(vt.TEXT_ARGS) | {"pad_probability": 0.}))


def worker_init(_):
    configure(); torch.set_num_threads(2)


def load_model():
    configure(); modeling.compile_elementwise(); vt.use_vision_attention("varlen")
    model = modeling.load(data.BASE, attention="noncausal", deltanet="causal", dtype=torch.bfloat16, kernel="varlen")
    model.model.visual.to("cuda"); vt.add_lora(model, False)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    vt.set_vit_checkpointing(model, False)
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) == 10822656
    assert sum(p.numel() for p in model.parameters()) < 10**9
    assert model.get_input_embeddings().weight is model.get_output_embeddings().weight
    return model


def implementation():
    paths = [Path(__file__), Path(data.__file__), Path(vt.__file__), Path(lora.__file__), Path(place.__file__),
             Path(modeling.__file__), Path(train.__file__), Path(fmt.__file__)]
    return {str(p): audit.sha(p) for p in paths}


def encoded_batch(rows, encoder):
    encoded = [encoder.row(r, None) for r in rows]
    return vt.to_device(vt.pack(encoded, encoder.tokenizer.convert_tokens_to_ids("<|image_pad|>"),
                                audit.read(data.BASE / "config.json")["vision_config"]))


def frozen_digest(model):
    h = hashlib.sha256()
    for name, p in model.named_parameters():
        if not p.requires_grad:
            h.update(name.encode()); h.update(p.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def validate_inputs():
    """Validate every training schema and render/decode every evaluation row before starting the GPU run."""
    reports = {}
    for name in ("train", *EVALS):
        rows = list(audit.rows(data.MIX / (name + ".jsonl")))
        for row in rows:
            assert row["id"] and row["source"] and isinstance(row["ordered"], bool)
            if row["modality"] == "text":
                assert isinstance(row["kind"], str)
                assert "state" in row
                k = len(fmt.options(row["question"])[0]); target = np.asarray(row["target"])
                assert target.shape == (k,) and np.isfinite(target).all()
                assert (target >= 0).all() and abs(target.sum()-1) < 1e-5
            else:
                assert row["modality"] == "image" and 0 <= row["gold"] < len(row["options"])
                assert row["images"] and isinstance(row["question"], str)
                assert all((data.MIX / p).is_file() for p in row["images"])
        info = {"rows": len(rows), "schema_valid": True}
        if name != "train":
            entries = [("mix", r) for r in rows]; seen = set(); longest = 0
            loader = torch.utils.data.DataLoader(vt.EvalChunks(entries, CFG.token_budget), batch_size=None,
                       num_workers=CFG.workers, worker_init_fn=worker_init, multiprocessing_context="spawn",
                       prefetch_factor=2)
            for groups in loader:
                for batch in groups:
                    for j,index in enumerate(batch["index"]):
                        assert index not in seen; seen.add(index)
                        longest = max(longest, int(batch["cu_cpu"][j+1]-batch["cu_cpu"][j]))
                    assert torch.isfinite(batch["targets"]).all()
                    assert torch.allclose(batch["targets"].sum(-1), torch.ones(len(batch["index"])))
            assert seen == set(range(len(rows)))
            info.update(rendered_rows=len(seen), longest_tokens=longest)
        reports[name] = info
        print("INPUT_VALIDATED", name, info, flush=True)
    train.atomic_json(ROOT / "input-validation-v2.json", reports)
    return reports


def check():
    configure(); prepared = audit.read(ROOT / "prepared.json")
    assert prepared["complete"] and prepared["source_sha256"] == audit.sha(data.__file__)
    for path, expected in prepared["files"].items(): assert audit.sha(path) == expected
    assert audit.sha(data.BASE / "model.safetensors-00001-of-00001.safetensors") == audit.BASE_SHA
    validation = validate_inputs()
    train.seed_all(SEED)
    rows = list(audit.rows(data.MIX / "train.jsonl")); encoder = vt.Encoder(text_pool=vt.text_pool(rows))
    sample = [next(r for r in rows if r["modality"] == m) for m in ("text", "image")]
    model = load_model().eval(); batch = encoded_batch(sample, encoder)
    with torch.no_grad():
        initial = vt.packed_logits(model, batch)
        # All zero-initialized B matrices must make the adapted model exactly the original function.
        saved = []
        for name, module in list(model.named_modules()):
            if isinstance(module, lora.LoRALinear):
                parent, _, leaf = name.rpartition("."); owner = model.get_submodule(parent)
                saved.append((owner, leaf, module)); setattr(owner, leaf, module.base)
        native = vt.packed_logits(model, batch)
        for owner, leaf, module in saved: setattr(owner, leaf, module)
    init_error = float((initial - native)[batch["code_mask"]].abs().max()); assert init_error == 0
    frozen_before = frozen_digest(model); model.train(); opt = train.MasterAdamW(model, 2e-4)
    smoke_rows = rows[:16]
    assert {r["modality"] for r in smoke_rows} == {"text", "image"}
    # Exercise exactly the same augmentation, token budget and packing used in the real run.
    smoke = vt.MixSteps([[(r, f"check:{step}:{j}") for j,r in enumerate(smoke_rows)]
                         for step in range(3)], CFG, vt.text_pool(rows))
    updates = []; torch.cuda.reset_peak_memory_stats()
    for step in range(3):
        opt.zero_grad(); started = time.monotonic(); loss_sum = 0.
        # Smoke parameters are discarded before the actual experiment.
        for cpu in smoke[step]:
            b = vt.to_device(cpu)
            train.set_checkpointing(model, len(b["ids"]) > CFG.checkpoint_above)
            loss = train.soft_ce(vt.packed_logits(model, b), b["targets"]).sum() / len(smoke_rows)
            assert torch.isfinite(loss); loss.backward(); loss_sum += float(loss.detach())
        norm = opt.clip_grad_norm(1.); assert math.isfinite(norm); opt.step(); torch.cuda.synchronize()
        updates.append({"loss": loss_sum, "seconds": time.monotonic() - started})
    assert frozen_digest(model) == frozen_before
    assert any(bool((p != 0).any()) for n, p in model.named_parameters() if n.endswith("lora_b"))
    # Real native path cross-check on a text and an image input (after updates, not only zero adapters).
    model.eval(); train.set_checkpointing(model, False)
    diffs = []
    with torch.no_grad():
        for row in sample:
            e = encoder.row(row, None); b = encoded_batch([row], encoder)
            packed = vt.packed_logits(model, b)[0, b["code_mask"][0]]
            if row["modality"] == "text":
                ids = torch.tensor([e["input_ids"]], device="cuda"); codes = torch.tensor([e["code_ids"]], device="cuda")
                raw = modeling.code_logits(model, ids, torch.ones_like(ids), codes, torch.ones_like(codes,dtype=torch.bool))[0]
            else:
                q, assigned, target = vt.item_view(row, None)
                pictures = encoder.pictures(row, root=data.MIX)
                text = fmt.render(encoder.tokenizer, vt.IMAGE * len(pictures), q, encoder.codes[:len(target)])
                p = encoder.processor(text=[text], images=pictures, return_tensors="pt").to("cuda")
                hidden = model.model(**p, use_cache=False).last_hidden_state[0,-1]
                w = model.get_output_embeddings().weight[torch.tensor(e["code_ids"],device="cuda")]
                raw = w.float() @ hidden.float()
            error = float((packed-raw).abs().max()); margin = float(raw.topk(2).values.diff().abs()[0])
            assert error < .5 and (int(raw.argmax()) == int(packed.argmax()) or margin <= 2*error)
            diffs.append({"modality": row["modality"], "max_logit_error": error,
                          "same_prediction": int(raw.argmax()) == int(packed.argmax()), "native_margin": margin})
    result = {"passed": True, "zero_lora_error": init_error, "frozen_base_unchanged": True,
              "adapter_parameters": 10822656, "total_parameters": sum(p.numel() for p in model.parameters()),
              "updates": updates, "native_comparison": diffs, "peak_gb": torch.cuda.max_memory_allocated()/2**30,
              "implementation": implementation(), "input_validation": validation, "smoke_parameters_discarded": True}
    train.atomic_json(ROOT / "check.json", result)
    print("CHECK", result, flush=True)


def freeze():
    assert not PLAN.exists(), "Keep immutable plans"
    checked = audit.read(ROOT / "check.json"); assert checked["passed"] and checked["implementation"] == implementation()
    prepared = audit.read(ROOT / "prepared.json")
    plan = {"experiment": "joint-g7-from-base-pilot-v1", "seed": SEED, "seeds": [SEED], "exploratory": True,
            "base": str(data.BASE), "base_sha256": audit.BASE_SHA, "prepared_sha256": audit.sha(ROOT/"prepared.json"),
            "implementation": implementation(), "inputs": prepared["files"] | {
                str(ROOT/name): audit.sha(ROOT/name) for name in
                ("training-image-hashes.json", "text-audit.json", "vision-audit.json", "restoration.json")},
            "rows": prepared["rows"], "text_rows": prepared["text_rows"], "image_rows": prepared["image_rows"],
            "step_rows": 16, "steps": prepared["rows"]//16, "epochs": 1, "learning_rate": 2e-4,
            "warmup_steps": prepared["rows"]//16//16, "schedule": "linear warmup then cosine to10%",
            "optimizer": "existing FP32-master AdamW (.9,.95), wd.01, clip1", "trainable": "all186 LM LoRA r16 alpha32 dropout.05; frozen ViT/base/tied embedding",
            "attention": "noncausal six full-attention layers; causal DeltaNet; existing kernels and native d3 readout",
            "evaluation": list(EVALS), "selection": "no checkpoint/merge search; fixed final update; both text and image metrics stored, no seed confirmation claim",
            "alignment": "post-training frozen state-like image/human-NLVR2-statement features; train-only PCA16+orthogonal Procrustes and shuffled-fit control; disjoint reserved image components; diagnostic only, not a native decision-performance intervention or PRH theorem replication",
            "release_gate": "full-board image perceptual dedup and all11 official benchmarks are incomplete; no automatic submission",
            "resources": "current RTX3090 only", "config": vars(CFG),
            "amendment": {"previous_plan_sha256": audit.sha(ROOT / "plan.json"),
                          "record_sha256": audit.sha(ROOT / "input-amendment-v2.json"),
                          "reason": "Repair missing diagnostic logging metadata kind; add full input rendering validation. No prompt, label, split, train row or hyperparameter changes."}}
    train.atomic_json(PLAN, plan)
    snapshot = ROOT / "source-snapshot-v2"; snapshot.mkdir()
    for path in plan["implementation"]: shutil.copyfile(path, snapshot / Path(path).name)


def verify_plan():
    p = audit.read(PLAN)
    assert implementation() == p["implementation"]
    for path, digest in p["inputs"].items(): assert audit.sha(path) == digest
    assert audit.sha(data.BASE / "model.safetensors-00001-of-00001.safetensors") == p["base_sha256"]
    return p


def evaluate(model, label):
    configure(); model.eval(); train.set_checkpointing(model, False); reports = {}
    for name in EVALS:
        rows = list(audit.rows(data.MIX / (name + ".jsonl")))
        entries = [("mix", r) for r in rows]
        loader = torch.utils.data.DataLoader(vt.EvalChunks(entries, CFG.token_budget), batch_size=None,
                  num_workers=CFG.workers, pin_memory=True, prefetch_factor=2, worker_init_fn=worker_init,
                  multiprocessing_context="spawn")
        predictions = [None] * len(rows)
        with torch.inference_mode():
            for groups in loader:
                for cpu in groups:
                    z = vt.packed_logits(model, vt.to_device(cpu)).float().cpu()
                    for j, index in enumerate(cpu["index"]):
                        k = int(cpu["code_mask"][j].sum()); logits = z[j,:k].double()
                        gold = cpu["targets"][j,:k] > 0; prediction = int(logits.argmax())
                        nll = float(torch.logsumexp(logits,0)-torch.logsumexp(logits[gold],0))
                        predictions[index] = {"id": rows[index]["id"], "source": rows[index]["source"],
                          "modality": rows[index]["modality"], "prediction": prediction, "gold": gold.nonzero().flatten().tolist(),
                          "logits": logits.tolist(), "correct": bool(gold[prediction]), "nll": nll,
                          "tokens": int(cpu["cu_cpu"][j+1]-cpu["cu_cpu"][j])}
        assert all(r is not None for r in predictions)
        by = defaultdict(list)
        for row in predictions: by[row["source"]].append(row)
        per = {s: {"rows": len(v), "accuracy": statistics.mean(r["correct"] for r in v),
                   "nll": statistics.mean(r["nll"] for r in v)} for s,v in by.items()}
        reports[name] = {"rows": len(rows), "macro_accuracy": statistics.mean(v["accuracy"] for v in per.values()),
                         "macro_nll": statistics.mean(v["nll"] for v in per.values()), "sources": per}
        with (ROOT / f"{label}-{name}.jsonl").open("w") as f:
            for row in predictions: f.write(json.dumps(row,ensure_ascii=False)+"\n")
        print("EVAL",label,name,{k:v for k,v in reports[name].items() if k!='sources'},flush=True)
    train.atomic_json(ROOT / f"{label}-evaluation.json", reports)
    return reports


def rotation_diagnostic(image, text, fit, rank=16, seed=20261104):
    image, text = np.asarray(image, dtype=float), np.asarray(text, dtype=float)
    fit = np.asarray(fit, dtype=bool); test = ~fit
    assert image.shape == text.shape and image.ndim == 2
    assert fit.shape == (len(image),) and 0 < rank <= image.shape[1]
    assert np.isfinite(image).all() and np.isfinite(text).all()
    if fit.sum() < rank*4 or test.sum() < 32:
        return {"eligible": False, "fit": int(fit.sum()), "evaluation": int(test.sum()), "rank": rank,
                "reason": "insufficient reserved image groups for the fixed rank/size rule"}
    ix, tx = image[fit].mean(0), text[fit].mean(0)
    _, _, vh = np.linalg.svd(np.concatenate([image[fit]-ix,text[fit]-tx]),full_matrices=False)
    projection = vh[:rank].T
    def unit(x): return x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-12)
    a,b=unit((image-ix)@projection),unit((text-tx)@projection)
    def solve(x,y):
        u,_,v=np.linalg.svd(x.T@y,full_matrices=False);return u@v
    rotation=solve(a[fit],b[fit]); shuffled=solve(a[fit],np.random.default_rng(seed).permutation(b[fit]))
    def metrics(x):
        y=b[test]; matrix=x@y.T
        return {"paired_cosine":float(np.mean(np.sum(x*y,axis=1))),
                "image_to_text_r1":float(np.mean(matrix.argmax(1)==np.arange(len(y))))}
    return {"eligible":True,"fit":int(fit.sum()),"evaluation":int(test.sum()),"rank":rank,
            "raw":metrics(a[test]),"orthogonal":metrics(a[test]@rotation),"shuffled_fit":metrics(a[test]@shuffled),
            "orthogonality_max_error":float(np.max(np.abs(rotation.T@rotation-np.eye(rank)))),
            "scope":"PCA and centering fit on fit groups only; evaluation untouched; no downstream logits modified"}


def alignment(model):
    configure(); model.eval(); pairs=list(audit.rows(data.MIX/"alignment-pairs.jsonl"))
    if not pairs:
        result={"eligible":False,"reason":"no independent human-described scene pairs"}
        train.atomic_json(ROOT/"alignment.json",result);return
    processor=AutoProcessor.from_pretrained(data.BASE,local_files_only=True)
    from decision_index.vision import images as img
    features=[]
    with torch.inference_mode():
        for pair in pairs:
            pictures=[img.pil(p,root=data.MIX,max_pixels=CFG.max_pixels).convert("RGB") for p in pair["images"]]
            modalities=[]
            for state,images in ((vt.IMAGE*len(pictures),pictures),(pair["description"],None)):
                # No task-specific decision question or answer label: a common scene representation request.
                prompt=processor.tokenizer.apply_chat_template([
                    {"role":"system","content":"Read the scene description or image as data."},
                    {"role":"user","content":"State:\n"+state}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
                batch=processor(text=[prompt],images=images,return_tensors="pt").to("cuda")
                h=model.model(**batch,use_cache=False).last_hidden_state[0,-1]
                modalities.append(h.float().cpu().numpy())
            features.append(modalities)
    array=np.asarray(features);np.savez(ROOT/"alignment-features.npz",image=array[:,0],text=array[:,1],
                                      fit=np.array([p["split"]=="fit" for p in pairs]))
    result=rotation_diagnostic(array[:,0].astype(float),array[:,1].astype(float),[p["split"]=="fit" for p in pairs])
    result["data_caveat"]="NLVR2 true human statements describe partial scenes; good-view kernel equality is not established; no theorem or decision-accuracy gain inferred"
    train.atomic_json(ROOT/"alignment.json",result);print("ALIGNMENT",result,flush=True)


def merge_and_save(model,opt):
    adapters=vt.adapter_state(model,opt);save_file({k:v.detach().cpu().contiguous() for k,v in adapters.items()},str(ROOT/"adapter.safetensors"))
    del adapters
    with torch.no_grad():
        for name,module in list(model.named_modules()):
            if isinstance(module,lora.LoRALinear):
                module.base.weight.copy_(module.base.weight.float()+module.scale*(module.lora_b.float()@module.lora_a.float()))
                parent,_,leaf=name.rpartition(".");setattr(model.get_submodule(parent),leaf,module.base)
    processor=AutoProcessor.from_pretrained(data.BASE,local_files_only=True)
    modeling.save(model,processor.tokenizer,ROOT/"model",{"attention":"noncausal","deltanet":"causal","layout":"d3",
                   "experiment":"joint-g7-from-base-pilot-v1","steps":2048,"seed":SEED})
    for name in ("preprocessor_config.json","video_preprocessor_config.json"):
        shutil.copyfile(data.BASE/name,ROOT/"model"/name)


def publish(result):
    """Publish both modalities together; internal accuracy cannot stand in for official Full."""
    target=Path("wiki/project/champion-joint-pilot.md")
    lines=["# 원본 Qwen 공동 학습 파일럿", "", "고정 seed 1개, 텍스트16,384 + 이미지16,384 문항, CE 2,048 update의 탐색 결과다. 공식 보드 점수나 순위가 아니다.", "",
           "| 평가 | 원본 정확도 | 저장·복원한 공동 모델 정확도 | 변화 (%p) |", "|---|---:|---:|---:|"]
    for name in EVALS:
        before=result["base"][name]["macro_accuracy"];after=result["trained"][name]["macro_accuracy"]
        lines.append(f"| {name} | {before*100:.2f}% | {after*100:.2f}% | {(after-before)*100:+.2f} |")
    a=result["alignment"]
    lines += ["", "표는 출처 평균 정답률이며 F1·nDCG·공식 Full을 재현하지 않는다. 기존 diagnostic은 이미 관찰한 평가다.", "",
              "회전 정렬은 학습에 쓰이지 않은 실제 이미지와 사람 서술로 별도 검증했다. 이 단계는 판단 출력에 적용하지 않았다."]
    if a["eligible"]:
        lines.append(f"정렬 평가 {a['evaluation']}쌍에서 cosine은 {a['raw']['paired_cosine']:.4f} → {a['orthogonal']['paired_cosine']:.4f}, shuffled 대조는 {a['shuffled_fit']['paired_cosine']:.4f}다. 기하 개선만으로 판단 정확도 향상을 추론하지 않는다.")
    else: lines.append("정렬의 사전 고정 최소 표본 조건을 충족하지 못해 효과를 판정하지 않았다.")
    lines += ["", "기본 가중치는 고정한 공식 원본이고 기존 Crystal/QEV 가중치를 사용하지 않았다. ViT는 동결했다. 어댑터 병합 후 전체 평가를 다시 했으며 저장·복원 로그 일치를 확인했다.", "",
              "전체 비전 보드의 근사 중복 검사는 아직 완료되지 않아 제출용 최종 학습으로 취급하지 않는다. 두 트랙의 공식 1위 목표는 아직 달성하지 않았다.", "",
              "상세 결과: [JSON](champion-joint-pilot-results.json). 설계·계보: [프로젝트](champion-v1.md). 실행: `runs/champion-v1/joint-g7-v1/`."]
    target.write_text("\n".join(lines)+"\n")


def run():
    plan=verify_plan();configure();train.seed_all(SEED)
    train.atomic_json(ROOT/"status.json",{"state":"evaluating_original_both_modalities","training_started":False})
    model=load_model();base=evaluate(model,"base")
    rows=list(audit.rows(data.MIX/"train.jsonl")); order=list(range(len(rows)));random.Random(SEED).shuffle(order)
    steps=[[(rows[i],f"view:{SEED}:{step}:{j}") for j,i in enumerate(order[step*16:(step+1)*16])]
           for step in range(plan["steps"])]
    loader=torch.utils.data.DataLoader(vt.MixSteps(steps,CFG,vt.text_pool(rows)),batch_size=None,num_workers=CFG.workers,
           pin_memory=True,prefetch_factor=2,worker_init_fn=worker_init,multiprocessing_context="spawn")
    model.train();opt=train.MasterAdamW(model,2e-4);started=time.monotonic();all_tokens=0
    train.atomic_json(ROOT/"status.json",{"state":"training_joint_ce","training_started":True,"step":0,"steps":plan["steps"]})
    with (ROOT/"train.jsonl").open("x",buffering=1) as log:
        for step,groups in enumerate(loader,1):
            opt.zero_grad();value=0.;tokens=0
            for cpu in groups:
                b=vt.to_device(cpu);train.set_checkpointing(model,len(cpu["ids"])>CFG.checkpoint_above)
                loss=train.soft_ce(vt.packed_logits(model,b),b["targets"]).sum()/16
                assert torch.isfinite(loss);loss.backward();value+=float(loss.detach());tokens+=len(cpu["ids"])
            norm=opt.clip_grad_norm(1.);assert math.isfinite(norm)
            w=plan["warmup_steps"];n=plan["steps"]
            scale=min(step/w,1.)*(.1+.45*(1+math.cos(math.pi*max(0,step-w)/max(1,n-w))))
            for group in opt.param_groups:group["lr"]=2e-4*group["lr_scale"]*scale
            opt.step();all_tokens+=tokens
            record={"step":step,"loss":value,"grad_norm":norm,"tokens":tokens,"total_tokens":all_tokens,"seconds":time.monotonic()-started}
            log.write(json.dumps(record)+"\n")
            if step%32==0 or step==n:
                print("TRAIN",record,flush=True);train.atomic_json(ROOT/"status.json",{"state":"training_joint_ce","training_started":True,"steps":n,**record})
            if step%512==0:
                save_file({k:v.detach().cpu().contiguous() for k,v in vt.adapter_state(model,opt).items()},str(ROOT/"latest-adapter.safetensors"))
                torch.save({"step":step,"optimizer":opt.optimizer.state_dict(),"torch_rng":torch.get_rng_state(),
                            "cuda_rng":torch.cuda.get_rng_state_all()},ROOT/"latest-optimizer.pt")
    trained=time.monotonic()-started
    train.atomic_json(ROOT/"status.json",{"state":"evaluating_both_modalities","training_started":True})
    adapter_eval=evaluate(model,"adapter");alignment(model)
    merge_and_save(model,opt);del opt;gc.collect();torch.cuda.empty_cache()
    # Reload the exported model and compare the same fixed text/image rows after serialization.
    sample=[next(r for r in rows if r["modality"]==m) for m in ("text","image")]
    encoder=vt.Encoder();b=encoded_batch(sample,encoder)
    with torch.inference_mode():reference=vt.packed_logits(model,b).detach().cpu()
    del model;gc.collect();torch.cuda.empty_cache()
    reloaded=modeling.load(ROOT/"model",attention="noncausal",dtype=torch.bfloat16,kernel="varlen");reloaded.model.visual.to("cuda");reloaded.eval()
    with torch.inference_mode():current=vt.packed_logits(reloaded,b).cpu()
    assert torch.equal(reference,current)
    # BF16 merging can change close decisions; report the actual exported model on all four panels.
    after=evaluate(reloaded,"exported")
    del reloaded;gc.collect();torch.cuda.empty_cache()
    deltas={k:{m:after[k][m]-base[k][m] for m in ("macro_accuracy","macro_nll")} for k in EVALS}
    result={"complete":True,"seed":SEED,"train_seconds":trained,"total_train_tokens":all_tokens,"base":base,"trained":after,
            "deltas":deltas,"adapter_before_merge":adapter_eval,"alignment":audit.read(ROOT/"alignment.json"),"export_replay_exact":True,
            "scope":"single-seed exploratory joint reference; internal source metrics, not official Full; alignment geometry only; final image contamination clearance incomplete"}
    train.atomic_json(ROOT/"report.json",result)
    shutil.copyfile(ROOT/"report.json",Path("wiki/project/champion-joint-pilot-results.json"))
    publish(result)
    verify_plan();train.atomic_json(ROOT/"status.json",{"state":"complete","training_started":True,"verified":True,"steps":plan["steps"]})
    print("COMPLETE",ROOT,flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("command",choices=("check","freeze","run"));args=parser.parse_args()
    torch.set_num_threads(4);torch.set_float32_matmul_precision("highest")
    try:globals()[args.command]()
    except Exception:
        train.atomic_json(ROOT/"status.json",{"state":"failed","stage":args.command,"traceback":traceback.format_exc()});raise


if __name__=="__main__":main()
