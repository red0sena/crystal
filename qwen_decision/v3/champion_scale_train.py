"""One fixed joint scale-up, followed by both official public evaluations."""
from __future__ import annotations
import argparse
import collections
import gc
import json
import math
from pathlib import Path
import random
import shutil
import time
import traceback

import torch
from safetensors.torch import load_file,save_file
from transformers import AutoProcessor

from qwen_decision.v3 import champion_joint_train as legacy,champion_scale_data as data
from qwen_decision.v3 import champion_audit as audit,train,vision_train as vt,modeling

ROOT=data.ROOT
PLAN=ROOT/"plan.json"
EVALS=("text-dev","format-dev","image-dev","text-diagnostic","vision-diagnostic")
EPOCHS=2
ORIGINAL_IMPLEMENTATION=legacy.implementation
ORIGINAL_VALIDATION=legacy.validate_inputs


def implementation():
    paths=ORIGINAL_IMPLEMENTATION()
    from qwen_decision.v3 import champion_joint_data,vision_mix,vision_convert_cauldron,vision_convert_screen,decontam,mix
    for module in (champion_joint_data,vision_mix,vision_convert_cauldron,vision_convert_screen,decontam,mix):
        paths[str(Path(module.__file__))]=audit.sha(module.__file__)
    paths[str(Path(__file__))]=audit.sha(__file__)
    for p in [Path("qwen_decision/v3/champion_scale_service.py"),Path("qwen_decision/v3/public_eval_engines.py"),Path("qwen_decision/v3/engine.py"),Path("qwen_decision/v3/vision_engine.py")]:
        paths[str(p)]=audit.sha(p)
    for p in Path("external/decision-index/decision_index").rglob("*"):
        if p.suffix in (".py",".json"):paths[str(p)]=audit.sha(p)
    return paths


def bind():
    # The pilot's tested model, encoding, optimizer checks and evaluation code are reused unchanged.
    legacy.data=data;legacy.ROOT=ROOT;legacy.SEED=data.SEED;legacy.PLAN=PLAN;legacy.EVALS=EVALS
    legacy.implementation=implementation;legacy.worker_init=worker_init
    legacy.validate_inputs=validated_inputs
    legacy.configure()


def validated_inputs():
    """Render all inputs on CPU ahead of the GPU queue; reuse only identical audited inputs/code."""
    prepared=audit.read(ROOT/"prepared.json")
    for p,h in prepared["files"].items():assert audit.sha(p)==h,p
    signature={"inputs":prepared["files"],"implementation":implementation(),"config":vars(legacy.CFG),
        "processor_files":{str(p):audit.sha(p) for p in data.BASE.glob("*") if p.suffix in (".json",".jinja",".txt")}}
    path=ROOT/"input-preflight-cache.json"
    if path.exists():
        cached=audit.read(path)
        if cached["signature"]==signature:
            print("INPUT_VALIDATION_REUSED_IDENTICAL_FILES_AND_CODE",flush=True)
            return cached["reports"]
    reports=ORIGINAL_VALIDATION()
    train.atomic_json(path,{"signature":signature,"reports":reports})
    return reports


def worker_init(_):
    bind();torch.set_num_threads(2)


class Schedule:
    """Every epoch visits each row exactly once, with text16:image4 in each update."""
    def __init__(self,rows,epochs=EPOCHS,seed=data.SEED):
        self.by={m:[r for r in rows if r["modality"]==m] for m in ("text","image")}
        self.take={"text":16,"image":4};self.epochs=epochs;self.seed=seed
        self.per_epoch=len(self.by["text"])//16
        assert self.per_epoch>0 and all(len(self.by[m])==self.per_epoch*self.take[m] for m in self.by)
        self.orders={}
        for epoch in range(epochs):
            for m,rs in self.by.items():
                order=list(range(len(rs)));random.Random(f"{seed}:order:{epoch}:{m}").shuffle(order)
                self.orders[epoch,m]=order

    def __len__(self):return self.per_epoch*self.epochs

    def __getitem__(self,index):
        if not 0<=index<len(self):raise IndexError(index)
        epoch,step=divmod(index,self.per_epoch);result=[]
        for m,rs in self.by.items():
            n=self.take[m];order=self.orders[epoch,m][step*n:(step+1)*n]
            result += [(rs[i],f"scale-view:{self.seed}:{epoch}:{step}:{m}:{j}") for j,i in enumerate(order)]
        random.Random(f"{self.seed}:batch:{index}").shuffle(result)
        return result


def freeze():
    if PLAN.exists():
        verify();return
    checked=audit.read(ROOT/"check.json")
    assert checked["passed"] and checked["implementation"]==implementation()
    prepared=audit.read(ROOT/"prepared.json")
    assert prepared["complete"] and prepared["source_sha256"]==audit.sha(data.__file__)
    vision=audit.read(ROOT/"vision-audit.json")
    assert vision["full_public_board_sha_and_dhash_audit_complete"]
    image_hashes=audit.read(ROOT/"training-image-hashes.json")
    for p,(h,_) in image_hashes.items():assert audit.sha(data.MIX/p)==h
    extra={str(ROOT/name):audit.sha(ROOT/name) for name in
           ("prepared.json","text-audit.json","text-coverage.json","vision-audit.json","training-image-hashes.json","extra-restoration.json","weakness-analysis.json","input-preflight-cache.json")}
    public=Path("runs/champion-v1/public-eval-v1")
    extra.update({str(p):audit.sha(p) for p in (public/"text-suite").glob("*") if p.is_file()})
    extra[str(public/"text-counted-rows.jsonl")]=audit.sha(public/"text-counted-rows.jsonl")
    extra[str(public/"vision-suite/vision-rows.jsonl")]=audit.sha(public/"vision-suite/vision-rows.jsonl")
    vision_files=audit.read(public/"joint-vision/plan.json")["files"]
    extra.update({p:h for p,h in vision_files.items() if p.startswith(str(public/"vision-suite"))})
    plan={"experiment":"joint-g7-scale-v2","base":str(data.BASE),"base_sha256":audit.BASE_SHA,
        "seed":data.SEED,"epochs":EPOCHS,"text_unique_rows":data.N_TEXT,"image_unique_rows":data.N_IMAGE,
        "step_rows":20,"text_per_step":16,"image_per_step":4,"steps":data.N_TEXT//16*EPOCHS,
        "learning_rate":2e-4,"warmup_steps":512,"schedule":"linear warmup then cosine to10%",
        "trainable":"same all186 LM LoRA r16 alpha32 dropout.05; frozen ViT/base/tied embedding",
        "optimizer":"existing FP32-master AdamW (.9,.95), wd.01, clip1",
        "checkpoint_steps":"first at128, then every2048 and final; adapter + FP32 masters + optimizer + RNG",
        "evaluation":list(EVALS),"selection":"fixed final update; no public checkpoint tuning; both modalities and paired presentation dev",
        "config":vars(legacy.CFG),"implementation":implementation(),"inputs":prepared["files"]|extra,
        "images":image_hashes,"hypothesis":"Pilot's ~147 text rows/source and narrow presentation coverage undertrain tasks; increase g7 coverage, class/template balance, presentation diversity and preserve vision with joint updates.",
        "scope":"one bundled scale-up candidate, not isolation of any single change; no rank claim without private evaluation",
        "alignment":"not added: prior frozen alignment did not change native decisions or establish downstream benefit",
        "resources":"current GPU only, wait until preceding public suite finishes; no uploads or external paid computation"}
    train.atomic_json(PLAN,plan)
    snapshot=ROOT/"source-snapshot";snapshot.mkdir()
    for i,p in enumerate(plan["implementation"]):shutil.copyfile(p,snapshot/(str(i)+"-"+Path(p).name))


def verify():
    p=audit.read(PLAN);assert p["implementation"]==implementation()
    assert audit.sha(data.BASE/"model.safetensors-00001-of-00001.safetensors")==p["base_sha256"]
    for file,h in p["inputs"].items():assert audit.sha(file)==h,file
    for file,(h,_) in p["images"].items():assert audit.sha(data.MIX/file)==h,file
    return p


def save_checkpoint(model,opt,step,tokens):
    dest=ROOT/"checkpoints"/f"step-{step:06}";dest.mkdir(parents=True,exist_ok=True)
    save_file({k:v.detach().cpu().contiguous() for k,v in vt.adapter_state(model,opt).items()},str(dest/"adapter.safetensors"))
    torch.save({"step":step,"total_tokens":tokens,"optimizer":opt.optimizer.state_dict(),"rng":torch.get_rng_state(),
                "cuda_rng":torch.cuda.get_rng_state_all()},dest/"optimizer.pt")
    train.atomic_json(ROOT/"last-checkpoint.json",{"directory":str(dest),"step":step,
        "files":{n:audit.sha(dest/n) for n in ("adapter.safetensors","optimizer.pt")}})


def restore_checkpoint(model,opt):
    if not (ROOT/"last-checkpoint.json").exists():return 0,0
    cp=audit.read(ROOT/"last-checkpoint.json");dest=Path(cp["directory"])
    for f,h in cp["files"].items():assert audit.sha(dest/f)==h
    state=load_file(str(dest/"adapter.safetensors"),device="cuda");masters={id(p):m for p,m in opt.pairs}
    with torch.no_grad():
        for name,p in model.named_parameters():
            if p.requires_grad:
                p.copy_(state[name]);masters[id(p)].copy_(state["master."+name])
    saved=torch.load(dest/"optimizer.pt",map_location="cpu",weights_only=False)
    opt.optimizer.load_state_dict(saved["optimizer"])
    torch.set_rng_state(saved["rng"]);torch.cuda.set_rng_state_all(saved["cuda_rng"])
    return saved["step"],saved["total_tokens"]


def export(model,opt,steps):
    legacy.merge_and_save(model,opt)
    # The reused export helper has pilot metadata literals; replace them before final replay/evaluation.
    path=ROOT/"model/v3_config.json";cfg=audit.read(path)
    cfg.update(experiment="joint-g7-scale-v2",steps=steps,seed=data.SEED)
    train.atomic_json(path,cfg)


def run():
    plan=verify();train.seed_all(data.SEED)
    model=legacy.load_model()
    baseline_path=ROOT/"base-evaluation.json"
    baseline=audit.read(baseline_path) if baseline_path.exists() else legacy.evaluate(model,"base")
    rows=list(audit.rows(data.MIX/"train.jsonl"));schedule=Schedule(rows)
    assert len(schedule)==plan["steps"]
    opt=train.MasterAdamW(model,plan["learning_rate"])
    start_step,tokens=restore_checkpoint(model,opt)
    dataset=vt.MixSteps(schedule,legacy.CFG,vt.text_pool(rows))
    loader=torch.utils.data.DataLoader(torch.utils.data.Subset(dataset,range(start_step,len(dataset))),batch_size=None,
        num_workers=legacy.CFG.workers,pin_memory=True,prefetch_factor=2,worker_init_fn=worker_init,
        multiprocessing_context="spawn",generator=torch.Generator().manual_seed(data.SEED))
    path=ROOT/"train.jsonl"
    if path.exists():
        old=[r for r in audit.rows(path) if r["step"]<=start_step]
        path.write_text("".join(json.dumps(r)+"\n" for r in old))
    model.train();started=time.monotonic()
    with path.open("a",buffering=1) as log:
        for step,groups in enumerate(loader,start_step+1):
            opt.zero_grad();loss_sum=0.;batch_tokens=0
            for cpu in groups:
                batch=vt.to_device(cpu);train.set_checkpointing(model,len(cpu["ids"])>legacy.CFG.checkpoint_above)
                loss=train.soft_ce(vt.packed_logits(model,batch),batch["targets"]).sum()/plan["step_rows"]
                assert torch.isfinite(loss);loss.backward();loss_sum+=float(loss.detach());batch_tokens+=len(cpu["ids"])
            norm=opt.clip_grad_norm(1.);assert math.isfinite(norm)
            warm=plan["warmup_steps"];n=plan["steps"]
            factor=min(step/warm,1.)*(.1+.45*(1+math.cos(math.pi*max(0,step-warm)/max(1,n-warm))))
            for group in opt.param_groups:group["lr"]=plan["learning_rate"]*group["lr_scale"]*factor
            opt.step();tokens+=batch_tokens
            record={"step":step,"steps":n,"epoch":step/schedule.per_epoch,"loss":loss_sum,"grad_norm":norm,
                    "tokens":batch_tokens,"total_tokens":tokens,"seconds_this_process":time.monotonic()-started}
            log.write(json.dumps(record)+"\n")
            if step%32==0:
                train.atomic_json(ROOT/"status.json",{"stage":"training","training_started":True,**record});print("TRAIN",record,flush=True)
            if step==128 or step%2048==0 or step==n:save_checkpoint(model,opt,step,tokens)
    adapter_eval=legacy.evaluate(model,"adapter")
    export(model,opt,plan["steps"]);del opt;gc.collect();torch.cuda.empty_cache()
    sample=[next(r for r in rows if r["modality"]==m) for m in ("text","image")]
    encoder=vt.Encoder();batch=legacy.encoded_batch(sample,encoder)
    with torch.inference_mode():expected=vt.packed_logits(model,batch).cpu()
    del model;gc.collect();torch.cuda.empty_cache()
    saved=modeling.load(ROOT/"model",attention="noncausal",deltanet="causal",dtype=torch.bfloat16,kernel="varlen")
    saved.model.visual.to("cuda");saved.eval()
    with torch.inference_mode():actual=vt.packed_logits(saved,batch).cpu()
    assert torch.equal(expected,actual)
    trained=legacy.evaluate(saved,"exported")
    result={"complete":True,"base":baseline,"trained":trained,"adapter_before_merge":adapter_eval,
            "export_replay_exact":True,"steps":plan["steps"],"total_train_tokens":tokens,
            "official_full_rank":None,"seed":data.SEED,"exploratory":True}
    train.atomic_json(ROOT/"report.json",result);verify()
    train.atomic_json(ROOT/"status.json",{"stage":"trained","training_started":True,"verified":True,"steps":plan["steps"]})


def public_eval(modality):
    from decision_index.suite.io import Suite
    public=Path("runs/champion-v1/public-eval-v1");dest=ROOT/("public-"+modality)
    verify();assert audit.read(ROOT/"report.json")["export_replay_exact"]
    model_hash={str(p):audit.sha(p) for p in (ROOT/"model").glob("*") if p.is_file()}
    if modality=="text":
        from decision_index.runner import run as runner
        from decision_index.pipeline import score_run
        runner("qwen_decision.v3.public_eval_engines:JointText",{"model":str(ROOT/"model")},
               public/"text-counted-rows.jsonl",dest,compact=True,corpus_sha256=audit.sha(public/"text-counted-rows.jsonl"))
        result=score_run(Suite(public/"text-suite","0.3"),dest/"results.jsonl","joint-g7-scale-v2",dest)
    else:
        from decision_index.vision.run import run as runner
        from decision_index.vision.score import score_run
        runner("qwen_decision.v3.vision_engine:VisionEngine",{"model":str(ROOT/"model"),"prompt":"decide"},
               public/"vision-suite",dest,workers=1,chunk=16,compact=True)
        result=score_run(public/"vision-suite/vision-rows.jsonl",dest/"results.jsonl",dest)
    for p,h in model_hash.items():assert audit.sha(p)==h
    results=list(audit.rows(dest/"results.jsonl"));counts=collections.Counter(r["status"] for r in results)
    expected=110087 if modality=="text" else 12210
    assert len({r["run_id"] for r in results})==expected and not counts["error"]
    train.atomic_json(dest/"completion.json",{"complete":True,"counts":dict(counts),"visited_requests":expected,
        "public_index":result.get("decision_index",result.get("public_vision_index")),"model_hashes":model_hash})


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("command",choices=("check-inputs","check","freeze","run","eval-text","eval-vision"));args=parser.parse_args()
    torch.set_num_threads(4);torch.set_float32_matmul_precision("highest");bind()
    try:
        if args.command=="check-inputs":validated_inputs()
        elif args.command=="check":legacy.check()
        elif args.command=="freeze":freeze()
        elif args.command=="run":run()
        else:public_eval(args.command.removeprefix("eval-"))
    except Exception:
        train.atomic_json(ROOT/"failure.json",{"stage":args.command,"traceback":traceback.format_exc()});raise


if __name__=="__main__":main()
