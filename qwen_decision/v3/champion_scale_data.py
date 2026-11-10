"""Audited g7 scale-up inputs; presentation changes preserve every gold target."""
from __future__ import annotations
from collections import Counter,defaultdict
from concurrent.futures import ProcessPoolExecutor
import hashlib
import heapq
import json
import os
from pathlib import Path
import random
import shutil
import time

from qwen_decision.v3 import champion_audit as audit,champion_joint_data as old
from qwen_decision.v3 import decontam,format as fmt,mix,train,vision_mix as vm,vision_convert_cauldron as cc

ROOT=Path("runs/champion-v1/joint-g7-v2")
PREVIOUS=old.ROOT
MIX=ROOT/"mix"
BASE=old.BASE
SEED=20261105
N_TEXT,N_IMAGE=131072,32768
MAX_TOKENS=16384
BOOST={"phishing":4.,"hover-train":3.,"apibank-train":3.,"banking77-train":3.,
       "clinc-train":3.,"vast-train":3.,"isarcasm-train":3.,"hellaswag-train":3.,
       "winogrande-train":3.,"poem-pairs-gen":2.,"contractnli-train":2.,"esci-train":2.,
       "mnli-train":1.5,"wanli-train":1.5,"ragtruth-train":1.5,"nli4ct-train":1.5}


def allocate(total,weights,capacity):
    """Weighted quotas capped by distinct available examples; never oversample a row."""
    if total>sum(capacity.values()):raise ValueError("not enough distinct eligible examples")
    out={s:0 for s in capacity};active={s for s,c in capacity.items() if c};left=total
    while left:
        denom=sum(weights[s] for s in active)
        shares={s:left*weights[s]/denom for s in active}
        capped={s for s in active if shares[s]>=capacity[s]-out[s]}
        if capped:
            for s in sorted(capped):
                n=capacity[s]-out[s];out[s]+=n;left-=n;active.remove(s)
            continue
        for s in sorted(active):
            n=int(shares[s]);out[s]+=n;left-=n
        for s in sorted(active,key=lambda s:(-(shares[s]-int(shares[s])),s))[:left]:out[s]+=1
        left=0
    assert sum(out.values())==total and all(out[s]<=capacity[s] for s in out)
    return out


def presentation(row,seed,force=False):
    """Move existing context / add opaque option keys; no new labels, facts or alternatives."""
    rng=random.Random(seed);out=dict(row);q=dict(row["question"]);changes=[]
    fixed=row["ordered"] or mix.positional_options(q)
    if q["type"]=="choice" and not fixed and (force or rng.random()<.5):
        _,texts=fmt.options(q)
        q["criteria"]={f"candidate_{i}":text for i,text in enumerate(texts)}
        changes.append("opaque_keys_preserving_full_option_text")
    if row["state"] not in (None,"",{},[]) and (force or rng.random()<.35):
        context=fmt.describe(row["state"])
        q["instructions"]="Context (data):\n"+context+"\n\nQuestion:\n"+(q.get("instructions") or fmt.DEFAULT_INSTRUCTIONS)
        out["state"]={};changes.append("context_in_question")
    out["question"]=q
    if changes:
        out["presentation_changes"]=changes
        out["id"]=row["id"]+":presentation:"+old.digest(seed)[:12]
    return out


def stratum(row):
    q=row["question"];gold=mix.gold_set(row)[0]
    if q["type"]=="noul":
        # Fixed question wording separates positive/negative statements before balancing truth.
        wording=q.get("instructions","")
        return ("noul",wording if len(wording)<250 else "long",gold)
    if row["source"] in {"banking77-train","clinc-train","vast-train","mnli-train","wanli-train"}:
        return ("class",list(q["criteria"])[gold].lower().replace("_"," "))
    if row["source"]=="isarcasm-train":return ("sarcasm-task",row["id"].split(":")[1],gold)
    return ("choice",len(row["target"]))


def fair_order(rows):
    bins=defaultdict(list)
    for r in rows:bins[str(stratum(r))].append(r)
    for key in bins:bins[key].sort(key=lambda r:old.digest("scale-balance",r))
    for i in range(max(map(len,bins.values()),default=0)):
        for key in sorted(bins):
            if i<len(bins[key]):yield bins[key][i]


_FILTER=_TOKENIZER=_CODES=None


def text_worker_init():
    from transformers import AutoTokenizer
    global _FILTER,_TOKENIZER,_CODES
    _FILTER=decontam.Filter(PREVIOUS/"restored/data/decontam")
    _TOKENIZER=AutoTokenizer.from_pretrained(BASE,local_files_only=True)
    _CODES,_=fmt.answer_codes(_TOKENIZER)


def select_source(args):
    source,candidates,limit=args;out=[];seen=set();dropped=Counter();longest=(0,None)
    for row in fair_order(candidates):
        identity=old.digest({"state":row["state"],"question":row["question"],"target":audit.target_signature(row)})
        if identity in seen:dropped["identical_variant"]+=1;continue
        seen.add(identity)
        why=_FILTER.flagged(decontam.row_texts(row["state"],{"q":row["question"]}))
        if why:dropped["public_jeval_overlap"]+=1;continue
        row=train.gold_targets([row])[0]
        row={**row,"modality":"text","original_id":row["id"],"id":row["id"]+":variant:"+identity[:16]}
        row=presentation(row,f"scale:{SEED}:{identity}")
        n=len(_TOKENIZER.encode(fmt.render(_TOKENIZER,row["state"],row["question"],_CODES[:len(row["target"])]),add_special_tokens=False))
        if n>MAX_TOKENS:dropped["over_16384_no_truncation"]+=1;continue
        row["prepared_tokens"]=n
        out.append(row)
        if n>longest[0]:longest=(n,row["id"])
        if len(out)>=limit:break
    return source,out,{"eligible":len(out),"dropped":dict(dropped),"longest_tokens":longest[0]}


def prepare_text():
    dest=MIX/"text-train.jsonl"
    if dest.exists():return
    raw=PREVIOUS/"restored/data/mix";manifest=audit.read(raw/"v3-gold7.manifest.json")
    expected=audit.read("runs/champion-v1/g7-review/review.json")["hub_files_match_g7_training_manifest"]
    for file,v in expected.items():assert audit.sha(PREVIOUS/"restored"/file)==v["sha256"]
    dev=list(audit.rows(raw/"v3-gold7.dev.jsonl"));devkeys=set().union(*(old.keys(r) for r in dev))
    devinputs={audit.input_key(r) for r in dev}
    fresh={audit.state_key(r["state"]) for r in audit.rows("data/fresh/fresh-v1.jsonl")}
    counts=Counter();dropped=Counter();heaps=defaultdict(list)
    for i,row in enumerate(audit.rows(raw/"v3-gold7.jsonl"),1):
        assert old.eligible_origin(row) and row["source"] not in mix.CORPUS_EXCLUDED
        assert len(fmt.options(row["question"])[0])==len(row["target"])
        counts[row["source"]]+=1
        if old.keys(row)&devkeys:dropped["dev_item_document"]+=1;continue
        if audit.input_key(row) in devinputs:dropped["dev_input"]+=1;continue
        if audit.has_context(row) and audit.state_key(row["state"]) in fresh:dropped["fresh_state"]+=1;continue
        rank=int(old.digest("scale-pool",SEED,row),16);entry=(-rank,i,row);h=heaps[row["source"]]
        cap=8192 if row["source"] in BOOST else 3072
        if len(h)<cap:heapq.heappush(h,entry)
        elif entry[:2]>h[0][:2]:heapq.heapreplace(h,entry)
        if i%200000==0:print("G7_AUDITED",i,flush=True)
    assert dict(counts)==manifest["counts"] and sum(counts.values())==999961
    weights={s:BOOST.get(s,1.) for s in heaps}
    initial=allocate(N_TEXT,weights,{s:len(h) for s,h in heaps.items()})
    jobs=[(s,[e[2] for e in h],min(len(h),initial[s]+512)) for s,h in sorted(heaps.items())]
    pools={};reports={}
    with ProcessPoolExecutor(6,initializer=text_worker_init) as pool:
        for s,rows,info in pool.map(select_source,jobs,chunksize=1):
            pools[s]=rows;reports[s]=info;print("TEXT_SOURCE",s,len(rows),flush=True)
    quotas=allocate(N_TEXT,weights,{s:len(v) for s,v in pools.items()})
    selected=[r for s in sorted(pools) for r in pools[s][:quotas[s]]]
    assert len(selected)==N_TEXT
    by=defaultdict(list)
    for r in dev:
        assert old.eligible_origin(r)
        by[r["source"]].append(r)
    chosen=[{**r,"modality":"text"} for s,rs in sorted(by.items()) for r in list(fair_order(rs))[:32]]
    chosen=train.gold_targets(chosen)
    variants=[presentation(r,f"format-dev:{r['id']}",force=True) for r in chosen]
    old.write(dest,selected);old.write(MIX/"text-dev.jsonl",chosen);old.write(MIX/"format-dev.jsonl",variants)
    train.atomic_json(ROOT/"text-audit.json",{"raw_rows":sum(counts.values()),"raw_sources":dict(counts),
        "quotas":quotas,"source_reports":reports,"early_dropped":dict(dropped),"dev_rows":len(chosen),
        "max_training_tokens":max(r["prepared_tokens"] for r in selected),
        "presentation_changes":dict(Counter(c for r in selected for c in r.get("presentation_changes",[]))),
        "label_policy":"same g7 human/programmatic + existing source-specific four exceptions; no new teachers",
        "selection":"Source weights fixed from public task families; no individual public errors are training rows; all g7 dev documents/items blocked; existing pinned public/JEVal filter reapplied."})


class LinkedSaver(vm.Saver):
    def _write(self,raw,ext):
        path=self.dir/f"{hashlib.sha256(raw).hexdigest()[:24]}.{ext}"
        previous=PREVIOUS/"mix"/path.relative_to(MIX)
        if not path.exists() and previous.exists():os.link(previous,path)
        return super()._write(raw,ext)


def _hashed(path):
    raw=Path(path).read_bytes()
    return path,hashlib.sha256(raw).hexdigest(),vm.dhash(raw)


def prepare_images():
    dest=MIX/"image-train.jsonl"
    if dest.exists():return
    from qwen_decision.v3 import vision_convert_screen as screen
    extras=audit.read(ROOT/"extra-restoration.json")
    for r in extras:assert audit.sha(r["local"])==r["sha256"]
    plan=audit.read(PREVIOUS/"vision-download-plan.json")
    for r in plan["files"]:
        assert audit.sha(PREVIOUS/"raw/cauldron"/r["file"])==r["sha256"]
    vm.MIX=MIX;vm.IMG=MIX/"img";cc.CAULDRON=PREVIOUS/"raw/cauldron"
    screen.CORD=ROOT/"raw/naver-clova-ix__cord-v2/data"
    screen.FUNSD=ROOT/"raw/nielsr__funsd/data";screen.FUNSD_ZIP=ROOT/"raw/funsd-dataset.zip"
    candidates=[]
    sources=[("cauldron/"+s,lambda saver,cap,rng,s=s:cc.convert(cc.SPECS[s][1],saver,cap,rng)) for s in plan["subsets"]]
    sources += [("cord",screen.cord),("funsd",screen.funsd)]
    pooldir=ROOT/"pools";pooldir.mkdir(exist_ok=True)
    for source,fn in sources:
        cache=pooldir/(source.replace("/","_")+".jsonl")
        if cache.exists():rows=list(audit.rows(cache))
        else:
            rows=[r for r in fn(LinkedSaver(source),6144,random.Random(f"scale:{SEED}:{source}")) if r]
            old.write(cache,rows)
        candidates+=rows;print("IMAGE_SOURCE",source,len(rows),flush=True)
    paths=sorted({str(MIX/p) for r in candidates for p in r["images"]})
    with ProcessPoolExecutor(6) as pool:hashed=list(pool.map(_hashed,paths,chunksize=64))
    hashes={str(Path(p).relative_to(MIX)):(s,d) for p,s,d in hashed}
    # Use every official source image, including unselected KIE rows, and the existing diagnostics.
    public=Path("runs/champion-v1/public-eval-v1/vision-suite")
    evalpaths={str(p) for p in public.glob("full/*/img/*") if p.is_file()}
    for r in audit.rows("runs/inference-policy-v1/examples.jsonl"):
        if r["modality"]=="vision":evalpaths.update(str(Path(p).resolve()) for p in r["images"])
    with ProcessPoolExecutor(6) as pool:ev=list(pool.map(_hashed,sorted(evalpaths),chunksize=64))
    evalsha={s for _,s,_ in ev};evaldh=[d for _,_,d in ev]
    order=list(hashes);close={order[i] for i in vm.near([hashes[p][1] for p in order],evaldh,4)}
    forbidden=close|{p for p,(s,_) in hashes.items() if s in evalsha}
    grouped=old.image_groups([r for r in candidates if not forbidden.intersection(r["images"])],hashes)
    reserved=[r for r in grouped if int(r["group"][:8],16)%5==0]
    training=[r for r in grouped if int(r["group"][:8],16)%5!=0]
    # Preserve every previously observed dev/diagnostic image as held out across candidates.
    oldreserved={p for r in audit.rows(PREVIOUS/"mix/image-dev.jsonl") for p in r["images"]}
    oldhashes=audit.read(PREVIOUS/"training-image-hashes.json")
    reserve_dh=[hashes[p][1] for r in reserved for p in r["images"]]+[oldhashes[p][1] for p in oldreserved]
    paths=sorted({p for r in training for p in r["images"]})
    near_dev={paths[i] for i in vm.near([hashes[p][1] for p in paths],reserve_dh,4)}
    training=[r for r in training if not near_dev.intersection(r["images"])]
    by=defaultdict(list)
    for r in training:by[r["source"]].append(r)
    quotas=allocate(N_IMAGE,{s:1.5 if s in ("cord","funsd") else 1. for s in by},{s:len(v) for s,v in by.items()})
    selected=[]
    for s,rs in sorted(by.items()):
        # Balance already-validated subtask/gold-position pairs without inventing negatives.
        bins=defaultdict(list)
        for r in sorted(rs,key=lambda r:old.digest("scale-image",r["id"])):bins[(r["subtask"],r["gold"])].append(r)
        ranked=[v[i] for i in range(max(map(len,bins.values()))) for _,v in sorted(bins.items()) if i<len(v)]
        selected += [old.subset_options(r,f"scale-options:{r['id']}")|{"modality":"image"} for r in ranked[:quotas[s]]]
    devby=defaultdict(list)
    for r in reserved:devby[r["source"]].append(r)
    dev=[r|{"modality":"image"} for s,rs in sorted(devby.items()) for r in sorted(rs,key=lambda r:old.digest("scale-image-dev",r["id"]))[:64]]
    selected_paths={p for r in selected for p in r["images"]}
    assert len(selected)==N_IMAGE and not selected_paths.intersection(forbidden|near_dev)
    old.write(dest,selected);old.write(MIX/"image-dev.jsonl",dev)
    train.atomic_json(ROOT/"training-image-hashes.json",{p:hashes[p] for p in sorted(selected_paths)})
    train.atomic_json(ROOT/"vision-audit.json",{"candidate_rows":len(candidates),"quotas":quotas,"train_rows":len(selected),
        "train_images":len(selected_paths),"dev_rows":len(dev),"official_and_diagnostic_images":len(evalpaths),
        "forbidden_candidate_images":len(forbidden),"dev_near_filtered_images":len(near_dev),"near_distance":4,
        "full_public_board_sha_and_dhash_audit_complete":True,"perceptual_method":"64-bit dHash, Hamming <=4; conservative approximate filter, not proof against every crop/edit",
        "train_option_counts":dict(Counter(len(r["options"]) for r in selected)),"source_splits":"Cauldron train + CORD train + FUNSD training_data; no evaluation labels or images for training"})


def run():
    if (ROOT/"prepared.json").exists():
        p=audit.read(ROOT/"prepared.json")
        assert p["source_sha256"]==audit.sha(__file__)
        for f,h in p["files"].items():assert audit.sha(f)==h
        return
    MIX.mkdir(parents=True,exist_ok=True);started=time.monotonic()
    prepare_text();prepare_images()
    texts=list(audit.rows(MIX/"text-train.jsonl"));images=list(audit.rows(MIX/"image-train.jsonl"))
    combined=texts+images;random.Random(SEED).shuffle(combined)
    # The preflight's real optimizer steps include the maximum text length, plus both modalities.
    longest=max(texts,key=lambda r:r["prepared_tokens"])
    combined=[longest,images[0]]+[r for r in combined if r["id"] not in {longest["id"],images[0]["id"]}]
    old.write(MIX/"train.jsonl",combined)
    for name in ("text-diagnostic","vision-diagnostic"):
        rows=list(audit.rows(PREVIOUS/"mix"/(name+".jsonl")))
        old.write(MIX/(name+".jsonl"),rows)
    files={str(p):audit.sha(p) for p in MIX.glob("*.jsonl")}
    train.atomic_json(ROOT/"prepared.json",{"complete":True,"rows":len(combined),"text_rows":len(texts),"image_rows":len(images),
        "files":files,"source_sha256":audit.sha(__file__),"base_sha256":audit.BASE_SHA,"seconds":time.monotonic()-started,
        "scope":"one scale-up candidate, not a causal ablation; original base; same CE/LoRA recipe, broader data and presentation coverage; both modalities evaluated"})
    print("PREPARED",len(combined),flush=True)


if __name__=="__main__":run()
