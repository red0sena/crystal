"""Pinned native-code leader vs exported joint-pilot comparison on identical diagnostic rows.

These diagnostics are already observed, not official Full or a new blind evaluation.
No training, teacher targets, tuning, uploads, or paid compute.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import gc
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import statistics
import subprocess
import time
import traceback

import torch
from decision_index.engines import Unsupported
from qwen_decision.v3 import format as fmt, leader_engines as native

ROOT = Path("runs/champion-v1/leader-compare-v1")
JOINT = Path("runs/champion-v1/joint-g7-v1")
EXAMPLES = Path("runs/inference-policy-v1/examples.jsonl")
PANELS = {"text": "text-diagnostic", "vision": "vision-diagnostic"}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8*1024**2), b""): h.update(block)
    return h.hexdigest()


def read(path): return json.loads(Path(path).read_text())
def rows(path): return [json.loads(line) for line in Path(path).open()]


def atomic(path, value):
    path = Path(path); tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+"\n"); tmp.replace(path)


def status(stage, **extra): atomic(ROOT/"status.json", {"stage":stage,"updated_unix":time.time(),**extra})


def response_values(question, response):
    answer = response["answers"]["answer"]
    keys,_ = fmt.options(question)
    if question["type"] == "noul":
        p = float(answer["noul"]); probabilities = [1-p,p]; prediction = int(p >= .5)
    else:
        assert answer["type"] == "choice" and set(answer["probabilities"]) == set(keys)
        probabilities = [float(answer["probabilities"][k]) for k in keys]
        prediction = keys.index(answer["choice"])
        assert probabilities[prediction] >= max(probabilities)-1e-7
    assert all(math.isfinite(p) and 0 <= p <= 1 for p in probabilities)
    assert abs(sum(probabilities)-1) < 1e-5
    return prediction, probabilities


def build(modality): return native.GLiClassNative() if modality=="text" else native.qev_vision()


def smoke_samples(candidates, modality):
    if modality=="text":
        sample=[next(r for r in candidates if r["question"]["type"]==kind) for kind in ("choice","noul")]
        largest=max(candidates,key=lambda r:len(fmt.options(r["question"])[0]))
        if largest["id"] not in {r["id"] for r in sample}:sample.append(largest)
        return sample
    return [next(r for r in candidates if len(r["images"])==n) for n in sorted({len(r["images"]) for r in candidates})]


def infer(engine, row):
    kwargs = {"images":row["images"]} if row["modality"]=="vision" else {}
    state = {} if kwargs else row["state"]
    response,_ = engine(state,{"answer":row["question"]},**kwargs)
    prediction,probabilities = response_values(row["question"],response)
    return prediction,probabilities,response.get("usage",{})


def check():
    torch.set_num_threads(4)
    examples = rows(EXAMPLES); report = {}
    for modality in PANELS:
        engine = build(modality)
        count = sum(p.numel() for p in engine.m.parameters()); assert count < 10**9
        candidates = [r for r in examples if r["modality"]==modality]
        sample=smoke_samples(candidates,modality)
        cases=[]
        for row in sample:
            started=time.monotonic();prediction,probabilities,usage=infer(engine,row)
            cases.append({"id":row["id"],"question_type":row["question"]["type"],"prediction":prediction,
                          "probabilities":probabilities,"usage":usage,"seconds":time.monotonic()-started})
        report[modality]={"parameters":count,"provenance":engine.provenance,"cases":cases}
        print("NATIVE_CHECK",modality,report[modality],flush=True)
        del engine;gc.collect();torch.cuda.empty_cache()
    atomic(ROOT/"check.json",{"passed":True,"panels":report,"source_sha256":sha(__file__),"adapter_sha256":sha(native.__file__)})


def freeze():
    assert not (ROOT/"plan.json").exists()
    checked=read(ROOT/"check.json");assert checked["passed"]
    assert checked["source_sha256"]==sha(__file__) and checked["adapter_sha256"]==sha(native.__file__)
    board=ROOT/"sources/multimodalart__jev-decision-index"
    text,vision=read(board/"data/index.json"),read(board/"data/vision.json")
    eligible=[m for m in text["models"] if 0 < ((m.get("meta") or {}).get("served_params") or 0) <= 10**9
              and (m.get("v03") or {}).get("v") is not None]
    top_text=max(eligible,key=lambda m:m["v03"]["v"])
    top_vision=max((m for m in vision["entrants"] if 0 < (m.get("params") or 0) <= 10**9),key=lambda m:m["full"])
    assert top_text["engine"]=="gliclass-v4-edge" and top_vision["engine"]=="qev-0.8b"
    inputs={str(EXAMPLES):sha(EXAMPLES)}
    for row in rows(EXAMPLES):
        if row["modality"]!="vision":continue
        assert len(row["images"])==len(row["image_sha256"])
        for path,digest in zip(row["images"],row["image_sha256"]):
            if path not in inputs:assert sha(path)==digest;inputs[path]=digest
    for name in PANELS.values():
        p=JOINT/f"exported-{name}.jsonl"; inputs[str(p)]=sha(p)
    model_files={}
    for download in read(ROOT/"downloads.json"):
        for name,value in download["files"].items():
            p=Path(download["directory"])/name
            assert sha(p)==value["sha256"]
            model_files[str(p)]=value["sha256"]
    commits={str(p):subprocess.check_output(["git","-C",str(p),"rev-parse","HEAD"],text=True).strip()
             for p in (Path("external/leader-gliclass"),Path("external/leader-qev"))}
    code={str(p):sha(p) for p in (Path(__file__),Path(native.__file__))}
    code.update({str(p):sha(p) for repo in commits for p in Path(repo).rglob("*.py")})
    env={name:importlib.metadata.version(name) for name in ("torch","transformers","pydantic","accelerate","numpy","pillow")}
    plan={"models":read(ROOT/"downloads.json"),"leaderboard_revision":read(board/"repo-record.json")["revision"],
          "leaderboard":{"text":{"engine":top_text["engine"],"full":top_text["v03"]["v"]},
                         "vision":{"engine":top_vision["engine"],"full":top_vision["full"]}},
          "inputs":inputs,"models_sha256":model_files,"source_sha256":code,"source_commits":commits,
          "backbone_cache":read(ROOT/"qev-backbone-cache.json"),"check":checked,"environment":env,
          "python":platform.python_version(),"gpu":torch.cuda.get_device_name(0),"counts":{"text":1335,"vision":2184},
          "metric":"source macro accuracy, unsupported included as wrong; conditional NLL and coverage also stored",
          "scope":"already-observed internal diagnostics; native author code in local environment, not an official-board reproduction; no latency rank claim; cached exported joint-model predictions checked by ID and gold",
          "dependencies":"separate target vendor; main environment unchanged; Python3.13/transformers5.18 differ from QEV package declared3.12/5.17; native outputs smoke-checked, not board prediction-agreement certified",
          "training":False,"teacher_targets":False,"automatic_upload":False}
    atomic(ROOT/"plan.json",plan)
    print("FROZEN",plan["leaderboard"],plan["counts"],flush=True)


def verify():
    plan=read(ROOT/"plan.json")
    for key in ("inputs","models_sha256","source_sha256"):
        for path,digest in plan[key].items():assert sha(path)==digest,path
    return plan


def summarize(records):
    by=defaultdict(list)
    for r in records:by[r["source"]].append(r)
    sources={}
    for source,items in by.items():
        sources[source]={"rows":len(items),"leader_accuracy":statistics.mean(x["correct"] for x in items),
                         "joint_accuracy":statistics.mean(x["joint_correct"] for x in items),
                         "coverage":statistics.mean(x["status"]=="ok" for x in items),
                         "leader_only_correct":sum(x["correct"] and not x["joint_correct"] for x in items),
                         "joint_only_correct":sum(x["joint_correct"] and not x["correct"] for x in items)}
        good=[x["nll"] for x in items if x["status"]=="ok"]
        sources[source]["answered_nll"]=statistics.mean(good) if good else None
    result={"rows":len(records),"sources":sources,"leader_accuracy":statistics.mean(x["leader_accuracy"] for x in sources.values()),
            "joint_accuracy":statistics.mean(x["joint_accuracy"] for x in sources.values()),
            "unsupported":sum(x["status"]!="ok" for x in records)}
    result["joint_minus_leader_pp"]=100*(result["joint_accuracy"]-result["leader_accuracy"])
    return result


def publish(report):
    lines=["# 1B 이하 선두 모델 직접 비교", "", "동일한 내부 진단 문항에 다운로드한 선두 모델의 공식 추론 코드를 적용했다. 공식 Full이나 비공개 평가 재현은 아니다.", "",
           "| 입력 | 선두 모델 | 선두 정확도 | 공동 파일럿 정확도 | 차이 (%p) |", "|---|---|---:|---:|---:|"]
    for modality,name in (("text","GLiClass v4 Edge"),("vision","QEV-0.8B")):
        r=report["panels"][modality]
        lines.append(f"| {modality} | {name} | {r['leader_accuracy']*100:.2f}% | {r['joint_accuracy']*100:.2f}% | {r['joint_minus_leader_pp']:+.2f} |")
    lines += ["", "텍스트1,335문항·이미지2,184문항, 출처 평균 정확도다. 미지원 문항은 오답으로 포함하며 실제 분모·coverage·출처별 차이는 JSON에 남겼다. 모델별 공식 입력 형식을 유지했고 평가 라벨로 보정하거나 재학습하지 않았다.", "",
              "GLiClass는 배포된 보정기를 포함한 모델 카드의 SystemOne 매핑, QEV는 accurate 모드·최대1.6MP·전체 깊이의 저자 kit engine을 사용했다. 원본 파일·버전·코드·환경은 고정했다. 로컬 Python3.13/transformers5.18 환경은 QEV 패키지의 선언된3.12/5.17과 다르므로 보드와 동일한 수치 재현을 주장하지 않는다.", "",
              "목표는 텍스트·이미지 두 트랙의1B 이하 전체1위다. 이 비교는 학습 과제와 형식의 부족을 찾는 진단이며 최종 순위 확인은 공식 평가가 필요하다.", "",
              "[상세 결과](champion-leader-comparison-results.json) · [프로젝트](champion-v1.md). 실행 증거: `runs/champion-v1/leader-compare-v1/`."]
    Path("wiki/project/champion-leader-comparison.md").write_text("\n".join(lines)+"\n")
    atomic("wiki/project/champion-leader-comparison-results.json",report)


def run():
    torch.set_num_threads(4);plan=verify();examples=rows(EXAMPLES);reports={}
    for modality,panel in PANELS.items():
        dataset=[r for r in examples if r["modality"]==modality]
        assert len(dataset)==plan["counts"][modality]
        joint={r["id"]:r for r in rows(JOINT/f"exported-{panel}.jsonl")}
        assert set(joint)=={r["id"] for r in dataset}
        output=ROOT/f"{modality}-predictions.jsonl"
        prior=rows(output) if output.exists() else []
        assert [r["id"] for r in prior]==[r["id"] for r in dataset[:len(prior)]]
        engine=build(modality);records=list(prior);started=time.monotonic()
        with output.open("a",buffering=1) as handle:
            for index,row in enumerate(dataset[len(prior):],len(prior)+1):
                reference=joint[row["id"]];assert reference["gold"]==[row["gold"]]
                record={"id":row["id"],"source":row["source"],"group":row["group"],"split":row["split"],
                        "gold":row["gold"],"joint_correct":reference["correct"],"joint_prediction":reference["prediction"]}
                before=time.monotonic()
                try:
                    prediction,probabilities,usage=infer(engine,row)
                    record.update(status="ok",prediction=prediction,probabilities=probabilities,usage=usage,
                                  correct=prediction==row["gold"],nll=-math.log(max(probabilities[row["gold"]],1e-12)))
                except Unsupported as error:
                    record.update(status="unsupported",correct=False,nll=None,reason=str(error))
                record["milliseconds"]=(time.monotonic()-before)*1000
                records.append(record);handle.write(json.dumps(record,ensure_ascii=False)+"\n")
                if index%50==0 or index==len(dataset):
                    status("native_inference",modality=modality,rows=index,total=len(dataset),seconds=time.monotonic()-started)
                    print("INFER",modality,index,len(dataset),flush=True)
        reports[modality]=summarize(records)
        atomic(ROOT/f"{modality}-summary.json",reports[modality])
        del engine;gc.collect();torch.cuda.empty_cache()
    report={"complete":True,"panels":reports,"plan_sha256":sha(ROOT/"plan.json"),"scope":plan["scope"]}
    verify();atomic(ROOT/"report.json",report);publish(report);status("complete",verified=True)
    print("COMPLETE",{k:{f:v[f] for f in ("leader_accuracy","joint_accuracy","joint_minus_leader_pp")} for k,v in reports.items()},flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("command",choices=("check","freeze","run"));args=parser.parse_args()
    status(args.command)
    try:globals()[args.command]()
    except Exception:status("failed",command=args.command,traceback=traceback.format_exc());raise


if __name__=="__main__":main()
