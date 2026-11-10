"""Finish current evaluation, then preflight/freeze/train/evaluate a joint scale-up."""
from __future__ import annotations
import collections
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT=Path("runs/champion-v1/joint-g7-v2")
PUBLIC=Path("runs/champion-v1/public-eval-v1")


def read(path):return json.loads(Path(path).read_text())


def write(path,data):
    path=Path(path);temp=path.with_suffix(".tmp");temp.write_text(json.dumps(data,ensure_ascii=False,indent=2)+"\n");temp.replace(path)


def status(stage,**extra):
    r={"stage":stage,"pid":os.getpid(),"updated_unix":time.time(),**extra}
    write(ROOT/"pipeline.json",r);print(json.dumps(r,ensure_ascii=False),flush=True)


def unit_state(name):
    text=subprocess.check_output(["systemctl","--user","show",name,"-p","ActiveState","-p","Result"],text=True)
    return dict(line.split("=",1) for line in text.splitlines() if "=" in line)


def analyze():
    from decision_index.scoring import index02
    from decision_index.suite.io import read_jsonl,Suite
    from decision_index.vision import spec as vision_spec
    from qwen_decision.v3.champion_scale_data import BOOST
    ours=read(PUBLIC/"joint-text/scores.json");leader=read(PUBLIC/"gliclass-published-score/scores.json")
    spec=index02.spec("0.3");aw=index02.area_weights(spec);gaps=[]
    for area in spec["areas"]:
        bw=index02.bench_weights(area["benchmarks"],spec)
        for n in area["benchmarks"]:
            a,b=ours["index_benchmarks"][str(n)],leader["index_benchmarks"][str(n)]
            info=ours["benchmarks"][str(n)]
            gaps.append({"catalog":n,"dataset":info["dataset"],"metric":info["metric"],
                "ours":info["score"],"leader":leader["benchmarks"][str(n)]["score"],
                "weighted_index_gap":100*aw[area["id"]]*bw[n]*(b["skill"]-a["skill"])})
    results={r["run_id"]:r for r in read_jsonl(PUBLIC/"joint-text/results.jsonl")}
    counts=collections.defaultdict(collections.Counter)
    for r in Suite(PUBLIC/"text-suite","0.3").rows(apply_exclusions=True):
        e=r["_evaluation"];n=e["catalog_id"]
        if n not in (20,40,41,56,61):continue
        res=results[e["run_id"]]["response"]["answers"]
        for key,q in r["questions"].items():
            gold=r["expected"].get(key)
            if gold is None:continue
            answer=res[key];pred=answer["choice"] if q["type"]=="choice" else answer["noul"]>=.5
            counts[f"{n}:{key}"][f"gold:{gold}"]+=1
            counts[f"{n}:{key}"][f"prediction:{pred}"]+=1
            counts[f"{n}:{key}"]["correct" if gold==pred else "incorrect"]+=1
    va=read(PUBLIC/"joint-vision/vision-scores.json");vb=read(PUBLIC/"qev-vision/vision-scores.json")
    weights={b["name"]:b["weight"] for b in vision_spec.BENCHMARKS};den=sum(weights.values())
    vision_gaps=[]
    for name,a in va["benchmarks"].items():
        b=vb["benchmarks"][name]
        vision_gaps.append({"dataset":name,"ours_accuracy":a["accuracy"],"leader_accuracy":b["accuracy"],
            "weighted_index_gap":100*weights[name]/den*(b["skill_clipped"]-a["skill_clipped"])})
    report={"text_public":{"ours":ours["decision_index"],"leader_published":leader["decision_index"],
            "weighted_gaps":sorted(gaps,key=lambda r:-r["weighted_index_gap"]),"predicted_label_counts":dict(counts)},
        "vision_public":{"ours":va["public_vision_index"],"leader_local":vb["public_vision_index"],
            "weighted_gaps":sorted(vision_gaps,key=lambda r:-r["weighted_index_gap"])},
        "prespecified_source_boost":BOOST,
        "interpretation":"~147 text rows/source pilot; strong request-format dependence and class collapse; not evidence that an alignment rotation or temperature is the missing intervention.",
        "training_boundary":"Use g7 train annotations only. No official evaluation item, error-case label or leader prediction becomes a training label. Public scores guide task-family priorities, so this is development, not blind confirmation."}
    write(ROOT/"weakness-analysis.json",report)
    return report


def publish():
    if not (ROOT/"weakness-analysis.json").exists():return
    a=read(ROOT/"weakness-analysis.json")
    lines=["# 두 모달리티 공동 학습 확대", "", "목표는 Jev Decision Index에서 1B 이하 텍스트·비전 모두 1위다. 이 실행은 원본 Qwen에서 시작하는 단일 후보이며 공식 비공개 Full 결과가 아니다.",
        "", "## 기존 파일럿에서 확인한 부족한 부분", "", "|텍스트 과제|지표|우리|선두|공개 지수 격차 기여|", "|---|---|---:|---:|---:|"]
    for r in a["text_public"]["weighted_gaps"][:12]:
        lines.append(f"|{r['dataset']}|{r['metric']}|{r['ours']:.4f}|{r['leader']:.4f}|{r['weighted_index_gap']:.2f}|")
    lines += ["", "피싱은 같은 이메일의 noul 정확도68.45%와 주 선택지 정확도51.35%가 크게 달랐다. 주 선택지에서는2,000건 중1,971건을 정상으로 골랐다. HoVer는4,000건 중3,741건을 supported로, VAST는3,006건 중103건만 neutral로 골랐다. 과제 전체 격차와 함께 표현 형식·클래스별 예측 쏠림을 개선 대상으로 삼는다.",
        "", "## 고정한 다음 후보", "", "원본 공식 Qwen에서 텍스트131,072 + 이미지32,768개를2epoch 학습한다. 매 update는 텍스트16·이미지4개이고 총16,384 update다. 기존 CE·all LM LoRA r16/alpha32/dropout.05·ViT 동결·FP32-master AdamW를 유지한다. LR2e-4, warmup512, cosine to10%다.",
        "", "g7의113개 출처를 기반으로 약한 과제군에 표본을 더 배분하고 가능한 클래스·질문 polarity를 균형화한다. 기존 문맥을 질문으로 이동하거나 의미를 유지한 선택지 설명에 임의 키를 붙인다. 정답·참/거짓 방향·기존 선택지 내용은 바꾸지 않는다. 이 보강의 목표는 표현 형식 이동이며 회전 정렬을 학습 손실로 추가하지 않는다.",
        "", "이미지는 기존12개 Cauldron train 출처에 기존 변환기가 지원하는 CORD train·FUNSD training_data를 추가한다. 공식 전체 이미지와 기존 진단 이미지에 SHA256 및 dHash 거리4 중복 검사를 적용하고 dev 이미지와도 분리한다. 이 근사 필터가 모든 크롭·편집 중복의 부재를 증명하지는 않는다.",
        "", "기존 공개 평가를 마친 뒤 같은 GPU에서 사전 점검→불변 계획→학습→저장/복원 검증→텍스트37개·비전11개 공개 재평가를 순서대로 실행한다. 고정 최종 update를 사용하며 공개 점수로 중간 checkpoint를 고르지 않는다. 입력·가중치·코드 해시와 재개용 optimizer/master/RNG를 보존한다.",
        "", "## 상태와 결과", ""]
    if (ROOT/"pipeline.json").exists():lines.append("서비스 단계: `"+read(ROOT/"pipeline.json")["stage"]+"`.")
    for modality,name,key in (("텍스트","scores.json","decision_index"),("비전","vision-scores.json","public_vision_index")):
        p=ROOT/("public-"+("text" if modality=="텍스트" else "vision"))/name
        if p.exists():lines.append(f"새 후보 {modality} 공개 지수: **{read(p)[key]:.2f}**.")
    lines += ["", "개별 변경의 효과를 분리하는 대조 실험이 아닌 묶음 개선 후보다. 공개 평가를 관찰해 우선순위를 정했으므로 blind 확인 결과로 부르지 않는다. 선두 가중치·예측은 학습 초기값이나 교사로 사용하지 않는다. 새 후보의 텍스트·비전을 모두 재고, 공식 Full 순위는 최종 비공개 평가 후에만 판정한다.",
              "", "실행 원본: `runs/champion-v1/joint-g7-v2/`. 서비스: `champion-scale-v2`. 원본 비교: [공식 공개 평가](champion-public-evaluation.md)."]
    Path("wiki/project/champion-joint-scale.md").write_text("\n".join(lines)+"\n")


def main():
    with (ROOT/"pipeline.lock").open("w") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            deadline=time.monotonic()+21600
            while True:
                prep=unit_state("champion-scale-prepare-v2")
                previous=unit_state("champion-public-eval-v1c")
                ready=(ROOT/"prepared.json").exists()
                evaldone=all((PUBLIC/job/"completion.json").exists() for job in ("joint-text","joint-vision","qev-vision"))
                if prep.get("ActiveState")=="failed":raise RuntimeError("Input preparation failed; inspect prepare.log")
                if previous.get("ActiveState")=="failed":raise RuntimeError("Previous public evaluation failed; preserve evidence before training")
                if ready and evaldone and previous.get("ActiveState") not in ("active","activating","deactivating"):
                    break
                if time.monotonic()>deadline:raise RuntimeError("Preparation/evaluation exceeded six-hour queue allowance")
                status("queued",inputs_prepared=ready,previous_public_evaluation_complete=evaldone)
                time.sleep(30)
            for job in ("joint-text","joint-vision","qev-vision"):
                c=read(PUBLIC/job/"completion.json");assert c["frozen_inputs_verified"] and not c["errors_require_review"]
            analyze();publish()
            for stage in ("check","freeze","run","eval-text","eval-vision"):
                status(stage)
                with (ROOT/(stage+".log")).open("a") as log:
                    subprocess.run([sys.executable,"-u","-m","qwen_decision.v3.champion_scale_train",stage],
                                   stdout=log,stderr=subprocess.STDOUT,check=True)
                publish()
            assert read(ROOT/"report.json")["export_replay_exact"]
            assert all(read(ROOT/("public-"+m)/"completion.json")["complete"] for m in ("text","vision"))
            status("complete",verified=True,training_complete=True,both_public_evaluations_complete=True)
            publish()
        except BaseException:
            status("failed",traceback=traceback.format_exc());publish();raise


if __name__=="__main__":main()
