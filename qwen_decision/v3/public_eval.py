"""Frozen, resumable official public-panel comparison; no training or uploads.

Text: every counted panel; out-of-index requests are omitted explicitly.
Vision: all eleven panels, verified against the kit when available.
The kit's completeness fields and failed/unsupported requests are preserved.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from decision_index.suite.io import Suite, atomic_json, dumps, read_jsonl
from decision_index.scoring import index02

ROOT = Path("runs/champion-v1/public-eval-v1")
MODEL = "runs/champion-v1/joint-g7-v1/model"
ADAPTER = "qwen_decision.v3.public_eval_engines:"
JOBS = {
    "joint-text": (ADAPTER+"JointText", {"model":MODEL}),
    "gliclass-text": (ADAPTER+"GLiClassText", {}),
    "joint-vision": ("qwen_decision.v3.vision_engine:VisionEngine", {"model":MODEL,"prompt":"decide"}),
    "qev-vision": (ADAPTER+"QEVVision", {}),
}
ACTIVE_JOBS = ("joint-text", "joint-vision", "qev-vision")


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f,"sha256").hexdigest()


def counted_ids():
    return {n for a in index02.spec("0.3")["areas"] for n in a["benchmarks"]}


def payload_hash(state, questions):
    from decision_index.suite.build.layout import REFERENCE_MODEL
    return hashlib.sha256(dumps({"model":REFERENCE_MODEL,"state":state,"questions":questions}).encode()).hexdigest()


def prepare_text():
    """Audit actual input bytes, not only stored payload-hash metadata."""
    from decision_index.pipeline import score_run
    refs = json.loads((ROOT/"references.json").read_text())
    ref = next(x for x in refs if x["repo"].startswith("knowledgator") and x["file"].endswith("results.jsonl"))
    if sha(ref["local"]) != ref["sha256"]:
        raise ValueError("published reference changed")
    counted = counted_ids()
    published = {}
    for r in read_jsonl(ref["local"]):
        if int(r["catalog_id"]) not in counted:
            continue
        actual = payload_hash(r["payload"]["state"],r["payload"]["questions"])
        if actual != r["payload_sha256"]:
            raise ValueError(f"published payload metadata mismatch: {r['run_id']}")
        published[r["run_id"]] = r
    counts, statuses = collections.Counter(),collections.Counter()
    probes, seen = [],set()
    extremes = {}
    suite = Suite(ROOT/"text-suite","0.3")
    path = ROOT/"text-counted-rows.jsonl"
    temp = path.with_suffix(".tmp")
    with temp.open("w") as f:
        for row in suite.rows(apply_exclusions=True):
            e = row["_evaluation"]; n = int(e["catalog_id"]); rid = e["run_id"]
            if n not in counted:
                continue
            actual = payload_hash(row["state"],row["questions"])
            if actual != e["payload_sha256"] or actual != published[rid]["payload_sha256"]:
                raise ValueError(f"counted payload mismatch: {rid}")
            if rid in seen:
                raise ValueError(f"duplicate evaluation row: {rid}")
            seen.add(rid); counts[n]+=1; statuses[published[rid]["status"]]+=1
            f.write(dumps(row)+"\n")
            if counts[n] == 1:
                probes.append(row)
            measures = {"questions":len(row["questions"]),
                        "options":max(len(q["criteria"]) for q in row["questions"].values()),
                        "chars":len(dumps({"state":row["state"],"questions":row["questions"]}))}
            for key,value in measures.items():
                if key not in extremes or value > extremes[key][0]:
                    extremes[key]=(value,row)
    if set(counts) != counted:
        raise ValueError("missing counted panels")
    temp.replace(path)
    probes = {r["_evaluation"]["run_id"]:r for r in probes+[x[1] for x in extremes.values()]}
    (ROOT/"text-probes.jsonl").write_text("".join(dumps(r)+"\n" for r in probes.values()))
    report = {"panels":len(counted),"requests":sum(counts.values()),"counts":dict(counts),
        "actual_inputs_match_published_reference":True,"counted_reference_statuses":dict(statuses),
        "rows_sha256":sha(path),"probes_sha256":sha(ROOT/"text-probes.jsonl"),
        "whole_suite_hash_matches":False,
        "exception":"RouterBench (6) and SGD (10) differ; both outside the counted index. All counted inputs are independently hashed and matched.",
        "omitted_catalogs":sorted(set(int(n) for n in json.loads((ROOT/"text-payload-audit.json").read_text())["counts"])-counted),
        "extremes":{k:{"value":v,"run_id":r["_evaluation"]["run_id"]} for k,(v,r) in extremes.items()},
        "published_reference":ref,"reuse":"Author-published full predictions are rescored separately; local runs remain separate."}
    atomic_json(ROOT/"text-counted-verification.json",report)
    print(dumps({k:v for k,v in report.items() if k not in ("published_reference","counts")}),flush=True)
    scores = score_run(suite,ref["local"],"author-published gliclass-v4-edge",ROOT/"gliclass-published-score")
    expected = json.loads(next(Path(x["local"]).read_text() for x in refs if x["repo"].startswith("knowledgator") and x["file"].endswith("index.json")))
    agreement = {"recomputed":scores["decision_index"],"published":expected["index"],
                 "match":scores["decision_index"]==expected["index"],"coverage":{n:b["coverage"] for n,b in scores["index_benchmarks"].items() if b["in_index"]}}
    atomic_json(ROOT/"gliclass-published-score"/"parity.json",agreement)
    if not agreement["match"]:
        raise ValueError(f"official scorer parity failed: {agreement}")
    print(dumps({"published_scorer_parity":agreement["match"],"index":scores["decision_index"]}),flush=True)


def check_text(job):
    from decision_index.engines import load_engine, validate, Unsupported
    import torch
    engine_name,options = JOBS[job]
    engine = load_engine(engine_name,**options)
    rows = list(read_jsonl(ROOT/"text-probes.jsonl"))
    report = {"job":job,"probes":[],"native_parity":None}
    for row in rows:
        entry = {"run_id":row["_evaluation"]["run_id"],"questions":len(row["questions"])}
        try:
            result,_ = engine(state=row["state"],questions=row["questions"])
            validate(row["questions"],result)
            entry["status"]="ok"
        except Unsupported as e:
            entry.update(status="unsupported",reason=str(e))
        report["probes"].append(entry)
        print(dumps(entry),flush=True)
    if job == "joint-text":
        from qwen_decision.v3.engine import Scorer
        from qwen_decision.v3.public_eval_engines import PackedScorer
        if type(engine.scorer) is Scorer:
            report["native_parity"]={"engine":"unchanged native Scorer", "packed_optimization":False,
                "rejected_probe":"packed-native-rejected.json"}
            report["verified"]=True
            atomic_json(ROOT/f"{job}-preflight.json",report)
            engine.close()
            return
        # Same weights/prompts/readout; compare padding and independent varlen boundaries.
        # Use representative short requests to avoid quadratic SDPA memory in this parity probe.
        scorer = engine.scorer
        from qwen_decision.v3.engine import Scorer
        from qwen_decision.v3.modeling import VARLEN
        items=[]
        for row in rows:
            candidates=scorer.prompts(row["state"],row["questions"])
            items.extend(c for c in candidates[:2] if len(c[1])<=2048)
        items=items[:24]
        if len(items)<12:
            raise ValueError("too few native parity cases")
        packed=[]; native=[]
        for group in scorer.batches(items):
            batch=[items[i] for i in group]
            packed.extend(scorer.forward(batch))
        scorer.model.model.language_model.config._attn_implementation="sdpa"
        for group in scorer.batches(items):
            batch=[items[i] for i in group]
            native.extend(Scorer.forward(scorer,batch))
        scorer.model.model.language_model.config._attn_implementation=VARLEN
        decisions=sum(max(range(len(p)),key=p.__getitem__)==max(range(len(n)),key=n.__getitem__) for p,n in zip(packed,native))
        report["native_parity"]={"cases":len(items),"same_decisions":decisions,
            "max_probability_error":max(abs(a-b) for p,n in zip(packed,native) for a,b in zip(p,n))}
        if decisions != len(items):
            raise ValueError(f"packed/native decision mismatch: {report['native_parity']}")
    report["verified"]=True
    atomic_json(ROOT/f"{job}-preflight.json",report)
    engine.close()


def prepare_vision():
    from decision_index.vision.rows import write_rows,verify
    write_rows(ROOT/"vision-suite")
    report=verify(ROOT/"vision-suite")
    atomic_json(ROOT/"vision-verification.json",report)
    if not report["ok"]:
        raise ValueError("vision suite differs from official pinned images or requests")
    print(dumps({"vision_sets":len(report["sets"]),"vision_benchmarks":len(report["benchmarks"]),"official_hashes_match":True}),flush=True)


def check_vision(job):
    from decision_index.engines import load_engine, validate, Unsupported
    name,options=JOBS[job]
    engine=load_engine(name,image_root=str((ROOT/"vision-suite").resolve()),**options)
    probes={}; extremes={}
    for row in read_jsonl(ROOT/"vision-suite/vision-rows.jsonl"):
        probes.setdefault(row["benchmark"],row)
        values={"images":len(row["images"]),"options":len(row["questions"]["answer"]["criteria"])}
        for k,v in values.items():
            if k not in extremes or v>extremes[k][0]:extremes[k]=(v,row)
    chosen={r["id"]:r for r in [*probes.values(),*(r for _,r in extremes.values())]}
    report={"job":job,"probes":[],"verified":False}
    for row in chosen.values():
        e={"run_id":row["id"],"benchmark":row["benchmark"],"images":len(row["images"])}
        try:
            result,raw=engine(state=row["state"],questions=row["questions"],images=row["images"],prompt=row.get("prompt"))
            validate(row["questions"],result)
            e.update(status="ok",image_tokens=(raw or {}).get("image_tokens"))
        except Unsupported as ex:e.update(status="unsupported",reason=str(ex))
        report["probes"].append(e);print(dumps(e),flush=True)
    report["verified"]=True
    atomic_json(ROOT/f"{job}-preflight.json",report)
    engine.close()


def freeze(job):
    files = [Path(__file__),Path("qwen_decision/v3/public_eval_engines.py"),
             Path("qwen_decision/v3/leader_engines.py"),Path("qwen_decision/v3/engine.py"),
             Path("qwen_decision/v3/modeling.py"),Path("qwen_decision/v3/format.py")]
    if job.endswith("text"):
        files += [ROOT/"text-counted-rows.jsonl",ROOT/"text-counted-verification.json",ROOT/f"{job}-preflight.json"]
    else:
        files += [ROOT/"vision-suite/vision-rows.jsonl",ROOT/"vision-verification.json",ROOT/f"{job}-preflight.json",Path("qwen_decision/v3/vision_engine.py")]
        image_files={p for r in read_jsonl(ROOT/"vision-suite/vision-rows.jsonl") for p in r["images"]}
        files += sorted(ROOT/"vision-suite"/p for p in image_files)
    model = Path(MODEL if job.startswith("joint") else "models/leader-gliclass-v4-edge" if job.startswith("gliclass") else "models/leader-qev-0.8b")
    files += sorted(p for p in model.rglob("*") if p.is_file())
    files += sorted(p for p in Path("external/decision-index/decision_index").rglob("*") if p.suffix in (".py",".json"))
    if job.startswith("gliclass"):
        files += sorted(Path("external/leader-gliclass/gliclass").rglob("*.py"))
    if job.startswith("qev"):
        files += sorted(Path("external/leader-qev/src").rglob("*.py"))
        files += [Path("runs/champion-v1/leader-compare-v1/qev-backbone-cache.json")]
    plan={"job":job,"engine":JOBS[job],"files":{str(p):sha(p) for p in files},
          "protocol":"Every counted text panel or all 11 vision panels. Official kit scorers; no test-guided tuning; no training."}
    dest=ROOT/job/"plan.json";dest.parent.mkdir(exist_ok=True)
    if dest.exists() and json.loads(dest.read_text()) != json.loads(json.dumps(plan)):
        raise ValueError(f"immutable plan mismatch: {job}")
    atomic_json(dest,plan)


def verify_plan(job):
    plan=json.loads((ROOT/job/"plan.json").read_text())
    for p,h in plan["files"].items():
        if sha(p)!=h:raise ValueError(f"frozen input changed: {p}")


def run(job):
    verify_plan(job)
    engine,options=JOBS[job]
    if job.endswith("text"):
        from decision_index.runner import run as runner
        from decision_index.pipeline import score_run
        runner(engine,options,ROOT/"text-counted-rows.jsonl",ROOT/job,compact=True,
               corpus_sha256=sha(ROOT/"text-counted-rows.jsonl"))
        scores=score_run(Suite(ROOT/"text-suite","0.3"),ROOT/job/"results.jsonl",job,ROOT/job)
        valid={n:b for n,b in scores["index_benchmarks"].items() if b["in_index"]}
        atomic_json(ROOT/job/"counted-completeness.json",{
            "panels":len(valid),"all_counted_rows_visited":scores["completed"]==json.loads((ROOT/"text-counted-verification.json").read_text())["requests"],
            "all_counted_rows_answered":all(b["coverage"]==1 for b in valid.values()),
            "kit_whole_suite_complete":scores["complete"],"note":"Only counted panels requested; original kit whole-suite completeness retained unchanged."})
    else:
        from decision_index.vision.run import run as runner
        from decision_index.vision.score import score_run
        runner(engine,options,ROOT/"vision-suite",ROOT/job,workers=1,chunk=16,compact=True)
        scores=score_run(ROOT/"vision-suite/vision-rows.jsonl",ROOT/job/"results.jsonl",ROOT/job)
    verify_plan(job)
    counts=collections.Counter(r["status"] for r in read_jsonl(ROOT/job/"results.jsonl"))
    atomic_json(ROOT/job/"completion.json",{"job":job,"finished":True,"frozen_inputs_verified":True,
        "counts":dict(counts),"public_index":scores.get("decision_index",scores.get("public_vision_index")),
        "errors_require_review":bool(counts["error"]),"official_full_rank":None})
    if counts["error"]:
        raise RuntimeError(f"{job} has runtime errors; inspect before interpretation")
    summarize()


def summarize():
    out={}
    for job in JOBS:
        p=ROOT/job/("scores.json" if job.endswith("text") else "vision-scores.json")
        if p.exists():out[job]=json.loads(p.read_text())
    if "gliclass-text" not in out:
        p=ROOT/"gliclass-published-score/scores.json"
        if p.exists():
            out["gliclass-text"]=json.loads(p.read_text())
            out["gliclass-text"]["comparison_source"]="author-published predictions, rehashed inputs and official rescore; not a fresh local full run"
    atomic_json(ROOT/"comparison.json",out)
    lines=["# 공식 공개 평가 비교", "", "목표는 1B 이하에서 텍스트와 비전 모두 1위다. 이 결과는 공개 평가이며 비공개를 포함한 Full 순위는 아니다.",
           "", "텍스트는 점수에 포함되는 37개 패널의 전체 문항을 사용한다. 문항별 실제 입력을 공개 참조 결과와 해시 대조했다. 점수에 반영되지 않는 과제는 생략했으므로 키트의 전체 suite complete=false는 그대로 둔다.",
           "", "텍스트 선두는 저자의 공개 전체 예측을 입력 대조 후 공식 재채점한 값이다. 로컬 GLiClass의 장문 preflight는 가용 GPU 메모리를 초과했다. 이전 직접 실행 진단과 구분한다.",
           "", "|실행|공개 지수|키트 complete|", "|---|---:|---|"]
    for job,s in out.items():
        lines.append(f"|{job}|{s.get('decision_index',s.get('public_vision_index'))}|{s['complete']}|")
    for modality,a,b in (("텍스트","joint-text","gliclass-text"),("비전","joint-vision","qev-vision")):
        if a not in out or b not in out:continue
        key="index_benchmarks" if modality=="텍스트" else "benchmarks"
        metric="skill" if modality=="텍스트" else "accuracy"
        gaps=[]
        for name,x in out[a][key].items():
            if key=="index_benchmarks" and not x["in_index"]:continue
            y=out[b][key][name]
            gaps.append((100*(x[metric]-y[metric]),name,x[metric],y[metric]))
        lines += ["",f"## {modality} 과제별 차이", "",f"|과제|우리 {metric}|선두 {metric}|차이(pp)|","|---|---:|---:|---:|"]
        lines += [f"|{n}|{100*x:.2f}|{100*y:.2f}|{d:+.2f}|" for d,n,x,y in sorted(gaps)]
    lines += ["", "원본 결과·검증·불변 실행 계획: `runs/champion-v1/public-eval-v1/`. 학습 데이터 증량 또는 학습 방식 변경은 이 평가를 확인한 뒤 결정한다.", ""]
    Path("wiki/project/champion-public-comparison.md").write_text("\n".join(lines))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("command",choices=["prepare-text","prepare-vision","check","freeze","run","summarize","pipeline"])
    p.add_argument("--job",choices=list(JOBS))
    args=p.parse_args()
    if args.command=="prepare-text":prepare_text()
    elif args.command=="prepare-vision":prepare_vision()
    elif args.command=="summarize":summarize()
    elif args.command=="pipeline":
        if not json.loads((ROOT/"gliclass-published-score/parity.json").read_text())["match"]:
            raise ValueError("reference scorer parity is required")
        for job in ACTIVE_JOBS:
            for command in ("check","freeze"):
                subprocess.run([sys.executable,"-u","-m","qwen_decision.v3.public_eval",command,"--job",job],check=True)
        for job in ACTIVE_JOBS:
            subprocess.run([sys.executable,"-u","-m","qwen_decision.v3.public_eval","run","--job",job],check=True)
    else:
        import torch
        torch.set_num_threads(4)
        {"check":check_text if args.job.endswith("text") else check_vision,"freeze":freeze,"run":run}[args.command](args.job)


if __name__=="__main__":main()
