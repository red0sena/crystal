# 2차 학습 데이터 출처 (오염 검사 포함)

[위키 홈](../README.md) · [작업 기록](../project/v3-log.md) · [전략](beat-d3-lite.md)

**한국어 요약(2026-10-16).** 비공개 same-skills와 new-domains에 대응하려고, index 시험 문항이 아닌 같은 역량의 공개 데이터를 골랐다. 아래 영문 본문은 조사 단계의 계획 문서다. 데이터셋별로 index와 JEVal이 실제로 쓰는 split, 라이선스, 오염 함정을 다룬다. 이 계획에 따라 **실제로 만든 출처**는 다음과 같다. 빌더는 `qwen_decision/v3/sources/`, 데이터는 `data/phase2/`에 있다. 모든 출처가 [오염 필터](../project/v3-log.md)를 거쳤다. "버림"은 필터가 뺀 수이고, 출처별 manifest에 사유 예시가 있다.

**갱신(2026-10-19):** 아래 표는 `data/phase2/` manifest 60개에서 다시 뽑았다. 처음 37개 이후 다음이 추가됐다.
- 실제 장문(lexglue·cuad·quality·musique), 키-값 찾기(kv-lookup), API-Bank 재구성.
- 금융(SEntFiN·TAT-QA).
- 3차(Social IQa, Cosmos QA, MultiNLI, FaithDial, Moral Stories, e-CARE, GoEmotions).
- 4차(TAT-QA 예/아니오, 예시 기반 분류, WNED, VitaminC, PAWS, SciTail, 금융 트윗).
- 5차(2026-10-19, `phase5.py`): TabFact, CondaQA, ROPES, FOLIO, ABCD, HelpSteer2, FiNER-139, Zeshel, Few-NERD, DROP, Taskmaster-1.
- 6차(2026-10-19, `phase6_{pref,reason,domain,lang}.py`): 선호·판단 9, 읽기·추론 11, 전문 분야·대화 9, 개체·언어 8. 상세는 [작업 기록](../project/v3-log.md).
- 7차(2026-10-19, `poetry_pairs.py`): 시 최소쌍(BPoMP와 같은 역량, 퍼블릭 도메인 시로 직접 생성). 계획 A7을 구현했다. 상세는 [작업 기록](../project/v3-log.md)의 "7차: 시 최소쌍".
- 7차 예외(2026-10-19, `minicheck.py`): MiniCheck C2D/D2C. 모델이 만든 정답이지만 사용자 결정으로 허용했다. 상세는 [작업 기록](../project/v3-log.md)의 "MiniCheck 예외 허용".
- 7차 입장·상식(2026-10-19, `phase7.py`): EZ-STANCE, P-Stance, args.me(정리된 토론 사이트 3곳), CommonsenseQA 2.0. 상세는 [작업 기록](../project/v3-log.md)의 "7차: 입장과 CommonsenseQA 2.0".
- 7차 새 분야(2026-10-19, `phase7_domains.py`): Aegis 2.0 프롬프트 안전성, Scruples 커뮤니티 윤리 판단, NVD CVE 취약점 분류. 상세는 [작업 기록](../project/v3-log.md)의 "7차: 새 분야".

"정답 출처"는 `mix.SOURCES`의 label_origin이고, "v3-gold6 사용"은 다음 학습 셋의 행 수다(g3는 v3-gold2, g4b는 v3-gold3을 썼다; 각 manifest 참조). 2026-10-19에 4차 출처 4개(tatqa-yesno, icl-classify, vitaminc-train, fintweet-train)를 감사 결과대로 다시 만들었다([작업 기록](../project/v3-log.md)의 "v3-gold4 데이터 감사 반영"). 이전 파일은 `data/phase2/archive/gold4-build/`에 있다.
HellaSwag·WinoGrande는 2026-10-18부터 train 전체를 쓴다. 이전 1.5만 개 파일과 manifest는 `data/phase2/archive/`에 있다.

| 출처 | 문항 | 버림 | 필터 모드 | 정답 출처 | v3-gold6 사용 | 라이선스(manifest) |
|---|---:|---:|---|---|---:|---|
| abcd-train | 37,214 | 0 | strict | human | 9,657 | MIT |
| acos-train | 22,644 | 0 | strict | human | 7,513 | no licence file (research release; SemEval-2016 restaurant r |
| aegis-prompts | 27,256 | 40 | strict | human | — (v3-gold7부터) | CC-BY-4.0 (nvidia Aegis 2.0; prompt labels only, LLM-jury response labels not used) |
| apibank-train | 8,463 | 183 | strict | model (도구 예외 허용) | 4,541 | MIT |
| aqua-rat-train | 14,141 | 859 | strict | human | 5,897 | Apache-2.0 |
| arc-train | 3,341 | 29 | strict | human | 2,770 | CC-BY-SA-4.0 |
| arena55k-train | 35,048 | 3,690 | strict | human | 9,369 | Apache-2.0 |
| argq-train | 20,000 | 0 | strict | human | 7,108 | CC-BY-SA-3.0 (card licensing text; YAML says cc-by-3.0) |
| argsme-train | 9,023 | 4 | strict | human | — (v3-gold7부터) | CC-BY-4.0 (webis/args_me; idebate·debatewise·debatepedia만) |
| banking77-train | 9,961 | 17 | strict | human | 9,661 | CC-BY-4.0 |
| boolq-train | 9,000 | 427 | strict | human | 4,684 | CC-BY-SA-3.0 |
| chid-train | 19,986 | 14 | strict | human | 7,052 | Apache-2.0 |
| civilcomments-train | 34,964 | 36 | strict | human | 9,358 | CC0-1.0 |
| claimstance-train | 1,038 | 1 | strict | human | 895 | CC-BY-SA-3.0 (card licensing text; YAML says cc-by-3.0) |
| clinc-train | 18,078 | 61 | strict | human | 6,702 | CC-BY-3.0 |
| cmrc-train | 14,931 | 69 | strict | human | 6,079 | CMRC2018 CC-BY-SA-4.0; DRCD CC-BY-SA-3.0 |
| code-output-gen | 4,999 | 1 | strict | programmatic | 3,445 | ours (generated) |
| commonsenseqa-train | 9,619 | 0 | strict | human | 4,852 | MIT |
| condaqa-train | 6,768 | 52 | strict | human | 4,046 | Apache-2.0 |
| conditionalqa-train | 1,083 | 34 | strict | human | — | CC-BY-SA-4.0 |
| contractnli-train | 5,049 | 2,652 | documents | human | 4,743 | CC-BY-4.0 |
| corr2cause-train | 10,000 | 0 | strict | programmatic | 4,949 | MIT (GitHub causalNLP/corr2cause; no licence tag on the Hub  |
| cosmosqa-train | 25,242 | 4 | strict | human | 7,934 | CC-BY-4.0 |
| creak-train | 10,175 | 1 | strict | human | 4,994 | MIT (LICENSE.txt) |
| csqa2-train | 9,789 | 0 | strict (+ 검증 과반 불일치·미검증 2,016문항 제외) | human | — (v3-gold7부터) | CC-BY-4.0 (github allenai/csqa2 LICENSE) |
| cuad-train | 4,804 | 0 | documents | human | 4,497 | CC-BY-4.0 (card YAML and Licensing Information; contracts fr |
| drop-train | 71,161 | 198 | strict | human | 13,379 | CC-BY-SA-4.0 |
| ecare-train | 17,013 | 0 | strict | human | 6,498 | MIT |
| esci-train | 14,221 | 779 | strict | human | 5,928 | Apache-2.0 |
| ezstance-train | 39,253 | 44 | strict | human | — (v3-gold7부터) | no licence file (Google Drive release; tweets): review before release |
| faithdial-train | 12,777 | 167 | strict | human | 5,602 | MIT |
| fewnerd-train | 39,982 | 18 | strict | human | 10,012 | CC-BY-SA-4.0 |
| fewrel-train | 31,598 | 402 | strict | human | 8,892 | MIT |
| finer-train | 44,613 | 9 | strict | human | 10,580 | CC-BY-SA-4.0 |
| fintweet-train | 15,946 | 0 | strict | human | 6,287 | MIT |
| folio-train | 1,208 | 0 | strict | human | 1,081 | CC-BY-SA-4.0 |
| gap-train | 2,433 | 7 | strict | human | 2,190 | Apache-2.0 |
| goemotions-train | 36,196 | 66 | strict | human | 19,044 | Apache-2.0 |
| gsm8k-train | 12,402 | 2,544 | strict | human | 5,529 | MIT |
| habermas-train | 2,227 | 59 | strict | human | 2,004 | CC-BY-4.0 (data), Apache-2.0 (code) |
| hatexplain-train | 15,368 | 0 | strict | human | 6,170 | MIT |
| headqa-train | 5,162 | 2 | strict | human | 3,504 | MIT (dataset); exams reused under the Spanish Ministerio de  |
| hellaswag-train | 39,566 | 314 | strict | human | 19,916 | MIT |
| helpsteer2-train | 9,353 | 464 | strict | human | 4,780 | CC-BY-4.0 |
| helpsteer3-train | 12,702 | 2,550 | strict | human | 5,597 | CC-BY-4.0 |
| hhrlhf-train | 29,175 | 825 | strict | human | 8,540 | MIT |
| hover-train | 15,104 | 3,066 | strict | human | 6,108 | CC-BY-SA-4.0 |
| humicroedit-train | 10,481 | 0 | strict | human | 5,071 | none stated (SemEval-2020 task release; Hub mirror: unknown) |
| icl-classify | 8,365 | 2,320 | strict | human | 8,064 | DBpedia: card CC0, text Wikipedia (CC-BY-SA); tweets MIT |
| infotabs-train | 16,200 | 333 | strict; table text by shingles only | human | 6,338 | Apache-2.0 (repository LICENSE) |
| isarcasm-train | 13,358 | 24 | strict | human | 5,743 | MIT |
| jglue-train | 28,988 | 0 | strict | human | 8,513 | CC-BY-SA-4.0 |
| klue-train | 30,001 | 0 | strict | human | 8,662 | CC-BY-SA-4.0 |
| kv-lookup | 4,000 | 0 | strict | programmatic | 3,700 | ours |
| lexglue-short-train | 22,919 | 2,098 | strict | human | 7,559 | CC-BY-4.0 |
| lexglue-train | 26,312 | 2,083 | documents | human | 11,860 | CC-BY-4.0. **SCOTUS 문항의 SCDB 쟁점 라벨은 upstream CC-BY-NC-3.0**(판결문은 퍼블릭 도메인). g6 1,363행·g7 2,684행 학습, 공개 전 검토 대기 |
| logic-puzzles-gen | 4,999 | 0 | strict | programmatic | 3,445 | ours (generated) |
| long-context | 15,999 | 1 | documents | ? | — | SQuAD 2.0 CC-BY-SA-4.0; CLINC150 CC-BY-3.0; BANKING77 CC-BY- |
| massive-intents | 11,413 | 99 | strict | human | 5,298 | CC-BY-4.0 |
| mathqa-train | 14,236 | 764 | strict | human | 5,901 | Apache-2.0 |
| maud-train | 30,991 | 1,179 | strict | human | 8,789 | CC-BY-4.0 |
| medmcqa-train | 9,999 | 1 | strict | human | 4,950 | Apache-2.0 |
| minicheck-synth | 14,366 | 25 | strict | model (2026-10-19 예외 허용) | — (v3-gold7부터) | MIT (lytang/C2D-and-D2C-MiniCheck; GPT-4 generated) |
| mmlu-aux-train | 19,984 | 15 | strict | human | 7,040 | MIT |
| mnli-train | 35,871 | 129 | strict | human | 9,480 | CC-BY-3.0 (card); genre sources per the MultiNLI paper |
| moralstories-train | 12,000 | 0 | strict | human | 5,436 | MIT |
| mtbench-human-train | 1,057 | 359 | strict | human | — | CC-BY-4.0 |
| mtop-train | 30,626 | 41 | strict | human | 8,752 | CC-BY-SA-4.0 |
| multiwoz-train | 29,357 | 49 | strict | human | 8,568 | MIT |
| musique-train | 12,000 | 0 | strict | human | 7,936 | CC-BY-4.0 |
| newyorker-matching-train | 9,753 | 558 | strict | human | 4,878 | CC-BY-4.0 |
| nli4ct-train | 1,867 | 33 | strict | human | 1,680 | no licence file (SemEval-2024 Task 2 data; CTRs from Clinica |
| nvd-cve | 39,966 | 14 | strict | human | — (v3-gold7부터) | NVD: no licensing restrictions (NIST); CVE text: MITRE CVE terms; CWE names: MITRE |
| oasst2-train | 24,912 | 1,858 | strict | human | 7,885 | Apache-2.0 |
| openbookqa-train | 4,945 | 7 | strict | human | 3,425 | unknown on the Hub (AllenAI release Apache-2.0; unverified) |
| paws-train | 19,991 | 7 | strict | human | 7,053 | free for any purpose (google-research-datasets/paws LICENSE) |
| phishing | 9,579 | 421 | strict | human (LLMGen: model, 제외) | 4,137 | {"kudzaiprichard/aura-phishing-email-corpus": "CC-BY-4.0 (Ze |
| poem-pairs-gen | 16,997 | 0 | strict (+ BPoMP 원본과 줄 단위 대조로 시 9편 제외) | programmatic | — (v3-gold7부터) | public-domain poems (Project Gutenberg); Gutenberg Poetry Corpus CC0; CMUdict BSD-style; WordNet 3.0 licence |
| pstance-train | 19,412 | 0 | strict | human | — (v3-gold7부터) | repository MIT; data on Google Drive (tweets, Twitter terms) |
| pubhealth-train | 9,803 | 1 | strict | human | 4,899 | MIT |
| pubmedqa-labeled | 995 | 5 | strict | human | 895 | MIT |
| quac-train | 29,013 | 987 | strict | human | 8,516 | CC-BY-SA-4.0 |
| quality-train | 2,518 | 3 | documents | human | 2,263 | CC-BY-4.0 (dataset); articles: Project Gutenberg / OANC (Sla |
| quoref-train | 15,301 | 93 | strict | human | 6,150 | CC-BY-4.0 |
| ragtruth-train | 11,246 | 3,843 | strict | human | 5,257 | MIT annotations; source texts CNN/DM, MS MARCO, Yelp (traini |
| redial-train | 26,098 | 0 | strict | human | 8,073 | CC-BY-4.0 |
| ropes-train | 10,245 | 137 | strict | human | 4,993 | CC-BY-4.0 |
| sciq-train | 9,992 | 8 | strict | ? | — | CC-BY-NC-3.0 (non-commercial) |
| scitail-train | 23,034 | 43 | strict | human | 7,577 | Apache-2.0 (github allenai/scitail LICENSE) |
| scruples-train | 37,597 | 37 | strict (+ 갈린 투표 14,190개 제외) | human (최다 득표 학습) | — (v3-gold7부터) | Apache-2.0 (allenai/scruples); anecdotes are Reddit posts |
| semcor-train | 29,977 | 5 | strict | human | 8,655 | Princeton WordNet licence (SemCor 1.6/WordNet; SemCor 3.0 ma |
| sentfin-train | 14,398 | 0 | strict | human | 5,967 | MIT (repository LICENSE) |
| sharc-train | 21,821 | 28 | strict | human | 7,370 | CC-BY-SA-3.0 |
| siqa-train | 33,348 | 0 | strict | human | 9,137 | CC-BY-4.0 (AllenAI release; Hub card untagged) |
| snips-train | 13,705 | 0 | strict | human | 5,819 | CC0-1.0 |
| snli-train | 29,995 | 5 | strict | human | 8,661 | CC-BY-SA-4.0 |
| spartqa-train | 20,000 | 0 | strict | programmatic | 7,054 | MIT (HLR/SpartQA_generation) |
| squad2-answerability | 9,834 | 166 | strict | human | 4,898 | CC-BY-SA-4.0 |
| stepgame-train | 14,990 | 0 | strict | programmatic | 6,092 | MIT |
| strategyqa-train | 1,603 | 0 | strict | human | 1,443 | MIT |
| svamp-train | 699 | 0 | strict | human | 629 | MIT |
| tabfact-train | 88,466 | 4,087 | strict | human | 14,924 | MIT (repository LICENSE); tables from Wikipedia (CC-BY-SA) |
| taskmaster-train | 2,404 | 237 | strict | human | 2,164 | CC-BY-4.0 |
| tatqa-train | 14,302 | 579 | strict; table text by shingles only | human | 5,883 | CC BY 4.0 (README) |
| tatqa-yesno | 14,304 | 576 | strict; table text by shingles only | human | 11,768 | CC-BY-4.0 |
| tldr-feedback-train | 29,814 | 186 | strict | human | 8,635 | Modified MIT (feedback data); Reddit TL;DR posts CC-BY-4.0 ( |
| tools-percall | 23,889 | 1,109 | strict | model (도구 예외 허용) | 7,714 | {"Team-ACE/ToolACE": "Apache-2.0", "glaiveai/glaive-function |
| tydiqa-train | 29,757 | 200 | strict | human | 8,625 | Apache-2.0 |
| ukpconvarg-train | 11,432 | 0 | strict | human | 5,374 | CC-BY-4.0 (arguments: createdebate CC-BY-3.0, convinceme pub |
| vast-train | 14,132 | 42 | strict | human | 5,911 | no licence file (research release; NYT comments) |
| vitaminc-train | 29,635 | 361 | strict | human | 8,609 | CC-BY-SA-3.0 |
| wanli-train | 19,967 | 33 | strict | human | 7,049 | CC-BY-4.0 |
| when2call-train | 18,281 | 14,702 | strict | model (도구 예외 허용) | 6,736 | CC-BY-4.0 |
| winogrande-train | 40,398 | 0 | strict | human | 20,130 | Apache-2.0 (GitHub release; Hub card blank) |
| wned-el | 3,350 | 46 | strict | human | 3,049 | KILT MIT; text Wikipedia (CC-BY-SA); upstream WNED: no licence |
| wtq-train | 11,889 | 369 | strict; table text by shingles only | human | 5,411 | CC-BY-SA-4.0 (repository LICENSE) |
| zeshel-train | 48,896 | 369 | strict | human | 11,080 | CC-BY-SA (Fandom wikis) |

처음 만든 37개 출처의 합계는 410,337문항이었고, 60개 출처 시점(2026-10-19 감사 반영)의 합계는 864,090문항이었다. 5·6차 출처를 더하면 108개 출처, 1,944,489문항이고(v3-gold6이 그중 104개를 쓴다), 7차 출처 9개(시 최소쌍, MiniCheck, 입장 3개, CSQA2, Aegis, Scruples, NVD)까지 더한 현재 117개 출처의 합계는 2,158,148문항이다(MT-Bench 사람 판정·ConditionalQA는 `mix.EXCLUDED`). SciQ는 비상업(CC-BY-NC) 라이선스라, 합성 장문 `long-context`는 실제 장문 출처로 대체해서 혼합에서 뺀다. 모델이 정답을 만든 출처(ToolACE·Glaive의 tools-percall, When2Call, API-Bank 학습셋)는 도구 영역에 한해 허용하고(2026-10-19부터 MiniCheck C2D/D2C도 예외), phishing의 GPT 생성 이메일(LLMGen)은 뺀다(2026-10-17 결정, [작업 기록](../project/v3-log.md)). 라이선스 파일이 없거나 확인되지 않은 출처(ACOS, VAST, NLI4CT, Humicroedit, OpenBookQA, EZ-STANCE)는 비교 실험에만 쓰고, 공개 모델에 쓸지는 제출 전에 검토한다. xLAM은 gated라 ToolACE와 Glaive로 대체했다.

주요 함정(측정됨): ToolRet 시험 질의가 xLAM·ToolACE 등에서 왔다. SATA-Bench 일부가 MultiRC train에서 나왔다. MASSIVE에 JEVal nlu-eval-68 발화가 들어 있다. New Yorker의 다른 fold에 시험 콘테스트가 있다. GSM8K train 문제가 index의 RouterBench·BRIGHT 행 안에 들어 있다. FinEntity, BFCL, BRIGHT, HLE, GPQA, MuSR, SATA, CRUXEval, POP909, PhishNChips core, BPoMP는 전부 제외한다.

---

# Phase 2: open training data for the Jev Decision Index skills (contamination-checked)

Date: 2026-10-16. Scope: Decision Index 0.3 (37 scored benchmarks, 5 areas). JEVal (carlosxiang/JEVal, 58 subtasks) is used only as a held-out proxy.

## How this was checked

- **Index sources and splits**: `external/decision-index/docs/suite.md`, plus the builders in `decision_index/suite/build/*.py` and the raw pinned sources under `external/decision-index/work/artifacts/benchmark-suite/raw/`. I read the builder code to see which file and split each benchmark actually uses.
- **JEVal splits**: `data/jeval/data/test.parquet` (column `source_item_id` and the parquet metadata `jeval.sources`). Each JEVal source records `records_available` and the sampled indices.
- **Licences and sizes**: Hugging Face API and datasets-server, the GitHub API, and the dataset cards.
- **Overlap checks marked "measured"**: I downloaded the data and compared normalised text. Anything not checked this way is marked *(unverified)*.

### Facts about the index that set the rules (verified in code)

| Index benchmark | What the index uses | Consequence for training |
|---|---|---|
| ACOS (38) | **Only** `*_quad_test.tsv` (1,394 test sentences; 400 reviews scored). Train/dev are read only to build the category inventory. | ACOS train/dev are usable. The brief said ACOS is used in full; that is **not** the case. Drop the 30 train/dev sentences that duplicate test sentences (measured). |
| FinEntity (39) | The whole `FinEntity.json` (979 docs), split `evaluation-only`. | **Exclude FinEntity entirely** (confirmed). |
| ESCI (37) | 5,000 pairs from the **test** split (`source_test_rows: 638,016`). | ESCI train is usable. Queries in train and test are disjoint by design. |
| PhishNChips (56) | `core_emails.csv`, all 2,000 rows. | Exclude core. The repo's other CSVs are not indexed (gray area; see Retrieval). |
| HoVer (61) | The dev release, all 4,000 claims. | HoVer train (18,171 claims) is usable. |
| RAGTruth (59) | All 2,700 test responses (450 sources). | Train (15,090 responses, 2,515 sources) is usable. Source overlap with test is 0 (measured). |
| When2Call (62) | `test/when2call_test_mcq.jsonl` (3,652). | `train_sft` (15k) and `train_pref` (9k) are usable. |
| New Yorker (64) | `matching` (cross-validation split 0) test, 528 rows, 141 contests. | Use split-0 train/validation only. Other folds and other rating data leak the test cartoons (measured; see Arts). |
| Habermas (50) | Only rows with `metadata.version` EVAL_* and `question.split` IID_TEST or OOD_TEST. | TRAINING_DATA_V1..V5 (15,595 rows; 1,439 TRAIN questions; 0 question overlap, measured) are usable. |
| cfcolor (23) | Targets come from `test_vec`; user history comes from `train_vec`; `probe_vec` is unused. | `train_vec` and `probe_vec` pairs are usable (same users, so disclose). |
| BPoMP (20) | 5,000 of 134,005 minimal pairs, grouped by original limerick. | Exclude the whole release. The originals come from an OEDILF scrape (paper, section 3). |
| POP909-CL (22) | 2,000 beats sampled from all 907 eligible songs, `evaluation-only`. | Exclude POP909 entirely. |
| CLadder (44) | 5,000 of the 10,112 rows in `cladder-v1-q-balanced`. | The balanced file merges all variants, so every variant overlaps it. Generate fresh questions instead (the repo ships its generator, MIT). |
| BFCL (1) | All of `simple`, `live_simple` and `live_multiple`. | Exclude all of BFCL. The other categories are also BFCL test data (gray area). |
| ToolRet (2) | ToolRet-Queries: 1,000 of 7,704 queries sampled, 685 scored. | **The queries come from xLAM/APIGen (1,000), ToolACE (1,000), ToolBench (1,100), API-Bank (101), Gorilla APIBench (~500 per hub), MetaTool (200), ToolAlpaca, UltraTool and others (35 sources)**. Deduplicate any tool corpus against all 7,704. |
| API-Bank (3) | `lv1-lv2-samples` (test) with the 53-API catalog. | The HF release `liminghao1630/API-Bank` has a separate `training-data/` folder (MIT), which is usable. |
| Home appliance (9) | A local generator (seed 2026091807); 88 of 160 test rows are scored; the dev split (80 rows) is published. | The dev split is usable, and so is an independent generator. Deduplicate full-state JSON against the 160 test rows. |
| GSM8K (30) | The test split, with 0.3 distractors taken from other **test** problems' golds in the same answer band. | Use GSM8K train and recreate the same distractor scheme inside train (0 train/test duplicates, measured). |
| MMLU-Pro (57), BBH (58), GPQA (25), HLE (45), MuSR (32), SATA (33), CRUXEval (43), BRIGHT (36), ChessBench (31) | All available test items (no train split exists, except ChessBench's). | Use substitutes. See the traps listed in each area. |
| SGD (10) | Not in the index now ("pending re-run, returns to the index"); uses `test/`. | Train only, and never the SGD test split. |

### JEVal facts (from `source_item_id` and `jeval.sources`; JEVal picks the first available of test, dev, train)

- **Datasets JEVal samples from its full pool, including what upstream calls train**:
  - boolq: 12,697 = train + val
  - mmlu-pro: 12,102 = test + validation
  - nlu-eval-68: 25,716 = all of HWU64
  - halueval-*: all of HaluEval
  - prism-alignment
  - decider
  - jevforge
  - For these, drop the 200 JEVal items by exact text match, or exclude the dataset.
- **Datasets where JEVal uses the same split as the index**: hellaswag/dev and winogrande/dev. Never train on those dev splits.
- **Datasets where JEVal uses test or dev only**, so train is fine: clinc150/test, fever/test, scifact/dev, commonsense-qa/dev, ARC test, bbq (all 58,492, so exclude BBQ), cuad/test, lex-glue-*/test, med-qa/test, go-emotions/test, truthfulqa (all 817, so exclude), gpqa (all), global-opinions (exclude), agentharm (exclude), longbench-*, helmet, financebench, finqa.

### Global deduplication rule (apply to every source below)

1. Build an exclusion index from three things: every index row (`suite/selected-rows` and `added-rows`, plus the GSM8K v0.3 rows), all 7,704 ToolRet queries, and all 11,257 JEVal states, questions and options.
2. Match on normalised text (lowercase, alphanumerics only, collapsed spaces) and also with a 13-gram containment check for long states.
3. For structured items, match on the item key: chess FEN, palette ID set, `contest_number`, limerick hash, URL, tool query, or full-state canonical JSON.
4. Exact duplicates found already (measured): CLINC150 train has 27 utterances that are in the index test set and val has 4; BANKING77 train has 9; HellaSwag train has 25 contexts that duplicate the index's validation split; ACOS train/dev have 30 sentences duplicating test.
5. Record the dropped counts in the disclosure.

Licence key: **NC** = non-commercial; **?** = no licence file found.

---

## 1. Knowledge and reasoning (GPQA-D, GSM8K, ChessBench, MuSR, SATA, CRUXEval, CLadder, HLE, MMLU-Pro, BBH, WinoGrande)

| # | Dataset | HF id / URL | Licence | Split to use | ~Size | Decision cast | Overlap risk and mitigation |
|---|---|---|---|---|---|---|---|
| K1 | GSM8K train with band distractors | `openai/gsm8k` (main) | MIT | train | 7,473 | choice over 4 and 10 numbers. Distractors are other **train** problems' golds in the same magnitude band, reproducing `answer-groups-v1` so that option values alone score at chance. | 0 train/test duplicates (measured). Never use GSM-Hard, GSM-Plus, GSM-Symbolic or GSM8K-Platinum, which are all rewrites of the GSM8K test set. |
| K2 | MathQA | `allenai/math_qa` | Apache-2.0 | train + validation | 29.8k + 4.5k | native 5-choice numeric; can also be re-optioned with band distractors | Low risk. Not in the index or JEVal. |
| K3 | AQuA-RAT | `deepmind/aqua_rat` (raw) | Apache-2.0 | train | 97k | native 5-choice | Low risk. Has known noisy items, so filter on rationale-answer agreement. |
| K4 | SVAMP / MetaMathQA / OpenMathInstruct-2 | `ChilleD/SVAMP`; `meta-math/MetaMathQA`; `nvidia/OpenMathInstruct-2` | MIT / MIT / CC-BY-4.0 | train | 1k / 395k / 14M (subsample) | numeric choice with band distractors | MetaMath and OMI-2 are built from GSM8K/MATH **train** (OMI-2 was decontaminated). Even so, run an exact and 13-gram match against GSM8K test. `microsoft/orca-math-200k` (MIT) has undocumented decontamination: flag it and dedupe. |
| K5 | ChessBench train action-values | `storage.googleapis.com/searchless_chess/data/train/action_value-*.bag` | CC-BY-4.0 (some CC0 Lichess parts); code Apache-2.0 | train (1 or 2 of the 2,148 shards, 1.2 GB each) | millions of positions | choice over **all legal moves**, gold = argmax Stockfish win% (accept ties); this is exactly the index format | The ChessBench paper reports that train and test share common positions, mostly openings *(from memory, unverified)*. **Deduplicate by FEN against the entire test bag**, not only the 5,000 scored positions. |
| K6 | Lichess puzzles | `Lichess/chess-puzzles` | CC0 | train | 6.16M | choice over all legal moves; gold = first puzzle move | Low risk. Still dedupe by FEN against the ChessBench test bag and `puzzles.csv`. |
| K7 | CLadder regenerated | `github.com/causalNLP/cladder` (generator under `causalbenchmark/`) | MIT | generate with **new seeds and story names** | unlimited | yes/no (choice) | Do **not** use any `cladder-v1*` file: all variants overlap the balanced file the index samples from. After generating, dedupe exact question text against v1. |
| K8 | Corr2Cause | `causal-nlp/corr2cause` (GitHub causalNLP/corr2cause) | MIT | train | ~206k *(size unverified)* | yes/no: does the causal relation hold, given the correlations? | Low risk. Different construction from CLadder. |
| K9 | MuSR regenerated | `github.com/Zayne-sprague/MuSR` (`musr_dataset_scripts/`) | MIT | regenerate murder-mystery, object-placement and team-allocation items with an LLM | thousands | native multiple choice | The index uses all 756 released items, so regenerate only and dedupe by story text. Alternatives: FOLIO train (`yale-nlp/FOLIO`, MIT, gated); CLUTRR (CC-BY-NC). |
| K10 | Procedural BBH-like tasks | `github.com/open-thought/reasoning-gym`; `MiniMaxAI/SynLogic` | Apache-2.0; MIT | generate / train | unlimited; 48.7k | parse each task's option set into choice | **Trap:** BBH items are subsets of 23 BIG-Bench tasks (for example `tasksource/bigbench`, Apache-2.0) and carry the BIG-bench canary. Do not train on those 23 BIG-Bench task files. Procedural generators are the safe route. |
| K11 | Code output prediction | `hkust-nlp/CodeIO-PyEdu-Reasoning` (ODC-BY); `semcoder/PyX` (MIT, 93k); self-made (execute functions from MBPP train or The Stack with random inputs) | ODC-BY / MIT | train | 93k+ | choice: correct output plus 3 to 9 distractors from mutated programs or other inputs' outputs (CRUXEval style) | CRUXEval has no train split (all 800 problems are test). Avoid CRUXEval-X and other translations of it. CodeI/O claims decontamination against CRUXEval *(unverified)*, so dedupe on code text against `cruxeval.jsonl`. |
| K12 | Select-all from multi-label data | MultiRC (`aps/super_glue` multirc); GoEmotions train; EURLEX57K train; Reuters-21578 | SuperGLUE terms (?) / Apache / CC-BY-SA / research | train | 27k / 43k / 45k / 10k | per-option yes/no (the index casts SATA as per-option yes/no) | **Measured trap:** SATA-Bench's reading-comprehension items are MultiRC **train** (216 of 216 paragraphs and 298 of 342 questions found in MultiRC train; 0 in validation). SATA's other sources are RealToxicityPrompts, Reuters-21578, PubMed-MeSH, EURLEX57K and `knowledgator/events_classification_biotech`. Drop every paragraph or document that appears in SATA-Bench. GoEmotions is fine on train (JEVal uses test). |
| K13 | MMLU auxiliary_train / MedMCQA / QASC / ARC train | `cais/mmlu` (auxiliary_train); `openlifescienceai/medmcqa`; `allenai/qasc`; `allenai/ai2_arc` | MIT / Apache-2.0 / CC-BY-4.0 / CC-BY-SA-4.0 | auxiliary_train; train | 99.8k / 182.8k / 8.1k / 3.4k | native multiple choice (pad to 10 options with sibling distractors for MMLU-Pro style) | **Trap:** MMLU-Pro = MMLU **test** (filtered) + TheoremQA + SciBench + STEM-site items. Never use MMLU test, TheoremQA or SciBench. MedMCQA is not in the index or JEVal. ARC: JEVal uses test, so train is fine. |
| K14 | OpenScience (synthetic graduate-level MCQ) | `nvidia/OpenScience` | CC-BY-4.0 (outputs from Qwen2.5-72B; the Qwen licence may attach to distributed models) | train configs `OS-Q*-4` / `-10` | ~millions | native 4 or 10 options, which matches the GPQA and MMLU-Pro formats | The card says it was decontaminated against MMLU, MMLU-Pro and GPQA test sets. It does not mention HLE, so dedupe against HLE text. Note: SciQ (CC-BY-**NC**-3.0) and SuperGPQA (ODC-BY, but itself a benchmark) are gray. Avoid GPQA main/extended, which are supersets of Diamond. |
| K15 | WinoGrande train | `allenai/winogrande` winogrande_xl | Apache-2.0 (GitHub; HF card blank) | train only | 40,398 | 2-option choice | 0 duplicates against validation (measured). The index and JEVal both use **validation**, so never touch it. |

Exclude completely from Knowledge: GPQA (all configs), HLE, MuSR release, SATA-Bench, CRUXEval and its derivatives, CLadder v1 files, BBH and its BIG-Bench source tasks, MMLU test, TheoremQA, SciBench, SimpleBench, and the GSM8K-test derivatives.

---

## 2. Language understanding (ContractNLI, ANLI, HellaSwag, ACOS, FinEntity, iSarcasmEval, VAST, NLI4CT, RAGTruth)

| # | Dataset | HF id / URL | Licence | Split | ~Size | Decision cast | Overlap risk and mitigation |
|---|---|---|---|---|---|---|---|
| L1 | ContractNLI train/dev | `stanfordnlp/contract-nli` (zip in repo) | CC-BY-4.0 | train + dev | 423 + 61 docs × 17 hypotheses ≈ 8.2k | 3-way choice (Entailment / Contradiction / NotMentioned); **long-context** | The index uses the 123 test docs. Train and dev are separate NDAs. Also good long-context data. |
| L2 | ANLI train | `facebook/anli` | **CC-BY-NC-4.0** | train_r1/r2/r3 (+ dev) | 162.9k | 3-way NLI choice | 0 identical premise+hypothesis pairs with test, but **the premises are shared by design** (R1: 828 of 846 test premises appear in train; measured). This is the official train split, so it is tolerated; disclose it. NC flag. |
| L3 | WANLI / SNLI / MNLI | `alisawuffles/WANLI`; `stanfordnlp/snli`; `nyu-mll/multi_nli` | CC-BY-4.0 / CC-BY-SA-4.0 / mixed (OANC, CC-BY-SA, MIT) | train | 103k / 550k / 393k | NLI choice; also binary entail / not-entail | Low risk. Not in the index or JEVal. |
| L4 | HellaSwag train | `Rowan/hellaswag` | MIT (per suite.md) | train only | 39,905 | 4-way continuation choice | Validation is used by **both** the index and JEVal. Drop the 25 train contexts that duplicate validation (measured). SWAG (`allenai/swag`, licence unknown, 73k) is an optional extra. |
| L5 | ACOS train/dev | `github.com/NUSTM/ACOS` | ? (no licence) | Laptop + Restaurant train and dev | 3,260 + 1,701 sentences | per (category, sentiment) pair yes/no, with the index's 64-pair inventory groups | Index uses test only (verified). Drop the 30 sentences duplicating test (measured). **Trap:** ACOS-Restaurant test is SemEval-2016 Restaurant **test**. ASQP `rest16/test.txt` contains 527 of the 1,394 ACOS test sentences (measured). Exclude SemEval-16 restaurant test and every derivative (ASQP, ASTE, TASD rest16 test). |
| L6 | MAMS / ASQP rest15 | `github.com/siat-nlp/MAMS-for-ABSA`; `github.com/IsakZhang/ABSA-QUAD` (rest15 only) | Apache-2.0 / ? | train + dev | 4.3k (ATSA+ACSA) / 1k | aspect-category × sentiment yes/no | ASQP rest15 train and test have 5 sentences each that are in ACOS test (measured); dedupe them. |
| L7 | Entity-level financial sentiment (FinEntity substitute) | SEntFiN 1.0 (`github.com/pyRis/SEntFiN`); `TheFinAI/fiqa-sentiment-classification`; `takala/financial_phrasebank` | MIT (repo) / MIT (gated) / **CC-BY-NC-SA-3.0** | all / train / all | ~10.7k headlines *(unverified)* / 1.2k / 4.8k | choice over Negative / Neutral / Positive for a **given entity span** (the index format) | **FinEntity is excluded entirely**. SEntFiN is headline-level and FinEntity is paragraph-level, so overlap is unlikely; still dedupe text. PhraseBank is NC and sentence-level only. |
| L8 | iSarcasmEval train | `github.com/iabufarha/iSarcasmEval` `train/train.{En,Ar}.csv` | MIT | train | ~3.5k En + ~3.1k Ar | task A yes/no; task B multi-label per-option yes/no; task C pairwise (tweet vs. its rephrase), all from the train columns | Index uses test A/B/C. **Exclude the `third-party annotations/` folder (test items).** Extras: `cardiffnlp/tweet_eval` irony (SemEval-2018 T3, licence unknown, 2.9k train). |
| L9 | VAST train/dev | `github.com/emilyallaway/zero-shot-stance` `vast_train.csv`, `vast_dev.csv` (`tasksource/vast`) | ? (no licence) | train + dev | 13,477 + 2,062 | 3-way stance choice | 0 post overlap with test (measured). **Exclude `vast_test-sentswap.csv`** (all 932 rows are test posts). Extra: SemEval-2016 T6 via `tweet_eval stance_*`. |
| L10 | NLI4CT train/dev (+ practice test) | `github.com/ai-systems/Task-2-SemEval-2024` `training_data.zip`; `practice_test.json` + gold | SemEval terms (?) | train + dev (+ practice) | 1,700 + 200 (+ 2,142) | Entailment / Contradiction on clinical-trial-report sections | Statements: 0 duplicates with test (measured). 200 train rows share a primary CTR with test, which is fine. The practice test is disjoint from test (measured) but is a test-phase file, so it is a gray option. MedNLI needs PhysioNet credentials and is not open. |
| L11 | RAGTruth train | `github.com/ParticleMedia/RAGTruth` `response.jsonl` (split=train) | MIT annotations; contexts are MS MARCO (NC) and Yelp (no redistribution) | train | 15,090 responses / 2,515 sources | noul: "contains content not supported by the context"; can also be split per sentence or span | 0 source overlap with test (measured). Rebuild locally and do not republish. |
| L12 | RAGBench | `rungalileo/ragbench` | CC-BY-4.0 | train + validation, **excluding the `cuad` and `finqa` configs** | ~80k *(unverified total)* | noul adherence / unsupported-claim | `cuad` comes from CUAD (JEVal cuad) and `finqa` from FinQA (JEVal finqa-yes-no). Dedupe the `msmarco` config against RAGTruth test questions (shared MS MARCO source). |
| L13 | FAVA / MiniCheck synthetic / SummEdits | `fava-uw/fava-data`; `lytang/C2D-and-D2C-MiniCheck`; `Salesforce/summedits` | CC-BY-4.0 / MIT / CC-BY-4.0 | train / c2d+d2c / all | 30k / 14.4k / 6.3k | noul "claim supported by document?" | Low risk. **Do not use** LLM-AggreFact (it includes the RAGTruth test set; CC-BY-**ND**), HaluBench (includes HaluEval; NC) or HaluEval (all of it is in JEVal). |
| L14 | Long-context decision | QuALITY (`emozilla/quality` or nyu-mll/quality); QASPER (`allenai/qasper`) | per-article licences (QuALITY); CC-BY-4.0 | train + dev | ~2.5k + 2.1k questions; 888 + 281 papers | 4-way choice; yes / no / unanswerable | Not in the index or JEVal. **Avoid LongBench and HELMET (JEVal).** |

Exclude completely from Language: FinEntity, ANLI test, ContractNLI test, NLI4CT test and gold_test, VAST test and test-sentswap, iSarcasm test and third-party annotations, HellaSwag validation/test, ACOS test, SemEval-16 restaurant test (and derivatives), HaluEval, LLM-AggreFact, HaluBench.

---

## 3. Retrieval and classification (BANKING77, CLINC150+OOS, BRIGHT, ESCI, PhishNChips, HoVer)

| # | Dataset | HF id / URL | Licence | Split | ~Size | Decision cast | Overlap risk and mitigation |
|---|---|---|---|---|---|---|---|
| R1 | BANKING77 train | PolyAI-LDN `banking_data/train.csv` (`legacy-datasets/banking77`) | CC-BY-4.0 | train | 10,003 | 77-way choice | The index uses test. **Drop the 9 train utterances identical to test utterances** (measured). |
| R2 | CLINC150 + OOS train/val | `clinc/clinc_oos` (plus) or `oos-eval` `data_full.json` | CC-BY-3.0 | train, val, oos_train, oos_val | 15,000 + 3,000 + 100 + 100 (+ the 250-example `plus` OOS) | 151-way choice (150 intents + OOS) | The index and JEVal both use test. **Drop the 27 train and 4 val utterances identical to test** (measured). |
| R3 | MASSIVE | `AmazonScience/massive` (or s3 tarball 1.1) | CC-BY-4.0 | en-US train/dev/test (other locales optional) | 11,514 / 2,033 / 2,974 (60 intents) | 60-way choice | **Measured trap:** 98 of the 200 JEVal `nlu-eval-68` utterances appear verbatim in MASSIVE en-US (55 in train). MASSIVE derives from SLURP, which reuses HWU64 text. Drop them. |
| R4 | HWU64 / NLU-Evaluation-Data | `github.com/xliuhw/NLU-Evaluation-Data` | CC-BY-4.0 | all, minus the 200 JEVal items | 25,716 (64 to 68 intents) | many-option choice | **JEVal nlu-eval-68 samples from the whole corpus.** Exclude those 200 items by text. |
| R5 | Bitext customer-support / SGD train | `bitext/Bitext-customer-support-llm-chatbot-training-dataset`; `google-research-datasets/dstc8-schema-guided-dialogue` train | CDLA-Sharing-1.0 / CC-BY-SA-4.0 | train | 26.9k (27 intents) / 16k dialogues | intent choice; SGD: current-intent choice given the schema (+ NONE) | SGD **test** is index item #10 (parked, expected back), so use SGD **train** only. |
| R6 | Amazon ESCI train | `github.com/amazon-science/esci-data` (`split=='train'`) | Apache-2.0 | train | ~2.0M judgments *(approx.)* | 4-way E/S/C/I choice (index format) | Index uses the test split only (verified). Train and test queries are disjoint. |
| R7 | WANDS (Wayfair) | `github.com/wayfair/WANDS` | MIT | all | 480 queries, ~233k labels *(approx.)* | 3-way Exact / Partial / Irrelevant, mapped to an ESCI-style choice | Not in the index or JEVal. |
| R8 | Document relevance (BRIGHT substitute) | MS MARCO hard negatives (`sentence-transformers/msmarco-*`), NQ, HotpotQA, FiQA-2018 train | MS MARCO research-only (**NC-like**); CC-BY-SA | train | millions | per-document yes/no relevance (noul), with BM25 top-k candidates as in the index | **BRIGHT is test-only and is excluded.** **Trap:** ReasonIR **HQ** data (`reasonir/reasonir-data`, CC-BY-NC) is generated from BRIGHT's own document datastore, so exclude it (the VL config is optional, NC). Raw StackExchange, LeetCode and AoPS dumps contain BRIGHT queries; dedupe by post text and URL if used. |
| R9 | PhishNChips non-core files | `AreLit/PhishNChips`: `real_phishing_validation.csv`, `infrastructure_phishing_expanded.csv` (Nazario corpus, CC-BY-4.0), `cross_domain_legitimate_v5.csv` | MIT project content + per-source terms | all, after dedupe | unknown *(row counts unverified)* | choice: phishing / legit (should the agent click) | **Core 2,000 = all index items.** These files are not indexed but come from the same lab (gray area). Dedupe by id, url_raw and body against `core_emails.csv`; `cross_domain_legitimate_v5` likely overlaps the 333 core cross-domain rows. |
| R10 | Phishing URL / email corpora | `pirocheto/phishing-url` (CC-BY-4.0, 11.4k); PhiUSIIL (UCI, CC-BY-4.0 *(unverified)*, 235k URLs); Nazario corpus (CC-BY-4.0); `ealvaradob/phishing-dataset` (Apache-2.0); `zefang-liu/phishing-email-dataset` (LGPL-3.0) | as listed | all | 10k to 235k | phishing yes/no. Best transfer: wrap URLs in **LLM-written workplace emails** as PhishNChips does (strategies: google_docs, github_pages, ipfs, firebase, url_shortener, cross-domain legitimate traps) | Core URLs come from PhishTank (700), OpenPhish (66), GitHub Phishing DB (234) and Tranco (662). **Dedupe exact URL and host against `core_emails.csv` `url_raw`.** Note that a free-hosting/shortener regex reaches 91.8%; include legitimate emails that use those hosts so the model does not learn the shortcut. |
| R11 | HoVer train | `hover-nlp/hover` + `wiki_wo_links.db` | CC-BY-SA-4.0 | train | 18,171 claims | SUPPORTED / NOT_SUPPORTED with oracle intro paragraphs (index format) | Index uses dev (all 4,000). Train claims are distinct. |
| R12 | FEVER / FEVEROUS / VitaminC | `fever/fever` v1.0 train; `fever/feverous`; `tals/vitaminc` | CC-BY-SA-3.0 | train | 145k claims / 71k / 371k | yes/no claim verification | JEVal fever samples 200 items from its "test" file of 6,613 records, which is a reformatted variant. Dedupe claims against JEVal. **SciFact dev is JEVal**; SciFact train is CC-BY-**NC**-2.0. |

Exclude completely from Retrieval: BRIGHT (and ReasonIR-HQ), PhishNChips `core_emails.csv`, the BANKING77, CLINC and ESCI test splits, HoVer dev, the JEVal nlu-eval-68 and clinc150 items, SciFact dev.

---

## 4. Tools and automation (BFCL per-function yes/no, ToolRet relevance yes/no, API-Bank next API, Home appliances, When2Call)

| # | Dataset | HF id / URL | Licence | Split | ~Size | Decision cast | Overlap risk and mitigation |
|---|---|---|---|---|---|---|---|
| T1 | xLAM function-calling 60k (APIGen) | `Salesforce/xlam-function-calling-60k` | CC-BY-4.0 (auto-gated) | train | 60,000 | per candidate function: yes if it appears in the gold calls, no otherwise (BFCL format). Inject 5 to 30 extra tools from other rows to match live_multiple option counts. | **ToolRet contains 1,000 APIGen queries.** Drop them (exact and fuzzy match on the query). |
| T2 | ToolACE | `Team-ACE/ToolACE` | Apache-2.0 | train | 11,300 | per-function yes/no, including multi-turn and no-call cases | **ToolRet contains 1,000 ToolACE queries.** Drop them. |
| T3 | Glaive FC v2 / Hermes FC v1 | `glaiveai/glaive-function-calling-v2`; `NousResearch/hermes-function-calling-v1` | Apache-2.0 / Apache-2.0 | train | 113k / ~11.5k | per-function yes/no; the Glaive rows with no applicable function give all-no negatives | Hermes' `glaive_func_calling` overlaps Glaive, so dedupe internally. Low risk against the index. |
| T4 | xLAM irrelevance | `MadeAgents/xlam-irrelevance-7.5k` | CC-BY-4.0 | train | 7,500 | all candidate tools = no | Built from xLAM, so apply the same ToolRet dedupe. |
| T5 | When2Call train | `nvidia/When2Call` `train_sft`, `train_pref` | CC-BY-4.0 | train | 15,000 + 9,000 | 4-way choice (direct answer / tool call / ask for info / cannot answer), exactly the index cast | The index uses the test MCQ. Check train questions against ToolRet and BFCL live queries *(source overlap unverified)*. |
| T6 | ToolRet training set | `mangopy/ToolRet-Training-20w` | **? (no licence)** | train | 208,826 | per-tool relevance yes/no (positive vs. listed hard negatives) | Same authors and sources as ToolRet-Queries. **Dedupe against all 7,704 ToolRet queries** (and the 35 source test sets). Licence flag. |
| T7 | API-Bank training data | `liminghao1630/API-Bank` `training-data/lv{1,2,3}-*.json` | MIT | train | ~2k dialogues *(unverified)* | next-API choice over the catalog given the dialogue (index cast: 53-API catalog) | The index uses `lv1-lv2-samples` (test). The training data uses a different API pool. ToolRet `apibank` has 101 queries; dedupe them. |
| T8 | ToolBench / APIGen-MT | `github.com/OpenBMB/ToolBench`; `Salesforce/APIGen-MT-5k` | Apache-2.0 / **CC-BY-NC-4.0** | train | 126k / 5k | per-API relevance yes/no; next-call choice | ToolRet `toolbench` (1,100) and `toolbench-sam` queries must be dropped. Gorilla APIBench **is a ToolRet source** (gorilla-huggingface/pytorch/tensor), so avoid it or dedupe. |
| T9 | Home appliances | dev split `work/artifacts/benchmark-suite/home/dev.jsonl` (80 rows) + **an independent simulator** written by us covering the 24 scenario families (exact_toggle, room_except, named_group, duplicate_name, already_satisfied, range_reject, atomic_block, future_schedule, conditional_rule, power_filter, context_pronoun, missing_device, relative_level, room_alias_ambiguous, last_wins, partial_unavailable, ...) | MIT (repo generator) / ours | dev + generated | 80 + unlimited | per-question choice: target-member yes/no, outcome, post-state, state read | Do not rerun the repo generator with the index seed. Use new household, room and device names, and **reject any generated state whose canonical JSON equals one of the 160 test states**. Extra: HomeBench (`github.com/BITHLP/HomeBench`, **no licence file**), which has valid and invalid multi-device instructions. |

Exclude completely from Tools: BFCL (every category; simple, live_simple and live_multiple are index items), ToolRet-Queries/Tools test queries, the API-Bank lv1-lv2 samples (lv3-samples are test too; gray), When2Call test, the Home appliance test states.

---

## 5. Arts and human taste (BPoMP, Humicroedit, POP909 chords, cfcolor, Habermas, New Yorker)

| # | Dataset | HF id / URL | Licence | Split | ~Size | Decision cast | Overlap risk and mitigation |
|---|---|---|---|---|---|---|---|
| A1 | Humicroedit subtask-2 train/val + FunLines | `SemEvalWorkshop/humicroedit` (subtask-2: train, validation, funlines) | ? (SemEval-2020 T7 terms; HF says unknown) | train + validation + funlines | 9,381 + 2,355 + 1,958 pairs | 2-way: which edit is funnier (drop ties, as the index does) | The index uses subtask-2 test (2,960, minus 332 ties). Subtask-1 train (9,652 + 8,248 FunLines) can also be paired by the same original headline. Check that no test original headline occurs *(unverified)*. |
| A2 | New Yorker caption contest (split 0 only) | `jmhessel/newyorker_caption_contest` configs `matching`, `ranking`, `explanation` (no suffix) | CC-BY-4.0 | train + validation | matching 9,792 + 531; ranking 9,576 | 5-way caption match (text-only: scene, description, entities); ranking: 2-way which caption is better | **Measured traps:** `matching_1` train contains **all 141** test contests; `ranking` (split 0) **test** shares 136 contests with matching test; `yguooo/newyorker_caption_ranking` (CC-BY-**NC**, contests 510+) contains **49** test contests. Use split-0 train/val only, and filter every external source by `contest_number`. |
| A3 | Habermas Machine training rounds | `storage.googleapis.com/habermas_machine/datasets/hm_all_candidate_comparisons.parquet` (`metadata.version` TRAINING_DATA_V1..V5) | CC-BY-4.0 (materials); Apache-2.0 (code) | TRAIN | 15,595 comparison rows, 1,439 questions | choice: which candidate statement this group ranks highest (summed ranks; ties accepted) | 0 question overlap with IID/OOD test (measured). The `hm_all_final_preference_rankings` and `position_statement_ratings` files may include eval cohorts, so filter on `question.split=='TRAIN'`. |
| A4 | Consensus / argument preference | Polis open data (`github.com/compdemocracy/openData`); IBM ArgQ-Rank-30k (`ibm-research/argument_quality_ranking_30k`); UKPConvArg1 | CC-BY-4.0 / CC-BY-3.0 / CC-BY-4.0 *(unverified)* | all / train | ~20 conversations / 21k / 11k pairs | Polis: which statement gets the broadest cross-group agreement; ArgQ: which argument is higher quality | Not in the index or JEVal. **Avoid PRISM and Anthropic global-opinions (JEVal).** |
| A5 | cfcolor personalised palettes | `dgp.toronto.edu/~donovan/cfcolor/cfcolor.zip` (`allMTurkRatings.mat`: `train_vec` 317,770 and `probe_vec` 104,206 ratings; `themeData.mat`: 13,343 themes) | per project page (?) | train_vec for history + probe_vec for targets (never test_vec) | ~50k+ pairs | 2-way: which palette this user rates higher, given 4 to 8 rated history palettes (gap ≥ 2) | Same users and palette pool as the test, but the official train/probe splits are tolerated; disclose. Never build a target from `test_vec`/`testRatings` (testRatings includes test). Gray option: aggregate theme ratings (`regressTargets`). COLOURlovers palettes are CC-BY-**NC**-SA. |
| A6 | Chord-from-notes, synthetic and annotated | Self-generated (13 templates × 12 roots, + NoChord / Other, inversions, passing and melody notes, 8-beat context); When-in-Rome (`github.com/MarkGotham/When-in-Rome`, CC-BY-SA-4.0); PDMX (`github.com/pnlong/PDMX`, MIT code, public-domain MusicXML with chord symbols *(unverified)*); Lakh MIDI (CC-BY-4.0) + music21 chordify | as listed | generated / all | unlimited / ~1k scores / 250k scores | 129-way pitch-class-set choice using the index's exact vocabulary (12 roots × {M, m, dim, aug, sus2, sus4, 7, maj7, m7, half-dim7, dim7, mMaj7, aug7}, merged aliases, + NoChord + Other) | **Exclude POP909 and POP909-CL completely** (every song is in the evaluation pool). Make sure no POP909 song is present in other compilations (dedupe by title and MIDI hash). |
| A7 | Limerick and poem minimal pairs | Edward Lear *A Book of Nonsense* (public domain); Gutenberg Poetry Corpus (Allison Parrish, public-domain texts); self-made perturbations with the same 14 BPoMP variants (corrupted rhyme, shuffled rhyming words or lines, delete 1 to 3 words, rhyming or non-rhyming EOL deletions, beam-search completions) | PD / CC0 | n/a | thousands to millions | 2-way: which is the original | **Measured trap: BPoMP originals are an OEDILF scrape** (paper §3, 1,624 authors), so any OEDILF-derived limerick set leaks. Hash-dedupe against all 134,005 BPoMP originals, not just the 5,000 scored ones. |

Exclude completely from Arts: the BPoMP release and OEDILF scrapes, POP909 and POP909-CL, cfcolor `test_vec`, the Habermas EVAL_* cohorts, New Yorker non-zero folds and split-0 test (all tasks), yguooo caption ratings for test contests, Humicroedit test.

---

## 6. New-domain yes/no judgment (30% private: broad knowledge plus knowing when to answer yes or no)

| # | Dataset | HF id | Licence | Split | ~Size | Cast | Risk and mitigation |
|---|---|---|---|---|---|---|---|
| N1 | BoolQ | `google/boolq` | CC-BY-SA-3.0 | train + validation **minus the 200 JEVal items** | 12,697 | passage yes/no | JEVal boolq samples from the full 12,697 pool, so dedupe by question and passage. |
| N2 | StrategyQA | `ChilleD/StrategyQA` | MIT | train | 1,603 | implicit multi-hop yes/no (no passage, so it tests broad knowledge) | Not in the index or JEVal. |
| N3 | PubMedQA | `qiaojin/PubMedQA` | MIT | pqa_artificial (+ pqa_labeled) | 211k (+1k) | yes / no (/ maybe) | Not in the index or JEVal. |
| N4 | SQuAD 2.0 answerability / QASPER / MuSiQue-answerable | `rajpurkar/squad_v2`; `allenai/qasper`; `bdsaglam/musique` (answerable config) | CC-BY-SA-4.0 / CC-BY-4.0 / CC-BY-4.0 *(unverified)* | train | 130k / ~5k Qs / 20k+ | noul: "the context answers the question" (abstain-or-answer judgement) | Low risk. |
| N5 | ETHICS | `hendrycks/ethics` | MIT | train | ~95k | yes/no acceptability (commonsense, justice, deontology, virtue) | Not in the index or JEVal. |
| N6 | Safety / refusal judgement | `nvidia/Aegis-AI-Content-Safety-Dataset-2.0`; `allenai/wildguardmix` (wildguardtrain) | CC-BY-4.0 / ODC-BY (gated) | train | 30k / 87k *(unverified)* | noul: unsafe prompt? should the assistant refuse? is the response harmful? | **AgentHarm is JEVal**, so do not use it. BeaverTails is CC-BY-NC. |
| N7 | Contrastive fact checks | `tals/vitaminc`; FEVER train | CC-BY-SA-3.0 | train | 371k / 145k | yes/no | Dedupe against JEVal fever. |
| N8 | Paraphrase / sense yes/no | `google-research-datasets/paws` (labeled_final); WiC (SuperGLUE) | PAWS: free with attribution (?); WiC: CC-BY-NC-4.0 | train | 49k / 5.4k | yes/no | Low risk. |
| N9 | Commonsense / science MCQ for breadth | CommonsenseQA train, ARC train, QASC, OpenBookQA train (licence "unknown" on HF), MedMCQA | MIT / CC-BY-SA / CC-BY / ? / Apache | train | 9.7k / 3.4k / 8.1k / 5k / 183k | choice; or per-option yes/no to train calibrated "no" | JEVal uses CommonsenseQA **dev** and ARC **test**, so train is fine. |
| N10 | jev-distill-v3 (already local) | `data/jev-distill-v3` | Apache-2.0 (Open-Jev stream CC0) | train | 656k | native noul/choice/score | The openjev_v2 domains are synthetic games and workflows (checked), with no index overlap. yuri_v1 uses HotpotQA, PubMedQA, MLQA, NarrativeQA and similar, which are not index sources. |

Never use for training (JEVal held-out): BBQ, TruthfulQA, HaluEval (all four), AgentHarm, PRISM-alignment, Anthropic global-opinions, CUAD test (and RAGBench-cuad), LexGLUE test splits, MedQA test, GoEmotions test, FinanceBench, FinQA yes/no, LongBench, HELMET, C-Eval/CMMLU/CMB/CMExam test, CommunityBench, ComPred, DISC-Law, AIDA-testc, decider, jevforge, jevbench, and GPQA/MMLU-Pro (all of both, which are also index items).

---

## Top picks (best expected transfer per unit of effort)

1. xLAM-60k + ToolACE + Glaive v2 + xlam-irrelevance, recast as per-function yes/no (after ToolRet dedupe). Covers BFCL and ToolRet.
2. When2Call train_sft + train_pref, as the native 4-way choice. Covers When2Call.
3. API-Bank `training-data`. Covers API-Bank.
4. CLINC150 train/val/oos + BANKING77 train + MASSIVE + HWU64, after test and JEVal dedupe. Covers the intent benchmarks.
5. ESCI train + WANDS. Covers ESCI.
6. HoVer train (oracle docs) + FEVER/VitaminC. Covers HoVer.
7. RAGTruth train + RAGBench (no cuad/finqa) + FAVA/MiniCheck. Covers RAGTruth.
8. ANLI train (NC) + WANLI + ContractNLI train/dev + NLI4CT train/dev. Covers ANLI, ContractNLI and NLI4CT.
9. iSarcasmEval train + VAST train/dev + ACOS train/dev (+ MAMS, SEntFiN). Covers sarcasm, stance, ACOS and FinEntity-style tasks.
10. GSM8K train with band distractors + MathQA/AQuA-RAT. Covers GSM8K.
11. ChessBench train shard + Lichess puzzles (FEN-deduped). Covers ChessBench.
12. CLadder generator (new seeds) + Corr2Cause. Covers CLadder.
13. reasoning-gym/SynLogic + regenerated MuSR + code-execution data (PyX/CodeI/O/self-made). Covers BBH, MuSR and CRUXEval.
14. MMLU auxiliary_train + MedMCQA + OpenScience. Covers MMLU-Pro and GPQA-style tasks.
15. Arts bundle: Habermas TRAIN, New Yorker split-0 train/val, Humicroedit train/val/FunLines, cfcolor train/probe, synthetic chord generator, PD-limerick perturbations.
16. New-domain: BoolQ (minus JEVal), StrategyQA, PubMedQA, SQuAD2 answerability, ETHICS, Aegis2/WildGuardMix.

## Not verified (to check before use)

- Licences: ACOS, VAST, NLI4CT, Humicroedit/FunLines, cfcolor, ToolRet-Training-20w, HomeBench, OpenBookQA, PAWS, QuALITY articles.
- Sizes marked approx.: ESCI train, WANDS, RAGBench total, API-Bank training, Corr2Cause, SEntFiN, WildGuardMix, CodeI/O.
- Whether When2Call train shares queries with BFCL live or ToolRet.
- CodeI/O's decontamination against CRUXEval.
- Row counts and overlap of the PhishNChips extra CSVs.
- That Humicroedit train contains no test original headlines.
- PDMX chord-symbol coverage.
