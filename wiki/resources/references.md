# 참고 문헌과 데이터 (v3, 2026-10-16 ~ 10-18)

[위키 홈](../README.md) · [v3 요약](../project/v3-summary.md) · [조사 기록 규칙](research-notes.md) · [데이터 출처 표](../designs/phase2-data-sources.md)

**확인 수준**

| 표기 | 뜻 |
|---|---|
| 카드 | 원문·공식 페이지에서 내용을 확인하고 논문 카드를 썼다 |
| 초록 | 초록이나 공식 페이지에서 제목·저자·핵심 결과만 확인했다 |
| 검색 | 웹 검색 요약으로 확인했다(2차 요약 포함) |
| 인용만 | 외부 검토가 인용했고 우리는 확인하지 않았다. 근거로 쓰지 않는다 |

2026-10-13 조사의 구조 관련 논문(GLiClass, Candidate-independent attention 등)은 [위키 홈](../README.md)의 논문 색인에 있다.

## 학습 방식·망각

| 논문 | 확인 | 쓰인 곳 |
|---|---|---|
| [WiSE-FT](../papers/wise-ft.md) (Wortsman 외, CVPR 2022) | 카드 | 원본과 섞기(`interpolate.py`, `merge.py`) |
| [L2-SP](../papers/l2-sp.md) (Li 외, ICML 2018) | 카드 | `train.py --anchor`(구현, 미평가) |
| [LoRA Learns Less and Forgets Less](../papers/lora-learns-less.md) (Biderman 외, TMLR 2024) | 카드 | 망각 해석. 전 층 LoRA r16이 지금까지 최선 레시피다(10-29, [LoRA 위치 비교](../project/lora-placement.md)) |
| [LP-FT](../papers/lp-ft.md) (Kumar 외, ICLR 2022) | 카드 | MLP head 반대 근거, 출력층 이동 해석 |
| [LiNeS](../papers/lines.md) (Wang 외, ICLR 2025) | 카드 | 층별 섞기 |
| [Model Stock](../papers/model-stock.md) (Jang 외, ECCV 2024) | 카드 | `merge.py --stock` |
| [체크포인트 평균 LAWA](../papers/checkpoint-averaging.md) (Sanyal 외 2023; Kaddour 2022) | 검색 | 에폭 평균 |
| [In-the-Wild 병합 비교](../papers/merging-in-the-wild.md) (Hitit 외, TMLR 2026) | 검색 | 단순 병합을 기준선으로 유지 |
| [Dataset Cartography](../papers/dataset-cartography.md) (Swayamdipta 외, EMNLP 2020) | 검색 | 미실행 후보 |
| T0 (Sanh 외, ICLR 2022), FLAN (Wei 외 2022; Chung 외 2022) | 검색 | 지시문·과제 다양성 근거 |
| Exploring Strategies for Generalizable Commonsense Reasoning (Ma 외, EMNLP 2021) | 검색 | 전체 미세조정 과적합, prefix-tuning 일반화 |
| Do Fine-tuned Commonsense LMs Really Generalize? (Kejriwal & Shen) | 검색 | 데이터셋 간 일반화 약함 |
| SAM for LM (Bahri 외 2022), SAM OOD (2412.05169) | 검색 | 미룸(학습 비용 2배) |
| Revisiting Few-sample BERT Fine-tuning (Zhang 외, ICLR 2021) | 검색 | 층별 학습률·재초기화 우선순위 낮춤 |
| PoE / Debiased focal loss (Utama 외 2020), PoE NLI (2604.19069) | 검색 | 방침 확인 필요로 보류. 일반 focal(γ 2)은 10-31에 시험했고 짧은 문항이 나빠졌다([Loss 비교](../project/loss-screen.md)) |
| MoFO (2407.20999) | 검색 | 미구현 후보 |
| Calibrate Before Use (Zhao 외 2021) | 원문 §5 | 10-19 g4b에서 prior 보정 α .25/.5/1을 시험했다: −0.39/−0.72/−4.69%p([다른 방법](../project/other-methods.md)) |
| AdaMerging (ICLR 2024), AIM (2502.02421), Locate-then-Merge (2505.16703), Model Breadcrumbs (ECCV 2024), TSV (CVPR 2025), TIES, DARE | 검색·초록 | 병합 후보 검토, 미실행 |
| LM-Cocktail, WARP, Safeguard FT (2412.19512) | 검색 | 원본 보간의 LLM 사례 |
| FINCH (2605.20005), 재노출 연구 (2512.22337), 문항별 망각 (2510.17776), mSFT (2603.21606), FLOW (2502.02797), FastMix (2606.14971), Post-Training Science (2609.01244), 불완전 학습 (2604.10079), Prism-1 (2610.22066), Label smoothing (Müller 외 2019) | 인용만 | 외부 검토 근거. 수치를 우리 근거로 쓰지 않음. label smoothing 자체는 10-31에 시험했고 CE와 같았다([Loss 비교](../project/loss-screen.md)) |
| ProLong (Gao 외, ACL 2025) | 검색 | **정정:** 장문 붕괴 진단 근거로 쓴 것은 잘못이었다 |

## 표현 정렬과 회전 동치 (2026-11-04 추가)

| 논문 | 확인 | 쓰인 곳 |
|---|---|---|
| [Proving and Improving the Platonic Representation Hypothesis with JEPAs](../papers/platonic-jepa.md), Balestriero·LeCun, arXiv:2610.39302v1 | 원문 본문·부록 A–C, 고정 커밋의 공식 정렬·지표·손실 코드 정적 대조 | [고정 plain Qwen 회전 pilot 및 사후 감사](../project/rotation-alignment.md), [실제 상태 경로의 CE+LeJEPA 후속](../project/jepa-decision.md). 논문의 독립 JEPA 학습을 재현한 것은 아님 |

## VLM 학습 방식 (2026-10-31 추가)

| 논문 | 확인 | 쓰인 곳 |
|---|---|---|
| Phi-4-multimodal (2503.01743), Image-LoRA (2512.19219), Idefics2 (2405.02246), X-Reasoner (2505.03981), Fine-tuning MLLMs Without Forgetting (2603.14493), Prismatic VLMs (2402.07865 §4.1), Cambrian-1 (2406.16860) | 원문 | [모달리티별 학습 방식](../papers/vlm-modality-finetuning.md). 이미지 LoRA 설계(위치, ViT 팔, 학습률, 데이터 섞기) |
| CogVLM (2311.03079), InternLM-XComposer2 Partial LoRA (2401.16420), Why are VLMs bad at image classification (2405.18415), EMT (2309.10313) | 초록 | 같은 곳 |
| Llama 3.2 Vision (Meta 블로그), Wings (2406.03496), Bring Reason to Vision (2505.05464), SmolVLM (2504.05299) | 검색 | 같은 곳 |
| Testing the Limits of Fine-Tuning for Visual Cognition (2502.15678), SFT Memorizes RL Generalizes (2501.17161), Hidden in plain sight (2506.08008), MoCLE (2312.12379), Qwen-VL (2308.12966 Table 3) | 원문 | [모달리티별 학습 방식 §8](../papers/vlm-modality-finetuning.md#8-이미지-학습이-처음-보는-벤치마크로-안-옮겨-가는-이유-11-02-추가). 이미지 학습이 처음 보는 벤치마크로 안 옮겨 가는 이유(11-02) |
| Eyes Wide Shut (2401.06209), AURORA (2412.03548), Synthetic Stimuli Real Gains (2511.11440), SPRITE (2512.16237), SITUATE (2602.00108) | 검색 | 같은 곳 |
| FLAN (2109.01652), LIMA (2305.11206), Caption This Reason That (2505.21538), Understanding Task Transfer in VLMs (2511.18787) | 초록 | [모달리티별 학습 방식 §8-1](../papers/vlm-modality-finetuning.md#8-1-텍스트-학습은-왜-옮겨-간-것처럼-보였나-11-02-추가). 텍스트 학습이 옮겨 간 것처럼 보인 이유(11-02) |
| Vision-Flan (2402.11690) | 검색 | 같은 곳 |

## 평가·편향

| 논문 | 확인 | 쓰인 곳 |
|---|---|---|
| [One Bias Score, Two Factors](../papers/option-bias-two-factors.md) (Ok 외, 2026) | 카드 | `bias_eval.py` |
| [객관식 코드·위치 편향](../papers/multiple-choice-bias.md) (Zheng 외 2024) | 카드 | RStd 지표 |
| PA-GRPO (2603.21016) | 초록 | 순열 일관성 학습 후보 |
| Punching Above Their Weight (2607.03801) | 초록 | 분류 head 비교의 범위 해석 |
| [검정력: With Little Power…](../papers/power-analysis.md) (Card 외, EMNLP 2020), Ng *ML Yearning* | 검색 | dev 크기 규칙 |
| 그룹 단위 분할·누수 관행 (GroupShuffleSplit 등, Kapoor & Narayanan 2022) | 검색 | `mix.hold_out` |

## 학습 데이터 (원본·라이선스)

출처별 manifest(`data/phase2/<출처>.manifest.json`, 지금 로컬에 없다. `data/`는 Hugging Face `Kyunghwan-kim/kkh-decision-data`에 있다)에 revision, 라이선스, 버린 수가 있다. 아래는 2026-10-17 이후 추가한 출처다. 그 전 출처는 [데이터 출처 표](../designs/phase2-data-sources.md)에 있다.

| 데이터 | 논문 | 라이선스(확인 위치) |
|---|---|---|
| SEntFiN 1.0 | Sinha 외 2022 | MIT (저장소 LICENSE) |
| TAT-QA | Zhu 외, ACL 2021 | CC BY 4.0 (README) |
| Social IQa | Sap 외, EMNLP 2019 | CC-BY-4.0 (AllenAI, Hub 카드 미표기) |
| Cosmos QA | Huang 외, EMNLP 2019 | CC-BY-4.0 |
| MultiNLI | Williams 외, NAACL 2018 | CC-BY-3.0 (카드) |
| FaithDial | Dziri 외, TACL 2022 | MIT |
| Moral Stories | Emelin 외, EMNLP 2021 | MIT |
| e-CARE | Du 외, ACL 2022 | MIT (저장소 LICENSE) |
| GoEmotions | Demszky 외, ACL 2020 | Apache-2.0 |
| DBpedia 계층 분류 | (Kaggle 추출) | 카드는 CC0이지만 본문은 위키백과 추출이라 CC-BY-SA로 본다([DBpedia 안내](https://www.dbpedia.org/about/)). 2026-10-18 정정 |
| 금융 트윗 주제·감성 | zeroshot | MIT |
| KILT WNED | Petroni 외, NAACL 2021; 원 데이터 Guo & Barbosa 2018 | KILT MIT, 본문 위키(CC-BY-SA). 원 WNED 배포처([Borealis DOI 10.7939/DVN/10968](https://borealisdata.ca/dataset.xhtml?persistentId=doi:10.7939/DVN/10968))는 "라이선스 없음, 사용 조건 정보 없음"(2026-10-18 확인). **사용 여부는 사용자 판단 대기** |
| VitaminC | Schuster 외, NAACL 2021 | CC-BY-SA-3.0 |
| PAWS-Wiki | Zhang 외, NAACL 2019 | 자유 이용 (Google LICENSE) |
| SciTail | Khot 외, AAAI 2018 | Apache-2.0 (저장소 LICENSE) |
| **5차(2026-10-19)** | | |
| TabFact | Chen 외, ICLR 2020 | MIT(저장소), 표는 위키(CC-BY-SA). HF 카드 CC-BY-4.0 |
| CondaQA | Ravichander 외, EMNLP 2022 | Apache-2.0 |
| ROPES | Lin 외, MRQA 2019 | CC-BY-4.0 |
| FOLIO | Han 외, 2022 | CC-BY-SA-4.0 (GitHub). HF판은 MIT·동의 절차 |
| ABCD | Chen 외, NAACL 2021 | MIT |
| HelpSteer2 | Wang 외, 2024 | CC-BY-4.0 (응답은 모델 작성, 점수는 사람) |
| FiNER-139 | Loukas 외, ACL 2022 | CC-BY-SA-4.0 |
| Zeshel | Logeswaran 외, ACL 2019 | CC-BY-SA (Fandom 위키; 공식 README) |
| Few-NERD | Ding 외, ACL 2021 | CC-BY-SA-4.0 |
| DROP | Dua 외, NAACL 2019 | CC-BY-SA-4.0 |
| Taskmaster-1 | Byrne 외, EMNLP 2019 | CC-BY-4.0 |
| **6차(2026-10-19)** | 출처별 manifest와 `phase6-candidates-vetted.json`에 근거 URL | |
| HelpSteer3 · Arena 55k · MT-Bench 사람 판정 | NVIDIA 2025 · LMSYS · Zheng 외 2023 | CC-BY-4.0 · Apache-2.0 · CC-BY-4.0(MT-Bench는 제외) |
| OASST2 · HH-RLHF · summarize_from_feedback | Köpf 외 2023 · Bai 외 2022 · Stiennon 외 2020 | Apache-2.0 · MIT · 수정 MIT(게시물 CC-BY-4.0) |
| IBM ArgQ-30k · Claim Stance · UKPConvArg1 | Gretz 외 2020 · Bar-Haim 외 2017 · Habernal & Gurevych 2016 | CC-BY-SA-3.0 · CC-BY-SA-3.0 · CC-BY-4.0 |
| Quoref · QuAC · ShARC · CREAK · SNLI | Dasigi 외 2019 · Choi 외 2018 · Saeidi 외 2018 · Onoe 외 2021 · Bowman 외 2015 | CC-BY-4.0 · CC-BY-SA-4.0 · CC-BY-SA-3.0 · MIT · CC-BY-SA-4.0 |
| InfoTABS · WikiTableQuestions · SVAMP · StepGame · SpartQA | Gupta 외 2020 · Pasupat & Liang 2015 · Patel 외 2021 · Shi 외 2022 · Mirzaee 외 2021 | Apache-2.0 · CC-BY-SA-4.0 · MIT · MIT · MIT |
| MAUD · PubHealth · HEAD-QA | Wang 외 2023 · Kotonya & Toni 2020 · Vilares & Gómez-Rodríguez 2019 | CC-BY-4.0 · MIT · MIT(+스페인 보건부 재사용 조건) |
| Civil Comments · HateXplain · MultiWOZ 2.2 | Borkan 외 2019 · Mathew 외 2021 · Zang 외 2020 | CC0 · MIT · MIT |
| MTOP · SNIPS · ReDial | Li 외 2021 · Coucke 외 2018 · Li 외 2018 | CC-BY-SA-4.0 · CC0 · CC-BY-4.0 |
| FewRel · GAP · SemCor 3.0 | Han 외 2018 · Webster 외 2018 · Miller 외 1993 | MIT · Apache-2.0 · Princeton WordNet 라이선스 |
| KLUE · ChID · CMRC2018 / DRCD · JGLUE · TyDi QA | Park 외 2021 · Zheng 외 2019 · Cui 외 2019 / Shao 외 2018 · Kurihara 외 2022 · Clark 외 2020 | CC-BY-SA-4.0 · Apache-2.0 · CC-BY-SA-4.0 / CC-BY-SA-3.0 · CC-BY-SA-4.0 · Apache-2.0 |

- **평가 전용(fresh-v1):** PIQA, QASC, QuaRTz, Balanced COPA, SuperGLUE CB·WiC, GoEmotions validation, ETHICS, SWAG, BigBench 7개 과제. 일부는 비상업 라이선스라 학습에는 쓰지 않는다.
- **6차 후보 검증:** `runs/analysis/phase6-candidates-vetted.json`(지금 로컬에 없다. 70개: 포함 50·보류 11·제외 9, 라이선스 원문 위치와 다운로드 경로). 포함 판정이어도 평가셋과 원문을 공유할 수 있는 MultiHiertt·TAT-DQA(FinQA와 같은 연차보고서), Re-DocRED(HotpotQA 문단), Social Chemistry·Scruples(ETHICS와 같은 AITA 글)는 쓰지 않는다. 받은 코드를 실행해야 하는 MBPP, 평가 전용인 Belebele도 뺐다. 외부 제안 목록(`additional-dataset-candidates-20261019.json`, 로컬에 없다)의 Qasper는 LongBench 하위 과제라 제외 대상이다.
- **제외:** SciQ, ANLI, Financial PhraseBank(비상업), FiQA(접근 제한), SHP(공개 라이선스 없음), Zeshel(당시 "공식 배포 경로 없음"으로 뺐으나 **틀렸다**: [공식 README](https://github.com/lajanugen/zeshel)에 Google Drive 배포 링크와 CC-BY-SA 표기가 있다. 2026-10-19 정정 후 5차에 넣었다(zeshel-train)), FinQA·FEVER·TREC·AIDA·HaluEval(JEVal과 같은 데이터셋).
