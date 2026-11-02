# 이미지 학습 데이터 출처 후보 (2026-10-31)

[위키 홈](../README.md) · [Vision 기준점](../project/vision-baseline.md) · [모달리티별 학습 방식](../papers/vlm-modality-finetuning.md) · [텍스트 데이터 출처](phase2-data-sources.md)

**상태(11-02): 다 받았다(MegaDepth만 403으로 실패). 학습 셋으로 바꾼 결과는 [이미지 학습 셋과 첫 LoRA 학습](../project/vision-train-screen.md)에 있다.** 사용자가 "다 받아줘"라고 해서 보드 train split도 포함했다(제출 때 공개). 비상업 출처는 비교 실험에만 쓴다. 목록과 받는 코드는 `qwen_decision/v3/vision_sources.py`이고, 받은 기록(revision, 크기)은 `data/raw/vision-manifest.jsonl`에 남는다. 행 수, 크기, 라이선스 태그는 10-31에 HF API(`dataset_info`, datasets-server `/size`)로 확인했다. 라이선스는 HF 태그나 카드 기준이라 원본과 다를 수 있다.

**추가(11-02): 분야 넓히기용 18개.** DTD, GTSRB, EuroSAT, RESISC45(`tanganke/*`), FGVC-Aircraft, PCam, EyePACS 망막, Rendered SST-2, SVHN, smallNORB(고도·방위각), dSprites 방향, DMLab, CLEVR 가장 가까운 물체 거리, Country211(`clip-benchmark/wds_*`, train shard 일부), VQA-RAD, PathVQA, IAM 손글씨를 받았다(약 7.5GB).
- 쓴 것과 뺀 것은 [분야 넓히기](../project/vision-train-screen.md#결과-분야-넓히기-clean-div-11-02-1638--1845)에 있다.
- 쓰지 않은 것: VQA-RAD(MathVista testmini의 원본), smallNORB 방위각과 dSprites(한 장으로 판단할 수 없는 라벨)다.
- 받지 않은 것: KITTI·nuScenes(CV-Bench의 Omni3D와 MME-RealWorld의 원본)다.
- `clip-benchmark` shard는 클래스 순으로 정렬된 것이 있다(Rendered SST-2, Aircraft). 그래서 Rendered SST-2는 전부 받았고, Aircraft는 shard 2개에 100개 기종 중 48개만 들어 있다.

**전부 받지 못한 것과 이유**
- **너무 큰 저장소는 일부만 받았다.** 디스크가 426GB 남아 있었다.
  - CO3D: 5.3TB 중 tv 카테고리 3GB
  - OS-Atlas: 816GB 중 linux·macOS 데스크톱과 RICO·모바일 부분. 웹 598GB는 뺐고, IG_data에는 Mind2Web 이미지가 들어 있어서 뺐다.
  - Android in the Wild: 494GB 중 general 13.6GB
  - AndroidControl: train 275 shard 중 70개
  - NYU Depth: 35GB 중 2 shard
  - Places365: 39 shard 중 8개
  - The Cauldron의 표 subset 두 개(robut_wikisql, robut_wtq, 10GB): 우선순위가 낮아서 뺐다.
- **약관 동의나 신청이 필요해서 받지 못했다:** ImageNet(gated), SPAR-7M의 원본 이미지(ScanNet 등 신청 필요, 주석만 공개), Winoground(gated).
- **받는 방법이 막혔다:** SROIE(Google Drive), WebLINX 스크린샷(별도 대형 저장소).
- **평가로 돌렸다:** Memotion과 WildReceipt는 학습에 쓰지 않고 [다른 벤치마크 시험셋](../project/vision-heldout.md)에 넣었다.

## 보드가 묻는 문제 (공개 vision 0.3)

모든 문항은 객관식이다. 이미지는 1~2장이고 선택지는 2~10개이며, 글자 하나로 답한다. 예시는 우리가 빌드한 평가 셋(`data/vision-work/vision/full/`)에서 옮겼다. wise80은 [기준점](../project/vision-baseline.md)의 skill이다.

| 분야 | 보드 셋 (행, 가중치) | 세부 과제와 문제 모양 | wise80 |
|---|---|---|---:|
| 실사진 공간·개수·거리 | CV-Bench (2,038, 1) | 개수 788("이미지에 쿠션이 몇 개?"), 관계 650("벽은 계단의 위인가 아래인가?"), 거리 600("빨간 상자의 의자에 냉장고와 문 중 어느 쪽이 실제로 더 가까운가?") | 0.604 |
| 실사진 공간 (운전 장면 등) | RealWorldQA (589, 1) | "세 물체 중 가장 작은 것은?"(3지선다), "가장 가까운 빨간 차가 가장 가까운 세미트럭보다 가까운가?"(Yes/No) | 0.420 |
| 세밀한 지각 | BLINK (961, 1; 574행이 2장) | 개수, 상대 깊이("원으로 표시한 A와 B 중 카메라에 더 가까운 점은?"), 공간 관계("차가 고양이 아래에 있나?"), 다시점("비디오의 두 프레임, 카메라가 시계 방향으로 돌았나?"), 시각·의미·기능 대응(두 이미지에서 같은 점 찾기) | 0.393 |
| 논문 차트 | CharXiv (532, 1) | arXiv 그래프. "정확도가 가장 많이 떨어진 모델은?", "두 번째로 시간이 긴 모델의 정확도는?", "점수 50 미만인 점은 몇 개?"(4지선다) | 0.629 |
| 인포그래픽 | InfographicVQA (244, 1) | "나이지리아의 인터넷 사용자는 몇 명인가?" 오답은 같은 인포그래픽의 다른 답이다(4지선다) | 0.820 |
| 영수증·서식 | KIE: CORD + FUNSD (2,465, 1) | 필드 값("거스름돈은 얼마인가?", "FAX NO. 칸에 무엇이 적혔나?"), 필드 유무("현금 결제 금액이 있나?" Yes/No) | 0.715 |
| 웹 조작 | Mind2Web (973, 1) | 쇼핑, 여행, 엔터테인먼트 사이트. 과제 문장과 이전 행동이 주어지고, 스크린샷에 A~D 상자가 그려져 있다. "다음에 조작할 요소는?" | 0.424 |
| 이미지와 문장 맞추기 | Winoground (800, 1) | 같은 단어를 순서만 바꾼 두 캡션 중 이미지에 맞는 것을 고른다. **아직 빌드하지 못했다(gated)** | — |
| 혐오 밈 | Hateful Memes (1,213, 1) | "이 밈은 혐오인가?" Yes/No. 이미지와 글자를 함께 봐야 한다 | 0.169 |
| 공학 계산 | R-Bench-M (665, 0.5) | 기계·재료 역학 그림과 긴 계산 문제, 수치 답 6지선다("Other options are incorrect" 포함) | 0.177 |
| 전문 과목 | MMMU-Pro vision (1,730, 0.25) | 역사, 미술, 물리, 의학 등 약 30과목, 과목당 60문항. **질문과 선택지(대부분 10개)가 스크린샷 안에 있다** | 0.118 |

## 규칙 (텍스트 데이터와 같다)

- **정답은 사람이나 프로그램이 만든 것만 쓴다.** 모델이 만든 라벨은 쓰지 않는다(교사 금지).
  - 빠지는 것: GPT로 만든 지시 데이터(LLaVA-665K, LRV, ShareGPT4V, Co-Instruct, MIMIC-IT), 검출기로 센 PixMo-Count train, T5가 만든 ChartQA 질문(20,901개), LLM이 만든 WebSight.
- **평가 셋은 어떤 형태로도 학습에 쓰지 않는다.** 보드 vision 셋의 평가 split, 그리고 MMMU, MMBench, SEED, MME, POPE, MathVista, ScreenSpot 같은 일반 벤치마크가 여기에 해당한다.
- **보드 벤치마크의 train split은 아직 정하지 않았다(사용자 결정 대기).** 보드 규칙은 "공개하면 허용"이다.
  - 해당 셋: InfographicVQA train, CORD train(800), FUNSD train(149), Mind2Web train(7,775), Hateful Memes train(8,500).
  - 보드가 쓰는 split: InfographicVQA validation, CORD validation과 test, FUNSD testing, Mind2Web test_website, Hateful Memes dev_seen과 test_seen.
- **라이선스가 비상업이거나 확인되지 않은 출처는 비교 실험에만 쓴다.** 공개 모델에 쓸지는 제출 전에 따로 검토한다(텍스트 데이터와 같은 규칙).
- **이미지 단위로 중복을 뺀다.** 평가 이미지 11,410장(Winoground를 받으면 800장 추가)과 비교한다. 재인코딩된 이미지도 잡도록 정확 해시와 지각 해시(pHash)를 둘 다 쓴다.
  - **CV-Bench가 가장 위험하다.** 이미지가 COCO 805장, ADE20K 633장, Omni3D 600장이다. GQA(Visual Genome은 COCO와 겹친다), VSR(COCO 2017), TallyQA, COCO 캡션, Localized Narratives(COCO와 ADE20K)가 같은 이미지 풀을 쓴다.
  - BLINK의 대응·깊이 과제는 SPair-71k, HPatches 같은 원본을 쓴다. 이 출처들을 쓰면 반드시 중복을 확인한다.
  - CharXiv는 arXiv 그림이라 DaTikz(arXiv TikZ)는 뺀다.
- 형식은 텍스트와 같다. 이미지를 State에 넣고, 질문과 선택지를 주고, 코드 하나를 고르게 한다. 자유 답 데이터는 같은 이미지의 다른 답이나 같은 출처의 답에서 오답을 뽑아 객관식으로 바꾼다. 보드가 InfographicVQA, KIE, CharXiv에 쓴 방식과 같다.

## 몇 문항이 적당한가

| 단위 | 크기 | 3090 학습 시간(추정) | 용도 |
|---|---:|---|---|
| 방식 거르기 | 16,384행 | 1~3시간 | ViT 얼림 대 ViT LoRA, 텍스트 섞는 비율 같은 방식 비교. seed 5개 |
| 본 학습 | 65,536행 | 4~12시간 | 고른 방식의 첫 본 학습 |
| 확장 | 13만~26만 행 | 1~2일 | 16k → 65k에서 계속 오를 때만 |
| **모아 둘 풀** | **약 30만~35만 문항** | | 출처당 상한 약 1만, 약 40개 출처 |

- 시간은 10-31 측정에서 나왔다. 1.6MP 이미지 1장짜리 문항이 0.48초(ViT 얼림)~0.66초(ViT LoRA)였다. 학습 이미지는 대부분 이보다 작으므로(GQA 약 0.2MP) 아래쪽에 가까울 것이다.
- **풀을 학습량보다 크게 모으는 이유:** 텍스트에서 같은 양이면 다양한 데이터가 나았다. BDH에서 4,096행 4회보다 16,384행 1회가 +1.7점이었다. 그래서 같은 문항을 반복하기보다 출처를 넓힌다.
- 출처당 상한 약 1만은 텍스트 v3-gold6과 같은 기준이다(출처당 최대 약 9,700행). 한 출처가 섞음을 지배하지 않게 한다.
- **텍스트 판정 데이터를 25~50% 섞는다.** 텍스트 능력을 지키고, "질문과 상관없이 클래스를 고르는" 지름길을 막기 위해서다. 2603.14493은 지시 데이터를 약 50% 섞어 이 지름길을 고쳤다.
- 출처마다 100~200문항을 dev로 떼어 둔다(약 5천 문항). 공개 vision 셋은 마지막 확인에만 쓴다.
- 문헌의 규모는 다음과 같다. LLaVA-1.5 SFT는 66.5만, Cambrian은 700만이다. 하지만 이들은 정렬부터 학습한 것이다. 우리는 이미 이미지를 아는 Qwen3.5에 판정 형식과 보드 능력만 더하므로 훨씬 적어도 될 것으로 본다(추정, 16k → 65k 곡선으로 확인한다).

## 후보 출처 (보드 능력별)

"정답"은 사람 / 프로그램(사람 주석이나 합성 장면에서 규칙으로 만듦)이다. "보드 train"은 위의 결정 대기 항목이다. The Cauldron은 `HuggingFaceM4/the_cauldron`(50개 셋의 train split만 있고, MMMU·MathVista·MMBench 시험 이미지는 이미 뺐다. 전체 169GB)이다.

### 1. 공간·관계·개수 (CV-Bench, BLINK 개수·공간, RealWorldQA)

| 출처 | 저장소 | 규모 | 정답 | 라이선스 | 비고 |
|---|---|---|---|---|---|
| GQA balanced | `lmms-lab-encoder/GQA` train | 질문 943,000 / 이미지 72,140 (저장소 30GB) | 프로그램(VG 장면 그래프) | MIT 태그 | 관계·속성·비교. VG 이미지라 CV-Bench 중복 확인 필수 |
| VSR | `cambridgeltl/vsr_random` train | 7,680 | 사람(참/거짓) | CC BY 4.0 | 공간 관계 66종. COCO 2017 이미지 |
| SAT | `array/SAT` train / static | 172,384 / 127,405 (7.6GB) | 프로그램(ProcTHOR 합성) | MIT | 거리, 깊이, 시점, 움직임(2장). test 150은 실제 사진 평가라 쓰지 않는다 |
| TallyQA | Cauldron `tallyqa` | 질문 183,986 / 이미지 98,680 (4.7GB) | 사람 | 원본 확인 필요 | 개수. COCO·VG 이미지 |
| Visual7W | Cauldron `visual7w` | 질문 69,817 | 사람(원래 4지선다) | 원본 확인 필요 | |
| A-OKVQA | `HuggingFaceM4/A-OKVQA` train | 17,056 | 사람(원래 객관식) | 태그 없음 | |
| VQAv2, COCO-QA, OK-VQA | Cauldron | 질문 44만 / 7.9만 / 9천 | 사람 / 프로그램(COCO-QA) | 원본 확인 필요 | 일반 질문. 상한을 낮게 둔다 |

### 2. 깊이·대응·여러 장 (BLINK 깊이·대응·다시점)

| 출처 | 저장소 | 규모 | 정답 | 라이선스 | 비고 |
|---|---|---|---|---|---|
| NYU Depth v2 | `sayakpaul/nyu_depth_v2` train | 뷰어 기준 3,100장(부분 변환일 수 있음, 3GB) | 센서 깊이 → 프로그램(두 점 중 가까운 쪽) | Apache 태그 | BLINK Relative_Depth 형식 |
| SPair-71k | `0jl/SPair-71k` | 쌍 7만 | 사람 키포인트 → 프로그램 | 확인 필요 | 의미 대응. BLINK가 같은 원본을 쓰므로 중복 확인 필수. 빌더 필요 |
| HPatches | `vbalnt/hpatches` | | 호모그래피 → 프로그램 | MIT 태그 | 시각 대응. 같은 주의 |
| NLVR2 | Cauldron `nlvr2` | 질문 86,373 / 이미지 50,426 (10.8GB) | 사람(참/거짓, 2장) | 원본 확인 필요 | 두 장 비교 |
| SAT 움직임 문항 | 위 SAT | | 프로그램(2장) | MIT | 다시점 |
| 비디오 프레임 쌍 | 카메라 pose가 있는 비디오 셋(CO3D, DL3DV, RealEstate10K 등, 저장소·라이선스 미확인) | 빌더 필요 | 기록된 카메라 pose → 프로그램(시계/반시계 방향, 어느 쪽이 먼저 등) | 확인 필요 | BLINK Multi-view_Reasoning이 "비디오의 두 프레임, 카메라가 어느 방향으로 움직였나" 형식이다. 중복 확인 필수 |

- **비디오 자체는 모으지 않는다(10-31).** vision 보드 0.3에는 비디오 문항이 없다. 키트 문서에 비디오가 없고, 공개 행은 이미지 1장 10,836개, 2장 574개다. 비공개 절반도 "같은 능력"을 본다. 비디오는 프레임이 많아 토큰, 학습 시간, 디스크가 크고 지연 규칙(1초)에도 걸린다. 비디오에서 뽑은 프레임 2장짜리 문항만 위처럼 쓴다.

### 3. 차트 (CharXiv)

| 출처 | 저장소 | 규모 | 정답 | 라이선스 | 비고 |
|---|---|---|---|---|---|
| ChartQA human | `HuggingFaceM4/ChartQA` train, `human_or_machine = human` | 7,398 | 사람 | GPL-3.0 태그 | machine 20,901개는 T5가 만든 질문이라 뺀다 |
| PlotQA | Cauldron `plotqa` | 질문 2,025만 / 그림 157,070 | 프로그램 | 원본 확인 필요 | 합성 차트 |
| FigureQA | Cauldron `figureqa` | 질문 133만 / 그림 10만 | 프로그램(참/거짓) | 원본 확인 필요 | |
| DVQA | Cauldron `dvqa` | 질문 233만 / 그림 20만 | 프로그램 | 원본 확인 필요 | 막대 차트 |
| MapQA | Cauldron `mapqa` | 질문 48만 | 프로그램 | 원본 확인 필요 | 지도 차트 |

### 4. 인포그래픽·문서·OCR (InfographicVQA)

| 출처 | 저장소 | 규모 | 정답 | 라이선스 | 비고 |
|---|---|---|---|---|---|
| DocVQA train | Cauldron `docvqa` | 질문 39,463 / 문서 10,189 (6.9GB) | 사람 | 원본(RRC) 확인 필요 | |
| InfographicVQA train | Cauldron `infographic_vqa` | 질문 10,074 / 2,118장 | 사람 | 원본(RRC) 확인 필요 | **보드 train** |
| TextVQA, ST-VQA | Cauldron | 질문 34,602 / 23,121 | 사람 | TextVQA CC BY 4.0 | 장면 글자 |
| OCR-VQA | Cauldron `ocrvqa` | 질문 80만 | 프로그램(책 메타데이터) | 원본 확인 필요 | 책 표지 |
| VisualMRC | Cauldron `visualmrc` | 질문 11,988 | 사람 | 원본 확인 필요 | 웹 페이지 독해 |
| 표 (TabMWP, TAT-QA, HiTab, FinQA, WTQ, SQA) | Cauldron | 합계 약 13만 | 사람 / 프로그램 | 원본 확인 필요 | 표 이미지. 우선순위 낮음 |

### 5. 영수증·서식 (KIE: CORD + FUNSD)

| 출처 | 저장소 | 규모 | 정답 | 라이선스 | 비고 |
|---|---|---|---|---|---|
| CORD train | `naver-clova-ix/cord-v2` train | 800장 | 사람 주석 → 프로그램 | CC BY 4.0 | **보드 train**. 보드와 같은 방식(필드 값 객관식, 필드 유무 Yes/No)으로 한 장에 여러 문항 |
| FUNSD train | `nielsr/funsd` train | 149장 | 사람 주석 → 프로그램 | 비상업 연구 | **보드 train**. 비교 실험만 |
| SROIE | `darentang/sroie` train | 626장 | 사람 주석 → 프로그램 | 확인 필요 | 영수증 |
| WildReceipt | `Theivaprakasham/wildreceipt` train | 1,267장 | 사람 주석 → 프로그램 | Apache 태그 | 영수증 |

### 6. 웹·앱 화면 (Mind2Web)

| 출처 | 저장소 | 규모 | 정답 | 라이선스 | 비고 |
|---|---|---|---|---|---|
| Mind2Web train | `osunlp/Multimodal-Mind2Web` train | 7,775 단계 (저장소 13.6GB) | 사람(실제 조작 기록) | 데이터 CC BY 4.0 | **보드 train**. 보드와 같은 set-of-marks 형식. test_task, test_domain도 평가 split이라 쓰지 않는다 |
| RICO-ScreenQA | `bevaya/RICO-ScreenQA` train | 68,980 (3.3GB) | 사람 | CC BY 4.0 | 앱 화면 질문 |
| Screen2Words | Cauldron `screen2words` | 15,743 | 사람(화면 요약) | 원본 확인 필요 | 요약 고르기로 변환 |

### 7. 이미지와 문장 맞추기 (Winoground)

| 출처 | 저장소 | 규모 | 정답 | 라이선스 | 비고 |
|---|---|---|---|---|---|
| ImageCoDe | `BennoKrojer/ImageCoDe` train | 16,594 | 사람 | AFL-3.0 | 비슷한 이미지 10장 중 설명에 맞는 것 고르기. Winoground처럼 미세한 차이를 본다 |
| COCO 캡션 | `yerevann/coco-karpathy` train + restval | 이미지 113,287 | 사람 캡션 → 프로그램 오답(다른 이미지의 캡션, 단어 순서 바꾸기) | 원본 확인 필요 | 이미지 URL만 있다. CV-Bench 중복 확인 필수 |

### 8. 혐오 밈 (Hateful Memes)

| 출처 | 저장소 | 규모 | 정답 | 라이선스 | 비고 |
|---|---|---|---|---|---|
| Hateful Memes train | `neuralcatcher/hateful_memes` train (Cauldron에도 있다) | 8,500 | 사람 | 연구용, 파생물 재배포 금지 | **보드 train이고 라이선스 문제도 있다.** 공개 모델에 써도 되는지 확인 전에는 비교 실험만 |

### 9. 과학·추론 (R-Bench-M, MMMU-Pro)

| 출처 | 저장소 | 규모 | 정답 | 라이선스 | 비고 |
|---|---|---|---|---|---|
| ScienceQA | `derek-thomas/ScienceQA` train | 12,726 (이미지 있는 것은 약 절반) | 사람(원래 객관식) | CC BY-SA 태그(원본은 CC BY-NC-SA) | |
| AI2D | Cauldron `ai2d` | 질문 9,708 | 사람(원래 객관식) | 원본 확인 필요 | 과학 도표 |
| TQA | Cauldron `tqa` | 질문 6,501 | 사람(원래 객관식) | 원본 확인 필요 | 교과서 도표 |
| IconQA | Cauldron `iconqa` | 질문 29,859 | 사람 / 프로그램 | 원본 확인 필요 | |
| RAVEN, CLEVR-Math, GeomVerse, Inter-GPS | Cauldron | 4.2만 / 79만 / 9천 / 2천 | 프로그램 / 사람(Inter-GPS) | 원본 확인 필요 | 도형·수학 |
| **우리 텍스트 gold를 이미지로 그리기** | `data/mix/` 텍스트 문항 | 원하는 만큼 | 기존 gold | 원 출처 그대로 | 문제와 선택지를 이미지 안에 그린다. MMMU-Pro vision 형식이다. 새 라벨이 필요 없다 |

### 10. 분류 (사용자 제안)

| 출처 | 저장소 | 규모 | 정답 | 라이선스 | 비고 |
|---|---|---|---|---|---|
| Oxford Pets | `timm/oxford-iiit-pet` train | 3,680 | 사람 | CC BY-SA 4.0 | |
| Oxford Flowers | `nkirschi/oxford-flowers` train | 7,169 | 사람 | unknown | |
| Food-101 | `ethz/food101` train | 75,750 (5GB) | 사람 | unknown(연구용) | |
| Stanford Cars | `tanganke/stanford_cars` train | 8,144 | 사람 | 연구용 | |
| Places365 | `ljnlonoljpiljm/places365-256px` train | 1,803,460 (24.7GB) | 사람 | 연구용 | 일부만 |
| ImageNet-1k | `ILSVRC/imagenet-1k` | 128만 | 사람 | gated, 비상업 | 계정 약관 동의 필요 |

- 오답은 같은 상위 범주에서 뽑는다(예: 다른 개 품종).
- 상한을 낮게 둔다(출처당 2천~5천). 분류만 많으면 "질문과 상관없이 클래스를 고르는" 습관이 생길 수 있다(2603.14493).

## 약한 분야 보강 (10-31 추가)

[기준점](../project/vision-baseline.md)에서 약했던 분야(세밀한 지각 0.39, 실사진 공간 0.42, 웹 조작 0.42, 혐오 밈 0.17, 공학·전문 과목 0.12~0.18)를 채울 후보를 더 찾았다. 행 수와 라이선스 태그는 HF API로 확인했다.

| 분야 | 출처 | 저장소 | 규모 | 정답 | 라이선스 | 비고 |
|---|---|---|---|---|---|---|
| 개수 | FSC-147 | `isentropic/FSC147` train | 6,146장 | 사람(물체마다 점을 찍음) | MIT 태그 | 개수 범위가 넓다(7~3,731). 구간 객관식으로 바꾼다 |
| 공간·깊이·거리 | SPAR-7M | `jasonzhango/SPAR-7M` | 문항 1,628만 | 프로그램(3D 스캔 주석) | 주석 MIT. 이미지는 ScanNet, ScanNet++, Matterport3D, Structured3D에서 직접 받아야 하고 비상업이다 | 양은 가장 많지만 원본 이미지 신청이 필요하다 |
| 깊이·대응 | MegaDepth | `vctvct123/Megadepth` | 110,675 | SfM 깊이와 카메라 → 프로그램 | 확인 필요 | 상대 깊이, 두 장 사이 같은 점 찾기 |
| 깊이 | Hypersim | `ritianyu/Hypersim` | 확인 필요 | 합성 렌더링의 정확한 깊이 → 프로그램 | 확인 필요 | |
| 다시점 | CO3D v2 | `mutou0308/co3dv2` | 확인 필요(뷰어 오류) | 기록된 카메라 pose → 프로그램 | Apache 태그(원본 CC BY-NC) | BLINK 다시점과 같은 "카메라가 어느 방향으로 돌았나" |
| 운전 장면 | nuScenes-QA (mini) | `KevinNotSmile/nuscenes-qa-mini` | 1,534 | 프로그램(3D 주석) | CC BY-NC-SA | RealWorldQA의 "가장 가까운 빨간 차가 트럭보다 가까운가" 형식에 가깝다. 전체본은 따로 찾아야 한다 |
| 웹·앱 화면 위치 찾기 | OS-Atlas | `OS-Copilot/OS-Atlas-data` | 수백만 요소(웹, 모바일, 데스크톱) | 접근성 트리·HTML에서 뽑은 지시와 상자 → 프로그램 | Apache 태그 | 상자 A~D를 그리고 지시에 맞는 요소를 고르게 하면 Mind2Web 형식이 된다 |
| 앱 조작 | Android in the Wild | `leosltl/Android-in-the-Wild` | 확인 필요 | 사람 조작 기록 | CC BY 4.0 | 다음 행동 고르기 |
| 앱 조작 | AndroidControl | `smolagents/android-control` train | 12,232 (67GB) | 사람 조작 기록 | 확인 필요 | 크다. 일부만 |
| 웹 조작 | WebLINX | `McGill-NLP/WebLINX` chat/train | 24,418 | 사람 조작 기록 | CC BY-NC-SA | 비교 실험만 |
| 혐오·공격적 밈 | Memotion | `Ahren09/MMSoc_Memotion` train | 5,593 | 사람(공격성 등급) | 확인 필요 | Hateful Memes train을 못 쓸 때의 대안 |
| 혐오 밈 (만든 것) | 우리 텍스트 혐오 gold를 사진 위에 글자로 얹기 | `data/mix/` 혐오·독성 출처 | 원하는 만큼 | 기존 gold | 원 출처 그대로 | 글을 읽고 판단하는 부분만 배운다. Hateful Memes의 "그림과 글이 합쳐져야 혐오"인 문항은 못 배운다 |
| 기하·공학 그림 | Geometry3K | `hiyouga/geometry3k` train | 2,101 | 사람 | MIT 태그 | R-Bench-M, MMMU-Pro 수학 |
| 기하 | GeoQA+ | `Yuting6/geoqa_plus` train | 6,027 | 사람 | 확인 필요 | |

- **공간·깊이 쪽은 프로그램으로 만든 정답이 충분하다.** SAT, SPAR-7M, MegaDepth, Hypersim, CO3D에서 BLINK의 깊이·대응·다시점 형식을 그대로 만들 수 있다. 이 분야를 가장 두껍게 채운다.
- **웹 쪽은 OS-Atlas가 크다.** 접근성 트리에서 나온 지시라 사람이 쓴 과제 문장과는 다르다. Mind2Web train, AITW, AndroidControl처럼 사람이 조작한 기록과 섞는다.
- **혐오 밈은 여전히 얇다.** Hateful Memes train이 거의 유일한 진짜 대응 데이터다. Memotion과 "글자 얹기"는 보조다.
- **공학·전문 과목은 학습 데이터로 크게 올리기 어렵다.** 가중치도 낮다(0.5, 0.25). 텍스트 gold를 스크린샷으로 그리는 방법(MMMU-Pro 형식)이 가장 싸다.

## 쓰지 않는 것

| 출처 | 이유 |
|---|---|
| PixMo-Count train | Detic 검출기로 센 라벨(모델 라벨). val/test는 사람이 확인했지만 PixMo의 평가 split이다 |
| ChartQA machine 질문 | T5가 만든 질문(20,901개) |
| LLaVA-665K, LRV, ShareGPT4V, Co-Instruct, MIMIC-IT(Cauldron `mimic_cgd`) | GPT가 만든 지시와 답 |
| WebSight, DaTikz, Chart2Text, VisText, RenderedText, IAM, TextCaps, Localized Narratives | 코드 생성이나 캡션 생성용이라 객관식으로 바꾸기 어렵다. WebSight는 LLM이 만들었고, DaTikz는 arXiv 그림이라 CharXiv와 겹칠 수 있다 |
| VQA-RAD | 의료 313장. 너무 작다 |
| 모든 평가 split | 보드 vision 셋, SAT test, PixMo val/test, 일반 벤치마크 |

## 디스크와 받는 방법

- 남은 디스크는 426GB다(10-31).
- 전부 받으면 Cauldron만 169GB이고, GQA 30GB, Places 25GB 등을 더하면 250GB를 넘는다. **출처당 상한만큼만 쓰므로 parquet shard를 일부만 받는다.** 예상은 80~120GB다.
- 이미지가 URL로만 있는 출처(COCO Karpathy)는 링크가 깨질 수 있다. 받은 수를 기록한다.

## 정할 것

1. 보드 벤치마크 train split 사용 여부(InfographicVQA, CORD, FUNSD, Mind2Web, Hateful Memes).
2. 비상업·미확인 라이선스 출처를 비교 실험에 써도 되는지(텍스트와 같은 규칙인지).
3. 분류 데이터의 비중.
