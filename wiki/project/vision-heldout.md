# 다른 벤치마크 vision 시험셋 (2026-10-31)

[위키 홈](../README.md) · [Vision 기준점](vision-baseline.md) · [이미지 데이터 출처](../designs/vision-data-sources.md)

**목적:** 이미지 학습으로 고른 모델이 학습한 벤치마크에만 맞춰진 것이 아닌지 본다. 사용자 요청은 "테스트 데이터셋은 아예 다른 벤치마크에서 빼서 평가"였다.
- 학습 출처(`vision_sources`의 train)에도 없고 보드 vision 셋에도 없는 벤치마크 17개에서 뽑았다. 벤치마크당 최대 300문항이고, 모두 4,772문항이다.
- 보드 공개 vision 셋은 마지막 확인으로 따로 둔다.
- 코드는 `qwen_decision/v3/vision_heldout.py`(build, run, score), 데이터는 `data/vision-heldout/`(rows.jsonl, 벤치마크별 items, images.json), 결과는 `runs/vision-heldout/`다.

## 만든 방법

- 벤치마크마다 seed를 고정해 세부 과제별로 고르게 뽑았다(seed 20261031).
- 선택지는 같은 seed로 섞었다. 예외: Yes/No, 숫자(작은 것부터), 이미지에 A~H가 그려진 상자 선택지.
- 행 형식은 키트(`decision_index.vision.rows.to_row`)와 같아서 키트 runner와 우리 `VisionEngine`이 그대로 돌린다.
- 점수는 벤치마크마다 chance를 뺀 skill이고, 지수는 17개 skill(0~1로 자름)의 단순 평균 × 100이다.
- `images.json`에 모든 이미지의 sha256과 64비트 difference hash를 남겼다. 이미지 학습 셋을 만들 때 이 이미지들(그리고 보드 이미지)과 겹치는 것을 뺀다.

## 섞이지 않게 뺀 것

- **MMStar:** 원래 다른 벤치마크에서 문항을 모은 셋이다(SEED 424, MMBench 366, MathVista 345, AI2D 196, MMMU 100, ScienceQA 69). AI2D와 ScienceQA는 학습 출처이고 MMMU는 보드 MMMU-Pro의 원본이라, SEED와 MMBench 문항만 남겼다(790개에서 287개 사용).
- **MathVista testmini:** 객관식 중 원래 출처가 학습 벤치마크인 것(FigureQA, AI2D, GeoQA+, Geometry3K, IconQA, TQA, ScienceQA 등)과 UniGeo(Geometry3K·GeoQA 문제를 다시 씀)를 뺐다. Super-CLEVR, FunctionQA, IQTest, GEOS, PaperQA 등 162문항이 남았다.
- **3DSRBench:** 이미지를 좌우로 뒤집어야 하는 "-flip" 문항은 뒤집힌 이미지가 없어서 뺐다.
- **Facebook_harmeme_dataset**이라는 HF 저장소는 이름과 달리 Hateful Memes(보드 셋의 원본)라서 쓰지 않았다. 밈은 Memotion으로 쟀고, 그래서 Memotion은 학습에서 뺐다.
- **WildReceipt**는 영수증 KIE 시험으로 쓰고 학습에서 뺐다.

## 벤치마크

| 분야 | 벤치마크 | 문항 | 선택지 | chance | 세부 과제 |
|---|---|---:|---|---:|---|
| 공간 | 3DSRBench | 300 | 2~4 | 0.458 | 높이, 위치, 방향 등 12종(COCO 사진) |
| 공간 | EmbSpatial-Bench | 300 | 4 | 0.250 | 가깝다/멀다/왼쪽/오른쪽/위/아래(실내 장면) |
| 공간 | What'sUp (Controlled A, B) | 300 | 4 | 0.250 | 물건 두 개의 위치 관계를 캡션으로 고르기 |
| 개수 | CountBenchQA | 300 | 4 | 0.250 | 2~10개 세기. 오답은 가까운 숫자 |
| 세밀한 지각 | MMVP | 300 | 2 | 0.500 | CLIP이 헷갈리는 사진 쌍 |
| 세밀한 지각 | NaturalBench | 298 | 2 | 0.500 | 같은 질문, 답이 다른 사진 두 장 |
| 세밀한 지각 | V* Bench | 191 | 2~4 | 0.360 | 고해상도 사진의 작은 물체 속성, 위치 |
| 실세계 고해상도 | MME-RealWorld (7/70 shard) | 299 | 5 | 0.200 | 자율주행, 도표·표, 복잡한 글자, 원격 탐사 |
| 차트 | ChartQAPro | 235 | 2~4 | 0.381 | 객관식, 참/거짓 확인 |
| 글자 많은 이미지 | SEED-Bench-2-Plus | 300 | 4 | 0.250 | 차트, 지도, 웹 |
| 웹 | VisualWebBench | 300 | 7~8 | 0.125 | 요소 찾기, 클릭할 곳, 클릭 뒤 페이지 |
| 이미지-문장 | SugarCrepe | 300 | 2 | 0.500 | 단어를 바꾸거나 더하거나 순서를 바꾼 오답 캡션 7종 |
| 영수증 | WildReceipt test | 300 | 2~4 | 0.394 | 보드 KIE 형식(필드 값, 필드 유무 Yes/No) |
| 밈 | Memotion test | 300 | 2 | 0.500 | 공격적인가(slight는 뺌) |
| 수학 | MathVista testmini (거른 것) | 162 | 2~8 | 0.370 | Super-CLEVR, FunctionQA, IQTest 등 |
| 일반 | MMStar (SEED·MMBench 문항) | 287 | 4 | 0.250 | 지각, 추론 6종 |
| 일반 | POPE | 300 | 2 | 0.500 | 물체가 있는가(환각) |

## 결과 (10-31, 이미지 학습 전)

[Vision 기준점](vision-baseline.md)과 같은 엔진과 같은 두 설정이다. 원래 Qwen은 chat(보드 기준 경로, causal), wise80은 decide(v3 형식, 양방향)다. 네 실행 모두 4,772행을 답했고 오류는 0이다. 실행 하나에 약 8.6분 걸렸다.

| 벤치마크 | 원래 Qwen | wise80 | 차이 |
|---|---:|---:|---:|
| **지수 (17개 평균)** | **40.40** | **50.91** | **+10.51** |
| What'sUp | 0.742 | 0.924 | +0.18 |
| SugarCrepe | 0.587 | 0.873 | +0.29 |
| VisualWebBench | 0.577 | 0.771 | +0.19 |
| CountBenchQA | 0.618 | 0.760 | +0.14 |
| WildReceipt | 0.626 | 0.720 | +0.09 |
| POPE | 0.773 | 0.733 | −0.04 |
| SEED-Bench-2-Plus | 0.484 | 0.556 | +0.07 |
| EmbSpatial | 0.400 | 0.547 | +0.15 |
| V* | 0.493 | 0.493 | 0 |
| NaturalBench | 0.396 | 0.483 | +0.09 |
| MMStar | 0.359 | 0.470 | +0.11 |
| MME-RealWorld | 0.423 | 0.436 | +0.01 |
| MMVP | 0.240 | 0.413 | +0.17 |
| MathVista | 0.020 | 0.255 | +0.24 |
| ChartQAPro | 0.120 | 0.155 | +0.03 |
| 3DSRBench | 0.009 | 0.065 | +0.06 |
| Memotion | −0.007 | −0.020 | 0 |

- **wise80 − 원래 Qwen = +10.51, 95% 구간 [+8.01, +12.55].** 벤치마크 안에서 행을 다시 뽑는 bootstrap 2,000번이다. 보드 공개 셋의 차이(+16.4)보다 작지만 같은 방향이다. 텍스트 판정 학습의 이득이 보드 셋에만 맞춰진 것이 아니라는 뜻이다.
- **원래 Qwen의 글자 편향이 여기서도 보인다.** 3DSRBench에서 300문항 중 255개를 A로 골랐고, SugarCrepe에서도 220개를 A로 골랐다. Memotion에서는 287개를 Yes로 답했다. wise80은 고르게 나뉜다(3DSRBench A 134, B 138).
- **둘 다 거의 우연 수준인 곳:** Memotion(밈 공격성, 둘 다 정답률 약 0.49), 3DSRBench(3D 위치, 0.01~0.07), ChartQAPro(0.12~0.16). Memotion은 라벨 잡음이 큰 셋으로 알려져 있어, 이 점수만으로 밈 능력을 판단하기는 어렵다.
- **wise80이 조금 낮은 곳:** POPE(물체가 있는가) −0.04. wise80이 No를 더 자주 골랐다(178 대 156).
- **같은 곳:** V*(고해상도 작은 물체), MME-RealWorld. 이미지를 읽는 능력 자체가 필요한 곳이라, 텍스트 학습으로는 오르지 않는 것으로 보인다.
- 이미지 학습 후보를 고를 때는 이 지수를 쓰고, 보드 공개 셋은 마지막에 한 번만 쓴다.
