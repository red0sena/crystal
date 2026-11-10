# 목표: Jev Decision Index에서 d3-lite 넘기

> **2026-11-04 목표 확대:** d3-lite를 넘은 기존 기록을 보존한다. 현재 목표와 실행 방침은 [1B 이하 텍스트·비전 1위 프로젝트](champion-v1.md)에 있다. 새 학습은 원본 가중치에서 시작하고, 아래 연구·실험 지식은 이어받는다.

[위키 홈](../README.md) · [작업 기록](v3-log.md) · [전략](../designs/beat-d3-lite.md)

**목표(2026-10-16 설정):** 우리 Qwen3.5-0.8B 모델로 [Jev Decision Index](https://huggingface.co/spaces/multimodalart/jev-decision-index)에서 `vllm-sr/d3-lite`(0.8B)를 넘는다. 아래 수치는 2026-10-15 생성된 보드(Decision Index 0.3.2) 데이터 파일 `data/index.json`, `own.json`, `v03.json`, `methodology.json`(보드 Space의 파일이고 이 저장소에는 없다)을 내려받아 확인한 값이다. 보드는 계속 바뀐다.

## 갱신(2026-11-02): 보드 0.4.1, 공개 점수는 Full에서 빠졌다

Space revision `c130faf`의 `data/index.json`(10-29 생성)과 `methodology.json`을 내려받아 확인했다.

**산식 변경**
- **v0.4(10-19):** Full = 0.60 × 비공개 same-skills + 0.40 × 비공개 new-domains다. 둘 다 조정하지 않은 stock 모델로 공개 척도에 맞춘다.
  - **공개 벤치마크는 계속 돌려 보여 주지만 Full에는 들어가지 않는다.** 순위는 "아무도 학습할 수 없는 시험"으로만 정한다.
  - 1등과 0.9점 안이면 같은 순위다.
- **v0.4.1(10-21):** 비공개 new-domains 과제를 넓히고 바꿨다. 더 어려워져서 new 점수가 전체적으로 낮아졌다.
- 아래 "점수 구조"와 d3-lite 32.33은 0.3.2 기준이라 지금과 다르다.

**1B 이하 Full 순위**(served parameters 기준)

| 모델 | 바탕 | 방식 | Full | 공개 | same | new |
|---|---|---|---:|---:|---:|---:|
| GLiClass v4 edge (0.4B) | DeBERTa-v3-large | encoder | **31.21** | 47.49 | **42.16** | 10.53 |
| OneJev1.6-0.8B | Qwen3.5-0.8B | 생성형 | 29.58 | 48.76 | 40.49 | 9.28 |
| crystal-0.8b | Qwen3.5-0.8B | 생성형 | 29.37 | 42.79 | 39.31 | 10.29 |
| vLLM-SR vela3-fast (0.4B) | UltraBERT-380M | encoder | 29.27 | 50.36 | 40.22 | 8.96 |
| Crivo-1B | ettin-reranker-1b | encoder | 29.25 | 47.77 | 38.75 | 10.75 |
| StartLux-Decision v1.1 | Qwen3.5-0.8B | 생성형 | 28.88 | **53.79** | 41.29 | 6.86 |
| Saina Helm 2 Omni | Qwen3.5-0.8B | 생성형 | 26.54 | 37.44 | 35.21 | 9.59 |
| **vLLM-SR d3-lite** | Qwen3.5-0.8B | 생성형 | **26.16** | 36.93 | 34.81 | 9.33 |
| Amber 0.4B | ModernBERT-large | encoder | 26.00 | 49.74 | 35.73 | 7.86 |
| jiwo 0.8B v2 | Qwen3.5-0.8B | 생성형 | 25.81 | 29.58 | 30.66 | **13.71** |

**encoder와 생성형 비교**
- 1B 이하 56개 중 encoder가 29개(GLi 계열 3개 별도), 생성형이 24개다.
- 상위 5개 평균은 encoder가 Full 27.9, 생성형이 28.1로 같다. 공개 점수만 encoder가 높다(47.0 대 43.9).
- d3-lite를 만든 vLLM-SR도 Vela 3.0을 encoder(UltraBERT)로 냈다.

**우리 모델: crystal-0.8b = wise80 g6+g7e2**
- 보드의 `crystal-0.8b`(PR 278, [HF `Kyunghwan-kim/crystal-0.8b`](https://huggingface.co/Kyunghwan-kim/crystal-0.8b))가 wise80이다.
  - `crystal_config.json`: base + 0.8 × (0.5 × (g6 − base) + 0.5 × (g7 epoch2 − base)), 추가 학습 없음.
- **공식 Full은 29.37로 1B 이하 3위이고, d3-lite(26.16)보다 3.2점 높다.** 공개 42.79, same 39.31, new 10.29다.

**GLiClass v4 edge와의 차이 분해**(보드 `index.json`의 벤치마크별 값)
- Full 차이 1.84의 거의 전부가 same-skills에서 나온다.
  - same: S 41.59 대 38.72. 0.6 × 2.87 = 1.72점이다.
  - new: O 15.64 대 15.35로 거의 같다. 0.4 × 0.29 = 0.12점이다.
- 벤치마크별 `combined`를 영역·gold 가중치로 합치면 보드 S와 0.5~0.7점 안으로 맞는다. 그래서 이 값을 벤치마크별 비공개 same 점수로 봤다(필드 정의는 문서에 없다).
- S 차이 2.87을 벤치마크별로 나누면 다음과 같다(합 2.88).

| GLiClass가 앞선 곳 | S 기여 | 우리 / GLiClass 비공개 | | 우리가 앞선 곳 | S 기여 | 우리 / GLiClass 비공개 |
|---|---:|---|---|---|---:|---|
| PhishNChips | +1.88 | 0.0 / 62.0 | | ACOS | −0.88 | 49.7 / 21.0 |
| WinoGrande | +0.96 | 15.9 / 49.2 | | GSM8K | −0.71 | 31.9 / 2.4 |
| iSarcasmEval | +0.95 | 0.0 / 30.8 | | BFCL | −0.61 | 85.8 / 70.8 |
| HellaSwag | +0.51 | 47.8 / 61.7 | | When2Call | −0.60 | 61.1 / 43.3 |
| NLI4CT | +0.37 | 52.3 / 64.2 | | CRUXEval | −0.35 | 24.4 / 9.8 |
| POP909 | +0.36 | 0.0 / 17.8 | | SATA-Bench | −0.26 | 21.0 / 10.2 |

**세 셋이 전체 차이보다 크다(3.79).** 우리 모델이 한쪽 답으로 쏠린다.
- **PhishNChips:**
  - 공개 정답률 0.52, 비공개 0.50으로 우연 수준이다. d3-lite도 비공개 0.75다.
  - 우리 공개 샘플 40행에서 판정 질문은 80/80 "legitimate", 행동 질문은 40/40 "click"이었다. noul P(phishing)은 모두 0.5 아래였다. **한 번도 phishing이라고 하지 않았다.**
  - 보드 문항은 회사 메일처럼 다듬어진 본문에 `link_url`이 붙은 형식이다.
  - 우리 학습 출처 `phishing`(aura corpus, 사람 라벨)은 텍스트 dev 정답률 1.0이다. 겉모양 단서로 풀리는 쉬운 셋으로 보인다(미확인).
- **iSarcasm:**
  - 비꼼 클래스 F1이 0.238로, 우연(0.223)과 같다.
  - 공개 샘플에서 yes는 116개 중 24개뿐이었다.
- **WinoGrande:** 공개 정답률 0.69, 비공개 skill 15.9다. 텍스트 dev(winogrande-train)는 0.96이라 차이가 크다.
- **GLiClass 설계와의 관계:** 그쪽은 양성 0.75 / 음성 0.25로 가중한 BCE, recall·F1 보상 RL, 쌍별 bias 보정을 쓴다. 모두 "음성 쪽으로 쏠림"을 막는 장치다([GLiClass](../papers/gliclass.md)).
- **우리에게 남은 여지:** PhishNChips만 d3-lite의 비공개 수준(약 50)으로 올려도 S +1.5, 즉 Full +0.9다.

**목표에 대한 뜻**
- d3-lite는 지금 1B 이하 8위(26.16)다. 1등은 GLiClass(31.21)다. **wise80(crystal-0.8b)은 공식 보드에서 이미 d3-lite를 넘었다.**
- 공개 train split 학습은 Full에 직접 이득이 없다. 비공개 same-skills로 옮겨 가는 만큼만 의미가 있다. 그래서 대리 지표는 공개 샘플보다 JEVal same·new와 fresh가 맞다.

## 점수 구조

Full = 20% 공개 벤치마크 + 50% 비공개 same-skills + 30% 비공개 new-domains. 비공개 두 파트는 21개 stock 모델로 공개 척도에 맞춰진다(equating). `own.json`의 계수로 역산하면 다음과 같다. Jev 1.13의 공개 수치로 검산했다.

Full ≈ 0.2·public + 0.506·same + 0.404·new − 3.04

- 원점수 1점의 가치: public 0.2, same-skills 0.506, new-domains 0.404. 공개 벤치마크만 올리는 전략은 효율이 낮다.
- 0.9점 이내는 동순위(tie band)다. 확실히 넘으려면 33.3 이상, 계획 목표는 35 이상이다.
- 공개 37개 벤치마크는 chance-corrected skill(0=무작위, 100=완벽)이며 미응답은 오답이다.
- same-skills: 공개 37개 중 34개에 대응하는 비공개 시험(새 데이터, 미공개). ContractNLI, cfcolor, HLE는 대응 시험이 없다.
- new-domains: "일부는 폭넓은 지식, 일부는 언제 yes/no로 답할지 판단하는 능력"을 본다(보드 설명).

## d3-lite (0.8B)

| 항목 | 값 |
|---|---|
| Full | 32.33 (public 36.93 / same 34.81 / new 25.70) |
| 비공개 영역별 same-skills | knowledge 12.9, language 29.7, retrieval 56.7, tools 43.8, arts 19.9 |
| 지연시간 | median 37ms (RTX PRO 6000) |

공개 구현에서 확인한 구조([model](https://huggingface.co/vllm-sr/d3-lite), `d3_format.py`, `d3_runtime.py`; 보드 측정 revision `a8d25a91`):

- Qwen3.5-0.8B fine-tune. readout은 255개 단일 토큰 코드(A–Z, 두 글자 코드)의 별도 FP32 행렬(`readout.safetensors`, 255×1024)이다. 원본 LM head 행과 코사인 0.99994, 상대 L2 1.1% 차이다. **갱신(2026-10-17):** 가중치 감사 결과 입력 임베딩도 원소의 67%가 바뀌었고(상대 L2 1.05%) 24층 모두 변했다. 임베딩을 동결했다는 앞선 추정은 틀렸다. full fine-tune인지 LoRA 병합인지는 가중치로 구분되지 않는다([작업 기록](v3-log.md)).
- `noncausal_full_attention`: 24층 중 full attention 6층만 padding key mask로 양방향이고, Gated DeltaNet 18층은 causal이다. 원래 perplexity-ai/pplx-decider의 Apache-2.0 hook이다.
- 질문당 forward 1회, 질문 간 상태 공유 없음, T=1. 순열 평균(`permutation_average`)은 구현돼 있지만 꺼져 있다.
- 학습 데이터·loss·교사는 공개되지 않았다. 이전 Decision 2.0 Eos(옵션별 scalar head, Full 17.47)와 달리 LM 코드 readout과 양방향 attention을 쓴다. CoCo-4B도 같은 전환 후 공개 지수가 51.93→53.93으로 올랐다.

**약점(공개 행 기준):** 공개 index에서 noul 형식은 RAGTruth와 PhishNChips 두 개뿐인데 둘 다 약하다. RAGTruth는 환각 클래스 F1 0.297로 우연(0.41~0.52)보다 낮다. 비공개 tools(43.8)는 OneJev 0.8B(57.9)보다 낮고, arts도 낮다. **강점:** new-domains 25.7이다. 다른 0.8B는 16~17 수준이라 2B급에 해당한다.

## 0.8B 경쟁 모델에서 얻은 교훈

| 모델 | Full | public / same / new | 특징 |
|---|---|---|---|
| d3-lite | 32.33 | 36.9 / 34.8 / 25.7 | 위 참조 |
| OneJev1.5-0.8B | 29.01 | **42.7** / 33.0 / 16.8 | 공개 데이터셋 train split 위주. public은 높지만 new-domains가 낮아 Full에서 진다 |
| jiwo-0.8b | 24.87 | 29.6 / 30.7 / 16.1 | 같은 255코드 readout |
| Sieve-0.8B | 23.06 | 32.3 / 26.0 / 16.1 | option fork + scalar head(순서 불변) |
| JPT-0.8B | 18.73 | 19.4 / 17.8 / 22.0 | 가벼운 LoRA. public은 낮지만 new-domains는 상대적으로 보존 |

공개 벤치마크 과적합은 이길 수 없는 길이다. 일반화(new-domains)를 지키는 학습이 핵심이다.

## 규칙(키트와 methodology에서 확인)

- 요청 형식: `{"state", "questions": {key: {"type": "choice"|"noul", "instructions", "criteria"}}}`. choice 옵션은 2~255개이고, 공개 suite 최대는 CLINC150의 151개다. 한 요청에 질문이 여러 개일 수 있다(ACOS 최대 64).
- 채점은 `choice`(argmax)와 noul `p≥0.5` 기준이다. 확률은 BRIGHT/ToolRet의 P(yes) 순위에만 쓰인다.
- 자르기 금지, 옵션 필터링 금지, 벤치마크별 프롬프트 금지, 정답 기반 재시도 금지. 지원하지 않는 요청은 오답이다.
- **오염:** index 시험 문항으로 학습하면 그 문항이 오답 처리된다(감사로 적발된 사례: Lumma, JevK5, reflex 4B 퇴출). 데이터셋의 train/dev split 사용은 공개하면 허용된다.
- 지연시간 median·mean·p80이 각각 1초 이하여야 한다. 점수에는 반영되지 않는다.
- 제출: 공개 0.3 suite 전체 실행(`complete: true`) → 결과를 Hub에 업로드 → `apolinario/decision-index`에 PR. 연구실이 비공개 시험을 우리 서빙 코드로 실행하고, 결정 일치율(BF16 98.5%)을 확인한다. 가중치는 Hub에 공개해야 한다.

## 갱신(2026-11-07): 1B 이하 선두 교체 확인

[보드 미러 데이터](https://huggingface.co/datasets/multimodalart/decision-index-leaderboard)(11-06 09:05Z 생성)를 받아 확인했다. 텍스트 1B 이하 순위는 다음과 같다.

| 순위 | 모델 | Full | same | new | public |
|---:|---|---:|---:|---:|---:|
| 1 | [GLiDE 0.8B](https://huggingface.co/fastino/GLiDE-0.8B) (Fastino, qwen3_5) | 36.36 | 49.66 | 16.40 | 54.32 |
| 2 | GLiDE 0.4B (encoder) | 33.84 | 46.41 | 15.00 | 51.10 |
| 3 | GLiClass v4 edge | 31.21 | 41.59 | 15.64 | 47.49 |
| 4 | PKU-KCL-Meteor-0.8B | 30.94 | 43.50 | 12.11 | 58.77 |
| 6 | crystal-0.8b (우리) | 29.37 | 38.72 | 15.35 | 42.79 |

GLiDE 카드는 학습 방법을 공개하지 않는다.

1B 이하 모델들 사이의 상관은 Full~same 0.99, Full~new 0.64, public~same 0.96, public~new 0.40이다. 표준편차는 same 12.7, new 3.4다. 순위는 사실상 same에서 갈린다.

비전 1B 이하는 QEV-0.8B 44.81, Valen-0.8B 43.69, d3-lite 42.11 순이다. 비전 Full은 공개와 비공개 점수의 평균이고, 1B 이하에서 둘의 상관은 0.96이다.
