# 목표: Jev Decision Index에서 d3-lite 넘기

[위키 홈](../README.md) · [작업 기록](v3-log.md) · [전략](../designs/beat-d3-lite.md)

**목표(2026-10-16 설정):** 우리 Qwen3.5-0.8B 모델로 [Jev Decision Index](https://huggingface.co/spaces/multimodalart/jev-decision-index)에서 `vllm-sr/d3-lite`(0.8B)를 넘는다. 아래 수치는 2026-10-15 생성된 보드(Decision Index 0.3.2) 데이터 파일 `data/index.json`, `own.json`, `v03.json`, `methodology.json`(보드 Space의 파일이고 이 저장소에는 없다)을 내려받아 확인한 값이다. 보드는 계속 바뀐다.

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
