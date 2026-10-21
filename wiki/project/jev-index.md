# 목표: Jev Decision Index에서 d3-lite 넘기

[위키 홈](../README.md) · [작업 기록](v3-log.md) · [전략](../designs/beat-d3-lite.md)

**목표(2026-10-16 설정):** 우리 Qwen3.5-0.8B 모델로 [Jev Decision Index](https://huggingface.co/spaces/multimodalart/jev-decision-index)에서 `vllm-sr/d3-lite`(0.8B)를 넘는다. 아래 수치는 2026-10-15 생성된 보드(Decision Index 0.3.2) 데이터 파일 `data/index.json`, `own.json`, `v03.json`, `methodology.json`(보드 Space의 파일이고 이 저장소에는 없다)을 내려받아 확인한 값이다. 보드는 계속 바뀐다.

## 점수 구조

**Decision Index 0.4 (키트 문서 2026-10-20, upstream `25c6ec7`):** Full = 60% 비공개 same-skills + 40% 비공개 new-domains다. new-domains는 과제 세트가 바뀌었다.
- 공개 점수는 Full에 들어가지 않는다. 다만 공개 0.3 suite 전체 실행은 계속 필수다. 제출 검증에 쓰이고 보드에도 표시되며, 공개 점수가 비공개보다 지나치게 높으면 따로 들여다본다.
- 학습 데이터가 비공개 시험과 겹친 흔적이 있으면 그 시험은 해당 모델 점수에서 빠진다. 동률 기준(0.9점)은 그대로다.
- 0.4에서 비공개 두 파트를 공개 척도에 맞추는 계수는 아직 모른다. 그래서 우리는 0.6·Δsame + 0.4·Δnew(JEVal skill 점수)로 어림한다.
- d3-lite의 0.4 Full은 아직 확인하지 못했다.

아래는 0.3 기준이다. Full = 20% 공개 벤치마크 + 50% 비공개 same-skills + 30% 비공개 new-domains. 비공개 두 파트는 21개 stock 모델로 공개 척도에 맞춰진다(equating). `own.json`의 계수로 역산하면 다음과 같다. Jev 1.13의 공개 수치로 검산했다.

Full ≈ 0.2·public + 0.506·same + 0.404·new − 3.04

- 원점수 1점의 가치: public 0.2, same-skills 0.506, new-domains 0.404. 공개 벤치마크만 올리는 전략은 효율이 낮다.
- 0.9점 이내는 동순위(tie band)다. 확실히 넘으려면 33.3 이상, 계획 목표는 35 이상이다.
- 공개 37개 벤치마크는 chance-corrected skill(0=무작위, 100=완벽)이며 미응답은 오답이다.
- 공개 지수는 다섯 영역의 고정 가중 평균이다(2026-10-19 키트 채점 결과 `note`에서 확인).
  - 가중치: knowledge 25.8%, language 25.8%, retrieval 20.0%, tools 18.3%, arts 10%.
  - 영역 안에서는 gold ★ 벤치마크가 1.2, 나머지가 1.0이다.
  - 그래서 벤치마크 하나의 비중은 영역마다 다르다. 예술 6개 중 하나는 약 1.7%, 언어 9개 중 하나는 약 2.9%다.
- RouterBench, SGD/SGD-X, MMLU, ARC-Easy, ARC-Challenge, SimpleBench는 보드에 보이지만 지수에 들어가지 않는다(shown, not counted).
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
- 제출(키트 README, 0.4에서도 같음):
  1. 공개 0.3 suite 전체를 실행하고 결과 폴더를 Hub 데이터셋에 올린다(`pipeline --upload <you>/<repo>`). `runs/<name>/scores.json`에 `"complete": true`가 있어야 한다.
  2. `apolinario/decision-index`에 PR을 보내 `submissions/README.md`(없으면 새로 만듦)에 한 줄을 추가한다. 모델명, Hub 모델 저장소, 결과 데이터셋 링크, 엔진·커밋, 하드웨어, 추론 설정을 적는다. d3-lite의 `RUN_NOTES.md`(`runs/board-d3-lite/`)가 좋은 예다.
  3. 관리자가 비공개 두 파트를 우리 방식 그대로 돌린다. 그러려면 가중치(공개 또는 관리자에게 접근 허용), 실행 코드, 설정이 필요하다. 공개 답안과의 일치도 비교한다. 지연 시간은 RTX PRO 6000 한 장에서 잰다.
- 우리 상태(2026-10-21):
  - 우리 GitHub과 HF 저장소는 모두 비공개다. 제출 후보 `crystal-0.8b`는 독립 실행 코드와 함께 비공개 Hub 저장소에 있다(`release/crystal-0.8b/`).
  - 로컬에서 다시 만든 suite는 연구실 원본과 `selected-rows` 해시가 다르다. RouterBench 6개와 SGD 10개 요청이 다르고, 둘 다 지수 밖이다. 그래서 로컬 실행에는 `--no-verify`가 필요하고, 실제 제출에는 원본 파일이 필요하다.
  - 공개 전체 실행은 A6000에서 약 2.3시간이다(14만 요청, 질문 32만 개). 지연 시간은 중앙값 32ms, 80% 지점 193ms다.
  - SCOTUS SCDB 라벨(CC-BY-NC-3.0 upstream)이 학습에 들어 있어, 가중치를 공개하기 전에 판단이 필요하다.
