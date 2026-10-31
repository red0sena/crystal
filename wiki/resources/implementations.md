# 공개 구현 참고

[위키 홈](../README.md)

아래는 논문과 별도의 model card·공개 코드 자료다. 대화에서 확인한 구조를 요약하며 학습 데이터·비공개 recipe를 추정해 사실로 적지 않는다.

| 구현 | 확인한 핵심 | 원본 |
|---|---|---|
| LiquidAI d1-3B | causal LFM2.5-VL, 기존 LM Head 후보 token readout, trunk/branch 상태 공유 | [model](https://huggingface.co/LiquidAI/d1-3B), [확인 revision](https://huggingface.co/LiquidAI/d1-3B/commit/da1fe36a861f24690f27f622dca1d8688503d113), [공식 설명](https://www.liquid.ai/blog/d1-open) |
| LiquidAI d1-omni-600M | bidirectional Encoder + 2층 Transformer DecisionHead, 후보 marker 표현과 공유 scalar MLP | [model](https://huggingface.co/LiquidAI/d1-omni-600M), [확인 revision](https://huggingface.co/LiquidAI/d1-omni-600M/commit/414f8d6438174f5b2133a9c21a478fc42625e308) |
| Jeff-Qwen3.5-0.8B legacy | 기존 선택 코드 logits, supervised 선택 loss와 calibration; legacy와 최신 adapter 구현 구분 | [legacy card](https://huggingface.co/jeff-legacy/Jeff-Qwen3.5-0.8B), [project](https://github.com/firelex/jeff) |
| Mapika Decider | 기존 LM Head 후보 readout, 버전별 학습·consistency/replay 차이 | [model card](https://huggingface.co/Mapika/decider-2b), [code](https://github.com/Mapika/decider) |
| GLiClass | label contents와 본문 표현으로 가변 후보 점수화 | [code](https://github.com/Knowledgator/GLiClass), [문서](https://docs.knowledgator.com/docs/frameworks/gliclass/intro/), [논문 카드](../papers/gliclass.md) |
| MetaEncoder | causal 마지막 token Bi-encoder·contrastive 학습 | [model](https://huggingface.co/facebook/meta-encoder), [eval code](https://github.com/facebook/meta-encoder-eval), [논문 카드](../papers/metaencoder.md) |
| InnerJev | reasoning 종료 후보 분포를 첫 답 token readout으로 증류 | [code](https://github.com/amazingljy1206/InnerJev), [JEVal](https://huggingface.co/datasets/carlosxiang/JEVal), [논문 카드](../papers/innerjev.md) |

## Decision Index 출전 모델 (2026-10-16 확인)

| 구현 | 확인한 핵심 | 원본 |
|---|---|---|
| vLLM-SR d3-lite (0.8B) | Qwen3.5-0.8B full FT. tied LM head를 255코드로 제한한 readout(별도 readout 행이 원본 embedding과 cosine 0.99995). full-attention 6층 양방향, DeltaNet causal. 질문당 forward 1회, T=1, 순열 평균 꺼짐. 학습 recipe 미공개 | [model](https://huggingface.co/vllm-sr/d3-lite), [보드 실행 결과](https://huggingface.co/datasets/vllm-sr/d3-lite-decision-index), [design.md](https://github.com/vllm-project/semantic-router/blob/main/src/model-runtime/docs/design.md) |
| vLLM-SR Decision 2.0 Eos (0.8B) | 옵션별 마지막 토큰 + 질의 토큰의 bilinear+MLP scalar head, causal. 학습 코드 일부 공개(valid-K CE + 선택적 Brier + replay KL). 3-seed soup | [model](https://huggingface.co/vllm-sr/Decision-2.0-Eos-0.8B) |
| OneJev1.5-0.8B | LM head 슬롯 readout. 약 32만 문항(공개 데이터 train split 다수), soft-target CE + Brier, lr 5e-6. 옵션을 255개까지 패딩하는 증강 | [code](https://github.com/OmniJev/OneJev) |
| Sieve-0.8B | 블록 1–16만 사용. state를 한 번 인코딩해 옵션별로 복제(option fork), scalar head로 순서 불변. 학습 문항의 17%가 index 데이터셋의 train/dev | [model](https://huggingface.co/sthanika-ai/Sieve-0.8B) |
| autotrust JEV | Jev 1.13 분포로 증류(LoRA r16 + 24슬롯 head). 사용한 코퍼스가 `SargeDev/jev-distill-corpus-v3`(Apache-2.0, Jev soft label 49.8만 행) | [model](https://huggingface.co/autotrust/JEV), [corpus](https://huggingface.co/datasets/SargeDev/jev-distill-corpus-v3) |

d3-lite 구조는 우리 엔진으로 재현해 확인했다. 결정 일치율은 JEVal 360문항에서 처음 98.6%였고, d3의 `readout.safetensors`를 읽게 한 뒤 100%가 됐다. 공개 suite 샘플에서는 99.36%다([작업 기록](../project/v3-log.md)). d3-lite 가중치는 지금 로컬에 없다.

## d1-3B와 d1-omni를 구분

둘 다 확률을 반환하지만 같은 head 구조가 아니다. d1-3B의 LM Head readout과 d1-omni의 별도 DecisionHead를 하나의 recipe로 설명하지 않는다. 공개 inference 구조가 확인됐다는 것과 전체 학습 recipe가 공개됐다는 것도 다르다.

## Benchmark 링크

- [Jev Decision Index](https://huggingface.co/spaces/multimodalart/jev-decision-index), [공개 코드](https://github.com/apolinario/decision-index).
- JevBench의 공개 사례/공식 결과와 Jev Decision Index의 Public/Full, [JEVal 논문](../papers/innerjev.md)의 점수를 서로 교환해 사용하지 않는다.

2026-10-16부터 목표 대상 d3-lite의 수치만 날짜와 함께 [목표 문서](../project/jev-index.md)에 기록한다. 그 밖의 변동하는 leaderboard 수치는 복사하지 않는다. 점수 비교 시 suite 버전·날짜·공개/비공개 범위·학습 overlap·latency 환경을 확인한다.

