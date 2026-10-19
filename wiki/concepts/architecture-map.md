# 구조 변경 지도

[위키 홈](../README.md) · [현재 Qwen](../project/current-qwen.md)

## 무엇을 바꾸는가

| 변경 수준 | 대표 연구 | 달라지는 부분 | 그대로 남는 부분 |
|---|---|---|---|
| 출력 계산·학습 목표 | [LLM-as-Jev](../papers/llm-as-jev.md), [InnerJev](../papers/innerjev.md) | 코드 확률 점수화·증류·regularization | 기존 causal Backbone·LM Head |
| 판단 readout 추가 | [Token·Layer probe](../papers/token-layer-readout.md), [Dual-Head](../papers/dual-head-reasoning-distillation.md) | pooling·MLP·보조 학습 | 기존 Backbone 계산 블록 |
| 후보 내용 기반 scorer | [GLiClass](../papers/gliclass.md), [MetaEncoder](../papers/metaencoder.md) | 본문·라벨 표현 및 shared scorer | 사전학습 Encoder/Decoder 기반 |
| 후보의 정보 흐름 변경 | [Candidate-independent Attention](../papers/candidate-independent-attention.md) | block mask·local position | 기본 Transformer block |
| 판단 아키텍처 재조합 | [Soft Contextualized Encoder](../papers/soft-contextualized-encoder.md), [Poly-encoders](../papers/poly-encoders.md) | query encoder·label set·Attention의 배치와 상호작용 | 기존 신경망 구성 요소 |
| 양방향 적응 | [LLM2Vec](../papers/llm2vec.md), [dQwen3.5](../papers/dqwen35.md) | Attention visibility 및 적응 목표 | 사전학습 가중치; hybrid의 causal 순환 계층은 별도 문제 |

새로운 판단 아키텍처는 확인했다. 하지만 이번 조사에서는 Transformer·DeltaNet의 기본 계산 블록을 처음부터 대체하는 Decision 전용 Backbone을 검증한 근거까지 확보하지 않았다. 문헌 전체에 그런 연구가 없다는 결론은 아니다.

## 후보를 비교하는 네 방식

| 방식 | 문맥 처리 | 후보끼리 직접 비교 | 후보 캐시 | 주의점 |
|---|---|---|---|---|
| 한 입력 + 코드 LM Head | 전체 목록을 함께 읽음 | 가능 | 보통 불가 | 코드 prior·표시 순서·단일 코드 수 제한 |
| 문맥 + 독립 후보 branch | 공통 prefix 공유 가능 | 독립 branch만으로 불가 | prompt 의존 후보는 재사용 제한 | 각 branch 계산·hybrid 상태 분리 |
| Bi-encoder | request와 후보 별도 | 기본 dot product에는 없음 | 가능 | 단일 벡터 병목; request에 목록을 추가하면 성질이 달라짐 |
| 독립 표현 + Set Attention | 문맥 공유, 후보 표현 따로 | 얕은 집합층에서 가능 | 조건에 따라 가능 | 앞단 독립성·위치 없는 집합층·공유 scorer 필요 |

'선택지끼리 비교한다'와 '표시 순서에 의존한다'는 같은 뜻이 아니다. 대칭적인 Set Attention은 관계를 보면서 표시 순서 영향을 제거하도록 설계할 수 있다. 반대로 마지막에 softmax로 경쟁시키는 것만으로 각 후보가 다른 후보의 내용을 이해하는 것은 아니다.

## 속도·메모리 해석

- 생성 제거는 decode를 없애지만 긴 입력의 prefill 비용은 남는다.
- 하나의 batched API call과 하나의 Backbone forward, 요청 전체의 총 계산량을 구분한다.
- 후보 embedding이 캐시된 Bi-encoder만 신규 request forward 하나로 끝날 수 있다. 새 후보에는 인코딩 비용이 있다.
- 공동 인코딩은 후보 수에 따른 입력 길이·Attention 비용이 커진다. 독립 branch도 후보별 suffix 비용과 cache 복제 메모리가 남는다.
- Full vocabulary를 저장하는 것과 전체 vocabulary logits를 계산하는 것은 다르다. 우리 scorer는 필요한 출력 행만 계산한다.

## 관련 문서

- [Attention 차단 상세](attention-isolation.md)
- [다음 설계](../designs/qwen-next.md)

