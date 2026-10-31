# 선택지 사이 Attention 차단

[위키 홈](../README.md) · 근거: [Candidate-independent Attention](../papers/candidate-independent-attention.md)

## 기본 mask

입력을 공통 prefix P(지문+질문)와 후보 블록 B1…BK로 나눈다. 아래 표의 허용 블록 내부도 causal하게, 즉 자기 위치와 앞 token만 읽는다.

| Query 쪽 / Key 쪽 | P | B1 | B2 | B3 |
|---|---|---|---|---|
| P | causal | 차단 | 차단 | 차단 |
| B1 | 허용 | causal | 차단 | 차단 |
| B2 | 허용 | 차단 | causal | 차단 |
| B3 | 허용 | 차단 | 차단 | causal |

Attention(Q,K,V)=softmax(QKᵀ/√d + M)V.

허용 위치 M=0, 차단 위치 M=−∞로 설정하면 차단 key의 softmax weight가 0이 된다. 실구현에서는 kernel별 boolean mask의 의미, padding, fully masked row, 지원하는 block mask를 확인한다. 단순 padding mask는 후보 블록 분리를 대신하지 못한다. 긴 문맥에서 dense L×L mask를 만들면 메모리 비용도 발생한다.

후보마다 동일한 local position sequence를 부여하고, 표시에만 쓰는 후보 index를 표현에 주입하지 않아야 의도한 대칭성에 가까워진다. ID는 결과를 원래 선택지와 연결하는 외부 식별자로 관리한다. 'A와 B 모두' 같은 문구는 ID를 무작정 지우면 의미가 바뀌므로 처리 규칙이 필요하다.

## readout도 바꿔야 한다

각 후보의 final token 또는 score token h_i에서 공유 scorer g로 z_i=g(h_i)를 얻고 전체 후보에 softmax를 적용한다. 입력 끝의 하나의 토큰만 읽어 A/B/C를 예측하는 기존 scorer에 mask만 붙이는 것은 이 구조와 다르다.

## Qwen3.5 hybrid의 추가 조건

**아래는 구현 제안이며 이 wiki에 구현 코드가 포함되어 있지 않다.** 10-25 순서 측정에서 학습 없이 4순서 평균한 이득이 짧은 문항 +0.7%p 안팎이라, 구현하지 않기로 했다([순서 측정](../project/order-probe.md)).

Qwen3.5는 Attention 외에 Gated DeltaNet recurrence와 causal convolution으로 이전 token 정보를 전달한다. Attention 경로만 차단하면 앞 후보 정보가 다른 경로로 넘어갈 수 있다.

```text
공통 지문·질문 prefill → KV + recurrent + conv 상태
                         ├─ 독립 복제 → 후보 1 → scorer
                         ├─ 독립 복제 → 후보 2 → scorer
                         └─ 독립 복제 → 후보 3 → scorer
```

안전한 첫 대조군은 각 후보를 같은 prefix 종료 상태에서 독립적으로 처리하는 것이다. 또는 packed sequence에서 모든 hybrid 계층의 상태 reset·분기 semantics를 직접 구현해야 한다. prefix가 candidate 내용을 이미 포함했다면 그 경로를 통해 후보가 섞일 수 있으므로 prefix 정의도 확인한다.

v2(지금은 쓰지 않음)는 **여러 질문**의 suffix를 독립적으로 처리하는 상태 공유 코드였다. v3와 지금 레시피는 d3 계약대로 질문당 forward 1회이고 상태를 공유하지 않는다. 그것은 한 질문 안의 **선택지별 branch head**를 학습한 것과 다르다. [현재 구현](../project/current-qwen.md)

## 한계

독립 후보는 다른 후보의 내용에 의존하는 '가장 저렴한 것', '위 조건 중 어느 것도 아님' 등을 충분히 판단하지 못할 수 있다. [Set Transformer](../papers/set-transformer.md)식 집합 비교를 추가하는 확장과 순서 대칭성·정확도를 함께 비교한다. g4b 위 잔차 Set Transformer는 10-19에 selection +0.37%p였지만 채택하지 않았다.

