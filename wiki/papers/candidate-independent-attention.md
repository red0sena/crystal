# Candidate-Independent Block-Causal Attention

[위키 홈](../README.md) · [구조 비교](../concepts/architecture-map.md)

- 논문: **Permutation-Robust Decision Modeling with Candidate-Independent Block-Causal Attention**
- 연도: 2026 · 주제: Attention 구조 / 순서 편향
- 원문: [alphaXiv](https://www.alphaxiv.org/abs/2610.01601) · [arXiv](https://arxiv.org/abs/2610.01601)
- 조사 기록: 2026-10-13. 아래는 공개 본문을 바탕으로 작성한 요약이다. 우리 모델에서 재현한 결과가 아니다.

## 해결하려는 문제

표시 순서가 바뀌면 causal visibility와 위치가 바뀌는 문제를 데이터 증강뿐 아니라 정보 흐름 자체로 줄인다. 임의 길이의 후보 내용을 점수화하며 A/B 코드의 LM Head 행에 의존하지 않는다.

## 논문에서 제안한 방식

공통 지문·질문을 causal하게 읽고, 각 선택지는 공통 prefix와 자기 블록만 읽게 한다. 선택지 사이 Attention을 차단하고 블록마다 논리적 위치 번호를 초기화한다. 각 선택지 끝의 표현에 동일한 점수층을 적용하고 후보 점수에 softmax를 취한다.

## 확인한 실험 근거

Open-Jev의 Gemma3-1B, Qwen3-1.7B, Qwen3-4B 비교. Qwen3-1.7B 테스트 정확도 81.84→84.96%, 순서 변경 시 답변 flip 18.16→1.20%. OOD 정확도는 75.09→74.37%로 내려갔다. 근거: §III–V, 표 I.

## 한계와 해석 범위

이론상 대응하는 후보의 점수는 순서에 독립적이어야 하지만 실제 측정 flip은 0이 아니었다. 순서 안정성, 분포 밖 정확도, 확률 보정은 별개다. 후보 독립 처리만으로는 다른 후보 내용에 의존하는 판단을 충분히 표현하기 어렵다.

## 우리 Qwen에 적용한다면

**아래는 적용 제안이며 미구현·미검증이다.** Qwen3.5는 일반 Qwen3와 달리 DeltaNet과 causal convolution을 포함한다. Attention mask만 변경해서는 후보 독립성을 보장하지 못한다. 공통 문맥 종료 상태를 각 후보에 복제하고 KV·recurrent·conv 상태를 모두 분리하는 구현을 제안한다. 아직 구현·평가하지 않았다.

## 연결 문서

- [attention-isolation ](../concepts/attention-isolation.md)
- [qwen-next ](../designs/qwen-next.md)

