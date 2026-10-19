# Token·Layer 선택 readout

[위키 홈](../README.md) · [구조 비교](../concepts/architecture-map.md)

- 논문: **A BERTology View of LLM Orchestrations: Token- and Layer-Selective Probes for Efficient Single-Pass Classification**
- 연도: 2026 · 주제: hidden state 집계 / 작은 분류층
- 원문: [alphaXiv](https://www.alphaxiv.org/abs/2601.13288) · [arXiv](https://arxiv.org/abs/2601.13288)
- 조사 기록: 2026-10-13. 아래는 공개 본문을 바탕으로 작성한 요약이다. 우리 모델에서 재현한 결과가 아니다.

## 해결하려는 문제

항상 마지막 층·마지막 token만 읽는 대신 과제에 유용한 정보를 여러 위치와 깊이에서 모은다. 같은 prefill 계산을 별도 classifier가 재사용한다.

## 논문에서 제안한 방식

Frozen serving LLM의 token×layer hidden states를 사용한다. 각 층에서 token을 집계하고 다시 층별 표현을 집계하여 분류한다. pooling, 약 100K scoring-attention gate, 최대 35M downcast MHA probe를 비교한다.

## 확인한 실험 근거

Safety·sentiment 과제에서 Llama3.2-3B, GPT-OSS-20B, Qwen3-30B-A3B를 평가. 근거: 도입부·방법·실험.

## 한계와 해석 범위

고정 클래스 classification이고 가변 후보 scorer의 직접 검증이 아니다. 모든 층·토큰을 materialize하면 긴 문맥 메모리가 커질 수 있으며 pooling·streaming 구현 여부를 확인해야 한다.

## 우리 Qwen에 적용한다면

**아래는 적용 제안이며 미구현·미검증이다.** 마지막 hidden state 병목을 조사하기 위한 대조군이다. 처음에는 일부 층과 후보 score 위치만 수집하는 작은 집계층부터 비교하는 것을 제안한다.

## 연결 문서

- [current-qwen ](../project/current-qwen.md)
- [gliclass ](gliclass.md)

