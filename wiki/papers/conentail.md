# ConEntail

[위키 홈](../README.md) · [구조 비교](../concepts/architecture-map.md)

- 논문: **ConEntail: An Entailment-based Framework for Universal Zero and Few Shot Classification with Supervised Contrastive Pretraining**
- 연도: 2023 · 주제: 통합 분류 데이터 / 대조 학습
- 원문: [alphaXiv](https://www.alphaxiv.org/abs/2210.07587) · [arXiv](https://arxiv.org/abs/2210.07587)
- 조사 기록: 2026-10-13. 아래는 공개 본문을 바탕으로 작성한 요약이다. 우리 모델에서 재현한 결과가 아니다.

## 해결하려는 문제

서로 다른 라벨·과제를 하나의 자연어 기반 메타 과제로 묶어 기존 분류 데이터를 활용한다. 고정 클래스 ID만 외우는 대신 label 의미와 예제 사이 관계를 학습한다.

## 논문에서 제안한 방식

여러 분류 과제를 'query가 [premise가 label에 해당함]을 함의하는가'라는 nested entailment로 통일한다. Few-shot에는 예제 premise, zero-shot에는 NULL premise를 사용한다. 관련 query–premise/label 표현을 대조 학습한다.

## 확인한 실험 근거

초록은 57 annotated classification datasets, 본문은 CrossFit 56 datasets와의 구성 관계를 설명한다. 동일 데이터 기반 discriminator/generator 비교를 제공한다. 근거: 초록·도입·방법.

## 한계와 해석 범위

이번 검색에서 'Universal Discriminator'라는 검색 제목이 이 논문으로 잘못 매칭되었다. 둘을 같은 논문으로 취급하지 않는다. 여기에는 실제 반환된 ConEntail 본문만 기록한다. MMLU reasoning·Qwen hybrid 구조 검증은 아니다.

## 우리 Qwen에 적용한다면

**아래는 적용 제안이며 미구현·미검증이다.** 다양한 분야·감정·문의 의도를 같은 선택 형식으로 학습하는 데이터 설계 참고다. Backbone 변경의 직접적인 성능 근거로 쓰지는 않는다.

## 연결 문서

- [gliclass ](gliclass.md)
- [evaluation ](../designs/evaluation.md)

