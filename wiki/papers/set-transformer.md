# Set Transformer

[위키 홈](../README.md) · [구조 비교](../concepts/architecture-map.md)

- 논문: **Set Transformer: A Framework for Attention-based Permutation-Invariant Neural Networks**
- 연도: 2019 · 주제: 선택지 집합 / 대칭성
- 원문: [alphaXiv](https://www.alphaxiv.org/abs/1810.00825) · [arXiv](https://arxiv.org/abs/1810.00825)
- 조사 기록: 2026-10-13. 아래는 공개 본문을 바탕으로 작성한 요약이다. 우리 모델에서 재현한 결과가 아니다.

## 해결하려는 문제

선택지는 순서 없는 집합이지만 서로 비교할 필요가 있다는 문제에 구조적 도구를 제공한다. 이 논문의 decoder는 텍스트를 autoregressive하게 생성한다는 뜻이 아니다.

## 논문에서 제안한 방식

집합 원소 사이 관계를 Attention으로 학습한다. SAB·ISAB는 원소 표현을 순열에 대응해 재배열되는 형태로 처리하고, PMA는 집합을 순서에 독립적인 결과로 모은다. 고정된 inducing point 개수 m으로 O(n²) Attention을 O(nm)으로 줄이는 구조도 제안한다.

## 확인한 실험 근거

ICML 2019. 집합 기반 회귀·클러스터링·분류 등의 이론·실험을 제공한다. 근거: §2–3, 실험.

## 한계와 해석 범위

원래 논문은 언어 모델 기반 선택형 문제 풀이 연구가 아니다. 후보별 점수를 얻으려면 equivariant 원소 표현에 공유 scorer를 붙이는 설계가 필요하다. 앞단 후보 표현이 이미 표시 순서에 의존하면 뒤의 집합층만으로 전체 대칭성이 보장되지 않는다.

## 우리 Qwen에 적용한다면

**2026-10-19 g4b 위에서 잔차 Set Transformer를 시험했고 채택하지 않았다.** selection은 79.19%로 기본(78.82%)보다 높아 scoring gate와 공동 1위였다. 동점 규칙으로 확인셋에는 scoring gate만 갔기 때문에 이 구조의 confirmation 결과는 없다([구조 검색](../project/architecture-search.md)). 독립 후보 표현 위에 Set Attention을 두는 아래 제안 그대로는 미구현이다.

처음 제안(기록): 독립적으로 얻은 후보 벡터 사이에 위치 정보 없는 얕은 Set Attention을 적용하고 공통 scorer로 점수화하는 방식을 제안한다. 후보 간 비교를 복원하면서 입력 표시 위치와 의미상 등급 순서를 구분해야 한다.

## 연결 문서

- [soft-contextualized-encoder ](soft-contextualized-encoder.md)
- [attention-isolation ](../concepts/attention-isolation.md)

