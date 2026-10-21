# TIES-Merging: 간섭을 줄이는 가중치 병합

[위키 홈](../README.md) · [작업 기록](../project/v3-log.md)

- 논문: **TIES-Merging: Resolving Interference When Merging Models** (Prateek Yadav, Derek Tam, Leshem Choshen, Colin Raffel, Mohit Bansal)
- 연도: 2023 (NeurIPS) · 주제: 다중 과제 병합 / 간섭
- 원문: [arXiv](https://arxiv.org/abs/2306.01708)
- 확인 수준: 초록·검색 요약(본문 완독 아님). 방식의 세 단계는 여러 2차 자료가 일치한다.

## 논문에서 제안한 방식

서로 다른 과제로 미세조정한 모델들의 변화량(task vector)을 합칠 때, 값이 작은 변화와 부호가 엇갈리는 변화가 서로를 깎는다(간섭). 세 단계로 줄인다.
1. **Trim:** 각 변화량에서 크기 상위 k%(보통 20%)만 남긴다.
2. **Elect sign:** 원소마다 남은 변화량의 합으로 부호를 정한다.
3. **Disjoint merge:** 정한 부호와 같은 변화량만 평균한다. 결과에 배율 λ를 곱해 원본에 더한다.

## 우리 Qwen에 적용한 결과 (2026-10-19)

`merge.py --ties <남길 비율>`로 구현했다(가중치가 있으면 부호 투표와 평균에 가중치를 쓴다). 작은 예제로 계산을 확인했다.
- g6 0.67 + g3e2 0.33, 20% 남김: JEVal 0.453 / 공개 42.70 / fresh 0.508. 같은 비율의 단순 평균(0.470 / 43.12 / 0.520)보다 모두 낮다.
- g6 + g3e2 + g4b, 20% 남김: 0.449 / 41.52 / 0.493.
- 해석: TIES는 서로 다른 과제 모델의 간섭을 겨냥한다. 우리 모델들은 같은 원본·비슷한 데이터로 학습해 간섭이 작고, 80%를 잘라 잃는 손해가 더 컸다.
- g7 시기(2026-10-21), g6 + g7e2 1:1, 50% 남김: JEVal 0.461 / 공개 44.62 / fresh 0.508. 같은 비율의 단순 평균(0.472 / 44.49 / 0.533)보다 JEVal·fresh가 낮고, d3-lite 대비 Full 추정 −1.08(단순 평균 +0.13). 50%만 잘라도 손해였다.

## 연결 문서

- [DARE](dare.md) · [Model Stock](model-stock.md) · [LiNeS](lines.md) · [체크포인트 평균·Soup](checkpoint-averaging.md)
