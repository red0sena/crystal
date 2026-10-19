# Dataset Cartography: 학습 동역학으로 문항 지도 그리기

[위키 홈](../README.md) · [작업 기록](../project/v3-log.md)

- 논문: **Dataset Cartography: Mapping and Diagnosing Datasets with Training Dynamics** (Swayamdipta 외)
- 연도: 2020 (EMNLP) · 주제: 문항 난이도 지도 / 분포 밖 일반화 / 라벨 오류
- 원문: [ACL Anthology](https://aclanthology.org/2020.emnlp-main.746) · [arXiv](https://arxiv.org/abs/2009.10795)
- 조사 기록: 2026-10-18 웹 검색 요약 기준. 효과 크기는 확인하지 않았다.

## 논문에서 제안한 방식

학습하는 동안 문항별로 정답 확신도의 평균과 에폭 간 변동을 잰다. 이 두 값으로 문항을 쉬움(높은 확신, 낮은 변동), 어려움(낮은 확신, 낮은 변동), 애매함(높은 변동)으로 나눈다.

## 확인한 실험 근거

- 애매한 문항으로 학습하면 분포 밖 일반화가 오르고 분포 안 성능은 거의 그대로였다.
- 어려운 구역에는 라벨 오류가 많았다.

## 우리 Qwen에 대한 의미 (미실행)

g3의 에폭 모델 세 개로 학습 문항 지도를 만들 수 있다. 우리 문제(학습한 데이터셋 안에서는 잘하고 다른 데이터에서는 덜 맞힘)와 직접 맞닿은 후보로 제안했다. 반복 없이 새 문항을 보는 g4b가 먼저 효과를 보여 우선순위가 밀렸다.

## 연결 문서

- [v3 요약](../project/v3-summary.md)
