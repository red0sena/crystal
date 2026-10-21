# DARE: 변화량을 무작위로 버리고 다시 키우기

[위키 홈](../README.md) · [작업 기록](../project/v3-log.md)

- 논문: **Language Models are Super Mario: Absorbing Abilities from Homologous Models as a Free Lunch** (Le Yu, Bowen Yu, Haiyang Yu, Fei Huang, Yongbin Li)
- 연도: 2024 (ICML) · 주제: 미세조정 변화량 희소화 / 병합
- 원문: [arXiv](https://arxiv.org/abs/2311.03099)
- 확인 수준: 초록·검색 요약(본문 완독 아님).

## 논문에서 제안한 방식

DARE(Drop And REscale)는 미세조정 변화량의 원소를 확률 p로 버리고, 남은 원소를 1/(1−p)배 키워 기댓값을 유지한다. 저자들은 SFT 변화량이 작을 때 p를 0.9 이상으로 해도 성능이 유지된다고 보고했다. 그 뒤 여러 모델을 평균하거나 TIES와 함께 쓴다(DARE-TIES).

## 우리 Qwen에 적용한 결과 (2026-10-19)

`merge.py --dare <p> [--ties 1]`로 구현했다(시드 고정, `--seed`).
- DARE p=0.7 + TIES(자르기 없음), g6 + g3e2 + g4b: JEVal 0.372 / 공개 34.90 / fresh 0.429. 이번에 시험한 방법 중 가장 낮고, 원본이 맞히던 문항을 가장 많이 잊었다(388개, 단순 평균 271개).
- 해석: 우리는 전체 파라미터를 학습해 변화량이 넓게 퍼져 있다. 70%를 버리고 3.3배로 키우면 잡음이 커진다. 같은 원본·비슷한 데이터 모델끼리는 단순 평균이 낫다.
- g7 시기(2026-10-21), DARE p=0.5만(TIES 없이), g6 + g7e2 1:1: JEVal 0.473 / 공개 44.65 / fresh 0.540, Full 추정 +0.20. 단순 평균(0.472 / 44.49 / 0.533, +0.13)과 잡음 범위 안에서 같다. 버린 만큼 키워 기댓값이 같으니 이득도 손해도 거의 없다.

## 연결 문서

- [TIES-Merging](ties-merging.md) · [Model Stock](model-stock.md) · [체크포인트 평균·Soup](checkpoint-averaging.md)
