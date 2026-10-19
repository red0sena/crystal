# One Bias Score, Two Factors: 객관식 편향의 두 요인

[위키 홈](../README.md) · [평가 계획](../designs/evaluation.md) · [객관식 편향](multiple-choice-bias.md)

- 논문: **One Bias Score, Two Factors: How Mitigations Affect Option Bias in Multiple-Choice Evaluation** (Hyunjong Ok, Minjae Park, Jaeho Lee)
- 연도: 2026 (arXiv 2610.20993, 10-14 제출) · 주제: 객관식 편향 측정
- 원문: [alphaXiv](https://www.alphaxiv.org/abs/2610.20993) · [arXiv](https://arxiv.org/abs/2610.20993)
- 조사 기록: 2026-10-17 arXiv 초록 확인. 심사 전 프리프린트다.

## 해결하려는 문제

특정 선택지 글자를 고르는 선택 편향과, 순서를 바꾸면 답이 바뀌는 순서 민감도가 하나의 점수에 섞여 있다.

## 논문에서 제안한 방식

통합 선택률 표준편차를 답 뒤집힘 비율(flip rate)과 위치 쏠림(positional skew) 두 요인으로 분해한다.

## 확인한 실험 근거

41개 체크포인트·12개 벤치마크에서, prior 보정은 주로 위치 쏠림을 줄이고 원래 순서와 역순을 합치는 방법은 주로 뒤집힘을 줄인다. 다만 후자는 일부 같은 오답으로 수렴하는 방식으로 줄인다.

## 우리 Qwen에 적용한 결과 (우리 실험)

`qwen_decision/v3/bias_eval.py`가 같은 분리로 잰다. JEVal choice 2,522문항, 4가지 순서에서 위치 쏠림은 g1·d3-lite 모두 작았고(RStd 0.025/0.020), 뒤집힘은 g1 0.315, d3-lite 0.213이었다. 순서 평균은 정확도를 1%p 남짓만 올렸다. 상세는 [작업 기록](../project/v3-log.md).

## 연결 문서

- [객관식 코드·위치 편향](multiple-choice-bias.md) · [평가 계획](../designs/evaluation.md)
