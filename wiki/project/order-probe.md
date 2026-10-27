# 선택지 순서 측정: 선택지 격리로 얻을 수 있는 몫의 추정

[위키 홈](../README.md) · 관련: [선택지 사이 Attention 차단](../concepts/attention-isolation.md),
[Candidate-independent Attention](../papers/candidate-independent-attention.md), [장문 echo 실험](long-context-echo.md)

**완료(2026-10-25). 학습 없는 측정이다.** 선택지끼리 서로 보지 못하게 격리하면 모델은 순서 불변이 된다.
그 몫을 학습 없이 가늠하려고, 이미 학습한 native 3개(`runs/long-context/echo-v1/native-seed{41,73,101}/model`)로
choice 문항을 선택지 순서 4가지(원래, 역순, k/4 회전, 그 역순)로 각각 채점했다. 확률은 원래 선택지 순서로 되돌려 평균했다.
순서가 의미를 갖는 문항, 위치를 참조하는 문항, noul은 순서를 바꾸지 않았다.

실행: `PYTHONPATH=external/decision-index .venv/bin/python -m qwen_decision.v3.order_probe`
→ `runs/long-context/order-probe-v1/{result.json, seed*-*.jsonl}`.

## 결과 (3 seed 평균)

| 평가셋 | 순서를 바꾼 문항 | 원래 순서 | 4순서 평균 | 차이 | 답이 바뀐 비율 |
|---|---:|---:|---:|---:|---:|
| selection | 3,237/4,147 | 78.02% | 78.70% | **+0.68%p** (+0.59/+0.90/+0.55) | 10.2% |
| JEVal 장문 | 684/684 | 74.47% | 73.21% | **−1.25%p** (−2.40/−2.08/+0.72) | 22.0% |
| BABILong 0–16k | 600/600 | 55.06% | 55.06% | 0.00%p (+0.17/−0.67/+0.50) | 18.9% |

순서를 바꾼 출처만의 출처 층화 bootstrap(문항별 3 seed 평균, 2,000회):
selection +0.72%p [+0.26, +1.18], JEVal 장문 −1.25%p [−2.26, −0.22], BABILong 0.00%p [−1.00, +1.00].

- **selection:** 선택지 2개 문항은 차이가 없었고(76.1→76.0%, 답 바뀜 6.0%), 5–10개 문항이 +1.2%p(15.6%),
  11개 이상이 +1.0%p(11.7%)였다. 이득은 fewnerd·commonsenseqa·chid·hellaswag·gsm8k, 손실은 headqa·medmcqa·lexglue 쪽이다.
- **JEVal 장문:** 손실은 라벨 50개인 LongBench TREC(−6.0%p, 답 바뀜 65.6%)과 TREC-e(−3.5%p, 61.8%)에서 나왔다.
  원래 라벨 순서가 정보로 쓰이는 것으로 보인다. 검색 과제(retrieval, HELMET)는 천장이라 답이 거의 바뀌지 않았다.
- **BABILong:** 다섯 문항 중 하나꼴로 답이 바뀌지만 평균하면 정확도는 그대로다.

## 해석

- 학습은 이미 매 step 선택지 순서를 무작위로 섞는다(`robust_adaptation.random_view`). 그래도 짧은 문항의 10%,
  장문의 20% 안팎이 순서에 따라 답을 바꾼다. 데이터 증강만으로는 순서 의존성이 없어지지 않았다.
- 순서 불변이 줄 수 있는 몫은 짧은 문항에서 +0.7%p 정도, 장문에서는 오히려 손해다. +0.7%p에는 4번 forward한
  앙상블 효과도 섞여 있다. 한 번 forward하는 격리 모델이 받을 몫은 이보다 작을 수 있다.
- 이 측정은 순서 의존성만 본다. "다른 선택지를 보는 것 자체가 노이즈"라는 가설은 직접 검증하지 않는다.
  격리는 학습으로 다른 표현을 배울 수 있으므로 이 값을 격리의 상한이나 하한으로 단정하지 않는다.

## 이전 시도

- 이 프로젝트에서 선택지 격리 구조는 구현한 적이 없다. 설계 메모만 있다([attention-isolation](../concepts/attention-isolation.md)).
- 순열 두 view의 JSD 일관성 학습(2026-10-19)은 단일 forward에서 confirmation −0.20%p로 확인된 개선이 없었다([robust-combinations](robust-combinations.md)).
- 외부: Candidate-Independent Block-Causal Attention(2026)은 Qwen3-1.7B에서 정확도 81.84→84.96%,
  순서 변경 flip 18.16→1.20%였으나 OOD는 75.09→74.37%로 내려갔다. 일반 Qwen3라 DeltaNet 분기 문제가 없다.
- Decision Index의 Sieve-0.8B는 option fork + scalar head(순서 불변)로 Full 23.06이었다. 같은 255코드 readout의
  jiwo-0.8b 24.87, d3-lite 32.33보다 낮다. 학습 데이터가 달라 구조만의 비교는 아니다.
