# 장문 학습·평가와 질문 우선 배치(echo)

[위키 홈](../README.md) · [작업 기록](v3-log.md) · [v3 요약](v3-summary.md) · [학습 레시피 실험](recipe-ema-lr.md)

2026-10-23~24. **결론: 3 seed 사전 등록 검정에서 echo 배치의 구조 효과는 입증되지 않았다.** 장문 점수 차이(echo − native)는 seed 41/73/101에서 +1.59/+0.95/−0.71%p, 평균 +0.61%p, seed×문항 계층 bootstrap 95% 구간 [−0.84, +2.07]%p다. 짧은 문항 selection은 평균 −0.11%p로 손실도 없었다.

## 동기

- Qwen3.5-0.8B의 24층 중 18층은 왼쪽에서 오른쪽으로 읽는 Gated DeltaNet이다. d3 프롬프트는 State → Question → Options 순서라서 이 18층은 질문을 모른 채 지문을 고정 크기 상태로 압축한다. 양방향으로 바꾼 full-attention은 6층뿐이다.
- 순환형 모델에서 지문을 두 번 읽히면 회상 과제가 평균 +11점 올랐고([Just read twice](https://arxiv.org/abs/2407.05483v1)), 질문을 문서 앞뒤에 두면 키-값 찾기가 거의 완벽해졌다([Lost in the Middle](https://aclanthology.org/2024.tacl-1.9.pdf)). 둘 다 causal/순환 모델 결과다.
- 이전 구조 실험은 1,536토큰 학습 제한과 짧은 문항 위주 selection(중앙값 299토큰, 1,536토큰 초과 3.5%)에서만 평가했다. 구조가 효과를 낼 수 있는 조건을 보지 못했다는 진단에서 출발했다.

## 데이터

### 장문 테스트셋 (평가 전용, 학습 금지)

`qwen_decision/v3/long_eval.py build`가 고정 revision 원본(`data/raw/longtest/`)에서 만든다. 결과는 `data/longctx/test/`.

| 셋 | 원천 | 라이선스 | 최종 문항 | 프롬프트 토큰 중앙값 / 최대 |
|---|---|---|---:|---|
| BABILong | `RMT-team/babilong` qa1–qa5, 0k–128k 각 100 | 코드 Apache-2.0 | 4,500 (실험은 과제당 20문항×9길이=900) | 168(0k) ~ 125,339(128k) / 129,382 |
| QASPER | v0.3 test, 주석자 합의 yes/no·답변 가능 여부, 가능 여부는 균형 | CC-BY-4.0 | 284 | 4,046~5,313 / 14,790 |
| LongBench v2 | `zai-org/LongBench-v2` 503 사지선다 | Apache-2.0 | 487 (실험은 short만 보조) | short 27,771 / 806,552 |
| DocNLI 장문 | test, 전제 5,000자 이상, 함의 600/600 | BSD-3 | 1,188 | 1,309 / 5,212 |
| JEVal 장문 | JEVal dev 절반의 LongBench v1 6종 + HELMET | 평가 전용 | 684 | 4,474~12,984 / 26,657 |

- 오염 검사(`long_eval.py overlap`): gold2·3·6 학습/dev 파일과 학습 풀 전체에 대해 정규화 지문 일치와 13단어 shingle(8행 넘게 반복되는 템플릿 제외)을 대조했다. 공유 shingle이 1% 이상이거나 지문이 같으면 제외했다. LongBench v2 16문항(GoEmotions 예시를 포함한 ICL 문항), DocNLI 12문항(RAGTruth 학습의 CNN/DailyMail 기사와 사실상 동일)이 빠졌다. 제외 목록은 `data/longctx/audit.json`.
- LongBench v2의 100문항은 모델 최대 길이 262,144토큰을 넘어 자르지 않으면 채점할 수 없다. 보드 규칙대로 오답이다.

### 학습 풀

- `data/longctx/train-long25-32k.jsonl`: v3-gold6만 쓴 32,768행. 1,536토큰 초과 행 8,192개(25%)와 이하 24,576개다. 자르지 않는다. 최장 61,066토큰.
- **이전 풀의 누락:** 이전 4,096 update 비교와 BDH 실험이 쓴 100개 출처 목록은 1,536토큰 이하 표본에서 출발해 짧은 행이 없는 4개 출처(QuALITY, MuSiQue, kv-lookup, API-Bank)를 통째로 뺐다. 이번 풀은 104개 출처 전체에서 장문을 뽑아 이 4개를 각 약 504행 넣었다.

## 설계

- **두 배치:** native(d3)와 echo(Question·Options → State → Question·Options). echo는 새 파라미터가 없다.
- **통제:** 같은 g4b에서 출발, 같은 행·순서·선택지 순열·코드 배정(step마다 해시 대조), 2,048 update × 16, body LR 3e-6, warmup 64 후 cosine 10%, FP32-master AdamW, 정답 CE, varlen packing.
- **3090 제약:** 2,048토큰 넘는 microbatch는 activation checkpointing(8,192 기준에서는 메모리 부족, 기록 `smoke-oom-checkpoint8192.log`). 최장 3행(최대 61k토큰) update 41초·21.2GB, 128k 평가 2문항 31~35초·10.3GB.
- **주 지표(학습 전 `plan.json`에 고정):** 장문 점수 = BABILong 900, JEVal 장문 684, QASPER 284, DocNLI 장문 1,188의 macro 정확도 평균. 짧은 문항 selection(4,147)은 −0.5%p 비열화 조건. LongBench v2 short는 참고.
- **유의성(seed 73·101 전 `significance-plan.json`에 고정):** seed 41·73·101. 성공 = 세 seed 모두 이득 + seed와 문항을 함께 재표본하는 계층 bootstrap(10,000회) 95% 하한 > 0. seed 41을 뺀 확인용 분석도 같이 보고한다.

## 결과

| seed | native 장문 | echo 장문 | 차이 | JEVal 장문 | QASPER | BABILong | DocNLI | selection |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 41 | 66.03 | 67.61 | **+1.59** | +4.29 | +2.77 | +0.89 | −1.60 | −0.74 |
| 73 | 66.34 | 67.28 | **+0.95** | +2.57 | +1.82 | −1.11 | +0.51 | +0.31 |
| 101 | 67.29 | 66.58 | **−0.71** | −0.75 | −0.79 | −0.11 | −1.18 | +0.09 |
| 평균 | 66.55 | 67.16 | **+0.61** | +2.04 | +1.27 | −0.11 | −0.76 | −0.11 |

(단위 %, 차이는 %p)

- 장문 점수: 평균 +0.61%p, 95% 구간 [−0.84, +2.07]. 세 seed 모두 양수가 아니고 구간이 0을 포함해 **성공 기준 실패**. seed 73·101만으로도 +0.12%p [−1.42, +1.73].
- selection: 평균 −0.11%p [−1.09, +0.88]. seed 41의 −0.74%p 손실은 재현되지 않았다.
- seed 41 하나만 보았을 때는 평가 문항 bootstrap 구간이 [+0.31, +2.90]으로 0을 넘었다. 학습 seed 분산을 빼면 과신한다는 점을 그대로 보여 준 사례다.
- 사후 관찰(확증 아님): LongBench TREC만 세 seed 모두 올랐다(+7.4/+7.4/+3.2%p). TREC-e는 +18.8/+8.3/−9.4%p로 뒤집혔고 native 자체가 seed마다 25.0~47.9%였다.

## 효과가 없었던 원인 분석

1. **가설이 겨냥한 병목이 이 모델에서는 이미 거의 풀려 있었다.** 찾기 과제는 native가 이미 천장이다: HELMET 100%, LongBench 검색 98.0~99.6%(최대 26k토큰). 양방향 full-attention 6층이 질문을 보고 지문을 다시 읽을 수 있기 때문으로 보인다. 앞선 근거(JRT, Lost in the Middle)는 causal·순환 모델의 결과이고, 우리 모델은 d3식 양방향 전환으로 그 이득의 상당 부분을 이미 갖고 있다.
2. **native가 지는 곳은 '찾기'가 아니라 '추론·집계'다.** BABILong은 방해 문장이 없는 0k에서도 qa2(사실 2개) 43.3%, qa3(사실 3개) 18.3%다. 길이나 읽는 순서와 무관하게 다단계 추론을 못 한다. 128k까지 늘려도 native 39.0%로 이 바닥 근처다. 다예시 분류(TREC·TREC-e·LSHT)도 36.5~47.9%로 낮다. echo는 추론 능력을 더하지 않는다.
3. **잡음이 신호보다 크다.** 6번의 학습(2배치×3seed)에서 장문 문항의 24.4%가 실행마다 정오가 바뀌었다. 같은 문항에서 echo가 native보다 더 맞힌 횟수의 평균은 문항당 +0.007회(약 +0.2%p)다. 출처별 native seed 표준편차는 TREC-e 11.5%p, TREC 6.8%p, JEVal 장문 전체 2.78%p였고, seed별 echo−native 차이의 표준편차는 장문 점수 기준 약 1.2%p다.
4. **길이가 늘수록 이득이 커진다는 예측이 맞지 않았다.** 장문 셋 전체를 길이로 나누면 echo−native는 2k 미만 −0.35, 2~8k +0.41, 8~32k +1.94, 32k 이상 −1.16%p다. 8~32k 이득은 대부분 JEVal 다예시 분류에서 나왔고, 32k 이상은 BABILong 추론 바닥 구간이라 여지가 없다.
5. **주 지표 구성이 둔감했다.** 네 셋 중 BABILong은 추론 바닥, DocNLI는 중앙값 1,309토큰으로 실제로는 장문이 아니고, JEVal 검색 4종은 천장이었다. 배치 차이에 민감한 문항은 다예시 분류 3종(약 290문항)뿐인데, 그 문항들이 가장 불안정했다.
6. **새 배치 적응 비용은 원인이 아니다.** echo의 학습 loss는 처음 256 update에서 native보다 0.03 높았지만 마지막 256 update에서는 −0.003~+0.007로 따라잡았다.

## 시사점

- 이 모델에서 읽는 순서를 바꾸는 구조는, native의 찾기가 실제로 무너지는 조건(현재 측정 범위 26k토큰 밖)이 있어야 효과를 낼 수 있다. 다음 구조 실험 전에 그 조건을 먼저 측정해야 한다.
- native가 실제로 지는 곳은 다단계 추론과 다예시 분류다. 읽는 순서가 아니라 추론·집계를 겨냥한 수단이 필요하다.
- 평균 효과 1%p 미만을 seed 분산(차이 표준편차 약 1.2%p) 아래에서 확정하려면 seed 15개 안팎이 필요하다. 이 GPU에서 70시간이 넘는다. 작은 효과를 겨냥한 구조 실험은 비용 대비 판정력이 낮다.
- 다예시 분류(TREC류) 가설은 사후 관찰이다. 주장하려면 새 독립 데이터에서 사전 등록 재시험이 필요하다.

## 재현과 파일

```bash
PYTHONPATH=external/decision-index .venv/bin/python -m qwen_decision.v3.long_eval build      # 테스트셋
PYTHONPATH=external/decision-index .venv/bin/python -m qwen_decision.v3.long_eval pool-long  # 25% 장문 풀
PYTHONPATH=external/decision-index .venv/bin/python -m qwen_decision.v3.long_eval audit      # 길이·오염
PYTHONPATH=external/decision-index .venv/bin/python -m qwen_decision.v3.long_context_experiment smoke
PYTHONPATH=external/decision-index .venv/bin/python -m qwen_decision.v3.long_context_experiment run --seeds 41,73,101
PYTHONPATH=external/decision-index .venv/bin/python -m qwen_decision.v3.long_context_experiment significance
```

- 코드: `qwen_decision/v3/long_eval.py`, `qwen_decision/v3/long_context_experiment.py`.
- 실행 기록: `runs/long-context/echo-v1/` — `plan.json`, `significance-plan.json`, `report.json`(seed 41), `significance.json`(최종), 배치·seed별 `train.jsonl`·`eval-*.jsonl`·`result.json`·`model/`.
- 소요: 배치·seed당 학습 약 81분 + 평가 약 51분(RTX 3090 24GB).
