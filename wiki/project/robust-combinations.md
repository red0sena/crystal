# 원본 보존 적응 학습과 추론 조합 (2026-10-19)

[위키 홈](../README.md) · [직전 readout 실험](readout-mlp.md)

**별도 성과:** 작업 중 새로 올라온 g6 완료본의 고정 recipe를 재구성했다.
g6 67% + g3 33% 조합은 fresh-v1의 15출처에서 g4b **66.24% → 70.36%**였고,
이후 따로 준비한 새 1,847문항·10출처에서도 **65.39% → 71.97%(+6.58%p)**였다.
아래의 소규모 adapter confirmation과는 다른 고정 조합·평가다.
[모델·직접 평가·한 번 forward 검증](g6-soup-replication.md).

비교 기준은 현재 실험에서 계속 사용한 **g4b-gold3-pad0/best**다. 공개 Qwen 원본이나
d3-lite를 이번 표의 baseline으로 바꿔 부르지 않는다. head 교체에서 개선을 확인하지 못해,
기존 출력층을 유지하면서 모든 층을 작게 적응시키는 방법과 후보를 다시 비교하는 방법을 조사했다.

## 논문에서 가져온 것과 이번에 직접 조합한 것

| 근거 | 확인한 방법과 적용 범위 | 이번 실험과의 관계 |
|---|---|---|
| [LoRA](https://arxiv.org/abs/2106.09685) | 원래 선형 가중치를 고정하고 낮은 rank의 BA를 학습한다. NLP 모델·분류 결과가 있으며 학습 후 원래 행렬에 합칠 수 있다. | 24개 text block 전체의 186개 선형 projection에 rank 8, alpha 16을 적용했다. 입력 임베딩·기존 code head·norm·conv는 고정한다. |
| [WiSE-FT](https://arxiv.org/abs/2109.01903), §3–4 | 원본과 적응 모델을 파라미터 공간에서 보간한다. 주된 실증 근거는 CLIP 등 **시각 분류**이며 우리 언어 판단의 상승을 보장하지 않는다. | g4b + c·평균(두 seed의 BA), c=.5/1.0을 실제 BF16 행렬로 합쳐 평가한다. 이 조합은 원 논문의 zero-shot 설정 재현이 아니다. |
| [Model soups](https://arxiv.org/abs/2203.05482) | 같은 출발 모델에서 미세조정한 모델들의 가중치를 평균한다. 하나의 모델로 추론한다. | seed 41·73을 항상 같은 비중으로 결합한다. **A 평균 × B 평균이 아니라 BA 업데이트 자체를 평균**한다. |
| [R-Drop](https://arxiv.org/abs/2106.14448) | dropout으로 얻은 두 예측 사이 일관성을 학습한다. | 이번에는 dropout을 추가하지 않았다. 같은 gold 문항의 **선택지·코드 순열 두 view** 사이 JSD를 넣는 별도 조합이다. R-Drop 재현이라고 주장하지 않는다. |
| [Consider, Then Choose / Runoff](https://www.alphaxiv.org/abs/2610.18233) | 후보를 추린 뒤 순서 영향을 줄여 다시 비교한다. 후보별 yes/no 단계가 포함된다. | 원래 예측의 top 2/4를 다시 listwise 비교한다. 이 shortlist reranker는 원 논문의 전체 Runoff 구현이 아니다. |
| [R3F](https://arxiv.org/abs/2008.03156), [SMART](https://arxiv.org/abs/1911.03437) | 미세조정 중 표현 변화·국소 불안정성을 제어하는 NLP 학습 방법이다. | 보존을 고려하는 학습 방향의 근거로 검토했다. embedding noise·adversarial/Bregman 최적화는 이번에 실행하지 않았다. |

alphaXiv의 추가 검색에서는 [Null-Basis LoRA](https://www.alphaxiv.org/abs/2609.25618),
[NA-LoRA](https://www.alphaxiv.org/abs/2606.31717),
[The Intruder Threshold](https://www.alphaxiv.org/abs/2607.23711)의 PDF도 읽었다.
각각 기존 activation의 null space에 업데이트 제한, SwiGLU gate별 학습 제어, spectral update budget을 다룬다.
post-RL reasoning 보존·다른 크기 모델 등의 결과를 g4b 정확도 개선으로 간주하지 않으며, 이번에는 구현하지 않았다.

## 데이터와 확인용 문서

- gold3 앞의 32,768행에서 길이·중복 조건을 통과한 표본을 고정 hash로 추출해 **16,384개/48출처**를 준비했다.
- 선택에는 이전부터 관찰한 **1,474개/47출처**를 그대로 쓴다. 새로운 확인 문항을 고르는 데 점수를 사용하지 않는다.
- gold3-dev의 이전 architecture·Pointer·입력·readout 실험 노출 ID를 모아, 연결된 **문서 그룹 전체**를 제외했다.
- 원래 학습 파일 **399,978행 전체**와 upstream ID·state·공백 정규화 state·item·document key를 감사했다.
  dev에서 학습과 key를 공유한 7개 ID의 그룹도 제외했다. 의미 수준의 근사 중복 전부를 검사한 것은 아니다.
- 남은 문서에서 모든 고정 cyclic view가 4,096토큰 이하인 **2,998개/39출처/2,505문서 그룹**을 확인용으로 고정했다.
  이 중에는 이전 1,536토큰 실험에서 다루지 못한 긴 문맥도 있다. 선택셋과 확인셋의 절대 정확도를 직접 비교하지 않는다.
- g4b는 과거 **gold3-dev 전체를 early stopping에 사용**했다. 이 확인셋은 이번 검색에 새로 남겨둔 문서이며,
  모델 생애 전체에서 한 번도 쓰이지 않은 독립 benchmark는 아니다.
- 신규 teacher label·rationale는 사용하지 않는다. 학습·선택·확인 데이터와 원본 모델 hash를 보존한다.

## 사전 고정한 비교

### 동일 계산량의 CE와 순열 일관성 학습

각 seed에서 같은 semantic 문항 draw와 같은 두 순열 view를 사용한다.
두 view의 code 점수를 **원래 선택지 의미 순서로 되돌린 뒤** 다음 손실을 계산한다.

`L = (CE(view1, gold) + CE(view2, gold))/2 + lambda * JSD(p1, p2)`

- 대조군 lambda=0, 제안 조합 lambda=.5. 두 방법 모두 같은 두 view를 forward/backward한다.
- 순서가 의미를 갖는 질문·선택지가 위치를 참조하는 질문·noul은 순서를 유지한다.
- 유효한 choice 문항에서 선택지를 섞고, 확률 .2로 표시 코드도 섞는다. padding 선택지는 추가하지 않는다.
- seed 41/73, effective batch 16, **256 update = 4,096문항 draw = 8,192 prompt view/run**이다.
  16,384개를 전부 학습한 에폭이라고 부르지 않는다.
- FP32 adapter AdamW, lr=1e-4, betas(.9,.95), weight decay=.01, clip 1,
  16 update warmup 후 cosine으로 최대 학습률의 .1까지 낮춘다.
- 128/256 update를 저장하고 두 seed의 **같은 step** 업데이트를 평균한다. c=.5/1.0을 비교하므로 계열당 후보는 4개다.
- 모든 선택·확인 평가는 adapter 연산을 따로 더한 근사치가 아닌, 실제로 합친 BF16 가중치로 실행한다.
  학습 중 원래 가중치는 정확히 복원한다. 합친 뒤 projection 추가 연산은 없다.

### 순서 평균과 후보 재비교

- cyclic 평균은 서로 다른 shift `0, floor(k/4), floor(k/2), floor(3k/4)`의 **확률 평균**이다.
  2·3선택지는 중복 view를 제거한다. 보호해야 하는 문항은 원래 순서 한 번만 읽는다.
- 기본 모델 및 CE/일관성 계열의 선택용 최선 soup 각각에서 cyclic 평균을 비교한다.
- shortlist는 기본 확률의 top 2/4만 남겨 한 번 또는 cyclic view로 다시 비교한다.
  원래 선택지 상대 순서와 실제 선택지 key/description을 유지한다.
- 재비교 확률은 기존 shortlist의 전체 확률 질량을 재분배한다. shortlist 밖의 확률은 보존한다.
  혼합 강도 .5/1.0, 후보 수 2/4, cyclic 유무로 **8개 설정**을 선택셋에서만 비교한다.
- shortlist에 정답이 없을 수도 있으며 이를 허용한다. 후보 선택에는 gold를 읽지 않는다.
  noul·순서 의미 문항·이미 이진인 문항은 기본 결과를 유지한다.

### 동결과 판정

단일 forward soup 최선 1개, cyclic 계열 최선 1개, shortlist 계열 최선 1개를
**공유 확인셋의 점수를 읽기 전에 모두 고정**한다. 결과를 보고 seed·혼합 비율·후보 수를 다시 고르지 않는다.
출처 내 문서 단위 paired bootstrap 5,000회를 사용한다. 기본 대비 세 비교를 함께 보고하므로
Bonferroni **98.3333% 양측 구간**의 하한이 0보다 클 때 확인된 개선으로 판정한다.
단순 점수 상승과 이 판정을 구분한다. 새 확인용 ID도 평가 후에는 관찰한 자료로 취급한다.

추가로 [kNN-LM](https://arxiv.org/abs/1911.00172)에서 착안한 gold memory 혼합도 비교했다.
다른 질문의 코드 A를 같은 의미의 label로 취급하지 않도록 질문·선택지 key/description·표시 순서가
정확히 같은 schema에서만 검색했다. 학습 사례 32개 이상·gold class 2개 이상인 schema별로
평균을 빼고 정규화한 마지막 벡터를 캐시했다. k=8/32, 혼합 .1/.25/.5를 비교했다.
27개 schema·3,797개 학습 사례가 있었으나 선택셋 지원은 99/1,474문항이었다.
최선은 **78.82%로 원본과 같아** 사전 조건 +.25%p를 통과하지 못했고 확인셋에서 평가하지 않는다.
이 후보까지 확인했다면 비교 수를 4로 늘리도록 별도 계획에 정했으나 실제 joint 비교 수는 3이다.

## 실행과 검증

```bash
.venv/bin/python -m qwen_decision.v3.robust_experiment prepare
.venv/bin/python -m qwen_decision.v3.shortlist_experiment plan
.venv/bin/python -m qwen_decision.v3.memory_experiment plan
.venv/bin/python -m qwen_decision.v3.robust_experiment baseline
.venv/bin/python -m qwen_decision.v3.robust_experiment fit
.venv/bin/python -m qwen_decision.v3.robust_experiment screen
.venv/bin/python -m qwen_decision.v3.shortlist_experiment screen
.venv/bin/python -m qwen_decision.v3.memory_experiment screen
# 모든 family 동결·gate 확인을 검사한 뒤 공유 확인셋 평가:
.venv/bin/python -m qwen_decision.v3.robust_report confirm
.venv/bin/python -m qwen_decision.v3.robust_report export
.venv/bin/python -m unittest tests.test_v3_robust -v
```

`runs/robust/search-v1/`에 gold 입력·예측·학습 손실·adapter·계획·동결·hash를 보존한다.
기존 모델과 serving 기본값은 실험 후보의 검증 전까지 유지한다.

원본 점수 초기 일치, 실제 행렬 합치기와 복원, BA 평균·adapter 재로딩, 선택지 의미 정렬,
padding을 포함한 JSD gradient, shortlist 질량 보존·정답 제외 허용, 실제 작은 Qwen decoder의
전 층 adapter gradient, memory schema·query gold 비의존성, fresh label·가변 chance 계산을 포함한
문서 bootstrap의 출처 평균 계산, 새 확인셋의 지문 형제 그룹 제외까지 포함한 **10개 테스트**가 통과했다.

0.8B 모델의 실제 GPU smoke check에서도 186개 projection, **5,411,328개 학습 parameter**의
gradient가 유한했다. RTX 3090에서 4×1,024토큰의 forward/backward 두 번째 실행은 .47초,
peak allocated 약 13.99GB였다. 이는 학습 가능성 점검이며 전체 학습·추론 성능 수치가 아니다.

## 완료한 학습·선택·별도 확인 결과

CE/CE+순열 JSD × seed 41/73의 **4개 run을 모두 256 update**까지 실행했다.
각 run은 5,411,328개의 adapter parameter를 학습했고, 4,096개 semantic draw를 두 view로 읽었다.
전체 text 층에 adapter를 두었지만 16,384개 준비 표본 전체나 모든 원래 가중치를 학습한 실험은 아니다.

두 seed의 **BA 업데이트 자체를 평균한 단일 모델**에서 고른 선택용 결과는 다음과 같다.
이는 seed별 정확도의 산술평균이 아니다.

| 선택 후보 | selection 출처 평균 |
|---|---:|
| 원본 g4b | 78.82% |
| CE, step 128, 원본과 .5 혼합 | 79.15% |
| CE+JSD, step 128, 원본과 .5 혼합 | 79.22% |
| CE soup + cyclic 평균 | 79.69% |
| JSD soup + cyclic 평균 | 79.69% |
| top-4 cyclic 재비교, 혼합 .5 | 78.89% |

단일 후보는 JSD soup, cyclic 계열 동점에서는 정해진 이름 순서로 CE soup,
shortlist는 top-4 cyclic .5를 고정했다. 가중치·후보 hash와 모든 family 선택이 끝난 뒤에만
이전 검색에서 관찰하지 않은 **2,998문항/39출처/2,505문서 그룹**을 평가했다.

| 고정 모델 | confirmation 정확도 | 기본 대비 | 세 비교의 98.3333% 구간 |
|---|---:|---:|---:|
| 원본 g4b | **80.83%** | — | — |
| JSD adapter soup, 한 번 forward | 80.64% | −0.20%p | [−0.72,+0.29]%p |
| CE adapter soup + cyclic 평균 | 81.14% | +0.31%p | [−0.49,+1.20]%p |
| top-4 cyclic 재비교, 혼합 .5 | 81.06% | +0.23%p | [−0.15,+0.62]%p |

모든 구간이 0을 포함해 **이번 세 추가 방법 중 확인된 개선은 없었다.** selection과 confirmation은
출처·길이 구성이 다르므로 두 열의 절대 점수 차이를 학습 효과로 해석하지 않는다.
g4b 자체의 과거 early stopping에 전체 gold3-dev가 쓰인 한계도 남는다.

> **10-31 갱신:** g4b에 붙인 r8 LoRA와 달리, 원래 Qwen에서 시작한 전 층 LoRA r16은 지금까지 최선 레시피다([LoRA 위치 비교](lora-placement.md)). 순서 일관성도 원래 Qwen에서 다시 쟀다([Loss 비교](loss-screen.md)). 출발점, rank, 예산이 달라 모순은 아니다.

confirmation의 실제 prompt 계산 수는 기본·단일 soup 각각 **2,998개**, cyclic soup **9,213개**였다.
shortlist는 기본 2,998개에 **7,914개를 추가**했다. token 수는 각각 3,075,185,
cyclic 9,043,742, shortlist 합계 10,215,233이었다. GPU를 다른 평가와 일부 공유했으므로
wall time을 production latency로 비교하지 않는다.

최종 파일에서 다시 구성한 selection 예측의 불일치는 단일 soup·cyclic·shortlist 모두 **0개**였다.
4개 run의 마지막 64 update에서 JSD 항은 CE 대비 약 4–5% 작아졌지만 정확도 개선을 입증하지 못했다.
이 작은 예산·강도 결과를 모든 일관성 학습의 상한으로 해석하지 않는다.

[전체 수치·joint 보정 구간·학습 로그 요약·hash·재로딩 검증](robust-combinations-results.json)에 보존했다.
이번에 더 좋은 사용 가능한 모델로 남긴 것은 [별도 g6/g3 조합과 새 문항 확인](g6-soup-replication.md)이다.
