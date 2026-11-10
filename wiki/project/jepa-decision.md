# 실제 상태 판정 경로에 LeJEPA 정렬 학습 추가

[위키 홈](../README.md) · [논문 분석](../papers/platonic-jepa.md) · [이전 Crystal 비교의 범위](crystal-state-methods.md#검증-질문-재점검)

[10seed 확대 비교](jepa-decision-seeds10.md)를 별도 진행한다. 아래는 보존한 원래3seed 결과다.

2026-11-04. **3조건×3seed 학습·평가·독립 검산 완료.**
좋은 관측 쌍을 쓰면 CKA와 대응 검색이 올라가고, bad-view 대비 seen 이미지 판정도 좋아졌다.
CE만 학습한 기준선 대비 이미지 정확도 차이는 불확실했고, 텍스트 능력과 unseen 질문 일반화는 안정적으로 보존하지 못했다.
**현재 근거는 ‘쌍 선택의 정렬 효과가 관측됨’이다. 일반 Decision 성능 개선이나 직교 동치의 입증으로 확대하지 않는다.**

[전체 결과 JSON](jepa-decision-results.json) · [독립 검산 JSON](jepa-decision-verification.json)

## 결과: 실제 같은 판정기의 이미지 전이

모든 정확도는 3seed 평균이며 색·모양·관계 과제의 macro 평균이다.
이미지는 train-only 중심화·직교 회전 후 **그때 학습한 동일 판정기**로 답했다.

| 학습 | text seen | image seen | image seen NLL | text unseen | image unseen |
|---|---:|---:|---:|---:|---:|
| CE | 98.79% | 46.36% | 2.534 | 70.18% | 40.61% |
| CE + bad-view LeJEPA | 95.21% | 42.77% | 1.658 | 64.19% | 34.71% |
| CE + good-view LeJEPA | 93.71% | 52.51% | 1.193 | 56.72% | 38.29% |

정확도는 높을수록, NLL은 낮을수록 좋다. seen/unseen은 학습된/제외된 질문 signature를 뜻한다.
두 그룹 모두 test 세계는 train 세계와 다르다.

| good의 비교 대상 | 질문 | 정확도 차이, 95% 구간(%p) | NLL 차이, 95% 구간 |
|---|---|---:|---:|
| CE | seen | +6.15 [−5.55,+16.92] | −1.340 [−2.082,−0.365] |
| bad | seen | +9.74 [+6.64,+12.82] | −0.464 [−0.712,−0.255] |
| CE | unseen | −2.32 [−11.62,+7.99] | −1.003 [−1.467,−0.506] |
| bad | unseen | +3.58 [−3.52,+10.87] | −0.249 [−0.608,+0.054] |

따라서 **같은 정규화 형식에서 좋은 관측 쌍을 고른 효과는 seen 이미지 판정에 나타났다.**
CE 대비 정확도 우위는 구간이0을 포함해 확정하기 어렵다. CE 대비 이미지 NLL은 두 질문 그룹에서 좋아졌다.
이 확률 점수 개선과 정확도 우위의 불확실성을 함께 유지한다.

## 결과: 표현의 기하

| 학습 | biased linear CKA | mutual10NN | 회전 후 검색 R@1 | 회전 후 paired cosine |
|---|---:|---:|---:|---:|
| CE | .2286 | .1465 | 20.70% | .4412 |
| bad | .2947 | .1923 | 30.47% | .4707 |
| good | .4066 | .2674 | 41.54% | .6049 |

CKA와 이웃 겹침은 회전으로 수치를 올릴 수 없는 지표다. 이 두 지표도 함께 올라,
이번 변화가 fitted R 뒤 cosine만 높아진 현상에 그치지는 않는다.
이는 같은 frozen Qwen과 공유 adapter에서 측정한 관측치이며, 독립 인코더의 정리 재현은 아니다.

## 텍스트 기준선과 seed 편차

| 학습 | seed | text seen | text unseen | 회전 image seen |
|---|---:|---:|---:|---:|
| CE | 401 | 99.12% | 58.78% | 46.91% |
| CE | 419 | 100.00% | 94.94% | 39.49% |
| CE | 443 | 97.25% | 56.80% | 52.69% |
| bad | 401 | 93.42% | 53.28% | 43.90% |
| bad | 419 | 92.22% | 42.09% | 46.52% |
| bad | 443 | 100.00% | 97.19% | 37.90% |
| good | 401 | 92.94% | 38.70% | 54.45% |
| good | 419 | 88.19% | 32.65% | 56.96% |
| good | 443 | 100.00% | 98.80% | 46.13% |

good−CE의 seen 이미지 차이는 seed별 +7.54,+17.47,−6.56%p다.
좋은 쌍이 모든 초기화에서 CE보다 좋은 이미지 정확도를 준 것은 아니다.
평균 기준 good은 text seen95% 점검을 통과하지 못했고, 세 조건 모두 text unseen90%를 통과하지 못했다.
또한 평균이 통과한 조건도 모든 seed가 통과한 것은 아니다.
미학습 질문의 이미지 결과를 회전·정렬 하나의 성패로 귀속할 수 없다.
구조화된 판정기의 표현 가능성은 oracle 상태에서 전체18,432개 유효 질문을 맞히는 검사로 확인했지만,
이 검사는 실제 학습이 그 상태를 안정적으로 찾아낸다는 보장이 아니다.

## 보조 대조: 평균, 회전, scale, 잘못 연결한 쌍

아래는 seen 이미지 정확도다. scale은 test를 보기 전에 정한 보조 대조이며 train 상태만으로 적합했다.

| 학습 | raw | 평균 보정 | 직교 회전 | 회전+단일 scale | 20개 잘못된 R 평균 |
|---|---:|---:|---:|---:|---:|
| CE | 40.15% | 44.47% | 46.36% | 45.84% | 22.08% |
| bad | 39.55% | 35.05% | 42.77% | 50.72% | 22.14% |
| good | 49.05% | 44.63% | 52.51% | 58.10% | 21.73% |

good에서 실제 쌍으로 적합한 R은 평균 보정44.63%와 잘못된 쌍21.73%보다 높은52.51%를 보였다.
다만 회전+scale58.10%를 주 지표로 갈아 끼워 정확도 개선을 확정하지 않는다. 해당 차이의 구간은 이번 계획에서 계산하지 않았다.

canonical train 상태의 좌표별 표준편차 평균은 good text4.64/image1.01이었다.
최적 scale의 seed 평균도2.77로, **이미 정확히 같은 크기의 직교 동치 상태가 된 것은 아니다.**
SIGReg는 augmented view에 적용하고 canonical text는 CE로 학습했으므로, canonical text에 표준정규분포를 강제하지 않았다.
이 입력 분포 차이와 크기 차이는 후속 설계에서 확인할 구체적인 지점이다.
good의 image effective rank는 평균17.62/64로, 표준편차가1 근처라는 이유만으로 완전한 whitening을 주장할 수도 없다.

## 검산 범위

- 9개 checkpoint에서 cached token bank→상태를 다시 실행했다. canonical 상태27,648행의 최대 절대 차이는0이었다.
- 25조건×3,072문항×9run = **691,200건**을 별도의 판정 확률식으로 재계산했다.
  장면·질문 그룹·과제별 정답수 불일치0, NLL 합 최대 차이 `6.58×10⁻⁶`이었다.
- 세계에서 정답을 독립 재구성하고, NumPy SVD로 train-only R과 직교성을 대조했다.
  Gram 행렬로 CKA를 재계산하고 이웃·검색·headline도 대조했다.
- seed·장면의 추출 횟수를 가중치로 쓰는 다른 집계 방식으로 모든 비교의 bootstrap5,000회 구간을 재계산했다.
- 공식 pinned LeJEPA 코드를 격리해 같은128방향으로 비교했다. 손실값은 일치하고 gradient 최대 차이는 `2.91×10⁻¹¹`이었다.
- 관련7개 검사가 통과했다. padding 불변성, CE/JEPA의 실제 상태 인코더 gradient,
  good/bad 쌍의 주변분포, 전체 oracle 답, SIGReg 수치 적분, 기하의 회전 불변성, view 사실 보존을 확인했다.

고정 Qwen 앞단의 특징 추출 전체를 별도 구현으로 재실행한 검산은 아니다.
해당 bank·앞단 소스·원본 모델 파일은 해시로 고정했다. 원본 결과와 검산 결과는 별도 파일로 보존한다.

## 이번에 묻는 질문

같은 세계에 대한 서로 다른 관측을 묶는 표현 학습을 정답 CE에 더하면, 질문 없는 상태의
이미지·텍스트 기하가 가까워지고 텍스트로 학습한 판정기가 이미지에서도 더 잘 작동하는가?

이전 실험은 native Crystal을 학습한 다음 평균 pooling·PCA·새 MLP를 붙였다.
이번에는 **실제로 답을 만드는 64차원 상태 자체에 CE와 LeJEPA를 함께 건다.**
평가에서 새 판정기를 학습하지 않는다. [논문](https://arxiv.org/html/2610.39302v1)의
good-view 아이디어를 Qwen 위의 작은 상태 인코더에 적용하는 통제 실험이다.
논문의 독립 모달리티 인코더 학습이나 기존 native Crystal 경로의 재현은 아니다.

## 실제 학습·추론 경로

```text
장면 텍스트 또는 이미지
  → 고정 원본 Qwen3.5-0.8B, noncausal, 질문·선택지 없음
  → 내용 토큰 시퀀스 [가변 길이, 1024]
  → 공유 상태 인코더 Eθ → z∈R⁶⁴
  → 동일하게 학습한 판정기 Dψ(z, 구조화된 질문) → 답 확률
```

Qwen은 변경하지 않는다. 학습 대상은 상태 인코더와 판정기 총 **489,440개 파라미터**다.
토큰별 정규화·1024→128 투영·잔차 FFN·Transformer 층 하나·8개 query attention pooling·64차원 출력을 쓴다.
padding mask를 보존하며, 길이가 128을 넘으면 중단한다. 조용히 자르지 않는다.

판정기는 `64→256→32`로 세 위치의 색·모양과 배치 축에 대한 중간 logits를 만든다.
색/모양 참조와 상대 위치 질문을 공유하는 미분 가능한 결합 연산으로 답한다.
**중간 물체 속성 정답으로 학습하지 않는다.** 원래 seen 질문의 답 CE만 쓴다.
다만 질문 종류와 참조 대상은 기존 query metadata로 주므로 자연어 질문 이해를 학습한 모델은 아니다.
새 질문 일반화에는 이 구조적 사전 지식도 기여한다. 기존 native Decision 및 기존 MLP와 절대 점수를 직접 비교하지 않는다.

CE 경로의 `z`와 표현 손실의 `z`가 같은 텐서 정의를 쓴다. 별도 projector만 정렬하거나
상태를 우회해 답하는 경로는 없다. 질문은 상태를 만든 뒤 판정기에만 제공한다.

## 데이터와 양성 쌍

세계·분할·질문·정답은 [이전 상태 실험](state-rotation-v2.md)과 동일하다.
세 물체의 색·모양·순서와 가로/세로 배치로 만든 합성 세계 1,536개를 train1,024/validation256/test256으로 나눴다.
학습은 같은 **8,922개 text seen 문항**이고, test는 seen2,244/unseen828문항이다.
unseen은 새로운 장면과 더불어 지정된 질문 signature 6개도 학습에서 제외한 조건이다.
예: diamond의 색, purple 물체의 모양, 지정된 shape 쌍의 상대 위치.
새 자연 이미지 분포나 임의의 새로운 추론 과제를 뜻하지 않는다.

train 세계마다 모달리티별로 관측 2개와 관측당 변형 2개, 총 4개 view를 고정 생성한다.

| 입력 | 별도 관측 | 한 관측의 변형 |
|---|---|---|
| 텍스트 | 모든 물체·속성·위치를 말하는 서로 다른 서술 양식 | 선택적인 관사 a/the만 일부 제거 |
| 이미지 | 같은 세계를 위치 jitter·크기를 달리해 다시 렌더링 | 그 렌더링에 ±2의 약한 픽셀 잡음 |

색·모양·공간 관계를 지우는 마스킹, crop, flip은 쓰지 않는다.
세계와 질문 정답은 늘리지 않았다. 추가한 것은 같은 train 세계의 관측 변형이다.
관측은 작은 결정적 view pool에서 나오므로 논문 정리의 통계적 독립 가정을 충족했다고 주장하지 않는다.
두 조건은 변형의 세기와 서술 양식도 달라, 통계적 독립성 한 요인만 분리한 검정은 아니다.
입력을 점검한 결과 4,096개 텍스트 view의 색·모양 사실 12,288개와 배치 방향이 모두 보존됐다.
bad text 쌍 2,048개 중113개(5.52%)는 변형 seed가 달라도 결과 문자열이 같았다.
내용 토큰 길이는 canonical text28, augmented text17–23, image64였다. 길이128 제한으로 잘린 입력은 없다.

| 조건 | 정답 CE | 모달리티 내부 양성 쌍 | LeJEPA 계수 |
|---|---|---|---:|
| CE | 동일 | 계산만 하고 학습 가중치 0 | 0 |
| bad | 동일 | 같은 관측의 서로 다른 변형 | .1 |
| good | 동일 | 같은 세계의 서로 다른 관측 | .1 |

모든 조건은 매 step 같은 canonical text와 모달리티별 4개 view를 모두 인코딩한다.
good/bad의 선택 view 주변분포는 균등하고, 동일 seed의 초기 가중치와 canonical 세계 순서도 같다.
개별 step에서 선택된 두 view의 실현값까지 같다는 뜻은 아니다.
**이미지·텍스트를 직접 붙이는 학습 loss와 이미지 답 CE는 없다.**
이미지와 텍스트의 상태 인코더 가중치는 공유하므로, 두 모달리티가 독립 학습됐다는 주장도 하지 않는다.

## 손실과 고정 프로토콜

모달리티마다 두 view의 평균 `c=(z₁+z₂)/2`를 만들고,

```text
Linv = mean((zv − c)²)
LJEPA = Linv + .02 × SIGReg(concat(z₁,z₂))
Ltotal = LCE + .1 × mean(LJEPA_text, LJEPA_image)
```

SIGReg는 batch 크기를 곱한 sliced Epps–Pulley 통계다. 표준 정규분포의 특성함수와
임의 투영의 경험 특성함수 차이를 Gaussian 가중치로 적분한다.
저자 재현 환경이 고정한 [stable-pretraining ab836bf](https://github.com/galilai-group/stable-pretraining/blob/ab836bf/stable_pretraining/methods/lejepa.py)의
손실식을 확인했다. 적분 구간 0–3, trapezoid17점은 같고, 방향 수는 1,024 대신 128이다.
이 손실을 그대로 더한 값과 볼록 혼합 `(1−λ)Linv+λSIGReg`는 구분한다.
두 view 모두에 gradient가 흐르고, teacher·stop-gradient·별도 predictor는 없다.

- seed401/419/443, 조건당 200epoch·1,600update. 세계 batch128.
- AdamW lr.001, weight decay.0001, gradient clip norm1. FP32, 동일 matmul 설정.
- 같은 질문·정답을 200번 노출한다. 이전 1pass Crystal 학습과 학습 예산이 같다는 뜻은 아니다.
- 세 과제의 평균 NLL로 CE를 정의한다. checkpoint는 **text seen validation NLL 최소**로만 선택한다.
- 이미지·unseen·test로 checkpoint나 계수를 고르지 않는다. 완성된 9개 학습을 모두 보고한다.
- 예비 seed379로 text seen validation 능력만 확인했다. 첫 attention pooling 구성은 약93%여서 보존 후 수정했다.
  토큰 간 문맥 층을 추가하고 pool query 초기화를 바꾼 구성은100%로 사전95% 기준을 통과했다.
  이 수정 중 이미지 및 unseen/test 성능은 평가하지 않았다.

## 평가에서 고정하는 것

학습된 인코더와 **그때 학습된 동일 판정기**를 고정한다.
train1,024세계의 canonical 이미지·텍스트 상태 쌍으로 각 평균과 정사각형 직교 Procrustes `R`을 적합한다.
held-out 장면을 행렬 적합이나 평균 계산에 쓰지 않는다.

| 평가 입력 | 상태를 판정기에 넣는 방식 |
|---|---|
| text | `ztext` |
| image raw | `zimage` |
| image mean | `zimage − μimage + μtext` |
| image rotation, **주 지표** | `(zimage − μimage) R + μtext` |
| image similarity, 보조 | 위 centered 회전에 train-only 단일 최적 scale도 적용 |
| image shuffle | train 쌍을 20번 잘못 연결해 각각 얻은 R로 판정한 평균 |

잘못 연결한 쌍은 fixed point가 없는 순열이다. 참 대응 행렬과 동일한 train 세계·평균·차원을 쓴다.
회전은 이미지 정답을 보지 않지만 **train 이미지·텍스트의 대응은 안다.** pair-free 실험이 아니다.

학습 방식 good−CE, good−bad의 차이는 같은 seed와 test 장면을 묶어 bootstrap5,000회로 계산한다.
seed와 장면을 각각 복원 추출한다. 정확도는 과제별 정확도의 평균, NLL도 과제별 평균이며 낮을수록 좋다.
seed가3개뿐이고 이전에 본 합성 test 분할이므로 구간도 탐색적 불확실성으로 읽는다.
text seen95%·unseen90% 기준은 시험 결과 해석의 점검 조건이며, 실패한 run을 제외하는 선택 규칙이 아니다.

검색R@1은 train-only R로 돌린 후 test256개 안에서 짝 caption을 찾는다.
회전에 불변인 내부 기하는 **biased linear CKA**와 중심화 cosine mutual10NN으로 별도 기록한다.
CKA는 논문의 unbiased estimator와 같지 않다. collapse 진단으로 좌표 표준편차·effective rank도 남긴다.

## 이 설계로 말할 수 있는 범위

good−CE는 CE에 view invariance와 Gaussian 분포 정규화를 **함께** 추가한 전체 효과다.
good−bad는 동일한 정규화 형식에서 쌍을 만드는 방식의 효과를 더 직접적으로 본다.
검색이나 CKA가 올라도 판정이 안 오르면 Decision 개선을 확인했다고 말하지 않는다.
이미지 전이가 실패했는데 text도 기준에 못 미치면 두 병목을 분리했다고 말할 수 없다.

공유 adapter, 이미 공동 사전학습된 Qwen 앞단, 구조화된 합성 질문, 작은 view pool이라는 제한이 있다.
또한 같은 세계의 관측끼리만 연결한 유한 graph에서 비퇴화 eigengap이나 최적해 유일성을 확인하지 않았다.
서로 독립적으로 학습한 일반 이미지·텍스트 인코더가 직교 동치라는 정리의 재현으로 해석하지 않는다.

## 재현 파일

- [학습·평가 코드](../../qwen_decision/v3/jepa_decision.py), [독립 검산 코드](../../qwen_decision/v3/jepa_decision_verify.py), [7개 검사](../../tests/test_v3_jepa_decision.py).
- 실행 위치 `runs/rotation-state/jepa-decision-v1/`. `plan.json`에 데이터·토큰 bank·원본 Qwen 파일·소스 해시와 조건을 고정했다.
- `prepare-source.py`는 특징 추출 시점, `experiment-source.py`는 본 비교 시점의 소스다.
  세계·view 생성·추출 함수의 AST가 동일함을 확인하고, 예비 구조 변경은 따로 보존했다.
- 조건/seed마다 `fit.json`, `model.pt`, `evaluation-states.pt`, `test-records.npz`를 보존한다.

```bash
PYTHONPATH=.:external/decision-index .venv/bin/python -m qwen_decision.v3.jepa_decision run
PYTHONPATH=.:external/decision-index .venv/bin/python -m qwen_decision.v3.jepa_decision_verify
```

새 입력 특징 생성은 `prepare`, text-only 예비 점검은 `pilot`, 계획 고정은 `freeze`다.
기존 run이 있는 디렉터리에서는 `prepare`가 덮어쓰지 않고 중단한다.
