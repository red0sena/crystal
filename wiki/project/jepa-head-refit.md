# 고정한 상태에서 판정기를 따로 학습하고 이미지로 전이하기

[위키 홈](../README.md) · [기존10seed 결과](jepa-decision-seeds10.md) · [논문 분석](../papers/platonic-jepa.md)

2026-11-04. **270개 판정기 학습·평가·검산 완료.**

기존 실험은 상태 인코더와 판정기를 CE+LeJEPA로 함께 학습했다. 이번에는 기존 상태를 고정하고,
텍스트만으로 판정기를 새로 학습한 뒤 같은 판정기에 회전한 이미지 상태를 넣는다.
학습된 표현을 판정기가 다시 읽을 수 있는지, 정규화와 정렬 잔차에 맞춘 노이즈가 각각 어떤 영향을 주는지 확인한다.
텍스트 성능이 개선된다고 가정하거나 낮은 seed를 제외하지 않는다.

good 표현의 이미지 seen 정확도는 원 판정기52.83%, 원래 크기에서 재학습53.68%,
중심화·정규화 후58.52%, 잔차 노이즈를 추가하면62.23%였다.
새 판정기 조건 안에서 정규화의 차이는+4.84%p [2.91,6.75], 노이즈의 추가 차이는+3.71%p [2.39,5.15]였다.
따라서 이 고정 표현과 합성 분할에서는 입력 크기를 맞추고 판정기에 정렬 잔차에 대한 강건성을 주는 방법이
이미지 전이에 도움이 됐다. 세 단계의 합산 변화만으로 한 요인의 효과를 주장하지 않는다.

같은 정규화·노이즈 조건의 good−CE 이미지 seen 차이는+11.51%p [5.91,16.66],
good−bad는+7.43%p [2.39,12.52]였다. 이는 해당 조건의 모델 비교이고 독립 인코더 정리의 입증은 아니다.
구간은 각각의95% 구간이며 복수 비교 보정을 하지 않았다.

텍스트의 낮은 일반화는 판정기 재학습만으로 해결되지 않았다.
good의 text unseen 평균은 원 판정기56.56%, raw_clean54.91%, unit_clean55.79%, unit_noise56.08%였다.
unit_noise의 text seen 정확도는unit_clean보다+0.53%p [0.09,1.04]였지만 NLL은+0.081 [0.038,0.138]로 나빠졌다.
정확도가 소폭 올랐다는 이유로 텍스트 성능 전반이 개선되거나 보존됐다고 단정하지 않는다.
상태의 정보 부족과 판정기의 학습 한계를 이번 비교만으로 완전히 분리할 수는 없다.

## 결과

정확도는10개 상태 인코더 seed×3개 판정기 seed의 개별 점수 평균이다. 앙상블이 아니다.

| 표현 학습 | 판정기 조건 | text seen | image R seen | text unseen | image R unseen |
|---|---|---:|---:|---:|---:|
| ce | original | 94.25% | 48.02% | 53.13% | 34.75% |
| ce | raw_clean | 93.83% | 47.54% | 48.62% | 34.23% |
| ce | unit_clean | 94.18% | 48.15% | 50.60% | 35.67% |
| ce | unit_noise | 93.68% | 50.72% | 51.72% | 36.07% |
| bad | original | 95.17% | 44.47% | 60.11% | 36.64% |
| bad | raw_clean | 94.58% | 45.00% | 57.01% | 36.97% |
| bad | unit_clean | 94.50% | 50.43% | 58.47% | 43.12% |
| bad | unit_noise | 92.71% | 54.80% | 57.24% | 43.59% |
| good | original | 93.88% | 52.83% | 56.56% | 39.44% |
| good | raw_clean | 93.69% | 53.68% | 54.91% | 39.03% |
| good | unit_clean | 93.81% | 58.52% | 55.79% | 41.82% |
| good | unit_noise | 94.34% | 62.23% | 56.08% | 43.83% |

### 사전 고정한 비교의 차이와95% 구간

정확도는%p, NLL은 원 단위다. 아래 구간은 개별95% 구간이며 다중 비교 보정은 없다.

| 비교 | 입력 | 질문 | 정확도 차이 [95% 구간] | NLL 차이 [95% 구간] |
|---|---|---|---:|---:|
| ce:raw_clean-minus-ce:original | text | seen | -0.41 [-1.59,+0.66] | +0.031 [+0.003,+0.060] |
| ce:raw_clean-minus-ce:original | text | unseen | -4.51 [-9.05,-1.31] | -0.172 [-0.690,+0.330] |
| ce:raw_clean-minus-ce:original | rotation | seen | -0.49 [-1.86,+1.02] | -0.164 [-0.304,-0.021] |
| ce:raw_clean-minus-ce:original | rotation | unseen | -0.52 [-3.74,+2.43] | -0.148 [-0.345,+0.043] |
| ce:unit_clean-minus-ce:raw_clean | text | seen | +0.35 [-0.47,+1.49] | -0.019 [-0.046,+0.001] |
| ce:unit_clean-minus-ce:raw_clean | text | unseen | +1.98 [-1.55,+6.98] | -0.003 [-0.441,+0.443] |
| ce:unit_clean-minus-ce:raw_clean | rotation | seen | +0.61 [-0.72,+2.06] | +0.652 [+0.423,+0.882] |
| ce:unit_clean-minus-ce:raw_clean | rotation | unseen | +1.44 [-0.69,+3.60] | +0.848 [+0.576,+1.126] |
| ce:unit_noise-minus-ce:unit_clean | text | seen | -0.51 [-1.01,-0.05] | +0.248 [+0.190,+0.306] |
| ce:unit_noise-minus-ce:unit_clean | text | unseen | +1.11 [-1.32,+3.58] | -1.657 [-2.175,-1.076] |
| ce:unit_noise-minus-ce:unit_clean | rotation | seen | +2.57 [+1.41,+3.81] | -1.451 [-1.647,-1.242] |
| ce:unit_noise-minus-ce:unit_clean | rotation | unseen | +0.41 [-1.28,+2.17] | -1.798 [-2.018,-1.582] |
| bad:raw_clean-minus-bad:original | text | seen | -0.58 [-2.22,+0.38] | +0.019 [-0.006,+0.054] |
| bad:raw_clean-minus-bad:original | text | unseen | -3.10 [-9.15,+0.66] | -0.457 [-0.865,-0.009] |
| bad:raw_clean-minus-bad:original | rotation | seen | +0.53 [-0.84,+2.02] | -0.047 [-0.133,+0.021] |
| bad:raw_clean-minus-bad:original | rotation | unseen | +0.33 [-2.43,+2.96] | -0.077 [-0.189,+0.014] |
| bad:unit_clean-minus-bad:raw_clean | text | seen | -0.08 [-1.36,+1.75] | +0.008 [-0.030,+0.040] |
| bad:unit_clean-minus-bad:raw_clean | text | unseen | +1.46 [-2.06,+7.48] | +0.129 [-0.192,+0.484] |
| bad:unit_clean-minus-bad:raw_clean | rotation | seen | +5.44 [+2.64,+8.36] | +0.859 [+0.553,+1.199] |
| bad:unit_clean-minus-bad:raw_clean | rotation | unseen | +6.15 [+2.32,+9.95] | +1.054 [+0.688,+1.413] |
| bad:unit_noise-minus-bad:unit_clean | text | seen | -1.80 [-4.54,+0.21] | +0.324 [+0.180,+0.496] |
| bad:unit_noise-minus-bad:unit_clean | text | unseen | -1.23 [-5.22,+3.14] | -1.153 [-1.718,-0.603] |
| bad:unit_noise-minus-bad:unit_clean | rotation | seen | +4.37 [+3.01,+5.83] | -1.163 [-1.542,-0.839] |
| bad:unit_noise-minus-bad:unit_clean | rotation | unseen | +0.47 [-1.82,+2.67] | -1.311 [-1.675,-0.972] |
| good:raw_clean-minus-good:original | text | seen | -0.19 [-1.31,+0.99] | +0.026 [+0.001,+0.053] |
| good:raw_clean-minus-good:original | text | unseen | -1.66 [-5.29,+1.73] | -0.326 [-0.756,+0.186] |
| good:raw_clean-minus-good:original | rotation | seen | +0.84 [-1.02,+2.81] | -0.042 [-0.179,+0.077] |
| good:raw_clean-minus-good:original | rotation | unseen | -0.41 [-3.50,+2.27] | -0.100 [-0.290,+0.064] |
| good:unit_clean-minus-good:raw_clean | text | seen | +0.12 [-0.87,+1.16] | -0.011 [-0.038,+0.014] |
| good:unit_clean-minus-good:raw_clean | text | unseen | +0.88 [-1.66,+4.12] | +0.113 [-0.268,+0.481] |
| good:unit_clean-minus-good:raw_clean | rotation | seen | +4.84 [+2.91,+6.75] | +0.391 [+0.259,+0.544] |
| good:unit_clean-minus-good:raw_clean | rotation | unseen | +2.79 [+0.31,+5.40] | +0.946 [+0.676,+1.229] |
| good:unit_noise-minus-good:unit_clean | text | seen | +0.53 [+0.09,+1.04] | +0.081 [+0.038,+0.138] |
| good:unit_noise-minus-good:unit_clean | text | unseen | +0.29 [-1.16,+1.91] | -1.261 [-1.869,-0.623] |
| good:unit_noise-minus-good:unit_clean | rotation | seen | +3.71 [+2.39,+5.15] | -0.637 [-0.798,-0.493] |
| good:unit_noise-minus-good:unit_clean | rotation | unseen | +2.02 [+0.09,+4.28] | -1.046 [-1.286,-0.840] |
| good:raw_clean-minus-ce:raw_clean | text | seen | -0.14 [-3.15,+2.85] | +0.003 [-0.076,+0.081] |
| good:raw_clean-minus-ce:raw_clean | text | unseen | +6.29 [-10.99,+24.81] | -0.383 [-1.526,+0.646] |
| good:raw_clean-minus-ce:raw_clean | rotation | seen | +6.14 [+0.79,+11.38] | -0.852 [-1.174,-0.581] |
| good:raw_clean-minus-ce:raw_clean | rotation | unseen | +4.80 [-0.14,+9.78] | -1.021 [-1.199,-0.854] |
| good:raw_clean-minus-bad:raw_clean | text | seen | -0.89 [-2.82,+1.28] | +0.014 [-0.031,+0.063] |
| good:raw_clean-minus-bad:raw_clean | text | unseen | -2.10 [-12.93,+8.69] | +0.337 [-0.458,+1.127] |
| good:raw_clean-minus-bad:raw_clean | rotation | seen | +8.68 [+4.12,+13.19] | -0.256 [-0.423,-0.112] |
| good:raw_clean-minus-bad:raw_clean | rotation | unseen | +2.06 [-1.36,+5.69] | -0.106 [-0.269,+0.033] |
| good:unit_clean-minus-ce:unit_clean | text | seen | -0.37 [-3.85,+2.92] | +0.011 [-0.070,+0.095] |
| good:unit_clean-minus-ce:unit_clean | text | unseen | +5.19 [-13.77,+24.85] | -0.268 [-1.581,+0.999] |
| good:unit_clean-minus-ce:unit_clean | rotation | seen | +10.37 [+5.78,+14.51] | -1.112 [-1.416,-0.771] |
| good:unit_clean-minus-ce:unit_clean | rotation | unseen | +6.15 [+0.71,+11.23] | -0.923 [-1.250,-0.576] |
| good:unit_clean-minus-bad:unit_clean | text | seen | -0.69 [-2.56,+0.82] | -0.006 [-0.047,+0.044] |
| good:unit_clean-minus-bad:unit_clean | text | unseen | -2.68 [-12.46,+6.53] | +0.321 [-0.490,+1.118] |
| good:unit_clean-minus-bad:unit_clean | rotation | seen | +8.09 [+3.14,+12.73] | -0.723 [-1.148,-0.322] |
| good:unit_clean-minus-bad:unit_clean | rotation | unseen | -1.30 [-5.19,+2.46] | -0.215 [-0.687,+0.242] |
| good:unit_noise-minus-ce:unit_noise | text | seen | +0.66 [-2.64,+3.79] | -0.156 [-0.225,-0.099] |
| good:unit_noise-minus-ce:unit_noise | text | unseen | +4.36 [-14.56,+23.68] | +0.128 [-0.414,+0.625] |
| good:unit_noise-minus-ce:unit_noise | rotation | seen | +11.51 [+5.91,+16.66] | -0.297 [-0.431,-0.150] |
| good:unit_noise-minus-ce:unit_noise | rotation | unseen | +7.76 [+2.09,+12.93] | -0.171 [-0.270,-0.064] |
| good:unit_noise-minus-bad:unit_noise | text | seen | +1.64 [-0.50,+4.61] | -0.249 [-0.368,-0.136] |
| good:unit_noise-minus-bad:unit_noise | text | unseen | -1.16 [-9.82,+7.93] | +0.212 [-0.123,+0.524] |
| good:unit_noise-minus-bad:unit_noise | rotation | seen | +7.43 [+2.39,+12.52] | -0.196 [-0.319,-0.072] |
| good:unit_noise-minus-bad:unit_noise | rotation | unseen | +0.24 [-3.56,+4.12] | +0.051 [-0.032,+0.133] |

### 검산과 해석 범위

1,658,880개 질문의 판정을 원래 decoder로 대조했고 장면·과제별 정답 개수와 분모가 일치했다.
유효 후보 log 확률 최대 오차는 1.91e-05였다. 근접 동점의 argmax 차이는 0건이며 원 실행의 선택을 유지했다.
학습을 독립 재실행하지 않았고 bootstrap 구간도 별도 구현으로 재검산하지 않았다. 모든 입력 파일의 해시와 train-only 회전·calibration·checkpoint 선택·평균은 검산했다.
텍스트가 낮은 seed도 모두 포함한다. text seen95%/unseen90% 통과 수와 과제별 점수는 전체 결과 JSON에 기록했다.
정규화·노이즈 비교는 이번 판정기의 진단이며 독립 인코더의 직교 동치나 일반 자연어·이미지 성능 향상을 입증하지 않는다.

## 실행 전에 고정한 설계

- 상태 인코더: 기존CE/bad/good 각각10seed, 총30개. 추가 학습 없이 저장된64차원 canonical 상태를 읽는다.
- 판정기 초기화:701,719,733. 모든 인코더와 조건에서 같은 세 초기화를 쓴다.
- 조건: 원래 크기에서 새 판정기를 학습하는 `raw_clean`, 중심화·단위 길이 입력의 `unit_clean`, 같은 입력에 노이즈를 넣는 `unit_noise`.
- 총270개 판정기를 각각 학습한다. 파라미터·optimizer 상태·gradient clipping·최선 checkpoint는 판정기마다 독립적이다.
- 데이터: 원래1024개 train 세계,8922개 seen 텍스트 질문만 정답 CE로 사용한다. 이미지 정답은 학습하지 않는다.
- 판정기: 기존과 같은64→256→32의 factored decoder. 질문의 구조화된 metadata는 그대로 제공한다.
- AdamW lr.001, wd.0001, 판정기별 gradient clip1, batch256세계,400epoch=1600step. 모든 조건에 같은 세계 순서를 쓴다.
- 고정 epoch를 모두 수행하고 clean text seen validation NLL이 가장 낮은 checkpoint를 판정기별로 저장한다.

`original`은 원래 함께 학습한 판정기 결과를 참고로 재사용한다. `raw_clean−original`은 판정기 재학습의 효과를 진단하지만,
기존 판정기는 인코더와 공동 학습했고 validation256개를 썼으므로 엄밀한 단일 요인 대조로 부르지 않는다.
새 세 조건 사이에서는 인코더·초기화·데이터·배치·선택 분할·optimizer·학습 길이를 고정했다.

## 정렬과 노이즈 calibration

직교 회전R과 두 모달리티의 평균은 기존 실험에서 train1024개 상태로 적합한 값을 그대로 사용한다.
원래 크기의 이미지 입력은 `(image−image_mean)R+text_mean`이다.
unit 조건에서는 `normalize(text−text_mean)`와 `normalize((image−image_mean)R)`를 같은 판정기에 넣는다.
평가 중 노이즈를 추가하지 않으며 실제 대응을 써서R을 다시 fitting하지 않는다.

validation256개 중 앞128개(1024:1152)는 정렬 잔차의 평균 cosine c를 측정하는 데만 쓴다. 정답은 쓰지 않는다.
뒤128개(1152:1280)는 clean text seen NLL에 따른 판정기 checkpoint 선택에만 쓴다.
test256개와 unseen 질문 정답은 학습·calibration·선택에 쓰지 않는다.

논문[§5.1의 residual-matched noise](https://arxiv.org/pdf/2610.39302v1#page=13)를 다음처럼 구현한다.

```text
sigma = sqrt(c^(-2) − 1)
z_train = normalize(text − text_mean) + epsilon
epsilon ~ N(0, sigma²/64 I)
```

sigma는 unit 상태에 대한 노이즈의 RMS L2 크기다. 좌표별 표준편차는sigma/√64다.
노이즈 추가 후 다시 정규화하지 않는다. 평균 cosine과 sigma의 관계는 등방성 노이즈를 이용한 근사적 규칙이며
실제 이미지 잔차가 등방 Gaussian이라는 가정까지 검증한 것은 아니다. test 결과에 맞춰 sigma를 조정하지 않는다.

## 집계와 해석

각 판정기의 text/image, seen/unseen, 색·모양·관계 과제 정확도와NLL을 저장한다.
표의 평균은 모델별 점수 평균이며 예측을 합치는 앙상블이 아니다.
10개 인코더 seed 안의3개 판정기 seed는 독립적인 인코더30개로 세지 않는다.
짝지은 bootstrap5000회는 인코더 seed·판정기 seed·test 세계를 각각 복원 추출한다.
판정기 초기화를 모든 인코더에서 공유하므로 각 draw에서 같은 판정기 seed 추출을 전체 인코더에 적용한다.
방법 비교에도 같은 추출을 적용한다.

사전에 고정한 비교는 각 표현 학습 조건 안에서 `raw_clean−original`, `unit_clean−raw_clean`, `unit_noise−unit_clean`과,
각 새 판정기 조건에서 `good−CE`, `good−bad`다. 정확도와NLL을 함께 남기고 성능이 가장 좋은 조건만 골라 보고하지 않는다.
각95% 구간에는 다중 비교 보정을 적용하지 않는다. 이미 확인한 합성 test 분할의 후속 진단이다.

표현이 담고 있지 않은 정보를 회전이나 노이즈가 복원해 준다고 가정하지 않는다.
판정기 재학습 뒤 텍스트가 좋아져도 인코더 자체의 일반성을 입증하지 않으며, 개선되지 않아도 정보 부재를 단정할 수 없다.
기존 공유 Qwen·공유 adapter를 재사용하므로 독립 모달리티 인코더의 JEPA 정리를 재현한 실험이 아니다.

## 코드·결과·검산

- [실행 코드](../../qwen_decision/v3/jepa_head_refit.py)
- [확률식·gradient·독립 optimizer·노이즈·집계 검사](../../tests/test_v3_jepa_head_refit.py)
- 완료 후 전체 결과: [JSON](jepa-head-refit-results.json)
- 완료 후 검산 범위: [JSON](jepa-head-refit-verification.json)
- 실행 디렉터리: `runs/rotation-state/jepa-head-refit-v1/`의plan/status/run.log.

기존 인코더·상태·보고서는 수정하지 않고 입력 해시로 보존 여부를 대조한다.
실행기는 학습 후 평가·검산·위키 저장을 순서대로 수행한다. 주기적인 모니터링은 사용하지 않는다.

```bash
PYTHONPATH=.:external/decision-index .venv/bin/python -m qwen_decision.v3.jepa_head_refit prepare
PYTHONPATH=.:external/decision-index .venv/bin/python -u -m qwen_decision.v3.jepa_head_refit run
```
