# Reasoning·증류·확률 보정

[위키 홈](../README.md)

## 서로 다른 목표

| 방법 | 배우거나 바꾸는 것 | 추론 출력 | 정답 선택이 바뀔 수 있는가 |
|---|---|---|---|
| 현재 CE 학습 | 후보 정답 점수 | 후보 분포 | 가능 |
| [Dual-Head](../papers/dual-head-reasoning-distillation.md) | 분류와 교사 설명의 공동 감독 | 분류층만 사용 | 가능 |
| [InnerJev](../papers/innerjev.md) | reasoning 이후 teacher 후보 분포 | 첫 답 위치 readout | 가능 |
| [Temperature scaling](../papers/temperature-scaling.md) | 별도 dev에서 양수 T 적합 | softmax(z/T) | 같은 후보에 동일 T면 argmax 불변 |

## 기본 선택 loss

정답 후보 y가 하나이면 L_CE=−log softmax(z)_y. 타깃이 분포 q이면 L_softCE=−Σ_i q_i log p_i. 후자는 teacher 분포나 ordinal 인접 등급의 soft target에 사용할 수 있지만 분포를 만든 근거가 다르다.

Dual-Head의 LM loss는 입력 이후 교사 설명·답을 포함하는 training sequence에 적용하고, 분류 readout은 설명 앞 입력 span만 사용한다. 설명 token을 입력에 추가한 후 bidirectional pooling으로 읽으면 누출 위험이 있으므로 구조 변경 시 특히 주의한다.

InnerJev는 설명 문장 자체를 학생이 재현하도록 학습하는 것이 아니라 설명 끝의 후보 분포를 바로 출력하도록 증류한다. teacher의 오답·miscalibration도 전달될 수 있다.

## '생성 없이 reasoning'의 의미

판단 forward 자체에도 여러 층의 계산이 있다. 그러나 설명을 생성하지 않는다고 긴 CoT와 동일한 연산을 숨겨서 수행한다고 주장할 수는 없다. 'Reasoning 감독이 분류 정확도에 도움이 되었는가'를 실제 평가로 확인한다. 학습 시 생성 비용과 온라인 inference 비용도 따로 기록한다.

## 확률과 ordinal score

확률은 제공한 후보 사이에서 정규화한 값이다. softmax 출력만으로 실제 정답률과 일치한다고 보장되지 않는다. Positive T는 분포의 확신을 바꾸지만 오답·표시 순서 편향을 직접 고치지 않는다.

분야·감정은 보통 nominal, 난이도·품질 등급은 ordinal이다. 등급의 의미상 순서와 화면 표시 위치를 구분하고 score는 argmax 외에 기대값 Σ_i v_i p_i도 평가한다. [Ordinal 편향 연구](../papers/ordinal-scale-bias.md)

## 개발·평가 분리

Train은 model weights 학습, model-selection dev는 head·checkpoint·loss 선택, calibration dev는 T 적합, test는 최종 보고에 사용한다. 같은 원본 문항에서 만든 정답·분야·난이도 요청은 group 단위로 split해야 한다. Test를 본 뒤 설계를 고르면 새로운 평가셋이 필요하다.

