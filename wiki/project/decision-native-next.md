# 후보별 판정 Query: 마지막 층의 연산 재배치

**실제로 학습했지만 정확도 개선은 확인하지 못했다.** 새 확인셋 3,394문항/35출처에서 동일 학습 기본 구조 70.70%, 후보별 Query 70.75%, 차이 +0.046%p, 문서 bootstrap 95% 구간 [−0.328,+0.412]%p였다. 단일 Query도 70.72%로 기본 구조와 거의 같았다. 아래에 구조 가설·학습 전 검증·통제 학습 결과를 구분해 기록한다.

## 제안

기존 모델은 24번째 마지막 full-attention 층에서도 모든 입력 토큰의 Q·attention 출력·FFN을 계산한 뒤, 마지막 실제 prompt 토큰 하나만 읽는다. 이 층의 attention K/V는 층에 들어온 표현에서 계산되며 FFN은 토큰별 연산이다. 따라서 마지막 판정 표현만 필요할 때 다른 토큰의 최종 Q·출력 projection·FFN을 생략할 수 있다. 앞선 23개 hybrid 층과 마지막 층의 전체 K/V 메모리는 유지한다.

여기에 선택지마다 판정 Query를 하나씩 둔다. 각 Query는 같은 전체 메모리를 보되, 질문과 그 선택지에 맞는 근거를 다르게 읽도록 학습한다. 같은 마지막 Transformer의 attention과 FFN을 거친 후보별 상태에서 기존 LM head의 해당 코드 행으로 점수를 낸다. 추가 모델, 후보별 전체 백본 재호출, 순서별 예측 평균은 없다.

```text
입력 → 기존 23개 층 → 전체 토큰 K/V 메모리
                         ↑         ↑         ↑
                     후보 A Q  후보 B Q  후보 C Q
                         ↓         ↓         ↓
                  기존 마지막 attention·FFN을 각 Query에 적용
                         ↓         ↓         ↓
                    기존 코드 A/B/C 행으로 개별 점수
```

기존 Pointer는 이미 나온 최종 벡터들의 유사도로 점수를 낸다. 이 구조는 후보마다 **지문의 어떤 토큰을 읽을지**를 마지막 attention 안에서 바꾼다. 마지막 판정 벡터 하나가 수학적으로 표현력이 부족하다는 주장은 아니다. 후보별 근거 선택을 명시한 학습 방식이 유리할 수 있다는 가설이다.

초기 Query는 원래 판정 위치의 상태를 그대로 복사한다. 후보·질문을 결합하는 작은 adapter만 0 출력으로 초기화한다. 원래 attention·FFN·정규화·LM 출력 가중치를 재사용한다. 후보 Query의 RoPE 위치도 원래 판정 토큰의 위치를 유지한다. 후보 내용이 Question과 비선형적으로 결합되도록 `LN(h), LN(c_i), LN(h)*LN(c_i)`를 함께 사용한다.

입력 토큰 수를 T, 후보 수를 K라 하면 마지막 층의 attention 상호작용은 T×T에서 K×T로, Q·출력 projection·FFN은 T개 토큰에서 K개 Query로 바뀐다. K/V projection은 T개를 유지한다. 전체 모델의 앞 23층은 그대로라 전체 추론이 이 비율만큼 빨라지지는 않는다. GPU kernel·배치·후보 수에 따라 실측해야 한다.

## 학습 전 검증 (2026-10-19)

기존 g4b 가중치를 고정하고 이미 관찰된 selection에서 고정 hash로 512문항/73출처를 뽑았다. 새 확인셋을 소비하거나 후보를 골라 정확도 이득을 측정한 실험이 아니다. 입력은 211,134토큰, 136배치, 후보 수 중앙값 3개였다. 실제 표본의 후보 수는 2–151개다.

| 경로 | 원래 선택과 불일치 | 준비된 GPU 입력의 3회 중앙값 |
|---|---:|---:|
| 기존 마지막 층 전체 계산 | — | 8.1799초 |
| 원래 판정 Query 1개만 계산 | 0/512 | 8.0171초 |
| 후보별 판정 Query, 새 adapter 0 출력 | 0/512 | 8.0502초 |

새 후보 경로는 이 조건에서 약 **1.59% 짧았다**. 이것은 학습 전 동일 가중치의 소규모 추론 측정이며 일반 서비스 지연시간 개선 보장은 아니다. 같은 BF16·SDPA·microbatch4/4096토큰을 사용했고, 경로별 8배치 warmup 후 실행 순서를 세 번 회전해 GPU 동기화 시간을 측정했다. Tokenization·collation·모델 로딩은 제외한다. 모든 입력은 미리 GPU에 준비했다.

실수 연산에서 원래 Query 복사는 원래 판정과 수학적으로 같다. **BF16 확률까지 bitwise 동일하지는 않았다.** Query 수가 바뀌면 행렬곱/attention의 수치 경로도 달라진다. 후보 경로의 최대 logit 차이는 0.0833, 최대 확률 차이는 0.00465였으며, 이 512문항의 argmax는 모두 같았다. 추가 adapter는 262,144개 파라미터다.

테스트 3개로 원래 경로·Query 1개·초기 후보 Query의 수치 일치, 패딩, 새 adapter와 기존 백본의 gradient, native 경로 복구를 검증했다. Q/FFN은 K행, K/V는 T행을 처리하는지도 검사했다. 실제 0.8B 검증은 hybrid DeltaNet 앞부분을 포함한다.

코드는 [decision_tail.py](../../qwen_decision/v3/decision_tail.py), 측정은 [decision_tail_probe.py](../../qwen_decision/v3/decision_tail_probe.py), 테스트는 [test_v3_decision_tail.py](../../tests/test_v3_decision_tail.py)다. 원문 수치는 [결과 JSON](decision-tail-feasibility-results.json)에 있다. 실행은 `python -m qwen_decision.v3.decision_tail_probe`. 원본 run은 `runs/decision-tail/feasibility-v1/`이다.

위 수치는 **Query를 학습하기 전** feasibility 실험의 기록이다. 이후 후보 Query의 별도 adapter·구조 manifest 저장/복원과 기존 `engine.Scorer`의 자동 복원을 구현했다. 일반 Transformers 로딩만으로는 후보 Query 계산이 복원되지 않으므로 저장된 `decision_architecture.json`과 `decision_tail.safetensors`를 함께 유지하고 이 저장소의 loader/engine을 사용해야 한다.

## 실제 학습과 새 문항 확인 (2026-10-19)

`runs/decision-tail/trained-v1/`에서 기본 구조·단일 Query·후보별 Query를 비교한다. 출발점은 같은 g4b이고 gold6 학습 풀 12,290문항/100출처에서 동일한 8,192개 입력을 512스텝×배치16으로 학습한다. 언어 백본과 tied LM head 전체를 갱신하고 vision은 고정한다. 백본 LR 3e-6, 후보 adapter LR 3e-4, rank64, AdamW FP32 master, warmup32와 cosine을 고정했다. 단일 Query에서는 사용하지 않는 adapter를 고정한다. 정답 CE 한 개만 쓰며 교사 라벨·모델/seed 평균·후보별 백본 재호출은 없다.

기본 구조 대조군은 직전 recurrent 실험의 **이미 완료한 동일 학습**을 재사용한다. 데이터·학습 설정·당시 구현·가중치를 검증하고, 새 두 구조의 모든 업데이트에서 512개 prompt 해시가 이 대조군 로그와 일치해야 진행한다. 이번에 기본 구조를 다시 학습했다고 주장하지 않는다. 업데이트/데이터 예산이 같으며 wall-clock 예산까지 같게 맞춘 실험은 아니다.

선택은 이미 관찰한 4,147문항/77출처에서 256·512스텝 중 독립적으로 정한다. 새 최종 확인은 3,394문항/35출처, 2,022문서 그룹이다. 기존 gold3 dev·fresh-v1/v2·JEVal·직전 recurrent 선택/확인 자료와 전체 gold2/3/6 학습 1,299,919행을 upstream ID·state·정규화 state·명시적 문서 연결로 대조하고 그룹 전체를 제외했다. 출처 도메인은 겹치며 의미상 유사한 paraphrase를 전부 탐지했다는 주장은 아니다.

주 비교는 후보별 Query 대 동일 학습 기본 구조 하나로 미리 고정했다. 출처별 평균 정확도 차이와 출처 내 문서 단위 paired bootstrap 5,000회의 95% 구간으로 평가했다. 단일 Query 비교는 보조 ablation이다. 실제 0.8B에서 3스텝 전체 역전파·저장/복원을 먼저 검증했고 재로딩 확률 차이는 0이었다.

세 arm 모두 선택셋에서 512스텝이 선택됐다. 선택셋 출처 평균은 기본 75.7419%, 단일 Query 75.8381%, 후보 Query 75.7200%였다. 모델/adapter/구조 파일 hash를 고정한 뒤 새 확인셋을 처음 평가했다. 후보별 Query·단일 Query 각각 512개 입력 hash가 이미 완료한 기본 대조군과 모두 일치했다.

| 새 확인셋 경로 | 출처 평균 정확도 | 문항 전체 정확도 |
|---|---:|---:|
| 동일 학습 기본 구조 | 70.6993% | 67.6783% |
| 단일 판정 Query | 70.7203% | 67.8845% |
| 후보별 조건부 Query | 70.7453% | 67.7666% |

주 비교의 차이는 **+0.04595%p**, 95% 구간 **[−0.3280,+0.4116]%p**로 0을 포함했다. 기본 구조의 오답 28개를 고치고 정답 25개를 새로 틀렸다. 단일 Query 대 기본의 차이는 +0.02100%p, 구간 [−0.1873,+0.2357]%p였다. 후보 Query 대 단일 Query는 +0.02495%p, 구간 [−0.3428,+0.3628]%p였다. 보조 두 비교에서 우위를 확인했다는 주장도 하지 않는다.

후보 Query의 학습·선택 평가·저장은 907.20초, 단일 Query는 872.22초였다. 후보 Query의 trainable parameter는 752,655,168개로 기본 대비 262,144개 추가이며 최대 CUDA allocation은 13.894GB였다. 이는 고정 update/데이터 예산의 한 seed 실험으로, 이 구조의 모든 학습률·학습량이 실패한다거나 원래 Transformer가 최적이라는 증명은 아니다. 현재 설정에서 후보별 근거 읽기를 명시해도 정확도가 뚜렷하게 좋아지지는 않았다.

학습/평가 코드는 [decision_tail_experiment.py](../../qwen_decision/v3/decision_tail_experiment.py), 결과 원문은 `runs/decision-tail/trained-v1/results.json`, 선택된 후보 모델은 `runs/decision-tail/trained-v1/candidates/step-512/`다. 추론은 `engine.Scorer`로 해당 경로를 읽거나 `decision_tail.load`와 `decision_tail.logits(..., mode="candidates")`를 사용한다. 단일 Query 체크포인트를 직접 호출할 때는 `mode="single"`을 지정한다. 원래 probe와 이번 fit의 source snapshot·원시 입력·문항별 확률·각 체크포인트를 모두 보존했다.

### 학습 후 비용과 실제 엔진 검증

고정 selection 512문항/73출처, 211,134토큰/136배치에서 준비된 GPU 입력의 세 번 중앙값을 비교했다. 후보 Query는 기본 대비 약 **1.30% 짧았지만**, 토큰화·선택지 offset·padding·전송·확률 변환을 포함하는 실제 `score_many`는 기본 대비 약 **0.85% 길었다**. 모델은 별도로 로딩해 각각 warmup 후 측정했으며 로딩 시간은 제외했다.

추가로 후보 체크포인트 하나를 고정하고, 전체 마지막 층 계산과 후보 Query 계산을 같은 512개 raw 요청에서 번갈아 세 번 실행했다. 이 비교는 비용만 측정하며 전체 층 경로를 정확도 후보로 다시 고른 실험이 아니다. 중앙값은 원래 계산 **8.5873초**, 후보 Query **8.6548초**, 약 **0.79% 느렸다**. GPU 일부 계산 절약이 입력 처리 비용까지 포함한 속도 개선으로 이어지지는 않았다. 이 환경·표본의 작은 차이이며 일반 서비스 지연시간에 대한 보장은 아니다.

저장된 후보의 전체 선택셋 4,147문항 재평가에서 예측 불일치0, 최대 확률 차이0이었다. 기존 엔진의 독립 로딩128문항도 예측 불일치0, 최대 확률 차이1.53e-7이었다. 35배치에서 백본이 각각 한 번, 내부 블록24개가 실행됐다. 마지막 Q는 패딩된 후보 수 K행, K/V는 prompt 길이 T행을 처리했다. 선택셋128문항에서 adapter를 0으로 지우면 예측1개와 확률 최대.02447이 달라져 학습한 후보 경로가 실제 작동함도 확인했다. 관련 테스트51개와 컴파일·diff whitespace 검사를 통과했다.

**이번 설정에서는 정확도·전체 호출 비용 양쪽에서 기본 구조보다 나은 결과를 얻지 못했다.** 수치·구간·입력/가중치 hash·고정 source·학습/실전 검증은 [전체 결과 JSON](decision-tail-trained-results.json)에 있다. 비용/엔진 검증 코드는 [decision_tail_audit.py](../../qwen_decision/v3/decision_tail_audit.py)다.

## 문헌과 다른 후보의 판단

- [DUMA, AAAI 2021](https://arxiv.org/abs/2001.09415): passage·question·answer의 관계를 co-attention으로 모델링해 MRC 성능을 비교했다. 후보/질문에 따른 근거 읽기의 근거다. 여기 제안한 native 마지막 층 재사용이나 속도 측정의 근거로 쓰지는 않는다.
- [Poly-encoders, ICLR 2020](https://arxiv.org/abs/1905.01969): 후보와 문맥의 interaction을 비용과 정확도의 절충으로 다룬다. 여기서는 새 encoder나 저장한 몇 개 코드로 문맥을 바꾸는 대신 기존 공동 인코딩과 전체 토큰 메모리를 유지한다. 이 조합이 새 논문 수준의 독창성이나 정확도 이득을 갖는다고 주장하지 않는다.
- [Jeva, 2026-10-13](https://arxiv.org/abs/2610.18773): 요청·후보를 독립 인코딩하고 후보 메모리로 작은 decision state를 갱신한다. 후보 캐시와 명시적 null을 포함한 본격적인 별도 설계다. 논문은 4B·단계별 대규모 학습과 일부 teacher 확률을 사용하므로 우리 0.8B의 작은 비교에서 같은 개선을 기대할 근거는 아니다. 특히 coverage gate 하나만 더하면 `p_i=ρq_i`여서 기존 후보 간 argmax는 그대로다. 닫힌 choice 정확도의 단독 해법으로 보지 않는다. 이는 Jeva의 전체 candidate-conditioned state 구조에 대한 부정이 아니다.
- [Candidate-Independent Block-Causal Attention, 2026-10](https://arxiv.org/abs/2610.01601): 순서 민감도를 구조적으로 줄이는 방향이다. 우리 기록의 순서 효과가 작았고, Qwen3.5의 DeltaNet은 attention mask만으로 후보 간 상태가 분리되지 않으므로 현재 첫 우선순위로 삼지 않는다.
- [Sparse Token Routing, 2026-08](https://arxiv.org/abs/2608.20632): 작은 모델/라우팅이 긴 독해에서 한계를 보이고 DistilBERT보다 원시 정확도가 낮았다는 음성 결과도 보고한다. 토큰을 중요하지 않다고 추정해 버리는 방법은 이 설계의 초기 단계에 넣지 않는다. 여기서는 전체 K/V 근거를 보존한다.

## 제안 당시 정한 정확도 실험 기준

같은 초기 checkpoint·정답 데이터·업데이트 예산으로 기존 마지막 층, 판정 Query 1개 경로, 후보별 조건부 Query를 비교한다. Adapter만 보지 말고 앞선 백본 적응도 동일하게 허용한다. 과거 Pointer/작은 readout 실패를 새 이름으로 반복하지 않도록 **원래 pretrained 최종 attention·FFN·LM head를 재사용하고 후보별 token-level evidence를 실제로 바꾸는가**를 검증한다.

기존 8,823문항은 이제 관찰된 자료라 새 후보를 고르는 확인셋으로 재사용하지 않는다. 후보별 attention을 시각적으로 그럴듯하다고 성능 근거로 삼지 않는다. 새로 분리한 문서에서 정확도와 동기화된 실제 지연시간을 함께 평가해 기존 구조의 정확도–비용 관계를 넘는지를 판단해야 한다.
