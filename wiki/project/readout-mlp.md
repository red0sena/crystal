# LM Head 대체 readout·MLP 비교 (2026-10-19)

[위키 홈](../README.md) · [이전 구조 검색](architecture-search.md)

LM Head도 hidden vector를 출력 점수로 바꾸는 선형 readout이다. 기존 v3는 전체 vocabulary를 계산하지 않고
LM Head의 답 코드 255개 행만 읽는다. 이번에는 이 출력 가중치를 독립 head로 바꾸거나, 선택지별 contextual
벡터에서 직접 점수 하나를 얻거나, hidden layer 하나를 추가해 비선형 점수화를 학습한다.

## 비교하는 head

backbone의 마지막 정규화 후 결정 벡터를 h, 선택지 끝 벡터를 o_i라고 한다. d=1,024, 중간 폭 m=512다.
모든 head는 하나의 backbone forward에서 점수를 얻으며 텍스트를 생성하지 않는다.

| 이름 | 점수 계산 | 초기화·학습 |
|---|---|---|
| code 대조군 | W_code h | 기존 255개 행을 FP32 독립 복사해 학습. 입력 임베딩과 묶지 않는다. |
| linear | W_new LN(h) | vocabulary와 무관한 새 255행 선형 head를 무작위 초기화해 학습. 새 mlp와 같은 입력 정규화. |
| mlp | W_2 GELU(W_1 LN(h)+b_1) | d→512→255, hidden layer 하나. 기존 LM 출력 가중치를 사용하지 않는다. |
| warm_mlp | W_out [h + W_down GELU(W_up LN(h)+b_up)] | 512폭의 residual MLP와 출력 W_out을 **함께** 학습. W_down=0, W_out=W_code로 시작해 초기 점수는 기존과 정확히 같다. |
| option_linear | wᵀ LN(o_i) | 모든 선택지에 동일한 1,024→1 scorer. LM 출력 행을 사용하지 않는다. |
| option_mlp | w_2ᵀ GELU(W_1 LN(o_i)+b_1) | 모든 선택지에 동일한 1,024→512→1 MLP. LM 출력 행을 사용하지 않는다. |

코드 head는 현재 표시된 코드 행만 gather하고, 후보 head는 표시된 선택지 벡터의 순서대로 점수를 낸다.
padding은 점수에서 제외하고, 정답 분포의 순서를 유지한다. 선택지 scorer 자체의 순열 등변성은 검사하되,
backbone 입력에 이미 코드·순서 정보가 있으므로 전체 모델이 순서 불변이라고 주장하지 않는다.

이전 `architecture.Readout('ffn')`은 128폭 residual MLP만 학습하고 **출력 행을 고정**했다.
이번 warm_mlp는 512폭이고 출력 행도 함께 학습한다. 새 MLP로 출력 전체를 대체하는 mlp 및
선택지 벡터만 점수화하는 option_mlp도 이번에 별도로 구현했다.

## 사전 비교 규칙

- 시작점은 같은 `runs/v3/g4b-gold3-pad0/best`다. backbone 구조·입력 prompt는 유지한다.
- gold3 train 원문 앞 32,768개를 읽어 1,536토큰 이하 문항을 사용한다. 자르지 않고 제외한다.
- selection/새 confirmation의 원문 ID·state·item·document key와 겹치는 학습 행을 제외한다.
- 기존 selection 1,474문항을 동일 토큰·target인지 확인하고 재사용한다.
- 확인셋은 원래 gold3-dev에서 이전 architecture selection/confirmation, Pointer dev, 입력 실험 confirmation의
  **문서 그룹 전체**를 제외한다. 공백 정규화한 동일 state도 같은 그룹으로 묶는다. 남은 문항에서 출처당 최대 64개를 고정 추출한다.
- 원래 g4b는 과거 gold3-dev 전체로 early stopping했다. 이 확인셋은 이번 readout 검색과 과거 이 채팅의 실험에서
  쓰지 않은 문서이며, 완전히 새로운 benchmark라고 부르지 않는다.
- 학습·selection·confirmation manifest와 데이터 hash를 보존한다. 신규 teacher label/rationale는 없다.

실제 학습 표본은 **30,548개/48출처**, selection은 **1,474개/47출처**, confirmation은
**2,027개/34출처/1,967문서 그룹**이다. 새 confirmation과 과거 노출 문항의 공유 key는 0개였고,
이번 학습 표본과 selection/confirmation의 공유 key도 없었다.
confirmation은 남아 있는 출처만 포함하므로 전체 47출처나 JEVal을 대표하는 점수로 부르지 않는다.

초기 confirmation은 2,028개였다. 별도 CPU 감사로 **원래 학습 파일 전체 399,978행**까지 검사하자,
OpenBookQA의 같은 state가 학습 4행과 공유되는 확인 문항 1개를 발견했다.
**확인 점수를 보기 전에** 해당 문서 그룹 전체(1문항)를 제외하고 데이터 hash와 manifest를 갱신했다.
삭제 전 파일, 감사 결과와 변경 이유·ID·전후 hash는 run 디렉터리에 보존한다.
남은 confirmation은 원문 ID·전체 state·공백 정규화 state·item·명시된 document key 기준으로
원래 전체 학습 자료와의 공유가 0개다. 의미 수준의 근사 중복 전부를 검사했다고 주장하지 않는다.

head parameter 수는 code/linear 261,120, mlp 655,360, warm_mlp 1,310,208,
option_linear 1,024, option_mlp 525,312다. backbone parameter 수는 변경하지 않는다.

### 1단계: backbone 고정

같은 forward에서 결정 벡터·선택지 벡터를 추출해 저장한다. 후보 벡터는 ragged 형식으로 보존해
전체 데이터 N×255×1,024 텐서를 만들지 않는다. 저장한 같은 특징에서 모든 head를 학습한다.

- head마다 학습률 2개 × seed 41/73, 총 24개 설정·seed 조합.
- batch 128 × 1,024 update, 같은 seed의 예제 draw 순서. 같은 예제 수·update 수이며 FLOPs까지 같지는 않다.
- code/warm_mlp: 1e-5, 1e-4; linear: .001, .003; mlp/option_mlp: .0001, .001; option_linear: .001, .01.
- AdamW, weight decay 0, gradient norm 1. 128 update마다 selection을 평가하고 step 0도 포함해 checkpoint를 고른다.
- 동일 kind/lr의 두 seed 평균으로 선형 readout 계열 하나, MLP 계열 하나, code 대조군의 학습률을 고정한다. 가장 좋은 seed만 선택하지 않는다.
- 무작위 초기화한 head의 step 0은 자체 초기 점수로 평가한다. 이를 기존 모델의 점수로 대체하지 않는다.
- 고정 특징 단계는 입력·코드 augmentation 없이 같은 rendered 문항을 사용한다.

### 2단계: 마지막 backbone 두 층도 학습

고정한 선형 readout, MLP, code 대조군을 두 seed 모두 추가 학습한다(총 6개 run).
앞의 22층과 임베딩·최종 norm은 고정하고 마지막 두 층과 head를 학습한다.

- 동일 draw의 batch 16 × 128 update. backbone 학습률 1e-6, head는 1단계에서 고정한 학습률.
- BF16 backbone은 기존 FP32 master AdamW를 사용한다. head도 FP32로 학습한다.
- 32 update마다 selection을 평가하고 step 0을 포함한다. 선택한 head·마지막 두 층을 별도 파일로 저장한다.
- 1단계 cache 점수와 실제 backbone 실행의 예측 일치 여부를 학습 전에 검사한다.
- 확인 전에 모델 파일 hash와 두 seed·두 학습 단계를 모두 고정한다.

### 확인

기본 모델, head-only의 선형 readout·MLP·대조군, 두 층도 학습한 선형 readout·MLP·대조군을 같은 확인 문항에서 비교한다.
두 seed 결과를 모두 기록하며, paired 문서 bootstrap으로 기존 모델과 seed가 같은 학습 대조군 대비 차이를 계산한다.
개선 판정은 두 비교의 95% 구간 하한이 모두 0보다 클 때다. confirmation을 본 뒤 family·학습률·seed를 바꾸지 않는다.
이번 실험에서 JEVal restricted를 추가 선택용으로 사용하지 않는다.

## 실행·구현

```bash
.venv/bin/python -m qwen_decision.v3.readout_experiment prepare
.venv/bin/python -m qwen_decision.v3.readout_audit
.venv/bin/python -m qwen_decision.v3.readout_experiment screen
.venv/bin/python -m qwen_decision.v3.readout_experiment tune
.venv/bin/python -m qwen_decision.v3.readout_experiment confirm
.venv/bin/python -m unittest tests.test_v3_readout -v
```

`qwen_decision/v3/readout.py`는 head, ragged batch 및 실제 backbone forward 경로다.
`readout_experiment.py`는 분할·학습·고정·평가를 담당한다. 기본 serving 경로는 변경하지 않는다.
원문별 예측·loss log·head·추가 학습한 두 층·모델 hash는 `runs/readout/search-v1/`에 보존한다.
분할 재현에는 앞선 실험들의 selection/노출 ID 파일도 필요하므로 해당 run 디렉터리를 함께 보관한다.

초기 code/warm MLP의 원래 점수 일치, 모든 head의 gradient, 선택지 mask·정렬,
문서 묶기, 실제 작은 Qwen의 padding·한 번의 forward·마지막 층 gradient를 포함한 테스트 6개가 통과했다.

## 1단계 결과: head만 학습

24개 설정·seed 조합을 완료했다. 각 종류에서 선택용 두 seed 평균이 가장 높은 학습률의 결과다.
아래는 **selection**이며 별도 확인 결과가 아니다.

| 방법 | 선택한 학습률 | 두 seed 평균 정확도 |
|---|---:|---:|
| 기존 동결 모델 | — | 78.82% |
| code 추가 학습 대조군 | .0001 | **79.12%** |
| 새 선형 readout | .001 | 78.56% |
| 새 MLP | .0001 | 78.70% |
| 기존 출력에서 시작한 MLP | .00001 | **79.05%** |
| 선택지별 선형 readout | .001 | 78.39% |
| 선택지별 MLP | .0001 | 78.46% |

`screen-frozen.json`에 새 선형 linear(.001), warm_mlp(.00001), code 대조군(.0001)을 고정했다.
각각 두 seed를 마지막 두 층과 함께 학습한다. MLP의 작은 selection 상승은 코드 head를 단순 추가 학습한
대조군보다 크지 않았다. 새 head의 무작위 초기화는 pretrained 출력층의 지식을 버리므로,
같은 예산의 이 결과만으로 충분히 학습한 새 readout의 상한을 판단하지 않는다.

## 2단계 결과: 마지막 두 층도 학습

6개 run을 모두 완료했다. 각 run은 마지막 두 층의 39,908,000 parameter와 head를 실제로 학습했다.
선택용 점수로 고른 checkpoint는 아래와 같다. step 0은 head-only 상태다.

| 고정 계열 | seed 41: 선택 step / 정확도 | seed 73: 선택 step / 정확도 |
|---|---:|---:|
| 새 선형 readout | 0 / 78.75% | 0 / 78.36% |
| warm MLP | 0 / 79.15% | 64 / 79.02% |
| code 대조군 | 0 / 79.15% | 0 / 79.09% |

6개 중 5개는 추가 학습 전이 최선이었다. MLP의 seed 73에서만 한 문항 정도의 selection 상승이 있었다.
이는 충분한 전체 backbone 미세조정 실험이 아니라, 마지막 두 층에 제한한 128 update 비교다.
저장한 18개 head/tail 파일의 hash를 confirmation 전에 고정했다.
cache에서 학습한 head와 실제 backbone에 붙인 head의 selection 예측 불일치는 6개 run 모두 0개였다.

이번 확인용 ID도 결과를 본 이후에는 관찰한 평가 자료다. 이 CLI를 같은 조건으로 재현하는 것과,
새 구조를 고르기 위해 새로운 confirmation을 마련하는 것은 구분해야 한다. 후속 검색은 이번 ID도 노출 목록에 추가해야 한다.

## 별도 확인 결과: 기존 모델을 넘는 개선 없음

최종 confirmation은 **2,027문항/34출처/1,967그룹**이다. 아래 수치는 출처 평균 정확도를
두 seed에 걸쳐 평균한 것이며 예측 ensemble 점수가 아니다. 단계별로 같은 문항을 평가했다.

| 모델 | head-only | 마지막 두 층 추가 학습을 시도한 뒤 선택된 checkpoint |
|---|---:|---:|
| 기존 동결 모델 | **79.40%** | — |
| code 추가 학습 대조군 | 78.79% | 78.79% |
| 새 선형 readout | 77.97% | 77.97% |
| warm MLP | 79.34% | 79.28% |

MLP의 마지막 단계는 seed 41 **79.12%**, seed 73 **79.44%**였다.
기존 모델 대비 차이는 각각 **−0.29%p / +0.04%p**, paired 문서 bootstrap 95% 구간은
**[−0.83,+0.27]%p / [−0.82,+0.91]%p**다. 두 seed 모두 개선 근거가 없다.
head-only MLP의 seed 73은 79.56%(기본 대비 +0.16%p)였으나 구간이 [−0.42,+0.76]%p여서
이 경우도 개선으로 채택하지 않는다. 결과를 본 후 이 seed나 단계로 바꿔 고르지 않았다.

새 선형 readout은 두 seed 모두 기본보다 낮았고, MLP는 한 seed에서만 아주 조금 높았다.
새 계열 중 **기본 모델과 seed가 같은 학습 대조군 모두에 대해 구간 하한이 0보다 큰 모델은 없었다.**
MLP의 평균이 추가 학습 code보다 높다는 사실만으로, 동결된 기존 모델 대비 개선이라고 부르지 않는다.

이 결과는 512폭 MLP, 이번 30,548개 짧은 문맥 gold 표본과 학습 예산에 한정한다.
전체 backbone을 충분히 재학습한 MLP/readout이 일반적으로 열등하다는 결론은 아니다.
기존 모델 가중치와 serving 기본값을 유지한다. 새 JEVal restricted 평가는 하지 않았다.

### 재로딩과 파일 검증

- 마지막 두 층을 학습한 6개 모델을 파일에서 다시 구성해 selection 1,474문항을 각각 재실행했다.
  저장 직전 예측과의 불일치는 모두 **0개**였다. 선택 step 0인 모델도 실제 재구성·실행했다.
- 초기 cache/live head 예측 불일치도 6개 모두 0개였다. 모든 head는 한 번의 backbone forward 경로를 사용한다.
- 단위·작은 Qwen 통합 테스트 6개가 통과했다. `py_compile`과 `git diff --check`도 통과했다.
- base model 파일 hash, 준비된 입력 hash, 고정한 head/tail 파일 hash를 다시 확인했다.
- [작은 수치·검증 기록](readout-mlp-results.json), 원문별 예측·loss·후보 고정·감사·별도 head/tail은
  `runs/readout/search-v1/`에 보존했다. 기존 모델 파일은 수정하지 않았다.
