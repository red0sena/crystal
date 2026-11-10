# 같은 합성 데이터에서 Crystal 학습 방식 비교

[위키 홈](../README.md) · [고정 Qwen의 상태 회전 v2](state-rotation-v2.md) · [기존 Crystal 레시피](hub-g7-merges.md) · [학습 코드](../../qwen_decision/v3/crystal_state_compare.py) · [attention 대조](../../qwen_decision/v3/crystal_state_attention_control.py) · [독립 검산](../../qwen_decision/v3/crystal_state_verify.py)

**2026-11-04 완료.** 같은 데이터에서 기존 Crystal의 noncausal＋정답CE full FT를 적용했다.
학습 전후 모두 noncausal로 평가하면 직접 이미지 판정의 seen 정확도는66.11%→72.69%였지만,
질문 없는 상태의 회전＋MLP 판정 전이는48.81%→45.03%였다. 후자의 차이 구간은0을 포함한다.
상태 검색R@1은51.95%→68.88%로 올라, 검색 정렬의 개선과 판정 전이의 개선을 구별해야 한다.

[주 비교 수치](crystal-state-methods-results.json) · [정상 attention 상태 대조](crystal-state-methods-attention-results.json) · [독립 검산](crystal-state-methods-verification.json)

## 검증 질문 재점검

**이 상태 실험은 CE 학습 후 표현의 재사용성을 보는 보조 진단이다. 논문의 표현 학습을 실제 Decision
학습에 추가했을 때의 효과를 직접 비교한 실험은 아니다.** 상태 평가의 전처리·MLP 학습 알고리즘은
비교군마다 같으므로 해당 파이프라인의 성능 비교는 유효하다. 다만 Crystal이 학습한 경로는
`장면＋질문＋선택지 → native code 판정`이고, 진단 경로는 `질문 없는 장면 → 평균 pooling → PCA64 → 별도 MLP`다.
후자의 점수로 전자의 전체 능력이나 논문 방식의 학습 효과를 대변할 수 없다.

특히 별도 MLP의 unseen text 기준선이 약31%여서, 그 모듈에서 미학습 질문을 잘 푸는 능력이 먼저
확보됐다고 보기 어렵다. native text100%와 이 수치를 혼동하지 않는다.
또한 회전을 학습 뒤에 적합하는 절차 자체는 정상이다. 이번에 하지 않은 것은 그 앞단의
JEPA 표현 학습이며, 단순히 회전을 loss에 넣지 않았다는 이유로 실험이 틀렸다는 뜻은 아니다.

논문의 아이디어가 실제 판정을 개선하는지 확인하려면 동일한 실제 판정 경로·데이터·예산에서
CE만 학습한 조건과, CE에 별도로 설계한 표현 학습을 추가한 조건을 비교해야 한다.
이를 [실제 상태 경로의 CE+LeJEPA 후속 실험](jepa-decision.md)으로 별도 진행했다. 이 문서의 Crystal 결과에 포함되지 않는다. 질문 없는 상태 재사용을 목표로 한다면
그 상태가 실제 판정을 만드는 경로를 두 조건에서 공통으로 정하고, text 판정 능력부터 검증해야 한다.

## 결과1: 추론 attention을 모두 causal로 통일한 주 비교

세 LM seed와각 LM의세 MLP seed를 평균했다. 원본은 하나의 인코더와세 MLP seed다.
이전과 같은 새 장면256개·seen2,244문항·unseen828문항을 썼다. 정확도는 세 과제의 평균이다.

| 언어 모델 학습 | 회전 후 seen 정확도 | 회전 후 unseen 정확도 | text 기준선 seen | 회전 후 상태 검색R@1 |
|---|---:|---:|---:|---:|
| 원본 고정, 동일 연산 경로 재측정 | 43.28% | 29.82% | 84.04% | 43.75% |
| causal full FT | 38.41% | 29.40% | 74.59% | 52.08% |
| Crystal full FT, **상태 추출은 causal** | 37.48% | 31.67% | 70.58% | 48.70% |

주 비교의 Crystal−causal FT seen 차이는−0.93%p,95% 구간[−7.16,+6.53]%p였다.
원본 고정 대비는−5.80%p,[−10.91,−0.17]%p였다.
그러나 이 표의 Crystal은 학습과 상태 추출의 attention이 다르고, text 기준선도 낮다.
**이 표만으로 정상 운용 Crystal이나 회전 자체의 실패를 주장하지 않는다.**
두 미세조정 모두 직접 판정을 위한 학습이며, 현재의 질문 없는 pooling·PCA·별도MLP에서
원래 판정 능력이 보존되는지를 별도로 검사한 것이다.

## 결과2: Crystal의 정상 noncausal 상태 평가

같은 원본에서 attention만 noncausal로 바꾼 무학습 모델을 대조군으로 사용했다.
두 행의 상태 추출·판정 구조·데이터가 같고, Crystal의 텍스트 full FT 유무가 다르다.

| noncausal 상태 인코더 | 검색R@1 | 이미지 평균 보정만, seen | 이미지 회전 후, seen | 회전 후 seen NLL | text 기준선 seen |
|---|---:|---:|---:|---:|---:|
| 학습 전 | 51.95% | 25.90% | 48.81% | 2.845 | 76.16% |
| Crystal 학습 후 | 68.88% | 23.72% | 45.03% | 5.476 | 84.75% |

회전 후 seen 정확도 차이는 **−3.79%p,[−8.92,+1.11]%p**다. 확실한 정확도 하락으로 단정하지 않고,
추가 학습으로 전이가 개선됐다는 근거가 없다고 해석한다.
NLL 차이는 **+2.631,[+1.804,+3.382]**로 나빠졌다. NLL은 낮을수록 좋다.
text seen 기준선은 오히려 좋아졌으므로, 이 대조의 이미지 전이 정체를 text seen 정확도 하락으로 설명할 수 없다.

Crystal에서도 회전은 평균 보정만 한23.72%를45.03%로 올렸다.
따라서 **회전의 유용성과, Crystal 학습을 더했을 때의 추가 이득은 다른 비교**다.
검색R@1의16.93%p 상승도 대응 caption을 더 잘 찾았다는 뜻이며, 상태 벡터의 exact 동치를 뜻하지 않는다.
학습 후 대응 cosine도 .6867로 잔차가 남는다.

미학습 질문 조합의 회전 후 정확도는30.76%→28.65%였다.
이 경로의 text 기준선은30.66%→31.68%로, 사전에 둔70% 기준을 모두 넘지 못했다.
그 조합의 판정 전이 실패를 회전 하나의 책임으로 분해할 수는 없다.

## 결과3: 질문과 선택지를 함께 읽는 native code 판정

아래는 회전이나 별도MLP를 쓰지 않는 기존 Decision 경로다. 각 모델의 정상 attention을 썼다.
답 후보와 gold는 상태 실험과 동일하다. 모든 미세조정의 native text 정확도는 seen·unseen 모두100%였다.

| 모델·추론 attention | 이미지 seen 정확도 | 이미지 unseen 정확도 | 이미지 seen NLL | 이미지 unseen NLL |
|---|---:|---:|---:|---:|
| 원본 causal | 65.82% | 55.21% | .853 | 1.059 |
| causal full FT | 74.56% | 62.60% | 3.586 | 6.015 |
| 원본, noncausal 전환만 | 66.11% | 56.75% | .826 | .996 |
| Crystal full FT, noncausal | 72.69% | 57.31% | 2.955 | 5.835 |

Crystal의 학습된 질문 유형에 대한 직접 이미지 판정 정확도는 올랐다.
반면 미학습 질문 조합은 평균 차이+.56%p였고, 두 그룹의 NLL은 나빠졌다.
정확도 향상을 확률 품질까지 좋아진 것으로 해석하지 않는다.
native 구간은 별도 사후 분석으로 seed·장면을 짝지어5,000회 계산해 검산JSON에 보존했다.

- Crystal−동일 noncausal 무학습 기준선: seen 정확도 **+6.58%p,[+2.93,+11.05]%p**,
  unseen **+.56%p,[−4.22,+5.60]%p**.
- 같은 대조의 NLL 차이는 seen **+2.129,[+1.835,+2.487]**, unseen **+4.840,[+4.161,+5.676]**였다.
- Crystal−causal FT의 native unseen 정확도 차이는 **−5.29%p,[−9.00,−1.51]%p**였다.
  이 native 비교는 사후 구간이며, 한 합성셋·한 pass의 결과다. 학습법의 일반적인 순위로 확장하지 않는다.

## 이 결과를 논문과 연결하면

정말로 모든 상태에서 `R z_I = z_T`라면 같은 질문과 같은 결정 함수의 출력도 같다.
이번에는 전역 least-squares 회전을 적합했고, 잔차와 검색 오류가 남았다.
가장 가까운 caption을 맞히는 조건이 판정 함수의 모든 경계·확률까지 보존하는 조건은 아니다.

Crystal의 정답CE는 질문·선택지를 읽은 **native 판정**을 학습한다.
질문 없는 평균 pooling·축소 공간·별도MLP에서도 그 능력을 그대로 꺼낼 수 있도록
직접 제약한 목적함수가 아니다. 동시에 이번 결과만으로 정보가 사라졌다고 단정할 수도 없다.
pooling, PCA, 질문 표현, 판정 모듈, 모달리티 간 잔차의 기여를 분리하지 않았다.

원 논문의 JEPA population optimum·공통 positive-pair kernel·injective view 조건을 이번CE 학습으로
만든 것은 아니다. 이 비교는 그 정리의 재현이나 반증이 아니라, 같은 합성 분포에서의 실제 적용 결과다.

## 무엇을 같게 두었나

- 이전 `plain-qwen-v2`의 장면 명세·질문 명세·test 문항 파일을 그대로 읽는다.
  세 물체 장면1,536개, train1,024 / validation256 / test256이다.
- 판정 학습 문항8,922개, validation2,251개, test3,072개다.
  의미 질문6개를 train·validation에서 제외하는 규칙과 정답·후보도 그대로다.
- LM 미세조정은 **텍스트 장면＋seen 질문**만 본다. 이미지 gold, 미학습 질문 gold, 외부 학습 데이터는 추가하지 않는다.
- 고정 상태 평가에는 이전과 같은 질문 없는 content pooling, train 전용 평균·공통PCA64,
  직교 Procrustes, text-only 조건부MLP와3개 head seed,20개 잘못된 대응 대조를 사용한다.
  MLP는 최대160epoch, validation seen NLL best, patience20을 그대로 적용했다.
- LM seed271·283·307은 두 학습법에서 짝지었다. 각 step의 문항·선택지·답 코드·gold를 해시로 비교한다.

## 학습법과 추론 조건을 분리한다

| LM 학습 | 업데이트 | 학습 attention | 주 상태 평가 attention |
|---|---|---|---|
| 원본 고정 | 없음 | 해당 없음 | causal |
| 일반 미세조정 | 언어 백본 전체와 tied embedding/head | causal | causal |
| Crystal 방식 미세조정 | 동일한 파라미터 | full-attention6층만 noncausal, DeltaNet18층은 causal | causal |

주 비교는 **상태 추출 때의 attention까지 causal로 같게 둔다.** 따라서 모델의 학습된 가중치와
그로부터 얻은 특징만 달라진다. 다만 Crystal의 학습/상태 평가 attention이 달라지는 조건이므로,
이 점수만으로 원래 Crystal 운용 방식의 좋고 나쁨을 판단하지 않는다.

이를 보완하려고 test를 열기 전에 별도 계획을 고정했다.
**학습 전 noncausal 모델과 Crystal 미세조정 후 noncausal 모델**에도 같은 상태 평가를 적용한다.
또한 native code head의 텍스트·이미지 판정은 각 레시피의 정상 attention으로 따로 측정한다.
native 판정은 질문·후보를 입력에 포함하며, 질문 없는 상태의 회전·MLP 전이와는 다른 지표다.

**기준선의 수치 연산도 맞췄다.** 기존 v2의 캐시 특징은 기본 HF 연산으로 추출했고, 학습 후 평가에는
FLA convolution과 compiled elementwise 연산이 활성화된다. test 평가를 시작하기 전에 이 차이를
발견해, 원본 causal 모델도 새 모델과 동일한 추론 경로로 재추출·평가하도록 보완 계획을 고정했다.
최종 주 비교는 이 기준선을 쓴다. 기존 캐시를 쓴 집계는 `report-cached-baseline.json`에 보존하고,
기준선 변화량도 최종 보고서와 검산에 남긴다. Crystal−일반 미세조정 비교는 처음부터 연산 경로가 같았다.

## 이번에 적용한 Crystal의 범위

기존 [g7](hub-g7-merges.md)의 핵심은 noncausal full FT와 정답 code CE였다.
이번에도 원래 Qwen3.5-0.8B revision `2fc06364715b967f1860aea9cf38778875588b17`에서 출발하고,
비전 타워는 고정한 채 언어 파라미터752,393,024개를 학습한다. 생성은 호출하지 않는다.

배포된 `crystal-0.8b` 가중치, 기존 g6/g7의 약100만 행, WiSE80 가중치 병합은 가져오지 않았다.
그것들을 가져오면 학습 데이터와 이전 학습 이력이 함께 달라지기 때문이다.
가변255개 후보를 위한 외부 distractor padding도 하지 않는다. 이번 문항의 원래4~6개 후보만 사용한다.
선택지 순열과20% code shuffle·20% code sampling은 두 미세조정에 똑같이 적용한다.

- 같은8,922문항을 **정확히 한 번**, effective batch16, microbatch8로 학습한다.558 update다.
- FP32-master AdamW, beta(.9,.95), 행렬 weight decay .01, 나머지0, clip1, LR1e-5.
- 첫6% update는 warmup, 이후 cosine으로 초기LR의10%까지 내린다.
- 세 과제의 몫을 맞춘 hard-gold CE를 표시된 답 코드 위에서 계산한다.
- 한 pass의 최종 checkpoint를 사용한다. validation은 seen 텍스트 NLL을 관찰하는 용도다.
  긴 학습의 수렴이나 배포 Crystal 레시피 전체를 재현한 실험은 아니다.

주 비교는 Crystal−일반 미세조정과 Crystal−원본 고정의 회전 후 seen 과제평균 정확도다.
텍스트 기준선, 회전 없는 이미지, heldout 질문, NLL, 검색, native 판정도 모두 보존한다.
미학습 질문의 text 기준선이 낮으면 이미지 전이 실패를 회전만의 책임으로 돌리지 않는다.

## 불확실성과 해석 범위

LM seed3개, 각 LM에서 동일한 MLP seed3개, test 장면256개를 **교차한 짝지은 bootstrap**5,000회로 재표집한다.
한 장면의 질문12개는 같이 움직인다. 원본 고정 모델은 하나의 인코더이며, 이를3개의 독립 LM fit처럼
취급하지 않는다. head seed를 LM 전체에서 공유해 재표집하므로 baseline 반복에 따른 허위 표본 수 증가를 피한다.
분할을 다시 뽑거나 회전을 별도 표본에서 재적합한 불확실성까지 포함하지는 않는다.
구간은 다중비교 보정하지 않은 개별95% 구간이다. seed3개로 학습의 모든 변동성을 안다고 주장하지 않는다.

같은 추출·판정 학습 알고리즘을 적용하더라도 인코더가 바뀌면 이미지·텍스트·질문 특징과
적합된 PCA·회전·MLP 가중치가 함께 바뀐다. 전체 파이프라인의 학습법 비교이며,
정확도 변화를 회전 행렬이나 질문 표현 한 요소만의 효과로 분해한 실험은 아니다.

이미 이전 답변에서 본 동일한 test를 후속 실험에 다시 사용했다. 이번 비교는 **탐색적 후속 실험**이며,
새 확인셋에서 독립 검증했다고 부르지 않는다. 자연 이미지·새 도메인·JEPA 학습의 결과도 아니다.

## 증거와 재현

실행 기록은 `runs/rotation-state/crystal-method-v1/`에 보존한다.
주 계획은 `plan.json`, attention 대조 계획은 `noncausal-control/plan.json`이다.
학습 전 속도·메모리 점검의 checkpoint 가중치는 폐기하고 각 fit은 다시 원본에서 시작했다.
smoke의 별도 디렉터리는 남겼으며 test 점수를 보고 학습 예산을 조정하지 않았다.

```bash
PYTHONPATH=.:external/decision-index .venv/bin/python -m unittest tests.test_v3_crystal_state_compare tests.test_v3_state_rotation
PYTHONPATH=.:external/decision-index .venv/bin/python -m qwen_decision.v3.crystal_state_compare freeze
PYTHONPATH=.:external/decision-index .venv/bin/python -m qwen_decision.v3.crystal_state_attention_control freeze
PYTHONPATH=.:external/decision-index .venv/bin/python -m qwen_decision.v3.crystal_state_compare run
PYTHONPATH=.:external/decision-index .venv/bin/python -m qwen_decision.v3.crystal_state_attention_control run
PYTHONPATH=.:external/decision-index .venv/bin/python -m qwen_decision.v3.crystal_state_verify
```

이미 존재하는 결과 폴더에 `freeze`를 다시 호출하지 않는다. 미완료 fit은 원본에서 다시 학습하고,
`fit.json`이 있는 완료 fit은 건너뛴다. 검산은 상태 MLP의 module forward나 원 gold·집계 helper를 호출하지 않는다.

## 완료 후 검산과 보존 상태

- 장면 명세에서 test3,072개 gold를 독립적으로 다시 도출했다.
- 11개 인코더 조건의 MLP33개·주요 입력4조건, 총405,504개 판정을 float64 행렬 연산과
  GELU 수식으로 재생했다. 장면별 정답 수 불일치는0이었다.
- native 텍스트·이미지49,152개 logits도 별도로 정규화하고 gold와 집계를 검산했다.
  정답 수 불일치는0이었다. 장면·과제 NLL 합의 최대 차이는 상태 경로 `4.89e-5`, native `2.23e-6`였다.
- 주 비교와 noncausal 대조의 회전 판정 신뢰구간을 인덱스 재표집 대신 multiplicity weight로
  다시 계산해 seen·unseen 정확도·NLL 구간이 일치함을 확인했다.
- 세 seed 각각558개, 총1,674쌍의 학습 입력 해시가 일치했다. 원 데이터·원 모델·고정한 학습 소스 해시와
  저장 checkpoint 해시도 검증했다. 관련 테스트7개가 통과했다.
- 개별 상태 보고서의 `qwen_trainable_parameters: 0`은 **특징 추출 시점**의 값이다.
  앞서 업데이트한 언어 파라미터752,393,024개는 각 `fit.json`에 별도로 기록돼 있다.
- 실제 실행한 대조 코드 사본은 `noncausal-control/source-at-execution.py`에 보존한다.
  실행 완료 뒤 새 디렉터리의 `freeze`가 선택적인 이전 계획 파일을 요구하던 부분만 고쳤다.
  `runner-maintenance.json`과 검산에서 `freeze`를 제외한 AST가 동일함을 확인했다.
- 원 run은 gitignore 대상이다. 위키의 결과·대조·검산JSON3개는 원본의 바이트 단위 사본이다.
  체크섬 목록은 계속 바뀔 수 있는 로그 파일을 제외해 만든다.
