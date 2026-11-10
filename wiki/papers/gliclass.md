# GLiClass

[위키 홈](../README.md) · [구조 비교](../concepts/architecture-map.md)

- 논문: **GLiClass: Generalist Lightweight Model for Sequence Classification Tasks**
- 연도: 2025 · 주제: 가변 라벨 내용 / 공동 인코딩
- 원문: [alphaXiv](https://www.alphaxiv.org/abs/2508.07662) · [arXiv](https://arxiv.org/abs/2508.07662)
- 조사 기록: 2026-10-13. 아래는 공개 본문을 바탕으로 작성한 요약이다. 우리 모델에서 재현한 결과가 아니다.

## 해결하려는 문제

라벨을 고정된 출력 차원이 아닌 자연어 입력으로 표현한다. 본문–라벨 및 라벨–라벨 상호작용을 한 공동 인코딩에 반영한다.

## 논문에서 제안한 방식

Uni-encoder는 각 라벨 앞에 LABEL marker를 붙이고 본문과 함께 양방향 Encoder에 넣는다. 본문과 라벨 표현을 별도로 pooling하고 dot product 또는 공유 MLP로 점수를 얻는다. Bi-encoder·fused bi-encoder·encoder-decoder 변형도 제시한다. 층별 재가중 모듈과 보조 표현 학습도 설명한다.

## 확인한 실험 근거

DeBERTa v3와 ModernBERT 등을 사용한 zero/few-shot 분류 연구. Multi-label 처리와 PPO 학습도 다룬다. 근거: §2.1 및 논문 실험.

## 한계와 해석 범위

Joint encoder라고 해서 label 순열 대칭성을 자동으로 보장하지 않는다. 분류 연구이며 복잡한 객관식 reasoning 향상을 보장하지 않는다. main uni-encoder는 bidirectional이므로 causal Qwen에 marker만 추가해서는 동일 구조가 되지 않는다.

## 우리 Qwen에 적용한다면

**아래는 적용 제안이며 미구현이다.** 가까운 시험(g4b 위, 10-19): [Pointer head](../project/pointer-pilot.md) −5.05%p, [대체 readout·MLP](../project/readout-mlp.md)와 [잔차 Set Transformer](../project/architecture-search.md)는 개선 없음, [후보별 판정 Query](../project/decision-native-next.md) +0.05%p [−0.33, +0.41]. 이 제안 그대로는 미구현이다. 분야·감정·문의 의도를 같은 가변 후보 인터페이스로 처리하는 데 적합한 참고다. Qwen 뒤에 candidate encoder와 공유 scorer를 붙이거나 제한된 공동 비교층을 두는 실험을 제안한다.

## GLiClass v4 Edge 모델 카드 (2026-11-02 조사)

[knowledgator/gliclass-v4-edge](https://huggingface.co/knowledgator/gliclass-v4-edge)(revision `885af1a`, 10-24 갱신)의 README를 읽었다. 직접 돌리지는 않았다.

**모델**
- DeBERTa-v3-large 인코더(435M)다. 텍스트만 다루고 이미지는 못 받는다.
- 질문마다 forward를 한 번 한다. state가 본문, instructions가 prompt, 선택지 설명이 라벨이다.
- 중앙값 11ms다(RTX PRO 6000).

**보고한 점수:** Decision Index 0.3 공개 전체 skill 47.49다(d3-lite 36.93). 0.8B 이하에서 가장 높다고 한다. 비공개가 들어간 Full 점수는 보고하지 않았다.

**학습 3단계**
1. **SFT:** 분류, NLI, QA, 재순위, 도구 호출, 판정 데이터를 라벨 고르기로 바꿔 한 번에 학습했다. loss는 focal이다.
2. **RL:** PPO식이다. 행동은 라벨 선택이고, 보상은 과제 지표(F1, 정답률, recall)다. value baseline, clipped ratio, 선택적 KL·entropy 항을 쓴다.
3. **보정(calibration):** backbone을 얼리고, 작은 보정기가 본문·라벨 표현과 logit 통계를 읽어 (본문, 라벨) 쌍마다 temperature와 bias를 낸다. 따로 둔 판정 데이터에서 NLL로 맞췄다.

**우리 채점과의 관계**
- 보드 채점은 choice argmax와 noul p≥0.5다([목표와 규칙](../project/jev-index.md)). 그래서 temperature는 답을 바꾸지 않고, 쌍별 bias만 답을 바꾼다. 결국 입력에 따라 달라지는 문턱값·prior 보정이다.
- 고정 prior 보정(Calibrate Before Use)은 10-19 g4b에서 정답률을 낮췄다(−0.39~−4.69%p).
- 한 번에 하나를 고르는 문제에서 정답률 보상 RL은 기대 정답률 p(gold)를 키우는 것과 같다. CE와 실제로 다른 점은 F1·recall처럼 문항 단위로 나뉘지 않는 지표를 직접 보상할 수 있다는 것이다.
- focal은 10-31 [Loss 비교](../project/loss-screen.md)에서 짧은 문항을 −1.5점 낮췄다.

**공개 벤치마크별 비교:** GLiClass 숫자는 카드의 공개 전체 값이다. wise80과 d3-lite는 우리 공개 샘플 값이다(벤치마크당 40그룹이라 잡음이 크다). d3-lite는 샘플 39.56, 공식 36.93이다.
- **GLiClass가 크게 높은 곳:**
  - PhishNChips: 86.8 대 wise80 0.0, d3-lite 0.0
  - VAST: 59.6 대 15.0
  - POP909: 23.1 대 0.0
  - CLINC: 92.2 대 82.7
  - ESCI: 39.3 대 25.2
  - Habermas: 33.8 대 23.1
  - HellaSwag: 77.1 대 66.7
  - CLadder: 42.2 대 25.0
- **wise80이 높은 곳:**
  - RAGTruth: 75.6 대 58.6
  - When2Call: 73.3 대 49.7
  - GSM8K: 20.0 대 2.1
  - BRIGHT: 36.5 대 22.4
  - ToolRet: 61.2 대 47.9
  - ContractNLI: 77.3 대 69.6
  - BFCL: 93.5 대 87.2
- **PhishNChips는 noul 판정인데 wise80이 0점이다.** 우리 텍스트 dev의 phishing 출처는 정답률 1.0이라, 문턱값 문제라기보다 보드 문항과 우리 학습 문항의 형식이 다른 것으로 보인다(미확인).

**코드와 논문에서 확인한 것(11-02, GitHub `Knowledgator/GLiClass` main, arXiv 2508.07662)**
- **focal:** 라벨마다 따로 하는 sigmoid BCE focal이다(RetinaNet식, 함수 기본값 α 0.25, γ 2).
  - 배포한 v4 edge의 `config.json`은 `focal_loss_alpha: 0.75`, `focal_loss_gamma: -1`이다. 즉 γ 변조 없이 양성 0.75, 음성 0.25로 가중한 BCE다. 이 값이 마지막 단계의 설정인지는 모른다.
  - 그 밖의 config: label 최대 25개, `shuffle_labels: true`, 2층 scorer encoder, bge-small label 모델.
  - 라벨마다 따로 yes/no를 내는 구조에서 쉬운 음성 라벨이 대부분을 차지하는 불균형을 줄이는 용도다.
  - 우리는 softmax로 하나를 고르므로 이 불균형이 없다. 우리 softmax focal은 짧은 문항을 −1.5점 낮췄다.
- **RL(`train_rl.py`, `RLTrainer`):**
  - 라벨마다 Bernoulli로 행동을 뽑는다. 보상은 문항 단위 micro-F1이다(정답률, recall 보상도 있다).
  - baseline은 별도 value 모델이고, clip은 0.2, 같은 배치를 2~3번 다시 쓴다. 참조 모델 KL과 entropy는 기본으로 꺼져 있다.
  - **논문에는 PPO와 지도학습만 한 경우를 비교한 수치가 없다**("modest but consistent gains"라는 서술뿐이다).
  - v3 후반 학습은 GPT-4o가 만든 라벨을 썼다. 우리 정답 전용 규칙과 맞지 않는다.
- **보정기(`calibration.py`):**
  - backbone을 얼리고 MLP가 쌍마다 inverse temperature β ∈ (0, 3)와 선택적 bias를 낸다. 전역 temperature·Platt로 시작해 held-out NLL로 맞춘다.
  - 코드 기본값은 bias를 끈다. **하지만 배포한 v4 edge의 `config.json`은 `calibrator_use_bias: true`다.** 그래서 쌍마다 base rate가 옮겨 가 0.5 기준 판정과 argmax가 바뀔 수 있다. 목적은 확률 품질(ECE, Brier)이지만 답에도 영향을 준다.
- **하나를 고르는 문제에서 정답률 보상의 기대 gradient는 ∇p(gold)다.** 그래서 −p loss와 같다([loss 문헌](loss-alternatives.md)). PPO 장치는 표본 분산만 줄인다. CE와 실제로 다른 경우는 F1처럼 문항 단위로 나뉘지 않는 지표뿐이다.

**우리에게 배울 점: 보드 지표에 맞춘 학습과 평가**
- **11-04 재검산:** 현재 고정한 보드의 공식 panel 37개에서 단순 정답률형은21개다(그중 GSM8K는4/10-choice 표기). 이전 기록의17개를 정정한다. 아래와 합쳐37개다([현재 근거·집계](../project/champion-v1.md#먼저-고칠-평가)).
  - macro-F1 8개: BANKING77, CLINC, ContractNLI, ANLI, ESCI, FinEntity, VAST, NLI4CT
  - 특정 클래스 F1 2개: RAGTruth, iSarcasm
  - 리뷰별 F1 1개: ACOS
  - 모두 맞아야 정답 3개: BFCL, SATA-Bench, Home appliance
  - nDCG@10 2개: ToolRet, BRIGHT
- 우리는 CE(정답률의 대리)로 학습한다. 대리 평가인 `proxy_eval.py`와 `fresh_eval.py`도 출처별 정답률 skill만 잰다. 그래서 지표에 맞춘 개선이 있어도 대리 평가에는 보이지 않는다.
- 교사 없이 정답 라벨만으로 해 볼 수 있는 순서:
  1. 대리 평가를 보드와 같은 지표로 채점한다.
  2. 출처 안에서 클래스 균형 가중치를 준다(macro-F1의 대리).
  3. 기대 지표를 직접 미분한다. 하나 고르기는 −p, F1 과제는 배치 단위 soft-F1, 여러 항목 요청은 "모두 맞을 확률"을 쓴다. 행동이 한 단계라 확률이 다 보이므로 표본을 뽑을 필요가 없다.
- 전역 양의 temperature는 argmax·이진0.5 판정을 바꾸지 않는다. 반면 앞서 확인한 입력·라벨별 bias는 답을 바꿀 수 있으므로, **보정기 전체가 Full 순위에 거의 영향이 없다고 일반화하지 않는다.**

**학습 데이터 규모(11-02 확인)**
- **v4의 학습 데이터 규모와 구성은 공개되지 않았다.** 카드에는 과제 종류만 있다. 두 번째 인용 논문(2609.02292, SCX Router)은 다른 모델이다.
- 공개된 것은 이전 버전 데이터다(HF `knowledgator`, 모두 `text`, `all_labels`, `true_labels` 형식).

| 데이터셋 | 행 | 비고 |
|---|---:|---|
| `gliclass-v2.0` | 1,196,218 | v3 논문의 사전학습 "1.2M"과 같은 규모. 분류, 감성, NLI |
| `gliclass-v2.0-RAC` | 612,142 | 검색 보강 분류(예시가 붙은 행) |
| `gliclass-v3-logic-dataset` | 7,776 | FineWeb, CommonsenseQA, MultiNLI 문장에 **LLM이 라벨을 붙였다**. 참 1~50, 거짓 1~50개 |

- 비교하면 우리 v3-gold6는 699,960행(104개 출처), v3-gold7은 999,961행(113개 출처)이다. 모두 사람이나 프로그램이 정한 정답이다.
- `knowledgator/decision-index-results`에 v4 edge, 2b, 9b의 공개 suite 행별 결과(`results.jsonl`)가 있다. 그들 코드를 돌리지 않고도 행 단위로 우리 모델과 비교할 수 있다.

**한계**
- 학습 셋이 공개 벤치마크의 train split을 얼마나 포함하는지 밝히지 않았다. 카드 스스로 KGJevBench 일부가 학습 데이터와 겹친다고 적었다.
- OneJev처럼 공개는 높고 비공개는 낮을 수 있다(공개 42.7, Full 29.0).

## 연결 문서

- [soft-contextualized-encoder ](soft-contextualized-encoder.md)
- [architecture-map ](../concepts/architecture-map.md)
