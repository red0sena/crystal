# CE가 정말 최선인가: loss 문헌 검토 (2026-10-30)

[위키 홈](../README.md) · [Loss 비교 실험](../project/loss-screen.md) · [판정 모델 학습 레시피](decision-training-recipes.md)

[Loss 비교](../project/loss-screen.md) 1단계에서 label smoothing, focal, CE + Brier 중 CE를 이긴 것이 없었다. "CE보다 나은 loss가 있다"는 논문들이 어떤 조건에서 그런 결과를 냈는지 원문을 확인했다. 숫자는 각 논문의 표에서 가져왔고, 우리가 재현한 것이 아니다.

**결론: 우리 과제에서는 CE가 맞는 기본값이다.** CE를 이겼다는 보고는 거의 모두 우리와 다른 조건에서 나왔다. 사전 지식이 강한 모델, 생성 과제, 조정 여부가 불분명한 대조군이 그 예다. loss 쪽에 남은 여지는 정답 하나를 다르게 세는 것이 아니다. **CE에 없는 신호를 더하는 것**이다. 순서 일관성과 가린 토큰 학습이 여기에 해당한다. 2단계(10-31 완료)에서 둘 다 사전 등록 검정을 통과하지 못했다. 순서 일관성(consist)만 보고 비교에서 짧은 문항 NLL −0.023(4/5 seed)이었다.

## 1. 이론: CE가 겨냥하는 곳

- CE(log loss)는 strictly proper scoring rule이다. 기대 loss를 최소로 만드는 출력이 실제 조건부 확률 하나뿐이다.
- Brier, spherical score도 strictly proper라서 **최적점은 CE와 같다.** 그래서 이들 사이의 차이는 최적화 경로와 문항별 가중치에서만 생긴다([Shao et al., ICML 2024](https://arxiv.org/abs/2405.18906)).
- log score는 정답 확률 하나만 보는 유일한 strictly proper rule이다(local). 다른 선택지에 확률을 어떻게 나눴는지는 loss에 들어가지 않는다.
- 우리 목표 지표에 비춰 보면 다음과 같다.
  - Decision Index는 ForecastBench만 Brier로 채점하고, 나머지는 대부분 argmax 정답률이다.
  - 실제 확률을 맞히면 두 채점 모두에서 최적이므로, CE의 최적점은 지표와 어긋나지 않는다.
  - CE는 0-1 오류에 대해 consistent surrogate다([Mao, Mohri, Zhong, ICML 2023](https://arxiv.org/abs/2304.07288)).

## 2. CE를 이겼다는 보고와 그 조건

| 논문 | 무엇이 CE를 이겼나 | 조건 | 우리 과제와의 차이 |
|---|---|---|---|
| [Beyond Log Likelihood](https://arxiv.org/abs/2510.00526) (ICLR 2026) | −p, thresholded NLL | **사전 지식이 강한 모델**(학습셋 평균 정답 확률 0.76~0.81, 수학). Qwen2.5-Math-7B 평균 22.67(NLL) → 36.51(−p) | 아래 설명 참고 |
| [Strictly Proper Scoring Rules](https://arxiv.org/abs/2405.18906) (ICML 2024) | Brier, spherical로 fine-tune | 번역 생성. LLaMA-7B WMT22 EN-DE BLEU 25.42 → 29.15 | 처음부터 학습하면 log score가 가장 좋았다. 생성 과제이고, 우리는 Brier를 이미 시험해 차이가 없었다 |
| [CV Losses for LLM Fine-Tuning](https://arxiv.org/abs/2409.13641) (EMNLP 2024 Findings) | 0.6 × CE + 0.4 × focal 또는 Lovász | 3~7B base 모델 LoRA, 답 글자 생성. HellaSwag EM 47.4(CE) → 71.7(focal) | seed 수를 보고하지 않았다. CE 대조군을 따로 조정했는지 불분명하다. 시험셋 결과를 보고 loss를 골랐다. GSM8K에서는 focal이 CE보다 낮았다 |
| [DFT](https://arxiv.org/abs/2508.05629) (ICLR 2026) | 토큰 loss × 정답 확률 | 수학 추론 | 정답이 하나인 과제에서는 약하다고 저자가 직접 적었다. Natural Questions에서는 SFT 36.62, DFT 30.14로 오히려 낮았다 |
| [Generative CE](https://arxiv.org/abs/2604.06689) (2026) | batch 안에서 클래스 비율로 정규화한 CE | 비전 모델을 처음부터 학습, 2천~5천 장, 클래스 불균형 | LLM에서는 시험하지 않았다. 우리 코드(A, B, …)는 문항마다 뜻이 달라서, batch에서 클래스 비율을 세는 전제가 맞지 않는다 |
| [Label smoothing 보정](https://arxiv.org/abs/2508.00264) (ICML 2025) | label smoothing | instruction tuning에서 보정이 무너지는 것을 막는다 | 정답률을 올린다는 주장은 아니다. 우리 ls도 보정(NLL)과 정답률 둘 다 CE와 같았다 |
| [Supervised contrastive](https://arxiv.org/abs/2011.01403) (ICLR 2021) | CE + 대조 학습 항 | 문항 20~1,000개의 few-shot | 우리는 1.6만 행이다 |
| GCE, SCE ([1](https://arxiv.org/abs/1805.07836), [2](https://arxiv.org/abs/1908.06112)) | 잡음에 강한 loss | 라벨 잡음이 많을 때 | 우리 gold 라벨의 잡음률은 모른다. PEFT 자체가 라벨 잡음에 더 강하다는 2026 보고도 있다([2604.12469](https://arxiv.org/abs/2604.12469)) |

**Beyond Log Likelihood를 우리에 맞춰 보면**
- 이 논문은 "모델이 이미 얼마나 아는가"를 학습 데이터의 평균 정답 확률로 잰다. 세 구간으로 나눴다.
  - 강한 모델: 0.76~0.81이고, −p가 이긴다.
  - 중간: 0.50~0.59(의료)이고, loss 선택이 "거의 구분되지 않는다".
  - 약한 모델: 0.01~0.07이고, NLL이 이긴다.
- 학습 전 Qwen(양방향)의 selection 평균 정답 확률은 0.41~0.43이다(source 평균 0.408, 행 평균 0.433). 약한 쪽과 중간 사이, 중간에 가깝다. 이 구간에서 loss 선택이 차이를 거의 내지 않는다는 예측은 ls, brier ≈ CE와 맞는다.
- **focal은 이 논문이 시험하지 않은 방향이다.** −p는 CE보다 확신 있는 문항에 비중을 더 준다. focal은 반대로 확신 있는 문항의 비중을 줄인다. 그래서 "focal이 나쁜 것은 이 논문의 예측과 같다"는 말은 맞지 않는다. focal이 나쁜 것은 우리 실험에서 직접 나온 결과다.
- 학습 후 전체 LoRA의 평균 정답 확률은 0.62~0.67이다(seed 131). 논문 기준으로 중간에서 강한 쪽으로 넘어가는 지점이다. "NLL로 먼저 학습하고, 끝부분에서 −p로 바꾼다"는 생각은 해 볼 수 있다. 다만 논문이 시험하지 않았고(저자가 적응형 loss를 후속 과제로 남겼다), DFT는 정답이 하나인 과제에서 약했다. 그래서 우선순위는 낮다.

## 3. CE에 없는 신호를 더하는 방향

같은 정답 하나를 다르게 세는 loss는 최적점이 같다. 반면 아래 방법들은 CE가 보지 못하는 정보를 준다.

| 방법 | 근거 | 우리 실험 |
|---|---|---|
| 선택지 순서 일관성 | [PA-GRPO](https://arxiv.org/abs/2603.21016)(ACL 2026): 순서를 바꾼 묶음으로 RL. SFT 계열인 PIF 대비 7개 벤치 평균 정답률 Llama-3.1-8B 59.5 → 68.7, Qwen3-8B 70.1 → 74.7, Qwen3-32B 79.2 → 79.8(논문은 평균을 보고하지 않는다. Table 1에서 계산한 값이고, 원표와 한 칸씩 대조하지는 않았다). 일반 CE SFT나 순서 섞기 SFT 대조군은 없다. [Guda et al.](https://arxiv.org/abs/2511.21709): 순서 편향 지표를 LoRA 학습 목표로 써서 편향 −58%. [Order Independence](https://arxiv.org/abs/2503.23483): 정답과 오답 사이 margin loss | consist(two views + JSD)와 대조군 twoview. 검정은 차이 없음. consist − all(보고만 함)은 selection NLL −0.023, 4/5 seed. 벌점 없이도 두 순서의 JSD가 0.010으로 작았다 |
| 양방향 전환 전 가린 토큰 학습 | BidirLM(2604.02045), LLM2Vec | mntp: 차이 없음. 가린 토큰 loss는 1.88까지 내렸지만 답에는 옮겨지지 않았다 |
| source별 가중치 조정 (지표가 source 평균이므로) | tDRO(2408.10613), GRAPE(NeurIPS 2025) | 시험하지 않음. fine-tuning에서는 DRO 이득이 작았다는 보고가 있다([2312.03151](https://arxiv.org/abs/2312.03151)) |

- PA-GRPO의 이득은 8B 모델에서 크고 32B에서는 거의 없다. 대조군이 일반 SFT가 아니라서, 순서를 섞어 학습하는 우리 CE(이미 step마다 선택지 순서를 섞는다)보다 얼마나 나을지는 알 수 없다.
- d3-lite가 어떤 loss로 학습했는지는 공개 자료(recipes.vllm.ai, Hugging Face 카드)에서 찾지 못했다.

## 4. 다음에 시험할 만한 것 (10-31 갱신, 우선순위 순)

1. **consist를 새 seed로 다시 잰다.** consist − all을 미리 등록하고, 같은 계산량의 대조군(CE를 2 epoch)을 같이 둔다. 두 순서를 보는 팔은 계산량이 2배라서, 지금 결과로는 이득이 순서 신호에서 왔는지 계산량에서 왔는지 가를 수 없다.
2. consist가 확인되면 PA-GRPO식 묶음 학습(같은 문항의 여러 순서를 한 batch에)을 본다. 다만 RL이 아니라 gold 라벨 CE 그대로 쓴다. 이미 두 순서의 JSD가 0.010으로 작아서, 순서를 더 늘려도 이득의 상한은 작을 것이다.
3. 정답이 하나인 과제에서 확률 가중 loss(DFT, −p)를 쓰는 것은 문헌상 근거가 약하다. 시험한다면 학습 후반에만 바꾸는 형태로 한다.
4. MNTP는 우리 규모(1.6만 행)에서 효과가 없었다. GPT-1의 보조 LM loss도 큰 데이터에서만 도움이 됐다. 다시 볼 이유가 적다.
