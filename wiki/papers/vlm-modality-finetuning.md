# 텍스트는 full FT, 이미지는 LoRA? 모달리티별 학습 방식 문헌 검토 (2026-10-31)

[위키 홈](../README.md) · [Vision 기준점](../project/vision-baseline.md) · [인사이트](../project/insights.md) · [LoRA Learns Less](lora-learns-less.md)

사용자 질문: "텍스트는 full FT로, 이미지는 LoRA로 학습시키는 경우가 있나?" 그래서 VLM에서 부위·모달리티마다 학습 방식을 다르게 한 연구를 찾았다. 표의 "확인" 열은 [참고 문헌](../resources/references.md)의 표기를 따른다. 숫자는 각 논문의 값이고 우리가 재현한 것이 아니다.

**결론: 흔한 방식이다. 우리 상황(wise80이 이미 텍스트 full FT 모델)에 가장 가까운 것은 "언어 모델을 얼리고 이미지용 LoRA나 별도 파라미터만 얹기"다.**
- Phi-4-multimodal, Llama 3.2 Vision, CogVLM, InternLM-XComposer2가 이 방식이다. 모두 "텍스트 입력의 성능은 바뀌지 않는다"를 설계 이유로 든다.
- 텍스트로만 학습해도 이미지 문항이 오른다는 보고(X-Reasoner)가 있다. 우리 [Vision 기준점](../project/vision-baseline.md)의 +16.4점과 같은 방향이다.
- ViT를 얼릴지 학습할지는 문헌끼리 엇갈린다. 우리가 팔로 비교해야 한다.
- 망각은 방식보다 학습률 문제라는 2026 보고가 있다. LoRA나 작은 학습률이면 거의 없다.

## 1. 언어 모델은 그대로 두고 이미지 능력만 얹기

| 논문 | 언어 모델 | 이미지 쪽 학습 | 텍스트 성능 | 확인 |
|---|---|---|---|---|
| [Phi-4-multimodal](https://arxiv.org/abs/2503.01743) (Microsoft 2025) | 텍스트 학습을 끝낸 Phi-4-mini를 **얼림** | 비전 인코더(SigLIP-400M) + 2층 MLP projector + **decoder의 모든 linear에 vision LoRA**(370M). 음성은 별도 speech LoRA(rank 320) | "pure text inputs에서 언어 성능이 그대로"라고 적었다. 수치 비교는 없다 | 원문 |
| [Llama 3.2 Vision](https://ai.meta.com/blog/llama-3-2-connect-2024-Vision-edge-mobile-devices/) (Meta 2024) | Llama 3.1을 **얼림** | cross-attention adapter와 이미지 인코더만 학습 | Llama 3.1 텍스트 모델의 drop-in 대체라고 했다 | 검색 |
| [CogVLM](https://arxiv.org/abs/2311.03079) (2023) | **얼림** | attention과 FFN에 이미지 토큰 전용 가중치(visual expert)를 추가해 학습 | "NLP 과제 성능을 전혀 잃지 않는다" | 초록 |
| [InternLM-XComposer2](https://arxiv.org/abs/2401.16420) (2024) | 원래 가중치 유지 | **Partial LoRA**: 이미지 토큰에만 LoRA를 적용한다 | 언어 지식을 보존하려는 설계 | 초록 |
| [Image-LoRA](https://arxiv.org/abs/2512.19219) (2025) | 얼림 | 이미지 토큰 구간에서만, value 경로에만, 골라낸 head에만 LoRA를 적용한다 | GSM8K: 학습 전 25.55, Image-LoRA 25.55, 일반 LoRA 24.79 | 원문 |
| [Wings](https://arxiv.org/abs/2406.03496) (NeurIPS 2024) | 유지 | attention 옆에 시각용 learner를 먼저 붙이고, 나중에 텍스트용 learner를 토큰별 routing으로 붙인다 | 텍스트 전용 망각의 원인을 이미지 앞뒤 텍스트 사이의 attention 이동으로 봤다 | 검색 |

- **Phi-4-multimodal의 단계:** projector만 학습 → projector + 인코더 → vision LoRA를 추가하고 인코더·projector와 함께 SFT → 인코더를 얼리고 여러 장 SFT. LoRA를 어떻게 켜고 끄는지는 본문에서 찾지 못했다(초록은 "modality-specific routers"라고만 한다).
- **Image-LoRA의 한계:** 이미지 토큰 비율이 낮으면 일반 LoRA보다 약하다. ScreenSpot-Pro(Qwen2.5-VL-7B)에서 텍스트:이미지 1:5는 37.35 대 37.18로 조금 앞서지만, 1:2는 18.24 대 20.48로 뒤진다. 학습 파라미터는 252만 개에서 63만 개로 줄었다.

## 2. 원래 있던 부위는 LoRA, 새 부위는 full

[Idefics2: What matters when building VLMs?](https://arxiv.org/abs/2405.02246) (NeurIPS 2024, 원문)
- 사전학습된 ViT(SigLIP)와 LLM(Mistral-7B)은 LoRA로, 새로 만든 연결부는 full FT로 학습했다.
- 이유: 백본을 full로 풀면 "loss가 발산"했다. LoRA는 "표현력을 더하면서 학습을 안정시킨다".
- Table 3(벤치마크 4개 평균): fully autoregressive 구조에서 백본을 얼리면 60.3, LoRA면 69.5다. 본문의 "+12.9"는 표와 맞지 않는다(표 기준 +9.2).
- ViT에도 LoRA를 붙였다. 종횡비를 유지한 입력에 맞춰 보간한 위치 임베딩에 적응시키기 위해서다.

## 3. 텍스트 학습을 이미지로 옮기기

| 논문 | 무엇을 했나 | 결과 | 확인 |
|---|---|---|---|
| [X-Reasoner](https://arxiv.org/abs/2505.03981) (2025) | Qwen2.5-VL-7B를 **텍스트만으로** 학습했다(OpenThoughts-114k SFT, 수학 RL) | MMMU-Pro 38.3 → 40.4(SFT만) → 42.5(SFT + RL). 이미지 없이 풀리는 문항을 빼도 이득이 남았다(MMMU-Pro 33.4 → 36.4) | 원문 |
| [Bring Reason to Vision](https://arxiv.org/abs/2505.05464) (ICML 2025) | 추론 LLM의 변화량을 VLM의 언어 모델에 섞었다(학습 없음). 비전 타워는 그대로 | 수학 추론이 오르고 지각 과제 손해는 작았다. 지각은 앞쪽 층, 추론은 중간~뒤쪽 층에 있다 | 검색 |

- 우리 결과와 같은 방향이다. 비전 텐서가 같은 wise80이 원래 Qwen보다 vision 지수가 16.4점 높았다. 차이점도 있다. X-Reasoner는 생성형 추론이고, 우리는 한 번 forward해서 고르는 판정이다.
- 섞기 방식은 우리에게도 열려 있다. 원래 Qwen에 이미지 LoRA를 학습한 다음 wise80의 텍스트 변화량과 섞는 방법이다. 우리 `merge.py`의 `θ = base + Σ w_i · α · (θ_i − base)`와 같은 틀이다.

## 4. ViT를 얼릴까, 학습할까

| 논문 | 결과 | 확인 |
|---|---|---|
| [Prismatic VLMs](https://arxiv.org/abs/2402.07865) (ICML 2024) | 비전 백본을 full FT하면 "성능이 크게 떨어진다"(p = 0.004). 위치 찾기(RefCOCO, OCID-Ref)에서 가장 많이 떨어졌다 | 원문 §4.1 |
| [Cambrian-1](https://arxiv.org/abs/2406.16860) (NeurIPS 2024) | "ViT를 푸는 것은 넓게 이롭다". 일반·vision-centric 과제가 오르고, OCR·차트는 언어로 사전학습한 인코더에서 오른다. 지식 과제는 거의 그대로다 | 원문 |
| [Fine-tuning MLLMs Without Forgetting](https://arxiv.org/abs/2603.14493) (2026) | Qwen2.5-VL-3B에서 비전 인코더 full FT(학습률 1e-6)는 OOD 망각이 −1.4 ~ +0.5점으로 작았다 | 원문 |

- 작은 모델 쪽 참고: [SmolVLM](https://arxiv.org/abs/2504.05299)(2025, 검색)은 256M~2.2B 모델에서, 작은 언어 모델은 큰 비전 인코더에서 얻는 이득이 작다고 보고했다. 우리 0.8B에서도 ViT를 크게 바꾸는 것보다 언어 쪽 학습이 더 중요할 수 있다는 정도로만 읽는다.
- 둘이 엇갈리는 이유로 보이는 것: Prismatic은 학습 데이터가 작은 한 단계 학습이었고, Cambrian은 데이터가 크고 지각 과제를 넓게 봤다. 문헌은 데이터 규모와 학습 목표를 원인 후보로 든다.
- 우리 기준점에서 약한 곳은 지각 셋(BLINK 0.39, CV-Bench 0.60)이고, 글자 읽는 셋(InfographicVQA 0.82, KIE 0.72)은 이미 높다. Cambrian 기준으로는 ViT를 조금 학습할 이유가 있다. Prismatic 기준으로는 크게 바꾸면 위치 찾기가 위험하다. 그러니 ViT를 얼린 팔과 ViT LoRA 팔을 같이 둔다.

## 5. 망각과 과적합

[Fine-tuning MLLMs Without Forgetting Is Easier Than You Think](https://arxiv.org/abs/2603.14493) (2026, 원문). Qwen2.5-VL-3B, ImageNet VQA로 학습하고 OOD 문항으로 망각을 쟀다.

| 설정 | 학습 과제 | OOD 텍스트 + 같은 분포 이미지 | OOD 텍스트 + OOD 이미지 |
|---|---|---|---|
| LLM full FT, 학습률 1e-5 | 91.56% | −16.6점 | −33.6점 |
| LLM LoRA r8, 학습률 1e-4 | 91.08% | +0.5점 | −3.0점 |
| LLM full FT, 학습률 1e-6 | 91.37% | +1.1점 | −1.5점 |
| projector만 | 86.86% | +0.3점 | +0.1점 |

- 결론: "망각은 피할 수 없는 것이 아니라 과한 최적화에서 온다." 학습 과제 성능은 거의 같다.
- **예외가 우리에게 중요하다.** 학습한 이미지에 다른 종류의 질문을 하면, 작은 학습률로도 53.35% → 42.95%로 떨어졌다.
  - 클래스 이름을 오답 선택지로 넣으면 42.95% → 5.55%가 됐다. 질문과 상관없이 "클래스 이름을 고르는" 지름길을 배운 것이다.
  - 다양한 지시 데이터(LLaVA-665K)를 절반 정도 섞으면 대부분 고쳐졌다.
  - 분류식 학습을 하면 같은 지름길이 생길 수 있다. 그러니 이미지 학습 데이터에 질문 종류를 섞고, 텍스트 판정 데이터도 함께 넣는 것이 안전하다.
- [EMT](https://arxiv.org/abs/2309.10313) (2023, 초록): 공개 MLLM 대부분이 자기 비전 인코더보다 이미지 분류를 못했다. LLaVA를 계속 fine-tune하면 처음에는 다른 이미지 셋도 오르다가, 길어지면 비전 인코더를 얼려 둬도 일반화를 잃었다.

## 6. 분류 문제로 학습하는 것에 대해

[Why are VLMs bad at image classification?](https://arxiv.org/abs/2405.18415) (NeurIPS 2024, 초록)
- VLM이 분류를 못하는 주된 원인은 구조가 아니라 데이터다. 정보는 이미 latent에 있는데, 데이터가 충분해야 꺼내 쓸 수 있다. 성능은 학습 때 그 클래스가 얼마나 자주 나왔는지를 따라간다.
- 분류 데이터를 VLM 학습에 넣으면 분류가 오르고, 일반 능력으로도 옮겨 갔다(ImageWikiQA +11.8%).

## 7. 이미지 학습이 텍스트에 주는 영향 (10-31 추가)

| 논문 | 결과 | 확인 |
|---|---|---|
| [Training-Free Mitigation of Language Reasoning Degradation](https://arxiv.org/abs/2412.03467) (2024) | LLaVA-1.6-Mistral과 Mistral 비교. **CommonsenseQA +5.3**(0.668 → 0.722). GSM8K −6.7, HellaSwag −6.3, OpenBookQA −4.2, ARC-Easy −3.1, WinoGrande −2.5, MMLU −1.2. Vicuna 백본에서도 CommonsenseQA +9.8, GSM8K −3.3. 좋아진 CommonsenseQA 문항 60개 중 약 60%가 그림으로 보일 수 있는 지식(장소, 물건과 행동)이었다. 원래 LLM의 변화량을 α 0.1로 다시 섞으면 GSM8K가 0.430으로 원래보다 높아졌다 | 원문 |
| [NVLM](https://arxiv.org/abs/2409.11402) (2024) | LLM을 푼 공개 VLM은 텍스트 4개 평균이 −6.3~−6.9점이었다(LLaVA-OneVision 72B, InternVL-2 76B, VILA-1.5 40B). LLM을 얼린 Llama 3-V는 0. NVLM은 좋은 텍스트 SFT 데이터와 이미지 수학 데이터를 함께 넣어 +2.5~+4.3점(MATH 59.7 → 73.1) | 원문 |
| [ViComTe](https://arxiv.org/abs/2205.01850), [VaLM](https://arxiv.org/abs/2205.10178) (2022) | 색깔·모양·크기 같은 시각 상식은 이미지로 학습한 모델이 텍스트 모델보다 낫다. VaLM은 물체 색 +8.7%, 크기 +37.8%였다. BERT급 연구라 지금 LLM에 그대로 옮기기는 어렵다 | 검색 |
| [OpenMMReasoner](https://arxiv.org/abs/2511.16334) (2025) | 멀티모달 RL을 하는 동안 텍스트 추론도 함께 올랐다. 생성형 RL이라 우리 설정과 다르다 | 검색 |
| [Glyph](https://arxiv.org/abs/2510.17800) (2025) | 긴 글을 이미지로 그려 VLM에 넣어 토큰을 3~4배 줄였다. 렌더링 설정에 민감하다 | 검색 |

- **정리:** 이미지 학습이 텍스트를 올리는 것은 "그림으로 볼 수 있는 상식"(물건의 장소·용도, 색, 크기) 쪽에 한정된다. 수학과 HellaSwag 같은 문장 이어 쓰기는 오히려 떨어졌다. 떨어지는 것은 텍스트 데이터를 섞거나 원래 변화량을 다시 섞어 막을 수 있다.

## 8. 이미지 학습이 처음 보는 벤치마크로 안 옮겨 가는 이유 (11-02 추가)

사용자 질문: "추정 아님? 웹 검색해서 논문 찾아봐, 지금 현상과 같은 거." [겹침을 뺀 재학습](../project/vision-train-screen.md#결과-겹침을-뺀-학습-clean-frozen-11-02-1029--1244)에서 본 현상마다 같은 현상을 보고한 논문을 찾았다.

| 우리 현상 | 같은 현상을 보고한 논문 | 그 논문의 숫자 | 확인 |
|---|---|---|---|
| ① 학습한 과제만 오르고 처음 보는 벤치마크는 제자리다. KIE·Hateful Memes는 train split을 빼자 wise80 점수로 돌아갔고, 시험셋은 +1이다 | [Testing the Limits of Fine-Tuning for Improving Visual Cognition in VLMs](https://arxiv.org/abs/2502.15678) (Schulze Buschoff 외, ICML 2025) | Qwen2-VL 7B, Llama 3.2 11B·90B에 QLoRA r16(ViT와 LM 모두), 1만 행, 10 epoch. 같은 분포에서는 0.78~0.83까지 올랐다. 실제 사진의 탑에서는 대부분 우연 수준이었고, 다른 인지 과제로도 옮겨 가지 않았다 | 원문 |
| ① 같은 현상 | [SFT Memorizes, RL Generalizes](https://arxiv.org/abs/2501.17161) (Chu 외, ICML 2025) | Llama-3.2-Vision-11B. 시각 OOD에서 SFT는 GP-VL 23.6% → 13.7%, V-IRL 16.7% → 11.1%로 내렸다. RL은 41.2%, 77.8%로 올렸다. SFT를 늘릴수록 시각 인식 정확도도 내려갔다 | 원문 |
| ② BLINK 대응 과제는 같은 과제 데이터(SPair-71k)가 있을 때만 올랐고, 빼자 사라졌다 | [Hidden in plain sight: VLMs overlook their visual representations](https://arxiv.org/abs/2506.08008) (Fu 외 2025) | VLM이 자기 비전 인코더보다 크게 못한다. DINOv2 기준 저수준 대응은 인코더 0.819 대 VLM 0.170, 의미 대응은 0.536 대 0.225다. 과제마다 5,000개로 LoRA를 하면 LM 쪽 학습이 ViT나 projector보다 크게 올랐다(의미 대응 TV 거리 0.337 → 0.078). 과제 사이 전이는 재지 않았다 | 원문 |
| ③ 시험셋 안에서 오르는 셋과 내리는 셋이 엇갈려 합이 0이다 | [MoCLE](https://arxiv.org/abs/2312.12379) (Gou 외 2023) | InstructBLIP을 모든 데이터로 지시 학습하면, 처음 보는 7개 과제 중 2개(VSR, TextVQA)만 이득이었다. 나머지 5개는 일부 데이터로만 학습한 전문가보다 낮았다. 종류가 다른 과제끼리 충돌한다(negative transfer) | 원문(수치는 그림에만 있다) |
| ④ 공개 VQA 셋을 더 학습해도 새로 배울 것이 적다 | [Qwen-VL](https://arxiv.org/abs/2308.12966) (Bai 외 2023) Table 3 | 2단계 multi-task 사전학습에 GQA, VGQA, VQAv2, DVQA, OCR-VQA, DocVQA, TextVQA, ChartQA, AI2D, COCO 캡션이 들어갔다. 모두 우리 학습 셋에 있는 출처다. Qwen2.5-VL부터는 데이터 목록을 공개하지 않았다(종류만 적었다) | 원문 |
| ⑤ 텍스트만 학습해도 vision이 크게 오른다(보드 +16, 시험셋 +10) | X-Reasoner(§3). SFT Memorizes도 "SFT가 출력 형식을 안정시킨다"고 적었다 | §3 참조 | 원문 |

- ④는 Qwen-VL 1세대의 목록이다. Qwen3.5 0.8B가 같은 셋을 봤다는 직접 근거는 없다. 계열이 같다는 데서 나온 추정이다.

**앞서 한 설명 바로잡기: "보는 능력이 상한"이라는 말은 반만 맞다.**
- Hidden in plain sight는 정보가 비전 특징 안에 이미 있고, 병목은 LM이 그 정보를 꺼내 쓰는 쪽이라고 본다. 과제별 데이터로 LM에 LoRA를 하면 크게 오른다. 우리 SPair 결과(의미 대응 +19, 빼면 0)와 같은 모양이다.
- [Eyes Wide Shut](https://arxiv.org/abs/2401.06209) (Tong 외, CVPR 2024, 검색)은 다르게 본다. CLIP 인코더가 가르지 못하는 패턴(방향, 개수 등)에서는 MLLM도 못한다고 했다. 과제마다 원인이 다를 수 있다.
- 그래서 우리 현상은 이렇게 읽는다. **ViT를 얼려도 과제에 맞는 데이터가 있으면 그 과제는 오른다. 하지만 그 과제에서만 오르고 다른 과제로는 옮겨 가지 않는다.** "못 본다"기보다 "과제마다 따로 배워야 한다"에 가깝다.

**무엇이 옮겨 가는가(문헌)**
- **정답이 정해진 합성 자극:**
  - [Synthetic Stimuli, Real Gains](https://arxiv.org/abs/2511.11440) (Rizzoli 외 2025, 검색): 정답을 정해 둔 합성 이미지 VQA로 학습하니 실제 COCO 사진에서도 올랐다. 위치 과제다.
  - [SPRITE](https://arxiv.org/abs/2512.16237) (2025, 검색): 프로그램으로 공간 문항 21만 개를 만들었다. 같은 규모의 공개 데이터보다 여러 공간 벤치마크에서 높았다.
  - [SITUATE](https://arxiv.org/abs/2602.00108) (2026, 검색): 배치와 가림을 조절한 합성 개수 셋이다. Qwen2.5-VL 7B로 OOD 일반화를 봤다.
- **중간 표현 학습:** [AURORA perception tokens](https://arxiv.org/abs/2412.03548) (Bigverdi 외, CVPR 2025, 검색). 깊이 지도·상자를 중간 토큰으로 학습해 BLINK 상대 깊이 +6.4, 개수 +10.8을 보고했다. 깊이 지도를 어디서 얻었는지는 확인하지 않았다. 모델이 만든 것이면 정답 전용 규칙에 맞지 않는다.
- **RL:** SFT Memorizes는 RL이 OOD로 옮겨 간다고 했다. 다만 여러 단계로 생성하는 과제다. 우리처럼 한 번 forward해서 고르는 문제에서는 정답률 보상의 기대 gradient가 −p loss와 같다([GLiClass](gliclass.md)). 그래서 그대로 옮겨 오기 어렵다.
- **과제 섞기:** Schulze Buschoff 외는 두 과제를 섞어 학습하면 두 과제 모두 조금씩만 내리고 잘했다고 했다. 섞은 과제는 덮지만, 섞지 않은 과제로 넓어지지는 않는다.

**우리에게 주는 의미**
- 지금 결과는 흔한 현상이다. 과제별 SFT는 그 과제를 올리고, 다른 과제로는 잘 옮겨 가지 않는다.
- 보드 비공개 vision을 올리려면, 비공개에 나올 종류의 과제를 평가 원본과 겹치지 않게 과제마다 만들어야 한다. 예를 들면 BLINK의 대응·깊이·다시점 과제다.
- 문헌에 있는 방법은 정답을 프로그램이 정하는 합성 데이터다. 대응(homography), 개수, 공간 배치가 후보다.

### 8-1. 텍스트 학습은 왜 옮겨 간 것처럼 보였나 (11-02 추가)

사용자 질문: "원래 그런가? 텍스트는 일반화가 잘 된 거 같은데." 우리 수치는 [원인 분석](../project/vision-train-screen.md#원인-분석-처음-보는-벤치마크에서-이득이-작은-이유-11-02)에 있다.

| 문헌 | 내용 | 확인 |
|---|---|---|
| [FLAN](https://arxiv.org/abs/2109.01652) (Wei 외, ICLR 2022)과 [Google 블로그](https://research.google/blog/introducing-flan-more-generalizable-language-models-with-instruction-fine-tuning/) | 텍스트 지시 학습의 처음 보는 과제 이득은 모델이 충분히 클 때만 나타났다. 작은 모델에서는 오히려 내렸다(블로그: "At smaller scales, the FLAN technique actually degrades performance"). 과제 묶음 수가 많을수록 처음 보는 과제가 올랐다 | 초록·블로그(그림의 수치는 확인 못 함) |
| [LIMA](https://arxiv.org/abs/2305.11206) (Zhou 외 2023) | "almost all knowledge in large language models is learned during pretraining". 지시 학습은 주로 응답 형식을 가르친다 | 초록 |
| [Vision-Flan](https://arxiv.org/abs/2402.11690) (Xu 외, ACL Findings 2024) | 사람이 정답을 단 과제 187개로 이미지 지시 학습을 했다. 과제 다양성과 성능이 함께 올랐다. GPT-4 합성 데이터는 능력보다 응답 스타일을 바꿨다 | 검색 요약 |
| [Caption This, Reason That](https://arxiv.org/abs/2505.21538) (NeurIPS 2025) | 작은 VLM을 시각 추론 과제로 미세조정하면 그 능력은 크게 올랐다. 하지만 "does not translate to large enhancements on challenging, out-of-distribution benchmarks" | 초록 |
| [Understanding Task Transfer in VLMs](https://arxiv.org/abs/2511.18787) (Sachdeva 외 2025) | VLM 3개에서 지각 과제 13개를 하나씩 미세조정하고, 다른 과제로의 전이를 쟀다. 오르는 전이와 방해하는 전이가 둘 다 있었다. "Finetuning on one task can unpredictably affect performance on others" | 초록 |

- **텍스트도 새 분야로는 잘 안 옮겨 간다.** FLAN이 작은 모델에서 본 것과 같다. 보드의 0.8B 미세조정 모델 19개가 그렇다(아래 우리 수치).
- **텍스트 학습이 이미지 문항으로 옮겨 간 것은 형식 쪽이다.** 객관식에서 한 번 읽고 고르는 법은 모든 문항에 쓰인다. LIMA와 Vision-Flan이 말하는 "지시 학습은 형식을 가르친다"와 같은 쪽이다.
- **이미지 학습은 그 위에 과제별 지각을 더한다.** 그래서 Caption This, Reason That이나 §8의 문헌처럼 학습한 과제 근처에서만 오른다.

## 우리에게 주는 의미

1. **wise80 위에 이미지 LoRA를 얹는 것은 문헌의 주류 방식이다**(Phi-4-multimodal, Llama 3.2와 같은 모양).
   - vision 보드는 따로 제출하므로, vision 엔진에서만 LoRA를 불러오면 텍스트 보드 점수는 wise80 그대로다.
   - 한 모델로 두 보드를 다 내고 싶다면 Partial LoRA(이미지 토큰에만 적용)를 쓰면 텍스트 입력이 정확히 같게 유지된다. 다만 Image-LoRA 결과처럼 이미지 토큰 비율이 낮은 문항에서는 약할 수 있다.
2. **어디에 붙일지:** Phi-4는 decoder의 모든 linear에 붙였다. 우리 텍스트 실험의 최선(전체 LoRA r16)과 같은 위치다. 기본안은 전체 LoRA r16 + merger 학습 + ViT 얼림이고, ViT LoRA를 팔로 둔다.
3. **학습률은 작게:** LoRA나 작은 학습률이면 망각이 거의 없다. full FT를 쓴다면 1e-6 수준이어야 했다(2603.14493).
4. **지름길 주의:** 질문 종류가 하나뿐인 분류 데이터로만 학습하면 "질문과 상관없이 클래스를 고르는" 습관이 생길 수 있다. 질문 형식을 다양하게 하고 텍스트 판정 데이터를 섞는다.
5. **다른 길:** 원래 Qwen에 이미지 LoRA를 학습하고 wise80의 변화량과 섞는 방법(Bring Reason to Vision 방식)도 시험할 수 있다.
6. **아직 모르는 것:** 텍스트 학습의 vision 이득 중 양방향 attention의 몫. 저장된 causal / 양방향 full FT fit(`runs/from-base/methods-v1/`)을 vision 셋에서 재면 가를 수 있다.
