# 잠재 작업공간과 recurrent reasoning: 적용 범위 대조

2026-10-19 alphaXiv 검색 후 아래 8편의 추출 원문에서 설계·실험·학습 조건의 해당 부분을 확인했다. 모든 부록을 완독하거나 공개 코드를 재현한 기록은 아니다.

| 논문 | 확인한 부분 | 근거의 범위 |
| --- | --- | --- |
| [Less is More: Recursive Reasoning with Tiny Networks](https://arxiv.org/abs/2510.04871), Alexia Jolicoeur-Martineau | 답 상태·사고 상태의 recursive update, deep supervision, 퍼즐 평가 | 작은 퍼즐 solver의 학습이다. 자연어 Decision retrofit 대조가 아니다. |
| [T²MLR: Transformer with Temporal Middle-Layer Recurrence](https://arxiv.org/abs/2607.15178), Ziyang Cai 외 | 중간 temporal fusion, §4.4 pretrained retrofit, Table3 | SmolLM2-1.7B와 같은 데이터로 fine-tune한 native 비교가 있다. GSM8K35.78→39.88, MATH50012.80→18.00%. 자기회귀 시간축의 recurrence다. |
| [From Chain-of-Thought to Loops: Non-Autoregressive Latent Reasoning via Looped Transformers](https://arxiv.org/abs/2610.11472), Gerard Grau García 외 | joint workspace refinement, Methods, Table1/2 | latent 감독은 추론문 기반이다. 코드 네 평가 평균63.6%, answer-only SFT62.7%, reasoning SFT63.2%; 우리 gold-only 조건의 재현은 아니다. |
| [INTACT: Inducing Latent Reasoning on a Frozen Backbone](https://arxiv.org/abs/2610.19568), Kutay Tire 외 | progressive substitution, distillation, latent-policy reinforcement | frozen LM+외부 encoder 두 단계다. 우리 backbone 구조만의 개선 근거로 사용할 수 없다. |
| [Latent Recurrent Thoughts: Recurrent Refinement of Proposed Latents for Reasoning with Frozen LLMs](https://arxiv.org/abs/2609.01117), Zhaoliang Chen·Jie Fu | proposer/refiner, Table1의 baseline 범위, answer-only StrategyQA 부록 | per-task-family 모듈+frozen8B. 본문은 업데이트하는 decoder 대조군을 명시적으로 제외한다. 보편적 구조 우월성으로 읽지 않는다. |
| [BDH-CQ: In-Context Learning with Recurrent Latent Reasoning](https://arxiv.org/abs/2608.09888), Björn Engdahl 외 | §3.1의 BDH provenance·recurrent memory, ARC 평가 범위 | post-Transformer primitive까지 바꾸는 더 급진적 대안이다. 150M/ARC-AGI-1 29.5%pass@2는 candidate construction/ranking을 포함한 전체 시스템 수치이며, 자연어 Decision 평가가 아니다. |
| [Think before you speak: Training Language Models With Pause Tokens](https://arxiv.org/abs/2310.02226), Sachin Goyal 외 | 초록 및 저자 PDF의 사전학습/후학습 조건 | pause pretraining+finetuning의 이득과 기존 pretrained 모델에 pause만 추가한 경우를 구분해야 한다. |
| [Unlocking the Working Memory of Large Language Models for Latent Reasoning](https://arxiv.org/abs/2605.30343), Lukas Aichberger·Sepp Hochreiter | RiM의 고정 memory block·single forward, Figure1/2, 두 단계 학습 | 먼저 reasoning-step 감독으로 workspace를 grounding하고, 이후 answer refinement만 학습한다. 전역 중간 상태에 후보별 가설을 주입하는 이번 구현과는 다르지만 단일 forward의 작업공간이라는 가까운 선행연구다. |

작업공간·추가 토큰·반복 계산 각각은 기존 연구가 있는 요소다. 이를 우리 모델에 조합한 것만으로 최초 발명이라고 주장하지 않는다. [이번 설계와 사전 고정한 선별 조건](../project/hypothesis-workspace.md)에서 원문과의 차이, 실제 구현과 검증 상태를 별도로 기록한다.

## 2026-10-19 후속: 구조 전환과 학습 데이터

- [The Dragon Hatchling: The Missing Link between the Transformer and Models of the Brain](https://arxiv.org/abs/2509.26507): 초록·[공개 BDH-GPU 코드](https://github.com/pathwaycom/bdh/blob/2b0d7a45b058d4309c84a10e0768d541fe18bdc2/bdh.py)·license를 확인했다. 공개 코드는 양수 feature, Q=K의 RoPE linear retrieval, multiplicative coactivation, shared refinement를 구현한다. [우리 적용](../project/bdh-memory.md)은 양방향/self 제외·native projection 재사용의 별도 설정이다.
- [LoLCATs: On Low-Rank Linearizing of Large Language Models](https://arxiv.org/abs/2410.10254): 원문에서 softmax attention 출력과의 MSE attention transfer 후 LoRA로 적응하는 두 단계를 확인했다. 구조 교체에서 정렬 학습의 중요성을 보여주는 관련 근거이며 우리 BDH 결과를 보증하지 않는다.
- [The Mamba in the Llama: Distilling and Accelerating Hybrid Models](https://arxiv.org/abs/2408.15237): 원문 및 공식 repo에서 projection 재사용·단계별 정렬·distillation을 구분하는 recipe를 확인했다. 우리 gold-only CE 실험에 같은 단계를 적용한 것으로 주장하지 않는다.

BDH-CQ 원문§3.3은 차원·정확한 update·구현 세부가 proprietary라고 명시한다. 그 성과를 공개 BDH 예제나 이번 retrofit의 성과로 대체해서 읽지 않는다.
