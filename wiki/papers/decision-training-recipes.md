# 판정 모델 학습 레시피 논문 묶음 (2026-10-26~27 조사)

[위키 홈](../README.md) · [학습 방식 비교 실험](../project/training-methods.md) · [평가 점검](../project/evaluation-validity.md) · [loss 문헌 검토](loss-alternatives.md)

사용자가 가져온 요약을 원문(arXiv 초록과 HTML)으로 다시 확인했다. 확인하지 못한 항목은 그렇다고 적었다. 모두 2026년 공개 프리프린트이고, 우리가 재현한 결과가 아니다.

| 논문 | 모델 | 무엇을 했나 | 확인한 핵심 수치 | 우리에게 가져온 것 |
|---|---|---|---|---|
| [Where Should LoRA Go?](https://arxiv.org/abs/2604.22127) | Qwen3.5-0.8B | LoRA를 attention / Gated DeltaNet / 전체 중 어디에 붙일지 비교. 과제당 2천 문항 × 3 epoch, r16, seed 1 | GSM8K 기준 0.297. attention만 학습하면 0.398, **DeltaNet만 학습하면 0.148(−14.8점)**, 전체는 0.375 | **frozen**: DeltaNet token mixer를 얼린다 |
| [Narrow Decision Models](https://arxiv.org/abs/2610.25553) | mmBERT-base | 선택지 차단 mask + 위치 초기화 + set head. log-loss + Brier, 3 seed | 같은 토큰 예산에서 pairwise보다 NLL 0.819 대 1.024. 다만 문항을 6배 더 봤다. set head를 빼면 정답률 0.705 → 0.655 | **brier**: CE + Brier. 선택지 차단 구조는 가져오지 않는다 |
| [DKnown-Jev](https://arxiv.org/abs/2610.24358) | Qwen3.5-2B | LoRA(attention projection) + 선택지별 scalar head, CE + 0.1×Brier, 학습 26,064 / 평가 1,120 | 92.86%로 Jev보다 0.18점(2문항) 높음. 저자도 안정적 우위가 아니라고 적었다. seed 미보고 | Brier 근거를 하나 더함 |
| [Releasing the Inner Jev (Jeff)](https://arxiv.org/abs/2610.17205) | Gemma 4 31B 등 | LM head 행에서 판정 head를 시작하고 LoRA r32로 학습, 3 seed. 41.8만 문항 풀에서 run당 약 7.6만 문항 | head만 학습하면 78.9(학습 전 79.6과 같음), 기본 레시피 84.4. 800/1,500/3,000 step은 84.7/83.6/83.3 | 본체 적응이 핵심이고 head는 아니다. 우리 code readout을 유지한다. OpenDecisions(2,352)는 추가 평가 후보 |
| [LLM-as-Jev](https://arxiv.org/abs/2610.02076) | Qwen3.5-4B, Qwen3-0.6B | 구조는 유지하고 listwise loss + KL anchor. 9천~3만 문항(최대 9.5만), 1 epoch 이하, seed 1 | 0.6B: 학습 전 JevBench 56.7, full FT 58.0, LoRA 61.0 (231문항이라 ±3점은 잡음 범위) | KL·L2-SP anchor는 우선순위 낮음 |
| [Signal and Noise](https://arxiv.org/abs/2508.13144) | 1B 사전학습 | 벤치마크의 신호 대 잡음 측정 | 정답률 대신 정답 likelihood 지표를 쓰면 작은 규모 결정 정확도가 77.0% → 83.7% | **주 지표를 정답 NLL로** |

**LoRA placement 세부(원문 HTML, 10-28 확인)**

- 설정: r16, α32, dropout 0.05, LR 2e-4 cosine, 유효 batch 16, 3 epoch, 1,024토큰, 8-bit Adam, seed 3407 하나.
- Qwen3.5 크기는 0.8B 하나만 실험했다. 비교 모델은 병렬형 hybrid인 Falcon-H1-0.5B다. 이쪽에서는 순환 경로를 학습시키는 것이 오히려 +8.6점 도움이 됐다.
- GSM8K 학습 Table 5에서는 softmax_plus_mlp도 0.148로 gdn_only와 같았다. GSM8K 평가가 128~256문항이라 이 칸들의 잡음이 크다.
- 이 설정을 그대로 옮긴 실험: [Attention-only LoRA](../project/lora-attention.md).

## 우리 실험에서 확인한 결과 (10-28~29, [학습 방식 거르기](../project/training-methods.md))

- **LoRA placement → frozen: 통과.** full FT에서 DeltaNet token mixer만 얼렸더니, 양방향 대비 selection 정답률 +1.4점(5/5 seed), 장문 +2.3점(4/5 seed)이었다. 논문의 "DeltaNet을 학습시키면 해롭다"와 같은 방향이다. 다만 1.6만 행 결과라 과적합 감소 효과와 아직 구분되지 않는다. 아래 위치 비교에서 DeltaNet LoRA가 해롭지 않아, 이 해석은 약해졌다.
- **Attention-only LoRA(softmax_only) 재현:** 짧은 문항은 full FT와 같았고, 장문은 +4.8점이었다([상세](../project/lora-attention.md)).
- **위치 비교(10-29, [상세](../project/lora-placement.md)): 논문과 달랐다.**
  - 논문에서는 attention-only가 가장 좋았다. 우리 과제에서는 r16 위치 중 짧은 문항이 가장 낮았다. gdn_only, mlp_only, softmax_plus_mlp, all_layers 모두 selection NLL이 attention-only보다 나았다(5/5 seed). 장문은 차이가 없었다.
  - DeltaNet에 붙인 LoRA(gdn_only)도 해롭지 않았다. 그러니 frozen의 이득은 "DeltaNet 학습이 해롭다"보다 "full FT가 너무 많이 바꾼다"로 보인다.
  - mlp_only, softmax_plus_mlp, all_layers는 frozen을 짧은 문항과 장문 둘 다 이겼다(보고만 한 비교).
- **NDM·DKnown-Jev → brier: 효과 없음.** CE + 1×Brier는 양방향과 NLL·정답률·Brier 모두 차이가 없었다. 전체 LoRA r16에서도 CE와 차이가 없었다(10-29, [Loss 비교](../project/loss-screen.md)).
- **Signal and Noise → 정답 NLL 주 지표: 유효.** 정답률만 봤다면 seed 3개로는 잡기 어려운 1점대 차이를, seed 5개와 NLL로 판정할 수 있었다.

## 가져오지 않은 것과 이유

- **선택지 차단 + set head (NDM):** 이득이 문항 처리 효율에서 나왔고, set head(선택지 비교)를 빼면 나빠졌다. 리더보드에서도 비슷한 구조의 Sieve-0.8B(23.06)가 d3-lite(32.33)보다 낮다. Qwen3.5는 DeltaNet 상태까지 갈라야 해서 구현도 크다.
- **선택지별 scalar head (DKnown-Jev, Jeff 변형):** 우리 readout 교체 실험(10-19)에서 code readout보다 나은 head가 없었다. Jeff에서도 head보다 본체 적응이 핵심이었다.
- **S₀ tuning, RL(ReTerm), diffusion 변환(FLARE):** 각각 소량 데이터용 경량 적응, 다단계 에이전트용, 10B 토큰 규모라 지금 목표와 맞지 않는다.
