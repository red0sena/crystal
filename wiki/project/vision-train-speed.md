# 이미지 LoRA 학습 속도 (2026-11-01)

[위키 홈](../README.md) · [Vision 기준점](vision-baseline.md) · [다른 벤치마크 시험셋](vision-heldout.md) · [이미지 데이터 출처](../designs/vision-data-sources.md)

**질문:** "LoRA로 이미지 학습할 때 학습 속도 최적화가 적용돼 있는지 확인하고, 웹 검색으로 최적화 방안을 찾아 적용하라."

**확인 결과:** 텍스트 학습 경로는 10-16~17에 최적화를 마쳤다([v3 로그](v3-log.md)의 학습 속도 절). 이미지 학습 코드는 없었다. `vision_engine.py`는 추론 전용이고, 한 행씩 forward하며 transformers 기본 경로를 쓴다.
- **텍스트에 이미 있던 것:**
  - BF16 계산, FP32 master(`MasterAdamW`).
  - 패딩 없는 packing. varlen attention, FLA scan·conv에 cu_seqlens를 넘긴다.
  - 원소별 블록 compile.
  - 토큰 예산 microbatch, 큰 microbatch만 체크포인팅.
- **이미지에서 비어 있던 것:**
  - 이미지 행의 packing(M-RoPE 3차원 위치).
  - ViT attention이 이미지마다 SDPA를 따로 부르고, 층마다 길이를 host로 읽는다.
  - 이미지 디코딩·처리기가 학습 루프 안에서 돈다.

그래서 이미지 학습 경로 `qwen_decision/v3/vision_train.py`를 새로 만들었다. 고정 실험 파일은 고치지 않고 import만 한다.

## 적용한 것 (`vision_train.py`)

| 항목 | 내용 | 수학 |
|---|---|---|
| 이미지 행 packing | 행마다 M-RoPE (t, h, w) 위치를 CPU에서 `get_rope_index`와 같게 만들고 이어 붙인다. 각 행의 위치는 0부터 다시 시작한다. 텍스트 경로의 varlen attention·FLA cu_seqlens를 그대로 쓴다 | 같음 |
| ViT varlen attention | 한 microbatch의 모든 이미지를 `varlen_attn` 한 번으로 계산한다. transformers 기본은 이미지마다 SDPA를 부르고 층마다 `.tolist()`로 동기화한다 | 같음(BF16 반올림 차이) |
| ViT 보조 텐서 미리 계산 | cu_seqlens, 회전 위치, 위치 임베딩 보간 인덱스·가중치를 worker에서 계산해 넘긴다(`transformers.vision_utils`가 kwargs에서 꺼내 쓴다) | 같음(비트 단위 일치 확인) |
| 동결 ViT | ViT에 어댑터가 없으면 `no_grad`로 돌린다. activation 저장과 backward가 없다 | 같음 |
| 데이터 worker | 디코딩, 키트와 같은 1.6MP 제한(`images.pil`), 이미지 처리기, microbatch 분할·packing을 DataLoader worker 8개에서 한다. 한 항목이 optimizer step 하나다 | 같음 |
| LoRA | `lora_experiment.LoRALinear`(rank 16, alpha 32, dropout .05). LM은 `lora_placement_experiment`의 attn-mlp(6.39M), ViT LoRA는 선택(qkv, proj, fc1, fc2) | — |

이미지 처리기는 이미 torchvision 백엔드(`Qwen2VLImageProcessor`, `TorchvisionBackend`)다. 평가 엔진도 같은 처리기라 학습과 평가의 전처리가 같다.

## 정확성 검사 (`vision_train check`, `runs/vision-train/check.json`)

Cauldron 6개 부분집합에서 12행(이미지 14장, NLVR2는 두 장)으로, 새 경로를 transformers 기본 경로(한 행씩, 이미지별 SDPA)와 비교했다.
- **M-RoPE 위치:** 12행 모두 `get_rope_index`와 정확히 같다.
- **미리 계산한 ViT 보조 텐서:** 모델이 계산한 것과 출력이 비트 단위로 같다.
- **ViT varlen attention:** 토큰 cosine 평균 0.999998, 최소 0.9968이다.
  - FP32 ViT와 비교하면 새 경로 평균 0.996967, 기본 경로 0.996966이다. 가장 나쁜 토큰(0.625)도 같다.
  - 즉 두 경로의 차이는 BF16 반올림 수준이다.
- **logit:** 최대 차이 0.10(logit 크기 7.9)이다. 텍스트 varlen 도입 때(0.125)와 비슷하다.
  - 12행 중 11행의 답이 같다. 다른 1행은 기본 경로의 1·2위 차이가 0.061인 동점에 가까운 행이다.
- **행 독립성:** 0번 행 logit을 다른 행의 입력 임베딩(이미지 토큰 포함)으로 미분한 값이 정확히 0이다.

## 속도 측정 (`vision_train bench`, `runs/vision-train/bench.jsonl`)

**조건:**
- **모델·학습:** wise80 g6+g7e2(양방향)에 LM LoRA attn-mlp(6.39M)를 붙였다. ViT는 동결이다(ViT LoRA는 따로 표시).
- **데이터:** Cauldron 6개 부분집합(AI2D, VQAv2, TextVQA, DocVQA, InfographicVQA, NLVR2)의 고정 행이다. step당 16행, 이미지 18.7장, LM 토큰 14,495개(그중 이미지 토큰 12,545개)다.
- **측정:** 1.6MP 제한(보드와 같음), RTX 3090이다. warmup 3 step 뒤 12 step의 중앙값이고, 단계마다 device 동기화로 끊었다.
- 학습한 가중치는 저장하지 않는다.

### 기본 경로 대비

| 경로 | step 시간 | 행/초 | 데이터 대기 | 최대 메모리 |
|---|---:|---:|---:|---:|
| transformers 기본(`hf`): 4행씩 패딩, 이미지별 ViT SDPA, 전 층 체크포인팅, 루프 안 전처리, compile 없음 | 4.66s | 3.52 | 0.80s | 3.4GB |
| **새 경로(`fast`)** | **1.30s** | **12.05** | 0.0001s | 15.2GB |

- **3.4배 빠르다.** 기본 경로는 실제 토큰 14,495개를 위해 패딩 포함 24,613개를 계산했다(41%가 패딩).
- 메모리는 늘었다. 체크포인팅을 끈 대가이고, 24GB 안에서 여유가 있다.

### 하나씩 끄기 (새 경로 1.30s 기준)

| 끈 것 | step 시간 | 차이 | 비고 |
|---|---:|---:|---|
| 데이터 worker(루프 안 전처리) | 2.04s | +58% | step당 0.76s를 디코딩·1.6MP 축소·처리기에 쓴다 |
| 체크포인팅 끄기 → 전 층 체크포인팅 | 1.67s | +29% | 메모리 2.9GB |
| 원소별 compile | 1.57s | +21% | 메모리도 20.1GB로 는다 |
| ViT varlen → 이미지별 SDPA | 1.31s | +0.8% | 거의 차이 없다. 정확하고 동기화가 없어 남겨 둔다 |

**microbatch 크기와 체크포인팅(3090):**

| 토큰 예산 / 체크포인팅 | step 시간 | 최대 메모리 |
|---|---:|---:|
| 16k / 체크포인팅(step이 microbatch 하나) | 1.61s | 5.2GB |
| 32k / 체크포인팅 | 1.62s | 4.2GB |
| 8k / 체크포인팅 | 1.65s | 7.7GB |
| 6k / 없음 | 1.33s | 11.8GB |
| **8k / 없음(기본값)** | **1.27~1.30s** | **15.2GB** |
| 12k / 없음 | 1.34s | 21.9GB |
| 16k / 없음 | 메모리 부족 | |

- LoRA는 FP32 master와 moment가 어댑터에만 있어, full FT보다 activation에 쓸 메모리가 훨씬 많다. 텍스트 full FT는 3090에서 2k 토큰 위를 체크포인팅해야 했다.

### ViT에도 LoRA를 붙일 때 (8.75M)

| 설정 | step 시간 | 최대 메모리 |
|---|---:|---:|
| 8k / ViT 체크포인팅 | 2.48s | 15.8GB |
| 8k / LM만 전 층 체크포인팅, ViT 없음 | 2.45s | 12.9GB |
| **4k / 체크포인팅 없음(`fast-vit` 기본값)** | **2.23s** | **13.5GB** |
| 8k / 체크포인팅 없음 | 메모리 부족 | |

ViT LoRA는 동결 ViT보다 1.7배 느리다. ViT backward(이미지당 패치 약 6,250개, 12층)가 생기기 때문이다.

### 해상도 (수학이 바뀜, 참고용)

| 이미지 제한 | step 시간 | 이미지 토큰/step |
|---|---:|---:|
| 1.6MP(보드와 같음, 기본값) | 1.30s | 12,545 |
| 0.8MP | 0.96s(−26%) | 9,072 |
| 0.4MP | 0.63s(−52%) | 5,851 |

- 해상도는 가장 큰 속도 수단이다. 하지만 모델이 보는 것이 바뀐다.
- 학습과 평가 해상도가 다르면 손해라는 보고가 있다(LLaVA-NeXT: ChartQA 45.2 대 49.2). 보드가 1.6MP이므로 기본값은 1.6MP로 둔다.
- 낮은 해상도는 거르기 단계에서 별도 후보로만 쓸 수 있다.

### 새 경로의 시간 분포 (CUDA event, `runs/vision-train/profile.json`)

| 단계 | ms/step | 비중 |
|---|---:|---:|
| ViT forward(동결) | 394 | 27% |
| LM forward | 442 | 31% |
| backward | 600 | 42% |
| optimizer | 10 | 1% |

- **ViT:** 가장 큰 커널은 ViT attention(head 64 flash forward, 140ms)이다. 1.6MP 이미지 하나가 패치 약 6,250개라 attention 비용이 길이의 제곱으로 는다.
- **나머지:** bf16 행렬곱과 residual 덧셈(79ms)이다.

### 예상 학습 시간 (3090, 이 행 구성 기준)

| 규모 | ViT 동결 | ViT LoRA |
|---|---:|---:|
| 거르기 1.6만 행(1,000 step) | 약 22분 | 약 37분 |
| 본 학습 6.5만 행 | 약 1.5시간 | 약 2.5시간 |

- 앞서 잡은 추정(1.6만 1~3시간, 6.5만 4~12시간)보다 훨씬 짧다.
- 실제 섞기는 이미지 크기·장수가 다르므로 첫 실행에서 다시 잰다.

## 적용하지 않은 것

- **동결 ViT 출력 캐시:** ViT forward(step의 27%)와 디코딩을 없앤다.
  - 같은 이미지를 여러 run이 다시 쓸 때만 이득이다(거르기의 여러 seed·arm).
  - 저장 용량이 merger 출력 1024차원 BF16 × 이미지당 약 670토큰 = 약 1.3MB/장이다. 1.6만 행이면 20GB를 넘는다.
  - 다운로드가 끝나면 디스크 여유가 약 55GB라 보류한다.
- **해상도 낮추기:** 위 이유로 기본값에서 뺐다.
- **처리기를 GPU에서 돌리기(`device='cuda'`):** worker로 데이터 대기가 이미 0이다.
- **Unsloth:** 커널 대부분이 이미 있는 packing·compile과 겹친다.
- **FlashQLA:** SM90 이상 전용이라 3090에서 못 쓴다.
- **ViT window attention:** Qwen3.5 ViT에 없는 구조라 모델이 바뀐다.

## 문헌 조사 요약

| 방법 | 기대 이득 | 수학 | 근거 |
|---|---|---|---|
| 이미지 토큰 예산(max_pixels) | 가장 큼(ViT·LM 모두 토큰에 비례) | 바뀜 | Qwen3-VL 공식 SFT 스크립트는 `max_pixels 50176`(약 49토큰). Qwen2-VL 표 7: MMMU는 64~3,136토큰에서 평평, InfoVQA는 28.9 → 77.3. LLaVA-NeXT: 학습과 추론 해상도가 다르면 손해(ChartQA 45.2 대 49.2) |
| 동결 ViT를 `no_grad`로, 출력 캐시 | 캐시는 ViT forward·디코딩까지 없앰 | 같음(증강이 없을 때) | LLaMA-Factory·ms-swift 기본값이 ViT 동결. 캐시 속도를 잰 공개 수치는 못 찾음 |
| ViT attention을 varlen 한 번으로 | 이미지 수에 비례 | 같음 | PyTorch varlen 문서. 공개 수치 없음 |
| 이미지 행 packing(M-RoPE) | NeMo +40~47%, HF padding-free 1.4~2배 | 같음(행 경계를 지킬 때) | Qwen 공식 flatten collator, LLaMA-Factory neat_packing |
| CPU 파이프라인(worker, 미리 줄이기) | 루프 안 전처리 시간만큼 | 거의 같음 | HF torchvision 백엔드(최대 33배) |
| Unsloth류 LoRA 커널 | 1.5배 주장(미검증) | 같음 | 대부분 이미 있는 packing·compile과 겹침 |
| FlashQLA(DeltaNet) | 2~3배 주장 | 같음 | SM90 이상 전용이라 3090에서 못 씀 |
