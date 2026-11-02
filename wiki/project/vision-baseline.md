# 공개 Vision 보드 기준점: 원래 Qwen과 wise80 (2026-10-31)

[위키 홈](../README.md) · [인사이트](insights.md) · [g7·섞기 결과](hub-g7-merges.md) · [목표와 규칙](jev-index.md)

**상태: 완료(2026-10-31 20:20 KST).** 이미지 학습을 하기 전에 지금 모델이 Vision 보드에서 어디쯤인지 쟀다. 학습은 하지 않았다. [수치 전체](vision-baseline-results.json).

**한 줄 결론: 텍스트로만 학습한 wise80이 원래 Qwen보다 vision 지수가 16.4점 높다(49.05 대 32.66, 95% 구간 [+14.5, +18.2]).**
- 비전 타워는 원래 Qwen과 같다. 바뀐 것은 언어 모델과 attention 방향(양방향)뿐이다.
- 이 중 약 4점은 원래 Qwen의 글자 편향(예: BLINK에서 87%를 "A"로 답함)이 사라진 몫이다. 편향을 빼도 13.6점 차이가 남는다(36.53 대 50.14).
- wise80은 프롬프트를 바꿔도 점수가 같다(decide 49.05, chat 49.14). 원래 Qwen은 프롬프트에 따라 2.9점 흔들린다.

## 설정

- **셋:** Decision Index 공개 vision 0.3, 키트 commit `f5a6e4f`. 11개 벤치마크 중 10개, 11,410행이다.
  - **Winoground(800행, 가중치 1)는 없다.** HF gated 셋인데 `Kyunghwan-kim` 계정이 약관에 동의하지 않아 받지 못했다.
  - 나머지 10개 셋은 고정 해시와 행 수, items 해시가 모두 맞았다.
  - Mind2Web은 상자 글자를 Arial Bold 대신 Liberation Sans Bold로 그려서 이미지 바이트가 보드와 다르다. 문항과 run id는 같다.
- **지수:** 키트 채점기(`vision score`) 그대로다. 벤치마크마다 (정답률 − chance) / (1 − chance)를 [0, 1]로 자르고 가중 평균한다.
  - "있는 것만" 지수는 있는 가중치 8.75로 다시 나눈 값이다(키트가 `complete: false`로 계산하는 값).
  - "Winoground 0점" 지수는 빠진 Winoground를 0으로 친 하한이다(× 8.75 / 9.75).
- **엔진:** `qwen_decision/v3/vision_engine.py`. transformers 5.18.0, bf16, RTX 3090, 한 행에 forward 한 번. 이미지는 키트와 같이 1.6MP로 제한했다(`images.pil`).
  - `chat`: 보드의 stock VLM 기준 경로(`vllm-vl`)를 따랐다. user 턴 하나(이미지, 질문, `A. 선택지` 줄, "Answer with the option's letter from the given choices directly."), 기본 chat template(빈 think 블록), R-Bench-M과 MMMU-Pro는 assistant가 `The answer is (`로 시작한 뒤 다음 토큰의 글자 logit을 읽는다.
  - `decide`: v3 텍스트 형식(`format.render`, d3 layout)에 이미지를 state로 넣었다. 마지막 토큰에서 답 코드 logit을 읽는다. readout prefix는 쓰지 않는다.
- **모델:**
  - 원래 Qwen3.5-0.8B(`models/qwen3.5-0.8b`), causal.
  - wise80 g6+g7e2(`runs/v3/wise80-g6-g7e2`), 양방향. 텍스트 판정 데이터로만 학습한 g6과 g7을 섞은 모델이다([g7·섞기 결과](hub-g7-merges.md)). 비전 텐서 153개(타워와 merger)가 원래 Qwen과 원소 단위로 같다(10-31 확인).
- 각 모델을 자기 프롬프트와 상대 프롬프트로 한 번씩, 모두 네 번 돌렸다.

## 결과

벤치마크별 skill(0~1)이다. 위 두 줄이 주 비교(각 모델의 자기 프롬프트)다.

| 실행 | 있는 것만 | Winoground 0점 | CV-Bench | BLINK | RealWorldQA | CharXiv | InfoVQA | Mind2Web | KIE | Hateful Memes | R-Bench-M | MMMU-Pro |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 원래 Qwen, chat | 32.66 | 29.31 | 0.350 | 0.114 | 0.348 | 0.348 | 0.667 | 0.215 | 0.669 | 0.110 | 0.058 | 0.036 |
| **wise80, decide** | **49.05** | **44.02** | 0.604 | 0.393 | 0.420 | 0.629 | 0.820 | 0.424 | 0.715 | 0.169 | 0.177 | 0.118 |
| 원래 Qwen, decide | 29.76 | 26.71 | 0.375 | 0.125 | 0.348 | 0.361 | 0.563 | 0.157 | 0.559 | 0.111 | 0.000 | 0.019 |
| wise80, chat | 49.14 | 44.10 | 0.624 | 0.386 | 0.409 | 0.627 | 0.803 | 0.448 | 0.725 | 0.149 | 0.193 | 0.124 |

행 수: CV-Bench 2,038, BLINK 961, RealWorldQA 589, CharXiv 532, InfographicVQA 244, Mind2Web 973, KIE 2,465, Hateful Memes 1,213, R-Bench-M 665, MMMU-Pro vision 1,730. 네 실행 모두 11,410행 전부 답했다(오류 0).

**차이의 구간**(벤치마크 안에서 행을 다시 뽑는 paired bootstrap 2,000번, "있는 것만" 지수):
- wise80 decide − 원래 Qwen chat: **+16.39 [+14.49, +18.24]**
- wise80 chat − wise80 decide: +0.09 [−0.79, +0.98]
- 원래 Qwen chat − 원래 Qwen decide: +2.90 [+0.95, +4.68]

**참고점:** 같은 키트로 다시 채점한 보드 공개 지수는 JEV-27B-VL 72.78, Solomon v1.1(27B) 69.80이다. Winoground를 포함한 값이라 위 숫자와 바로 비교할 수 없다.

**d3-lite(11-02 추가):** [모델 README](https://huggingface.co/vllm-sr/d3-lite)에 따르면 보드 Vision 0.3.2 점수는 42.1(공개 47.9, 비공개 36.3)이다. README에 벤치마크별 skill도 있다: CV-Bench 65.5, BLINK 37.7, RealWorldQA 35.9, CharXiv 65.7†, InfographicVQA 74.9†, Mind2Web 53.1†, Winoground 57.0, KIE 93.2†, Hateful Memes 11.8, R-Bench-M 10.5, MMMU-Pro 8.3. †는 d3 쪽이 셋을 근사로 다시 만든 것이다.
- 이 숫자를 키트 가중치로 Winoground 없이 다시 계산하면 50.87이다. 위 표의 "있는 것만" 지수와 같은 기준이다.
- Winoground를 넣어 다시 계산하면 51.5다. 보드 공개 점수 47.9와 3.6점 차이가 나는데, 원인은 모른다.
- **11-02에 같은 행으로 직접 쟀다:** 10개 지수는 47.19다(wise80 49.05). Winoground 57.0을 더하면 48.19로, 보드 47.9와 맞는다. 차이는 README의 † 셋에서 나왔다. 자세한 것은 [이미지 학습 결과](vision-train-screen.md#d3-lite와-비교-11-02-실측)에 있다.

## 원래 Qwen의 글자 편향

원래 Qwen은 정답과 상관없이 한 글자를 많이 고른다. 표는 가장 많이 고른 글자와 그 비율이다.

| 벤치마크 | 정답에서 가장 많은 글자 | 원래 Qwen chat | wise80 decide |
|---|---|---|---|
| CV-Bench | A 40% | A 68% | A 40% |
| BLINK | A 37% | A 87% | B 38% |
| RealWorldQA | A 40% | A 57% | B 39% |
| CharXiv | D 26% | D 47% | D 29% |
| Mind2Web | A 28% | D 43% | A 31% |
| MMMU-Pro vision | B 12% | A 46% | F 16% |

- 벤치마크마다 위치별 평균 logit을 빼고 다시 고르면(정답을 쓰지 않는 분석용 보정, 제출 방식이 아니다) 원래 Qwen chat은 32.66 → 36.53, wise80 decide는 49.05 → 50.14다.
- 그러니 16.4점 중 약 4점이 편향 몫이고, 약 13.6점은 편향을 빼도 남는다. 남는 차이가 큰 곳은 CharXiv(0.343 → 0.619), CV-Bench(0.412 → 0.603), BLINK(0.217 → 0.401), Mind2Web(0.242 → 0.418)이다.
- wise80은 이미 편향이 거의 없다. 텍스트 학습(`train.py`)이 선택지 순서와 코드 배정을 섞어서, 답이 위치나 글자와 묶이지 않게 배웠기 때문으로 보인다.
- wise80은 두 프롬프트에서 같은 답을 88.9% 골랐다. 원래 Qwen은 57.6%였다.

## 해석

- **텍스트 판정 학습이 이미지 문항으로 옮겨 갔다.** 비전 타워가 그대로인데 "선택지를 보고 하나를 고르는" 능력이 좋아졌다. 이미지를 읽는 능력이 늘었다기보다, 읽은 것을 답으로 바꾸는 부분이 나아졌다고 보는 것이 맞다.
- **이것이 학습 덕분인지 양방향 attention 덕분인지는 가르지 못했다.** wise80은 양방향이라 이미지 토큰이 질문과 선택지를 볼 수 있다. 원래 Qwen은 causal이다. 둘을 가르려면 같은 데이터로 causal과 양방향을 각각 학습한 모델이 필요하다. 이미 학습된 모델의 attention 방향만 바꾸면 학습 분포에서 벗어나 공정한 비교가 아니다.
- **이미지 학습이 아직 아무것도 하지 않은 출발점이 꽤 높다.** InfographicVQA 0.82, KIE 0.72처럼 글자를 읽는 셋은 이미 높고, BLINK 0.39, Hateful Memes 0.17, R-Bench-M 0.18, MMMU-Pro 0.12가 낮다. 이미지 학습의 여지는 공간·지각(CV-Bench, BLINK, RealWorldQA)과 Hateful Memes에 있다.
- **지연은 문제가 아니다.** 네 실행 모두 중앙값 84~85ms, 평균 152~156ms, p80 169~176ms다(RTX 3090, 보드 기준은 RTX PRO 6000에서 1,000ms 이하).

## 한계

- Winoground가 빠졌다. 제출하려면 HF에서 약관에 동의하고 다시 빌드해야 한다.
- Mind2Web 이미지 바이트가 보드와 다르다(글꼴). 상자 글자의 모양이 달라서 점수가 조금 다를 수 있다.
- 보드 기준 경로는 vLLM이다. 여기서는 같은 프롬프트를 transformers로 한 행씩 돌렸다. 수치 정밀도 차이로 일부 답이 바뀔 수 있다.
- 이미지 공개 셋에서 고른 것이 없으므로(모델 4개, 미리 정한 두 비교) 선택 편향은 작다. 하지만 비공개 절반은 재지 않았다. Full 점수는 공개 50% + 비공개 50%다.

## 다시 돌리기

```bash
export PYTHONPATH=.:external/decision-index
.venv/bin/python -m qwen_decision.v3.vision_baseline rows                # 빌드된 셋만 rows-built.jsonl로
.venv/bin/python -m qwen_decision.v3.vision_baseline run wise80-decide   # base-chat, base-decide, wise80-chat도 같다
.venv/bin/python -m qwen_decision.v3.vision_baseline score               # runs/vision/summary.json
```

셋 빌드는 `decision-index suite rebuild --vision --work data/vision-work`다. 실행 하나는 약 29분 걸렸다. 결과는 `runs/vision/<실행>/`(results.jsonl, vision-scores.json, environment.json)에 있다.
