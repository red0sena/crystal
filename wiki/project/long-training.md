# 구조 후보의 학습 예산 확대 비교

**상태: 완료(2026-10-20).** 결과는 아래 "결과"와 [결과 JSON](long-training-results.json)에 있다. 원래 결과표 `runs/long-training/64k-4096-v1/result.md`는 이 기계에만 있다(git 제외, Hugging Face 미업로드).

## 결과 (2026-10-20, selection 4,147문항 출처 평균, seed 41)

| 구조 | 1,024 | 2,048 | 4,096 | 원본 대비 (95%) |
|---|---:|---:|---:|---|
| 원본 | 75.27% | 75.87% | 79.14% | — |
| BDH | 74.42% | 74.12% | 78.16% | −0.99%p [−2.00, −0.09] |
| 블록 반복 | 75.72% | 76.19% | 79.05% | −0.09%p [−0.76, +0.53] |
| 가설 출구 | 75.15% | 75.45% | 78.45% | −0.69%p [−1.65, +0.17] |

- 사전 기준(원본 대비 +0.5%p)을 넘은 구조가 없다. 새 확인셋은 열지 않았다.
- 구간은 문서 단위 paired bootstrap이고, 학습 seed 변동은 포함하지 않는다. 구조 비교 수로 보정하면 BDH 구간도 0을 포함한다([−2.22, +0.08]).

2026-10-20. 이전512/1,024 update의 작은 차이가 짧은 학습 탓인지 확인하기 위해, 같은 g4b에서 시작하는 긴 추가 학습으로 비교했다. 이전 실패를 구조 자체의 보편적 한계로 취급하지 않는다. 원본에도 같은 학습을 적용한다.

## 비교 계획

| 군 | 구조 | 이전 학습 | 이번 학습 |
| --- | --- | ---: | ---: |
| native | native hybrid24층·기존 코드 head | 512 또는1,024 update | 4,096 update |
| bdh | native16층·공유 연상 메모리3회·기존 코드 head | 1,024 update | 4,096 update |
| recurrent | native24층·내부16:20층 공유 추가 pass·학습 gate | 512 update | 4,096 update |
| workspace | native24층·중간 후보 상태·후보 상태에서만 코드 점수화 | 256 update 후 조기 종료 | 4,096 update |

첫 비교 범위는 위4종·seed41이다. 학습량 확대가 우선이며 역사적 모든 head/설정의 전수 재실험은 아니다. 코드에는 원본/BDH를 seed41/73으로 반복하는 대체 범위도 있지만, 실제 범위는 run의 고정 plan으로 구분한다. 여러 모델의 가중치나 예측을 섞지 않는다.

- 기존 gold6 학습699,960행에서 **고유 ID65,536행/100출처**를 준비한다. 이전 BDH 큰 풀16,384행을 포함한다. distinct state와 출처별 수는 manifest에 기록한다. 의미적 정보량이 정확히4배라는 뜻은 아니다.
- batch16×4,096 update로65,536회 노출한다. 한 pass 동안 각 학습 ID를 한 번 사용하며 seed별 순서·선택지/코드 증강을 구조 사이에 맞춘다.
- 원본 입력의 **1,536토큰 제한은 유지**한다. 이번 실험은 긴 문맥 학습 확대를 동시에 넣지 않는다. selection은 기존4,147문항/77출처이며4,096토큰까지다.
- body LR3e-6, BDH core/workspace LR1e-4, recurrent gate LR.001이다. warmup64 뒤4,096 update의 cosine을10%까지 낮춘다. FP32-master AdamW betas(.9,.95), WD.01(gate0), global clip1, BF16 forward/backward, 정답CE다.
- `torch.use_deterministic_algorithms(True)`와 `CUBLAS_WORKSPACE_CONFIG=:4096:8`을 네 구조 모두 적용한다. CPU 데이터 순서 외에 GPU backward의 비결정성도 줄인 설정이다. 기존 짧은 run과의 연산 환경 차이로 기록하며 정확도 개선 효과를 가정하지 않는다.
- 각 구조가 실제로 사용하는 언어 백본과 tied embedding/head 전체를 학습한다. vision과 BDH 미실행 뒤8층은 frozen이다.
- **1,024/2,048/4,096 update**의 selection·공통 학습 probe·가중치·optimizer를 저장한다. 정확도에 따른 조기 종료는 하지 않는다. NaN·데이터/입력 불일치 등 실행 오류는 중단 사유다.
- 중간 학습 probe에는 아직 보지 않은 행이 있다. 최종4,096에서는512개 모두 학습한 상태이며, 그때의 probe와 중간 혼합 점수를 구분한다.
- 주 비교는 같은4,096 update다. 중간 snapshot 최선 시점 재선택을 주 결과로 쓰지 않는다. 이전 짧은 run과는 데이터 표본·학습률 곡선이 달라, 절대 점수 차이를 순수 step 수의 인과 효과로 부르지 않는다.

## 오차 범위의 해석

기존 문서 bootstrap 구간은 고정된 두 학습 모델에 대해 **평가 문서를 바꿨을 때의 차이**를 나타낸다. 학습 seed·학습 표본 변동이나 더 긴 학습에서의 결과까지 포함하지 않는다. 이번4구조 범위도 한 seed이므로 그 한계는 남는다.

최종 감사는 출처 내 문서 그룹 paired bootstrap5,000회로 일반95% 구간과 한 seed의3개 구조 비교에 대한 Bonferroni 구간을 기록한다. 반복 사용한 selection의 탐색적 결과이며 독립 확인 성공이 아니다. 개선 후보가 나오면 모델·설정을 고정하고 별도로 감사한 미사용 확인 문항이 필요하다. 이 runner는 새 확인셋을 자동으로 열지 않는다.

## 재현·중단 후 재개

구현은 [long_training.py](../../qwen_decision/v3/long_training.py)다. 기존 구조 모듈과 이전 run을 수정하지 않는다. 모든 학습 prompt·target·code·후보 위치를 CPU에서 미리 생성하고 format/offset 일치를 검사한다. 구조별 매 update의 입력/정답 hash가 native와 같아야 학습이 진행된다.

체크포인트에는 BF16 모델만 아니라 **FP32 master 가중치·Adam moments·parameter 이름/순서·Python/NumPy/Torch/CUDA RNG**를 보존한다. 실제 모델 smoke에서3번째 update 저장→4번째 update 실행을 비교해, 재개 후 다음 loss·모든 FP32 master hash·평가 확률이 일치해야 본학습을 고정한다. 중단 후 같은 `run` 명령으로 마지막 완전한 체크포인트에서 재개한다. 저장되지 않은 log suffix나 불완전한 checkpoint는 별도 이름으로 보존한다.

```bash
PYTHONPATH=external/decision-index .venv/bin/python -m qwen_decision.v3.long_training prepare
PYTHONPATH=external/decision-index .venv/bin/python -m qwen_decision.v3.long_training cache
PYTHONPATH=external/decision-index .venv/bin/python -m qwen_decision.v3.long_training smoke
PYTHONPATH=external/decision-index .venv/bin/python -m qwen_decision.v3.long_training freeze
PYTHONPATH=external/decision-index .venv/bin/python -m qwen_decision.v3.long_training run
```

출력은 `runs/long-training/64k-4096-v1/`다. `prepare`와 `smoke`는 기존 출력을 덮어쓰지 않는다. source·입력을 고정한 이후 변경되면 본학습을 거부한다. 본학습 완료 뒤 runner가 네 모델의 전체 selection 예측을 다시 재현하고 통계를 기록한다.

## 준비·시작 기록

**(실행 당시 기록)** 본학습은 2026-10-20에 끝났다. 고유 ID65,536행/100출처를 준비했고 정규화 state는62,419개다. 모든 증강 prompt/target/code/후보 위치의 cache를 생성하고 offset과 원본 prompt 일치를 검사했다.65,536개 학습 index를 각각1회 사용하는 schedule, 유한한 정답 확률의 합1, 코드/후보 수 일치, 위치 범위를 추가 검사했다. 최대 증강 입력1,536토큰·최대 후보151개·잘못된 행0이었다(`preflight-inputs.json`).

최종 코드에서 관련69개 테스트를 다시 통과했다(`tests-69.log`). 실제0.8B 네 구조 각각에서 저장 전후 모델·FP32 master·Adam 상태가 byte-identical했고, 복원 후 다음 loss·전체 master hash·평가 확률도 정확히 일치했다(`smoke-*/result.json`, `resume-comparison.json`). 이 결과와 source·입력 hash를 고정한 뒤2026-10-20 본학습을 시작했다. 원본의 초기 selection71.4304%가 재현됐고 실제 optimizer update가 진행 중이다. 완료·정확도 개선은 아직 주장하지 않는다.

본학습 전 검사에서 새 runner의 재개 후 학습률 그룹 alias 문제를 찾아 수정했다. `optimizer.load_state_dict()`가 내부 group 목록을 바꾼 뒤 wrapper의 참조도 갱신해야 한다. 학습률이 다음 step에서 변하는 CPU regression을 추가했다. 초기 검사 실패의 코드·체크포인트·로그는 `smoke-native.failed-*`에 보존했다.

그 뒤 모델/FP32 master/Adam 상태가 복원 전후 모두 byte-identical해도 다음 backward의 norm이 달라지는 현상을 관찰했다. 해당 비교의 loss는 같았고 master 최대 차이는 약2.32e-7이었다. 결정론 설정을 켜고 같은 모델을 같은 상태로 두 번 되돌려 실행한 대조에서는 다음 loss/norm/master hash가 모두 일치했다(`determinism-probe.json`). 이를 바탕으로 본실험을 결정론 설정에 고정하고, 네 구조의 실제 저장·재개 재검증도 모두 통과했다.

실행은 로컬 supervisor `runs/long-training/64k-4096-v1/worker.py`가 관리한다. 채팅의 대기와 분리된 프로세스이며, 종료 코드가 `worker-result.json`에 기록된다. source를 수정하지 않고 진행 상태를 읽는 명령은 다음과 같다.

```bash
.venv/bin/python runs/long-training/64k-4096-v1/status.py
```

중단·실패 후 다시 실행할 때는 기존 worker 생존 여부를 확인하는 다음 launcher를 사용한다. 같은 명령의 중복 학습 실행은 거부하고, 미완료 run은 마지막 완전한 optimizer checkpoint에서 재개한다.

```bash
.venv/bin/python runs/long-training/64k-4096-v1/worker.py
```
