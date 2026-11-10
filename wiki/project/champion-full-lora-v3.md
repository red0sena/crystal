# 보강 데이터 전체 LoRA 학습: 길이·GPU 실측과 실행

실행 위치는 `runs/champion-v1/full-g7-lora-v3`다. 텍스트·비전 모두 1B 이하 1위가 목표이며, 이 실행의 완료나 내부 지표를 공식 비공개 순위로 해석하지 않는다. 원본 Qwen3.5-0.8B에서 CE LoRA를 학습한다. 회전 정렬은 이번 본학습에 포함하지 않는다.

## 11-10 현재 요약 (11-05~11-10 흐름)

| 날짜 | 한 일 | 결과 |
|---|---|---|
| 11-05~06 | 보강 데이터 전체 LoRA(v3) 1 epoch | 공개 텍스트 18.76으로 크게 하락, 사용자 요청으로 중단 |
| 11-07 | 하락 원인 대조 R1(v2 배합)·R2(v3 배합), 각 16k update | 원인은 배합이 아니라 길고 식지 않은 학습 일정. R1 text-dev 75.73, 공개 텍스트 33.35 |
| 11-08 | R3: R1 + 가전·화음 합성 각 2% | 합성은 전이(POP909 .53, Home .31)했지만 yes/no 쏠림으로 BFCL·API-Bank 손해, 공개 텍스트 33.54 |
| 11-08 | 평가 형식 재구성(acos·isarcasm), 순위 손실 시험 | 순위 손실은 버그 수정 후 유의하지 않아 미채택 |
| 11-08 | 데이터 확장 data-scale-v1 | 공인 공개 데이터 55개 추가, 선택지 수 비율 유지, 2,070,120행 |
| 11-08~10 | g7 방식 full FT(fullft-scale-v1), 3 epoch 계획 | 11-08 OOM(깨진 23.9만 토큰 phishing 행) → 65,536 토큰 초과 9행만 건너뛰고 재개 → 11-10 사용자 요청으로 update 94,208(1.46 epoch)에서 중단 |
| 11-10 | best(update 81,920) 소표본 공개 비교 | 아래 표 |

현재 최선 후보는 `runs/champion-v1/fullft-scale-v1/best/`(update 81,920, text-dev macro 0.8068; R1 0.7573)다.

| 소표본 비교(공식 지수 아님) | best | Crystal(배포) | 1위 | R1 | v2 |
|---|---|---|---|---|---|
| 텍스트 37개 × 40행, 정확도 평균 | 0.622 | 0.629 | GLiClass 0.666 | 0.595 | 0.605 |
| 이미지 11개 × 60행, 정확도 평균 | 0.592 | 0.639 | QEV 0.645 | 0.673 | 0.635 |

- 텍스트: Crystal과 통계적 동률(−0.7 [−2.7, +1.3]), 1위보다 −4.5 [−6.9, −2.0]. 1위와의 격차는 VAST·ANLI·POP909·피싱·Home·ToolRet·BRIGHT에 몰려 있다.
- 이미지: 텍스트로만 학습해 Crystal·QEV보다 4~5점 낮다(Mind2Web·CV-Bench 큰 격차). 다음 순서는 best 위에 이미지 LoRA를 얹는 것이다.
- 원칙은 그대로다: 공개 문항은 학습 라벨·체크포인트 선택에 쓰지 않고, 교사 soft-label distillation도 쓰지 않는다.

## 데이터 전체를 GPU에 넣는가

전체 파일은 디스크에 두고 CPU 작업 프로세스가 인덱스로 필요한 행을 읽는다. GPU에는 모델·LoRA·optimizer/master와 **현재 마이크로배치의 계산값**이 들어간다. 텍스트16개·이미지4개로 한 update를 만들되, 길이순으로 마이크로배치를 나누어 각각 backward하고 gradient를 누적한 다음 optimizer를 한 번 갱신한다. 모든20행의 전체 계산 그래프를 동시에 유지하지 않는다.

`token_budget=4096`은 여러 문항을 묶는 토큰 예산이다. 개별 문항이 이보다 길면 단독 마이크로배치로 처리하며 문항을 자동으로4,096토큰에 자르지 않는다. 기존 도우미 `train.microbatches`의 동작을 그대로 재사용한다.

## 입력 길이 전수 확인과 수정

원래 준비된 텍스트1,130,546행을 모두 실제 프롬프트로 렌더링·토큰화했다. 최장387,425토큰은 새 피싱 보강의 원본 메일에서 나왔다. 이 메일의 본문은1,025,016자로 대규모 수신인 나열을 포함한다. 다른 극단 메일에는 다수의 메시지 헤더가 포함되어 있었다. 이 문제가 모든 긴 문서를 불필요하다고 뜻하지는 않는다.

[Qwen 공식 모델 카드](https://huggingface.co/Qwen/Qwen3.5-0.8B)는 기본 문맥 길이를262,144토큰으로 명시한다. 이 숫자는 해당 길이의 fine-tuning이 현재 GPU에 들어간다는 보장이 아니며, 우리 모델의 attention 수정 후 성능 보장도 아니다.

새 이메일 보강분 중65,536토큰을 넘는3개 원본의9개 질문을 검토 대상으로 보관했다. 추가 이메일에만 적용하며 기존 g7 핵심 데이터는 변경하지 않았다. 통째로 분리했으므로 문서의 오른쪽을 자르거나 서로 다른 청크에 전체 이메일 라벨을 복사하지 않는다. 원본 준비 파일은 보존하고 별도의 `inputs/text-train-reviewed.jsonl`과 수정 인덱스를 만들었다. 변경 근거와 제외 행은 `input-review.json` 및 `inputs/withheld-email-views.jsonl`에 있다.

최종 학습 텍스트는 **1,130,537행**, 이미지 **55,656행**이다. 텍스트는2개 파일(검토 후 본문 파일+별도 URL 보강 파일)을 모두 읽는다. 수정 후 길이 분포는 다음과 같다.

| 길이 통계 | 토큰 |
|---|---:|
| 중앙값 | 211 |
| 90백분위 | 768 |
| 99백분위 | 약4,886 |
| 99.9백분위 | 16,129 |
| 최장 | 49,552 |

4,096토큰 초과13,534행,16,384토큰 초과1,098행,32,768토큰 초과6행이다. 이 행들은 남겨 두었다. 학습 이미지 프롬프트의 최장은3,402토큰이며 모든 이미지를 실제 processor 경로로 렌더링했다. g7 dev33,821행·URL dev4,218행·이미지 dev11,417행도 모두 CPU 렌더링했다. 증거는 `prepared.json`과 `length-memory-analysis.json`이다.

## 실제 GPU 메모리 측정

RTX3090에서 다른 작업을 유지한 채 실측했다. 장치 총량은23.55GiB, 모델 로드 전 사용 가능량은15.70GiB였다. BF16 원본+학습할 LoRA+FP32 masters의 초기 allocated는약1.69GiB다. optimizer 상태는 첫 update 때 추가되며 아래 측정은 loss/backward/optimizer step까지 포함한다. 측정용 업데이트는 버리고 본학습은 다시 원본에서 시작한다.

독립 메모리 probe(`memory-probe.json`)의 결과:

| 입력 | 실제 토큰 | 피크 allocated(GiB) | 결과 |
|---|---:|---:|---|
| 텍스트16+이미지4, 마이크로배치 누적 | 합계13,337 | 8.73 | 통과 |
| 텍스트 단독 | 4,096 | 9.82 | 통과 |
| 텍스트 단독 | 8,192 | 2.98 | 통과 |
| 텍스트 단독 | 16,364 | 4.13 | 통과 |
| 텍스트 단독 | 32,480 | 6.38 | 통과 |
| 실제 최장 정상 문서 | 49,552 | 8.78 | 통과 |

4,096토큰을 **초과**하면 activation checkpointing을 켜므로8k가4k보다 작게 나왔다. 같은 설정의 단순 길이 대비 곡선으로 해석하지 않는다. `allocated`는 이 PyTorch 프로세스의 tensor 메모리이며 다른 프로세스나 allocator 여유분은 포함하지 않는다. 최장 문서의 reserved는10.59GiB였다. 첫 측정 시간에는 컴파일·워밍업이 섞이므로 전체 학습 ETA로 직접 쓰지 않는다.

정리한 입력으로 전체 실행기의 GPU 사전 점검도 별도로 통과했다. 최대 텍스트·최대 이미지·새 URL·Mind2Web 사례를 처리했고, 원본 대비 zero-LoRA 차이는0, 고정 가중치 불변과 native/packed 예측 일치를 확인했다. 이 점검 전체의 피크 allocated는9.84GiB였다. 결과는 `check.json`이다. 한 번의 probe는 모든 가능한 배치 조합이나 이후 GPU 사용량 변화를 보장하지 않는다.

## 현재 방식은 일반적인 장문 학습인가

메모리 절약 기법들은 보편적이다. [PyTorch2.10 체크포인팅 문서](https://docs.pytorch.org/docs/2.10/checkpoint.html)는 계산을 다시 수행하는 대신 저장하는 중간값을 줄이는 방식을 설명하며 `use_reentrant=False`를 권한다. 현재 코드도 그 설정을 사용한다.

[PyTorch variable-length attention 튜토리얼](https://docs.pytorch.org/tutorials/intermediate/variable_length_attention_tutorial.html)은 샘플 경계를 나타내는 누적 길이와 패킹으로 padding 계산을 줄이는 방식을 설명한다. 우리는 문항 경계를 유지하고 기존 검증된 varlen 경로를 쓴다. 단순히 문항을 이어붙여 다른 문항을 참조하게 만드는 방식이 아니다. [FlashAttention-2 논문](https://arxiv.org/abs/2307.08691)은 효율적인 attention이 메모리를 절약해도 긴 문맥의 계산 비용 자체가 사라지지 않는다는 근거다.

[Hugging Face의 메모리 절약 문서](https://huggingface.co/docs/trl/main/en/reducing_memory_usage)는 길이 분포 확인, LoRA, packing과 truncation/splitting의 차이를 설명한다. 라이브러리의 기본 `bfd` packing이 긴 문서의 꼬리를 버릴 수 있다는 점에 주의한다. 현재 도우미는 그와 달리 긴 문항을 단독 처리한다. 언어 모델링에서 모든 토큰을 chunk로 보존하는 방식도 전체 문서를 보고 하나를 고르는 현재 과제에 그대로 적용할 수는 없다. 이것은 과제 구조에 따른 판단이다.

이번 학습은 새로운 문맥 길이 확장 학습이 아니다. 이미 장문을 지원하는 backbone을 선택형 판단에 적응시킨다. 표준 next-token SFT와 다른 부분은 여섯 full-attention 층의 양방향 설정, causal DeltaNet 유지, 마지막 hidden state에서 선택지 코드 점수만 읽는 CE다. 전체 위치×전체 vocabulary의 logits를 만들지 않아 일반적인 생성 SFT의 메모리 수치를 그대로 적용해서는 안 된다. 이 부분은 저장소의 기존 Decision 방식이며 보편적인 Qwen 기본 학습과 구별한다.

## 본학습 일정과 지속 실행

텍스트 전체를2epoch 사용한다. 마지막9개 텍스트도 누락하지 않으므로 epoch당70,659update, 총141,318update다. 각 update에 이미지4개를 넣어 이미지 표시 횟수는565,272회, 이미지 풀은약10.16회 반복된다. 각 반복마다 독립적으로 섞는다. 이 비율은 텍스트 확대가 이미지 학습을 희석하지 않도록 이전의16:4비율을 유지한 것이며 최적 비율임을 입증한 결과가 아니다.

LoRA는 언어 모델186개 모듈에r16/alpha32/dropout.05, 학습 파라미터10,822,656개다. ViT·원본·tied embedding은 고정한다. LR2e-4, warmup2%, cosine to10%, FP32-master AdamW(.9,.95), weight decay.01, clip1을 사용한다. checkpoint는64update에서 처음 저장하고 이후4,096update·epoch 경계·최종에 저장한다. FP32 masters/optimizer/RNG를 함께 복원한다. 공개 점수로 checkpoint를 고르지 않는다.

서비스 `champion-full-lora-v3`가 입력 점검→극단 보강 입력 검토→GPU 점검→freeze→본학습→저장·복원 일치→내부 dev→텍스트37개·비전11개 공개 평가를 순서대로 실행한다. 사용자 linger가 활성화되어 앱 종료·로그아웃 뒤에도 서비스가 유지된다. 자동 재시작은 실패 시60초 뒤,2시간에3회로 제한한다. 다른 GPU 작업을 종료하지 않는다.

실시간 상태는 `pipeline.json`/`status.json`, 매 update 기록은 `train.jsonl`, 복구 지점은 `last-checkpoint.json`이다. `status.json`의 ETA는 초기16update를 제외한 최근 최대256update의 실제 시간을 사용하며 export·내부/공개 평가 시간은 별도다. 이것은 학습률 warmup2,827update가 끝났다는 뜻이 아니다.

## 정상 학습 시작 확인: 11월5일14:02 KST

실제 업데이트219/141,318를 확인했다. 기록된 모든 loss/gradient norm이 유한하고,64update 체크포인트의 파일 해시·학습 tensor 유한성·186개 LoRA B의 실제 변경·optimizer step64·CPU/CUDA RNG 저장을 검증했다. 원본·입력·실행 코드의 불변 해시도 다시 검증했다. 서비스는 active, 자동 재시작0회다. 상세 증거는 `training-start-confirmation.json`에 있다.

초기16update를 제외한203update의 평균은 회당1.176초다. 이 시점 예상 학습 종료는 **11월7일12:07 KST 전후**이며, 길이 분포와 다른 GPU 사용량에 따라 변한다. export·내부 dev·텍스트/비전 전체 평가는 그 뒤의 별도 시간이다. 최종 성능이나 순위 상승은 아직 확인하지 않았다.

## 11월6일: 첫 epoch 체크포인트와 Crystal 비교

사용자 요청에 따라 첫 epoch의70,659update 체크포인트를 별도로 평가한다. 해당 adapter SHA256은 `42f601b9f16b7c820440dd0424f09da336d402139df9d73d38a3ffc76ec986d0`, optimizer SHA256은 `00c6475533a89ad20a3b31841252fec23394ac0b88fc5115be9f5771d6faf365`다. 평가 입력은 기존에 고정한 text-dev33,821행, phishing-url-dev4,218행, image-dev11,417행 전체다. 일부 사례만 뽑은 점수로 전체 결과를 대신하지 않는다.

비교 조건은 이번 epoch1, [현재 배포 Crystal](https://huggingface.co/Kyunghwan-kim/crystal-0.8b/tree/cd943da732d0c26d88ae6dfd7daaacb7a86c8bdd), 원본 Qwen이다. Crystal은 이미지 학습 `clean-div`가 반영된 배포 revision이며, 과거 텍스트 보드가 평가한 `decision-index-0.4.1` 태그와 구별한다. 실제 다운로드 가중치 SHA256 `ec0b2fce3f7ea33036a850ce25062d8b21ee0fbc4b27a9d303b8d2626aa0b624`를 Hub LFS 해시와 대조했다. main을 실시간으로 따라가지 않고 revision을 고정한다.

Crystal 배포 토크나이저의 전처리·merges·special-token 설정은 현재 학습 초기값의 토크나이저와 다르다. 같은 문항·선택지 순서·d3 의도를 유지하되 각 모델의 native tokenizer/processor를 사용한다. Crystal의255개 answer code와 token ID가 배포 `crystal_config.json`과 일치하는지 확인한다. 배포 가중치에 다른 토크나이저를 강제로 붙여 비교하지 않는다. 이것은 실제 모델 간 비교이며 tokenizer까지 통제한 순수 학습 방식 비교는 아니다.

이번 epoch1은 FP32 optimizer masters를 forward에 직접 쓰지 않고 저장 당시 BF16 adapter 가중치를 정확히 복원한다. dropout/증강은 끄고, 모델별 dataset별 row/source 정확도 및 CE·KL을 기록한다. 내부 source 평균 정확도는 공식 benchmark별 F1이나 Jev 지수를 대체하지 않는다. Crystal의 과거 학습과 내부 dev가 완전히 독립적이라는 보장은 이 비교로 새로 증명하지 않는다. 공개 checkpoint 선택은 하지 않는다.

GPU 여유가약4.8GiB라 본학습을 동시에 실행하지 않는다. 첫 epoch 체크포인트 저장 직후 학습을 잠시 중지했고, 그 뒤 약30개의 미저장 업데이트는 재개 시 같은 RNG/optimizer에서 다시 계산한다. 평가 unit `champion-epoch1-eval-v3`의 종료 작업이 `resume-training.py`로 본학습 transient unit을 재생성한다. 평가 성공·실패·시간 제한 종료 모두 재개를 시도하며, 기존 학습 소스·계획·optimizer는 변경하지 않는다. 앱 종료 후에도 user service로 진행한다. 평가로 중단한 시간은 기존 종료 예상에 추가되므로 재개 후 ETA를 다시 확인한다.

평가 코드는 `qwen_decision/v3/champion_epoch1_eval.py`, 상태는 `runs/champion-v1/full-g7-lora-v3/epoch1-eval/status.json`, 고정 계획은 같은 디렉터리의 `plan.json`, 모델별 문항 예측과 summary는 각 arm 하위 폴더, 최종 비교는 `completion.json`이다. 실제 완료 파일이 나오기 전에는 성능 비교가 끝났다고 해석하지 않는다.

### 단일 정답 비율 재확인

원본 학습 파일 전체에 실제 `gold_targets` 변환과 이미지 `item_view` 규칙을 적용해 세었다. 텍스트1,130,537행 중1,129,932행(99.9465%)이 one-hot이고, 이미지55,656행은 모두 one-hot이다. 80:20 풀 가중 평균의 target entropy는0.0003158에 불과하다. 이는 최근 학습 batch를 그대로 재현한 측정은 아니지만, loss약0.64를 복수 정답의 불가피한 하한으로 설명할 수 없음을 보여준다. 증거는 `target-entropy-analysis.json`이다.

## 11월6일: 현재 1B 이하 Full 선두와 epoch1 공개 전체 비교 추가

사용자 요청에 따라 Crystal 비교에 더해 텍스트·이미지 선두와 비교한다. 11월6일13:48 KST에 [보드 원본](https://huggingface.co/spaces/multimodalart/jev-decision-index/tree/f0dbd5521d35156e694dc5ba43d553d43747681c/data)을 다시 받았다. 엄격히 `0 < served_params/params <=1,000,000,000`을 적용하고 text `v03.v`와 vision `full`로 정렬하면 텍스트 선두는 GLiClass v4 edge(439,790,595개, Full31.21, Public47.49), 비전은 QEV-0.8B(852,985,920개, Full44.81, 보드 Public52.73)다. 점수순 선두이며 통계적 동률을 새로 해결한 순위는 아니다. snapshot revision은 `f0dbd5521d35156e694dc5ba43d553d43747681c`, 증거는 `epoch1-leader-eval/current-leaders.json`과 `board/data/*.json`이다.

이번 epoch1은 공식 공개 텍스트37개 패널110,087요청과 비전11개 벤치마크12,210요청 전체에 적용한다. 기존 공개 키트의 metric·chance 보정·모달리티 가중치를 재사용한다. 서로 다른 모델의 내부 정확도를 공식 지수처럼 비교하지 않는다. 결과는 text `scores.json`, vision `vision-scores.json`, 과제별 선두 격차를 포함한 `comparison.json`에 남긴다.

GLiClass 기준은 이미 payload 전수 대조와 공식47.49 재채점을 통과한 저자 공개 전체 예측이다. 현재 GPU의 GLiClass 전체 장문 경로는 메모리 부족이 확인돼 있으므로 다시 잘라서 실행하거나 새 로컬 전체 실행으로 부르지 않는다. QEV는 이미 이 머신에서 accurate 모드로12,210요청 모두 처리한 결과(공개 지수52.8899508762)를 재사용하며, 전 요청의 run ID·payload hash·image hash를 이번 동일 suite와 대조했다. 보드 Public52.73과 로컬52.88995는 환경 차이를 포함한 서로 다른 실행값으로 구별한다. 모델을 학습 교사나 초기값으로 쓰지 않는다.

우리 epoch1은 저장된 BF16 LoRA를 원본 모델에 정확히 복원하고 unmerged 상태로 native padded SDPA 텍스트/비전 엔진에서 추론한다. 패킹 경로의 과거23/24 일치 실패를 무시하고 공개 평가에 패킹을 채택하지 않는다. 프롬프트는 d3, orders1, 증강/dropout 없음, 이미지 예산1,638,400px로 고정한다. 원본852,985,920개와 adapter10,822,656개를 합쳐 실제 추론 파라미터863,808,576개이므로1B 미만이다. FP32 optimizer masters는 추론에 쓰지 않으며 모델별로 입력 길이·온도·선택지를 평가 정답에 맞춰 조절하지 않는다.

기존 내부 비교 실행기는 수정하지 않았다. 그 서비스 종료 후 `resume-training.py`가 `epoch1-leader-eval/queued.json`의 공개 비교를 한 번 실행하고, 공개 비교 unit `champion-epoch1-public-v3`의 종료 작업은 `--resume-only`로 두 번째 epoch를 재개한다. 재개 루프를 막기 위해 followup 실행 시도 표식을 남긴다. 공개 비교가 시작되지 못하거나 실패·시간 제한 종료가 발생해도 기존 학습 재개를 시도하며, 실패 상태를 숨기지 않는다. 현재 내부 비교→공개 선두 비교→나머지 학습 순서이며, 평가 시간만큼 기존 ETA가 늦어진다.

실행 코드 `qwen_decision/v3/champion_epoch1_public.py`, 고정 계획 `runs/champion-v1/full-g7-lora-v3/epoch1-leader-eval/plan.json`, 상태 `status.json`, 대기열 `queued.json`을 확인한다. GPU 최악 입력 사전 점검과 본평가는 내부 비교가 GPU를 반환한 뒤 별도 프로세스로 진행한다. 공개 점수는 개발 중 관찰한 지표이며, 비공개 Full이나 공식1위를 의미하지 않는다. 목표는 계속 두 트랙 모두의1B 이하 Full1위다.

### 11월6일14시: 내부 평가 행 ID 검사 수정과 재개 순서

첫 epoch 모델의 text-dev33,821행 추론은 모두 완료됐으나, 집계 직전 전역 ID가 모두 고유하다는 잘못된 assertion에서 중단됐다. 실제 고정 dev에는 Fintweet2쌍·MNLI1쌍의 ID 충돌이 있으며 서로 다른 행이다. 문항을 제거하거나 ID로 합치지 않고 고정 파일의 행 번호(index)를 식별자로 사용한다. 모든 index가 정확히 한 번 존재하고 ID/source가 해당 행과 일치하는지 확인했으며, 저장 logits에서 정답·CE·KL까지 다시 계산해 검증했다. 예측 파일 해시는 변경하지 않았다.

복구한 epoch1 text-dev는 행 정확도67.6473%, source macro 정확도66.8361%, CE0.7565767이다. 이것은 고정 내부 dev의 점수이며 Crystal/원본 비교나 공개 지수는 아직 아니다. 학습 중 증강·dropout이 켜진 혼합 loss와 직접 같은 조건의 값으로 취급하지 않는다. 이전 실패 소스·계획·traceback은 `epoch1-eval/failed-before-row-index-fix*`로 보존하고, 수정 근거와 복구 파일 해시는 `row-index-repair.json`에 기록했다. 중복 ID의 서로 다른 정답을 유지하는 검사 및 중복 index/손상된 cached loss를 거부하는2개 회귀 검사가 통과했다.

공개 선두 비교는 내부 집계 실패 후 예정대로 시작됐고 계속 실행 중이다. 이를 중단하거나 GPU 평가를 동시에 겹치지 않는다. 공개 비교 종료 후 `epoch1-eval/repair-queued.json`이 남은 epoch1 URL/이미지와 Crystal·원본 전체 dev 비교를 수행한다. 완료된33,821개 예측은 검증된 해시 그대로 재사용해 다시 추론하지 않는다. 이후 `resume-training.py --training-only`로 나머지 학습을 재개한다. 따라서 현재 순서는 **공개 선두 비교→남은 내부 비교→두 번째 epoch 재개**이며 이전 순서 설명을 갱신한다.

## 11월6일: 사용자 요청으로 전체 중단 후 성능 하락 조사

17:48 KST의 사용자 요청에 따라 본학습·공개 평가·내부 비교를 중지했다. `pause-request.json`을 두고 재개 실행기의 모든 경로보다 앞에서 종료하도록 했으며, 남은 대기열을 `*-disabled-by-user.json`으로 보관했다. 서비스 상태를 읽어 현재 프로젝트의 실행 PID가 없음을 확인했다. 중단 후에는 아래 소수 문항의 forward-only 진단만 수행했고 optimizer·추가 학습·전체 평가·모델 업로드는 실행하지 않았다. 진단이 끝난 뒤 GPU 모델도 해제했다. 직접 재개 스크립트를 실행하는 점검은 자동 승인 검토에서 거부됐으며, 파일·서비스 읽기 확인으로 대체했다.

### 실제 성능 저하와 평가 구현 점검

- 이번 epoch1 공개 텍스트37개 패널110,087요청은 모두 ok이며 공식 공개 지수18.76이다. 이전 v2는35.19, 고정 GLiClass 공개 참조는47.49다. 비전은7,177/12,210요청 상태에서 중단됐으며 전체 비전 지수는 산출하지 않는다.
- 내부에서 state·question·실제 gold가 모두 동일한3,551문항을 맞췄다. 원본49.51%, 이전 v2 unmerged74.85%, 이전 v2 merged74.74%, 이번 epoch1 66.74%다. 단순히 서로 다른 검증 문항을 비교해서 생긴 하락은 아니다. raw target이 달랐던22행도 실제 gold 정규화 후 일치했다.
- 35개 공개 패널의 짧은105개 prompt에서 현재 checkpoint의 eager 단독/패딩 배치/패킹 배치 선택이 모두 일치했다. compile 여부는 단독 경계 사례1개 차이, 배치/패킹 선택은 모두 일치했다. 별도 긴 prompt6개에서도 현재 모델의 세 경로 선택이 모두 같았다. 이는 유한 진단이며 모든 요청의 수치 동일성을 증명하지 않는다.
- 고정 내부 진단855문항에서 LoRA unmerged/merged 정확도는 모두65.61%, 문항별 선택 일치는99.53%다. merge 유무가 큰 하락을 설명한다는 증거가 없다. 실제 forward BF16 adapter·step70,659·원본/adapter/optimizer 파일 해시를 다시 확인했다.

### 확인된 설계 변화와 유력한 원인

전체 데이터 사용으로 확대하면서 이전의 출처·클래스 균형 선택을 유지하지 않았다. 출처 비중은 아래처럼 달라졌다. 행 수를 늘렸다는 사실과 전체 loss에서 차지하는 기여를 구별해야 한다.

| 출처 | 이전 텍스트 행 비중 | 이번 비중 |
|---|---:|---:|
| WANLI | 1.12% | 9.87% |
| API-Bank | 2.24% | 0.25% |
| CLINC | 2.24% | 0.78% |
| BANKING77 | 2.24% | 0.85% |
| HoVer | 2.24% | 0.71% |

WANLI 라벨 오류를 발견했다는 뜻은 아니다. HellaSwag처럼 비중이 증가했어도 성능이 떨어진 과제가 있으므로 데이터 배합 하나로 모든 하락을 설명하지 않는다. 피싱은 공개 정확도가53.50%→74.05%로 개선돼, 보강이 모든 과제에서 무효였던 것도 아니다.

이전 v2는16,384update 후 감쇠가 끝났지만 이번 epoch1만70,659update다. 최고 LR2e-4를 유지하면서 일정을 늘려, 이 시점의 LR은1.129e-4로 이전 최종2e-5보다 높다. LR 일정 합은 이전 전체 대비6.52배다. 이 합은 Adam의 실제 이동 거리와 같지 않은 proxy다. 별도로 계산한 LoRA delta의 그룹별 Frobenius norm은 원본 대비 이전 DeltaNet/MLP/full-attention 약15.3/20.0/14.8%에서 이번59.1/69.2/59.5%로 커졌다. 큰 변화량만으로 나쁘다고 판단하지 않고 아래 동일 문항 간섭으로 점검했다.

| 같은 내부 진단855문항 | 정확도 | CE |
|---|---:|---:|
| 이전 v2 최종 | 75.09% | 0.651 |
| 이번 epoch1 그대로 | 65.61% | 0.781 |
| 이번 adapter 강도0.50 | 69.12% | 0.698 |
| 이번 adapter 강도0.75 | 69.47% | 0.711 |

강도0.75는 기존 오답62개를 고치고 기존 정답29개를 틀려 순33개를 개선했다. 같은 checkpoint의 적용 강도만 바꾼 결과이므로 지나친 adaptation이 이 진단에서 손해를 준다는 증거다. 다만 이전 v2까지 회복되지 않았고, 강도를 공개 지수나 새 배포 정책으로 채택하지 않았다. 학습률·배합 각 원인의 기여율은 이 진단만으로 분리할 수 없다.

checkpoint 진단은4,096회62.69%,16,384회61.64%,32,768회64.44%,49,152회63.74%,65,536회66.20%,70,659회65.61%다. 이전과 같은16,384회에서도 이미 약해져 있으므로 마지막 순간에 갑자기 붕괴했다고 부를 근거는 없다. 이번은2epoch 일정 중간이며 낮은 LR로 마무리하는 단계도 아직 지나지 않았다. 중단한 나머지 epoch가 어떤 결과였을지는 모른다.

이전에는 준비 시 opaque option key를 보존 의미와 함께 쓰는46,788행, context를 question으로 옮긴44,722행이 있었다. 새 builder는 기존 `presentation` 처리를 호출하지 않는다. 표준 옵션·code 순서 증강은 여전히 켜져 있다. 다만 같은855문항의 표현 대조에서 이전 v2는75.09→74.04%, 이번은65.61→66.08%로 큰 추가 저하가 없었다. 이 누락은 설계 변경이지만 현재 하락의 주원인이라는 증거는 약하다.

### 결론과 중단 상태

현재 가장 유력한 설명은 **전체 데이터로 확대할 때 바뀐 출처/클래스 배합과 그 규모에 맞추지 않은 LoRA 학습률·감쇠 일정이 함께 작용한 것**이다. 평가 구현만의 문제나 LoRA라는 학습법 자체의 필연적 한계로 단정할 근거는 없다. 같은 rank16/all186 구조로 이전 모델은 더 높은 성능을 냈다. rank 용량의 영향, 단일 seed 편차, 배합·학습률 각각의 인과 기여는 새 controlled training 없이 분리하지 못했다.

전체 행을 쓰면서도 출처별 loss 기여를 설계하고, 업데이트 수에 맞춘 LR/감쇠와 고정 내부 검증을 먼저 확인하는 수정 설계가 필요하다. 데이터 규모·배합·일정을 동시에 바꾼 뒤 한 epoch 동안 loss만 추적한 설계·검증이 충분하지 않았다. 추가 epoch나 full fine-tuning을 자동 해결책으로 채택하지 않는다. **사용자 중단 상태를 유지하며 새 학습·전체 평가·자동 재개를 시작하지 않는다.**

구조화 보고서와 재현 스크립트·원본 JSON 증거는 `runs/champion-v1/full-g7-lora-v3/cause-investigation/`에 있다. `report.json`, `matched-dev-comparison.json`, `data-and-optimization-audit.json`, `checkpoint-curve.json`, `adapter-strength-diagnostic.json`, `adapter-norm-growth.json`, `presentation-diagnostic.json`, `merge-diagnostic.json`, `path-probe-*.json`, `long-path-probe-eager.json`을 함께 확인한다. 진단 부분집합은 이미 관찰한 개발 데이터이며 새로운 공식·blind 성능 결과가 아니다.

## 11월6일: 웹 문헌을 대조한 복구 설계 — 제안이며 미실행

사용자의 웹 검색 요청에 따라 원 논문·저자 보고서와 기존 위키를 대조했다. 아래는 실행 승인을 받거나 성능을 확인한 새 레시피가 아니다. `pause-request.json`과 서비스 중단을 유지하며, 기존 학습 코드·입력·checkpoint·자동 재개 설정을 변경하지 않았다. 구조화 제안은 `cause-investigation/recovery-research-plan.json`이다.

### 문헌이 뒷받침하는 범위와 우리에게 남은 가설

| 근거 | 확인한 내용 | 이번 프로젝트에 적용할 판단 |
|---|---|---|
| [Data Mixing Optimization for SFT, 2025](https://arxiv.org/html/2508.11953v1) | 검증 loss를 바탕으로 SFT 데이터 배합을 최적화한다. Qwen2.5-0.5B를 포함한 실험이 있다. | 행 수 비례가 최적 배합이라는 가정을 버릴 근거다. 생성 SFT 결과이며 우리 Decision CE의 최적 가중치를 알려주지는 않는다. |
| [DoReMi, NeurIPS 2023](https://arxiv.org/abs/2305.10429) / [DoGE, ICLR 2024](https://arxiv.org/abs/2310.15393) | 작은 proxy로 도메인 비중을 학습해 큰 모델의 사전학습에 적용한다. | 데이터 규모와 출처 중요도를 구별하는 참고다. 여기서 제안하는 고정 가중 CE는 이 알고리즘들을 구현한 것이 아니다. 새 proxy 학습은 첫 복구 단계에 추가하지 않는다. |
| [(How) Learning Rates Regulate Catastrophic Overtraining, COLM 2026](https://arxiv.org/html/2604.13627v2) | 여러 1B 모델의 SFT에서 같은 train loss에 도달해도 높은 fine-tuning LR가 더 큰 OOD 성능 손실을 보였다. | 평균 train loss 감소만으로 성공을 판단하지 않고 낮은 LR를 대조한다. Qwen3.5 hybrid Decision LoRA를 직접 실험한 결과는 아니며, 우리에게 최적인 LR를 확정하지 않는다. 논문의 사전학습 LR 감쇠에 관한 결과와 fine-tuning LR 결과를 혼동하지 않는다. |
| [LoRA Without Regret, Thinking Machines, 2025](https://thinkingmachines.ai/blog/lora/) | 충분한 rank와 MLP를 포함한 배치가 중요하다. 학습이 진행되며 A/B의 상호작용도 바뀐다. rank 비교는 alpha=32를 고정했다. | 현재 all186 배치를 유지한다. 데이터가 커진 만큼 r16 용량 부족도 열린 가설이다. 기존 모델이 같은 r16으로 더 좋았다는 사실만으로 이 가능성 전체를 배제하지 않는다. |
| [LoRA Learns Less and Forgets Less, 2024](https://arxiv.org/abs/2405.09673) | 코드·수학의 일부 instruction/continued-pretraining 조건에서 낮은 rank가 full FT보다 덜 학습하고 덜 잊었다. | LoRA가 항상 충분하다거나 full FT가 항상 우월하다는 일반화 모두 피한다. |
| [mSFT, AACL 2026, v7](https://arxiv.org/html/2603.21606v7) | 하위 데이터셋마다 과적합 시점이 달라 조기 과적합 출처를 제외하고 checkpoint로 되돌리는 방식을 제안한다. Qwen2.5-0.5B도 평가했다. | 과제별 dev 곡선을 측정한다. 정적 비중 수정만으로 충분하다고 보장하지 않는다. 논문 방식은 출처 제외·rollback을 포함하므로 전체 데이터 사용 요구를 유지하는 첫 대조에 그대로 도입하지 않는다. |
| [WiSE-FT, 2022](https://arxiv.org/abs/2109.01903) | zero-shot/학습 후 가중치 보간으로 분포 이동 강건성을 개선했다. | 우리의 adapter 강도 완화 진단과 관련은 있지만 CLIP 계열 결과를 Decision 모델의 보장으로 옮기지 않는다. 855개 텍스트에서의 개선은 이미지·공개 전체 성능 개선을 입증하지 않는다. |

문헌을 종합하면 우선순위는 **출처별 목표 기여도와 LR를 분리해 대조하고, 과제·모달리티별 검증으로 확인하는 것**이다. 이번 회귀의 인과 기여율은 아직 확정하지 못했다. 더 많은 seed는 잘못된 레시피 자체를 고치지 않으며, 유망한 후보를 찾은 뒤 재현성을 확인하는 데 쓴다.

### 1. 전체 행을 유지하는 출처별 가중 CE

텍스트는 1,130,537행을 유지하고, 각 epoch에서 전부 한 번씩 읽는다. 출처 s의 현재 텍스트 행 비중을 p_s, 이전 v2의 텍스트 출처 비중을 b_s로 두고, 첫 대조의 목표 비중을 다음처럼 제안한다.

```
q_s = 0.5 * p_s + 0.5 * b_s
w_s = q_s / p_s
L = 0.8 * mean_text(w_s * CE) + 0.2 * mean_image(CE)
```

b_s는 현재 입력에 존재하는 출처에서 다시 정규화한다. 실제 감사 데이터에서는 이전 131,072행의 모든 출처가 현재 입력에 있어 재정규화 후 총량이 그대로다. 새 URL 출처도 p_s 항 때문에 양의 기여를 유지한다. 이 50:50은 **검증할 초기 가설**이며 문헌이 제시한 최적값이 아니다. 이전 출처 비중은 참고 prior일 뿐 공식 지수의 과제별 가중치와 같지 않다.

| 출처 | 제안하는 행당 CE 가중치 |
|---|---:|
| WANLI | 0.557 |
| API-Bank | 4.990 |
| CLINC | 1.935 |
| BANKING77 | 1.810 |
| HoVer | 2.074 |
| 새 phishing URL | 0.500 |

113개 출처 전체에서 가중치는 0.500~5.158이며 현재 행 분포에서의 기대값은 1이다. 이미지 가중치와 16:4 표시 비율은 첫 대조에서 고정한다. 마지막 짧은 텍스트 batch도 두 모달리티 평균을 따로 계산한다. 마이크로배치마다 평균을 내어 다시 더하거나, 현재 batch 가중치 합으로 나누면 목표와 다른 loss가 될 수 있다. 전체 출처 통계로 고정한 w_s를 적용해 각 모달리티의 전체 update 행 수로 나눈 뒤 backward를 누적해야 한다.

이 식은 고정된 모델에서 목표 출처 분포의 평균 CE와 그 gradient를 맞추는 importance weighting이다. 유한 batch의 분산·gradient clipping·Adam 경로까지 재샘플링과 같다는 뜻은 아니다. clip 발생률과 출처별 가중/비가중 CE를 함께 기록한다. 원본 한 건에서 파생한 다수 질문은 독립 사례 수로 해석하지 않는다. 클래스 균형은 별도 요인이므로 첫 2×2 대조에는 추가하지 않는다.

### 2. 배합과 학습률을 분리하는 최소 대조

| 팔 | 텍스트 기여도 | 최고 LR | 목적 |
|---|---|---:|---|
| A | 현재 행 비중 | 2e-4 | 기존 v3 checkpoint 재사용 가능 |
| B | 위의 가중 CE | 2e-4 | 배합만 바꾼 효과 |
| C | 현재 행 비중 | 5e-5 | LR만 바꾼 효과 |
| D | 위의 가중 CE | 5e-5 | 두 수정의 결합과 상호작용 |

초기 예상 우선 후보는 D다. 5e-5는 탐색값이며 적정값을 확인한 결과가 아니다. A를 남겨 낮은 LR가 무조건 낫다는 결론을 전제하지 않는다. 모든 팔은 원본 Qwen에서 출발하며 기존 v2/Crystal/선두 모델은 비교 기준으로만 사용한다. 중단 checkpoint의 optimizer를 다른 LR 레시피에 이어 붙이지 않는다.

동일 seed20261106·동일 원본 초기화/LoRA 초기화·전체 입력 파일·행 순서·증강 키·all186/r16/alpha32/dropout.05·CE·16:4·optimizer·길이/이미지 예산·warmup2,827·전체 cosine horizon141,318을 고정한다. 첫 대조에서는 LR의 진폭만 바꾸고, 짧은 점검에 맞춰 cosine을 새로 끝내지 않는다. A의4,096/8,192/16,384 checkpoint를 활용하면 동일 대조군을 다시 학습할 필요가 없다. 새 실행기의 변경이 가중치1일 때 기존 loss/gradient·누적 방식과 일치하는지는 CPU 단위 검증으로 먼저 확인한다.

4,096·8,192update에서 실패 징후와 과제별 방향을 보고, 결론이 불명확하면 같은 예산으로16,384까지 비교한다. 이는 학습 레시피 진단이며 전체 데이터 학습 결과가 아니다. 짧은 구간의 승자가 한 epoch 후에도 이긴다고 보장하지 않는다. 불안정·명백한 양 모달리티 퇴보 외의 임의 조기 탈락 규칙은 쓰지 않는다. 선별한 후보는 같은 일정에서 전체 행 한 바퀴까지 검증하고, 두 번째 epoch의 필요 여부는 dev로 판단한다. 이 절차·중단 조건·검증 구분은 실행 전에 고정해야 한다.

### 3. 텍스트·이미지를 동시에 지키는 검증

전체 평균 train loss와 작은 855문항 진단만으로 checkpoint를 고르지 않는다. 기존 text-dev33,821·URL-dev4,218·image-dev11,417행에서 source macro 정확도/NLL, 주요 과제별 metric, 장문·unseen 구분을 함께 본다. 짧은 주기 검증은 고정하고 원본 사건/문서/이미지 단위로 그룹을 유지한다. 공정 비교에는 이전 v2와 같은 행·gold인 교집합도 병기한다.

원본·v2·Crystal을 참조하되, 내부 점수와 공개 Jev 지수를 섞지 않는다. 텍스트 개선만으로 이미지를 희생한 후보를 최종 채택하지 않는다. 두 트랙 Pareto 결과와 seed 편차를 보고 허용할 하락 범위를 실행 전에 정한다. 이미 관찰한 dev를 새 blind holdout이라고 부르지 않으며, 확인용 분할이 필요하면 기존 split 이력과 중복·오염을 먼저 점검한다. 후보를 정한 뒤 대표 대조와 후보를 총3seed 이상으로 확인한다. 공개 패널은 매 LR/가중치/seed의 튜닝 수단으로 반복 사용하지 않고, 선택을 고정한 뒤 전체 평가한다.

후보 전체 평가도 텍스트·이미지 각각 현재 선두와 비교해야 한다. 이전 v2 회복은 통과 지점일 뿐 목표 달성이 아니다. 비공개 Full은 공개 점수로 확정할 수 없다.

### 4. 이후에만 검토할 수정과 당장 반복하지 않을 실험

- 배합/LR를 통제한 뒤에도 학습·검증 양쪽에서 용량 제한 징후가 남으면 all186의 r16 대 r64를 비교한다. 문헌의 rank 비교처럼 alpha32를 고정하는 설계를 우선 검토하고 LR를 확인한다. 예전 attention-r160은 alpha도320으로 커졌으므로 순수 용량 대조가 아니었다. r64는 adapter가 약4배가 되지만 현재 장문 GPU 적합성·속도는 실측하지 않았으므로 실행 가능/ETA를 단정하지 않는다.
- full FT는 자동 해결책으로 채택하지 않는다. 기존 [5-seed 위치 비교](lora-placement.md)에서 all-r16은 짧은69.4%/장문63.1%, full FT는66.6%/58.3%였다. 작은 데이터 조건의 결과이므로 큰 데이터에서 full FT 가능성을 영구 배제하는 뜻도 아니다.
- attention-only 전환, focal/Brier/label smoothing을 첫 복구에 넣지 않는다. 기존 [loss 비교](loss-screen.md)의 사전 등록16개 검정에서 CE를 이긴 후보가 없었고, focal은 짧은 문항을 해쳤다. 두 순서 consistency는 작은 사후 신호이고 계산량이 늘어난다.
- 누락된 preparation 표현 증강은 실제 차이지만 이번 대조 진단에서 주원인 지지가 약했다. 배합/LR 효과를 확인한 뒤 독립적으로 비교한다. 회전 정렬·reranking·추론 반복도 이번 퇴보 원인을 직접 고치는 근거가 부족해 첫 대조에 합치지 않는다.
- 과제별 과적합이 실제 dev 곡선에서 확인되면 mSFT식 출처 조절/rollback이나 동적 비중을 다음 후보로 검토한다. 현재 checkpoint의 강도0.5~0.75는 저비용 복구 기준일 수 있지만 아직 텍스트 소규모 진단 결과이며 두 트랙의 새 모델로 채택하지 않는다.

이번 검색으로 **“LR만 낮추면 해결” 또는 “rank만 늘리면 해결”을 확정하지 않았다.** 가장 적은 새 요인으로 배합·LR를 분리하고, 전체 데이터와 두 모달리티를 유지한 채 확인할 구체적 순서를 정리했다. 학습·전체 평가는 계속 중단 상태다.

## 11월6일: v2 레시피 고정 배합 대조 R1·R2 — 준비 완료, 미실행

2×2 대조 전에 v2 레시피에서 텍스트 출처 배합만 바꾸는 두 실행을 먼저 둔다. 문헌 대조 근거: 감쇠가 끝나지 않은 중간 checkpoint끼리의 비교는 최종 순위를 보장하지 않으며([Hägele et al. 2024](https://arxiv.org/abs/2405.18392)), SGD에서는 행별 재가중보다 재샘플링이 gradient 분산이 작다([Why resampling outperforms reweighting](https://arxiv.org/abs/2009.13447)). 그래서 출처별 CE 가중치 대신 출처별 샘플링을 쓰고, 두 팔 모두 16,384 update cosine을 끝까지 돈다.

| 공통 | 값 |
|---|---|
| 시작점 | 원본 Qwen3.5-0.8B, seed 20261106, 같은 LoRA 초기화 |
| 일정 | 16,384 update, LR 2e-4, warmup 512, cosine to 10% (v2 식) |
| 배치 | 텍스트 16 + 이미지 4, 텍스트 262,144회 추출, 이미지 v3 풀 55,656행 약 1.18회 (두 팔 동일 순서) |
| 구조·최적화 | all186 r16/alpha32/dropout.05, FP32-master AdamW, clip 1, v3 파이프라인 설정 |

| 팔 | 텍스트 배합 | 예: WANLI / API-Bank / CLINC |
|---|---|---|
| R1 | v2 출처 비중(112개 기존 출처), 새 URL은 자연 비중 1.58% 유지 | 1.10% / 2.20% / 2.20% |
| R2 | 전체 풀 자연 비중 (= v3 배합) | 9.87% / 0.25% / 0.78% |

출처 안에서는 섞은 순서로 중복 없이 뽑고, 다 쓰면 다음 순열로 넘어간다. R1에서 반복이 가장 많은 출처는 contractnli 2.13회, API-Bank 2.05회로 v2의 2 epoch와 비슷하다. 해석: **R1 대 R2 = 배합 효과**, **R2 대 v3 epoch1 = 학습 길이/LR 일정 효과**, v2 = 고정 기준점. v2에서 재현하지 않은 것: 출처 내 클래스 균형, presentation 변형, padding 선택지 pool, v2 이미지 풀. R1이 v2에 못 미치면 이 차이들이 다음 용의자다.

평가는 학습 후 병합하지 않은 BF16 adapter로, v3 고정 dev(text 33,821·URL 4,218·image 11,417행)에서 v2·R1·R2를 같은 채점기로 돌린다. v3 epoch1의 text-dev 예측은 재사용한다. v3 dev는 v2 학습 데이터와 정확 id/item/파일명 기준 겹침이 0건이다(근사 중복까지 증명한 것은 아님). 공개 패널로 checkpoint를 고르지 않는다.

- 코드: `qwen_decision/v3/mix_ablation.py`, 실행 관리 `mix_ablation_service.py`, 테스트 `tests/test_v3_mix_ablation.py` (7개 통과)
- 고정 계획: `runs/champion-v1/mix-ablation-v1/{r1,r2}/plan.json` (출처별 목표 비중·추출 수·반복 횟수 포함), 출처 색인은 원인 조사 감사 수치와 113개 출처 모두 일치
- GPU 확인: R1 update 2회 정상, 최대 9.7 GiB (저장 없음)
- 예상 시간: 학습 각 약 5~6시간 + 평가 3팔. 중지는 `runs/champion-v1/mix-ablation-v1/pause-request.json` 생성

### 11월7일 결과: 하락의 주원인은 학습 일정, 배합은 작은 추가 효과

R1 학습 7.6시간, R2 5.4시간, 평가는 10:09 KST에 완료됐다. 두 팔 모두 loss·gradient가 전부 유한했다. 같은 v3 dev와 같은 채점기, 병합하지 않은 BF16 adapter로 평가했으며 seed는 1개다.

| 모델 | text-dev 행 / source macro | URL-dev | image-dev 행 / macro |
|---|---:|---:|---:|
| v2 | 74.69 / 74.24 | 67.76 | 84.18 / 85.59 |
| **R1** (v2 배합, 16k) | **75.73 / 75.22** | 92.65 | 88.12 / 87.52 |
| R2 (v3 배합, 16k) | 74.97 / 74.07 | **93.67** | **88.30 / 87.70** |
| v3 epoch1 (v3 배합, 70,659 / 141,318) | 67.65 / 66.84 | — | — |

text-dev 짝 비교(McNemar 근사 z): R2 대 v3 epoch1은 5,067 대 2,591 (z 28.3), R1 대 R2는 2,690 대 2,433 (z 3.6), R1 대 v2는 2,938 대 2,587 (z 4.7), R2 대 v2는 2,862 대 2,768 (z 1.3)이다. source macro에서 R1−R2 = +1.15%p이고, 출처 단위 bootstrap 95% 구간은 0.30~2.17이다.

- **배합이 같은데 일정만 다른 R2와 v3 epoch1의 차이가 7.3%p다.** 하락의 대부분은 학습 길이와 감쇠 전 높은 LR(141k horizon 중간)로 설명된다. 데이터 자체나 LoRA의 한계가 주원인이라는 근거는 없다.
- 배합 효과는 실재하지만 작다(+0.8%p 행, +1.2%p macro). R1은 HoVer +17.5, API-Bank +8.0, BANKING77 +7.7에서 이기고, R2는 WANLI +7.3, TabFact +6.5에서 이긴다. 데이터 비중대로 해당 출처가 움직였다.
- 전체 풀을 쓴 R1·R2는 v2보다 텍스트가 같거나 낫고, URL(+25%p)과 이미지(+4%p)는 크게 낫다. URL·이미지 개선은 v2에 없던 보강 데이터 효과가 섞여 있다.
- 결론: 다음 후보는 **R1**이다. "더 오래 학습하면 나아지는가"는 이 실험으로 답하지 못했다. v3의 70k는 감쇠 전 상태였기 때문이다. 이 질문은 32k처럼 완결된 cosine이나 WSD 가지로 따로 확인해야 한다. 공개 지수·선두 비교와 추가 seed는 아직이다.

증거: `runs/champion-v1/mix-ablation-v1/eval/comparison.json`, 팔별 예측 `eval/{v2,r1,r2}/*.jsonl`, 학습 로그 `{r1,r2}/train.jsonl`, 최종 adapter `{r1,r2}/checkpoints/step-016384/`.

## 11월7일: 목표 기준 재점검과 다음 방향 — 기술 공백 메우기 (R3)

보드를 다시 받아 보니 텍스트 1B 이하 1위가 GLiDE 0.8B(Full 36.36)로 바뀌어 있었다. 텍스트 Full은 비공개 same·new로만 정해지고, 순위는 사실상 same에서 갈린다(1B 이하 상관 0.99). 공개 지수는 same과 상관 0.96이다([jev-index](jev-index.md) 갱신 절 참고).

**벤치마크별 차이 분석:** GLiDE와 crystal의 공개 원점수 차이는 몇몇 과제에 몰려 있다. POP909 +0.73, Home appliance simulator +0.65, VAST +0.31, PhishNChips +0.31, SGD +0.29, ACOS +0.27, iSarcasm +0.26, BBH +0.22, CLadder +0.16이다. 반대로 HellaSwag·ContractNLI·ESCI·WinoGrande에서는 crystal이 앞선다. 우리 파이프라인(v2·v3 모두)은 Home appliance 0.000, POP909 0.015를 받았고, 학습 출처에 두 과제가 없다.

- Home appliance(v2): 문항별 정답률은 상태 읽기 0.99, 대상 기기 0.74, 처리 방식·결과 0.39다. 한 사례의 문항 약 18개를 모두 맞혀야 점수를 받는 방식이라 0점이 됐다. 규칙을 적용하는 추론 실력이 부족하다.
- POP909: 음표에서 화음 pitch-class 집합을 읽는 능력이 없다.
- 공개 지수 산식(영역 가중, 영역 안 벤치마크 평균)으로 보면, 둘을 GLiDE 수준으로 올릴 때 공개 지수는 약 +4점(tools 18.3%·arts 10%) 오른다. 배합 조정의 +1보다 크다.

**R3 설계:** R1 배합에 프로그램 생성 기술 데이터 두 출처를 각 2%(5,243회) 더한다. 나머지 일정·seed·이미지·최적화는 R1과 같다.

- `home-policy-synth` 21,574행(3,000 장면): 우리 손으로 만든 독립 생성기다. 키트 생성기·가구·템플릿을 쓰지 않았다. 방·기기 종류(blinds·fan 포함)·이름·정책(all_or_nothing/best_effort, away 잠금)·요청 문구·질문 문구를 다양화했다. 공개 요청 27개와 정확히 겹치는 요청은 0건이다.
- `symbolic-chord-synth` 12,000행: 합성 진행·voicing·전위·5음 생략·비화성음으로 만든다. POP909 데이터는 쓰지 않았다. 127개 pitch-class 집합에 NoChord·Other를 더했다.
- 정답은 생성 규칙으로 정확히 계산된다. 단위 테스트 7개가 통과했다. dev는 다른 seed로 만들었고 학습 상태와 겹침이 0이다.
- 위험: 공개 지수는 텍스트 Full에 들어가지 않는다. 비공개 same 세트가 우리 생성기와 다른 형식이면 이득이 줄어든다. 그래서 공개 템플릿 복제가 아니라 기술 일반화를 목표로 생성기를 독립적으로 만들었다.
- 실행: `skill-mix-r3` 서비스는 R1 공개 평가가 끝나면 R3 학습 → 내부·합성 dev 평가 → 공개 텍스트·비전 평가 순으로 진행한다. 코드는 `qwen_decision/v3/skill_synth.py`, `skill_mix.py`, `skill_mix_public.py`, `skill_mix_service.py`, 테스트는 `tests/test_v3_skill_synth.py`, 계획은 `runs/champion-v1/skill-mix-v1/r3/plan.json`이다.
- 다음 후보 공백: SGD(학습 출처 없음), BBH·CLadder·MuSR·SATA·ChessBench(출처 없음), 출처는 있지만 낮은 VAST·iSarcasm·ACOS·PhishNChips(쏠림 의심).

## 11-08 R3 결과 (R1 + home-policy/chord 합성 각 2%)

- 내부: text-dev 75.74/75.32(R1 75.73/75.22), URL 93.39, image 88.45. 합성 dev: home 95.9%(R1 67.5%), chord 84.0%(R1 7.0%).
- 공개 텍스트 33.54(R1 33.35, v2 35.19), 공개 비전 52.81(R1 53.86).
- 전이 성공: POP909 0.015→0.530, Home appliance 0.091→0.307 (공개 키트 템플릿 미사용 생성기).
- 상쇄된 손실: BFCL 0.790→0.606(v2 0.858), API-Bank 0.518→0.331(v2 0.685), VAST 0.520→0.419. BFCL "no" 답 비율 v2 61%→R1 68%→R3 78%.
- 원인 가설: home 합성의 기기별 target yes/no 문항(no 65%)이 BFCL의 도구별 yes/no 형식과 같아 "no" 편향 전이. R1의 v2 대비 툴 손실은 별개(v3 풀 변화) — 미조사.
- 다음: membership 문항 yes/no 50:50 균형 + 툴 소스 노출 v2 수준 복원 후 재학습.

## 11-08 full FT(g7 방식) 3090 실측

`qwen_decision/v3/fullft_probe.py`, 결과 `runs/champion-v1/fullft-probe-v1/`. 언어 backbone 전체+임베딩 학습, FP32 master AdamW, 실제 v3 텍스트 행. probe 업데이트는 버림.
- 정적 메모리(가중치+master+moment) 적재 후 장문 단일 행: 8k 12.7GiB, 16k 12.7, 32k 14.7, 최장 49,552토큰 16.9GiB(reserved 20.8) — 모두 통과.
- 32행 무작위 업데이트 24회: token_budget 16k/ckpt 16k(g7 설정) OOM, 8k/8k OOM, **4k/4k 통과 10,279 tok/s, 1.64 s/step, peak 18.0GiB**, 8k/ckpt 4k 9,512 tok/s.
- 추정(토큰 가중): 1 epoch 14.3h, **2 epoch 28.6h**(텍스트 풀 113만 행 기준, 합성 4% 미포함, 중간 dev 평가 미포함).

## 11-08 GLiDE 학습법 조사와 ACOS 형식 점검

출처: [How to Train a SOTA Frontier Decision Model](https://fastino.ai/blog/training-glide-sota-decision-model)(10-22, 11-06 갱신), [GLiDE family](https://fastino.ai/blog/glide-decision-model-family)(11-05). 0.8B/0.4B의 base·데이터·레시피는 공개되지 않음.
- (1) 평가기와 같은 분해·자연 base rate로 학습 문항 재구성: ACOS 27.2→61.6, 약한 과제 oversampling만은 30.1. (2) 여러 교사의 soft target(가중 product of experts, 정답과 다르면 gold 우선). (3) 학생 자신의 closed-book 정답 여부로 "모름" 선택지 학습. 실패: pause token, 1B 미만 recurrent depth, 완성 모델에 짧은 패치 FT(숨은 셋 하락), bulk train split(공개↑ 숨은↓).
- 우리 ACOS 점검: 학습 acos-train은 리뷰당 2.23문항, yes/no 거의 반반(noul 3441/2573, choice 1920/1953). 공개 평가는 리뷰당 58.2문항(모든 category×sentiment)이라 대부분 no. GLiDE가 지적한 것과 같은 불일치이며 ACOS 0.128(GLiDE 0.580)의 유력한 원인.

## 11-08 공개 suite 문항 형식 분포 (형식 재구성의 근거)

- 공개 텍스트 질문 272,229개 중 **61.5%가 후보별 yes/no 분해 문항**(한 상태에 여러 후보를 각각 yes/no). 학습 데이터(1/20 표본)는 **21.7%**.
- 분해형 벤치마크: ACOS(요청당 58.2문항), ToolRet(32, nDCG), BRIGHT(32, nDCG), SATA(9.4), iSarcasm-B(6), BFCL(2.8, case exact) — 모두 1.00 binary. 이 중 ACOS·SATA·iSarcasm은 우리 점수가 매우 낮다.
- 함의: (1) 집합 선택 과제를 train split에서 같은 분해·자연 base rate로 재구성, (2) nDCG 과제는 한 상태 안 후보들의 yes 확률 순위가 점수이므로 후보 간 순위 손실(IR의 listwise) 검토.
- 이미 시험한 것(반복 금지): noncausal, MNTP, loss 6종(ls/focal/brier/twoview/consist/mntp), pointer·MLP·후보별 query head, EMA, 순서 평균(+0.7 짧은 문항/−1.25 장문), recurrent depth, 질문 머리말.

## 11-08 평가 형식 재구성 데이터 (format-rebuild-v1)

코드 `qwen_decision/v3/format_rebuild.py`, 테스트 `tests/test_v3_format_rebuild.py`(5개 통과), 출력 `runs/champion-v1/format-rebuild-v1/`. 사용자 결정: 형식 재구성은 쓰고 교사 분포 학습은 쓰지 않는다. train split만 쓰고 공개 문항과 같은 상태(정규화 텍스트)는 제외했다.

| 출처 | 행 | 내용 | yes 비율 |
|---|---:|---|---:|
| acos-format | 38,675 | NUSTM/ACOS train, 평가와 같은 질문 문장·도메인 전체 inventory에서 균일 음성(기존은 같은 카테고리 hard 음성) | 5.2% (Laptop 자연 0.35%, Restaurant 3.4%) |
| acos-format dev | 34,671 | dev 파일을 평가와 같은 전체 분해로(Restaurant 전체, Laptop 80리뷰), 리뷰별 F1용 | 0.95% |
| isarcasm-format | 28,972 | A(En·Ar) 전체 텍스트 자연 비율, B(En) 모든 텍스트×6 범주(기존은 풍자 텍스트 위주), C 쌍 | A 25%, B 5.0% |
| bfcl-format | 7,369 | tools-percall을 결정 시점별로 묶어 평가의 도구별 yes/no로, 후보 수는 공개 BFCL 요청당 후보 수 분포를 따름 | 46.4% (공개 약 35%) |
| toolret-format | 38,390 / dev 6,400 | 질의 = 마지막 사용자 발화, 후보 = 호출 도구 + 도구 9,081개 중 fts5 BM25 상위(평가와 같은 방식) 31개 | 3.1% |

- 학습 mix에서는 기존 acos-train, isarcasm-train을 이 출처로 바꾼다. tools-percall은 유지한다.
- 미완: SATA(train split 없음), BRIGHT형 추론 검색.

## 11-08 순위 손실 소규모 시험 (rank-pilot-v1)

코드 `qwen_decision/v3/rank_pilot.py`, 결과 `runs/champion-v1/rank-pilot-v1/`. 원본 Qwen + all186 LoRA r16, toolret-format 1,200질의(1질의=1업데이트, 32후보), 같은 순서·seed 41, 증강 없음. dev 200질의(학습 질의·정답 도구와 분리).

**첫 결과(CE 0.807 vs 0.835)는 무효.** `format_rebuild.percall_label`이 noul 목표 `[P(false), P(true)]`를 거꾸로 읽어 bfcl/toolret 라벨이 뒤집혀 있었다(BM25 상위 32개 안 정답 213/1,200질의). 고친 뒤 데이터를 다시 만들고(정답 포함 1,053/1,200) 두 팔을 다시 돌렸다. 무효 결과는 `rank-pilot-v1/invalid-noul-label-bug/`에 보관.

| 팔 (라벨 수정 후) | nDCG@10 | dev 정확도 | 예측 yes 비율(정답 3.1%) |
|---|---:|---:|---:|
| CE | 0.818 | 97.48% | 2.20% |
| CE + 1.0×listwise(후보 margin softmax) | 0.831 | 97.33% | 2.89% |

- 차이 +0.013, 질의 단위 paired bootstrap 95% [−0.008, +0.034], 나아짐 33 / 나빠짐 26. **유의하지 않음**(seed 1개). 지금 본 학습에 넣을 근거는 부족하다.

## 11-08 데이터 확장: 약 200만 행 (data-scale-v1)

사용자 요청: g7+보강 mix와 선지 개수 비율을 유지하면서, 공인 데이터셋을 다양한 분야에서 대량으로 더해 약 200만 행으로 만들기. 증강은 최소한으로.

코드 `qwen_decision/v3/data_scale.py`(출처별 변환기, `mix` 명령), 테스트 `tests/test_v3_data_scale.py`. 출처 파일 `runs/champion-v1/data-scale-v1/sources/`, 최종 mix `runs/champion-v1/data-scale-v1/mix/text-train.jsonl`, 출처별 수와 비율은 `mix/manifest.json`.

- 규칙: train split만, 데이터셋 자체 라벨만(사람 라벨; sarcasm-headlines만 매체 기준 programmatic), 모델 라벨이나 distillation 없음. 공개 suite + JEVal로 다시 만든 오염 필터(13-gram)를 모든 행에 적용. 기존 pool과 다른 신규 출처에 이미 쓴 상태는 다시 쓰지 않음. 내부 held-out(PIQA, SWAG, COPA, QASC, QuaRTz, WiC, ETHICS, TimeDial, epistemic, presupposition, CB), ANLI train, TREC 제외.
- 증강: 2지 분류를 40% 확률로 "○○인가?" yes/no로 묻기, 출처마다 우리 문장 2~3개. 큰 주제 데이터셋(Yahoo, DBpedia)은 다른 출처에 쓰지 않은 새 항목만 "○○ 주제인가?"로 물음(정답 50%).
- 1차 41개 출처(뉴스·위키 개체·리뷰 감정·트윗·NLI·패러프레이즈·독해·의료/과학 객관식·대화·초록 문장 역할 등)를 더하니 yes/no가 18.6%로 줄고 4지·51+가 넘쳤다. 2차 12개 출처로 채웠다: Civil Comments 독성(명확한 항목만), PAWS, Yelp polarity, SciTail, Yahoo/DBpedia yes/no, VitaminC, OCNLI, DREAM, emotion(unsplit), Amazon 별점, 트윗 감정 확대. medmcqa·mmlu-aux·aqua·dbpedia L2/L3는 앞부분만 사용.
- MultiRC는 문단의 대부분이 공개 suite 문단과 13-gram이 겹쳐 27,243행 중 236행만 남았다(정상 제외).
- 평가 형식 재구성 중 ACOS·iSarcasm만 넣고(기존 acos-train·isarcasm-train 대체), BFCL·ToolRet 재구성은 벤치마크 맞춤 우려로 넣지 않았다.

| 구성 | 행 |
|---|---:|
| 기존 pool(text-train + phishing-url) − acos-train − isarcasm-train | 1,102,880 |
| acos-format + isarcasm-format | 67,647 |
| 신규 공인 출처 55개 | 900,492 |
| **합계** | **2,071,019** |

| 선지 수 | 기존 g7+보강 | 새 mix |
|---|---:|---:|
| yes/no | 22.4% | 22.7% |
| 2지(기타) | 18.6% | 18.6% |
| 3 | 21.6% | 20.9% |
| 4 | 14.9% | 14.7% |
| 5–10 | 14.3% | 14.9% |
| 11–50 | 5.1% | 5.3% |
| 51+ | 3.1% | 3.1% |

- 40,000행 표본 인코딩 오류 0. 토큰 길이: 기존 pool 평균 465(p99 4,725), 신규 평균 205(p99 1,233, 최대 1,881).
- 한 epoch 약 7.2억 토큰. 3090 full FT 4k/4k 실측 10,279 tok/s 기준 **epoch당 약 20시간**(추정).
- dev는 기존 `full-g7-v3/mix/text-dev.jsonl` 그대로(checkpoint 선택용, 공개 점수로 고르지 않음).

## 11-08 data-scale mix로 full FT 3 epoch (fullft-scale-v1)

사용자 요청: "풀파인튜닝으로 3에폭까지". 코드 `qwen_decision/v3/fullft_scale_train.py`, 실행 `runs/champion-v1/fullft-scale-v1/`(`plan.json`, `train.jsonl`, `status.json`, `eval.json`, `run.log`).

- 데이터: `data-scale-v1/mix/text-train.jsonl` 2,070,120행. 학습 전 선택용 dev와 상태가 같은 신규 출처 행 899개 제거(SciTail 356, iSarcasm 236, BoolQ 141 등; 빈 상태·아주 짧은 상태는 제외 기준에서 뺌).
- 레시피(g7과 같음): 언어 백본 전체 + 묶인 임베딩 학습, 비전 고정. noncausal attention, causal DeltaNet. BF16 계산 + FP32 master AdamW(.9,.95, 행렬 wd .01, clip 1). LR 3e-6, update당 32행, warmup 5% 후 10%까지 cosine. gold CE. 증강 noul→choice .2, 코드 섞기 .2, 코드 샘플 .2, padding 없음.
- 4k 토큰 microbatch, 4k 넘을 때만 checkpointing. epoch당 64,692 update, 3 epoch = 194,076 update. 모든 행을 epoch마다 한 번씩, epoch마다 새로 섞음.
- 선택: `full-g7-v3/mix/text-dev.jsonl`(33,821행) dev macro-source 정확도. step 0, 16,384 update마다, epoch 끝마다 측정. 공개 점수로 고르지 않음. 저장: `best/`, `epoch-1/`, `epoch-2/`, `last/`.
- 재개: 2,048 update마다 FP32 master·AdamW moment·RNG 저장(최근 2개 유지, 각 8.5 GB). 배치와 증강은 (seed, step)으로만 정해져 재개해도 같은 입력. smoke 시험(6에서 중단 → 4에서 재개 → 8 완료) 통과.
- 속도: smoke에서 update당 0.82~0.85초(약 1만 토큰). 학습 전 원본 Qwen dev macro(400행 표본) 0.506.
- 시작(11-08 15:19): 전체 dev(33,821행) step 0 macro 0.4815, 전체 정확도 0.484. update당 약 1.0~1.1초.
- **11-08 18:08 OOM으로 중단(update 8,992).** 원인은 phishing 행 하나(238,607 토큰). 길이를 다시 재 보니 65,536 토큰을 넘는 행은 9개뿐이고, 모두 phishing의 깨진 이메일 문서 3개(115k·239k·387k 토큰)의 변형 3개씩이다. 하나는 메일링 리스트 덤프가 통째로 한 행에 들어간 것이고, 하나는 수신자 주소 약 100만 개가 본문인 스팸이다. 387k는 모델 위치 한도(262,144)도 넘는다. g7 학습(`train.py --max-tokens 65536`)도 이 9개를 건너뛰었다.
- 실제 긴 문서는 그대로 학습한다. 16k 토큰 넘는 행 995개, 32k 넘는 행 15개(lexglue 49.5k 등)는 한 행짜리 microbatch로 학습되고, 3090 probe에서 49,552 토큰까지 통과했다(최대 reserved 20.8 GB).
- 11-09 13:56 재개(8,192 체크포인트부터): `MAX_TOKENS=65536` 넘는 행은 자르지 않고 건너뛰고 `train.jsonl`에 `skipped_over_max_tokens`로 남긴다. 그래도 OOM이 나면 그 update만 버리고(optimizer step 없음) `oom_dropped_update`로 기록한다. `PYTORCH_ALLOC_CONF=expandable_segments:True`. 15시간 정지로 끝나는 시점은 약 11-11 오후로 밀렸다.
- dev macro / 전체 정확도: step 0 0.4815/0.484 → 16,384 0.7599/0.7669 → 32,768 0.7797/0.7862 → 49,152 0.7939/0.8002 → 64,692(1 epoch 끝) 0.8004/0.8067 → 65,536 0.8008/0.8068 → **81,920 0.8068/0.8134(best)**. 같은 dev에서 LoRA R1은 0.7573. 재개 뒤 OOM으로 버린 update 0번, 65,536 토큰 초과로 건너뛴 행 11번(2 epoch째 재등장 포함).
- **11-10 16:07 사용자 요청("학습 끝내줘")으로 update 94,208(epoch 1.46)에서 중단.** 94,208 체크포인트 저장 직후 SIGTERM. 남은 모델: `best/`(step 81,920, dev 0.8068), `epoch-1/`(step 64,692, dev 0.8004). 94,208 가중치는 `checkpoints/step-094208`에만 있고 export·평가하지 않았다. 중단 시점 LR은 cosine 중간이라 아직 내려가지 않은 상태다. 같은 명령으로 다시 실행하면 94,208부터 이어진다.

### 11-10 best(81,920) 공개 텍스트 소표본 비교

사용자 요청("전체 말고 일부만 작게"): 공개 텍스트 37개 벤치마크마다 정답 있는 행 40개씩(seed 20261110), 1,480요청. best/는 R1과 같은 엔진 정책(d3, orders 1, noncausal, 8k batch token)으로 실행했고 1,480개 모두 ok, 245초. 나머지 모델은 저장된 전체 결과에서 같은 요청만 가져왔다(GLiClass는 저자 공개 예측). 지표는 문항 최상위 선택 정확도(noul은 p>.5), 벤치마크 37개 평균. 공식 지수(chance 보정·영역 가중)가 아니고 학습·선택에 쓰지 않는다. 코드 `qwen_decision/v3/fullft_sample_text.py`, 결과 `runs/champion-v1/fullft-scale-v1/public-sample/comparison-37.json`.

| 모델 | 37개 평균 정확도 | best와 차이(95% bootstrap) |
|---|---|---|
| GLiClass v4 edge (≤1B 텍스트 1위, 공개 47.49) | **0.666** | best −4.5 [−6.9, −2.0] |
| **best (fullft 81,920)** | 0.622 | — |
| v2 joint-g7-v2 (공개 35.19) | 0.605 | best +1.7 [−0.5, +4.0] |
| R1 LoRA (공개 33.35) | 0.595 | best +2.6 [+0.5, +5.0] |

- best가 1위보다 크게 진 곳: VAST −33, POP909 −28, ANLI −28, PhishNChips −26, Home −16, WinoGrande −15, ToolRet −14, BRIGHT −11, CLINC −10.
- best가 이긴 곳: iSarcasm +18, When2Call +13, CRUXEval +10, GPQA +8, ACOS +7, BFCL +3. 37개 중 15승 1무 21패.
- R1 대비 BRIGHT(0.546 vs 0.879)·ToolRet(0.738 vs 0.875)는 크게 떨어졌다.

배포 Crystal(HF `cd943da`, 자체 토크나이저, 같은 엔진 정책)도 같은 1,480요청으로 돌렸다(모두 ok). Crystal 0.629, best 0.622, 차이 −0.7 [−2.7, +1.3]으로 통계적 동률이며 14승 2무 21패. best가 진 곳: Home −15, HellaSwag −13, ANLI −13, VAST·GSM8K·ESCI·ContractNLI −10. 이긴 곳: BPoMP +45, BRIGHT +14, Phish +11, WinoGrande +10, API-Bank +10.

### 11-10 이미지 소표본 비교

공개 비전 11개 벤치마크마다 60행(seed 20261110), 660요청. best/(비전 타워는 원본 그대로, 이미지 학습 없음)와 배포 Crystal은 R1과 같은 비전 정책(VisionEngine, prompt decide, noncausal, 1,638,400px)으로 실행했고 모두 ok. QEV-0.8B(≤1B 이미지 1위)·v2·R1은 저장된 전체 결과에서 같은 요청을 가져왔다. 지표는 최상위 선택 정확도의 11개 평균이다(공식 지수 아님). 코드 `qwen_decision/v3/fullft_sample_vision.py`, 결과 `public-sample/vision-comparison.json`.

| 모델 | 11개 평균 | best와 차이(95%) |
|---|---|---|
| R1 LoRA | **0.673** | best −8.0 [−11.7, −4.2] |
| QEV-0.8B (이미지 1위) | 0.645 | best −5.2 [−8.9, −1.5] |
| Crystal cd943da | 0.639 | best −4.7 [−8.2, −1.4] |
| v2 joint-g7-v2 | 0.635 | best −4.3 [−7.7, −0.6] |
| **best (fullft 81,920)** | 0.592 | — |

- best는 텍스트로만 학습해서 이미지가 약하다. 크게 진 곳은 Mind2Web(best 0.433, Crystal 0.600, QEV 0.833, R1 0.867), CV-Bench(0.600, Crystal 0.767), BLINK, Moderation이다. MMMU-Pro와 R-Bench-M은 best가 가장 높다.
- 이 소표본 정확도로는 R1과 Crystal도 QEV와 비슷하거나 높지만, 공식 비전 지수(보정·가중)에서는 QEV가 앞선다. 이미지 쪽은 best 위에 이미지 LoRA를 얹는 게 다음 순서다.
