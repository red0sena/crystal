# Decision Model 연구 위키

생성 없이 가변 선택지의 점수·확률을 반환하는 모델을 만들기 위한 논문과 설계 기록이다. 우리 Qwen3.5-0.8B 실험과 연결해 읽을 수 있다.

**조사 기준일: 2026-10-13(망각·편향 문헌은 2026-10-17 추가). v3 기록은 2026-10-16부터.** alphaXiv에서 논문을 검색하고 PDF 본문을 확인한 내용을 정리했다. 2026년 연구는 공개 프리프린트 결과로 취급하며 학회 심사·독립 재현을 확인했다고 주장하지 않는다. 자료에 등장하는 정확도·속도는 해당 논문 조건의 결과이며 우리 모델 성능이 아니다.

## v3: Jev Decision Index 도전 (2026-10-16~)

목표는 우리 Qwen3.5-0.8B로 [Jev Decision Index](https://huggingface.co/spaces/multimodalart/jev-decision-index)에서 `vllm-sr/d3-lite`(0.8B, Full 32.33)를 넘는 것이다.

0. **[쉬운 정리: 지금까지와 앞으로](project/v3-easy-guide.md)** (2026-10-25): 한 일, 안 된 이유, 앞으로 할 수 있는 것(추천 순서 포함)을 쉬운 말로 정리했다. 처음엔 이것부터 읽는다.
   - [v3 전체 요약](project/v3-summary.md): 방침, 데이터 버전, 학습 실험(g1·g3·g4b), 가중치 섞기, 진단, 현재 최선 후보를 수치와 함께 정리한 상세판.
1. [목표와 리더보드 규칙](project/jev-index.md): 점수 산식, d3-lite 구조와 약점, 0.8B 경쟁 모델, 제출·오염 규칙.
2. [작업 기록](project/v3-log.md): 시간순 상세 기록이다. 측정값만 적는다. 주요 내용은 다음과 같다.
   - 준비: suite 재구성과 d3-lite 재현, JEVal 대리셋, 오염 필터, 학습 속도 개선.
   - 학습 방식과 g1: 정답 전용 학습(교사 금지, 도구만 예외), 정답 기준선 g1(공개 샘플 33.68 대 d3-lite 39.65).
   - 분석: 원본과 가중치 섞기, d3-lite 가중치 감사, 순서 편향·문항별 망각 측정.
   - 데이터: 내부 dev 누수 수정, 3·4차 데이터(사람 정답, 라이선스 확인), dev 분할 규칙 개정(검정력 기준).
   - 학습: g3(3에폭), g4b(새 데이터 40만, 패딩 없음), fresh-v1 평가셋.
   - 섞기: g4b/g3 이력에 이어 완료된 g6/g3 67:33 조합을 로컬 재구성·평가했다(아래 11번).
3. [전략](designs/beat-d3-lite.md): 수단의 우선순위와 상태, 지금까지 알게 된 것, 다음 단계.
4. [학습 데이터 출처](designs/phase2-data-sources.md): 108개 출처 194만 문항(v3-gold6이 104개 사용). 정답 출처(사람/프로그램/모델), 학습 셋 사용량, 라이선스, 오염 함정을 정리했다.
5. [참고 문헌과 데이터](resources/references.md): v3에서 참고한 논문과 데이터셋, 확인 수준(카드/초록/검색/인용만)과 쓰인 곳.
6. [Pointer head 소규모 비교](project/pointer-pilot.md): v3 backbone을 유지하는 코드 head/Pointer 학습·평가 실험과 실행 방법.
7. [논문 기반 구조 검색](project/architecture-search.md): 99개 설정·seed 조합, 별도 confirmation·JEVal 검증. 이번 범위에서 개선은 확인되지 않았다.
8. [다른 방법 조사와 입력 실험](project/other-methods.md): 추가 14편 검토. 입력 반복·정의·prefix·prior 보정 실험과 반사실 쌍 학습·R-Drop·반복 깊이의 적용 조건.
9. [LM Head 대체 readout·MLP](project/readout-mlp.md): 6종·24개 설정/seed, 마지막 두 층 추가 학습, 별도 2,027문항 확인. 이번 조건에서는 개선 없음.
10. [원본 보존 학습·추론 조합](project/robust-combinations.md): LoRA·순열 일관성 4개 run, 가중치·cyclic 평균, 후보 재비교, gold memory. 별도 2,998문항에서 이번 소규모 추가 방법의 개선은 확인되지 않았다.
11. **[g6/g3 조합 모델 재구성·새 문항 확인](project/g6-soup-replication.md)**: g6 67% + g3 33%를 직접 만들었다. 기존 fresh 재측정에 이어, 학습·이전 평가 지문을 제외한 새 1,847문항·10출처에서도 **g4b 65.39% → 71.97%(+6.58%p)**였다. 추가 추론 없는 단일 checkpoint, 기존 엔진 1,335문항 재로딩 불일치 0개.
12. [Transformer 내부 블록 반복](project/recurrent-depth.md): 24→28블록 실행, 가중치 공유·게이트, 두 구조 모두 전체 언어 백본을 512 update씩 학습했다. 새 8,823문항에서 같은 추가 학습의 기본 구조 75.20%, 반복 구조 75.29%. 차이 +0.10%p의 95% 구간이 0을 포함했고, 추론은 15.27% 느려 개선으로 채택하지 않았다.
13. [후보별 판정 Query와 마지막 층 연산 재배치](project/decision-native-next.md): 공통 23층 뒤 마지막 층의 Q·출력·FFN을 후보 Query에만 적용. 실제 512스텝 통제 학습 후 새 3,394문항에서 기본 70.70%, 후보 70.75%, 차이 +0.046%p·95% 구간 [−0.328,+0.412]%p로 정확도 개선은 확인하지 못했다. 저장/복원·기존 엔진 연결 포함.
14. **폐기한 후보 —** [중간 블록 delta 재사용 실험](project/v3-log.md): native24층 안에 변화량 depth routing을 추가해 실제512스텝 학습. 새2,220문항에서65.70%→65.85%, 차이+0.154%p·95% 구간[−0.282,+0.576]%p로 개선은 확인하지 못했다. 실제 엔진 호출 약5.4% 증가. 사용자 요청으로 채택 후보에서 제외하고 증거는 보존했다. [수치·모델/엔진 검증](project/delta-memory-results.json).
15. [중간 층의 가설 작업공간과 판정 출구 변경](project/hypothesis-workspace.md): 앞16층 뒤 선택지별 상태를 추가해 남은8층에서 원문·가설을 공동 갱신했다. 첫 버전512스텝은 기본75.7419%, 새75.7470%(+0.0050%p). 가설에서만 판정하는 버전은 같은256스텝에서73.1286% 대 기본73.9417%(−0.8131%p)로 사전 규칙에 따라 중단했다. 두 버전 모두 채택하지 않았고 새 확인셋을 열지 않았다. 관련63개 테스트·전체선택4,147문항 저장 복원을 검증했다. [TRM·T²MLR·RiM·LLoCoT·BDH-CQ 등 8편 대조](papers/latent-workspaces.md).
16. [상위8층을 BDH 메모리로 교체·고유 데이터4배 비교](project/bdh-memory.md): 앞16층 뒤 연상 메모리·coactivation을3회 갱신하는 코어로 뒤8층을 대체했다. 세 군 모두1,024스텝 실제 학습. 기존 선택셋4,147문항에서 같은16,384행을 학습한 원본75.5504%, BDH74.9540%(−0.5964%p)로 정확도 개선은 없었다. BDH에서4,096행4회→16,384행1회는73.2656%→74.9540%(+1.6884%p)여서 이 조건의 데이터 다양성 효과는 보였다. 새로운 확인셋은 열지 않았다. 공개 BDH의 별도 Qwen retrofit이며 비공개 BDH-CQ 재현이 아니다.
17. [학습 예산 확대](project/long-training.md): 원본·BDH·내부 블록 반복·가설 출구4종을 같은65,536행과4,096스텝으로 다시 비교한다. 이전 BDH의4배, 다른 후보의8~16배 update다. 모든 입력 검사·관련69개 테스트·네 구조의 실제 FP32 optimizer 저장/재개 검증을 통과하고 설정을 고정해 실행했다. 짧은 학습 실패를 구조 자체의 한계로 단정하지 않는다. **완료(2026-10-20):** 원본 79.14%, BDH 78.16%, 블록 반복 79.05%, 가설 출구 78.45%로 원본을 넘은 구조가 없다.
18. Gated DeltaNet 역방향 scan([수치](project/reverse-scan-results.json)): 4,096 update에서 −0.51%p(95% [−1.34, +0.19]). 개선 없음.
19. [MNTP 전환 학습](project/mntp-conversion.md): 2026-10-21 사용자 요청으로 중단, 비교 결과 없음.
20. [DUMA 공동 매칭·Theseus 점진 교체](project/dual-theseus.md): 세 변형 모두 원본 79.14% 미만.
21. 증거 쌍 손실([수치](project/evidence-pairs-results.json)): CONDAQA·VitaminC 짝 손실, 모든 효과 구간이 0 포함.
22. 단일 forward 의미 매칭([수치](project/single-forward-semantic-results.json)): semantic−native +0.05%p, 3 seed 계층 구간 0 포함.
23. [Talking-Heads](project/talking-heads.md): 마지막 두 full-attention 층 head 혼합, 3 seed에서 차이 없음.
24. [학습 레시피: EMA·구간 평균·학습률](project/recipe-ema-lr.md): EMA −0.54%p, 구간 평균 −0.57%p, LR 1e-5 −3.76%p. 개선 없음.
25. **[장문 테스트셋과 질문 우선 배치(echo)](project/long-context-echo.md)**: 공개 장문 5종을 평가 전용으로 만들고 25% 장문 풀로 native/echo를 3 seed 비교했다. 장문 점수 +0.61%p, seed×문항 95% [−0.84, +2.07]로 구조 효과 입증 실패. 찾기는 이미 천장이고 실패는 추론·다예시 분류에 있으며, seed 잡음이 효과보다 크다는 원인 분석을 함께 기록했다.
26. [선택지 순서 측정](project/order-probe.md): 학습 없이 4순서 평균. selection +0.68%p(답 바뀜 10%), JEVal 장문 −1.25%p(22%), BABILong 0(19%). 선택지 격리가 줄 몫은 짧은 문항 +0.7%p 이하로 추정.
27. **[원래 Qwen에서 출발하는 구조 비교](project/from-base-structures.md)** (완료 10-26): 원본(causal)·양방향·결정 지점 반복 읽기(hop)를 학습 전 가중치에서 같은 104개 출처 65,536행으로 각각 full fine-tuning해 3 seed로 비교했다. 사전 등록한 여섯 검정을 모두 통과하지 못했다. 양방향−원본은 selection +0.51, 장문 −0.21이다. hop의 두 번째 읽기는 첫 읽기와 98~99% 같은 답을 골라 다시 읽기가 작동하지 않았다. hop−원본 selection은 세 seed 모두 양수(+1.19)였지만 구간 하한이 −0.18이었다.
28. **[우리 평가가 맞나: 구조 변경 논문들의 평가 방식과 비교](project/evaluation-validity.md)** (10-26): Gated Attention·Ettin·LLM2Vec·Pause tokens·Retrofitted recurrence·Coconut·Candidate-independent attention·LoLCATs와 Signal and Noise·DataDecide를 비교했다. 우리 평가는 엄격하지만 둔감했다. seed 3개와 정답률로는 짧은 문항 약 2.5점, 장문 약 7점 미만의 효과를 잡지 못하는데, 문헌상 구조 효과는 0.5~3점이다. 같은 9개 모델을 정답 NLL로 사후 재분석하면 hop−원본 짧은 문항과 양방향−원본 장문이 기준을 넘는다. hop의 이득은 다시 읽기가 아니라 학습 방식에서 나왔다.
29. **[학습 방식 거르기](project/training-methods.md)** (완료 10-28: DeltaNet 얼리기만 통과, 6.55만 행 확인 대기): 원래 Qwen에서 출발해 다섯 레시피(causal, noncausal, DeltaNet 얼리기, CE+Brier, hop식 두 번 읽기 학습)를 fit당 16,384행 × seed 5로 비교한다. 주 지표는 정답 NLL이다. 근거 논문은 [판정 모델 학습 레시피 묶음](papers/decision-training-recipes.md)에 정리했다.

## 처음 읽을 문서

1. [현재 Qwen의 구조와 학습](project/current-qwen.md): 실제 구현한 것과 아직 없는 것을 구분.
2. [구조 변경 지도](concepts/architecture-map.md): LM Head 유지, 후보 내용 점수화, 집합 비교, Backbone 변경 비교.
3. [선택지 사이 Attention 차단](concepts/attention-isolation.md): 마스크, 위치 초기화, hybrid cache 분리.
4. [Reasoning·증류·확률 보정](concepts/reasoning-and-calibration.md): 서로 다른 목표와 손실.
5. [다음 Qwen 설계안](designs/qwen-next.md)과 [평가 계획](designs/evaluation.md): 문헌 기반 제안. 평가는 v3에서 일부 구현됐다.

## 목적별 읽기

| 목적 | 먼저 읽을 논문 |
|---|---|
| 후보 내용에서 직접 점수 얻기 | [GLiClass](papers/gliclass.md), [Soft Contextualized Encoder](papers/soft-contextualized-encoder.md) |
| 코드·표시 순서 편향 줄이기 | [Candidate-independent Attention](papers/candidate-independent-attention.md), [객관식 편향](papers/multiple-choice-bias.md) |
| 후보끼리 비교하면서 순서 대칭성 유지 | [Set Transformer](papers/set-transformer.md), [Soft Contextualized Encoder](papers/soft-contextualized-encoder.md) |
| 긴 입력 처리와 후보 캐시 | [Poly-encoders](papers/poly-encoders.md), [ColBERT](papers/colbert.md), [MetaEncoder](papers/metaencoder.md) |
| 추론 시 설명 없이 reasoning 학습 효과 활용 | [Dual-Head](papers/dual-head-reasoning-distillation.md), [InnerJev](papers/innerjev.md) |
| 마지막 토큰 외 표현 활용 | [Token·Layer readout](papers/token-layer-readout.md), [GLiClass](papers/gliclass.md) |
| 기존 구조를 유지하는 대조군 | [LLM-as-Jev](papers/llm-as-jev.md), [InnerJev](papers/innerjev.md) |
| 미세조정 망각 줄이기 | [WiSE-FT](papers/wise-ft.md), [L2-SP](papers/l2-sp.md), [LoRA Learns Less](papers/lora-learns-less.md), [LP-FT](papers/lp-ft.md) |
| 가중치 섞기 | [Model Stock](papers/model-stock.md), [LiNeS](papers/lines.md), [체크포인트 평균·Soup](papers/checkpoint-averaging.md), [In-the-Wild 병합](papers/merging-in-the-wild.md) |
| 객관식 편향 측정 | [One Bias Score, Two Factors](papers/option-bias-two-factors.md), [객관식 편향](papers/multiple-choice-bias.md) |
| 난이도·품질 등급 및 확률 | [Ordinal scale bias](papers/ordinal-scale-bias.md), [Temperature scaling](papers/temperature-scaling.md) |
| Encoder 전환과 주변 연구 | [LLM2Vec](papers/llm2vec.md), [dQwen3.5](papers/dqwen35.md) |
| 통합 분류 데이터·점수 손실 | [ConEntail](papers/conentail.md), [RankT5](papers/rankt5.md) |

## 전체 논문 색인

| 논문 | 연도 | 핵심 주제 |
|---|---|---|
| [Candidate-Independent Block-Causal Attention](papers/candidate-independent-attention.md) | 2026 | Attention 구조 / 순서 편향 |
| [Dual-Head Reasoning Distillation](papers/dual-head-reasoning-distillation.md) | 2025 | 학습 때만 reasoning / 분류층 |
| [Poly-encoders](papers/poly-encoders.md) | 2020 | 공통 문맥 / 후보 비교 |
| [Set Transformer](papers/set-transformer.md) | 2019 | 선택지 집합 / 대칭성 |
| [LLM2Vec](papers/llm2vec.md) | 2024 | Decoder를 표현 모델로 전환 |
| [LLM-as-Jev](papers/llm-as-jev.md) | 2026 | 기존 LM Head / 후보 문자열 점수 |
| [GLiClass](papers/gliclass.md) | 2025 | 가변 라벨 내용 / 공동 인코딩 |
| [Soft Contextualized Encoder](papers/soft-contextualized-encoder.md) | 2026 | 새 판단 아키텍처 / 집합 비교 |
| [MetaEncoder](papers/metaencoder.md) | 2026 | 후보 캐시 / 대조 학습 |
| [InnerJev / JEVal](papers/innerjev.md) | 2026 | reasoning 분포 증류 / 기존 LM Head |
| [Token·Layer 선택 readout](papers/token-layer-readout.md) | 2026 | hidden state 집계 / 작은 분류층 |
| [ColBERT](papers/colbert.md) | 2020 | 토큰 단위 late interaction |
| [RankT5](papers/rankt5.md) | 2022 | 스칼라 점수 / ranking loss |
| [Ordinal scale-utilization bias](papers/ordinal-scale-bias.md) | 2026 | 등급 편향 / 진단 |
| [ConEntail](papers/conentail.md) | 2023 | 통합 분류 데이터 / 대조 학습 |
| [dQwen3.5](papers/dqwen35.md) | 2026 | hybrid 정보 흐름 / 주변 연구 |
| [객관식 코드·위치 편향](papers/multiple-choice-bias.md) | 2024 | 코드 prior / 객관식 안정성 |
| [Temperature scaling](papers/temperature-scaling.md) | 2017 | 확률 보정 |
| [WiSE-FT](papers/wise-ft.md) | 2022 | 원본·미세조정 가중치 보간 |
| [L2-SP](papers/l2-sp.md) | 2018 | 원본 가중치 쪽 L2 제약 |
| [LoRA Learns Less and Forgets Less](papers/lora-learns-less.md) | 2024 | 미세조정 방식과 망각 |
| [LP-FT](papers/lp-ft.md) | 2022 | 출력층 먼저 학습 / 특징 왜곡 |
| [One Bias Score, Two Factors](papers/option-bias-two-factors.md) | 2026 | 객관식 편향의 두 요인 |
| [Model Stock](papers/model-stock.md) | 2024 | 각도 기반 병합 비율 |
| [LiNeS](papers/lines.md) | 2025 | 층별 스케일링 병합 |
| [체크포인트 평균·Soup](papers/checkpoint-averaging.md) | 2023 | 가중치 평균 |
| [In-the-Wild 병합 비교](papers/merging-in-the-wild.md) | 2026 | LLM 병합 방법 비교 |
| [Dataset Cartography](papers/dataset-cartography.md) | 2020 | 문항 지도 / OOD 일반화 |
| [검정력](papers/power-analysis.md) | 2020 | 평가셋 크기 |
| [TRM·T²MLR·RiM·LLoCoT 등 잠재 작업공간](papers/latent-workspaces.md) | 2025–2026 | 반복·가설 상태·학습 조건 대조 |
| [BDH·BDH-CQ, LoLCATs·Mamba in the Llama](papers/latent-workspaces.md) | 2024–2026 | 연상 메모리·공개 구현 한계·구조 전환 정렬 |
| [판정 모델 학습 레시피 묶음: LoRA placement·NDM·DKnown-Jev·Jeff·LLM-as-Jev·Signal and Noise](papers/decision-training-recipes.md) | 2025–2026 | Qwen3.5 DeltaNet 학습 위치 / Brier 손실 / 학습·평가 규모 / 연속 지표 |

## 구현 자료와 기존 실험

- [d1·d1-omni·Jeff·Decider·d3 구현 참고](resources/implementations.md): 논문과 소스코드 근거를 구분. Decision Index 출전 모델 포함.
- v3 코드: `qwen_decision/v3/`(프롬프트·모델·엔진·학습·평가·오염 필터), 데이터 빌더 `qwen_decision/v3/sources/`.
  - 학습 셋: `mix.py`.
  - 학습: `train.py`(에폭별 저장, 원본 당기기 포함).
  - 평가: `proxy_eval.py`(JEVal), `board_check.py`(공개 샘플), `fresh_eval.py`(fresh-v1), `bias_eval.py`, `forget_report.py`, `format_probe.py`.
  - 섞기·분석: `merge.py`(WiSE·LiNeS·Model Stock·평균), `interpolate.py`, `weight_drift.py`, `check_packed.py`(회귀 검사), `bench_train.py`.
  - 실행 스크립트: `runs/v3/queue/`.
- [조사 범위와 인용 규칙](resources/research-notes.md): 완독·재현 여부와 검색 오매칭 기록.
- [v1 실험 보고서](../docs/experiment_report_ko.md), [v2 실험 보고서](../docs/experiment_report_v2_ko.md).
- [재현 안내](../docs/reproduce_ko.md): v1/v2 재현 문서다. 당시 모델·원시 데이터·가상환경은 요청에 따라 삭제됐다. v3 환경(`.venv-di`, `models/`)은 따로 있다.
- [기계가 읽을 수 있는 출처 색인](sources.json).

## 유지 방식

새 연구는 논문 카드 → 구조 지도 → 적용 제안 순서로 반영한다. 실험을 실제 수행하면 날짜·데이터 split·설정·결과 파일을 붙이고 미검증 표기를 갱신한다. 기존 실험 보고서는 역사적 기록으로 보존한다.
