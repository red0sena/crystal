# Decision Model 연구 위키

생성 없이 가변 선택지의 점수·확률을 반환하는 모델을 만들기 위한 논문과 설계 기록이다. 우리 Qwen3.5-0.8B 실험과 연결해 읽을 수 있다.

**조사 기준일: 2026-10-13(망각·편향 문헌은 2026-10-17 추가). v3 기록은 2026-10-16부터.** alphaXiv에서 논문을 검색하고 PDF 본문을 확인한 내용을 정리했다. 2026년 연구는 공개 프리프린트 결과로 취급하며 학회 심사·독립 재현을 확인했다고 주장하지 않는다. 자료에 등장하는 정확도·속도는 해당 논문 조건의 결과이며 우리 모델 성능이 아니다.

## v3: Jev Decision Index 도전 (2026-10-16~)

목표는 우리 Qwen3.5-0.8B로 [Jev Decision Index](https://huggingface.co/spaces/multimodalart/jev-decision-index)에서 `vllm-sr/d3-lite`(0.8B, Full 32.33)를 넘는 것이다.

0. **[v3 전체 요약](project/v3-summary.md)**: 방침, 데이터 버전, 학습 실험(g1·g3·g4b·g6), 가중치 섞기, 진단, 현재 최선 후보를 한 페이지에 정리했다. 처음엔 이것부터 읽는다.
1. [목표와 리더보드 규칙](project/jev-index.md): 점수 산식, d3-lite 구조와 약점, 0.8B 경쟁 모델, 제출·오염 규칙.
2. [작업 기록](project/v3-log.md): 시간순 상세 기록이다. 측정값만 적는다. 주요 내용은 다음과 같다.
   - 준비: suite 재구성과 d3-lite 재현, JEVal 대리셋, 오염 필터, 학습 속도 개선.
   - 학습 방식과 g1: 정답 전용 학습(교사 금지, 도구만 예외), 정답 기준선 g1(공개 샘플 33.68 대 d3-lite 39.65).
   - 분석: 원본과 가중치 섞기, d3-lite 가중치 감사, 순서 편향·문항별 망각 측정.
   - 데이터: 내부 dev 누수 수정, 3·4차 데이터(사람 정답, 라이선스 확인), dev 분할 규칙 개정(검정력 기준).
   - 학습: g3(3에폭), g4b(새 데이터 40만, 패딩 없음), fresh-v1 평가셋, g6(104개 출처 70만, JEVal 0.465·공개 40.28).
   - 데이터 감사와 5·6차 데이터(46개 출처 추가, 에이전트 병렬 빌드와 직접 검수).
   - 섞기: g6 0.67 + g3e2 0.33(공개 43.12·fresh 0.520, d3-lite 초과)과 g6 + g4b(JEVal 0.476)가 현재 최선 후보다. 논문 방식(TIES·DARE·LiNeS·Model Stock)은 단순 평균보다 낮았다.
3. [전략](designs/beat-d3-lite.md): 수단의 우선순위와 상태, 지금까지 알게 된 것, 다음 단계.
4. [학습 데이터 출처](designs/phase2-data-sources.md): 108개 출처 194만 문항(v3-gold6이 104개 사용). 정답 출처(사람/프로그램/모델), 학습 셋 사용량, 라이선스, 오염 함정을 정리했다.
5. [참고 문헌과 데이터](resources/references.md): v3에서 참고한 논문과 데이터셋, 확인 수준(카드/초록/검색/인용만)과 쓰인 곳.

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
| 가중치 섞기 | [Model Stock](papers/model-stock.md), [LiNeS](papers/lines.md), [TIES-Merging](papers/ties-merging.md), [DARE](papers/dare.md), [체크포인트 평균·Soup](papers/checkpoint-averaging.md), [In-the-Wild 병합](papers/merging-in-the-wild.md) |
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
| [TIES-Merging](papers/ties-merging.md) | 2023 | 자르기·부호 투표 병합 |
| [DARE](papers/dare.md) | 2024 | 무작위 버리기·재스케일 병합 |
| [체크포인트 평균·Soup](papers/checkpoint-averaging.md) | 2023 | 가중치 평균 |
| [In-the-Wild 병합 비교](papers/merging-in-the-wild.md) | 2026 | LLM 병합 방법 비교 |
| [Dataset Cartography](papers/dataset-cartography.md) | 2020 | 문항 지도 / OOD 일반화 |
| [검정력](papers/power-analysis.md) | 2020 | 평가셋 크기 |

## 구현 자료와 기존 실험

- [d1·d1-omni·Jeff·Decider·d3 구현 참고](resources/implementations.md): 논문과 소스코드 근거를 구분. Decision Index 출전 모델 포함.
- v3 코드: `qwen_decision/v3/`(프롬프트·모델·엔진·학습·평가·오염 필터), 데이터 빌더 `qwen_decision/v3/sources/`.
  - 학습 셋: `mix.py`.
  - 학습: `train.py`(에폭별 저장, 원본 당기기 포함).
  - 평가: `proxy_eval.py`(JEVal), `board_check.py`(공개 샘플), `fresh_eval.py`(fresh-v1), `bias_eval.py`, `forget_report.py`, `format_probe.py`.
  - 섞기·분석: `merge.py`(WiSE·LiNeS·Model Stock·TIES·DARE·가중 평균), `interpolate.py`, `weight_drift.py`, `check_packed.py`(회귀 검사), `bench_train.py`.
  - 실행 스크립트: `runs/v3/queue/`.
- [조사 범위와 인용 규칙](resources/research-notes.md): 완독·재현 여부와 검색 오매칭 기록.
- [v1 실험 보고서](../docs/experiment_report_ko.md), [v2 실험 보고서](../docs/experiment_report_v2_ko.md).
- [재현 안내](../docs/reproduce_ko.md): v1/v2 재현 문서다. 당시 모델·원시 데이터·가상환경은 요청에 따라 삭제됐다. v3 환경(`.venv-di`, `models/`)은 따로 있다.
- [기계가 읽을 수 있는 출처 색인](sources.json).

## 유지 방식

새 연구는 논문 카드 → 구조 지도 → 적용 제안 순서로 반영한다. 실험을 실제 수행하면 날짜·데이터 split·설정·결과 파일을 붙이고 미검증 표기를 갱신한다. 기존 실험 보고서는 역사적 기록으로 보존한다.

