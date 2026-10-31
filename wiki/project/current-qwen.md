# 현재 Qwen 모델과 새 설계의 출발점

[위키 홈](../README.md) · 근거: [v1 보고서](../../docs/experiment_report_ko.md), [v2 보고서](../../docs/experiment_report_v2_ko.md), [scorer v2](../../qwen_decision/core/scorer_v2.py)

## 실제 구현

입력 state·question·options를 하나의 prompt로 만든다. 선택지 내용은 prompt 안에 들어가고 출력에는 단일 token 코드가 대응한다. Thinking을 끄고 Backbone을 처리한 뒤 마지막 실제 입력 hidden state와 LM Head의 해당 코드 행만 곱한다. 후보 logits에 softmax를 적용한다. generate()와 설명 token 생성은 없다.

Full vocabulary의 나머지 행은 저장되어 있다. LM Head와 입력 embedding이 tied이므로 출력 행을 줄여도 입력 embedding 전체를 제거하는 것은 아니다. 이미 scorer가 후보 행만 계산하므로 작은 고정 head가 자동으로 큰 속도·메모리 이득을 준다고 가정하지 않는다.

v2는 2–64 후보, choice·noul·score, 질문 유형·후보 수 기반 temperature calibration을 지원한다. 공통 state prefix를 여러 **질문**에 재사용하고 질문 suffix별 Attention KV·DeltaNet recurrent·conv 상태를 독립적으로 복사한다. 이 기능은 선택지 내용을 직접 점수화하는 새 head와 다르다.

## 실제 학습

- v1: 63,054 train / 14,078 dev / 30,801 test. 언어 Backbone과 tied embedding/LM Head full fine-tuning, vision 동결. 선택 후보 logits에 CE를 사용했다. 독립 두 seed 모델의 가중치 50:50 평균을 dev에서 선택했다.
- v2: 108,971 train / 20,996 dev / 53,212 test. 후보 수·유형과 데이터 다양성을 확장했다. CE와 score soft targets를 사용하며 모델 선택과 temperature 적합을 dev에서 수행했다. 선택된 checkpoint와 post-hoc 평균 모델을 구분해야 한다.
- 선택지 내용·코드 매핑을 섞고 타깃도 remap하여 코드·위치 편향을 완화하려 했다. 데이터 증강은 정확한 구조적 순열 대칭성 보장이 아니다.

## 해석할 때 주의할 기록

MMLU-Pro official test를 자체 분할하여 학습에 일부 사용했으므로 전체 official test를 독립적인 미학습 시험으로 부르면 안 된다. v1의 일부 많은 후보 평가에는 응답 코드 coverage 문제가 있었다. v2 None-of-the-above 증강에는 candidate count shortcut 이슈가 기록되어 있다. 다음 실험의 데이터·평가에서 수정해야 한다. 상세 수치와 한계는 기존 보고서를 우선한다.

## 현재 파일 상태

v1/v2 시절의 모델·원시 데이터·체크포인트·가상환경은 사용자 요청에 따라 삭제됐고 코드·작은 설정/결과 기록·재현 문서는 보존됐다([재현 안내](../../docs/reproduce_ko.md)). **v3(2026-10-16~)는 새로 만든 환경이다.** 10-31 현재 로컬에는 가상환경 `.venv`, 모델 `models/qwen3.5-0.8b`, 데이터 `data/mix/`(v3-gold2·3·6)·`data/longctx/`·`data/jeval/`·`data/fresh/`, 학습 결과 `runs/`가 있다. 처음에 쓰던 `.venv-di`, `models/d3-lite`, `data/phase2/`는 지금 없다. g1~g7 학습의 best·final checkpoint와 일부 섞기 모델은 Hugging Face `Kyunghwan-kim/kkh-decision-model`에 있다. 10-19 이후 실험 run 폴더와 로컬에서 다시 만든 g6/g3 조합 가중치는 이 기계에만 있다([v3 요약](v3-summary.md)).

## v3 (2026-10-16~)

[Jev Decision Index 목표](jev-index.md)를 위해 v3를 새로 만들었다(`qwen_decision/v3/`). v2와 다른 점은 이렇다.
- 프롬프트와 코드가 d3 계약을 따른다. 코드는 255개다.
- full-attention 6층을 양방향으로 바꿀 수 있다(`noncausal`).
- ~~학습 target이 교사 분포에 대한 soft CE다.~~ **2026-10-17부터 정답 전용이다.** 교사 모델의 출력(증류, argmax)은 쓰지 않고, 표시된 코드들에 대한 CE를 정답(`train.gold_targets`)으로만 계산한다. 도구 영역만 생성 모델이 만든 정답을 허용한다.
- 키트 Engine으로 공개 suite와 JEVal 대리셋을 평가한다.

현재 학습 경로(`qwen_decision/v3/train.py`, 2026-10-17):
- 언어 백본과 tied 임베딩/LM head를 full fine-tune하고 비전은 동결한다. BF16 계산, FP32 master 가중치·모멘트(`MasterAdamW`).
- 패딩 없이 이어 붙인 배치(varlen attention, FLA scan·conv), 원소별 블록 `torch.compile`, 큰 microbatch만 체크포인팅.
- 증강: 선택지 내용 섞기, 코드 배정 섞기(약 60%는 A, B, C 순서), 17~255지선다 무작위 패딩(`--pad-probability`, 기본 0.35. g4b는 0), noul을 2지선다로 표시. 선택지끼리 위치로 참조하는 문항은 순서를 고정한다.
- 선택: 출처별 정답 dev 정확도 평균으로 best를 고른다. `--epochs`이면 에폭마다 저장하고, `--anchor`이면 원본 가중치 쪽으로 당긴다(분리형 L2-SP).
- 학습 셋은 `qwen_decision/v3/mix.py`가 만든다(원본 항목·문서·근사 중복 단위 dev 분할, 누수 시 빌드 실패).

**원래 Qwen에서 출발하는 비교(2026-10-25~)의 학습 경로**는 `train.py` 위에 얹은 실험 모듈이다.
- full FT 비교: `from_base_experiment.py`(구조), `training_method_experiment.py`(causal, noncausal, DeltaNet 얼리기 등).
- LoRA: `lora_experiment.py`(attention), `lora_placement_experiment.py`(위치). 지금까지 최선은 양방향 + 전체 LoRA r16(alpha 32, dropout 0.05, LR 2e-4, 1,024 update × 16행, cosine)이다([LoRA 위치 비교](lora-placement.md)).
- loss: `loss_experiment.py`. CE를 이긴 loss는 없었다([Loss 비교](loss-screen.md)).
- 학습 정답은 gold 라벨만 쓰고, 학습 데이터는 토큰을 자르지 않는다.

결과와 진행 상황은 [작업 기록](v3-log.md)에 있다. 같은 날 루트의 스크립트는 `qwen_decision/` 패키지(core, prepare, train, evaluate, audit, analysis, sources)로 옮겨졌다. 위 링크의 `scorer_v2.py`는 `qwen_decision/core/scorer_v2.py`에 있다.

## 아직 구현하지 않은 것

가변 후보 content encoder, 위치 없는 Set Attention을 주 경로로 쓰는 구조, 새로운 reasoning distillation은 문헌 기반 제안으로 남아 있다([설계안](../designs/qwen-next.md)). 가까운 것들은 g4b 위에서 작게 시험했고 개선이 없었다: Pointer head([결과](pointer-pilot.md)), 잔차 Set Transformer([구조 검색](architecture-search.md)), 대체 readout·MLP([결과](readout-mlp.md)), 후보별 판정 Query([결과](decision-native-next.md)). v3에서 구현한 구조 변경 전체는 [위키 홈](../README.md)의 6~27번에 있다.

