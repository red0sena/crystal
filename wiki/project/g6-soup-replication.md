# g6·g3 가중치 조합의 로컬 재구성과 평가

> **10-31 추가:** 이 조합은 10-19에 공개 샘플 43.12(d3-lite 39.65), JEVal dev 0.470(0.482), fresh 0.520(0.509)으로도 채점돼 있었다. 그 뒤 g7과 섞은 모델이 세 평가 모두 이 조합보다 높았다([g7·섞기 결과](hub-g7-merges.md)).

[위키 홈](../README.md) · [함께 진행한 다른 조합 실험](robust-combinations.md)

**현재 비교 기준인 g4b보다 높은 점수를 내는 단일 모델을 만들었다.** 완료된 g6 가중치 67%와
g3의 2에폭 가중치 33%를 FP32로 평균해 BF16 checkpoint로 저장했다. 새로운 구조의 효과가 아니라,
서로 다른 데이터·학습 이력을 결합한 결과다.

**추가 확인:** 비율을 고른 뒤 남겨둔 새 문항 **1,847개·10출처**에서도 g4b **65.39% → 71.97%**,
**+6.58%p**를 확인했다(95% 구간 **[+4.61,+8.58]%p**). 아래 fresh-v1 재측정과 별도다.

## 비율을 고른 뒤의 새 문항 확인

기존 fresh-v1과 같은 원천 과제군의 공식 validation/test에서 미사용 문항을 골랐다.
가중치 비율·checkpoint는 먼저 고정했으며 새 점수로 바꾸지 않았다.

| 고정 모델 | 새 1,847문항의 출처 평균 정확도 |
|---|---:|
| g4b best | 65.39% |
| g6 67% + g3 33% | **71.97%** |

차이 **+6.58%p**, 문서 bootstrap 5,000회 95% 구간 **[+4.61,+8.58]%p**,
새로 맞힌 문항 **182개**, 새로 틀린 문항 **57개**였다.

- 기존 fresh-v1 및 **로컬 JEVal 파일 전체**와 ID·정규화 지문이 겹치는 그룹을 제외했다.
- 모델의 실제 학습 파일 hash를 manifest와 대조하고 g3 **199,981행**, g4b **399,978행**,
  g6 **699,960행**, 합계 **1,299,919행 전체**의 ID·정규화 지문을 검사했다. 각 gold dev도 제외했다.
- 소문자화·구두점/공백 정규화·JSON canonicalization을 사용했다. 의미가 비슷한 모든 문장을 탐지한 것은 아니다.
- 미사용 그룹이 20개 이상 남은 출처를 전부 포함하고, 출처당 hash 순서로 최대 200그룹을 골랐다.
  같은 지문에서는 한 문항만 썼다. 모델 예측이나 정답 여부로 문항을 선택하지 않았다.
- epistemic reasoning, TimeDial, COPA, ETHICS, PIQA, QASC, QuaRTz, SWAG, WiC는 각각 200개,
  presuppositions as NLI는 남은 47개였다. GoEmotions는 학습 출처라 포함하지 않았다.
- 새 label·rationale를 생성하지 않았다. 준비한 평가 파일과 두 모델의 hash를 추론 전에 고정했다.

이는 **같은 10개 과제군의 새 문항**에서의 확인이다. 모든 새로운 과제에서 더 좋다는 결론은 아니다.
이 문항들도 결과를 본 뒤에는 관찰한 평가 자료다. 다음 recipe 선택에 재사용하지 않는다.
[새 확인 문항의 수치·계획·전체 학습 감사·데이터 hash](g6-soup-holdback-results.json)에 보존했다.

```bash
# 기록된 공개 원문 snapshot 복원 (각 repo revision 고정)
.venv/bin/python -m qwen_decision.v3.download_fresh_sources
# private data repo의 gold2/3/6 학습 파일과 각 dev도 원래 hash로 복원한 뒤:
.venv/bin/python -m qwen_decision.v3.fresh_holdback prepare --output runs/robust/fresh-v2-reproduction
.venv/bin/python -m qwen_decision.v3.fresh_holdback confirm --output runs/robust/fresh-v2-reproduction
```

데이터 준비에는 pandas가 필요하다(실행 환경 3.0.6). 기존 출력 폴더는 덮어쓰지 않는다.
같은 확인 문항을 다시 실행한 것은 재현이며, 또 다른 새 독립 확인이라고 부르지 않는다.

## 같은 평가에서 직접 측정한 결과

fresh-v1에서 학습 출처인 GoEmotions를 제외한 **15출처, 1,235문항**을 똑같이 입력했다.
아래는 출처별 정확도의 평균이다. 최대 957토큰이며 문항을 자르지 않았다.

| 모델 | 출처 평균 정확도 |
|---|---:|
| 비교 기준 g4b best | 66.24% |
| g6 단독 | 66.73% |
| g3 2에폭 단독 | 63.51% |
| **g6 67% + g3 33%** | **70.36%** |

g4b 대비 **+4.11%p**, 새로 맞힌 문항 117개, 새로 틀린 문항 61개였다.
출처 내 문서 단위 paired bootstrap 5,000회의 95% 구간은 **[+2.01, +6.18]%p**다.
원래 비교를 측정한 뒤 g6·g3 단독도 설명용 대조군으로 추가했다.
세 비교에 Bonferroni를 적용한 98.3333% 구간은 다음과 같다.

| 조합 모델의 비교 상대 | 정확도 차이 | 98.3333% 구간 |
|---|---:|---:|
| g4b | +4.11%p | [+1.56, +6.74]%p |
| g6 | +3.63%p | [+1.73, +5.65]%p |
| g3 | +6.85%p | [+4.09, +9.66]%p |

TimeDial +24.0%p, analogical similarity +9.38%p 등이 올랐고 SWAG는 −4.0%p였다.
좋아진 과제만 골라 전체 지표를 다시 계산하지 않았다. GoEmotions를 포함한 전체
16출처·1,335문항에서는 g4b **66.54%**, 조합 **70.46%**였다.

## 발견 경로와 한계

별도 adapter 실험 중 Hub에 **2026-10-19 08:19:27 UTC**에 올라온 완료된 g6와 병합 평가를 발견했다.
그 안의 고정된 67:33 recipe를 다시 만들고 직접 평가했다. **g6 전체 학습은 이 채팅에서 수행하지 않았다.**
이 채팅에서는 checkpoint 복원, recipe 재구성, 동일 입력 비교, 구성요소 대조와 기존 엔진 검증을 수행했다.

이 recipe는 upstream에서 이미 fresh-v1로 평가됐다. **위 fresh-v1 결과는 이미 관찰된 평가셋의 재측정**이며,
새 독립 시험에서 처음 발견한 성공이라고 부르지 않는다. 위 구간은 고정 모델과 문항에서의 차이다.
벤치마크를 통한 recipe 선택의 영향이나 모든 새 데이터의 일반화까지 보장하지 않는다.

upstream의 16출처 정확도는 70.18%, 이번 재구성·평가에서는 70.46%였다.
완전히 같은 수치의 재현이라고 부르지 않는다. GPU·소프트웨어·배치 차이 중 원인은 분리하지 않았다.
이번 비교는 같은 로컬 평가 경로에서 두 모델을 직접 실행한 값이다.

## 고정한 checkpoint와 사용

- 모델 repo `Kyunghwan-kim/kkh-decision-model` revision `a0b44b4d531145ecb30136cea886381205132d93`.
- 원래 Qwen revision `2fc06364715b967f1860aea9cf38778875588b17`.
- g6 best는 최종·best 모두 step **21,874**로 upstream last에 대응한다.
- g3 best는 step **12,500 = 6,250 update/epoch × 2**로 epoch-2에 대응한다.
- fresh 데이터 revision `c15b054aafa3013cb421597031aee87ee150ad4b`.
- 재구성 모델 SHA256 `c90f56aea77f0dfa5fc4c50b7fe08cbe1ca5a168eefa8648ee97b5d15b18ac2c`.

기존 `merge.py`로 `theta = base + .67*(g6-base) + .33*(g3-base)`를 실행했다.
사용 가능한 별도 checkpoint는 **`runs/robust/rebuilt-g6w67-g3e2/`**에 있다.
원본 g4b 파일과 serving 기본값은 바꾸지 않았다. 재구성 명령은 다음과 같다.

```bash
OMP_NUM_THREADS=4 .venv/bin/python -m qwen_decision.v3.merge \
  --base models/qwen3.5-0.8b \
  --tuned runs/robust/hf-models-a0b44b4/runs/v3/g6-gold6-pad0/best \
          runs/robust/hf-models-a0b44b4/runs/v3/g3-gold2-ep3-lr3e-6-a0/best \
  --weights .67 .33 --alpha 1 --out runs/robust/rebuilt-g6w67-g3e2
.venv/bin/python -m qwen_decision.v3.reference_comparison
```

출력이 이미 있으면 별도 디렉터리에서 재실행한다. 추가 label·rationale나 미세조정은 사용하지 않았다.

## 기존 추론 API 검증

기존 `qwen_decision.v3.engine.Scorer`로 전체 **1,335문항**을 다시 실행했다.
저장 예측과 **불일치 0개**, 최대 확률 차이 **1.53e-7**이었다.
텍스트 backbone hook은 예상 배치 334개에서 정확히 **334회** 호출됐다.
`orders=1`이며 추가 모델·투표·후보 재평가 없이 원래 구조의 한 번 forward를 쓴다.

```bash
PYTHONPATH=external/decision-index .venv/bin/python \
  -m qwen_decision.v3.validate_reference_engine
```

기존 kit에서는 model 옵션을 `runs/robust/rebuilt-g6w67-g3e2`로 지정한다.
프로젝트에 고정된 Decision Index submodule commit `f5a6e4fd85fed05e77878dfc4966d0568bfb45ec`를
PYTHONPATH로 사용하고 기존 `huggingface-hub==2.2.0`을 유지했다.
평가 중 GPU를 별도 학습과 공유했으므로 wall time을 production latency로 사용하지 않는다.

[수치·가중치/데이터 hash·구성요소 대조·엔진 검증 기록](g6-soup-replication-results.json)에 결과를 보존했다.
원문별 확률·예측은 `runs/robust/reference-comparison/`, 입력 가중치는 고정 Hub snapshot에 있다.
