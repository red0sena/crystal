# Talking-Heads: 기존 attention 내부의 head 혼합 비교

**완료(2026-10-23). 개선 없음, 진행 기준 미통과.** 9개 fit와 최종 평가를 마쳤다
(`runs/talking-heads/upper2-v1/report.json`). 주 평가 정확도 평균은 stock_sdpa 77.76%,
native 77.97%, diagonal 77.94%, talking 77.94%다. talking−stock_sdpa +0.18%p(계층 95%
[−0.39, +0.70]), talking−native −0.03%p([−0.63, +0.52]), talking−diagonal 0.00%p
([−0.47, +0.42]), head 간 혼합 제거 영향 +0.03%p([−0.21, +0.26])로 모든 구간이 0을
포함했다. `exploratory_advance_gate`는 false다.

(아래는 실행 전 계획 기록이다.) 단위 테스트8개와 실제0.8B 모델의 세 구조
smoke가 통과했고, talking의 optimizer 재개까지 정확히 일치했다.
`runs/talking-heads/upper2-v1/worker.py`로 실행했다.

## 가설과 구조

최근 semantic residual reader는 추가 경로를 제거해도 예측이 거의 바뀌지 않았다.
이번에는 마지막 두 full-attention 층(0-based 19, 23)의 기존 attention 연산 안에
head 혼합을 넣는다. 다른 네 full-attention, 18 causal DeltaNet, QK norm, RoPE,
기존 Qwen 출력 gate·projection, tied code readout을 유지한다.

[Talking-Heads Attention](https://arxiv.org/abs/2003.02436)의 softmax 전후 head 혼합을
Qwen3.5 GQA에 적용한 별도 retrofit이다. 논문의 사전학습 결과를 이번 짧은
후학습의 개선 보장으로 사용하지 않는다.

각 층에서 두 8×8 행렬을 사용한다. 두 층 합계 추가 파라미터는 256개다.
pre/post 행렬은 FP32 항등행렬로 초기화한다. head 혼합 후 padding mask를 적용해
마스킹한 무한대를 head끼리 혼합하는 오류를 피한다. weighted value까지 FP32로
계산한 뒤 원래 BF16 출력 dtype으로 돌린다.

## 대조군과 학습

| 평가 조건 | 추가 학습 | 구분하려는 효과 |
|---|---|---|
| stock_sdpa | 이전에 완료한 동일 조건 원본 3개 재사용 | 실제 원래 구현 대비 개선 |
| native | 이번에 3개 학습, 선택한 두 층만 같은 FP32 명시 계산 | fused SDPA와 수치 구현 차이 |
| diagonal | 이번에 3개 학습, 행렬의 대각 원소만 사용 | head별 temperature·출력 크기 조절 |
| talking | 이번에 3개 학습, 행렬 전체 사용 | head 사이 정보 혼합 |

diagonal은 256개 파라미터를 저장하지만 유효 자유도는 32개다. 파라미터 자유도가
같다는 주장은 하지 않는다. native/diagonal/talking의 초기 출력은 동일 연산에서
정확히 같다. 원래 fused SDPA와는 수학적으로 같은 함수여도 BF16 수치가 다를 수
있으며, stock_sdpa 비교를 따로 보고한다.

공통 시작점은 `runs/long-training/64k-4096-v1/native-seed41/step-4096`이다.
seed 41/73/101 각각 batch16×512회, target 4,096개+anchor 4,096개를 한 번씩 읽는다.
세 seed는 같은 시작 체크포인트의 추가 학습 변동만 나타낸다. 언어 backbone과 tied
head 전체를 학습하고 vision은 고정한다. 기존 gold CE만 사용하며 학습용 새 teacher
label·rationale·모델 조합은 없다.

body LR3e-6, 혼합 행렬 LR1e-3, FP32-master AdamW(.9,.95), body WD.01,
행렬 WD0, clip1, 32회 warmup 뒤 cosine 최종10%다. 정확도에 따른 조기 종료나
best checkpoint 선택 없이 마지막512회를 비교한다. 동일 입력·정답·코드·선택지
순서를 캐시한다. 이전 원본3개의512회 입력/정답 hash와 이번 캐시가 모두 같음을
확인하고, 시작점·학습량·학습률과 체크포인트 hash를 검사해 재사용한다.

## 새 평가와 기전 진단

이전 평가 문서 전체를 제외한 독립 문서에서 주 평가1,152개, retention384개,
dev88개를 고정한다. 주 평가는 CONDAQA128개와 VitaminC/BoolQ/SNLI/MNLI 각256개,
retention은 여섯 출처 각64개다. 출처별 정확도를 같은 가중치로 평균한다.
CONDAQA 잔여 독립 문서가 적어 수를 제한했다. 기존 학습/평가와의 정규화 원문·ID·
명시 문서 metadata 중복을 감사하지만 의미적 중복이나 원래 사전학습 오염까지
배제한다고 주장하지 않는다.

모든 최종 모델을 고정한 뒤 다음 네 비교에 문서/seed paired bootstrap5,000회와
Bonferroni4 동시 구간을 적용한다.

1. talking−stock_sdpa
2. talking−native
3. talking−diagonal
4. 같은 talking 모델의 active−no_cross

no_cross는 학습한 대각 원소를 보존하고 head 간 원소만0으로 만든다. identity는
두 행렬 모두 항등행렬로 되돌린다. 내부 층을 바꾸므로 각 intervention마다 전체
backbone을 다시 한 번 실행한다. 여러 intervention 출력을 조합해 정답을 내지
않는다. 각 보통 추론은 원래24층을 각각 한 번씩 실행하는지 감사한다.

별도로 고유한 record의 status를 묻는 프로그램 생성128문항에 무관한 사실을
0/8/32/96개 붙인512문항을 평가한다. 같은 record·정답·선택지와 근거의 상대 위치를
유지하고 방해 사실은 중첩해 추가한다. 이 자료는 학습에 사용하지 않는다.
긴 문맥에서의 추가 손실을 같은 base task와 seed로 묶어 비교하는 탐색적 검색
진단이며 자연어 추론 개선의 증거로 대체하지 않는다.

진행 기준은 네 동시 구간 하한>0, 첫 세 비교가 세 seed 모두 양수, 두 native
대조군 대비 retention의 단측95% 하한>−1%p다. 이는 다음 실험을 위한 탐색 기준이며
자동 확장하지 않는다. 준비한 같은 GPU batch의 지연시간·peak memory도 보고한다.
명시 attention 때문에 계산량과 메모리가 커질 수 있고 FLOPs가 같다는 주장은 하지 않는다.

## 재현과 보존

실행 모듈은 `qwen_decision.v3.talking_heads_experiment`, 구조는 `talking_heads.py`,
데이터 준비는 `talking_heads_data.py`다. 단계는 data→prepare→cache→reference→smoke→
freeze→run이다. unit 결과와 환경 metadata를 freeze 전에 기록한다. 실제 모델
smoke는 세 구조의 학습·저장/재로드·단일24층 실행을 검사하고, talking은 optimizer
재개 후 세 번째 업데이트의 master/moment와 출력까지 비교한다.

불완전한 학습은 보존 디렉터리로 옮긴 뒤 공통 시작점부터 해당 fit을 재시작한다.
최종 체크포인트는 weights-only다. 이전 실험 코드와 모델은 수정하지 않는다.
새 모델은 `qwen_decision.v3.talking_heads.load`로 읽어야 혼합 행렬이 적용된다.
