# 다음 Qwen Decision 구조 제안

[위키 홈](../README.md) · [평가 계획](evaluation.md)

**상태: 이 조합 그대로는 미구현이다(10-31 갱신).** 아래는 조사한 요소를 조합한 우리의 실험안이다. 단일 논문이 이 조합의 성능을 검증한 것은 아니다.
- 가까운 요소는 v3에서 g4b 위에 작게 시험했고, 모두 기존 code readout을 넘지 못했다.
  - A에 가까운 것: [후보별 판정 Query](../project/decision-native-next.md)(+0.05%p, 구간이 0 포함), [Pointer head](../project/pointer-pilot.md)(−5.05%p).
  - B에 가까운 것: [잔차 Set Transformer](../project/architecture-search.md)(selection +0.37%p였지만 동점 규칙으로 확인셋에 가지 않아 채택 안 함), [대체 readout·MLP](../project/readout-mlp.md).
  - C: teacher 증류는 10-17 방침으로 하지 않는다.
- 선택지 격리의 기대 이득은 작다. 학습 없이 4순서 평균으로 잰 이득이 짧은 문항 +0.7%p다(앙상블 효과 포함, 상한으로 단정하지 않음)([순서 측정](../project/order-probe.md)).

## A. 후보 독립 branch + 공유 scorer

공통 지문·질문을 Qwen으로 읽고, 동일 prefix 종료 상태를 후보별로 복제한다. 각 후보 내용과 동일 score delimiter를 읽은 hidden state에 작은 공유 scalar MLP를 적용한다. 후보 logits 사이 softmax로 결정한다.

참고: [Candidate-independent Attention](../papers/candidate-independent-attention.md), [RankT5](../papers/rankt5.md).

코드 토큰 prior와 코드 pool 한계를 제거할 수 있는 구조다. 후보 수는 출력층 고정 차원 대신 시간·메모리·입력 제한에 영향을 받는다. 독립 branch에서 후보끼리 논리 비교가 부족하고 suffix·cache 비용이 커질 수 있다. [Hybrid 분리 조건](../concepts/attention-isolation.md)

## B. 공유 문맥 + 후보 encoder + 작은 비교층

```text
지문·질문 → Qwen → 문맥 token 표현 H 또는 여러 요약 벡터
후보 i → 공유 후보 encoder → 후보 표현 c_i
                       ↓
            후보별 Cross-Attention(c_i, H)
                       ↓
              선택적 Set Attention
                       ↓
                공유 scalar scorer
                       ↓
                    softmax
```

참고: [Poly-encoders](../papers/poly-encoders.md), [GLiClass](../papers/gliclass.md), [Soft Contextualized Encoder](../papers/soft-contextualized-encoder.md), [Set Transformer](../papers/set-transformer.md).

초기에는 Backbone을 고정하고 새 head만 학습하여 유용한 표현이 있는지 본다. 개선이 부족하면 상위층·LoRA 적응을 별도 단계로 비교한다. 후보 표현은 표시 순서와 독립적으로 만들고, 후보 집합 비교층에는 표시 위치를 주입하지 않는다. Ordinal level은 의미적 수치 정보를 유지한다. 문맥 token 전체를 쓰는 구조와 압축 구조의 메모리·장문 근거 보존을 비교한다.

## C. 구조 유지 + reasoning 감독

**(2026-10-17 방침: 교사에게서 배우지 않는다. 아래 teacher 증류 항목은 v3에서 실행하지 않으며, 방침이 바뀌지 않는 한 제안으로만 남는다.)** 기존 code LM Head readout을 유지한 채 [InnerJev](../papers/innerjev.md)식 teacher 분포 증류를 대조군으로 둔다. A/B의 scorer가 안정된 뒤 [Dual-Head](../papers/dual-head-reasoning-distillation.md)식 설명 loss를 추가하여 효과를 분리한다. 0.8B teacher가 약하면 더 강한 teacher를 쓰는 실험은 논문의 self-distillation과 구분한다.

## 실험 순서

1. 원본·v1·v2의 입력 및 평가 규약을 고정하고 미학습 test를 확보.
2. 기존 readout과 A를 비교: 동일 데이터·학습량, CE, 순서 및 코드 교차 검사.
3. A의 제한이 보이면 B의 작은 집합 비교층을 추가하고 candidate interaction 과제 평가.
4. CE-only와 reasoning 감독을 비교. 같은 frozen teacher targets를 사용해 구조 효과와 학습 효과를 구분.
5. 모델·설정을 dev에서 선택한 후 별도 dev calibration, 최종 test 1회 보고.

현재 논문들은 방법 후보를 제공한다. 새로운 head가 기존 pretrained LM Head보다 반드시 좋거나 0.8B reasoning 한계를 해결한다고 전제하지 않는다.

