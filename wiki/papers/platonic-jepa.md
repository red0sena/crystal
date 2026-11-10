# PRH를 JEPA의 조건부 정리로 만들기: 회전 동치의 의미와 한계

[위키 홈](../README.md) · [우리 Qwen 실험과 사후 감사](../project/rotation-alignment.md)

- 논문: **Proving and Improving the Platonic Representation Hypothesis with JEPAs**
- 저자: Randall Balestriero, Yann LeCun.
- 검토 대상: **arXiv:2610.39302v1**, 2026-10-29 제출, PDF 37쪽. 아래 페이지는 PDF의 1-based 페이지다.
- 원문: [초록](https://arxiv.org/abs/2610.39302v1) · [PDF](https://arxiv.org/pdf/2610.39302v1) · [HTML](https://arxiv.org/html/2610.39302v1).
- 조사일: 2026-11-04. 본문 §1–7, 증명 부록 A, 설정 B, 추가 결과 C를 대조했다.
  저자들의 대규모 학습을 독립 재현하거나 증명을 형식 검증했다는 뜻은 아니다. 심사·게재 상태는 별도로 확인하지 않았다.
- 공식 구현: [galilai-group/prh-proof](https://github.com/galilai-group/prh-proof), 정적 코드 검토는
  커밋 [`a2b9949fa907d4fbe562742d4b1ffa7205ab7427`](https://github.com/galilai-group/prh-proof/tree/a2b9949fa907d4fbe562742d4b1ffa7205ab7427)에 고정했다.
  [검토한 공식 파일과 SHA256](platonic-jepa-sources.json)을 함께 보존한다.

## 1. 읽고 나서 유지할 결론

PRH는 서로 다른 모달리티의 모델들이 규모가 커질수록 세계의 공통 통계적 표현에 가까워진다는 가설이다.
이 논문은 그 일반 가설 전체를 무조건 증명하기보다, JEPA에서 성립하는 충분조건과 학습 설계를 제시한다.
PRH 원 논문은 이 논문의 참고문헌 [22]이며, 여기서는 현재 논문의 기술과 인용 범위에서만 설명한다.

**정해진 이상적 조건 아래, 각 모달리티의 JEPA가 동일한 세계의 동일한 관계를 학습하면
최적 표현이 모달리티별 직교 변환만 다르게 된다. 좋은 관측 쌍은 이를 유도하는 학습 설계다.**

이는 임의의 기존 이미지·텍스트 인코더가 이미 정확한 회전 동치라는 명제가 아니다.
회전 행렬을 적합하는 절차와, 인코더 학습이 회전 가능한 기하를 만들도록 하는 절차도 다르다.
우리의 고정 Qwen 판정 벡터 실험은 전자를 작은 합성 분포에서 시험했다.

| 주장 | 근거의 종류 | 정확한 범위 |
|---|---|---|
| 별도 학습한 표현이 하나의 직교 변환으로 일치 | Theorem 1 | 공통 세계·관측 가역성·공통 positive 분포·회전까지 유일한 모집단 최적해 |
| whitening MSE에서 유일성의 충분조건 | Proposition 1 | 지정 차원 경계의 eigengap이 있는 spectral 해 |
| positive 관계가 다르면 정렬 오차가 제한됨 | Theorem 2 | 공통 주변분포를 가진 spectral 해, kernel 차이와 eigengap으로 정한 상계 |
| nuisance에 쓴 표현 차원은 모달리티 간 일치를 제한 | Theorem 3 / Proposition 3 | 중심화·whitening, 조건부 독립 nuisance 및 지정 corruption 모델 |
| 다음 토큰 예측은 JEPA 형태로 쓸 수 있음 | Proposition 4 | softmax 손실의 대수적 분해. 앞 정리들의 결론을 자동 상속하지 않음 |
| 좋은 쌍이 같은 데이터의 다른 목적함수보다 잘 정렬 | 실험 §3–4 | 논문이 사용한 데이터·모델·평가 프로토콜 |
| 109M이 72B보다 좋음 | 실험 §4.3 | 특정 시각 인코더와의 표현 정렬 점수. 언어·추론 능력 전반의 우위가 아님 |
| 회전을 찾으면 두 모달리티를 연결할 수 있음 | 실험 §5 | 별도 학습한 인코더와 캡션 decoder를 연결. 정확한 동치가 관측됐다는 뜻은 아님 |

## 2. ‘회전하면 같다’는 수학적 의미

[§2.1, Theorem 1 / Proposition 1, PDF p.3; 증명 A.1–A.2, p.20](https://arxiv.org/pdf/2610.39302v1#page=3).

세계는 유한 상태 집합 `Z`, 상태 분포는 `p(z)>0`다. 모달리티 `m`은 상태를
`x=g_m(z)`로 관측한다. 처음 정리에서는 `g_m`이 **단사**이므로 각 관측에서 상태를 구별할 수 있다.
학습 쌍의 세계 상태 `(z,z')`는 대칭 분포 `π_m`에서 나오며 두 주변분포가 `p`다.

인코더의 목적함수는 다음 형태다.

$$
L_m(f)=\mathbb E_{(z,z')\sim\pi_m}\ell(f(g_m(z)),f(g_m(z')))+\Omega(f).
$$

`Ω`는 임베딩 분포에만 의존하는 붕괴 방지 항이며, `ℓ`과 `Ω`는 직교 변환에 불변이다.
가까운 쌍의 MSE만 줄이면 모든 입력을 같은 상수 벡터로 보낼 수 있으므로 붕괴 방지 제약이 필요하다.
분석의 predictor는 identity다. **학습된 predictor나 action에 조건화된 미래 예측 JEPA는
이 분석에 포함하지 않는다.**

모든 `π_m`이 같고, 세계 위 목적함수의 최적해들이 하나의 직교 변환 궤도에 속하면,

$$
f_m^\star(g_m(z))=Q_m h^\star(z),\qquad Q_m\in O(k).
$$

따라서 모든 상태에 대해 같은 행렬 `Q_m Q_{m'}^T`가 두 모달리티의 표현을 연결한다.
샘플마다 다른 회전이 아니다. `O(k)`는 `Q^T Q=I`인 **직교군**으로, 엄밀히는 반사도 허용한다.
논문의 rotation을 항상 determinant `+1`인 순수 회전으로 읽으면 범위를 좁히게 된다.

증명의 핵심은 `h=f∘g_m`로 쓰면 단사 관측이 학습 문제의 좌표계만 바꾼다는 것이다.
그러나 **공통 문제의 최적해가 직교 변환까지 유일하다는 가정이 필수**다.
일반적인 finite-width 신경망이나 SGD가 그 모집단 최적해에 도달한다는 보장은 여기서 나오지 않는다.
같은 표현이라는 결론 자체도 그 표현이 모든 가능한 판단에 충분한 통계이거나,
세계의 모든 정보를 가역적으로 보존한다는 결론은 아니다.

### spectral 해에서 유일성이 생기는 조건

`ℓ(a,b)=||a-b||²`이고 `E_p h=0`, `E_p hh^T=I_k`를 강제한다.
`Π_zz'=π(z,z')`, `D=diag(p)`, `K_π=D^{-1/2}ΠD^{-1/2}`로 정의한다.
상수 방향을 뺀 상위 고유방향으로 spectral embedding `Φ_π`를 만들면,

$$
\min L_\pi=2\sum_{i=1}^k(1-\mu_i),\qquad
\mu_k>\mu_{k+1}\ \Longrightarrow\ h^\star=Q\Phi_\pi.
$$

증명은 MSE를 `2k-2Σ_j〈h_j,T_πh_j〉`로 전개한 뒤 Ky Fan 원리를 적용한다.
gap은 상위 `k`차원 부분공간을 고유하게 만들고, 그 안의 기저 선택만 직교 변환으로 남긴다.

LeJEPA 실험의 SIGReg는 전체 분포를 등방 Gaussian 쪽으로 보낸다. 논문은 이 **전체 분포 제약의
유일성까지 증명하지 않는다.** 유한 상태의 이산 임베딩 분포가 연속 Gaussian과 정확히 같을 수도 없다.
Proposition 1의 hard whitening과 실험의 SIGReg를 같은 조건으로 취급하지 않는다.
SIGReg의 회전 불변성도 무작위 projection 방향에 대한 기대값에서의 성질이다.
유한 개 방향을 고정한 수치 계산이 모든 회전에 정확히 불변이라는 뜻은 아니다.
VICReg·Barlow Twins도 논문에서 지정한 중심화·hard whitening 극한 조건 아래에 연결되며,
일반적인 실용 설정 전부에 Proposition 1을 그대로 적용하지 않는다.

## 3. 같은 장면의 관측 쌍만 있으면 충분한가

**좋은 관측 쌍이라는 말만으로 위 정리의 가정을 모두 채우지는 않는다.**
다음은 이 점을 설명하는 우리의 수학적 예시이며 논문에 대한 반례는 아니다.
논문이 요구한 유일성/eigengap을 제거하면 결론이 성립하지 않는 예다.

상태가 3개, `p=(1/3,1/3,1/3)`, 모든 positive가 같은 상태끼리여서 `Π=D`라고 하자.
그러면 `K=I_3`이고 고유값이 모두 1이므로 `k=1`에서 gap이 없다.

$$
h_A=\sqrt{3/2}(1,-1,0),\qquad h_B=(1,1,-2)/\sqrt2.
$$

둘 다 평균 0, 두 번째 모멘트 1이고 positive MSE는 0이다. 하지만 `O(1)={+1,-1}` 중
어느 것으로도 둘을 같게 만들 수 없다. 최적 부호를 적용한 평균 제곱 차이는 2다.
[수치 확인](../project/rotation-alignment-audit.json)의 `same_state_only_counterexample`에 보존했다.

직관적으로, 한 장면의 여러 관측을 같게 만드는 것과 **서로 다른 장면들이 배치되는 기하를
유일하게 정하는 것**은 별도 문제다. 실제 학습에서는 데이터의 연결, augmentation의 겹침,
신경망의 inductive bias 등이 중요하다. 정리는 이 모두를 자동으로 해결했다고 주장하지 않는다.

## 4. positive 관계가 조금 다르면 무엇을 보장하는가

[§2.2, Theorem 2, PDF p.3–4; A.3, p.20–21](https://arxiv.org/pdf/2610.39302v1#page=4).

`π_1`, `π_2`가 **같은** 주변분포 `p`를 가지며 `Φ_1`, `Φ_2`가 spectral 해이고,
두 번째 kernel의 gap이 `δ=μ_k-μ_{k+1}>0`라면,

$$
\min_{Q\in O(k)}\left(\mathbb E_p\|\Phi_1-Q\Phi_2\|^2\right)^{1/2}
\le {2\sqrt2\over\delta}
\min\left(\sqrt{k}\|K_{\pi_1}-K_{\pi_2}\|_2,
\|K_{\pi_1}-K_{\pi_2}\|_F\right).
$$

Davis–Kahan 형태의 섭동 정리를 적용한 결과다. kernel 차이가 작고 gap이 클수록 상계가 작다.
상계가 크면 유용한 보장이 아닐 수 있다. **공통 주변분포를 가진 spectral 해**에 대한
RMS 거리의 상계이며, 임의의 기존 신경망의 오차를 수치로 보장하는 식은 아니다.
caption이 이미지의 일부만 담는 정보 손실 문제를 직접 해결한 정리도 아니다.

## 5. good views, bad views, nuisance

[§2.3, PDF p.4–6; 증명 A.4 p.21–23](https://arxiv.org/pdf/2610.39302v1#page=4).

관측을 `x=g_m(z,ν)`로 확장한다. `ν`는 어휘 선택·시점·조명 같은 모달리티별 nuisance다.
다만 다른 모달리티와의 정렬에 불필요하다는 뜻이다. 카메라를 조작하는 world model에는 시점이 필요할 수 있다.
이 절의 기본 모형은 상태와 nuisance의 독립성도 가정한다. 모든 실데이터의 시점·문체가
상태와 실제로 독립이라는 주장을 하는 것은 아니다.

- **Good views:** 세계의 관계 `π`로 연결된 상태를 독립적인 nuisance로 관측한 두 입력.
  실험에서는 같은 이미지의 서로 다른 caption이나 같은 clip의 다른 frame을 사용한다.
- **Bad views:** 한 관측을 두 번 변형해 그 관측의 nuisance를 공유하는 두 입력.
  한 사진의 두 crop, 한 문장의 두 masking이 해당한다.

Theorem 3는 중심화·whitening된 표현에 대해 `C_f=Cov(E[f(x)|z])`,
`ρ(f)=tr(C_f^{1/2})/k`를 정의한다. 상태에 조건화했을 때 두 모달리티의 nuisance가 독립이면,

$$
\min_Q\mathbb E\|Qf_1-f_2\|^2\ge 2k(1-\min_m\rho(f_m)).
$$

이는 **제곱 거리의 하계**다. Theorem 2의 RMS 거리 상계와 단위를 혼동하지 않는다.
`ρ`도 일반적인 설명 분산 `tr(C_f)/k`와 다르다. 제곱근 행렬의 trace를 사용한다.

Proposition 3는 상태와 nuisance에 각각 독립적으로 작용하며 관측분포를 보존하는 corruption을 가정한다.
상태 유지 정도의 고유값 `α_i`와 nuisance 유지 정도 `β_j`의 곱이 후보 방향이 된다.
필요한 spectral separation 아래 `β_1>α_k`이면 nuisance 방향이 상위 `k`개에 진입한다.
그 수가 `r`이면 다른 모달리티와의 최적 제곱 정렬 오차가 적어도 `2r` 남는다.
**모든 crop이 언제나 나쁘거나 반드시 nuisance를 학습한다는 무조건적 명제가 아니다.**

Theorem 4는 일반적인 관측 쌍 분포 `W_m`이 유도하는 세계 kernel `π_m^W`를 다룬다.
상태와 nuisance의 혼합이 작고, 세계의 `k`번째 고유값이 다음 세계 방향과 nuisance 유지 정도보다 크면,
관측의 spectral 해는 유도된 세계 kernel의 해에 가깝다. 두 모달리티의 최적 정렬 거리와 세계 쪽 거리의
차이도 각 모달리티 근사오차 `τ_1+τ_2` 이내다. 데이터 양뿐 아니라 **어떤 상태를 연결하는
데이터와 view인가**가 학습할 세계 kernel을 결정한다는 정리다.

## 6. 다음 토큰 예측에 대한 주장의 범위

[§2.4, Proposition 4, PDF p.6–7; A.5 p.23](https://arxiv.org/pdf/2610.39302v1#page=6).

문맥 표현 `h(c)`와 출력 token 벡터 `w_v`에 대한 softmax loss는

$$
-\log p(y|c)=-\langle h(c),w_y\rangle+\log\sum_v\exp\langle h(c),w_v\rangle
$$

로 분해된다. 저자는 첫 항을 prediction, 두 번째를 anti-collapse에 대응시킨다.
문맥과 다음 token을 같은 문서 내부의 positive로 취급하는 일반화 JEPA라는 해석이다.

충분한 표현 차원과 full-support 조건부 분포를 가정한 최적 logit 분해에서는,
`h→Gh`, `w→G^{-T}w`라는 **일반 가역 선형변환 `GL(k)`**도 logits를 보존한다.
whitening한 spectral JEPA의 `O(k)` 식별성과 다르다. 이는 손실 수준의 분해이며,
임의의 `G`를 실제 tied embedding·Transformer 파라미터에서 모두 실현할 수 있다는 주장은 아니다.

문서의 문체·어휘를 공유하므로 nuisance를 유지하기 쉽다는 설명은 저자의 이론적 해석이다.
논문은 다음 토큰 예측이 하나의 공간의 spectral JEPA가 아니므로,
Proposition 3 / Theorem 4를 연결하는 설명을 **heuristic**이라고 명시한다.

**CE라는 손실식 자체가 bad views를 뜻하지 않는다.** 우리 gold 선택지 CE는
다음 토큰 언어모델의 자기지도 학습과 학습 쌍의 의미가 다르다.
이 논문에서 Decision의 gold CE를 버려야 한다는 결론은 나오지 않는다.

## 7. 회전을 찾는 일과 회전 가능한 표현을 학습하는 일

[§2.2 Proposition 2, PDF p.4; A.3 p.21](https://arxiv.org/pdf/2610.39302v1#page=4).

이상적인 중심화·whitening 표현의 평균과 공분산만으로는 대응 없는 두 공간의 회전을 알 수 없다.
반면 정확하고 잡음 없는 `Y=QX`가 성립하고, 대응하는 `n≥k`개 벡터가 `R^k`를 span하면,
대응 쌍으로 회전을 유일하게 결정한다. **일반 실데이터에서 k쌍이면 충분하다는 표본복잡도 결과가 아니다.**

유한 점군의 고차 구조가 비대칭이면 대응 없는 기하로도 회전을 추정할 수 있다.
그러나 등방 Gaussian처럼 분포 자체가 모든 회전에 불변이면, 대응 없는 분포만으로 방향을 식별할 수 없다.
공유 기하를 쓰는 방법도 그 기하에 충분한 식별 구조가 있어야 한다.

### 실용 구현에는 중심화와 직사각형 사상이 들어간다

공식 [`prh/align.py`](https://github.com/galilai-group/prh-proof/blob/a2b9949fa907d4fbe562742d4b1ffa7205ab7427/prh/align.py)의
`Rotation.fit` / `Rotation.__call__`은 행벡터를 **`(x-μ_a) @ R + μ_b`**로 보낸다.
실제 파이프라인은 순수 원점 회전 외에 평균 보정도 포함한다.

이미지와 텍스트의 차원이 다르면 행 또는 열이 정규직교인 직사각형 행렬을 사용한다.
행벡터 `x∈R^768`을 `R∈R^{768×384}`로 보낼 때 `R^T R=I_384`이지만
`R R^T=I_768`은 아니다. 일반 입력의 전체 길이·거리·정보를 모두 보존하는 정방 직교변환이 아니다.
논문도 이 다른 차원의 설정이 정리의 직접적인 적용 범위 밖이라고 명시한다.

직사각형 SVD 구현은 `tr(R^T A^T B)`를 최대화한다. 차원을 줄이는 경우 이것이 일반적인
최소제곱 문제를 항상 푸는 것은 아니며, source가 isotropic이면 동치라는 조건을 코드가 설명한다.
따라서 실험 전부를 “아무것도 잃지 않는 순수 회전 한 번”으로 요약하지 않는다.

## 8. 정렬 지표를 구분해야 하는 이유

[§3 Metrics / B PRH protocol; 공식 metrics](https://github.com/galilai-group/prh-proof/blob/a2b9949fa907d4fbe562742d4b1ffa7205ab7427/prh/metrics.py).

| 지표 | 비교하는 것 | 회전 적합 |
|---|---|---|
| mutual 10-NN | 각 모달리티 내부의 10개 이웃 표본 ID가 얼마나 겹치는가 | 불필요 |
| linear CKA | 두 표본 Gram 행렬의 관계 | 불필요. 논문은 unbiased HSIC 사용 |
| 회전 후 image→caption R@1 / R@5 | 별도 쌍에서 찾은 사상으로 올바른 caption을 검색하는가 | 필요 |
| 대응 벡터 cosine | 대응 이미지·텍스트 벡터의 방향이 가까운가 | 회전·중심화에 영향받음 |
| Decision 정답률·NLL | gold 선택지를 잘 판단하는가 | 논문의 주 PRH 지표가 아님 |

이미 전처리한 `X`에 정방 직교행렬 `R`만 적용하면

$$
(XR)(XR)^T=XX^T.
$$

따라서 내부 거리, cosine 이웃, linear CKA는 변하지 않는다.
**회전은 이미 있는 기하의 좌표를 맞추며, 내부 기하 자체를 더 좋게 만들지는 않는다.**
반면 모달리티 사이의 직접 cosine, 검색 결과, 고정 출력층의 판단은 달라질 수 있다.

WIT PRH protocol은 층별로 절댓값의 95% quantile에서 원소를 clip하고 L2 정규화한 뒤,
층 쌍의 최대 점수를 보고한다. 원소별 clipping은 회전과 가환하지 않으므로,
회전 뒤 clipping을 다시 적용하는 경우까지 위 불변성을 주장하면 안 된다.
우리 cosine 0.979와 논문의 mutual 10-NN 0.184는 서로 다른 값이다.

## 9. 실험이 지지하는 주장

아래는 **논문 보고값**이다. 우리 Qwen 측정값과 구분한다.

### 같은 작은 데이터에서 view 구성만 바꾸기

[§3 Tables 1–2, PDF p.7–12](https://arxiv.org/pdf/2610.39302v1#page=7).

Flickr30k의 29k 이미지와 145k caption으로 이미지·텍스트를 별도로 학습한다.
텍스트는 33M BERT이며 good은 같은 이미지의 다른 caption, bad는 같은 caption의 다른 span,
다음 토큰 대조군은 같은 BERT를 causal로 바꾼다. 이미지·텍스트 사이의 loss는 없지만,
**어떤 caption들이 같은 이미지에 속하는지**는 good 구성에 사용한다.

| 텍스트 학습 | 별도 소형 이미지 JEPA와 mutual 10-NN | 같은 이미지 JEPA와 R@1 | frozen DINOv2와 R@1 |
|---|---:|---:|---:|
| good-view JEPA | .062 | .044 | .246 |
| bad-view JEPA | .047 | .027 | .133 |
| 다음 토큰 예측 | .042 | .021 | .118 |

R@1은 1,014 validation 쌍으로 사상을 맞추고 1,000 test caption에서 검색한 값이다.
다음 토큰 대조군에도 같은 이미지의 caption 2개를 이어 보여 주지만 이득이 작다.
관계 정보를 주는 것뿐 아니라 목적함수에서 그 관계를 어떻게 활용하는지가 중요하다는 실험이다.

good-view의 trimming을 없애면 DINOv2 R@1은 .246→.185다.
SIGReg 가중치·view 수에 따라 학습이 붕괴하는 설정도 있어 clipping 등이 필요했다.
이미지 쪽 good views를 MSR-VTT의 다른 frame으로 시험했지만,
대규모 이미지 good-view JEPA까지 검증한 것은 아니다. 대부분의 이미지 쪽은 crop형이다.

### 대규모 비교와 ‘72B를 넘었다’의 의미

[§4.3 Table 3 / C Tables 14–18, PDF p.11–12, 33–35](https://arxiv.org/pdf/2610.39302v1#page=12).

| 텍스트 인코더 | WIT mutual 10-NN / DINOv2-B | 같은 WIT unbiased CKA | COCO-5k mutual 10-NN |
|---|---:|---:|---:|
| good-view JEPA 109M, 3 seed 평균 | .184 | .507 | .161 |
| 같은 데이터·109M의 다음 토큰 대조군 | .111 | 미측정 | 미측정 |
| 같은 조건의 bad-view JEPA | .088 | 미측정 | 미측정 |
| Qwen2.5-72B | .154 | .418 | .108 |
| Llama-3-70B | .151 | .380 | .113 |
| CLIP B/32 text | .124 | .339 | .132 |

good과 같은 데이터의 대조군 사이 차이가 view 설계의 효과를 판단하는 중심 증거다.
사전학습 LLM과는 학습 데이터·목표·능력 분포가 다르므로, 이 비교를 **일반 언어·추론 능력의
효율 우위**로 해석하지 않는다. CLIP text와 비교한 것도 CLIP 자신의 이미지 인코더와의 합의가 아니라,
독립적인 DINOv2 등과의 합의다.

### ‘42배’에 붙여야 하는 조건

[§4.2, PDF p.11](https://arxiv.org/pdf/2610.39302v1#page=11).

- 같은 109M 구조·같은 코퍼스 계열에서 good은 21.6M 학습 sample로 WIT .109,
  다음 토큰 대조군은 900M sample로 .111이다. `900/21.6≈41.7`이다.
- good sample은 텍스트 3개를 읽으므로 **텍스트 token 수로는 약14배**다.
- 유일한 이미지 출처 그룹 수는 2.16M 대 940M이므로 약435배다.
- learning rate는 `1e-4` 대 `2e-4`, batch는256 대2,048이다. 동일 계산량 비교는 아니다.
- 저자는 이보다 이른 다음 토큰 checkpoint를 평가하지 않았다. 해당 점수에 처음 도달하는
  최소 sample 수가900M이라고 확정한 결과는 아니다.

“모든 Decision 학습이42배 빨라진다”, “GPU 시간·라벨·token이42분의1이면 된다”는
예산 추정에 이 숫자를 사용하면 안 된다.

### 경험적인 scaling law

[§4.4 / C, PDF p.12–13, 25–26](https://arxiv.org/pdf/2610.39302v1#page=12).

good 비율 `q`, sample 수 `M`, 유일한 이미지 출처 그룹 수 `N`으로
`R@1=a+d log M+b log(1+q^α M^β N^γ/c)`를 적합한다.
미학습85 run의 예측 RMSE .0093은 seed가 다른 run 사이 RMSE .0096과 비슷했다.
그러나320M sample의 긴 학습에서 good-only 예측 .455에 실측 .418처럼 과대 예측한다.
지수도 재적합에 따라 달라진다. Theorem 4에서 유도한 보편 법칙은 아니다.

## 10. ‘이미지를 보지 않았다’와 ‘paired 정보가 없다’의 차이

[§4.3 / §7 / C Table 7, PDF p.11, 15, 24–25](https://arxiv.org/pdf/2610.39302v1#page=15).

Recap-DataComp의 각 이미지에는 alt-text와 LLaVA recaption 2개를 쓴다.
텍스트JEPA가 pixel을 직접 입력하지 않는 것은 맞다. 다만 LLaVA는 CLIP ViT-L/14와
LLaMA-3-8B를 사용하는 이미지 captioner이므로 **이미지·텍스트 학습 지식이 설명문에 간접적으로 들어간다.**

이미지 captioner 없이 alt-text와 텍스트 전용 모델의 paraphrase를 쓰면 WIT .060,
같은 이미지의 alt-text와 LLaVA recaption을 쓰면 .107이었다.
같은 alt-text bad views의 .055보다 .060이 좋으므로 captioner 없는 view 효과도 보고하지만,
대규모 우위 전체를 시각 지식 없는 독립 언어 학습의 효과로 돌릴 수는 없다.
동일 텍스트를 쓰는 good/bad/다음 토큰 대조군은 이 정보량 차이를 억제한다는 데 가치가 있다.

paired 정보는 caption 그룹, 사상 적합용 대응, Recap captioner라는 서로 다른 지점에 들어간다.
**cross-modal loss가 없다는 것과 paired 정보가 전혀 없다는 것은 다르다.**

## 11. captioner와 완전 pair-free 사상 추정은 별도 실험이다

[§5.1 / Tables 9–12, PDF p.13, 27–30](https://arxiv.org/pdf/2610.39302v1#page=13).

caption decoder는 텍스트JEPA의 caption 표현에서 문장을 복원하도록 학습한다.
이미지에는 frozen DINOv2/DINOv3, 사상, decoder를 연결한다.
이미지·caption을 동시에 받는 network loss는 없어도 **사상은30k Flickr 대응으로 적합한다.**
최선 captioner의 고차원 사상에는 약1.9M 자유도가 있다. 행렬 하나라고 학습이 없는 것은 아니다.

COCO CIDEr .838은100M 이미지 출처 텍스트로 학습한 text JEPA, Qwen2.5-1.5B decoder,
DINOv2-g와 DINOv3-L 특징, 32sample consensus 등을 조합한 결과다.
109M 최대 정렬 모델의 표와 captioner 설정을 같은 실험으로 섞지 않는다.
CIDEr는 논문에서 소수로 표기하며 통상100배 표기라면83.8이다. 값의 범위가 반드시0–1이라는 뜻은 아니다.

기본 captioner의 held-out 이미지·caption cosine은 약.29–.33이며,
decoder에는 그 오차를 견디도록 Gaussian noise를 넣어 학습한다.
**실용 모델에서 정확한 등식이 달성됐기 때문에 caption이 나온다는 설명은 틀리다.**

[§5.2 / Table 4, PDF p.14; C p.28](https://arxiv.org/pdf/2610.39302v1#page=14)의
pair-free 실험은 Schnaus 등의 기하·optimal transport 방법을 사용한다.
COCO train의 서로 다른 절반에서 이미지와 caption을 가져와 대응 없이 사상을 추정한다.
109M JEPA는 CIFAR-10 top-1 86.0%, CIFAR-100 top-5 67.2%, ImageNet-100 top-5 64.4%다.
같은 조건 Qwen3-8B의 ImageNet-100 top-5는47.6%, CLIP text는68.7%다.
모든 방법·지표에서 최고라는 뜻은 아니다.

FOSCTTM .018은0이 완전·.5가 chance인 별도 지표다.
저자는3학습seed×5solver seed의15run을 독립으로 취급한 검정의 계층 의존성도 명시한다.
pair-free는 사상 추정 단계의 조건이며, 사전 view 그룹·Recap 지식까지 사라지는 것은 아니다.

## 12. 결론을 과장하게 만들 수 있는 제한

- **이론과 실험의 간격:** 유한·단사·공통 주변분포·모집단 최적해·지정 제약이라는 이론 조건을,
  실험은 어느 것도 엄밀히 모두 만족하지 않는다고§7에 명시한다.
- **표현의 위치:** 실험의 LeJEPA는 projector와 SIGReg를 쓰지만 주 평가에는 backbone 표현을 사용한다.
  부록은 projector 출력의 결과도 별도로 보고한다. hard whitening 정리의 조건을
  실제 backbone 임베딩이 정확히 만족한다고 가정하지 않는다. projector의 Flickr R@1은
  .030, 별도 whitening 후 .034였으며, 원 backbone 검색값과 구분한다(C Table5 / Projector outputs).
- **가중치 선택:**940M의3seed 평균 .184는 실효 SIGReg 가중치 .01을 쓴다.
  실효 가중치가2배인 다른 run은 .121이며, 이를 본 뒤 WIT에서 가중치를 선택했다.
  4run 평균은 .168이다. 같은 설정의 나쁜 seed를 버린 경우와도, 전혀 선택에 쓰지 않은 test와도 구분한다.
- **스케일 요소 혼합:** 대규모화에서 유일한 데이터·학습sample·batch·실효 SIGReg 가중치가 함께 바뀐다.
  전체 상승을 데이터 양 하나의 인과효과로 분해할 수 없다.
- **평가 중복:** WIT caption에서 적어도43/1,024의 일치가 발견됐다.
  이를 뺀 저자 검사9모델의 변화는 각각 .002 이내지만, 미검출 중복·WIT 이미지 중복을 배제하지 못한다.
  제외 후 .193을3seed 평균 .184와 직접 빼지 않는다. seed0 계열의 .191과 구분해야 한다.
- **평가 시 선택:** WIT은1,024쌍에서 층 쌍 최대값을 보고한다.
  subsampling 구간도 최선 층을 고정하고 이웃 통계를 평균형 통계로 근사한다.
  층 선택·모든 하이퍼파라미터 탐색·미지 도메인을 포함하는 완전한 불확실성은 아니다.
- **조정의 비대칭:** 하이퍼파라미터는 주로 good-view 쪽에서 조정했고 대규모 bad-view 대조군은1seed다.
- **bad라는 이름의 범위:** DINOv2와 대부분의 이미지JEPA도 저자의 정의에서는 bad-view다.
  모든 하위 과제 성능이 나쁘다는 등급이 아니다.

## 13. 우리 Decision 모델에 연결하는 방법

현재 VLM은 이미지 특징을 기존 언어 입력 공간에 넣고 공통 백본·code head로 판단한다.
별도로 학습한 unimodal encoder들을 처음 연결하는 논문 설정과 다르다.
공통 VLM을 쓴다는 사실만으로 실용적인 회전 보정 실험이 무효가 되지는 않는다.
다만 무엇을 인코더 출력으로 정의하고, 미학습 장면뿐 아니라 어떤 질문·판정으로 전이를 시험했는지를 구분해야 한다.

| 적용 방향 | 확인할 질문 | 상태 |
|---|---|---|
| 고정 Qwen의 마지막 판정 표현에 직교 사상 | 기존 이미지·텍스트 readout 차이를 사상으로 보정할 수 있는가 | [합성pilot 실시](../project/rotation-alignment.md) |
| 질문 이전의 고정 Qwen 상태 정렬과 text-only 조건부 모듈 | 상태를 여러 질문에 재사용하면서 텍스트 판정을 이미지로 일부 옮길 수 있는가 | [수정 실험 실시](../project/state-rotation-v2.md). seen 일부 전이, unseen 텍스트 기준선 부족 |
| 동일 합성 데이터에 Crystal 정답CE full FT 추가 | Decision 학습 뒤 같은 상태 정렬·판정 전이가 개선되는가 | [동일 데이터 비교 완료](../project/crystal-state-methods.md). 정상 attention에서 검색은 개선, 판정 전이 추가 개선 미확인. JEPA 학습이 아님 |
| 독립 이미지encoder와 text Decision 표현 연결 | 공유 정보·차원·질문 의존성을 맞춰 판정 모듈을 전용할 수 있는가 | 미실시 |
| 상태 인코더의 good-view 학습 | 실제 답을 만드는 상태에 CE+LeJEPA를 걸면 기하와 판정 전이가 좋아지는가 | [공유 adapter 후속 비교](../project/jepa-decision.md). 독립 모달리티 인코더의 JEPA 학습과 구분 |
| 의미 보존 Decision 입력 쌍의 일관성 학습 | gold CE에 표현 변경의 강건성을 더할 수 있는가 | 기존 선택지 순서 일관성 실험과 구분 필요 |

caption이 판단에 필요한 사실을 빠뜨리거나, 이미지 token열과 전역 벡터를 혼동하거나,
같은 상태의 다른 질문들을 무조건 같은 표현으로 모으면 안 된다.
정확한 등식이라면 하위 함수를 전용할 수 있지만 근사적 정렬에는 **판정 margin과 사상 오차의 관계**도 필요하다.

열벡터에서 정렬 오차가 `e`이고 후보 가중치가 `w_a,w_b`라면 두 후보 logit 차이의 변화는
`||w_a-w_b||·||e||` 이하이다. 정답 margin이 이 값을 넘으면 순위를 보존할 수 있다.
평균 cosine이 높다는 사실만으로 모든 문항의 margin 보존은 보장하지 못한다.
정렬 뒤 선형 head를 쓰는 계산이 head 회전과 동치라는 점도 그 자체로 실패의 근거가 아니다.
인코더 기하의 유사성, 실제 대응 회전의 추정 가능성, 하위 판단의 전이를 각각 다른 대조로 검사해야 한다.

## 14. 출처와 다시 확인할 위치

| 확인 사항 | 원문·구현 |
|---|---|
| Theorem1·spectral 유일성 | §2.1 / A.1–A.2, PDF p.3 / 20 |
| mismatch·회전 식별 | §2.2 / A.3, PDF p.3–4 / 20–21 |
| nuisance·일반 kernel | §2.3 / A.4, PDF p.4–6 / 21–23 |
| 다음 토큰 손실·GL(k) | §2.4 / A.5, PDF p.6–7 / 23 |
| 최적화·특징·평가 설정 | B, PDF p.23–24 |
| captioner·noise·사상 자유도 | §5.1 / C Tables9–12, PDF p.13 / 27–30 |
| Recap 증류·seed·중복·평가 제약 | §7 / C Tables7,14–18, PDF p.15 / 25 / 33–35 |
| 행렬 fit·중심화 | 고정 커밋 `prh/align.py` |
| 이웃·CKA·층 탐색·subsampling | 고정 커밋 `prh/metrics.py` |
| 실험 손실 | 고정 커밋 `prh/forward.py`, `prh/views.py` |
| 공식 의존성 | `docs/design.md`, `docs/reproduce.md`: stable-pretraining `ab836bf699a2be712dfcf980a5eb70d36391e876` |

공식 코드는 정적으로 읽어 식·대표 표와 대조했다. 공개 테스트 전체와 대규모 학습은 실행하지 않았다.
정리의 충분조건, 실험적 지지, 우리 적용 가설을 각각 별도의 근거 수준으로 유지한다.
