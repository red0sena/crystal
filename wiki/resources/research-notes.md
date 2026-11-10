# 조사 범위와 기록 규칙

[위키 홈](../README.md)

## 출처 범위

2026-10-13 대화의 alphaXiv 검색과 answer_pdf_queries가 반환한 원문 페이지를 중심으로 정리했다. Methods와 관련 실험을 확인한 요약이며 모든 부록·코드를 완독하거나 각 실험을 독립 재현했다는 뜻은 아니다. 객관식 편향과 temperature scaling은 앞선 대화의 기초 참고 자료다. dQwen3.5와 ConEntail은 검색 중 확인한 주변 연구로 추가했다.

alphaXiv 검색 결과의 기관·인기도·자동 요약만으로 성능을 결론 내리지 않는다. 숫자를 기재한 논문 카드에는 표·절을 함께 적었다. PDF 본문은 저장하지 않고 원문 URL과 자체 요약만 보존했다.

## 검색 결과와 실제 확인을 구분

- **2026-10-19 구조 전환/데이터 후속:** 공개 BDH-GPU source를 commit2b0d7a4에 고정해 실제 상위층 교체를 구현했다. BDH-CQ는 내부 update 비공개라는 원문§3.3의 제한을 기록했다. LoLCATs·Mamba in the Llama의 정렬/증류 단계도 원문과 공식 자료에서 대조했다. [적용과 결과](../project/bdh-memory.md): native 대비 −0.60%p로 채택하지 않았다.

- **2026-10-19 추가:** TRM·T²MLR·LLoCoT·INTACT·Latent Recurrent Thoughts·BDH-CQ·Pause Tokens·RiM은 alphaXiv 추출 원문의 설계·실험·학습 조건에서 해당 부분을 확인했다. [문헌 카드](../papers/latent-workspaces.md), [실제 workspace 구현·선별 실험](../project/hypothesis-workspace.md). 이전 residual routing·head·동일 블록 반복과 구분하며 어느 원논문의 직접 재현으로도 주장하지 않는다.

- 'Prompt-based Universal Classification' 검색은 LinkPrompt 공격 논문으로 잘못 매칭되어 관련 근거에서 제외했다.
- 'Zero-Shot Text Classification with Universal Discriminator' 검색은 ConEntail로 매칭되었다. 확인한 실제 ConEntail만 카드로 만들었고 Universal Discriminator를 읽었다고 주장하지 않는다.
- NV-Embed·GLiNER·Deep Sets 등은 관련 후보로 언급됐지만 이 위키에는 상세 본문 조사를 완료한 카드로 추가하지 않았다.
- 모든 검색 hit를 목록에 넣기보다 실제 읽고 설명한 근거를 중심으로 남겼다.

## 주장 표기

- **논문 방식/결과:** 출처가 실제 구현·실험한 부분.
- **공개 구현:** model card·소스코드가 보여주는 동작. 학습 recipe 전체 공개와 다름.
- **우리 제안:** 기존 요소를 Qwen3.5-0.8B에 조합하는 미구현 가설.
- **우리 실험:** v1/v2 보고서와 v3 실험 페이지(`project/`)의 데이터·설정·결과. 논문과 같은 조건이라는 뜻이 아님.

새로운 structure는 새 scorer, information flow, encoder/candidate decomposition, Backbone primitive의 변경 수준으로 나누어 기록한다. '새 구조가 없다'와 '완전히 새로운 Backbone은 이번 조사에서 찾지 못했다'는 다르다.

## 재사용과 업데이트

상대 Markdown 링크를 사용하여 GitHub·일반 Markdown viewer에서 탐색할 수 있다. sources.json은 문헌 색인이며 benchmark prediction 또는 paper 본문의 복제 파일이 아니다. 후속 논문은 원문 ID와 실제 반환 title을 대조하고 주요 절·한계를 확인한 뒤 추가한다.

## 2026-11-04: PRH·JEPA와 고정 Qwen 회전 정렬

- [논문 카드](../papers/platonic-jepa.md): arXiv:2610.39302v1의 본문과 부록 A–C에서 정리·가정·증명 구조·설정·한계를 대조했다.
  공식 구현은 `a2b9949fa907d4fbe562742d4b1ffa7205ab7427`에 고정해 정적 검토했다.
  저자 학습의 독립 재현·공식 테스트 전체 실행·형식 증명 검증을 했다는 뜻은 아니다.
- [우리 실험](../project/rotation-alignment.md): 같은 VLM의 마지막 판정 표현에 적합한 직교 사상을 적용한 합성 pilot다.
  독립 인코더 동치·JEPA 정리 재현·무학습·pair-free·자연 이미지 전이로 부르지 않는다.
- [사후 감사](../project/rotation-alignment-audit.json): 원 기록 SHA256,672문항 gold·이미지 중복,
  모든 test logits의 독립 재계산, 층화 bootstrap, 평균 방향 대조, 중심화·회전 불변 지표를 확인했다.
  추가 대조와 exact McNemar 검정은 원 test를 본 뒤의 분석이며 새 확인 실험이 아니다.
- 정정한 해석: 높은 대응 cosine과 내부 기하 정렬은 다른 지표다. 평균·공통 방향이 원인이라는 설명도
  처음에는 가설이었으므로 사후 대조의 실제 관측과 구분했다. 마지막 표현의 회전은 효과적인 출력 head 변경과 같다.
- 원 run과 원 보고서는 보존하고 사후 감사 JSON을 별도로 뒀다. 원문 전체나 외부 코드는 위키에 복제하지 않았다.
- **수정 설계 실제 실행:** [질문 없는 상태 회전 v2](../project/state-rotation-v2.md).
  세 물체1,536장, train 전용 평균/PCA64/직교 사상, text-only 조건부 MLP3seed를 사용했다.
  원래 code head를 회전하는 실험과 분리했다. 새 장면·seen 질문에서 일부 전이를 관찰했고,
  의미 조합을 뺀 unseen 질문은 text 기준선부터 실패했다. 이 실패를 회전 자체의 실패로 귀속하지 않는다.
- **같은 데이터의 학습법 비교:** [Crystal full FT 후속](../project/crystal-state-methods.md).
  원본에서 출발해 같은8,922개 텍스트 seen 문항을 한 번씩, causal·Crystal 각3seed로 학습했다.
  모든 학습 입력 해시를 맞추고 causal 공통 평가와 정상 noncausal 평가를 모두 보존했다.
  정상 noncausal의 검색R@1은51.95%→68.88%, 회전 판정 전이는48.81%→45.03%(정확도 차이 구간0포함)였다.
  native 이미지 seen 정확도는올랐지만 NLL은 나빠졌다. 검색·별도모듈 전이·native 판정을 혼동하지 않는다.
- **실제 상태 경로의 표현 학습:** [CE+LeJEPA 비교](../project/jepa-decision.md).
  고정 Qwen 토큰→학습하는 공유64D상태→동일 판정기에서 CE/bad/good를 각3seed·200epoch로 비교했다.
  good의 CKA는 CE 대비.229→.407, 회전 이미지 seen은46.36%→52.51%이나 차이 구간은0을 포함했다.
  bad 대비는+9.74%p [+6.64,+12.82]였다. CE 대비 이미지 NLL은 좋아졌지만 text seen과 unseen 능력은 안정적으로 보존하지 못했다.
  별도 수식으로691,200건을 재계산해 장면·과제별 집계를 대조했고, 상태·사상·기하·bootstrap 구간도 검산했다.
  공유 adapter·구조화된 질문·canonical text의 정규화 차이를 명시했다. 독립 JEPA 인코더 정리의 재현이 아니다.


## 2026-10-17 추가 조사 (망각·편향·데이터)

이날은 웹 검색과 외부 검토(alphaXiv 대조)로 학습 방식 문헌을 봤다. 카드로 만든 것은 초록이나 공식 페이지에서 제목·저자·핵심 결과를 직접 확인한 논문뿐이다.

- **카드 작성(확인함):** [WiSE-FT](../papers/wise-ft.md), [L2-SP](../papers/l2-sp.md), [LoRA Learns Less and Forgets Less](../papers/lora-learns-less.md), [LP-FT](../papers/lp-ft.md), [One Bias Score, Two Factors](../papers/option-bias-two-factors.md).
- **초록만 확인, 카드 없음:**
  - PA-GRPO(arXiv 2603.21016, 순열 일관성을 보상하는 GRPO).
  - Punching Above Their Weight(arXiv 2607.03801): Qwen3 0.6B~8B에서 LoRA 분류 head가 라벨 생성보다 2~3% 높다. 후보마다 입력을 따로 처리하는 구조라 우리 255행 readout과 다르다.
  - Calibrate Before Use(Zhao et al. 2021, contextual calibration). 10-19 [다른 방법](../project/other-methods.md)에서 원문 §5를 확인했다.
  - MoFO(arXiv 2407.20999, 모멘텀 큰 파라미터만 업데이트).
- **외부 검토가 인용, 우리가 본문·초록 미확인:** FINCH(2605.20005), 공개 텍스트 재노출 연구(2512.22337), 문항별 망각 측정(2510.17776), FastMix(2606.14971), Post-Training Science(2609.01244), 불완전 학습 진단(2604.10079). 이 위키에서 이 논문들의 수치를 근거로 쓰지 않는다.
- **정정 기록:**
  - ProLong을 장문 붕괴 진단의 근거로 쓴 것은 잘못이었다(장문 continued training 뒤의 짧은 SFT를 다룬 연구).
  - soft label이 망각을 줄인다는 주장은 과장이었다.
  - d3-lite가 임베딩을 동결했다는 추정은 가중치 감사로 틀린 것이 확인됐다.

## 2026-10-18 추가 조사 (병합·일반화·평가셋 크기)

- 카드를 썼다: [Model Stock](../papers/model-stock.md), [LiNeS](../papers/lines.md). 둘 다 arXiv 본문에서 식을 확인했다.
- 검색 요약으로 확인하고 카드를 썼다: [체크포인트 평균](../papers/checkpoint-averaging.md), [In-the-Wild 병합](../papers/merging-in-the-wild.md), [Dataset Cartography](../papers/dataset-cartography.md), [검정력](../papers/power-analysis.md). 수치는 요약에 나온 것만 옮겼다.
- 10-18까지의 목록과 확인 수준은 [참고 문헌과 데이터](references.md)에 있다. 10-19 이후 문헌은 [구조 검색](../project/architecture-search.md), [다른 방법](../project/other-methods.md), [조합 실험](../project/robust-combinations.md), [평가 점검](../project/evaluation-validity.md), [학습 레시피 묶음](../papers/decision-training-recipes.md), [loss 문헌 검토](../papers/loss-alternatives.md)에 있다.
