# 조사 범위와 기록 규칙

[위키 홈](../README.md)

## 출처 범위

2026-10-13 대화의 alphaXiv 검색과 answer_pdf_queries가 반환한 원문 페이지를 중심으로 정리했다. Methods와 관련 실험을 확인한 요약이며 모든 부록·코드를 완독하거나 각 실험을 독립 재현했다는 뜻은 아니다. 객관식 편향과 temperature scaling은 앞선 대화의 기초 참고 자료다. dQwen3.5와 ConEntail은 검색 중 확인한 주변 연구로 추가했다.

alphaXiv 검색 결과의 기관·인기도·자동 요약만으로 성능을 결론 내리지 않는다. 숫자를 기재한 논문 카드에는 표·절을 함께 적었다. PDF 본문은 저장하지 않고 원문 URL과 자체 요약만 보존했다.

## 검색 결과와 실제 확인을 구분

- 'Prompt-based Universal Classification' 검색은 LinkPrompt 공격 논문으로 잘못 매칭되어 관련 근거에서 제외했다.
- 'Zero-Shot Text Classification with Universal Discriminator' 검색은 ConEntail로 매칭되었다. 확인한 실제 ConEntail만 카드로 만들었고 Universal Discriminator를 읽었다고 주장하지 않는다.
- NV-Embed·GLiNER·Deep Sets 등은 관련 후보로 언급됐지만 이 위키에는 상세 본문 조사를 완료한 카드로 추가하지 않았다.
- 모든 검색 hit를 목록에 넣기보다 실제 읽고 설명한 근거를 중심으로 남겼다.

## 주장 표기

- **논문 방식/결과:** 출처가 실제 구현·실험한 부분.
- **공개 구현:** model card·소스코드가 보여주는 동작. 학습 recipe 전체 공개와 다름.
- **우리 제안:** 기존 요소를 Qwen3.5-0.8B에 조합하는 미구현 가설.
- **우리 실험:** v1/v2 보고서의 데이터·설정·결과. 논문과 같은 조건이라는 뜻이 아님.

새로운 structure는 새 scorer, information flow, encoder/candidate decomposition, Backbone primitive의 변경 수준으로 나누어 기록한다. '새 구조가 없다'와 '완전히 새로운 Backbone은 이번 조사에서 찾지 못했다'는 다르다.

## 재사용과 업데이트

상대 Markdown 링크를 사용하여 GitHub·일반 Markdown viewer에서 탐색할 수 있다. sources.json은 문헌 색인이며 benchmark prediction 또는 paper 본문의 복제 파일이 아니다. 후속 논문은 원문 ID와 실제 반환 title을 대조하고 주요 절·한계를 확인한 뒤 추가한다.


## 2026-10-17 추가 조사 (망각·편향·데이터)

이날은 웹 검색과 외부 검토(alphaXiv 대조)로 학습 방식 문헌을 봤다. 카드로 만든 것은 초록이나 공식 페이지에서 제목·저자·핵심 결과를 직접 확인한 논문뿐이다.

- **카드 작성(확인함):** [WiSE-FT](../papers/wise-ft.md), [L2-SP](../papers/l2-sp.md), [LoRA Learns Less and Forgets Less](../papers/lora-learns-less.md), [LP-FT](../papers/lp-ft.md), [One Bias Score, Two Factors](../papers/option-bias-two-factors.md).
- **초록만 확인, 카드 없음:**
  - PA-GRPO(arXiv 2603.21016, 순열 일관성을 보상하는 GRPO).
  - Punching Above Their Weight(arXiv 2607.03801): Qwen3 0.6B~8B에서 LoRA 분류 head가 라벨 생성보다 2~3% 높다. 후보마다 입력을 따로 처리하는 구조라 우리 255행 readout과 다르다.
  - Calibrate Before Use(Zhao et al. 2021, contextual calibration).
  - MoFO(arXiv 2407.20999, 모멘텀 큰 파라미터만 업데이트).
- **외부 검토가 인용, 우리가 본문·초록 미확인:** FINCH(2605.20005), 공개 텍스트 재노출 연구(2512.22337), 문항별 망각 측정(2510.17776), FastMix(2606.14971), Post-Training Science(2609.01244), 불완전 학습 진단(2604.10079). 이 위키에서 이 논문들의 수치를 근거로 쓰지 않는다.
- **정정 기록:**
  - ProLong을 장문 붕괴 진단의 근거로 쓴 것은 잘못이었다(장문 continued training 뒤의 짧은 SFT를 다룬 연구).
  - soft label이 망각을 줄인다는 주장은 과장이었다.
  - d3-lite가 임베딩을 동결했다는 추정은 가중치 감사로 틀린 것이 확인됐다.

## 2026-10-18 추가 조사 (병합·일반화·평가셋 크기)

- 카드를 썼다: [Model Stock](../papers/model-stock.md), [LiNeS](../papers/lines.md). 둘 다 arXiv 본문에서 식을 확인했다.
- 검색 요약으로 확인하고 카드를 썼다: [체크포인트 평균](../papers/checkpoint-averaging.md), [In-the-Wild 병합](../papers/merging-in-the-wild.md), [Dataset Cartography](../papers/dataset-cartography.md), [검정력](../papers/power-analysis.md). 수치는 요약에 나온 것만 옮겼다.
- 전체 목록과 확인 수준은 [참고 문헌과 데이터](references.md)에 있다.
