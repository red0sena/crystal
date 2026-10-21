# LiNeS: 층 깊이에 따라 미세조정 변화를 다르게 남기기

[위키 홈](../README.md) · [작업 기록](../project/v3-log.md)

- 논문: **LiNeS: Post-training Layer Scaling Prevents Forgetting and Enhances Model Merging** (Ke Wang, Nikolaos Dimitriadis 외)
- 연도: 2025 (ICLR) · 주제: 층별 스케일링 / 망각 완화 / 병합
- 원문: [arXiv](https://arxiv.org/abs/2410.17146)
- 조사 기록: 2026-10-18 arXiv HTML 본문에서 식(Eq. 1)을 확인했다.

## 논문에서 제안한 방식

미세조정 변화량(task vector)의 ℓ번째 층 블록에 λ(ℓ) = α + β·(ℓ−1)/(L−1)을 곱한다. 앞층(일반 특징)은 원본 가까이 두고 뒷층(과제 특징)은 남긴다. 블록 밖 파라미터에는 α를 쓴다(부록 의사코드 주석).

## 확인한 실험 근거

- 망각 실험은 CLIP 비전 모델이다.
- LLM 실험은 LoRA로 미세조정한 LLaMA-2 7B의 Rewarded Soups 병합(Reddit 요약)이다. 저자들은 선호 공간 전체에서 기존 방식보다 낫다고 보고한다.

## 우리 Qwen에 적용한 결과 (우리 실험)

`merge.py --alpha --beta`로 구현했다. 우리 임베딩은 출력 코드 행과 공유돼 있어 α를 받으면 출력 쪽도 가장 많이 되돌아간다(`--outside`로 따로 지정 가능).
- 정방향(앞층 0.5 → 뒷층 1.0)은 대리셋에 유리했다. g3 e2: JEVal 0.423 / 공개 38.19. g4b: 0.445 / 36.36.
- 역방향(1.0 → 0.5, d3-lite의 이동 양상과 같음)은 공개 점수에 유리했다. g3 e2: 공개 40.39.
- g7 시기(2026-10-21), g6 + g7e2 1:1에 0.8 → 1.0: JEVal same 0.580(22개 후보 중 공동 최고), new 0.410, 공개 44.36, Full 추정 +1.05. 모든 층에 원본 20%를 섞은 후보(+1.49)보다 new가 낮다. 층별 비율과 원본 비율을 함께 바꾼 격자(AdaMerging의 싼 근사)는 아직 돌리지 않았다.

## 연결 문서

- [Model Stock](model-stock.md) · [WiSE-FT](wise-ft.md) · [LP-FT](lp-ft.md)
