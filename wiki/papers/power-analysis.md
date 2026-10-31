# 평가셋 크기는 검정력으로 정한다

[위키 홈](../README.md) · [평가](../designs/evaluation.md)

- 논문: **With Little Power Comes Great Responsibility** (Card, Henderson, Khandelwal, Jia, Mahowald, Jurafsky)
- 연도: 2020 (EMNLP) · 주제: NLP 실험의 통계적 검정력
- 원문: [PDF](https://web.stanford.edu/~jurafsky/pubs/2020.emnlp-main.745.pdf)
- 조사 기록: 2026-10-18 웹 검색 요약 기준. 함께 참고한 것은 Ng, *Machine Learning Yearning*의 "dev 1천~1만 개가 흔하다, 감지하려는 차이로 크기를 정한다"(2차 요약).

## 확인한 실험 근거

- 여러 NLP 비교 실험이 검정력이 부족했다. 많은 GLUE 비교가 작은 test 때문에 그렇다.
- 기계번역에서 약 2,000문장이면 BLEU 1점 차이를 약 75% 검정력으로 감지한다.

## 우리 Qwen에 적용한 결과 (우리 실험)

`mix.hold_out` dev 크기를 출처당 min(300, 10%)로 정했다(2026-10-18).
- 같은 문항으로 체크포인트를 짝지어 비교하면 출처 정확도 약 5점 변화를 검정력 80%로 감지한다(불일치 문항 약 12% 가정).
- 출처 평균(매크로)은 약 0.7점을 감지한다.
- 출처별 판단에 150문항(±3.1점)은 부족했다.
- **2026-10-26 점검:** 위 계산은 체크포인트 한 쌍의 문항 짝 비교다. seed 변동을 넣으면 정답률 + seed 3개로는 짧은 문항 약 2.5점, 장문 약 7점 미만을 못 잡았다. 그래서 주 지표를 정답 NLL + seed 5개로 바꿨다([평가 점검](../project/evaluation-validity.md)).

## 연결 문서

- [평가](../designs/evaluation.md) · [작업 기록](../project/v3-log.md)
