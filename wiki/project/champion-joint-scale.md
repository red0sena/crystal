# 두 모달리티 공동 학습 확대

목표는 Jev Decision Index에서 1B 이하 텍스트·비전 모두 1위다. 이 실행은 원본 Qwen에서 시작하는 단일 후보이며 공식 비공개 Full 결과가 아니다.

## 기존 파일럿에서 확인한 부족한 부분

|텍스트 과제|지표|우리|선두|공개 지수 격차 기여|
|---|---|---:|---:|---:|
|PhishNChips phishing decisions|accuracy|0.5135|0.9340|2.55|
|HoVer claim verification|accuracy|0.5292|0.8380|1.87|
|API-Bank|accuracy|0.4173|0.8287|1.70|
|VAST|macro-F1|0.3717|0.7305|1.48|
|HellaSwag|accuracy|0.5147|0.8280|1.38|
|BPoMP|accuracy|0.5036|0.9170|1.36|
|BANKING77|macro-F1|0.5511|0.9080|1.32|
|WinoGrande|accuracy|0.5643|0.8019|1.23|
|CLINC150+OOS|macro-F1|0.6181|0.9229|1.12|
|iSarcasmEval|Sarcasm F1 · track A, English|0.0446|0.5109|1.02|
|ContractNLI|macro-F1|0.5335|0.7896|1.02|
|ANLI|macro-F1|0.4112|0.6171|1.02|

피싱은 같은 이메일의 noul 정확도68.45%와 주 선택지 정확도51.35%가 크게 달랐다. 주 선택지에서는2,000건 중1,971건을 정상으로 골랐다. HoVer는4,000건 중3,741건을 supported로, VAST는3,006건 중103건만 neutral로 골랐다. 과제 전체 격차와 함께 표현 형식·클래스별 예측 쏠림을 개선 대상으로 삼는다.

## 고정한 다음 후보

원본 공식 Qwen에서 텍스트131,072 + 이미지32,768개를2epoch 학습한다. 매 update는 텍스트16·이미지4개이고 총16,384 update다. 기존 CE·all LM LoRA r16/alpha32/dropout.05·ViT 동결·FP32-master AdamW를 유지한다. LR2e-4, warmup512, cosine to10%다.

g7의113개 출처를 기반으로 약한 과제군에 표본을 더 배분하고 가능한 클래스·질문 polarity를 균형화한다. 기존 문맥을 질문으로 이동하거나 의미를 유지한 선택지 설명에 임의 키를 붙인다. 정답·참/거짓 방향·기존 선택지 내용은 바꾸지 않는다. 이 보강의 목표는 표현 형식 이동이며 회전 정렬을 학습 손실로 추가하지 않는다.

이미지는 기존12개 Cauldron train 출처에 기존 변환기가 지원하는 CORD train·FUNSD training_data를 추가한다. 공식 전체 이미지와 기존 진단 이미지에 SHA256 및 dHash 거리4 중복 검사를 적용하고 dev 이미지와도 분리한다. 이 근사 필터가 모든 크롭·편집 중복의 부재를 증명하지는 않는다.

기존 공개 평가를 마친 뒤 같은 GPU에서 사전 점검→불변 계획→학습→저장/복원 검증→텍스트37개·비전11개 공개 재평가를 순서대로 실행한다. 고정 최종 update를 사용하며 공개 점수로 중간 checkpoint를 고르지 않는다. 입력·가중치·코드 해시와 재개용 optimizer/master/RNG를 보존한다.

## 상태와 결과

서비스 단계: `complete`.
새 후보 텍스트 공개 지수: **35.19**.
새 후보 비전 공개 지수: **48.20**.

개별 변경의 효과를 분리하는 대조 실험이 아닌 묶음 개선 후보다. 공개 평가를 관찰해 우선순위를 정했으므로 blind 확인 결과로 부르지 않는다. 선두 가중치·예측은 학습 초기값이나 교사로 사용하지 않는다. 새 후보의 텍스트·비전을 모두 재고, 공식 Full 순위는 최종 비공개 평가 후에만 판정한다.

실행 원본: `runs/champion-v1/joint-g7-v2/`. 서비스: `champion-scale-v2`. 원본 비교: [공식 공개 평가](champion-public-evaluation.md).
