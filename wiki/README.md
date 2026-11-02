# Decision Model 연구 위키

생성 없이 가변 선택지의 점수·확률을 반환하는 모델을 만들기 위한 논문과 설계 기록이다. 우리 Qwen3.5-0.8B 실험과 연결해 읽을 수 있다.

**조사 기준일: 2026-10-13(망각·편향 문헌은 2026-10-17, 학습 레시피·loss 문헌은 2026-10-26~30 추가). v3 기록은 2026-10-16부터, 마지막 갱신은 2026-11-02.** alphaXiv에서 논문을 검색하고 PDF 본문을 확인한 내용을 정리했다. 2026년 연구는 공개 프리프린트 결과로 취급하며 학회 심사·독립 재현을 확인했다고 주장하지 않는다. 자료에 등장하는 정확도·속도는 해당 논문 조건의 결과이며 우리 모델 성능이 아니다.

## v3: Jev Decision Index 도전 (2026-10-16~)

목표는 우리 Qwen3.5-0.8B로 [Jev Decision Index](https://huggingface.co/spaces/multimodalart/jev-decision-index)에서 `vllm-sr/d3-lite`(0.8B, Full 32.33)를 넘는 것이다. **갱신(11-02):** 보드 0.4(10-19)부터 Full은 비공개 same 60% + new 40%이고, 공개 점수는 빠졌다. 지금 d3-lite Full은 26.16(1B 이하 8위)이고, 1등은 GLiClass v4 edge(DeBERTa encoder, 31.21)다([목표와 규칙](project/jev-index.md)). **우리 wise80은 `crystal-0.8b`로 올라가 Full 29.37(1B 이하 3위)로 d3-lite를 넘었다.** GLiClass와의 차이는 거의 PhishNChips, WinoGrande, iSarcasm 세 셋에서 나온다.

**지금까지의 성과(2026-11-02)**

| 항목 | 우리 | d3-lite | 비고 |
|---|---:|---:|---|
| 보드 Full(비공개 same 60% + new 40%) | **29.37** (`crystal-0.8b` = wise80, 1B 이하 3위) | 26.16 | 1위 GLiClass v4 edge 31.21 |
| 보드 공개 vision(10개, 키트 runner) | 49.05(텍스트만) → **49.9~51.0**(이미지 LoRA, 겹침 뺌) | 47.19 | 이미지 LoRA 대비 d3-lite +2.7~3.8, 구간이 0을 벗어난다 |
| 처음 보는 vision 벤치마크 17개 | 50.91(텍스트만) → 52.4 | 51.86 | d3-lite와 같다(구간이 0을 포함한다) |

- **텍스트(목표 지표): d3-lite를 넘었다.** g6과 g7 2에폭을 섞고 원본 쪽으로 당긴 wise80이 보드에 `crystal-0.8b`로 올라가 Full 29.37이 됐다. GLiClass와의 차이는 거의 PhishNChips, WinoGrande, iSarcasm에서 나온다([목표와 규칙](project/jev-index.md)).
- **이미지: wise80에 ViT를 얼린 LoRA를 얹었다(37번).**
  - 겹침을 빼기 전의 58.15는 보드 셋의 train split과 원본에서 온 점수라 성과로 쓰지 않는다.
  - 평가와 겹치는 출처를 뺀 학습 세 번이 보드 49.9~51.0, 시험셋 51.9~52.4였다. wise80 대비로는 보드 +0.8~+1.9, 시험셋 +1.0~+1.5다.
  - 개수·웹 보강과 분야 21개 추가를 해 봤지만, 처음 보는 벤치마크 전체는 오르지 않았다. 텍스트 dev는 분야를 넓힌 학습에서 0.848로 거의 지켰다.
- **배운 것**([인사이트](project/insights.md)):
  - 학습한 분포에서는 이미지 dev가 +7.4점인데, 처음 보는 분포에서는 정답률이 +0.5점이다. 시험셋의 변화는 seed 잡음과 크기가 비슷하다.
  - 텍스트 학습이 이미지 시험셋을 +10점 올린 것은 고르는 형식 덕이다. 0.8B에서는 텍스트든 이미지든 과제별 몫이 새 분야로 잘 안 옮겨 간다. 보드의 0.8B 미세조정 19개에서 same과 new의 상관은 0.09다.
  - 구조 변경은 모두 원본을 넘지 못했다. 가장 좋은 레시피는 양방향 + MLP 포함 LoRA r16 + CE다(27~32번).
- **남은 일:** 이미지 LoRA 모델을 vision 트랙 후보로 둘지 정하기, PhishNChips 음성 클래스 붕괴 분석(GLiClass와의 차이), seed 반복으로 이미지 학습 차이 확인.

0. **[쉬운 정리: 지금까지와 앞으로](project/v3-easy-guide.md)** (2026-10-25 작성, 10-31 갱신): 한 일, 안 된 이유, 앞으로 할 수 있는 것(추천 순서 포함)을 쉬운 말로 정리했다. 처음엔 이것부터 읽는다.
   - **[지금까지의 인사이트](project/insights.md)** (2026-10-31): 구조, 학습 방식, 평가에서 배운 것과 열린 질문을 한 페이지로 모았다.
   - [v3 전체 요약](project/v3-summary.md): 방침, 데이터 버전, 학습 실험(g1·g3·g4b), 가중치 섞기, 진단, 현재 최선 후보를 수치와 함께 정리한 상세판.
1. [목표와 리더보드 규칙](project/jev-index.md): 점수 산식, d3-lite 구조와 약점, 0.8B 경쟁 모델, 제출·오염 규칙.
2. [작업 기록](project/v3-log.md): 시간순 상세 기록이다. 측정값만 적는다. 주요 내용은 다음과 같다.
   - 준비: suite 재구성과 d3-lite 재현, JEVal 대리셋, 오염 필터, 학습 속도 개선.
   - 학습 방식과 g1: 정답 전용 학습(교사 금지, 도구만 예외), 정답 기준선 g1(공개 샘플 33.68 대 d3-lite 39.65).
   - 분석: 원본과 가중치 섞기, d3-lite 가중치 감사, 순서 편향·문항별 망각 측정.
   - 데이터: 내부 dev 누수 수정, 3·4차 데이터(사람 정답, 라이선스 확인), dev 분할 규칙 개정(검정력 기준).
   - 학습: g3(3에폭), g4b(새 데이터 40만, 패딩 없음), fresh-v1 평가셋.
   - 섞기: g4b/g3 이력에 이어 완료된 g6/g3 67:33 조합을 로컬 재구성·평가했다(아래 11번).
   - 원래 Qwen에서 출발한 비교(10-25~31): 구조 비교, 학습 방식 거르기, LoRA 위치, loss 비교(아래 27~32번).
3. [전략](designs/beat-d3-lite.md): 수단의 우선순위와 상태, 지금까지 알게 된 것, 다음 단계(10-31 갱신).
4. [학습 데이터 출처](designs/phase2-data-sources.md): 108개 출처 194만 문항(v3-gold6이 104개 사용). 정답 출처(사람/프로그램/모델), 학습 셋 사용량, 라이선스, 오염 함정을 정리했다.
   - [이미지 학습 데이터 출처 후보](designs/vision-data-sources.md) (10-31, 조사만): 보드 vision 능력 10개별 후보 약 40개(Cauldron, GQA, SAT, ImageCoDe 등), 정답·라이선스·중복 위험, 권장 규모(풀 30만~35만, 거르기 1.6만, 본 학습 6.5만).
5. [참고 문헌과 데이터](resources/references.md): v3에서 참고한 논문과 데이터셋, 확인 수준(카드/초록/검색/인용만)과 쓰인 곳.
6. [Pointer head 소규모 비교](project/pointer-pilot.md): v3 backbone을 유지하는 코드 head/Pointer 학습·평가 실험과 실행 방법.
7. [논문 기반 구조 검색](project/architecture-search.md): 99개 설정·seed 조합, 별도 confirmation·JEVal 검증. 이번 범위에서 개선은 확인되지 않았다.
8. [다른 방법 조사와 입력 실험](project/other-methods.md): 추가 14편 검토. 입력 반복·정의·prefix·prior 보정 실험과 반사실 쌍 학습·R-Drop·반복 깊이의 적용 조건.
9. [LM Head 대체 readout·MLP](project/readout-mlp.md): 6종·24개 설정/seed, 마지막 두 층 추가 학습, 별도 2,027문항 확인. 이번 조건에서는 개선 없음.
10. [원본 보존 학습·추론 조합](project/robust-combinations.md): LoRA·순열 일관성 4개 run, 가중치·cyclic 평균, 후보 재비교, gold memory. 별도 2,998문항에서 이번 소규모 추가 방법의 개선은 확인되지 않았다.
11. **[g6/g3 조합 모델 재구성·새 문항 확인](project/g6-soup-replication.md)**: g6 67% + g3 33%를 직접 만들었다. 기존 fresh 재측정에 이어, 학습·이전 평가 지문을 제외한 새 1,847문항·10출처에서도 **g4b 65.39% → 71.97%(+6.58%p)**였다. 추가 추론 없는 단일 checkpoint, 기존 엔진 1,335문항 재로딩 불일치 0개.
12. [Transformer 내부 블록 반복](project/recurrent-depth.md): 24→28블록 실행, 가중치 공유·게이트, 두 구조 모두 전체 언어 백본을 512 update씩 학습했다. 새 8,823문항에서 같은 추가 학습의 기본 구조 75.20%, 반복 구조 75.29%. 차이 +0.10%p의 95% 구간이 0을 포함했고, 추론은 15.27% 느려 개선으로 채택하지 않았다.
13. [후보별 판정 Query와 마지막 층 연산 재배치](project/decision-native-next.md): 공통 23층 뒤 마지막 층의 Q·출력·FFN을 후보 Query에만 적용. 실제 512스텝 통제 학습 후 새 3,394문항에서 기본 70.70%, 후보 70.75%, 차이 +0.046%p·95% 구간 [−0.328,+0.412]%p로 정확도 개선은 확인하지 못했다. 저장/복원·기존 엔진 연결 포함.
14. **폐기한 후보 —** [중간 블록 delta 재사용 실험](project/v3-log.md): native24층 안에 변화량 depth routing을 추가해 실제512스텝 학습. 새2,220문항에서65.70%→65.85%, 차이+0.154%p·95% 구간[−0.282,+0.576]%p로 개선은 확인하지 못했다. 실제 엔진 호출 약5.4% 증가. 사용자 요청으로 채택 후보에서 제외하고 증거는 보존했다. [수치·모델/엔진 검증](project/delta-memory-results.json).
15. [중간 층의 가설 작업공간과 판정 출구 변경](project/hypothesis-workspace.md): 앞16층 뒤 선택지별 상태를 추가해 남은8층에서 원문·가설을 공동 갱신했다. 첫 버전512스텝은 기본75.7419%, 새75.7470%(+0.0050%p). 가설에서만 판정하는 버전은 같은256스텝에서73.1286% 대 기본73.9417%(−0.8131%p)로 사전 규칙에 따라 중단했다. 두 버전 모두 채택하지 않았고 새 확인셋을 열지 않았다. 관련63개 테스트·전체선택4,147문항 저장 복원을 검증했다. [TRM·T²MLR·RiM·LLoCoT·BDH-CQ 등 8편 대조](papers/latent-workspaces.md).
16. [상위8층을 BDH 메모리로 교체·고유 데이터4배 비교](project/bdh-memory.md): 앞16층 뒤 연상 메모리·coactivation을3회 갱신하는 코어로 뒤8층을 대체했다. 세 군 모두1,024스텝 실제 학습. 기존 선택셋4,147문항에서 같은16,384행을 학습한 원본75.5504%, BDH74.9540%(−0.5964%p)로 정확도 개선은 없었다. BDH에서4,096행4회→16,384행1회는73.2656%→74.9540%(+1.6884%p)여서 이 조건의 데이터 다양성 효과는 보였다. 새로운 확인셋은 열지 않았다. 공개 BDH의 별도 Qwen retrofit이며 비공개 BDH-CQ 재현이 아니다.
17. [학습 예산 확대](project/long-training.md): 원본·BDH·내부 블록 반복·가설 출구4종을 같은65,536행과4,096스텝으로 다시 비교한다. 이전 BDH의4배, 다른 후보의8~16배 update다. 모든 입력 검사·관련69개 테스트·네 구조의 실제 FP32 optimizer 저장/재개 검증을 통과하고 설정을 고정해 실행했다. 짧은 학습 실패를 구조 자체의 한계로 단정하지 않는다. **완료(2026-10-20):** 원본 79.14%, BDH 78.16%, 블록 반복 79.05%, 가설 출구 78.45%로 원본을 넘은 구조가 없다.
18. Gated DeltaNet 역방향 scan([수치](project/reverse-scan-results.json)): 4,096 update에서 −0.51%p(95% [−1.34, +0.19]). 개선 없음.
19. [MNTP 전환 학습](project/mntp-conversion.md): 2026-10-21 사용자 요청으로 중단, 비교 결과 없음.
20. [DUMA 공동 매칭·Theseus 점진 교체](project/dual-theseus.md): 세 변형 모두 원본 79.14% 미만.
21. 증거 쌍 손실([수치](project/evidence-pairs-results.json)): CONDAQA·VitaminC 짝 손실, 모든 효과 구간이 0 포함.
22. 단일 forward 의미 매칭([수치](project/single-forward-semantic-results.json)): semantic−native +0.05%p, 3 seed 계층 구간 0 포함.
23. [Talking-Heads](project/talking-heads.md): 마지막 두 full-attention 층 head 혼합, 3 seed에서 차이 없음.
24. [학습 레시피: EMA·구간 평균·학습률](project/recipe-ema-lr.md): EMA −0.54%p, 구간 평균 −0.56%p, LR 1e-5 −3.76%p. 개선 없음.
25. **[장문 테스트셋과 질문 우선 배치(echo)](project/long-context-echo.md)**: 공개 장문 5종을 평가 전용으로 만들고 25% 장문 풀로 native/echo를 3 seed 비교했다. 장문 점수 +0.61%p, seed×문항 95% [−0.84, +2.07]로 구조 효과 입증 실패. 찾기는 이미 천장이고 실패는 추론·다예시 분류에 있으며, seed 잡음이 효과보다 크다는 원인 분석을 함께 기록했다.
26. [선택지 순서 측정](project/order-probe.md): 학습 없이 4순서 평균. selection +0.68%p(답 바뀜 10%), JEVal 장문 −1.25%p(22%), BABILong 0(19%). 선택지 격리가 줄 몫은 짧은 문항 +0.7%p 안팎으로 추정(앙상블 효과 포함).
27. **[원래 Qwen에서 출발하는 구조 비교](project/from-base-structures.md)** (완료 10-26): 원본(causal)·양방향·결정 지점 반복 읽기(hop)를 학습 전 가중치에서 같은 104개 출처 65,536행으로 각각 full fine-tuning해 3 seed로 비교했다. 사전 등록한 여섯 검정을 모두 통과하지 못했다. 양방향−원본은 selection +0.51, 장문 −0.21이다. hop의 두 번째 읽기는 첫 읽기와 98~99% 같은 답을 골라 다시 읽기가 작동하지 않았다. hop−원본 selection은 세 seed 모두 양수(+1.19)였지만 구간 하한이 −0.18이었다.
28. **[우리 평가가 맞나: 구조 변경 논문들의 평가 방식과 비교](project/evaluation-validity.md)** (10-26): Gated Attention·Ettin·LLM2Vec·Pause tokens·Retrofitted recurrence·Coconut·Candidate-independent attention·LoLCATs와 Signal and Noise·DataDecide를 비교했다. 우리 평가는 엄격하지만 둔감했다. seed 3개와 정답률로는 짧은 문항 약 2.5점, 장문 약 7점 미만의 효과를 잡지 못하는데, 문헌상 구조 효과는 0.5~3점이다. 같은 9개 모델을 정답 NLL로 사후 재분석하면 hop−원본 짧은 문항과 양방향−원본 장문이 기준을 넘는다. hop의 이득은 다시 읽기가 아니라 학습 방식에서 나왔다.
29. **[학습 방식 거르기](project/training-methods.md)** (완료 10-28: DeltaNet 얼리기만 통과. 10-29에 MLP를 포함한 LoRA r16이 이를 넘었다): 원래 Qwen에서 출발해 다섯 레시피(causal, noncausal, DeltaNet 얼리기, CE+Brier, hop식 두 번 읽기 학습)를 fit당 16,384행 × seed 5로 비교했다. 주 지표는 정답 NLL이다. 근거 논문은 [판정 모델 학습 레시피 묶음](papers/decision-training-recipes.md)에 정리했다.
30. **[Attention-only LoRA](project/lora-attention.md)** (완료 10-28: 짧은 문항은 full FT와 같고, 장문은 +4~5점. DeltaNet 얼리기 대비 짧은 문항 −2.2, 장문 +2.6): 논문(arXiv 2604.22127) 설정 그대로 attention q/k/v/o에만 LoRA r16을 붙이고 나머지는 모두 얼렸다. 거르기와 같은 데이터·seed로 full FT(causal, noncausal, frozen)와 비교했다.
31. **[LoRA 위치 비교](project/lora-placement.md)** (완료 10-29: MLP를 포함한 LoRA r16이 DeltaNet 얼리기를 짧은 문항과 장문 둘 다 이김. attention만은 짧은 문항이 가장 약하고, attention rank 160은 장문이 full FT 수준으로 떨어짐): 양방향 구조에서 LoRA를 DeltaNet / MLP / attention+MLP / 전체에 붙이고, 전체와 파라미터 수를 맞춘 attention rank 160을 대조군으로 넣어 '위치 대 양'을 갈랐다. 학습 전 Qwen 점수도 함께 쟀다(짧은 문항 47.9%, 장문 49.1%).
32. **[Loss 비교](project/loss-screen.md)** (완료 10-31: 사전 등록 검정 16개 중 CE를 이긴 것이 없다. focal은 짧은 문항 정답률 −1.5점으로 나쁨. 두 순서 + 일관성 벌점(consist)은 보고 비교에서 짧은 문항 NLL −0.023(4/5 seed)이지만 계산량이 2배라 확인이 필요하다. MNTP는 효과 없음): 전체 LoRA r16(양방향)을 그대로 두고 loss만 바꿨다. label smoothing, focal, CE + Brier, 두 입력 CE, 순서 일관성(Jensen-Shannon), 가린 토큰 맞히기(MNTP) 보조 손실을 CE와 비교했다.
33. **[g7과 g6·g7 섞기: Hugging Face에만 있던 결과](project/hub-g7-merges.md)** (10-19~21 학습·평가, 10-31 기록): v3-gold7(100만 행, 113개 출처) 3 epoch 학습한 g7과 g6·g7 섞기 70여 개를 공개 샘플, JEVal dev, fresh에서 d3-lite와 같은 문항으로 채점한 결과다. wise80 g6+g7e2가 세 평가 모두 d3-lite와 같거나 높았다. 선택에 쓴 셋이라 확인이 필요하다.
34. **[공개 Vision 보드 기준점](project/vision-baseline.md)** (10-31, 이미지 학습 전): Winoground를 뺀 10개 벤치마크(11,410행)에서 원래 Qwen과 wise80을 쟀다. 텍스트로만 학습한 wise80이 49.05로 원래 Qwen 32.66보다 16.4점 높다(95% [+14.5, +18.2]). 비전 텐서는 원래 Qwen과 같다. 약 4점은 원래 Qwen의 글자 편향이 사라진 몫이다. 학습 효과와 양방향 attention 효과는 아직 가르지 못했다.
35. **[다른 벤치마크 vision 시험셋](project/vision-heldout.md)** (10-31): 학습 출처와 보드 셋에 없는 벤치마크 17개에서 4,772문항을 뽑았다. 이미지 학습 전 원래 Qwen 40.40, wise80 50.91(+10.51, 95% [+8.0, +12.6]). 이미지 학습 후보는 이것으로 고르고 보드 셋은 마지막에 쓴다.
36. **[이미지 LoRA 학습 속도](project/vision-train-speed.md)** (11-01): 이미지 학습 경로 `vision_train.py`를 새로 만들었다. 이미지 행 packing(M-RoPE), ViT varlen, 동결 ViT `no_grad`, 데이터 worker, 8k 토큰 microbatch(체크포인팅 없음)를 넣었다. transformers 기본 경로보다 3.4배 빠르다(step 4.66s → 1.30s, 3090). 기본 경로와 BF16 수준으로 일치하고, 행 사이 누출이 0이다. ViT LoRA는 1.7배 느리다(2.23s). 1.6만 행 거르기는 약 22분이다.
37. **[이미지 학습 셋과 첫 LoRA 학습](project/vision-train-screen.md)** (11-01~02): 받은 출처 58개를 보드와 같은 객관식으로 바꾸고(14.3만 문항, 평가 이미지와 겹치는 0.9% 제거), 분야 15개에 똑같이 나눠 텍스트를 1/3 섞었다. 16k 거르기에서 ViT 동결이 ViT LoRA를 이겼다(시험셋 +2.56). 65k 본 학습(ViT 동결)으로 보드 공개 vision 지수가 49.05 → 58.15(+9.1, 95% [+7.7, +10.5])로 올랐다. 보드 train split이 없는 6개만 봐도 +4.6이고 BLINK가 +0.14다. 다른 벤치마크 시험셋은 +1.1(구간이 0을 포함)이고, 텍스트 dev는 −0.020이다.
   - **d3-lite 실측(11-02):** 같은 행으로 재니 보드 공개 vision은 d3-lite 47.19(보드 공식 47.9와 맞음) 대 본 학습 58.15(+11.0, 95% [+9.3, +12.5])다. 다른 벤치마크 시험셋은 51.86 대 51.97로 같다.
   - **겹침을 뺀 재학습(11-02):** 평가 벤치마크의 train split과 원본 12개 출처를 빼고 다시 학습하니 보드 공개 vision이 50.12다. wise80보다 +1.07(95% [−0.20, +2.35]), d3-lite보다 +2.94다. 본 학습의 +9.1은 대부분 겹친 출처에서 왔다. 시험셋(51.89)과 텍스트 dev는 그대로다.
   - **개수·웹 페이지 보강(11-02):** 겹침 뺀 학습에 개수 크롭(1~12개)과 SeeClick 웹 페이지 출처를 넣었다. CountBench 5~7개(+13.1)와 VisualWebBench action_prediction(+13.0)은 돌아왔지만, 옆 문항(CountBench 8~10개, element_ground)이 내려 전체는 거의 그대로다. 보드 50.96(+0.83, 구간이 0을 포함), wise80 대비 +1.91 [+0.72, +3.20], 시험셋 52.44, 텍스트 dev 0.839다.
   - **이득이 작은 원인(11-02):** 학습한 분포의 dev는 +7.4점인데 시험셋 정답률은 +0.5점이다. 시험셋에서는 확신 없던 문항이 seed 잡음과 비슷한 크기로 흔들릴 뿐이다(얻은 301, 잃은 278). 확신하고 틀린 오답 30%는 거의 안 고쳐진다(6%, dev에서는 38%). 데이터를 4배로 늘려도 시험셋은 그대로였다.
   - **분야 넓히기(11-02):** 새 분야(질감, 표지판, 기종, 위성, 의료, 글자 읽기, 거리·3D, 위치)와 새 형식(여러 장 고르기, 바뀐 점, 개수 비교, 다음 화면) 출처 21개를 넣고 출처마다 같은 몫으로 섞었다. 시험셋 52.40(보강 대비 −0.03), 보드 49.89(−1.07, 구간 0 근처)로 전체는 그대로다. 텍스트 dev는 0.848로 돌아왔고, train split 없는 6개는 wise80 대비 +2.18 [+0.38, +4.00]이다. VQA-RAD는 MathVista 시험 이미지와 겹쳐 뺐다.
   - **텍스트와 비교(11-02):** 텍스트도 새 분야로는 잘 안 옮겨 간다. 보드의 0.8B 미세조정 19개에서 same-skills와 new-domains의 상관은 0.09다. 텍스트 학습이 이미지 시험셋을 +10.4 올린 것은 고르는 형식(보기 A 편향 제거 약 3점과 나머지)이라, 모든 문항에 쓰인다. 이미지 학습에는 과제별 몫만 남았다.

## 처음 읽을 문서

1. [현재 Qwen의 구조와 학습](project/current-qwen.md): 실제 구현한 것과 아직 없는 것을 구분.
2. [구조 변경 지도](concepts/architecture-map.md): LM Head 유지, 후보 내용 점수화, 집합 비교, Backbone 변경 비교.
3. [선택지 사이 Attention 차단](concepts/attention-isolation.md): 마스크, 위치 초기화, hybrid cache 분리.
4. [Reasoning·증류·확률 보정](concepts/reasoning-and-calibration.md): 서로 다른 목표와 손실.
5. [다음 Qwen 설계안](designs/qwen-next.md)과 [평가 계획](designs/evaluation.md): 문헌 기반 제안. 평가는 v3에서 일부 구현됐다.

## 목적별 읽기

| 목적 | 먼저 읽을 논문 |
|---|---|
| 후보 내용에서 직접 점수 얻기 | [GLiClass](papers/gliclass.md), [Soft Contextualized Encoder](papers/soft-contextualized-encoder.md) |
| 코드·표시 순서 편향 줄이기 | [Candidate-independent Attention](papers/candidate-independent-attention.md), [객관식 편향](papers/multiple-choice-bias.md) |
| 후보끼리 비교하면서 순서 대칭성 유지 | [Set Transformer](papers/set-transformer.md), [Soft Contextualized Encoder](papers/soft-contextualized-encoder.md) |
| 긴 입력 처리와 후보 캐시 | [Poly-encoders](papers/poly-encoders.md), [ColBERT](papers/colbert.md), [MetaEncoder](papers/metaencoder.md) |
| 추론 시 설명 없이 reasoning 학습 효과 활용 | [Dual-Head](papers/dual-head-reasoning-distillation.md), [InnerJev](papers/innerjev.md) |
| 마지막 토큰 외 표현 활용 | [Token·Layer readout](papers/token-layer-readout.md), [GLiClass](papers/gliclass.md) |
| 기존 구조를 유지하는 대조군 | [LLM-as-Jev](papers/llm-as-jev.md), [InnerJev](papers/innerjev.md) |
| 미세조정 망각 줄이기 | [WiSE-FT](papers/wise-ft.md), [L2-SP](papers/l2-sp.md), [LoRA Learns Less](papers/lora-learns-less.md), [LP-FT](papers/lp-ft.md) |
| 가중치 섞기 | [Model Stock](papers/model-stock.md), [LiNeS](papers/lines.md), [체크포인트 평균·Soup](papers/checkpoint-averaging.md), [In-the-Wild 병합](papers/merging-in-the-wild.md) |
| 객관식 편향 측정 | [One Bias Score, Two Factors](papers/option-bias-two-factors.md), [객관식 편향](papers/multiple-choice-bias.md) |
| 난이도·품질 등급 및 확률 | [Ordinal scale bias](papers/ordinal-scale-bias.md), [Temperature scaling](papers/temperature-scaling.md) |
| Encoder 전환과 주변 연구 | [LLM2Vec](papers/llm2vec.md), [dQwen3.5](papers/dqwen35.md) |
| 통합 분류 데이터·점수 손실 | [ConEntail](papers/conentail.md), [RankT5](papers/rankt5.md) |
| CE 대신 다른 loss를 쓸지 | [CE가 정말 최선인가: loss 문헌 검토](papers/loss-alternatives.md), [판정 모델 학습 레시피](papers/decision-training-recipes.md) |
| 이미지 능력을 어떻게 얹을지(텍스트는 그대로, 이미지는 LoRA) | [모달리티별 학습 방식 문헌 검토](papers/vlm-modality-finetuning.md), [Vision 기준점](project/vision-baseline.md) |

## 전체 논문 색인

| 논문 | 연도 | 핵심 주제 |
|---|---|---|
| [Candidate-Independent Block-Causal Attention](papers/candidate-independent-attention.md) | 2026 | Attention 구조 / 순서 편향 |
| [Dual-Head Reasoning Distillation](papers/dual-head-reasoning-distillation.md) | 2025 | 학습 때만 reasoning / 분류층 |
| [Poly-encoders](papers/poly-encoders.md) | 2020 | 공통 문맥 / 후보 비교 |
| [Set Transformer](papers/set-transformer.md) | 2019 | 선택지 집합 / 대칭성 |
| [LLM2Vec](papers/llm2vec.md) | 2024 | Decoder를 표현 모델로 전환 |
| [LLM-as-Jev](papers/llm-as-jev.md) | 2026 | 기존 LM Head / 후보 문자열 점수 |
| [GLiClass](papers/gliclass.md) | 2025 | 가변 라벨 내용 / 공동 인코딩 |
| [Soft Contextualized Encoder](papers/soft-contextualized-encoder.md) | 2026 | 새 판단 아키텍처 / 집합 비교 |
| [MetaEncoder](papers/metaencoder.md) | 2026 | 후보 캐시 / 대조 학습 |
| [InnerJev / JEVal](papers/innerjev.md) | 2026 | reasoning 분포 증류 / 기존 LM Head |
| [Token·Layer 선택 readout](papers/token-layer-readout.md) | 2026 | hidden state 집계 / 작은 분류층 |
| [ColBERT](papers/colbert.md) | 2020 | 토큰 단위 late interaction |
| [RankT5](papers/rankt5.md) | 2022 | 스칼라 점수 / ranking loss |
| [Ordinal scale-utilization bias](papers/ordinal-scale-bias.md) | 2026 | 등급 편향 / 진단 |
| [ConEntail](papers/conentail.md) | 2023 | 통합 분류 데이터 / 대조 학습 |
| [dQwen3.5](papers/dqwen35.md) | 2026 | hybrid 정보 흐름 / 주변 연구 |
| [객관식 코드·위치 편향](papers/multiple-choice-bias.md) | 2024 | 코드 prior / 객관식 안정성 |
| [Temperature scaling](papers/temperature-scaling.md) | 2017 | 확률 보정 |
| [WiSE-FT](papers/wise-ft.md) | 2022 | 원본·미세조정 가중치 보간 |
| [L2-SP](papers/l2-sp.md) | 2018 | 원본 가중치 쪽 L2 제약 |
| [LoRA Learns Less and Forgets Less](papers/lora-learns-less.md) | 2024 | 미세조정 방식과 망각 |
| [LP-FT](papers/lp-ft.md) | 2022 | 출력층 먼저 학습 / 특징 왜곡 |
| [One Bias Score, Two Factors](papers/option-bias-two-factors.md) | 2026 | 객관식 편향의 두 요인 |
| [Model Stock](papers/model-stock.md) | 2024 | 각도 기반 병합 비율 |
| [LiNeS](papers/lines.md) | 2025 | 층별 스케일링 병합 |
| [체크포인트 평균·Soup](papers/checkpoint-averaging.md) | 2023 | 가중치 평균 |
| [In-the-Wild 병합 비교](papers/merging-in-the-wild.md) | 2026 | LLM 병합 방법 비교 |
| [Dataset Cartography](papers/dataset-cartography.md) | 2020 | 문항 지도 / OOD 일반화 |
| [검정력](papers/power-analysis.md) | 2020 | 평가셋 크기 |
| [TRM·T²MLR·RiM·LLoCoT 등 잠재 작업공간](papers/latent-workspaces.md) | 2025–2026 | 반복·가설 상태·학습 조건 대조 |
| [BDH·BDH-CQ, LoLCATs·Mamba in the Llama](papers/latent-workspaces.md) | 2024–2026 | 연상 메모리·공개 구현 한계·구조 전환 정렬 |
| [판정 모델 학습 레시피 묶음: LoRA placement·NDM·DKnown-Jev·Jeff·LLM-as-Jev·Signal and Noise](papers/decision-training-recipes.md) | 2025–2026 | Qwen3.5 DeltaNet 학습 위치 / Brier 손실 / 학습·평가 규모 / 연속 지표 |
| [CE가 정말 최선인가: Beyond Log Likelihood·Proper scoring rules·CV losses·DFT·GenCE·PA-GRPO](papers/loss-alternatives.md) | 2023–2026 | loss 선택 조건 / 사전 지식 강도 / 순서 일관성 |
| [모달리티별 학습 방식: Phi-4-multimodal·Llama 3.2·CogVLM·Partial LoRA·Image-LoRA·Idefics2·X-Reasoner·Prismatic·Cambrian](papers/vlm-modality-finetuning.md) | 2023–2026 | 언어 모델 동결 + 이미지 LoRA / ViT 학습 여부 / 텍스트→이미지 전이 / 망각 |

## 구현 자료와 기존 실험

- [d1·d1-omni·Jeff·Decider·d3 구현 참고](resources/implementations.md): 논문과 소스코드 근거를 구분. Decision Index 출전 모델 포함.
- v3 코드: `qwen_decision/v3/`(프롬프트·모델·엔진·학습·평가·오염 필터), 데이터 빌더 `qwen_decision/v3/sources/`.
  - 학습 셋: `mix.py`.
  - 학습: `train.py`(에폭별 저장, 원본 당기기 포함).
  - 평가: `proxy_eval.py`(JEVal), `board_check.py`(공개 샘플), `fresh_eval.py`(fresh-v1), `bias_eval.py`, `forget_report.py`, `format_probe.py`.
  - 섞기·분석: `merge.py`(WiSE·LiNeS·Model Stock·평균), `interpolate.py`, `weight_drift.py`, `check_packed.py`(회귀 검사), `bench_train.py`.
  - g1~g4b 실행 스크립트는 `runs/v3/queue/`에 있었지만, 지금은 로컬에 없다.
  - 원래 Qwen에서 출발한 비교(10-25~): `from_base_experiment.py`(구조), `training_method_experiment.py`(학습 방식), `lora_experiment.py`(attention LoRA), `lora_placement_experiment.py`(LoRA 위치), `loss_experiment.py`(loss). 출력은 `runs/from-base/`다.
  - 그 밖의 `*_experiment.py`는 g4b에서 출발한 구조 실험(10-19~24)이다.
- [조사 범위와 인용 규칙](resources/research-notes.md): 완독·재현 여부와 검색 오매칭 기록.
- [v1 실험 보고서](../docs/experiment_report_ko.md), [v2 실험 보고서](../docs/experiment_report_v2_ko.md).
- [재현 안내](../docs/reproduce_ko.md): v1/v2 재현 문서다. 당시 모델·원시 데이터·가상환경은 요청에 따라 삭제됐다. v3 환경은 `.venv`와 `models/qwen3.5-0.8b`다. g1~g7 학습의 best·final checkpoint와 일부 섞기 모델은 Hugging Face `Kyunghwan-kim/kkh-decision-model`에 있다. 10-19 이후 실험 run 폴더(`runs/`의 실험별 폴더)와 로컬에서 다시 만든 g6/g3 조합 가중치는 이 기계에만 있다.
- [기계가 읽을 수 있는 출처 색인](sources.json).

## 유지 방식

새 연구는 논문 카드 → 구조 지도 → 적용 제안 순서로 반영한다. 실험을 실제 수행하면 날짜·데이터 split·설정·결과 파일을 붙이고 미검증 표기를 갱신한다. 기존 실험 보고서는 역사적 기록으로 보존한다.
