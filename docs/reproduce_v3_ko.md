# v3 재현 안내

v3(정답 전용 학습, g6·g7, 가중치 섞기)의 환경, 모델 복구, 추론, 평가, 섞기, 학습 절차다(2026-10-21 작성). v1/v2 재현은 [재현 안내](reproduce_ko.md)에 있다. 결과와 해석은 [v3 요약](../wiki/project/v3-summary.md)과 [작업 기록](../wiki/project/v3-log.md)에 있다.

아래 명령은 이 저장소에서 실제로 쓴 환경(`.venv-di`)과 실행 스크립트에서 옮겼다. 새 머신에서 처음부터 끝까지 다시 실행해 보지는 않았다.

## 1. 환경

- Python 3.11, torch 2.10.0+cu128, transformers 5.17.0, flash-linear-attention 0.5.2, `external/decision-index`(submodule, editable 설치). 전체 목록은 [`requirements-v3.lock`](../requirements-v3.lock)이다.
- 루트 `pyproject.toml`과 `uv.lock`은 v1/v2용이다(transformers 5.18.0). v3에는 쓰지 않는다. 5.18.0에서 v3가 깨지는지는 확인하지 않았다.
- lock 파일의 `-e file:///home/khkim/...` 줄은 이 머신의 절대 경로라서 빼고, submodule을 따로 설치한다.

```bash
git submodule update --init
uv venv .venv-di --python 3.11
grep -v '^-e ' requirements-v3.lock > /tmp/requirements-v3.txt
uv pip install --python .venv-di -r /tmp/requirements-v3.txt \
    --extra-index-url https://download.pytorch.org/whl/cu128 --index-strategy unsafe-best-match
uv pip install --python .venv-di -e external/decision-index
```

## 2. 모델과 데이터 복구

```bash
hf download Qwen/Qwen3.5-0.8B --revision 2fc06364715b967f1860aea9cf38778875588b17 --local-dir models/qwen3.5-0.8b
hf download vllm-sr/d3-lite --revision a8d25a91ef75552803f7009d25b7bd435991abd9 --local-dir models/d3-lite
hf download Kyunghwan-kim/kkh-decision-model --include "runs/*" --local-dir .
hf download Kyunghwan-kim/kkh-decision-data --repo-type dataset --include "data/*" --include "external/*" --local-dir .
```

- 모델 저장소에는 각 run의 `best`만 있다. `final`은 `best`를 가리키던 링크였다(`ln -s best runs/v3/<run>/final`).
- 섞기 레시피(`runs/v3/<후보>/merge.json`)는 g6의 `last`와 g7의 `epoch-2`를 입력으로 적어 두었다. 이 두 파일은 각각 `best`와 SHA256이 같다(2026-10-21 확인). 레시피를 그대로 쓰려면 이 두 링크만 만든다.

```bash
ln -s best runs/v3/g6-gold6-pad0/last
ln -s best runs/v3/g7-gold7-ep3-pad0/epoch-2
```

- 다른 run의 `last`나 `epoch-N`을 `best`로 대신하면 안 된다. 예를 들어 g7 에폭 1·3은 `best`와 다른 가중치이고 올리지 않았다.

## 3. 추론

현재 최선 후보는 `runs/v3/wise80-g6-g7e2`다(원본 20% + g6 40% + g7 에폭 2 40%).

```python
from qwen_decision.v3.engine import Scorer

scorer = Scorer("runs/v3/wise80-g6-g7e2")
probs = scorer.score(
    'Kurt Cobain starred in the 1990 television show "Twin Peaks".\n'
    "Does the preceding sentence contain non-contemporaneous (anachronistic) elements?",
    {"q": {"type": "choice", "instructions": "Choose the correct answer.",
           "criteria": {"opt_1": "Yes", "opt_2": "No"}}},
)
print(probs["q"])  # 선택지 순서대로의 확률, 예: [0.39, 0.61]
```

- **attention 모드:** `Scorer`는 우리 체크포인트의 `v3_config.json`(g1 이후 모두 noncausal)을 읽는다. d3 체크포인트(`models/d3-lite`, `models/d3-flash`)는 `decision_config.json`의 `attention_mode`를 읽는다.
  - 2026-10-21 전에는 d3 체크포인트를 경로만으로 열면 causal로 잡혔다. 그래서 `attention="noncausal"`을 따로 줘야 했다.
  - 이미 기록된 d3-lite 점수는 이 문제의 영향을 받지 않았다. JEVal과 공개 샘플은 d3 자체 엔진으로 쟀다. fresh-v1(0.5085)은 noncausal로 다시 잰 값과 같다(causal이면 0.467).
- 원본 Qwen3.5-0.8B에는 두 설정 파일이 모두 없어 causal로 열린다. d3식 구조와 비교하려면 `attention="noncausal"`을 준다.
- Decision Index 키트로는 `--engine qwen_decision.v3.engine:V3Engine --option model=<체크포인트>`로 돌린다(아래 4절).

## 4. 평가

`runs/v3/queue/run_g3.sh`가 학습 뒤에 쓰는 명령과 같다. `<이름>`은 결과 폴더 이름이다.

```bash
# JEVal dev (출처당 60문항, 2,971문항): 비공개 same / new 대리
.venv-di/bin/python -m qwen_decision.v3.proxy_eval --engine qwen_decision.v3.engine:V3Engine \
    --option model=runs/v3/wise80-g6-g7e2 --split dev --out runs/proxy/<이름>
# fresh-v1 (16개 출처 1,335문항)
.venv-di/bin/python -m qwen_decision.v3.fresh_eval score --model runs/v3/wise80-g6-g7e2 --out runs/fresh/<이름>.json
# 공개 suite 표본 (벤치마크당 40그룹)
.venv-di/bin/python -m decision_index run --suite-dir runs/board/sample-40 --no-verify \
    --engine qwen_decision.v3.engine:V3Engine --option model=runs/v3/wise80-g6-g7e2 --out runs/board/<이름>
.venv-di/bin/python -m qwen_decision.v3.board_check score --suite-dir runs/board/sample-40 \
    --results runs/board/<이름>/results.jsonl --out runs/board/score-<이름>
```

- d3-lite의 JEVal 기준점은 d3 자체 엔진으로 쟀다: `PYTHONPATH=models/d3-lite ... proxy_eval --engine d3_engine:D3Engine --option model=models/d3-lite`.
- skill은 출처마다 max(0, (정확도 − 우연) / (1 − 우연))을 구한 뒤 출처 평균을 낸 값이다.
- fresh-v1의 GoEmotions는 학습한 데이터셋(validation split)이다. GoEmotions를 뺀 평균도 같이 본다.
- 섞기 후보는 JEVal dev와 공개 샘플로 골랐다. 그래서 선택에 쓰지 않은 확인은 fresh-v1과 JEVal restricted(아직 미사용)뿐이다.

## 5. 섞기 다시 만들기

```bash
.venv-di/bin/python -m qwen_decision.v3.merge --tuned runs/v3/g6-gold6-pad0/last runs/v3/g7-gold7-ep3-pad0/epoch-2 \
    --weights 0.5 0.5 --alpha 0.8 --out runs/v3/wise80-g6-g7e2
```

- 식은 원본 + 0.8 × (0.5·Δg6 + 0.5·Δg7e2)이다(Task Arithmetic, 벡터별 λ 0.4). CPU에서 1~2분 걸린다.
- 22개 후보의 레시피는 각 폴더의 `merge.json`에, 점수는 `runs/v3/queue/merges-g7.jsonl`에 있다.
- 대기열과 판정 스크립트(`runs/v3/queue/run_merges_g7*.sh`, `judge_merges.py`)는 git이 아니라 HF 모델 저장소에 있다.

## 6. 학습

```bash
MIX=v3-gold7 EPOCHS=3 PAD=0 EVAL_EVERY=5000 RUN=g7-gold7-ep3-pad0 runs/v3/queue/run_g3.sh
```

- 학습 셋 `data/mix/v3-gold7.jsonl`(dev는 `.dev.jsonl`)은 HF 데이터 저장소에 있다. 각 run의 `manifest.json`에 적힌 sha256으로 같은 파일인지 확인한다.
- `run_g3.sh`는 check_packed → preflight → 학습 → 에폭별 JEVal·공개 샘플 채점 → 잊음 보고서 순서로 돈다. 스크립트는 HF 모델 저장소의 `runs/v3/queue/`에 있다.
- g7은 GPU 한 장(49GB 중 약 43.5GB)에서 37.5시간 걸렸다.
- 학습 셋을 출처부터 다시 만들려면 각 빌더의 `--fetch`로 `data/raw/`를 받아야 한다(올리지 않았다). 현재 코드로 다시 만든 파일은 학습에 쓴 파일과 바이트가 다를 수 있다([v3 작업 기록](../wiki/project/v3-log.md)의 재현성 수정).
