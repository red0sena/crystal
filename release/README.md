# 배포본

| 폴더 | Hub 저장소 | 내용 |
|---|---|---|
| [`crystal-0.8b/`](crystal-0.8b/README.md) | [Kyunghwan-kim/crystal-0.8b](https://huggingface.co/Kyunghwan-kim/crystal-0.8b) (비공개, 2026-10-21) | 모델 카드와 그림(`assets/`, `make_figures.py`로 다시 그림), 독립 실행 코드(`crystal/`), 예제, `requirements.txt`, `NOTICE`, `crystal_config.json` |

- git에는 가중치와 토크나이저를 넣지 않는다. Hub 저장소를 다시 만들 때는 `runs/v3/wise80-g6-g7e2/`의 `model.safetensors`, `config.json`, `generation_config.json`, `chat_template.jinja`, `tokenizer.json`, `tokenizer_config.json`을 이 폴더 옆에 놓고 올린다.
- `crystal/format.py`와 `crystal/modeling.py`는 `qwen_decision/v3/`의 같은 파일을 복사한 것이다. 바뀐 곳은 두 군데다.
  - 설정 파일 이름이 `crystal_config.json`이다.
  - flash-linear-attention이 없어도 불러진다.
- `crystal/runtime.py`는 `qwen_decision/v3/engine.Scorer`에서 Decision Index 의존성을 뺀 것이다. 확인한 것은 둘이다.
  - 키트 엔진 `crystal.engine:CrystalEngine`과 `V3Engine`은 1,468개 질문에서 답과 확률이 같았다.
  - Hub에서 새로 받은 폴더에서도 예제가 돌아간다.
