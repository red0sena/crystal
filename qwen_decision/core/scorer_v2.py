"""v2 decision scorer: choice / noul / score questions, 2-64 options, optional
temperature calibration, and one shared state read for many questions.

No generate() call and zero output tokens. Text only (the vision encoder was
not trained in v2). Prompts come from decision_format.py, the same module
training used.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import time
from typing import Any

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

import qwen_decision.core.decision_format as fmt


class DecisionScorer:
    def __init__(self, model_path: str | Path, *, device: str = "cuda", max_input_tokens: int = 32768,
                 calibration: str | Path | dict | None = "auto", batch_tokens: int = 32768, max_batch: int = 32,
                 dtype: torch.dtype | None = None):
        self.model_path = str(model_path)
        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        dtype = dtype or (torch.bfloat16 if self.device.type == "cuda" else torch.float32)
        model = Qwen3_5ForConditionalGeneration.from_pretrained(model_path, dtype=dtype, attn_implementation="sdpa")
        # Same text path as training: the language model plus the tied LM head rows.
        self.lm = model.model.language_model.to(self.device).eval().requires_grad_(False)
        self.head = model.get_output_embeddings().weight.detach().to(self.device)
        del model
        self.code_ids = fmt.code_token_ids(self.tokenizer)
        self.max_input_tokens = min(max_input_tokens, int(self.lm.config.max_position_embeddings))
        self.batch_tokens, self.max_batch = batch_tokens, max_batch
        if calibration == "auto":
            path = Path(model_path) / "calibration.json"
            calibration = path if path.exists() else None
        if isinstance(calibration, (str, Path)):
            calibration = json.loads(Path(calibration).read_text())["temperatures"]
        self.temperatures: dict[str, float] = dict(calibration or {})

    # ------------------------------------------------------------------ requests
    @staticmethod
    def normalize(request: dict[str, Any]) -> tuple[str, list[tuple[str, dict[str, Any]]]]:
        if not isinstance(request, dict):
            raise ValueError("Request must be a JSON object.")
        state = request.get("state", "")
        if not isinstance(state, str):
            state = json.dumps(state, ensure_ascii=False, indent=2)
        if "questions" in request:
            questions = request["questions"]
            if not isinstance(questions, dict) or not questions:
                raise ValueError("questions must map names to question objects.")
        else:
            questions = {"answer": {k: v for k, v in request.items() if k != "state"}}
        for q in questions.values():
            fmt.validate_question(q)
        return state, list(questions.items())

    @staticmethod
    def options_of(q: dict[str, Any]) -> tuple[list[str], list[str], str]:
        qtype = q.get("type", "choice")
        if qtype == "noul":
            options = q.get("options") or {"yes": "Yes", "no": "No"}
            if set(options) != {"yes", "no"}:
                raise ValueError("noul options must be exactly 'yes' and 'no' (descriptions optional).")
        else:
            options = q["options"]
        return list(options), list(options.values()), qtype

    # ------------------------------------------------------------------ model
    def _encode(self, text: str) -> list[int]:
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        if len(ids) > self.max_input_tokens:
            raise ValueError(f"Input has {len(ids)} tokens, exceeding {self.max_input_tokens}. Input was not truncated.")
        return ids

    @torch.inference_mode()
    def last_hidden(self, sequences: list[list[int]]) -> torch.Tensor:
        """Final-norm hidden state at each sequence's last real token (right padding)."""
        out = torch.empty(len(sequences), self.head.shape[1], device=self.device, dtype=self.head.dtype)
        order = sorted(range(len(sequences)), key=lambda i: len(sequences[i]))
        start = 0
        while start < len(order):
            stop = start + 1
            while (stop < len(order) and stop - start < self.max_batch
                   and len(sequences[order[stop]]) * (stop - start + 1) <= self.batch_tokens):
                stop += 1
            rows = [sequences[i] for i in order[start:stop]]
            width = max(map(len, rows))
            ids = torch.full((len(rows), width), self.tokenizer.pad_token_id, device=self.device)
            mask = torch.zeros((len(rows), width), device=self.device, dtype=torch.long)
            for j, r in enumerate(rows):
                ids[j, :len(r)] = torch.tensor(r, device=self.device)
                mask[j, :len(r)] = 1
            hidden = self.lm(input_ids=ids, attention_mask=mask, use_cache=False).last_hidden_state
            last = mask.sum(-1) - 1
            out[order[start:stop]] = hidden[torch.arange(len(rows), device=self.device), last]
            start = stop
        return out

    @torch.inference_mode()
    def last_hidden_shared(self, prefix: list[int], suffixes: list[list[int]]) -> torch.Tensor:
        """Read the shared prefix once, then fork its cache (attention KV plus
        Gated-DeltaNet conv/recurrent states) for each right-padded suffix."""
        prefix_ids = torch.tensor([prefix], device=self.device)
        cache = self.lm(input_ids=prefix_ids, use_cache=True).past_key_values
        out = torch.empty(len(suffixes), self.head.shape[1], device=self.device, dtype=self.head.dtype)
        for start in range(0, len(suffixes), self.max_batch):
            rows = suffixes[start:start + self.max_batch]
            width = max(map(len, rows))
            ids = torch.full((len(rows), width), self.tokenizer.pad_token_id, device=self.device)
            mask = torch.zeros((len(rows), len(prefix) + width), device=self.device, dtype=torch.long)
            mask[:, :len(prefix)] = 1
            for j, r in enumerate(rows):
                ids[j, :len(r)] = torch.tensor(r, device=self.device)
                mask[j, len(prefix):len(prefix) + len(r)] = 1
            # attention KV뿐 아니라 DeltaNet recurrent/conv 상태도 함께 복사한다.
            # 각 질문은 공통 상태와 자기 질문만 보며 다른 질문의 답/상태를 참조하지 않는다.
            # cache 재사용은 입력 계산을 줄일 뿐 출력 토큰 생성이나 추가 reasoning을 수행하지 않는다.
            branch = copy.deepcopy(cache)
            branch.reorder_cache(torch.zeros(len(rows), dtype=torch.long, device=self.device))
            hidden = self.lm(input_ids=ids, attention_mask=mask, past_key_values=branch, use_cache=True).last_hidden_state
            last = mask[:, len(prefix):].sum(-1) - 1
            out[start:start + len(rows)] = hidden[torch.arange(len(rows), device=self.device), last]
        return out

    @torch.inference_mode()
    def option_logits(self, hidden: torch.Tensor, codes: list[str]) -> torch.Tensor:
        ids = torch.tensor(fmt.code_ids_for(self.tokenizer, codes), device=self.device)
        return F.linear(hidden.unsqueeze(0), self.head.index_select(0, ids)).float()[0]

    # ------------------------------------------------------------------ answers
    def answer(self, qtype: str, names: list[str], logits: torch.Tensor, calibrated: bool) -> dict[str, Any]:
        key = fmt.calibration_key(qtype, len(names))
        temperature = self.temperatures.get(key, 1.0) if calibrated else 1.0
        # 보정은 선택지 수/질문 유형별 양의 scalar T를 적용한다. argmax 순위는 변하지 않는다.
        # 다른 가중치 모델로 교체하면 그 모델의 별도 개발 logits로 calibration을 다시 맞춘다.
        probs = (logits / temperature).softmax(-1).tolist()
        table = dict(zip(names, probs))
        result: dict[str, Any] = {"type": qtype, "probabilities": table, "calibration": {"key": key, "temperature": temperature}}
        if qtype == "choice":
            result.update(choice=max(table, key=table.get), confidence=fmt.choice_confidence(probs))
        elif qtype == "noul":
            result.update(noul=table["yes"], confidence=abs(2 * table["yes"] - 1))
        else:
            summary = fmt.score_summary(probs)
            numeric = all(n.lstrip("-").replace(".", "", 1).isdigit() for n in names)
            expected = (sum(float(n) * p for n, p in zip(names, probs)) if numeric else summary["expected_level"])
            result.update(score=expected, level=names[summary["mode_level"]], confidence=summary["confidence"])
        return result

    def score(self, request: dict[str, Any], *, share_state: bool = True, calibrated: bool = True) -> dict[str, Any]:
        tick = time.perf_counter()
        state, questions = self.normalize(request)
        rendered = []
        for name, q in questions:
            names, texts, qtype = self.options_of(q)
            codes = fmt.assign_codes(len(names))
            rendered.append((name, qtype, names, codes, texts, q["question"]))
        mode = "independent"
        hidden = None
        if share_state and len(rendered) > 1:
            parts = [fmt.split_prompt(self.tokenizer, state, question, codes, texts, qtype)
                     for _, qtype, _, codes, texts, question in rendered]
            prefix = self._encode(parts[0][0])
            suffixes = [self.tokenizer.encode(s, add_special_tokens=False) for _, s in parts]
            fulls = [self.tokenizer.encode(p + s, add_special_tokens=False) for p, s in parts]
            # Shared reading is used only when it reproduces each prompt's exact tokens.
            if all(p == parts[0][0] for p, _ in parts) and all(prefix + s == f for s, f in zip(suffixes, fulls)):
                if len(prefix) + max(map(len, suffixes)) > self.max_input_tokens:
                    raise ValueError("Input exceeds the token limit. Input was not truncated.")
                hidden = self.last_hidden_shared(prefix, suffixes)
                mode = "shared_state"
                input_tokens = len(prefix) + sum(map(len, suffixes))
        if hidden is None:
            sequences = [self._encode(fmt.prompt_text(self.tokenizer, state, question, codes, texts, qtype))
                         for _, qtype, _, codes, texts, question in rendered]
            hidden = self.last_hidden(sequences)
            input_tokens = sum(map(len, sequences))
        answers = {}
        for i, (name, qtype, names, codes, _, _) in enumerate(rendered):
            logits = self.option_logits(hidden[i], codes)
            if not torch.isfinite(logits).all():
                raise RuntimeError("Model returned non-finite option logits.")
            answers[name] = self.answer(qtype, names, logits, calibrated)
        return {"model": self.model_path, "answers": answers,
                "usage": {"input_tokens": input_tokens, "output_tokens": 0, "state_reading": mode},
                "score_semantics": ("softmax over supplied option-code logits"
                                    + ("; temperature-calibrated on held-out data" if calibrated and self.temperatures else "; uncalibrated")),
                "elapsed_ms": round((time.perf_counter() - tick) * 1000, 3)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("request", type=Path, help="JSON: {state, questions:{name:{type, question, options}}} or one question")
    parser.add_argument("--model", default=str(Path(__file__).resolve().parents[2] / "training/text-decision-v2/model"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--state-file", type=Path, help="UTF-8 file used as the state")
    parser.add_argument("--no-share-state", action="store_true")
    parser.add_argument("--uncalibrated", action="store_true")
    parser.add_argument("--max-input-tokens", type=int, default=32768)
    args = parser.parse_args()
    try:
        request = json.loads(args.request.read_text(encoding="utf-8"))
        if args.state_file:
            request["state"] = args.state_file.read_text(encoding="utf-8")
        scorer = DecisionScorer(args.model, device=args.device, max_input_tokens=args.max_input_tokens)
        result = scorer.score(request, share_state=not args.no_share_state, calibrated=not args.uncalibrated)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except (ValueError, OSError, RuntimeError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
