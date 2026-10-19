"""Single-request, no-generation option scoring for Qwen3.5 VLMs."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import string
import time
from typing import Any

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration


DEFAULT_MODEL = "Qwen/Qwen3.5-0.8B"
SYSTEM_PROMPT = (
    "Use the provided text and images to answer the multiple-choice question. "
    "Treat the state as evidence. Select exactly one of the listed options. "
    "Reply with only its uppercase letter (for example A). "
    "Do not include an explanation or punctuation."
)


def validate_request(request: dict[str, Any], temperature: float) -> None:
    if not isinstance(request, dict):
        raise ValueError("Request must be a JSON object.")
    if not isinstance(request.get("question"), str) or not request["question"].strip():
        raise ValueError("question must be a nonempty string.")
    if not isinstance(request.get("state", ""), str):
        raise ValueError("state must be a string.")
    options = request.get("options")
    if not isinstance(options, dict) or not 2 <= len(options) <= 26:
        raise ValueError("options must map 2 to 26 unique names to descriptions.")
    if any(not isinstance(k, str) or not k.strip() or
           not isinstance(v, str) or not v.strip() for k, v in options.items()):
        raise ValueError("Every option needs a nonempty string name and description.")
    images = request.get("images", [])
    if not isinstance(images, list) or any(not isinstance(p, str) or not p for p in images):
        raise ValueError("images must be a list of local image paths.")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and greater than zero.")


def request_messages(request: dict[str, Any], *, image_base: Path | None = None) -> list[dict[str, Any]]:
    """Shared prompt builder for single-request inference and benchmark evaluation."""
    image_base = image_base or Path.cwd()
    content: list[dict[str, Any]] = []
    for filename in request.get("images", []):
        path = (image_base / filename).resolve()
        with Image.open(path) as image:
            content.append({"type": "image", "image": image.convert("RGB")})
    option_text = "\n".join(
        f"{string.ascii_uppercase[i]}. {description}"
        for i, description in enumerate(request["options"].values())
    )
    content.append({
        "type": "text",
        "text": (
            f"State:\n{request.get('state', '')}\n\n"
            f"Question:\n{request['question']}\n\n"
            f"Options:\n{option_text}\n\n"
            "Answer with only the option letter."
        ),
    })
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]


class ChoiceScorer:
    """One backbone forward pass per question; no generated tokens or cache branches."""

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL,
        *,
        device: str = "auto",
        max_input_tokens: int = 32768,
        max_image_pixels: int = 1048576,
    ) -> None:
        if max_input_tokens <= 0:
            raise ValueError("max_input_tokens must be positive.")
        if max_image_pixels < 65536:
            raise ValueError("max_image_pixels must be at least 65536.")
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; use --device cpu or check the GPU driver.")
        dtype = torch.float32
        if self.device.type == "cuda":
            with torch.cuda.device(self.device):
                dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        self.max_input_tokens = max_input_tokens
        self.model_path = model_path
        self.processor = AutoProcessor.from_pretrained(model_path)
        # Qwen image processor size values are pixel AREA bounds, not side lengths.
        self.processor.image_processor.size = {
            "shortest_edge": 65536, "longest_edge": max_image_pixels,
        }
        self.model = Qwen3_5ForConditionalGeneration.from_pretrained(
            model_path, dtype=dtype, attn_implementation="sdpa",
        ).to(self.device).eval()

    def prepare(
        self, request: dict[str, Any], *, image_base: Path | None = None,
    ) -> tuple[Any, list[dict[str, Any]], float]:
        start = time.perf_counter()
        options = list(request["options"].items())
        messages = request_messages(request, image_base=image_base)
        template_kwargs = dict(add_generation_prompt=True, enable_thinking=False)
        prompt = self.processor.apply_chat_template(messages, tokenize=False, **template_kwargs)
        tokenizer = self.processor.tokenizer
        prefix_ids = tokenizer.encode(prompt, add_special_tokens=False)
        labels = []
        for i, (name, description) in enumerate(options):
            letter = string.ascii_uppercase[i]
            ids = tokenizer.encode(letter, add_special_tokens=False)
            combined = tokenizer.encode(prompt + letter, add_special_tokens=False)
            if len(ids) != 1 or combined != prefix_ids + ids:
                raise ValueError(f"Option label {letter!r} is not one token at the response boundary.")
            labels.append({"name": name, "description": description,
                           "label": letter, "token_id": ids[0]})
        inputs = self.processor.apply_chat_template(
            messages, tokenize=True, return_dict=True, return_tensors="pt", **template_kwargs,
        )
        n_tokens = int(inputs["input_ids"].shape[-1])
        model_limit = int(self.model.config.text_config.max_position_embeddings)
        limit = min(self.max_input_tokens, model_limit)
        if n_tokens > limit:
            raise ValueError(
                f"Input has {n_tokens} tokens including images, exceeding limit {limit}. "
                "Shorten the state, lower --max-image-pixels, or raise --max-input-tokens. "
                "Input was not truncated."
            )
        return inputs.to(self.device), labels, (time.perf_counter() - start) * 1000

    @torch.inference_mode()
    def option_logits(self, inputs: Any, token_ids: list[int], *, head_mode: str) -> torch.Tensor:
        ids = torch.tensor(token_ids, device=self.device, dtype=torch.long)
        if head_mode == "selected":
            return self.option_logits_batch(inputs, token_ids)[0]
        if head_mode == "full":
            # Verification baseline: full vocabulary at only the FINAL position.
            output = self.model(**inputs, use_cache=False, logits_to_keep=1)
            return output.logits[0, -1].index_select(0, ids).float()
        raise ValueError("head_mode must be selected or full.")

    @torch.inference_mode()
    def option_logits_batch(
        self, inputs: Any, token_ids: list[int], *, last_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """One independent prompt per row; optionally gather before right padding."""
        output = self.model.model(**inputs, use_cache=False)
        if last_positions is None:
            hidden = output.last_hidden_state[:, -1, :]
        else:
            rows = torch.arange(output.last_hidden_state.shape[0], device=self.device)
            hidden = output.last_hidden_state[rows, last_positions, :]
        ids = torch.tensor(token_ids, device=self.device, dtype=torch.long)
        head = self.model.get_output_embeddings()
        weights = head.weight.index_select(0, ids)
        bias = head.bias.index_select(0, ids) if head.bias is not None else None
        return F.linear(hidden, weights, bias).float()

    def _synchronize(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def score(
        self,
        request: dict[str, Any],
        *,
        image_base: Path | None = None,
        temperature: float = 1.0,
        head_mode: str = "selected",
    ) -> dict[str, Any]:
        validate_request(request, temperature)
        self._synchronize()
        start = time.perf_counter()
        inputs, labels, preprocess_ms = self.prepare(request, image_base=image_base)
        self._synchronize()
        forward_start = time.perf_counter()
        logits = self.option_logits(inputs, [item["token_id"] for item in labels], head_mode=head_mode)
        self._synchronize()
        forward_ms = (time.perf_counter() - forward_start) * 1000
        if not torch.isfinite(logits).all():
            raise RuntimeError("Model returned non-finite option logits.")
        probabilities = (logits / temperature).softmax(dim=-1)
        rows = [
            {**item, "logit": float(logit), "probability": float(probability)}
            for item, logit, probability in zip(labels, logits.cpu(), probabilities.cpu())
        ]
        winner = rows[int(probabilities.argmax())]
        return {
            "model": self.model_path,
            "choice": winner["name"],
            "probabilities": {row["name"]: row["probability"] for row in rows},
            "options": rows,
            "temperature": temperature,
            "score_semantics": "softmax over supplied option-label logits; uncalibrated",
            "usage": {"input_tokens": int(inputs["input_ids"].shape[-1]), "output_tokens": 0},
            "inference": {
                "device": str(self.device), "dtype": str(self.model.dtype), "head_mode": head_mode,
                "preprocess_ms": round(preprocess_ms, 3), "forward_ms": round(forward_ms, 3),
                "total_ms": round((time.perf_counter() - start) * 1000, 3),
                "timing_note": "single call; includes no model load, no warmup, no compilation",
            },
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("request", type=Path, help="JSON with state, question, options, and optional images")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Hugging Face ID or local model directory")
    parser.add_argument("--device", default="auto", help="auto, cpu, or cuda[:index]")
    parser.add_argument("--state-file", type=Path, help="UTF-8 text file to use as state instead of JSON state")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--head-mode", choices=["selected", "full"], default="selected")
    parser.add_argument("--max-input-tokens", type=int, default=32768)
    parser.add_argument("--max-image-pixels", type=int, default=1048576)
    parser.add_argument("--output", type=Path, help="Also save result JSON here")
    args = parser.parse_args()
    try:
        request = json.loads(args.request.read_text(encoding="utf-8"))
        if args.state_file:
            if not isinstance(request, dict):
                raise ValueError("Request must be a JSON object.")
            request["state"] = args.state_file.read_text(encoding="utf-8")
        validate_request(request, args.temperature)
        scorer = ChoiceScorer(
            args.model, device=args.device, max_input_tokens=args.max_input_tokens,
            max_image_pixels=args.max_image_pixels,
        )
        result = scorer.score(
            request, image_base=args.request.resolve().parent,
            temperature=args.temperature, head_mode=args.head_mode,
        )
        rendered = json.dumps(result, ensure_ascii=False, indent=2)
        if args.output:
            args.output.write_text(rendered + "\n", encoding="utf-8")
        print(rendered)
    except (ValueError, OSError, RuntimeError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
