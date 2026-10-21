"""Decision Index kit engine for crystal-0.8b (github.com/apolinario/decision-index).

    PYTHONPATH=/path/to/crystal-0.8b python -m decision_index run --engine crystal.engine:CrystalEngine \
        --option model=/path/to/crystal-0.8b --out runs/crystal

One request per call; its question prompts batched under a token budget. Text only: requests with images or
videos, other question types and more than 255 options are reported as unsupported (they count as wrong).
"""
from __future__ import annotations

from pathlib import Path
import time

import torch
from decision_index.engines import Engine, Unsupported as KitUnsupported

from . import format as fmt
from .runtime import Crystal, Unsupported

HERE = Path(__file__).resolve().parents[1]


class CrystalEngine(Engine):
    name = "crystal-0.8b"
    latency = ("Device-synchronized in-process request wall time including prompt rendering and tokenization; "
               "one request per call, its question prompts batched under a token budget; excludes model loading.")

    def __init__(self, model: str | None = None, device: str = "cuda:0", orders: int = 1, batch_tokens: int = 32768,
                 max_batch: int = 8, **options):
        if options:
            raise TypeError(f"unknown engine options {sorted(options)}")
        model = model or str(HERE)
        super().__init__(model=model, device=device, orders=orders, batch_tokens=batch_tokens, max_batch=max_batch)
        started = time.perf_counter()
        self.crystal = Crystal.from_pretrained(model, device=device, orders=orders, batch_tokens=batch_tokens,
                                               max_batch=max_batch)
        self.provenance = {"kind": "crystal-0.8b", "model": model, "attention": self.crystal.attention,
                           "deltanet": self.crystal.deltanet, "layout": self.crystal.layout, "orders": self.crystal.orders,
                           "backbone_dtype": "bfloat16", "readout_dtype": "float32", "readout": "tied LM head",
                           "load_seconds": time.perf_counter() - started,
                           "policy": "One prompt per question (per option order); last-token readout over the "
                                     "question's answer codes; argmax choice; no truncation; no option filtering; "
                                     "one fixed prompt for every request."}

    def synchronize(self):
        if str(self.crystal.device).startswith("cuda"):
            torch.cuda.synchronize(self.crystal.device)

    def runtime(self):
        gpu = torch.cuda.get_device_name(self.crystal.device) if str(self.crystal.device).startswith("cuda") else "cpu"
        return {"torch": torch.__version__, "gpu": gpu}

    def __call__(self, state, questions, **inputs):
        if inputs.get("images") or inputs.get("videos"):
            raise KitUnsupported("text-only model")
        try:
            probabilities = self.crystal.probabilities(state, questions)
        except Unsupported as e:
            raise KitUnsupported(str(e)) from e
        answers = {key: fmt.to_answer(question, probabilities[key]) for key, question in questions.items()}
        return {"model": self.name, "answers": answers}, None
