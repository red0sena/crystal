"""Decision Index engine for v3 checkpoints (and the untouched base model).

  python -m decision_index run --engine qwen_decision.v3.engine:V3Engine \
      --option model=<checkpoint dir> [--option orders=2] ...

One forward pass per question prompt; questions of a request are batched under
a token budget. ``orders`` > 1 also scores each question with its options
reversed (then rotated) and averages the probabilities in the original option
order: one generic inference rule for every request, no per-benchmark logic.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer

from decision_index.engines import Engine, Unsupported

from qwen_decision.v3 import format as fmt
from qwen_decision.v3 import modeling


def option_orders(k: int, orders: int) -> list[list[int]]:
    """Display orders as permutations of the original option indices."""
    base = list(range(k))
    found = [base, base[::-1]]
    for i in range(1, orders):
        shift = round(i * k / orders) % k
        found += [base[shift:] + base[:shift], (base[shift:] + base[:shift])[::-1]]
    unique = []
    for order in found:
        if order not in unique:
            unique.append(order)
    return unique[:orders]


def reorder(question: dict[str, Any], order: list[int]) -> dict[str, Any]:
    """The question with its options displayed in `order`; noul becomes a 2-option choice."""
    keys, texts = fmt.options(question)
    if question["type"] == "noul":
        # Keyed by the rendered text with no description, so each option line reads exactly as in the noul prompt.
        if texts[0] == texts[1]:
            raise Unsupported("noul question with identical true/false descriptions")
        return {"type": "choice", "instructions": question.get("instructions"),
                "criteria": {texts[i]: None for i in order}}
    return {**question, "criteria": {keys[i]: question["criteria"][keys[i]] for i in order}}


class Scorer:
    """Loaded checkpoint that returns per-question probabilities in options() order."""

    def __init__(self, model: str | Path, device: str = "cuda:0", attention: str | None = None,
                 layout: str | None = None, batch_tokens: int = 32768, max_batch: int = 8, orders: int = 1,
                 deltanet: str | None = None):
        config = modeling.read_config(Path(model))
        self.attention = attention or config["attention"]
        self.layout = layout or config["layout"]
        self.deltanet = deltanet or config["deltanet"]
        self.device = device
        self.model = modeling.load(Path(model), attention=self.attention, deltanet=self.deltanet,
                                   dtype=torch.bfloat16, device=device)
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.codes, self.code_ids = fmt.answer_codes(self.tokenizer)
        self.readout = modeling.load_readout(Path(model), self.code_ids, device)
        if self.readout is None and not self.model.config.get_text_config().tie_word_embeddings:
            raise ValueError("an untied checkpoint needs its readout.safetensors: the LM head is not part of it")
        self.batch_tokens, self.max_batch, self.orders = int(batch_tokens), int(max_batch), int(orders)

    def prompts(self, state, questions: dict[str, Any]) -> list[tuple[str, list[int], list[int], int]]:
        """(question key, token ids, code token ids, order index) for every prompt to run."""
        items = []
        for key, question in questions.items():
            if question["type"] not in ("choice", "noul"):
                raise Unsupported(f"unsupported question type {question['type']!r}")
            k = len(fmt.options(question)[0])
            if not 2 <= k <= fmt.MAX_OPTIONS:
                raise Unsupported(f"question {key!r} has {k} options; this model answers 2 to {fmt.MAX_OPTIONS}")
            for index, order in enumerate(option_orders(k, self.orders)):
                shown = reorder(question, order) if index else question
                text = fmt.render(self.tokenizer, state, shown, self.codes[:k], self.layout)
                items.append((key, self.tokenizer.encode(text, add_special_tokens=False), self.code_ids[:k], index))
        return items

    @torch.no_grad()
    def forward(self, batch):
        width = max(len(ids) for _, ids, _, _ in batch)
        kmax = max(len(c) for _, _, c, _ in batch)
        pad = self.tokenizer.pad_token_id
        input_ids = torch.tensor([ids + [pad] * (width - len(ids)) for _, ids, _, _ in batch], device=self.device)
        mask = torch.tensor([[1] * len(ids) + [0] * (width - len(ids)) for _, ids, _, _ in batch], device=self.device)
        codes = torch.tensor([c + [c[0]] * (kmax - len(c)) for _, _, c, _ in batch], device=self.device)
        code_mask = torch.tensor([[True] * len(c) + [False] * (kmax - len(c)) for _, _, c, _ in batch], device=self.device)
        logits = modeling.code_logits(self.model, input_ids, mask, codes, code_mask, readout=self.readout)
        return [row[:len(c)].softmax(-1).tolist() for row, (_, _, c, _) in zip(logits, batch)]

    def batches(self, items):
        order = sorted(range(len(items)), key=lambda i: len(items[i][1]))
        group = []
        for i in order:
            longest = max([len(items[j][1]) for j in group] + [len(items[i][1])])
            if group and (len(group) >= self.max_batch or longest * (len(group) + 1) > self.batch_tokens):
                yield group
                group = []
            group.append(i)
        if group:
            yield group

    def score(self, state, questions: dict[str, Any]) -> dict[str, list[float]]:
        items = self.prompts(state, questions)
        results: list[list[float] | None] = [None] * len(items)
        for group in self.batches(items):
            try:
                outputs = self.forward([items[i] for i in group])
            except torch.OutOfMemoryError:
                torch.cuda.empty_cache()
                outputs = [self.forward([items[i]])[0] for i in group]
            for i, probabilities in zip(group, outputs):
                results[i] = probabilities
        averaged: dict[str, list[float]] = {}
        for (key, _, codes, index), probabilities in zip(items, results):
            k = len(codes)
            order = option_orders(k, self.orders)[index]
            original = [0.] * k
            for position, option in enumerate(order):
                original[option] = probabilities[position]
            total = averaged.setdefault(key, [0.] * k)
            for i in range(k):
                total[i] += original[i]
        return {key: [p / sum(values) for p in values] for key, values in averaged.items()}

    def score_many(self, requests: list[tuple[Any, dict[str, Any]]]) -> list[dict[str, list[float]]]:
        """score() for many (state, questions) requests at once, batching prompts across requests.

        For offline labelling: the same prompts, forward pass and order averaging as score(), only
        the batches mix requests.
        """
        items = [((r, key), ids, codes, index) for r, (state, questions) in enumerate(requests)
                 for key, ids, codes, index in self.prompts(state, questions)]
        results: list[list[float] | None] = [None] * len(items)
        for group in self.batches(items):
            try:
                outputs = self.forward([items[i] for i in group])
            except torch.OutOfMemoryError:
                torch.cuda.empty_cache()
                outputs = [self.forward([items[i]])[0] for i in group]
            for i, probabilities in zip(group, outputs):
                results[i] = probabilities
        averaged: list[dict[str, list[float]]] = [{} for _ in requests]
        for ((r, key), _, codes, index), probabilities in zip(items, results):
            k = len(codes)
            order = option_orders(k, self.orders)[index]
            total = averaged[r].setdefault(key, [0.] * k)
            for position, option in enumerate(order):
                total[option] += probabilities[position]
        return [{key: [p / sum(values) for p in values] for key, values in a.items()} for a in averaged]


class V3Engine(Engine):
    name = "qwen-decision-v3"
    latency = ("Device-synchronized in-process request wall time including prompt rendering and tokenization; "
               "one request per call, its question prompts batched under a token budget; excludes model loading.")

    def __init__(self, model: str | None = None, device: str = "cuda:0", attention: str | None = None,
                 layout: str | None = None, batch_tokens: int = 32768, max_batch: int = 8, orders: int = 1,
                 deltanet: str | None = None, **options):
        if options:
            raise TypeError(f"unknown engine options {sorted(options)}")
        if not model:
            raise ValueError("pass --option model=<checkpoint dir>")
        super().__init__(model=model, device=device, attention=attention, layout=layout,
                         batch_tokens=batch_tokens, max_batch=max_batch, orders=orders, deltanet=deltanet)
        started = time.perf_counter()
        self.scorer = Scorer(model, device, attention, layout, batch_tokens, max_batch, orders, deltanet)
        self.provenance = {"kind": "qwen-decision-v3", "model": str(model), "attention": self.scorer.attention,
                           "deltanet": self.scorer.deltanet,
                           "layout": self.scorer.layout, "orders": self.scorer.orders, "backbone_dtype": "bfloat16",
                           "readout_dtype": "float32", "readout": "d3 readout.safetensors" if self.scorer.readout else "tied LM head",
                           "load_seconds": time.perf_counter() - started,
                           "policy": "One prompt per question (per option order); last-token readout over the "
                                     "question's answer codes; probabilities averaged over option orders; argmax "
                                     "choice; no truncation; no option filtering; one fixed prompt for every request."}

    def synchronize(self):
        torch.cuda.synchronize(self.scorer.device)

    def runtime(self):
        return {"torch": torch.__version__, "gpu": torch.cuda.get_device_name(self.scorer.device)}

    def __call__(self, state, questions, **inputs):
        if inputs.get("images") or inputs.get("videos"):
            raise Unsupported("text-only engine")
        probabilities = self.scorer.score(state, questions)
        answers = {key: fmt.to_answer(question, probabilities[key]) for key, question in questions.items()}
        return {"model": self.name, "answers": answers}, None
