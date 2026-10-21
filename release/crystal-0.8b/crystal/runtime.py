"""crystal-0.8b runtime: answer typed decision questions without generating text.

    from crystal import Crystal

    model = Crystal.from_pretrained("path/to/crystal-0.8b")       # or a Hub id
    result = model.system_one(state="...", questions={"route": {...}, "receipt": {...}})

One prompt and one forward pass per question; the questions of a request are batched under a
token budget. The answer is the softmax over the question's answer codes at the last prompt token.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer

from . import format as fmt
from . import modeling


class Unsupported(ValueError):
    """A request this model cannot answer: another question type, or fewer than 2 / more than 255 options."""


def option_orders(k: int, orders: int) -> list[list[int]]:
    """Display orders as permutations of the original option indices (identity first, then reversed, ...)."""
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
        if texts[0] == texts[1]:
            raise Unsupported("noul question with identical true/false descriptions")
        return {"type": "choice", "instructions": question.get("instructions"),
                "criteria": {texts[i]: None for i in order}}
    return {**question, "criteria": {keys[i]: question["criteria"][keys[i]] for i in order}}


class Crystal:
    """A loaded crystal-0.8b checkpoint.

    orders: 1 scores each question once (the setting all reported scores use). 2 or more also scores it with
    the options reversed (then rotated) and averages, which costs that many forward passes per question.
    """

    def __init__(self, path: str | Path, device: str | None = None, orders: int = 1, batch_tokens: int = 32768,
                 max_batch: int = 8):
        path = Path(path)
        config = modeling.read_config(path)
        self.path = path
        self.attention, self.layout, self.deltanet = config["attention"], config["layout"], config["deltanet"]
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = modeling.load(path, attention=self.attention, deltanet=self.deltanet, dtype=torch.bfloat16,
                                   device=self.device)
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.codes, self.code_ids = fmt.answer_codes(self.tokenizer)
        self.readout = modeling.load_readout(path, self.code_ids, self.device)
        self.batch_tokens, self.max_batch, self.orders = int(batch_tokens), int(max_batch), int(orders)

    @classmethod
    def from_pretrained(cls, name_or_path: str | Path, **kwargs) -> "Crystal":
        """A local folder, or a Hub repo id (downloaded with huggingface_hub; private repos need `hf auth login`)."""
        path = Path(name_or_path)
        if not path.is_dir():
            from huggingface_hub import snapshot_download

            path = Path(snapshot_download(str(name_or_path)))
        return cls(path, **kwargs)

    # ------------------------------------------------------------------ prompts and batches
    def _prompts(self, state, questions: dict[str, Any]):
        items = []
        for key, question in questions.items():
            if question.get("type") not in ("choice", "noul"):
                raise Unsupported(f"question {key!r}: type {question.get('type')!r} is not supported (choice or noul)")
            k = len(fmt.options(question)[0])
            if not 2 <= k <= fmt.MAX_OPTIONS:
                raise Unsupported(f"question {key!r} has {k} options; this model answers 2 to {fmt.MAX_OPTIONS}")
            for index, order in enumerate(option_orders(k, self.orders)):
                shown = reorder(question, order) if index else question
                text = fmt.render(self.tokenizer, state, shown, self.codes[:k], self.layout)
                items.append((key, self.tokenizer.encode(text, add_special_tokens=False), self.code_ids[:k], index))
        return items

    @torch.no_grad()
    def _forward(self, batch):
        width = max(len(ids) for _, ids, _, _ in batch)
        kmax = max(len(c) for _, _, c, _ in batch)
        pad = self.tokenizer.pad_token_id
        input_ids = torch.tensor([ids + [pad] * (width - len(ids)) for _, ids, _, _ in batch], device=self.device)
        mask = torch.tensor([[1] * len(ids) + [0] * (width - len(ids)) for _, ids, _, _ in batch], device=self.device)
        codes = torch.tensor([c + [c[0]] * (kmax - len(c)) for _, _, c, _ in batch], device=self.device)
        code_mask = torch.tensor([[True] * len(c) + [False] * (kmax - len(c)) for _, _, c, _ in batch], device=self.device)
        logits = modeling.code_logits(self.model, input_ids, mask, codes, code_mask, readout=self.readout)
        return [row[:len(c)].softmax(-1).tolist() for row, (_, _, c, _) in zip(logits, batch)]

    def _batches(self, items):
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

    # ------------------------------------------------------------------ public API
    def probabilities(self, state, questions: dict[str, Any]) -> dict[str, list[float]]:
        """Per question, the probability of every option in the order the request lists them
        (for noul: [P(false), P(true)])."""
        items = self._prompts(state, questions)
        results: list[list[float] | None] = [None] * len(items)
        for group in self._batches(items):
            try:
                outputs = self._forward([items[i] for i in group])
            except torch.OutOfMemoryError:
                torch.cuda.empty_cache()
                outputs = [self._forward([items[i]])[0] for i in group]
            for i, p in zip(group, outputs):
                results[i] = p
        averaged: dict[str, list[float]] = {}
        for (key, _, codes, index), p in zip(items, results):
            k = len(codes)
            order = option_orders(k, self.orders)[index]
            total = averaged.setdefault(key, [0.] * k)
            for position, option in enumerate(order):
                total[option] += p[position]
        return {key: [v / sum(values) for v in values] for key, values in averaged.items()}

    def system_one(self, state, questions: dict[str, Any]) -> dict[str, Any]:
        """Answers in the Decision Index format.

        choice -> {"type": "choice", "choice": <best key>, "probabilities": {key: p, ...}}
        noul   -> {"type": "noul", "noul": P(true)}
        """
        probabilities = self.probabilities(state, questions)
        return {"model": "crystal-0.8b",
                "answers": {key: fmt.to_answer(q, probabilities[key]) for key, q in questions.items()}}

    __call__ = system_one
