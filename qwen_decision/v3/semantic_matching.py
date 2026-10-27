"""A function-preserving, single-encoding semantic evidence readout.

All arms keep the native backbone and code head. A zero-output matching
branch adds an internal logit correction. Generic and semantic cross-attention
and semantic self-attention have identical parameters and initialization.
Semantic spans refer to actual question/claim/hypothesis fields, not just the
generic task instruction. No labels or input tokens determine those spans.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from safetensors.torch import load_file, save_file

from qwen_decision.v3 import dual_matching as dual, format as fmt, modeling

POLICIES = {"generic": ("generic", "dual"), "semantic": ("semantic", "dual"),
            "self": ("semantic", "self")}
INTERVENTIONS = ("active", "branch_off", "question_zero", "evidence_zero", "generic_route")


def field_ranges(state):
    """Character ranges of top-level JSON string contents, including escaping.

    Building the exact serialization avoids accidentally selecting an identical
    question string quoted inside the evidence or a field name in its content.
    """
    parts, spans, offset = [], {}, 1
    for key, value in state.items():
        prefix = json.dumps(key, ensure_ascii=False) + ": "
        serialized = json.dumps(value, ensure_ascii=False)
        if isinstance(value, str) and value:
            spans[key] = (offset + len(prefix) + 1, offset + len(prefix) + len(serialized) - 1)
        part = prefix + serialized
        parts.append(part)
        offset += len(part) + 2
    if "{" + ", ".join(parts) + "}" != fmt.describe(state):
        raise ValueError("state serialization changed")
    return spans


def content_spans(renderer, raw, encoded):
    if renderer.args.layout != "d3":
        raise ValueError("fixed d3 layout required")
    question = raw["question"]
    if question["type"] == "choice":
        keys = list(question["criteria"])
        question = {**question, "criteria": {keys[i]: question["criteria"][keys[i]] for i in encoded["order"]}}
    labels = dict(zip(renderer.code_ids, renderer.codes))
    labels = [labels[c] for c in encoded["code_ids"]]
    user, candidates = fmt._user_prompt_spans(raw["state"], question, labels, "d3")
    state = fmt.describe(raw["state"]) if raw["state"] not in (None, "") else "(empty)"
    asked = fmt.describe(question.get("instructions") or fmt.DEFAULT_INSTRUCTIONS)
    state_start = len("State:\n")
    question_start = state_start + len(state) + len("\n\nQuestion:\n")
    text = fmt.render(renderer.tokenizer, raw["state"], question, labels, "d3")
    start = text.rfind(user)
    if start < 0:
        raise ValueError("chat template changed the user prompt")
    tokens = renderer.tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    if tokens["input_ids"] != encoded["input_ids"]:
        raise ValueError("span reconstruction changed input tokens")
    special = set(renderer.tokenizer.all_special_ids)

    def positions(lo, hi):
        result = [i for i, ((a, b), t) in enumerate(zip(tokens["offset_mapping"], tokens["input_ids"]))
                  if b > a and a < start + hi and b > start + lo and t not in special]
        if not result:
            raise ValueError(f"empty semantic span: {raw['id']}")
        return result

    passage = positions(state_start, state_start + len(state))
    generic = positions(question_start, question_start + len(asked))
    evidence, query, role = passage, generic, "state_and_instruction"
    if isinstance(raw["state"], dict):
        spans = field_ranges(raw["state"])
        for p, q in (("passage", "question"), ("evidence", "claim"), ("premise", "hypothesis")):
            if p in spans and q in spans:
                evidence = positions(*(state_start + x for x in spans[p]))
                query = positions(*(state_start + x for x in spans[q]))
                role = f"{p}:{q}"
                if set(evidence) & set(query):
                    raise ValueError("evidence/query token spans overlap")
                break
    return {"generic_passage_positions": passage, "generic_question_positions": generic,
            "semantic_evidence_positions": evidence, "semantic_query_positions": query,
            "candidate_positions": [positions(a, b) for a, b in candidates], "semantic_role": role}


class ResidualMatching(dual.DualMatching):
    def __init__(self, width, *, arm="semantic", heads=8, layers=2, device=None, dtype=None):
        policy, mode = POLICIES[arm]
        super().__init__(width, heads=heads, layers=layers, mode=mode, device=device, dtype=dtype)
        self.arm, self.policy = arm, policy
        nn.init.zeros_(self.scorer.weight)

    def forward(self, hidden, rows, *, checkpointed=False, intervention="active"):
        if intervention not in INTERVENTIONS:
            raise ValueError(intervention)
        results = []
        for x, row in zip(hidden, rows):
            policy = "generic" if intervention == "generic_route" else self.policy
            pp = row["generic_passage_positions" if policy == "generic" else "semantic_evidence_positions"]
            qq = row["generic_question_positions" if policy == "generic" else "semantic_query_positions"]
            passage = x[pp][None]
            if intervention == "evidence_zero":
                passage = torch.zeros_like(passage)
            choices = row["candidate_positions"]
            chunk = max(1, min(8, 2048 // (len(pp) + len(qq) + max(map(len, choices)))))
            scores = []
            for at in range(0, len(choices), chunk):
                indices = [qq + c for c in choices[at:at + chunk]]
                width = max(map(len, indices))
                idx = torch.tensor([v + [0] * (width - len(v)) for v in indices], device=x.device)
                mask = torch.arange(width, device=x.device)[None] < torch.tensor([len(v) for v in indices], device=x.device)[:, None]
                qa = x[idx] * mask[:, :, None]
                if intervention == "question_zero":
                    keep = torch.arange(width, device=x.device)[None, :, None] >= len(qq)
                    qa = qa * keep
                p = passage.expand(len(indices), -1, -1)
                pmask = torch.ones(p.shape[:2], dtype=torch.bool, device=x.device)
                if self.training and checkpointed:
                    score = checkpoint(self.match, p, qa, pmask, mask, use_reentrant=False)
                else:
                    score = self.match(p, qa, pmask, mask)
                scores.append(score)
            result = torch.cat(scores)
            results.append(result * 0 if intervention == "branch_off" else result)
        width = max(map(len, results))
        return torch.stack([F.pad(x, (0, width - len(x))) for x in results])


def install(text, arm, *, heads=8, layers=2):
    if hasattr(text, "semantic_matching"):
        raise ValueError("matching already installed")
    w = text.embed_tokens.weight
    module = ResidualMatching(w.shape[-1], arm=arm, heads=heads, layers=layers, device=w.device, dtype=w.dtype)
    text.add_module("semantic_matching", module)
    return module


def from_hidden(model, hidden, attention_mask, code_ids, code_mask, rows, *, intervention="active"):
    h = hidden[torch.arange(len(hidden), device=hidden.device), modeling.last_positions(attention_mask)]
    union, columns = torch.unique(code_ids, return_inverse=True)
    w = model.get_output_embeddings().weight.index_select(0, union)
    base = F.linear(h.float(), w.float()).gather(1, columns)
    module = getattr(model.model.language_model, "semantic_matching", None)
    if module is not None and intervention != "branch_off":
        checkpointed = any(getattr(l, "gradient_checkpointing", False) for l in model.model.language_model.layers)
        base = base + module(hidden, rows, checkpointed=checkpointed, intervention=intervention)
    return base.masked_fill(~code_mask, modeling.MASK)


def logits(model, input_ids, attention_mask, code_ids, code_mask, rows, *, intervention="active"):
    hidden = model.model.language_model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).last_hidden_state
    return from_hidden(model, hidden, attention_mask, code_ids, code_mask, rows, intervention=intervention)


def multi_logits(model, input_ids, attention_mask, code_ids, code_mask, rows, interventions=INTERVENTIONS):
    """Diagnostics share the EXACT encoder states; ordinary inference uses logits."""
    hidden = model.model.language_model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).last_hidden_state
    return {mode: from_hidden(model, hidden, attention_mask, code_ids, code_mask, rows, intervention=mode)
            for mode in interventions}


def save(model, tokenizer, out, cfg):
    out = Path(out)
    text = model.model.language_model
    module = getattr(text, "semantic_matching", None)
    if module is None:
        modeling.save(model, tokenizer, out, cfg)
        return
    del text.semantic_matching
    try:
        modeling.save(model, tokenizer, out, cfg)
    finally:
        text.add_module("semantic_matching", module)
    save_file({n: v.detach().cpu().contiguous() for n, v in module.state_dict().items()}, str(out / "semantic_matching.safetensors"))
    spec = {"kind": "residual_semantic_matching", "version": 1, "arm": module.arm,
            "heads": module.heads, "layers": module.layers, "width": module.width,
            "backbone_calls": 1, "initial_correction": "exactly_zero", "native_code_head_retained": True}
    (out / "decision_architecture.json").write_text(json.dumps(spec, indent=2) + "\n")
    (out / "README.md").write_text("# Residual semantic matching decision model\n\n"
        "Research checkpoint. Load with qwen_decision.v3.semantic_matching.load and score with its logits, "
        "including the exact semantic span metadata. The stock engine does not execute this readout.\n")


def load(path, *, device="cuda"):
    path = Path(path)
    cfg = modeling.read_config(path)
    model = modeling.load(path, attention=cfg["attention"], deltanet=cfg["deltanet"], device=device)
    file = path / "decision_architecture.json"
    if file.exists():
        spec = json.loads(file.read_text())
        if spec["kind"] != "residual_semantic_matching" or spec["version"] != 1:
            raise ValueError("wrong architecture loader")
        module = install(model.model.language_model, spec["arm"], heads=spec["heads"], layers=spec["layers"])
        module.load_state_dict(load_file(str(path / "semantic_matching.safetensors")), strict=True)
    return model
