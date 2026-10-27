"""DUMA-inspired token matching after ONE joint backbone encoding.

Unlike the paper's candidate-by-candidate encoder calls, all choices share the
unchanged d3 prompt and one Qwen forward. The matching MHA is shared across
both directions and two layers. No residual gate bypasses the new readout.
The self-attention control has exactly the same parameters and initialization.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.utils.checkpoint import checkpoint
from safetensors.torch import load_file, save_file

from qwen_decision.v3 import format as fmt, modeling


def content_spans(renderer, raw, encoded):
    """Resolve semantic spans against the EXACT cached order/code assignment.

    Token offsets can include a preceding space. Select tokens overlapping the
    content span, excluding special tokens, as format.encode_options does.
    Choice codes and section headings are never explicitly selected.
    """
    if renderer.args.layout != "d3":
        raise ValueError("this experiment fixes the existing d3 layout")
    question = raw["question"]
    if question["type"] == "choice":
        keys = list(question["criteria"])
        question = {**question, "criteria": {keys[i]: question["criteria"][keys[i]]
                                             for i in encoded["order"]}}
    lookup = dict(zip(renderer.code_ids, renderer.codes))
    labels = [lookup[t] for t in encoded["code_ids"]]
    user, options = fmt._user_prompt_spans(raw["state"], question, labels, "d3")
    state = fmt.describe(raw["state"]) if raw["state"] not in (None, "") else "(empty)"
    asked = fmt.describe(question.get("instructions") or fmt.DEFAULT_INSTRUCTIONS)
    state_start = len("State:\n")
    question_start = state_start + len(state) + len("\n\nQuestion:\n")
    if not user.startswith("State:\n" + state + "\n\nQuestion:\n" + asked + "\n\nOptions:\n"):
        raise ValueError("unexpected prompt structure")
    text = fmt.render(renderer.tokenizer, raw["state"], question, labels, "d3")
    start = text.rfind(user)
    if start < 0:
        raise ValueError("chat template must preserve user content")
    tokens = renderer.tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    if tokens["input_ids"] != encoded["input_ids"]:
        raise ValueError("span reconstruction changed the matched prompt")
    special = set(renderer.tokenizer.all_special_ids)

    def positions(lo, hi):
        result = [i for i, ((a, b), token) in enumerate(zip(tokens["offset_mapping"], tokens["input_ids"]))
                  if b > a and a < start + hi and b > start + lo and token not in special]
        if not result:
            raise ValueError(f"empty content token span: {raw['id']} {lo}:{hi}")
        return result

    return {"passage_positions": positions(state_start, state_start + len(state)),
            "question_positions": positions(question_start, question_start + len(asked)),
            "candidate_positions": [positions(a, b) for a, b in options]}


class DualMatching(nn.Module):
    def __init__(self, width, *, heads=8, layers=2, mode="dual", device=None, dtype=None):
        super().__init__()
        if mode not in ("dual", "self") or width % heads or layers < 1:
            raise ValueError("invalid matching configuration")
        self.width, self.heads, self.layers, self.mode = width, heads, layers, mode
        self.q = nn.Linear(width, width, bias=False, device=device, dtype=dtype)
        self.k = nn.Linear(width, width, bias=False, device=device, dtype=dtype)
        self.v = nn.Linear(width, width, bias=False, device=device, dtype=dtype)
        self.o = nn.Linear(width, width, bias=False, device=device, dtype=dtype)
        self.scorer = nn.Linear(2 * width, 1, bias=False, device=device, dtype=dtype)
        for layer in (self.q, self.k, self.v, self.o, self.scorer):
            nn.init.normal_(layer.weight, std=.02)

    def attention(self, query, context, query_mask, key_mask):
        b, t, d = query.shape
        q = self.q(query).view(b, t, self.heads, d // self.heads).transpose(1, 2)
        k = self.k(context).view(b, context.shape[1], self.heads, d // self.heads).transpose(1, 2)
        v = self.v(context).view(b, context.shape[1], self.heads, d // self.heads).transpose(1, 2)
        # Explicit math backend makes checkpoint/restart evolution deterministic.
        with sdpa_kernel(SDPBackend.MATH):
            value = F.scaled_dot_product_attention(q, k, v, attn_mask=key_mask[:, None, None, :],
                                                   dropout_p=0., is_causal=False)
        value = value.transpose(1, 2).reshape(b, t, d)
        return self.o(value) * query_mask[:, :, None]

    def match(self, passage, qa, passage_mask, qa_mask):
        for _ in range(self.layers):
            if self.mode == "dual":
                # Compute BOTH directions from the same previous-layer states.
                p = self.attention(passage, qa, passage_mask, qa_mask)
                q = self.attention(qa, passage, qa_mask, passage_mask)
                passage, qa = p, q
            else:
                size = passage.shape[1]
                together = torch.cat((passage, qa), dim=1)
                mask = torch.cat((passage_mask, qa_mask), dim=1)
                together = self.attention(together, together, mask, mask)
                passage, qa = together[:, :size], together[:, size:]
        p = (passage.float() * passage_mask[:, :, None]).sum(1) / passage_mask.sum(1)[:, None]
        q = (qa.float() * qa_mask[:, :, None]).sum(1) / qa_mask.sum(1)[:, None]
        return F.linear(torch.cat((p, q), -1), self.scorer.weight.float()).squeeze(-1)

    def forward(self, hidden, rows, *, checkpointed=False, disable_evidence=False):
        results = []
        for x, row in zip(hidden, rows):
            passage = x[row["passage_positions"]][None]
            if disable_evidence:
                passage = torch.zeros_like(passage)
            question = row["question_positions"]
            choices = row["candidate_positions"]
            if len(choices) != len(row["target"]):
                raise ValueError("candidate span/target cardinality mismatch")
            # Bound matching activations independently of option count/length.
            chunk = max(1, min(8, 2048 // (len(row["passage_positions"]) +
                                          len(question) + max(map(len, choices)))))
            scores = []
            for at in range(0, len(choices), chunk):
                indices = [question + choice for choice in choices[at:at + chunk]]
                width = max(map(len, indices))
                positions = torch.tensor([p + [0] * (width - len(p)) for p in indices], device=x.device)
                mask = torch.arange(width, device=x.device)[None] < torch.tensor(
                    [len(p) for p in indices], device=x.device)[:, None]
                qa = x[positions] * mask[:, :, None]
                p = passage.expand(len(indices), -1, -1)
                pmask = torch.ones(p.shape[:2], dtype=torch.bool, device=x.device)
                if self.training and checkpointed:
                    value = checkpoint(self.match, p, qa, pmask, mask, use_reentrant=False)
                else:
                    value = self.match(p, qa, pmask, mask)
                scores.append(value)
            results.append(torch.cat(scores))
        width = max(map(len, results))
        return torch.stack([F.pad(r, (0, width - len(r)), value=modeling.MASK) for r in results])


def install(text, mode="dual", heads=8, layers=2):
    if hasattr(text, "dual_matching"):
        raise ValueError("matching already installed")
    weight = text.embed_tokens.weight
    module = DualMatching(weight.shape[-1], heads=heads, layers=layers, mode=mode,
                          device=weight.device, dtype=weight.dtype)
    text.add_module("dual_matching", module)
    return module


def logits(model, ids, mask, rows, *, disable_evidence=False):
    text = model.model.language_model
    hidden = text(input_ids=ids, attention_mask=mask, use_cache=False).last_hidden_state
    checkpointed = any(getattr(l, "gradient_checkpointing", False) for l in text.layers)
    return text.dual_matching(hidden, rows, checkpointed=checkpointed, disable_evidence=disable_evidence)


def save(model, tokenizer, out, cfg):
    out = Path(out)
    text = model.model.language_model
    module = text.dual_matching
    del text.dual_matching
    try:
        modeling.save(model, tokenizer, out, cfg)
    finally:
        text.add_module("dual_matching", module)
    save_file({k: v.detach().cpu().contiguous() for k, v in module.state_dict().items()},
              str(out / "dual_matching.safetensors"))
    spec = {"kind": "joint_dual_matching", "version": 1, "mode": module.mode,
            "heads": module.heads, "layers": module.layers, "width": module.width,
            "backbone_calls": 1, "shared_directions_and_layers": True}
    (out / "decision_architecture.json").write_text(json.dumps(spec, indent=2) + "\n")
    (out / "README.md").write_text("# Joint token matching decision model\n\n"
        "Load with qwen_decision.v3.dual_matching.load and call its logits with content spans. "
        "One joint Qwen encoding; an adaptation of DUMA, not its per-candidate encoding recipe. "
        "The stock engine does not execute the matching readout.\n")


def restore_architecture(model, path):
    path = Path(path)
    spec = json.loads((path / "decision_architecture.json").read_text())
    if spec["kind"] != "joint_dual_matching" or spec["version"] != 1:
        raise ValueError("unknown matching architecture")
    module = install(model.model.language_model, spec["mode"], spec["heads"], spec["layers"])
    if module.width != spec["width"]:
        raise ValueError("matching width differs")
    module.load_state_dict(load_file(str(path / "dual_matching.safetensors")), strict=True)
    return model


def load(path, *, device="cuda"):
    path = Path(path)
    cfg = modeling.read_config(path)
    model = modeling.load(path, attention=cfg["attention"], deltanet=cfg["deltanet"], device=device)
    if (path / "decision_architecture.json").exists():
        restore_architecture(model, path)
    return model
