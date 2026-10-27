"""Identity-initialized Talking-Heads retrofit of the last two full attentions.

Shazeer et al., arXiv:2003.02436. Qwen's Q/K normalization, rotary positions,
output gate, projections, DeltaNet and tied code readout are untouched. All
trial arms use the SAME explicit FP32 attention arithmetic at selected layers.
The native arm is therefore an arithmetic-matched native control, not a claim
of bitwise identity with stock fused SDPA. No autoregressive decoding or loops.
"""
from contextlib import contextmanager
import json
from pathlib import Path

import torch
from torch import nn
from safetensors.torch import load_file, save_file
from transformers import AttentionInterface
from transformers.integrations.sdpa_attention import sdpa_attention_forward

from qwen_decision.v3 import modeling

KERNEL = "decision_talking_heads_v1"
KIND = "upper_two_talking_heads_v1"
ARMS = ("native", "diagonal", "talking")
MODES = ("active", "no_cross", "identity")
INTERVENTIONS = MODES


class HeadMix(nn.Module):
    def __init__(self, heads, arm, *, device=None):
        super().__init__()
        if arm not in ARMS:
            raise ValueError(arm)
        self.arm, self.mode = arm, "active"
        self.register_buffer("eye", torch.eye(heads, device=device, dtype=torch.float32), persistent=False)
        if arm != "native":
            self.pre = nn.Parameter(self.eye.clone())
            self.post = nn.Parameter(self.eye.clone())

    def matrices(self):
        if self.arm == "native" or self.mode == "identity":
            return self.eye, self.eye
        if self.arm == "diagonal" or self.mode == "no_cross":
            return self.pre * self.eye, self.post * self.eye
        return self.pre, self.post

    def forward(self, q, k, v, key_mask, scale):
        if key_mask is None or key_mask.dtype != torch.bool or key_mask.ndim != 4:
            raise ValueError("noncausal boolean padding mask required")
        if not bool(key_mask.any(-1).all()):
            raise ValueError("attention row has no valid key")
        groups = q.shape[1] // k.shape[1]
        if q.shape[1] % k.shape[1] or k.shape != v.shape:
            raise ValueError("invalid grouped-query attention")
        k, v = k.repeat_interleave(groups, 1), v.repeat_interleave(groups, 1)
        # Mask AFTER head mixing: mixing masked -inf entries is undefined.
        scores = (q.float() @ k.float().transpose(-1, -2)) * scale
        pre, post = self.matrices()
        scores = torch.einsum("bhij,hg->bgij", scores, pre.float())
        weights = scores.masked_fill(~key_mask, -torch.inf).softmax(-1)
        weights = torch.einsum("bhij,hg->bgij", weights, post.float())
        return (weights @ v.float()).transpose(1, 2).to(q.dtype).contiguous()


def attention(module, query, key, value, attention_mask, dropout=0., scaling=None, **kwargs):
    mix = getattr(module, "head_mix", None)
    if mix is None:
        return sdpa_attention_forward(module, query, key, value, attention_mask,
                                      dropout=dropout, scaling=scaling, **kwargs)
    if dropout or kwargs.get("cu_seq_lens_q") is not None:
        raise ValueError("this trial requires no dropout and padded independent rows")
    return mix(query, key, value, attention_mask, scaling or query.shape[-1] ** -.5), None


AttentionInterface.register(KERNEL, attention)


def install(text, arm):
    if hasattr(text, "_talking_heads"):
        raise ValueError("already installed")
    full = [i for i, layer in enumerate(text.layers) if layer.block_type == "full_attention"]
    if len(full) < 2:
        raise ValueError("need two full attention layers")
    selected = full[-2:]
    for i in selected:
        module = text.layers[i].self_attn
        module.add_module("head_mix", HeadMix(text.config.num_attention_heads, arm,
                                             device=module.q_proj.weight.device))
    text.config._attn_implementation = KERNEL
    text._talking_heads = {"kind": KIND, "arm": arm, "layers": selected,
        "heads": text.config.num_attention_heads, "initialization": "pre=post=identity_fp32",
        "arithmetic": "FP32 qk, head mixing, masked softmax, head mixing and weighted values; BF16 output",
        "native_control": "same explicit arithmetic, identity matrices; other four attentions stock SDPA"}
    return text._talking_heads


def modules(text):
    return [text.layers[i].self_attn.head_mix for i in text._talking_heads["layers"]]


@contextmanager
def using_mode(text, mode):
    if mode not in MODES:
        raise ValueError(mode)
    mm = modules(text); previous = [m.mode for m in mm]
    try:
        for m in mm:
            m.mode = mode
        yield
    finally:
        for m, old in zip(mm, previous):
            m.mode = old


intervention = using_mode


def summary(text):
    result = {}
    for i, m in zip(text._talking_heads["layers"], modules(text)):
        if m.arm == "native":
            continue
        for name in ("pre", "post"):
            w = getattr(m, name).detach().float()
            result[f"{i}.{name}"] = {"offdiag_l2": float((w * (1 - m.eye)).norm()),
                "diagonal_change_l2": float((w.diagonal() - 1).norm())}
    return result


def logits(model, input_ids, attention_mask, code_ids, code_mask, rows=None, *, intervention="active"):
    if not hasattr(model.model.language_model, "_talking_heads"):
        if intervention != "active":
            raise ValueError("stock reference only has active inference")
        return modeling.code_logits(model, input_ids, attention_mask, code_ids, code_mask)
    with using_mode(model.model.language_model, intervention):
        return modeling.code_logits(model, input_ids, attention_mask, code_ids, code_mask)


def multi_logits(model, input_ids, attention_mask, code_ids, code_mask, rows=None, interventions=MODES):
    # These interventions change INTERNAL layers. Each needs its OWN one-pass
    # inference. Unlike the old readout trial, sharing final hidden states would
    # invalidate the intervention. No outputs are combined to make a decision.
    return {mode: logits(model, input_ids, attention_mask, code_ids, code_mask, intervention=mode)
            for mode in interventions}


def load(path, *, arm=None, device="cuda"):
    path = Path(path)
    model = modeling.load(path, attention="noncausal", deltanet="causal", device=device)
    spec_path = path / "decision_architecture.json"
    if spec_path.exists():
        spec = json.loads(spec_path.read_text())
        if spec["kind"] != KIND or arm is not None and spec["arm"] != arm:
            raise ValueError("architecture mismatch")
        arm = spec["arm"]
    if arm is None:
        raise ValueError("specify arm for a stock checkpoint")
    actual = install(model.model.language_model, arm)
    if spec_path.exists():
        if actual != spec:
            raise ValueError("saved architecture differs")
        expected = {n: p for n, p in model.named_parameters() if ".head_mix." in n}
        state = load_file(str(path / "talking_heads.safetensors"))
        if set(expected) != set(state) or any(state[k].shape != p.shape for k, p in expected.items()):
            raise ValueError("incomplete head mix weights")
        with torch.no_grad():
            for k, p in expected.items():
                p.copy_(state[k])
    return model


def save(model, tokenizer, out, config):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    text = model.model.language_model
    spec = text._talking_heads
    state = {n: p.detach().cpu().contiguous() for n, p in model.named_parameters() if ".head_mix." in n}
    held = [(text.layers[i].self_attn, text.layers[i].self_attn.head_mix) for i in spec["layers"]]
    try:
        for module, _ in held:
            del module.head_mix
        text.config._attn_implementation = "sdpa"
        modeling.save(model, tokenizer, out, config | {"attention": "noncausal", "deltanet": "causal"})
    finally:
        text.config._attn_implementation = KERNEL
        for module, mix in held:
            module.add_module("head_mix", mix)
    save_file(state, str(out / "talking_heads.safetensors"))
    (out / "decision_architecture.json").write_text(json.dumps(spec, indent=2) + "\n")
    (out / "README.md").write_text("# Talking-Heads experiment\n\nLoad using qwen_decision.v3.talking_heads.load. Plain Transformers omits the head mixing.\n")
