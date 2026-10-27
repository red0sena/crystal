"""Shared-weight reverse DeltaNet scans with an identity-preserving blend.

The original full-attention layers, FFNs and code head are unchanged. Each
DeltaNet computes F(x) and reverse(F(reverse(x))) with the same parameters;
the output is F + tanh(g) * (R - F). This is an experiment, not a new claim
about bidirectional recurrent networks. Reversal is within each real row.
"""
from contextlib import contextmanager
import json
from pathlib import Path

from safetensors.torch import load_file
import torch
from torch import nn

from qwen_decision.v3 import modeling

KIND = "gated_reverse_deltanet_v1"


def reverse_valid(x, mask):
    if mask is None:
        return x.flip(1)
    if mask.ndim != 2 or mask.shape != x.shape[:2]:
        raise ValueError("reverse scan needs a two-dimensional padding mask")
    valid = mask.bool()
    positions = torch.arange(x.shape[1], device=x.device)[None]
    lengths = valid.sum(-1, keepdim=True)
    if not torch.equal(valid, positions < lengths) or not bool((lengths > 0).all()):
        raise ValueError("reverse scan requires nonempty right-padded rows")
    indices = torch.where(valid, lengths - 1 - positions, positions)
    return x.gather(1, indices[..., None].expand_as(x))


def install(text):
    if hasattr(text, "_reverse_scan"):
        raise ValueError("reverse scan already installed")
    indices = []
    for index, layer in enumerate(text.layers):
        if layer.block_type != "linear_attention":
            continue
        module = layer.linear_attn
        if hasattr(module, "backward_gate"):
            raise ValueError("another reverse path is already installed")
        module.register_parameter("backward_gate", nn.Parameter(torch.zeros(
            text.config.hidden_size, device=next(module.parameters()).device, dtype=torch.float32)))
        original = module.forward

        def forward(hidden_states, cache_params=None, attention_mask=None,
                    _original=original, _module=module, **kwargs):
            if cache_params is not None or kwargs.get("use_cache"):
                raise ValueError("reverse decision scans do not support caches")
            if kwargs.get("cu_seq_lens_q") is not None or kwargs.get("cu_seq_lens_k") is not None:
                raise ValueError("reverse decision scans do not support packed rows")
            direct = _original(hidden_states, attention_mask=attention_mask, **kwargs)
            if not text._reverse_scan_enabled:
                return direct
            reverse = _original(reverse_valid(hidden_states, attention_mask),
                                attention_mask=attention_mask, **kwargs)
            reverse = reverse_valid(reverse, attention_mask)
            gate = _module.backward_gate.tanh()
            return (direct.float() + gate * (reverse.float() - direct.float())).to(direct.dtype)

        module.forward = forward
        indices.append(index)
    if not indices:
        raise ValueError("no DeltaNet layers")
    text._reverse_scan = {"kind": KIND, "layers": indices, "gate": "float32_tanh_per_channel",
                          "blend": "forward + tanh(gate) * (reverse - forward)",
                          "padding": "reverse_valid_tokens_only", "shared_weights": True}
    text._reverse_scan_enabled = True
    return text._reverse_scan


@contextmanager
def disabled(text):
    enabled = text._reverse_scan_enabled
    text._reverse_scan_enabled = False
    try:
        yield
    finally:
        text._reverse_scan_enabled = enabled


def gate_summary(text):
    return {str(i): {"mean_abs": float(text.layers[i].linear_attn.backward_gate.tanh().abs().mean()),
                     "max_abs": float(text.layers[i].linear_attn.backward_gate.tanh().abs().max())}
            for i in text._reverse_scan["layers"]}


def restore_architecture(model, path):
    path = Path(path)
    spec = json.loads((path / "decision_architecture.json").read_text())
    if spec.get("kind") != KIND:
        raise ValueError("unknown reverse scan architecture")
    text = model.model.language_model
    if install(text) != spec:
        raise ValueError("reverse scan manifest does not match model")
    state = load_file(str(path / modeling.GATES_NAME))
    expected = modeling.gate_state(text)
    if set(state) != set(expected) or any(state[k].shape != expected[k].shape for k in state):
        raise ValueError("reverse scan gate checkpoint is incomplete")
    missing, unexpected = text.load_state_dict(state, strict=False)
    if unexpected or any(k.endswith("backward_gate") for k in missing):
        raise ValueError("reverse scan gates failed to load")
    return model


def load(path, *, device="cuda"):
    path = Path(path)
    cfg = modeling.read_config(path)
    if cfg["attention"] != "noncausal" or cfg["deltanet"] != "causal":
        raise ValueError("reverse experiment expects native noncausal-attention/causal-DeltaNet weights")
    model = modeling.load(path, attention="noncausal", deltanet="causal", device=device)
    if (path / "decision_architecture.json").exists():
        return restore_architecture(model, path)
    install(model.model.language_model)
    return model


def save(model, tokenizer, out, config):
    out = Path(out)
    text = model.model.language_model
    modeling.save(model, tokenizer, out, config | {"attention": "noncausal", "deltanet": "causal"})
    (out / "decision_architecture.json").write_text(json.dumps(text._reverse_scan, indent=2) + "\n")
    (out / "README.md").write_text(
        "# Reverse DeltaNet decision experiment\n\n"
        "Load with qwen_decision.v3.reverse_scan.load(path). Keep the architecture manifest "
        "and backward_gates.safetensors with the stock weights. The generic decision engine "
        "rejects this unsupported architecture; plain Transformers omits the reverse path.\n")
