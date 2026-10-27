"""A weight-shared, gated extra pass through an internal hybrid Transformer block.

This changes the executed network, not the output head or a model mixture.
The checkpoint's layers remain the only owners of their shared parameters.
One extra pass is a finite-depth adaptation, not a reproduction of Huginn.
"""
from __future__ import annotations

import json
from pathlib import Path
import weakref

import torch
from torch import nn
from safetensors.torch import load_file, save_file

from qwen_decision.v3 import modeling


class RecurrentPass(nn.Module):
    block_type = "full_attention"

    def __init__(self, layers, hidden_size):
        super().__init__()
        # Registering the shared modules again would duplicate checkpoint keys.
        self._layers = tuple(weakref.ref(layer) for layer in layers)
        weight = next(layers[0].parameters())
        self.recurrence_gate = nn.Parameter(torch.zeros(hidden_size, device=weight.device, dtype=torch.float32))

    def forward(self, hidden_states, *, attention_mask=None, past_key_values=None, use_cache=False, **kwargs):
        if use_cache or past_key_values is not None:
            raise ValueError("recurrent decision blocks require use_cache=False")
        if kwargs.get("cu_seq_lens_q") is not None:
            raise ValueError("this experiment uses padded SDPA, not packed rows")
        if attention_mask is not None and (attention_mask.ndim != 4 or attention_mask.dtype != torch.bool
                                           or attention_mask.shape[1:3] != (1, 1)):
            raise ValueError("expected noncausal [B,1,1,T] boolean key mask")
        original = hidden_states
        padding = attention_mask[:, 0, 0] if attention_mask is not None else None
        for reference in self._layers:
            layer = reference()
            if layer is None:
                raise RuntimeError("shared block no longer exists")
            mask = attention_mask if layer.block_type == "full_attention" else padding
            hidden_states = layer(hidden_states, attention_mask=mask, past_key_values=None,
                                  use_cache=False, **kwargs)
        # FP32 subtraction/multiply avoids rounding a small gate away in BF16.
        return (original.float() + self.recurrence_gate.tanh() *
                (hidden_states.float() - original.float())).to(original.dtype)


def install(text, start=16, stop=20):
    """Run [start:stop] once normally, then once more before the remaining coda."""
    if hasattr(text, "_decision_recurrence"):
        raise ValueError("recurrence already installed")
    if not 0 <= start < stop <= len(text.layers):
        raise ValueError("invalid recurrent block range")
    if text.config._attn_implementation != "sdpa":
        raise ValueError("this recurrence experiment requires padded noncausal SDPA")
    original_types = list(text.config.layer_types)
    if [layer.block_type for layer in text.layers] != original_types:
        raise ValueError("layer types and executed layers disagree")
    loop = RecurrentPass(list(text.layers[start:stop]), text.config.hidden_size)
    original = list(text.layers)
    text.layers = nn.ModuleList(original[:stop] + [loop] + original[stop:])
    text.config.layer_types = original_types[:stop] + ["full_attention"] + original_types[stop:]
    text.config.num_hidden_layers = len(text.layers)
    text._decision_recurrence = {"kind": "gated_shared_internal_block", "start": start, "stop": stop,
                                 "extra_passes": 1, "original_layer_types": original_types}
    return loop


def loop_module(text):
    spec = text._decision_recurrence
    return text.layers[spec["stop"]]


def save(model, tokenizer, out, config):
    """Stock weights plus an explicit architecture manifest and the learned gate."""
    out = Path(out)
    text = model.model.language_model
    if not hasattr(text, "_decision_recurrence"):
        modeling.save(model, tokenizer, out, config)
        return
    spec = dict(text._decision_recurrence)
    loop = loop_module(text)
    modified_layers, modified_types = text.layers, text.config.layer_types
    text.layers = nn.ModuleList([layer for layer in text.layers if layer is not loop])
    text.config.layer_types = spec["original_layer_types"]
    text.config.num_hidden_layers = len(text.layers)
    try:
        modeling.save(model, tokenizer, out, config)
        save_file({"recurrence_gate": loop.recurrence_gate.detach().cpu().contiguous()},
                  str(out / "recurrence.safetensors"))
        (out / "decision_architecture.json").write_text(json.dumps(spec, indent=2) + "\n")
        (out / "README.md").write_text(
            "# Recurrent Decision model\n\n"
            "This checkpoint executes an additional shared internal Transformer block.\n"
            "Load with `qwen_decision.v3.recurrent.load(path)` or the repository's "
            "`qwen_decision.v3.engine.Scorer`; both restore the saved computation graph.\n\n"
            "Plain Transformers loading reads only the stock weights and omits the extra path. "
            "Keep decision_architecture.json and recurrence.safetensors with the weights.\n"
        )
    finally:
        text.layers, text.config.layer_types = modified_layers, modified_types
        text.config.num_hidden_layers = len(modified_layers)


def restore_architecture(model, path):
    """Attach the saved computation graph to already loaded stock weights."""
    path = Path(path)
    manifest = path / "decision_architecture.json"
    if manifest.exists():
        spec = json.loads(manifest.read_text())
        if spec["kind"] != "gated_shared_internal_block" or spec["extra_passes"] != 1:
            raise ValueError("unknown decision architecture")
        loop = install(model.model.language_model, spec["start"], spec["stop"])
        loop.load_state_dict(load_file(str(path / "recurrence.safetensors")), strict=True)
    return model


def load(path, *, device="cuda"):
    path = Path(path)
    config = modeling.read_config(path)
    model = modeling.load(path, attention=config["attention"], deltanet=config["deltanet"], device=device)
    return restore_architecture(model, path)
