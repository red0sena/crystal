"""Middle-layer hypothesis tokens with bidirectional evidence write-back.

This is a research architecture, not a reproduction of TRM or LLoCoT. After
an encoder prefix, one centered, content-conditioned slot per candidate joins
the native full-attention blocks. Context and slots both change. Causal
DeltaNet blocks update context only; slots use their shared pointwise MLP.
Thus the workspace introduces no artificial left-to-right candidate recurrence.
All request state is passed as tensors; checkpoint replay has no mutable hooks.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from safetensors.torch import load_file, save_file
from transformers.masking_utils import create_recurrent_attention_mask

from qwen_decision.v3 import modeling


class HypothesisWorkspace(nn.Module):
    def __init__(self, hidden, start, device, scale=.5):
        super().__init__()
        self.start, self.scale = start, scale
        self.projection = nn.Linear(hidden, hidden, bias=False, device=device, dtype=torch.float32)
        # The new route is active on the first forward pass, not zero gated.
        with torch.no_grad():
            self.projection.weight.copy_(torch.eye(hidden, device=device))

    def seed(self, hidden, last, positions, valid):
        options = hidden.gather(1, positions[..., None].expand(-1, -1, hidden.shape[-1])).float()
        mean = (options * valid[..., None]).sum(1, keepdim=True) / valid.sum(1)[:, None, None]
        contrast = options - mean
        contrast = contrast * torch.rsqrt(contrast.square().mean(-1, keepdim=True) + 1e-6)
        base = hidden[torch.arange(len(hidden), device=hidden.device), last].float()[:, None]
        magnitude = base.square().mean(-1, keepdim=True).sqrt()
        slots = base + self.scale * magnitude * self.projection(contrast)
        return (slots * valid[..., None]).to(hidden.dtype)


def install(text, start=16, scale=.5):
    if hasattr(text, 'hypothesis_workspace'):
        raise ValueError('workspace already installed')
    if not 0 < start < len(text.layers) or text.config._attn_implementation != 'sdpa':
        raise ValueError('workspace requires a middle-layer boundary and padded SDPA')
    if not 0 < scale <= 1:
        raise ValueError('workspace seed scale must be in (0,1]')
    workspace = HypothesisWorkspace(text.config.hidden_size, start,
                                    text.embed_tokens.weight.device, scale)
    text.add_module('hypothesis_workspace', workspace)
    return workspace


def _slot_ffn(layer, slots):
    return slots + layer.mlp(layer.post_attention_layernorm(slots))


def hidden(model, input_ids, attention_mask, option_positions, valid, *,
           enabled=True, write_back=True, return_workspace=False):
    text = model.model.language_model
    ws = text.hypothesis_workspace
    if input_ids.ndim != 2 or attention_mask.shape != input_ids.shape:
        raise ValueError('workspace supports right-padded text, no packing or cache')
    if option_positions.shape != valid.shape or not valid.any(-1).all():
        raise ValueError('candidate position/validity mismatch')
    if torch.any(attention_mask[:, 1:] > attention_mask[:, :-1]):
        raise ValueError('workspace requires right padding')
    if torch.any(valid & ((option_positions < 0) | (option_positions >= attention_mask.sum(-1)[:, None]))):
        raise ValueError('candidate points outside real input')
    b, t = input_ids.shape
    h = text.embed_tokens(input_ids)
    pos = torch.arange(t, device=h.device)[None].expand(b, -1)
    rope = text.rotary_emb(h, pos[None].expand(3, -1, -1))
    full = attention_mask[:, None, None, :].bool()
    recurrent = create_recurrent_attention_mask(config=text.config, inputs_embeds=h,
                                               attention_mask=attention_mask, past_key_values=None)
    last = modeling.last_positions(attention_mask)
    for i, layer in enumerate(text.layers):
        if enabled and i == ws.start:
            slots = ws.seed(h, last, option_positions.masked_fill(~valid, 0), valid)
            h = torch.cat([h, slots], 1)
            joined_pos = torch.cat([pos, last[:, None].expand_as(valid)], 1)
            joined_rope = text.rotary_emb(h, joined_pos[None].expand(3, -1, -1))
            keys = torch.cat([attention_mask.bool(), valid], 1)
            joined_mask = keys[:, None, None, :]
            if not write_back:
                # Causal intervention: the context cannot read workspace K/V.
                joined_mask = joined_mask.expand(-1, 1, t + valid.shape[1], -1).clone()
                joined_mask[:, :, :t, t:] = False
        active = enabled and i >= ws.start
        if not active or layer.block_type == 'full_attention':
            h = layer(h, position_embeddings=joined_rope if active else rope,
                      attention_mask=joined_mask if active else
                                     (full if layer.block_type == 'full_attention' else recurrent),
                      position_ids=joined_pos if active else pos,
                      past_key_values=None, use_cache=False)
        else:
            context = layer(h[:, :t], position_embeddings=rope, attention_mask=recurrent,
                            position_ids=pos, past_key_values=None, use_cache=False)
            slots = h[:, t:]
            if layer.training and getattr(layer, 'gradient_checkpointing', False):
                slots = checkpoint(_slot_ffn, layer, slots, use_reentrant=False)
            else:
                slots = _slot_ffn(layer, slots)
            h = torch.cat([context, slots], 1)
    result = text.norm(h[:, :t])
    return (result, h[:, t:]) if return_workspace else result


def logits(model, input_ids, attention_mask, code_ids, code_mask, option_positions, **kwargs):
    h = hidden(model, input_ids, attention_mask, option_positions, code_mask, **kwargs)
    last = h[torch.arange(len(h), device=h.device), modeling.last_positions(attention_mask)]
    union, columns = torch.unique(code_ids, return_inverse=True)
    weight = model.get_output_embeddings().weight.index_select(0, union).float()
    return F.linear(last.float(), weight).gather(1, columns).masked_fill(~code_mask, modeling.MASK)


def save(model, tokenizer, out, config):
    out = Path(out)
    text = model.model.language_model
    ws = text.hypothesis_workspace
    # Keep stock keys/config usable and persist the actual architecture separately.
    del text.hypothesis_workspace
    try:
        modeling.save(model, tokenizer, out, config)
    finally:
        text.add_module('hypothesis_workspace', ws)
    save_file({k:v.detach().cpu().contiguous() for k,v in ws.state_dict().items()},
              str(out/'hypothesis_workspace.safetensors'))
    spec = {'kind':'middle_hypothesis_workspace', 'version':1, 'start':ws.start,
            'scale':ws.scale, 'slot_mixer':'full_attention_and_shared_native_ffn',
            'position':'original_decision_token', 'write_back':True}
    (out/'decision_architecture.json').write_text(json.dumps(spec, indent=2)+'\n')
    (out/'README.md').write_text('# Experimental hypothesis workspace\n\n'
        'Load using qwen_decision.v3.hypothesis_workspace.load and score using its logits(). '
        'Plain Transformers and the standard Decision engine do not restore this prototype. '
        'Keep both architecture and workspace files beside the stock weights.\n')


def restore_architecture(model, path):
    path = Path(path)
    spec = json.loads((path/'decision_architecture.json').read_text())
    if (spec['kind'] != 'middle_hypothesis_workspace' or spec['version'] != 1
            or spec['slot_mixer'] != 'full_attention_and_shared_native_ffn'
            or spec['position'] != 'original_decision_token' or spec['write_back'] is not True):
        raise ValueError('unknown workspace architecture')
    ws = install(model.model.language_model, spec['start'], spec['scale'])
    ws.load_state_dict(load_file(str(path/'hypothesis_workspace.safetensors')), strict=True)
    return model


def load(path, *, device='cuda'):
    path = Path(path)
    cfg = modeling.read_config(path)
    model = modeling.load(path, attention=cfg['attention'], deltanet=cfg['deltanet'], device=device)
    if (path/'decision_architecture.json').exists():
        restore_architecture(model, path)
    return model
