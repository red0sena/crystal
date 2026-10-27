"""Prototype: spend the final Transformer layer on decision queries only.

The preceding hybrid layers still encode the full, unchanged prompt once.
The last full-attention block reads full K/V memory but computes Q, output
projection and FFN for either one decision query or K candidate queries.
Candidate queries start as exact copies of the original decision state;
a zero-output adapter can learn candidate-specific evidence retrieval.

Saved checkpoints retain the stock weights, plus an adapter and architecture
manifest. Load through this module or the repository's decision engine.
"""
from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from safetensors.torch import load_file, save_file
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    ALL_ATTENTION_FUNCTIONS, apply_rotary_pos_emb, eager_attention_forward,
)

from qwen_decision.v3 import modeling


def gather(hidden, positions):
    return hidden.gather(1, positions[..., None].expand(-1, -1, hidden.shape[-1]))


def query_block(layer, memory, queries, positions, position_embeddings, attention_mask):
    """The native final block, with distinct query and K/V sequence lengths."""
    attn = layer.self_attn
    norm_memory = layer.input_layernorm(memory)
    norm_queries = layer.input_layernorm(queries)
    b, n, _ = queries.shape
    t = memory.shape[1]
    q, gate = torch.chunk(attn.q_proj(norm_queries).view(b, n, -1, 2*attn.head_dim), 2, -1)
    gate = gate.reshape(b, n, -1)
    q = attn.q_norm(q.reshape(b,n,-1,attn.head_dim)).transpose(1,2)
    k = attn.k_norm(attn.k_proj(norm_memory).view(b,t,-1,attn.head_dim)).transpose(1,2)
    v = attn.v_proj(norm_memory).view(b,t,-1,attn.head_dim).transpose(1,2)
    cos, sin = position_embeddings
    # A decision query retains the original decision token's RoPE position,
    # even when it represents a particular candidate. K/V positions stay native.
    qc, qs = gather(cos, positions), gather(sin, positions)
    q, _ = apply_rotary_pos_emb(q,q,qc,qs)
    k, _ = apply_rotary_pos_emb(k,k,cos,sin)
    interface = ALL_ATTENTION_FUNCTIONS.get_interface(attn.config._attn_implementation,
                                                    eager_attention_forward)
    output,_ = interface(attn,q,k,v,attention_mask,dropout=attn.attention_dropout if layer.training else 0.,
                         scaling=attn.scaling)
    output = output.reshape(b,n,-1).contiguous()*gate.sigmoid()
    hidden = queries+attn.o_proj(output)
    return hidden+layer.mlp(layer.post_attention_layernorm(hidden))


class DecisionTail(nn.Module):
    block_type = "full_attention"

    def __init__(self, native_layer, hidden_size, rank=64):
        super().__init__()
        if native_layer.block_type != "full_attention":
            raise ValueError("the final decoder block must be full attention")
        self.native = native_layer
        weight = next(native_layer.parameters())
        self.up = nn.Linear(3*hidden_size,rank,bias=False,device=weight.device,dtype=torch.float32)
        self.down = nn.Linear(rank,hidden_size,bias=False,device=weight.device,dtype=torch.float32)
        nn.init.zeros_(self.down.weight)
        self.request = None
        self.result = None

    def forward(self, hidden_states, position_embeddings, attention_mask=None,
                past_key_values=None, use_cache=False, **kwargs):
        if self.request is None:
            raise ValueError("DecisionTail must be called through decision_tail.logits")
        if use_cache or past_key_values is not None or kwargs.get("cu_seq_lens_q") is not None:
            raise ValueError("prototype supports padded noncausal inference, no cache/packing")
        if attention_mask is None or attention_mask.ndim != 4 or attention_mask.shape[1:3] != (1,1):
            raise ValueError("expected noncausal key-only padding mask")
        last, option_positions, valid, mode = self.request
        base = gather(hidden_states,last[:,None])
        if mode == "single":
            queries, positions = base, last[:,None]
        elif mode == "candidates":
            options = gather(hidden_states,option_positions)
            shared = base.expand_as(options)
            # Nonlinear joint conditioning: a common additive question query
            # would disappear under candidate-relative attention centering.
            a = F.layer_norm(shared.float(),(shared.shape[-1],))
            c = F.layer_norm(options.float(),(options.shape[-1],))
            update = self.down(F.silu(self.up(torch.cat([a,c,a*c],-1))))
            queries = (shared.float()+update*valid[...,None]).to(hidden_states.dtype)
            positions = last[:,None].expand_as(option_positions)
        else:
            raise ValueError(mode)
        self.result = query_block(self.native,hidden_states,queries,positions,
                                  position_embeddings,attention_mask)
        # The public backbone output is intentionally unused by logits().
        # Keep its shape/normalization contract without manufacturing full
        # context-token Q/output/FFN results that will never be read.
        return hidden_states


def install(text, rank=64):
    if isinstance(text.layers[-1],DecisionTail):
        raise ValueError("decision tail already installed")
    if text.config._attn_implementation != "sdpa":
        raise ValueError("prototype requires noncausal padded SDPA")
    tail = DecisionTail(text.layers[-1],text.config.hidden_size,rank)
    text.layers[-1] = tail
    return tail


@contextmanager
def native(text):
    """Use the untouched native block for the exact same weights and prompt."""
    tail = text.layers[-1]
    if not isinstance(tail,DecisionTail):
        raise ValueError("no decision tail")
    text.layers[-1] = tail.native
    try:
        yield
    finally:
        text.layers[-1] = tail


def logits(model,input_ids,attention_mask,code_ids,code_mask,option_positions,mode="candidates"):
    text = model.model.language_model
    tail = text.layers[-1]
    if not isinstance(tail,DecisionTail) or tail.request is not None:
        raise ValueError("missing or already-active decision tail")
    tail.request = (modeling.last_positions(attention_mask),option_positions,code_mask,mode)
    try:
        text(input_ids=input_ids,attention_mask=attention_mask,use_cache=False)
        hidden = text.norm(tail.result)
        union,columns = torch.unique(code_ids,return_inverse=True)
        weight = model.get_output_embeddings().weight.index_select(0,union).float()
        if mode == "single":
            scores = F.linear(hidden[:,0].float(),weight)
            scores = scores.gather(1,columns)
        else:
            selected = weight[columns]
            scores = (hidden.float()*selected).sum(-1)
        return scores.masked_fill(~code_mask,modeling.MASK)
    finally:
        tail.request = tail.result = None


def save(model, tokenizer, out, config, mode="candidates"):
    out = Path(out)
    tail = model.model.language_model.layers[-1]
    if not isinstance(tail, DecisionTail):
        modeling.save(model, tokenizer, out, config)
        return
    if mode not in {"single", "candidates"}:
        raise ValueError(mode)
    with native(model.model.language_model):
        modeling.save(model, tokenizer, out, config)
    save_file({"up.weight": tail.up.weight.detach().cpu().contiguous(),
               "down.weight": tail.down.weight.detach().cpu().contiguous()},
              str(out / "decision_tail.safetensors"))
    spec = {"kind": "candidate_query_tail", "rank": tail.up.out_features,
            "mode": mode, "position": "original_decision_token", "version": 1}
    (out / "decision_architecture.json").write_text(json.dumps(spec, indent=2)+"\n")
    (out / "README.md").write_text(
        "# Candidate-query Decision Transformer\n\n"
        "Load with `qwen_decision.v3.decision_tail.load(path)` and call "
        "`decision_tail.logits`, or use `qwen_decision.v3.engine.Scorer`. "
        "Keep decision_tail.safetensors and decision_architecture.json with the stock weights. "
        "Plain Transformers loading omits the modified final-block computation.\n")


def restore_architecture(model, path):
    spec = json.loads((Path(path)/"decision_architecture.json").read_text())
    if (spec["kind"] != "candidate_query_tail" or spec["version"] != 1
            or spec["mode"] not in {"single", "candidates"}
            or spec["position"] != "original_decision_token"):
        raise ValueError("unknown decision-tail architecture")
    tail = install(model.model.language_model, spec["rank"])
    state = load_file(str(Path(path)/"decision_tail.safetensors"))
    tail.up.load_state_dict({"weight": state["up.weight"]}, strict=True)
    tail.down.load_state_dict({"weight": state["down.weight"]}, strict=True)
    tail.mode = spec["mode"]
    return model


def load(path, *, device="cuda"):
    path = Path(path)
    cfg = modeling.read_config(path)
    model = modeling.load(path, attention=cfg["attention"], deltanet=cfg["deltanet"], device=device)
    if (path/"decision_architecture.json").exists():
        restore_architecture(model, path)
    return model
