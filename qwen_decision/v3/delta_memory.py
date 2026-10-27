"""Identity-initialized, token-wise retrieval of earlier Transformer block deltas.

An adaptation of Delta Attention Residuals, not a reproduction of its training.
Every four native hybrid layers form a block. A zero-gated depth router adds
selected earlier block updates before the next block. All 24 native layers
execute once; no option/readout change or extra sequence-attention pass.

The tensor carried between groups includes its delta bank. Checkpoint replay
therefore has explicit tensor inputs, with no mutable forward-hook memory.
"""
from contextlib import contextmanager
import json
from pathlib import Path

import torch
from torch import nn
from transformers.modeling_layers import GradientCheckpointingLayer
from safetensors.torch import load_file, save_file

from qwen_decision.v3 import modeling


class DeltaBlock(GradientCheckpointingLayer):
    block_type = "full_attention"  # the outer loop dispatches a (full, recurrent) mask pair

    def __init__(self, layers, index, groups, hidden):
        super().__init__()
        self.native = nn.ModuleList(layers)
        self.index, self.groups, self.hidden = index, groups, hidden
        if index < groups-1:
            device = next(layers[0].parameters()).device
            self.query = nn.Parameter(torch.zeros(hidden,device=device,dtype=torch.float32))
            self.gate = nn.Parameter(torch.zeros(hidden,device=device,dtype=torch.float32))

    def forward(self, hidden_states, position_embeddings, attention_mask=None,
                position_ids=None, past_key_values=None, use_cache=False, **kwargs):
        if use_cache or past_key_values is not None or kwargs.get("cu_seq_lens_q") is not None:
            raise ValueError("delta memory supports padded noncausal execution without cache")
        if not isinstance(attention_mask,tuple) or len(attention_mask)!=2:
            raise ValueError("missing full/recurrent mask pair")
        full, recurrent = attention_mask
        shape = hidden_states.shape[:-1]
        states = hidden_states.reshape(*shape,self.groups+1,self.hidden)
        entry = states[...,0,:]
        h = entry
        for layer in self.native:
            h = layer(h,position_embeddings=position_embeddings,
                      attention_mask=full if layer.block_type=="full_attention" else recurrent,
                      position_ids=position_ids,past_key_values=None,use_cache=False,**kwargs)
        # Updates, not accumulated hidden states, are the retrieval sources.
        delta = (h.float()-entry.float()).to(h.dtype)
        bank = [states[...,i+1,:] for i in range(self.index)]+[delta]
        if self.index < self.groups-1:
            values = torch.stack([torch.zeros_like(delta)]+bank,dim=0).float()
            keys = values*torch.rsqrt(values.square().mean(-1,keepdim=True)+1e-6)
            scores = (keys*self.query).sum(-1)
            weights = scores.softmax(0)
            selected = (weights[...,None]*values).sum(0)
            h = (h.float()+self.gate.tanh()*selected).to(h.dtype)
        # Future slots remain zero. The bank stays differentiable and is part
        # of the checkpoint input/output, not module state between requests.
        bank += [states[...,i+1,:] for i in range(self.index+1,self.groups)]
        return torch.stack([h]+bank,dim=-2).flatten(-2)


def _pack(module,args,kwargs):
    if args or kwargs.get("past_key_values") is not None or kwargs.get("use_cache"):
        raise ValueError("delta memory requires keyword, uncached inputs")
    if kwargs.get("cu_seq_lens_q") is not None:
        raise ValueError("delta memory does not support packing")
    x = kwargs.get("inputs_embeds")
    masks = kwargs.get("attention_mask")
    if x is None or x.shape[-1]!=module.config.hidden_size or not isinstance(masks,dict):
        raise ValueError("install delta memory after enable_noncausal")
    full = masks["full_attention"]
    if full is None or full.ndim!=4 or full.shape[1:3]!=(1,1):
        raise ValueError("expected a noncausal key padding mask")
    n = len(module.layers)
    kwargs["inputs_embeds"] = torch.cat([x]+[torch.zeros_like(x) for _ in range(n)],dim=-1)
    kwargs["attention_mask"] = {**masks,"full_attention":(full,masks["linear_attention"])}
    return args,kwargs


def _unpack(module,args):
    return (args[0][...,:module.weight.shape[0]],)+args[1:]


def install(text,block_size=4):
    if hasattr(text,"_decision_delta_memory"):
        raise ValueError("delta memory already installed")
    if text.config._attn_implementation!="sdpa" or len(text.layers)%block_size:
        raise ValueError("requires padded SDPA and complete equal-size blocks")
    original = list(text.layers)
    if [x.block_type for x in original]!=list(text.config.layer_types):
        raise ValueError("native layer dispatch differs from config")
    n = len(original)//block_size
    if n<2: raise ValueError("need at least two blocks")
    spec = {"kind":"gated_block_delta_memory","version":1,"block_size":block_size,
            "original_layer_types":list(text.config.layer_types),"groups":n,
            "null_source":"fixed zero","key_norm":"parameter-free RMS, eps1e-6",
            "gate":"per-channel tanh, zero initialized"}
    text.layers = nn.ModuleList([DeltaBlock(original[i*block_size:(i+1)*block_size],i,n,
                                           text.config.hidden_size) for i in range(n)])
    text.config.layer_types = ["full_attention"]*n
    text.config.num_hidden_layers = n
    text._decision_delta_memory = spec
    text._delta_memory_handles = [text.register_forward_pre_hook(_pack,with_kwargs=True),
                                   text.norm.register_forward_pre_hook(_unpack)]
    disable_native_checkpointing(text)
    return text


def disable_native_checkpointing(text):
    for block in text.layers:
        for layer in block.native: layer.gradient_checkpointing=False


@contextmanager
def native(text):
    spec = text._decision_delta_memory
    groups = text.layers
    for handle in text._delta_memory_handles: handle.remove()
    text.layers = nn.ModuleList([layer for block in groups for layer in block.native])
    text.config.layer_types = spec["original_layer_types"]
    text.config.num_hidden_layers = len(text.layers)
    try:
        yield
    finally:
        text.layers = groups
        text.config.layer_types = ["full_attention"]*len(groups)
        text.config.num_hidden_layers = len(groups)
        text._delta_memory_handles = [text.register_forward_pre_hook(_pack,with_kwargs=True),
                                       text.norm.register_forward_pre_hook(_unpack)]


def routing_parameters(text):
    return {f"{i}.{name}":p for i,block in enumerate(text.layers)
            for name,p in block.named_parameters(recurse=False)}


def save(model,tokenizer,out,config):
    out=Path(out);text=model.model.language_model
    if not hasattr(text,"_decision_delta_memory"):
        modeling.save(model,tokenizer,out,config);return
    spec=dict(text._decision_delta_memory)
    params=routing_parameters(text)
    with native(text): modeling.save(model,tokenizer,out,config)
    save_file({k:p.detach().cpu().contiguous() for k,p in params.items()},str(out/"delta_memory.safetensors"))
    (out/"decision_architecture.json").write_text(json.dumps(spec,indent=2)+"\n")
    (out/"README.md").write_text(
        "# Block-delta Decision Transformer\n\n"
        "Load with `qwen_decision.v3.delta_memory.load(path)` or the repository's "
        "`qwen_decision.v3.engine.Scorer`. Normal `modeling.code_logits` works after restore. "
        "Keep decision_architecture.json and delta_memory.safetensors with stock weights. "
        "Plain Transformers loading omits depth routing.\n")


def restore_architecture(model,path):
    path=Path(path);spec=json.loads((path/"decision_architecture.json").read_text())
    if spec["kind"]!="gated_block_delta_memory" or spec["version"]!=1:
        raise ValueError("unknown delta-memory architecture")
    text=model.model.language_model
    install(text,spec["block_size"])
    if text._decision_delta_memory!=spec: raise ValueError("architecture metadata differs")
    parameters=routing_parameters(text);state=load_file(str(path/"delta_memory.safetensors"))
    if state.keys()!=parameters.keys(): raise ValueError("routing parameter keys differ")
    with torch.no_grad():
        for name,value in state.items(): parameters[name].copy_(value)
    return model


def load(path,*,device="cuda"):
    path=Path(path);cfg=modeling.read_config(path)
    model=modeling.load(path,attention=cfg["attention"],deltanet=cfg["deltanet"],device=device)
    if (path/"decision_architecture.json").exists(): restore_architecture(model,path)
    return model
