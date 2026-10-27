"""Replace upper Qwen blocks with a shared positive associative-memory core.

BDH-GPU's positive feature/memory/coactivation equations are adapted from
https://github.com/pathwaycom/bdh (MIT, Copyright 2025 Pathway Technology,
Inc.; see wiki/resources/bdh-license.txt). This is a noncausal retrofit,
not the proprietary BDH-CQ system. BDH's paired feature RoPE is applied
before associative retrieval. Every refinement rebuilds request-local memory
and excludes each token's self association, with no mutable request cache.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from safetensors.torch import load_file,save_file
from transformers.masking_utils import create_recurrent_attention_mask

from qwen_decision.v3 import modeling


def norm(x):
    return F.layer_norm(x.float(),(x.shape[-1],)).to(x.dtype)


def feature_rope(features):
    n = features.shape[-1]
    if n%2:raise ValueError('even per-head feature width required')
    frequency = 65536.**(-(torch.arange(n,device=features.device,dtype=torch.float32)//2*2)/n)
    phase = torch.arange(features.shape[-2],device=features.device,dtype=torch.float32)[:,None]*frequency
    rotated = torch.stack((-features[...,1::2],features[...,::2]),-1).flatten(-2)
    return features*phase.cos().to(features.dtype)+rotated*phase.sin().to(features.dtype)


def associate(features,values,mask):
    """(positive QK^T with diagonal removed) V, without a T-by-T matrix.

    features: B,H,T,N; values: B,1,T,D. Padded keys contribute zero.
    Length scaling is independent of queries and is followed by LayerNorm.
    """
    keys = features*mask[:,None,:,None]
    memory = keys.transpose(-1,-2)@values
    read = features@memory
    self_weight = (features*keys).sum(-1,keepdim=True)
    read = read-self_weight*values
    count = (mask.sum(-1)-1).clamp_min(1).to(read.dtype)[:,None,None,None]
    return read/count


class MemoryCore(nn.Module):
    def __init__(self,mlp,start,heads=4,cycles=3):
        super().__init__()
        intermediate,hidden = mlp.up_proj.weight.shape
        if intermediate%heads or (intermediate//heads)%2 or cycles<1:
            raise ValueError('invalid memory heads/cycles')
        self.start,self.heads,self.cycles = start,heads,cycles
        # Reuse one removed layer's pretrained projections; no zero gate.
        self.encoder = nn.Parameter(mlp.up_proj.weight.detach().clone())
        self.encoder_v = nn.Parameter(mlp.gate_proj.weight.detach().clone())
        self.decoder = nn.Parameter(mlp.down_proj.weight.detach().clone())

    def step(self,x,mask,memory_enabled=True):
        b,t,d = x.shape;h = self.heads;n = self.encoder.shape[0]//h
        phi = F.relu(F.linear(x,self.encoder)).view(b,t,h,n).transpose(1,2)
        if memory_enabled:
            read = norm(associate(feature_rope(phi),x[:,None],mask))
        else:
            read = x.new_zeros((b,h,t,d))
        weight = self.encoder_v.view(h,n,d)
        psi = F.relu(torch.einsum('bhtd,hnd->bhtn',read,weight))
        coactive = (phi*psi).transpose(1,2).reshape(b,t,h*n)
        update = norm(F.linear(coactive,self.decoder))
        return norm(x+update)*mask[:,:,None]

    def forward(self,x,mask,*,memory_enabled=True,checkpointed=False):
        x = norm(x)*mask[:,:,None]
        for _ in range(self.cycles):
            if self.training and checkpointed:
                x = checkpoint(self.step,x,mask,memory_enabled,use_reentrant=False)
            else:
                x = self.step(x,mask,memory_enabled)
        return x


def install(text,start=16,heads=4,cycles=3):
    if hasattr(text,'bdh_memory') or not 0<start<len(text.layers):
        raise ValueError('invalid/already installed memory core')
    if text.config._attn_implementation!='sdpa':
        raise ValueError('padded SDPA required')
    core = MemoryCore(text.layers[start].mlp,start,heads,cycles)
    text.add_module('bdh_memory',core)
    return core


def hidden(model,input_ids,attention_mask,*,memory_enabled=True):
    text = model.model.language_model;core = text.bdh_memory
    if input_ids.ndim!=2 or attention_mask.shape!=input_ids.shape:
        raise ValueError('right-padded text only, no cache/packing')
    if not attention_mask[:,0].all() or (attention_mask[:,1:]>attention_mask[:,:-1]).any():
        raise ValueError('nonempty right-padded text required')
    x = text.embed_tokens(input_ids)
    b,t = input_ids.shape
    pos = torch.arange(t,device=x.device)[None].expand(b,-1)
    rope = text.rotary_emb(x,pos[None].expand(3,-1,-1))
    full = attention_mask[:,None,None,:].bool()
    recurrent = create_recurrent_attention_mask(config=text.config,inputs_embeds=x,
                                               attention_mask=attention_mask,past_key_values=None)
    for layer in text.layers[:core.start]:
        x = layer(x,position_embeddings=rope,
                  attention_mask=full if layer.block_type=='full_attention' else recurrent,
                  position_ids=pos,past_key_values=None,use_cache=False)
    x = core(x,attention_mask.bool(),memory_enabled=memory_enabled,
             checkpointed=any(getattr(l,'gradient_checkpointing',False) for l in text.layers[:core.start]))
    return text.norm(x)


def logits(model,input_ids,attention_mask,code_ids,code_mask,*,memory_enabled=True):
    h = hidden(model,input_ids,attention_mask,memory_enabled=memory_enabled)
    h = h[torch.arange(len(h),device=h.device),modeling.last_positions(attention_mask)]
    union,columns = torch.unique(code_ids,return_inverse=True)
    w = model.get_output_embeddings().weight.index_select(0,union).float()
    return F.linear(h.float(),w).gather(1,columns).masked_fill(~code_mask,modeling.MASK)


def save(model,tokenizer,out,config):
    out = Path(out);text = model.model.language_model;core = text.bdh_memory
    del text.bdh_memory
    try:
        modeling.save(model,tokenizer,out,config)
    finally:
        text.add_module('bdh_memory',core)
    save_file({k:v.detach().cpu().contiguous() for k,v in core.state_dict().items()},str(out/'bdh_memory.safetensors'))
    spec = {'kind':'bdh_memory_replacement','version':1,'start':core.start,'heads':core.heads,
            'cycles':core.cycles,'position':'bdh_feature_rope',
            'attention':'bidirectional_positive_kernel_no_self','init':'removed_start_mlp'}
    (out/'decision_architecture.json').write_text(json.dumps(spec,indent=2)+'\n')
    (out/'README.md').write_text('# BDH-inspired decision core\n\n'
        'Load with qwen_decision.v3.bdh_memory.load and call its logits(). '
        'Native blocks from start onward are retained for provenance but are not executed. '
        'Plain Transformers or the standard engine do not implement this architecture. '
        'Not a reproduction of proprietary BDH-CQ.\n')


def restore_architecture(model,path):
    path = Path(path);s = json.loads((path/'decision_architecture.json').read_text())
    if (s['kind']!='bdh_memory_replacement' or s['version']!=1
            or s['attention']!='bidirectional_positive_kernel_no_self'
            or s['position']!='bdh_feature_rope' or s['init']!='removed_start_mlp'):
        raise ValueError('unknown BDH memory architecture')
    core = install(model.model.language_model,s['start'],s['heads'],s['cycles'])
    core.load_state_dict(load_file(str(path/'bdh_memory.safetensors')),strict=True)
    return model


def load(path,*,device='cuda'):
    path = Path(path);cfg = modeling.read_config(path)
    model = modeling.load(path,attention=cfg['attention'],deltanet=cfg['deltanet'],device=device)
    if (path/'decision_architecture.json').exists():
        restore_architecture(model,path)
    return model
