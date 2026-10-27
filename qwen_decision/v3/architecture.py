"""Identity-initialized decision architectures and safe frozen-layer surgery.

These are adaptations inspired by LayerScale/CaiT, token-layer probes and
Transformer Layers as Painters, not reproductions of those papers' results.
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from qwen_decision.v3 import modeling


class SummaryCollector:
    """Capture only three summaries per layer, never a persistent L x T x D tensor."""
    def __init__(self, text):
        self.text = text
        self.handles = [layer.register_forward_hook(self._hook(i)) for i, layer in enumerate(text.layers)]
        self.batch = None
        self.values = {}

    def _hook(self, index):
        def capture(module, args, output):
            if self.batch is None:
                return
            h = output
            batch = self.batch
            rows = torch.arange(len(h), device=h.device)
            mask = batch['attention_mask']
            last = h[rows, mask.sum(-1) - 1]
            mean = (h.float() * mask[..., None]).sum(1) / mask.sum(1, keepdim=True)
            opt = h[rows[:, None], batch['positions']].float()
            opt = (opt * batch['valid'][..., None]).sum(1) / batch['valid'].sum(1, keepdim=True)
            summaries = torch.stack([last, mean.to(h.dtype), opt.to(h.dtype)], 1)
            self.values[index] = self.text.norm(summaries).detach()
        return capture

    @torch.no_grad()
    def __call__(self, batch):
        self.batch, self.values = batch, {}
        try:
            h = self.text(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'],
                          use_cache=False).last_hidden_state
            rows = torch.arange(len(h), device=h.device)
            base = h[rows, batch['attention_mask'].sum(-1) - 1]
            features = torch.stack([self.values[i] for i in range(len(self.text.layers))], 1)
            features[:, -1, 0] = base  # exact original final norm/readout, including BF16 rounding
            return base, features
        finally:
            self.batch, self.values = None, {}

    def close(self):
        for handle in self.handles:
            handle.remove()


class Readout(nn.Module):
    """Keep the checkpoint's code basis; modify its connection to internal representations."""
    def __init__(self, kind, weight, layers, inner=64):
        super().__init__()
        self.kind = kind
        self.register_buffer('weight', weight.detach().float().clone())
        hidden = weight.shape[1]
        if kind == 'code':
            self.weight = nn.Parameter(self.weight)
        elif kind in ('depth', 'pool', 'scoring'):
            self.gain = nn.Parameter(torch.zeros(()))
            if kind == 'scoring':
                self.token_score = nn.Parameter(torch.zeros(layers, hidden))
                self.layer_score = nn.Parameter(torch.zeros(hidden))
            else:
                self.mix = nn.Parameter(torch.zeros(layers if kind == 'depth' else 3))
        elif kind == 'class_attention':
            self.query = nn.Linear(hidden, inner, bias=False)
            self.key = nn.Linear(hidden, inner, bias=False)
            self.value = nn.Linear(hidden, inner, bias=False)
            self.out = nn.Linear(inner, hidden, bias=False)
            nn.init.zeros_(self.out.weight)
            self.inner = inner
        elif kind == 'ffn':
            self.up = nn.Linear(hidden,128,bias=False)
            self.down = nn.Linear(128,hidden,bias=False)
            nn.init.zeros_(self.down.weight)
        elif kind != 'base':
            raise ValueError(kind)

    def representation(self, base, features):
        base = base.float()
        if self.kind in ('base', 'code'):
            return base
        if self.kind=='ffn':
            return base+self.down(F.silu(self.up(F.layer_norm(base,(base.shape[-1],)))))
        x=features.float()
        if self.kind == 'depth':
            pooled = torch.einsum('l,bld->bd', self.mix.softmax(0), x[:, :, 0])
        elif self.kind == 'pool':
            pooled = torch.einsum('s,bsd->bd', self.mix.softmax(0), x[:, -1])
        elif self.kind == 'scoring':
            token_weights = torch.einsum('ld,blsd->bls', self.token_score, x).tanh().softmax(-1)
            layer = torch.einsum('bls,blsd->bld', token_weights, x)
            layer_weights = torch.einsum('d,bld->bl', self.layer_score, layer).tanh().softmax(-1)
            pooled = torch.einsum('bl,bld->bd', layer_weights, layer)
        else:
            x = x.flatten(1, 2)
            q, k, v = self.query(base), self.key(x), self.value(x)
            weights = (torch.einsum('bd,bnd->bn', q, k) / math.sqrt(self.inner)).softmax(-1)
            return base + self.out(torch.einsum('bn,bnd->bd', weights, v))
        return base + self.gain.tanh() * (pooled - base)

    def forward(self, base, features, columns, valid):
        h = self.representation(base, features)
        return F.linear(h, self.weight).gather(1, columns).masked_fill(~valid, modeling.MASK)


class ContrastReadout(nn.Module):
    """DoLa-inspired bounded layer contrast on ALL candidate codes, without option filtering.

    Weighted contrast s_final + alpha*(s_final-s_early), not DoLa's full vocabulary decoder.
    Dynamic premature layers are selected by JS divergence within a global late-layer window.
    """
    def __init__(self,weight,window,alpha,mode):
        super().__init__()
        self.register_buffer('weight',weight.detach().float().clone())
        self.window,self.alpha,self.mode=window,alpha,mode

    def forward(self,base,features,columns,valid):
        final=F.linear(base.float(),self.weight).gather(1,columns).masked_fill(~valid,modeling.MASK)
        window=min(self.window,features.shape[1]-1)
        h=features[:,-window-1:-1,0].float()
        early=F.linear(h,self.weight).gather(2,columns[:,None].expand(-1,window,-1))
        early=early.masked_fill(~valid[:,None],modeling.MASK)
        if self.mode=='mean':
            premature=early.mean(1)
        elif self.mode=='js':
            logp,logq=final.log_softmax(-1)[:,None],early.log_softmax(-1)
            p,q=logp.exp(),logq.exp()
            logm=((p+q)/2).clamp_min(1e-30).log()
            js=.5*((p*(logp-logm)).sum(-1)+(q*(logq-logm)).sum(-1))
            index=js.argmax(-1)
            premature=early[torch.arange(len(early),device=early.device),index]
        else:
            raise ValueError(self.mode)
        return (final+self.alpha*(final-premature)).masked_fill(~valid,modeling.MASK)


class OptionReadout(nn.Module):
    """Residual candidate scorer preserving all pretrained code logits at initialization."""
    def __init__(self,kind,weight):
        super().__init__()
        self.kind=kind
        hidden=weight.shape[1]
        self.register_buffer('weight',weight.detach().float().clone())
        if kind=='pointer-residual':
            self.query=nn.Linear(hidden,64,bias=False)
            self.key=nn.Linear(hidden,64,bias=False)
            nn.init.zeros_(self.query.weight)
        elif kind=='set-residual':
            self.input=nn.Linear(2*hidden,128)
            layer=nn.TransformerEncoderLayer(128,4,256,dropout=0.,activation='gelu',batch_first=True,norm_first=True)
            self.encoder=nn.TransformerEncoder(layer,2,enable_nested_tensor=False)
            self.out=nn.Linear(128,1)
            nn.init.zeros_(self.out.weight)
            nn.init.zeros_(self.out.bias)
        else:
            raise ValueError(kind)

    def forward(self,base,features,columns,valid):
        base,options=base.float(),features.float()
        original=F.linear(base,self.weight).gather(1,columns)
        if self.kind=='pointer-residual':
            bonus=torch.einsum('bd,bkd->bk',self.query(base),self.key(options))/8.
        else:
            x=torch.cat([base[:,None].expand(-1,options.shape[1],-1),options],-1)
            x=self.encoder(self.input(x),src_key_padding_mask=~valid)
            bonus=self.out(x).squeeze(-1)
        return (original+bonus).masked_fill(~valid,modeling.MASK)


def install_branch_scales(text, *, channels=False, bound=.2):
    """Scale attention/DeltaNet and MLP branches around 1, preserving pretrained behavior.

    Unlike scratch-trained LayerScale (near-zero scales), the origin must be ONE.
    The frozen branch is unmodified. Attention tuples retain their auxiliary outputs.
    Handles can be removed; gate tensors are exported separately from the stock weights.
    """
    parameters, handles = {}, []
    for i, layer in enumerate(text.layers):
        mixer = layer.linear_attn if layer.block_type == 'linear_attention' else layer.self_attn
        for name, module in [('mixer', mixer), ('mlp', layer.mlp)]:
            if hasattr(module, 'decision_scale'):
                raise ValueError('branch scales already installed')
            size = text.config.hidden_size if channels else 1
            gate = nn.Parameter(torch.zeros(size, device=next(module.parameters()).device, dtype=torch.float32))
            module.register_parameter('decision_scale', gate)
            parameters[f'{i}.{name}'] = gate

            def scale(mod, args, output, _bound=bound):
                value = output[0] if isinstance(output, tuple) else output
                # Construct scale in FP32 before casting the resulting activation back to BF16.
                scaled = (value.float() * (1. + _bound * mod.decision_scale.tanh())).to(value.dtype)
                return (scaled, *output[1:]) if isinstance(output, tuple) else scaled

            handles.append(module.register_forward_hook(scale))
    return parameters, handles


def remove_branch_scales(text, handles):
    for handle in handles:
        handle.remove()
    for module in text.modules():
        if hasattr(module, 'decision_scale'):
            del module.decision_scale


def install_inner_scales(text, bound=1.):
    """IA3 adaptation: K/V projection outputs of softmax layers and every FFN inner activation.

    DeltaNet recurrences stay intact. Qwen's key normalization is retained. A bounded
    1+tanh(g) scale preserves the pretrained function at g=0; no T-Few loss is used.
    """
    parameters, handles = {}, []
    for i, layer in enumerate(text.layers):
        targets = [('ffn',layer.mlp.down_proj,True)]
        if layer.block_type == 'full_attention':
            targets += [('key',layer.self_attn.k_proj,False),('value',layer.self_attn.v_proj,False)]
        for name, module, input_scale in targets:
            size = module.in_features if input_scale else module.out_features
            gate = nn.Parameter(torch.zeros(size,device=module.weight.device,dtype=torch.float32))
            module.register_parameter('decision_scale',gate)
            parameters[f'{i}.{name}'] = gate

            def before(mod, inputs, _bound=bound):
                x=inputs[0]
                value=(x.float()*(1.+_bound*mod.decision_scale.tanh())).to(x.dtype)
                return (value,*inputs[1:])

            def after(mod, inputs, output, _bound=bound):
                return (output.float()*(1.+_bound*mod.decision_scale.tanh())).to(output.dtype)

            handles.append(module.register_forward_pre_hook(before) if input_scale else module.register_forward_hook(after))
    return parameters, handles


def layer_order(depth, kind='base', index=None):
    order = list(range(depth))
    if kind == 'skip':
        del order[index]
    elif kind == 'swap':
        order[index], order[index + 1] = order[index + 1], order[index]
    elif kind != 'base':
        raise ValueError(kind)
    return order


def apply_layer_order(text, original, order):
    """Move hybrid layers AND their mask dispatch metadata together (no KV cache)."""
    text.layers = nn.ModuleList([original[i] for i in order])
    text.config.layer_types = [layer.block_type for layer in text.layers]
    text.config.num_hidden_layers = len(order)
