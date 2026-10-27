"""Independent decision readouts; no vocabulary-sized LM projection or generation."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from qwen_decision.v3.pointer import MASK

KINDS = ('code', 'linear', 'mlp', 'warm_mlp', 'option_linear', 'option_mlp')


class DecisionReadout(nn.Module):
    def __init__(self, kind, code_weight, inner=512):
        super().__init__()
        if kind not in KINDS:
            raise ValueError(kind)
        self.kind, self.inner = kind, inner
        classes, hidden = code_weight.shape
        self.uses_options = kind.startswith('option_')
        if kind in ('code', 'linear', 'warm_mlp'):
            self.output = nn.Linear(hidden, classes, bias=False, dtype=torch.float32)
            if kind != 'linear':
                with torch.no_grad():
                    self.output.weight.copy_(code_weight.detach().float())
        if kind in ('mlp', 'warm_mlp', 'option_mlp'):
            self.up = nn.Linear(hidden, inner, dtype=torch.float32)
            width = hidden if kind == 'warm_mlp' else (1 if self.uses_options else classes)
            self.down = nn.Linear(inner, width, bias=False, dtype=torch.float32)
            if kind == 'warm_mlp':
                nn.init.zeros_(self.down.weight)
        if kind == 'option_linear':
            self.output = nn.Linear(hidden, 1, bias=False, dtype=torch.float32)

    def forward(self, decision, options, columns, valid):
        if self.uses_options:
            if options is None:
                raise ValueError('option readout needs contextual option vectors')
            x = F.layer_norm(options.float(), (options.shape[-1],))
            scores = (self.output(x) if self.kind == 'option_linear'
                      else self.down(F.gelu(self.up(x)))).squeeze(-1)
        else:
            x = decision.float()
            if self.kind == 'linear':
                x = F.layer_norm(x, (x.shape[-1],))
            if self.kind == 'mlp':
                scores = self.down(F.gelu(self.up(F.layer_norm(x, (x.shape[-1],)))))
            else:
                if self.kind == 'warm_mlp':
                    x = x + self.down(F.gelu(self.up(F.layer_norm(x, (x.shape[-1],)))))
                scores = self.output(x)
            scores = scores.gather(1, columns)
        return scores.masked_fill(~valid, MASK)


def vectors(text, batch):
    """Exactly one text backbone call; read final and per-option content positions."""
    h = text(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], use_cache=False).last_hidden_state
    rows = torch.arange(len(h), device=h.device)
    final = h[rows, batch['attention_mask'].sum(-1)-1]
    options = h[rows[:, None], batch['positions']]
    return final, options


def live_logits(model, head, batch):
    final, options = vectors(model.model.language_model, batch)
    return head(final, options if head.uses_options else None, batch['columns'], batch['valid'])


def cache_batch(cache, indices, uses_options):
    """Gather ragged candidate vectors without allocating N x 255 x hidden."""
    index = torch.as_tensor(indices, dtype=torch.long, device=cache['base'].device)
    width = int(cache['lengths'][index].max())
    valid = torch.arange(width, device=index.device)[None] < cache['lengths'][index, None]
    opts = None
    if uses_options:
        offsets = cache['offsets'][index, None] + torch.arange(width, device=index.device)[None]
        offsets = offsets.masked_fill(~valid, 0)
        opts = cache['options'][offsets]
    return {'decision': cache['base'][index], 'options': opts,
            'columns': cache['columns'][index, :width], 'valid': valid,
            'targets': cache['targets'][index, :width]}


def cached_logits(head, batch):
    return head(batch['decision'], batch['options'], batch['columns'], batch['valid'])
