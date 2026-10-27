"""Gold-only, two-view adaptation with mergeable LoRA and safe option alignment."""
from contextlib import contextmanager
import math

import torch
from torch import nn
from torch.nn import functional as F

from qwen_decision.v3 import format as fmt


class LowRankLinear(nn.Module):
    def __init__(self, base, rank=8):
        super().__init__()
        self.base = base.requires_grad_(False)
        self.a = nn.Parameter(torch.empty(rank, base.in_features, device=base.weight.device))
        self.b = nn.Parameter(torch.zeros(base.out_features, rank, device=base.weight.device))
        nn.init.kaiming_uniform_(self.a, a=math.sqrt(5))
        self.scale = 2.0  # LoRA alpha=2*rank

    def forward(self, x):
        update = F.linear(F.linear(x.float(), self.a), self.b) * self.scale
        return self.base(x) + update.to(x.dtype)


def install(text, rank=8):
    text.requires_grad_(False)
    names = []
    for name, module in list(text.named_modules()):
        if isinstance(module, nn.Linear):
            parent, _, leaf = name.rpartition('.')
            setattr(text.get_submodule(parent), leaf, LowRankLinear(module, rank))
            names.append(name)
    if not names:
        raise ValueError('no linear projections to adapt')
    return names


def adapter_state(text):
    return {name: p.detach().cpu().contiguous() for name, p in text.named_parameters() if p.requires_grad}


def restore(text, state):
    parameters = {name: p for name, p in text.named_parameters() if p.requires_grad}
    if parameters.keys() != state.keys():
        raise ValueError('adapter parameter names differ')
    with torch.no_grad():
        for name, value in state.items():
            parameters[name].copy_(value)


@contextmanager
def merged(text, coefficient=1.0, states=None):
    """Evaluate actual merged BF16 weights; restore exact training weights afterwards."""
    changed = []
    try:
        for name, module in list(text.named_modules()):
            if not isinstance(module, LowRankLinear):
                continue
            parent, _, leaf = name.rpartition('.')
            owner = text.get_submodule(parent)
            original = module.base.weight.detach().clone()
            changed.append((owner, leaf, module, original))
            with torch.no_grad():
                if states is None:
                    delta = module.b @ module.a
                else:
                    # Average updates BA, not independently averaged low-rank factors.
                    delta = sum(s[name + '.b'].to(original.device) @ s[name + '.a'].to(original.device)
                                for s in states) / len(states)
                delta = delta * (module.scale * coefficient)
                module.base.weight.copy_(original.float() + delta)
            setattr(owner, leaf, module.base)
        yield
    finally:
        for owner, leaf, module, original in reversed(changed):
            with torch.no_grad():
                module.base.weight.copy_(original)
            setattr(owner, leaf, module)


def view(renderer, example, order=None, codes=None):
    """Order maps displayed columns to the original semantic option indices."""
    k = len(example['target'])
    order = list(range(k)) if order is None else list(order)
    if sorted(order) != list(range(k)):
        raise ValueError('order must be a permutation')
    fixed = example['ordered'] or renderer.positional(example) or example['question']['type'] != 'choice'
    if fixed and order != list(range(k)):
        raise ValueError('cannot permute ordered, positional, or noul options')
    question = example['question']
    if question['type'] == 'choice':
        keys = list(question['criteria'])
        question = {**question, 'criteria': {keys[i]: question['criteria'][keys[i]] for i in order}}
    assigned = list(range(k)) if codes is None else list(codes)
    if len(assigned) != k or len(set(assigned)) != k:
        raise ValueError('codes must be distinct')
    prompt = fmt.render(renderer.tokenizer, example['state'], question,
                        [renderer.codes[i] for i in assigned], renderer.args.layout)
    return {'id': example['id'], 'source': example['source'],
            'input_ids': renderer.tokenizer.encode(prompt, add_special_tokens=False),
            'code_ids': [renderer.code_ids[i] for i in assigned],
            'target': [example['target'][i] for i in order], 'order': order,
            'group': example.get('group', example['id'])}


def random_view(renderer, example, rng):
    k = len(example['target'])
    order, codes = list(range(k)), list(range(k))
    fixed = example['ordered'] or renderer.positional(example) or example['question']['type'] != 'choice'
    if not fixed:
        rng.shuffle(order)
        if rng.random() < .2:
            rng.shuffle(codes)
    return view(renderer, example, order, codes)


def cyclic_views(renderer, example):
    k = len(example['target'])
    fixed = example['ordered'] or renderer.positional(example) or example['question']['type'] != 'choice'
    shifts = [0] if fixed else sorted({0, k // 4, k // 2, 3 * k // 4})
    indices = list(range(k))
    return [view(renderer, example, indices[s:] + indices[:s]) for s in shifts]


def align(logits, encoded):
    inverse = []
    for row in encoded:
        order, k = row['order'], len(row['order'])
        inverse.append([order.index(i) for i in range(k)] + list(range(k, logits.shape[1])))
    return logits.gather(1, torch.tensor(inverse, device=logits.device))


def jensen_shannon(left, right):
    log_p, log_q = left.log_softmax(-1), right.log_softmax(-1)
    log_m = torch.logaddexp(log_p, log_q) - math.log(2)
    return .5 * ((log_p.exp() * (log_p - log_m)).sum(-1)
                 + (log_q.exp() * (log_q - log_m)).sum(-1))
