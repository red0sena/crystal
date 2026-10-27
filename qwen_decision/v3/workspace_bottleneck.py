"""Require decisions to leave through candidate workspace states.

Uses the middle-layer read/write workspace but removes the direct readout from
the original context decision token. Each final candidate state is scored with
its own existing code row. The native head is shared, not an ensemble.
"""
import json
from pathlib import Path

from safetensors.torch import load_file

from qwen_decision.v3 import hypothesis_workspace as workspace, modeling

install = workspace.install


def logits(model,input_ids,attention_mask,code_ids,code_mask,option_positions):
    _,slots = workspace.hidden(model,input_ids,attention_mask,option_positions,code_mask,
                               return_workspace=True)
    states = model.model.language_model.norm(slots).float()
    weight = model.get_output_embeddings().weight[code_ids].float()
    return (states*weight).sum(-1).masked_fill(~code_mask,modeling.MASK)


def save(model,tokenizer,out,config):
    out = Path(out)
    workspace.save(model,tokenizer,out,config)
    spec = json.loads((out/'decision_architecture.json').read_text())
    spec.update(kind='candidate_workspace_bottleneck',readout='diagonal_code_row_on_candidate_state')
    (out/'decision_architecture.json').write_text(json.dumps(spec,indent=2)+'\n')
    (out/'README.md').write_text('# Experimental candidate workspace bottleneck\n\n'
        'Use qwen_decision.v3.workspace_bottleneck.load and its logits(). '
        'The context decision-token readout has been removed. '
        'Plain Transformers or hypothesis_workspace.logits do not implement this decision path.\n')


def load(path,*,device='cuda'):
    path = Path(path)
    cfg = modeling.read_config(path)
    model = modeling.load(path,attention=cfg['attention'],deltanet=cfg['deltanet'],device=device)
    if (path/'decision_architecture.json').exists():
        spec = json.loads((path/'decision_architecture.json').read_text())
        if (spec['kind']!='candidate_workspace_bottleneck' or spec['version']!=1
                or spec['readout']!='diagonal_code_row_on_candidate_state'
                or spec['position']!='original_decision_token' or spec['write_back'] is not True
                or spec['slot_mixer']!='full_attention_and_shared_native_ffn'):
            raise ValueError('unknown candidate-workspace bottleneck')
        ws = install(model.model.language_model,spec['start'],spec['scale'])
        ws.load_state_dict(load_file(str(path/'hypothesis_workspace.safetensors')),strict=True)
    return model
