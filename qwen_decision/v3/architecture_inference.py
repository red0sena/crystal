"""Run a frozen architecture-search specification with one backbone forward per batch."""
from pathlib import Path

import torch
from safetensors.torch import load_file

from qwen_decision.v3 import architecture as arch, modeling


class Scorer:
    def __init__(self, manifest, candidate=None):
        config=manifest['config']
        self.model=modeling.load(Path(manifest['args']['base']),attention=config['attention'],
                                deltanet=config['deltanet'],kernel='sdpa').eval().requires_grad_(False)
        self.code_ids=manifest['code_ids']
        self.head,self.collector,self.handles=None,None,None
        self.uses_options=False
        text=self.model.model.language_model
        if candidate is None:
            return
        family=candidate['family']
        if family in ('readout','contrast','options'):
            weight=self.model.get_output_embeddings().weight[self.code_ids]
            if family=='options':
                self.uses_options=True
                self.head=arch.OptionReadout(candidate['kind'],weight).cuda()
                self.head.load_state_dict(load_file(str(Path(candidate['directory'])/'head.safetensors')))
            elif family=='contrast':
                self.head=arch.ContrastReadout(weight,candidate['window'],candidate['alpha'],candidate['mode']).cuda()
            else:
                self.head=arch.Readout(candidate['kind'],weight,len(text.layers)).cuda()
                self.head.load_state_dict(load_file(str(Path(candidate['directory'])/'head.safetensors')))
            self.head.eval().requires_grad_(False)
            if not self.uses_options and (family=='contrast' or candidate['kind'] not in ('code','ffn')):
                self.collector=arch.SummaryCollector(text)
        elif family=='surgery':
            arch.apply_layer_order(text,list(text.layers),candidate['order'])
        elif family=='gates':
            parameters,self.handles=(arch.install_inner_scales(text) if candidate['kind']=='ia3'
                else arch.install_branch_scales(text,channels=candidate['channels']))
            state=load_file(str(Path(candidate['directory'])/'scales.safetensors'))
            with torch.no_grad():
                for k,p in parameters.items():
                    p.copy_(state[k].to(p.device))
        else:
            raise ValueError(family)

    @torch.no_grad()
    def logits(self,batch):
        if self.head is not None:
            if self.collector is not None:
                base,features=self.collector(batch)
            else:
                h=self.model.model.language_model(input_ids=batch['input_ids'],attention_mask=batch['attention_mask'],
                                                  use_cache=False).last_hidden_state
                base=h[torch.arange(len(h),device=h.device),batch['attention_mask'].sum(-1)-1]
                features=h[torch.arange(len(h),device=h.device)[:,None],batch['positions']] if self.uses_options else None
            return self.head(base,features,batch['columns'],batch['valid'])
        codes=torch.tensor(self.code_ids,device=batch['columns'].device)[batch['columns']]
        return modeling.code_logits(self.model,batch['input_ids'],batch['attention_mask'],codes,batch['valid'])

    def close(self):
        if self.collector is not None:
            self.collector.close()
        if self.handles is not None:
            arch.remove_branch_scales(self.model.model.language_model,self.handles)
