"""A schema-conditioned kNN residual classifier using gold training memory.

No source names or evaluation labels enter retrieval. Unknown option schemas
fall back to the original model. This is inspired by kNN-LM, not its perplexity
benchmark reproduction. Uses existing compatible frozen hidden-state caches.
"""
import argparse
from collections import defaultdict
from pathlib import Path

import torch
from torch.nn import functional as F

from qwen_decision.v3 import architecture_search as search, modeling, train
from qwen_decision.v3 import pointer_experiment as pilot, readout_experiment as previous
from qwen_decision.v3 import robust_adaptation as robust, robust_experiment as study

CACHE = Path('runs/readout/search-v1')


def schema(row):
    if row['ordered'] or row['question']['type']!='choice':
        return None
    # Include the complete instructions, descriptions and displayed key order.
    # A token code such as A must never be treated as a common semantic label
    # across unrelated questions or differently ordered classification schemas.
    q=row['question']
    return search.digest([q.get('instructions'),list(q['criteria'].items())])


def make_memory(raw, vectors, targets):
    by=defaultdict(list)
    for i,row in enumerate(raw):
        key=schema(row)
        if key is not None:by[key].append(i)
    memory={}
    for key,indices in by.items():
        if len(indices)<32:continue
        k=len(raw[indices[0]]['target'])
        labels=targets[indices,:k].float()
        if len(labels.argmax(-1).unique())<2:continue
        x=vectors[indices].float()
        center=x.mean(0)
        memory[key]={'center':center,'vectors':F.normalize(x-center,dim=-1),'targets':labels}
    return memory


def probabilities(memory, examples, vectors, neighbors, temperature=.05):
    result={}
    by=defaultdict(list)
    for i,row in enumerate(examples):
        if schema(row) in memory:by[schema(row)].append(i)
    for key,indices in by.items():
        m=memory[key]
        x=F.normalize(vectors[indices].float()-m['center'],dim=-1)
        similarity=x @ m['vectors'].T
        scores,nearest=similarity.topk(min(neighbors,len(m['vectors'])),-1)
        probs=(scores.div(temperature).softmax(-1).unsqueeze(-1)*m['targets'][nearest]).sum(1)
        for index,p in zip(indices,probs):result[examples[index]['id']]=p.tolist()
    return result


def records(examples,base,knn,weight):
    result=[]
    for row in examples:
        p=base[row['id']]['probabilities']
        if row['id'] in knn:p=[(1-weight)*a+weight*b for a,b in zip(p,knn[row['id']])]
        pred=max(range(len(p)),key=p.__getitem__)
        result.append({k:row[k] for k in ('id','source','group')} |
            {'prediction':pred,'correct':row['target'][pred]>=max(row['target'])-1e-6,'probabilities':p,'memory_supported':row['id'] in knn})
    return result


@torch.no_grad()
def live_vectors(model,renderer,examples):
    encoded=[robust.view(renderer,row) for row in examples]
    features=torch.empty(len(encoded),1024,dtype=torch.bfloat16)
    index={row['id']:i for i,row in enumerate(examples)}
    model.eval()
    for group in train.microbatches(encoded,study.BATCH):
        ids,mask,_,_,_=train.collate(group,renderer.tokenizer.pad_token_id)
        hidden=model.model.language_model(input_ids=ids,attention_mask=mask,use_cache=False).last_hidden_state
        x=hidden[torch.arange(len(group),device=ids.device),modeling.last_positions(mask)].cpu()
        for row,value in zip(group,x):features[index[row['id']]]=value
    return features


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['plan','screen','confirm'])
    p.add_argument('--output',type=Path,default=study.OUT)
    args=p.parse_args();out=args.output
    torch.set_num_threads(4)
    if args.stage=='plan':
        if (out/'memory-plan.json').exists():raise ValueError('preserve existing plan')
        if (out/'confirmation-started.json').exists():raise ValueError('confirmation already started')
        train.atomic_json(out/'memory-plan.json',{'neighbors':[8,32],'weight':[.1,.25,.5],'temperature':.05,
            'memory':'exact displayed question schema; >=32 training examples, >=2 gold classes; centered normalized hidden vectors',
            'teacher':'gold targets only, no new model-generated labels','unknown_schema':'original model probabilities unchanged',
            'candidate_count':6,'confirmation_gate':'best selection macro at least +.25 pp over original; otherwise no fourth confirm arm',
            'joint_confirmation':'if included, four frozen arms use Bonferroni 98.75% two-sided document intervals',
            'implementation_sha256':train.sha256_file(Path(__file__))})
        return
    protocol=study.read(out/'memory-plan.json')
    if train.sha256_file(Path(__file__))!=protocol['implementation_sha256']:raise ValueError('memory implementation changed')
    manifest=study.read(out/'manifest.json');study.verify(out,manifest)
    if args.stage=='screen':
        if (out/'memory-frozen.json').exists() or (out/'confirmation-started.json').exists():raise ValueError('screen already frozen or confirmation accessed')
        lookup={r['id']:r for r in pilot.read_rows(Path('data/mix/v3-gold3.jsonl'),32768)}
        encoded=previous.rows(CACHE/'train.jsonl')
        raw=[lookup[r['id']] for r in encoded]
        cached=torch.load(CACHE/'train.pt',map_location='cpu',weights_only=True)
        if len(raw)!=len(cached['base']):raise ValueError('cache alignment differs')
        memory=make_memory(raw,cached['base'],cached['targets'])
        torch.save(memory,out/'gold-memory.pt')
        examples=previous.rows(out/'selection.jsonl')
        cache=torch.load(CACHE/'selection.pt',map_location='cpu',weights_only=True)
        old=previous.rows(CACHE/'selection.jsonl')
        if [r['id'] for r in old]!=[r['id'] for r in examples]:raise ValueError('selection cache ids differ')
        # Check the reused features against the actual baseline predictions.
        weight=torch.load(CACHE/'code-weight.pt',map_location='cpu',weights_only=True)
        z=F.linear(cache['base'].float(),weight.float()).gather(1,cache['columns'])
        valid=torch.arange(z.shape[1])[None]<cache['lengths'][:,None]
        pred=z.masked_fill(~valid,-1e9).argmax(-1).tolist()
        base={r['id']:r for r in previous.rows(out/'selection-base-single.jsonl')}
        mismatch=sum(base[r['id']]['prediction']!=v for r,v in zip(examples,pred))
        if mismatch:raise ValueError('cached baseline differs')
        candidates=[];coverage={}
        for neighbors in protocol['neighbors']:
            knn=probabilities(memory,examples,cache['base'],neighbors)
            coverage={'schemas':len(memory),'supported_selection_rows':len(knn),'training_rows':sum(len(m['vectors']) for m in memory.values())}
            for strength in protocol['weight']:
                rr=records(examples,base,knn,strength)
                name=f'memory-k{neighbors}-w{strength:g}'
                candidate={'name':name,'neighbors':neighbors,'weight':strength,'selection':search.metrics(rr)}
                candidates.append(candidate)
                search.save_records(out/f'selection-{name}.jsonl',rr)
                print('MEMORY',name,candidate['selection']['macro'],coverage,flush=True)
        winner=sorted(candidates,key=lambda c:(-c['selection']['macro'],c['weight'],c['neighbors']))[0]
        delta=winner['selection']['macro']-search.metrics(list(base.values()))['macro']
        train.atomic_json(out/'memory-frozen.json',{'winner':winner,'candidates':candidates,'coverage':coverage,
            'confirmation_eligible':delta>=.0025,'before_shared_confirmation':True,'cache_baseline_prediction_mismatches':mismatch,
            'memory_sha256':train.sha256_file(out/'gold-memory.pt'),
            'cache_sha256':{str(CACHE/name):train.sha256_file(CACHE/name) for name in ('train.jsonl','train.pt','selection.jsonl','selection.pt','code-weight.pt')},
            'confirmation_sha256':manifest['files']['confirmation.jsonl']})
    else:
        frozen=study.read(out/'memory-frozen.json')
        if not frozen['confirmation_eligible']:raise ValueError('memory did not pass predeclared selection gate')
        if (out/'memory-confirmation.json').exists():raise ValueError('already confirmed')
        if train.sha256_file(out/'gold-memory.pt')!=frozen['memory_sha256']:raise ValueError('memory changed')
        examples=previous.rows(out/'confirmation.jsonl')
        base={r['id']:r for r in previous.rows(out/'confirmation-base.jsonl')}
        memory=torch.load(out/'gold-memory.pt',map_location='cpu',weights_only=True)
        model=study.load_model();renderer,_=search.renderer_for(search.BASE)
        features=live_vectors(model,renderer,examples)
        torch.save(features,out/'confirmation-memory-features.pt')
        spec=frozen['winner'];knn=probabilities(memory,examples,features,spec['neighbors'])
        rr=records(examples,base,knn,spec['weight'])
        search.save_records(out/'confirmation-memory.jsonl',rr)
        delta=study.paired_interval(list(base.values()),rr,confidence=1-.05/4)
        delta['paired_document_bootstrap_joint_98_75ci']=delta.pop('paired_document_bootstrap_97_5ci')
        result={'candidate':spec,'metrics':search.metrics(rr),'vs_base':delta,'supported_confirmation_rows':len(knn)}
        train.atomic_json(out/'memory-confirmation.json',result)
        print('CONFIRM MEMORY',result['metrics']['macro'],delta,flush=True)


if __name__=='__main__':main()
