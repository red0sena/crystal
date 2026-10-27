"""Audit all original training rows and prune overlaps before confirmation."""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from qwen_decision.v3 import pointer_experiment as pilot, train
from qwen_decision.v3.readout_experiment import group_keys, read_json, rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=Path('runs/readout/search-v1'))
    args=p.parse_args()
    out=args.output
    manifest=read_json(out/'manifest.json')
    source=Path('data/mix/v3-gold3.jsonl')
    assert train.sha256_file(source)==manifest['sha256'][str(source)]
    if (out/'confirmation-amendment.json').exists():
        amendment=read_json(out/'confirmation-amendment.json')
        assert train.sha256_file(out/'confirmation.jsonl')==amendment['after_sha256']
        assert train.sha256_file(out/'confirmation-before-full-audit.jsonl')==amendment['before_sha256']
        print(json.dumps({'already_audited':True,'confirmation_n':amendment['after_n'],
                          'remaining_shared_keys':amendment['remaining_confirmation_rows_with_original_training_shared_keys']}))
        return
    if (out/'confirmation-results.json').exists() or (out/'confirmation.pt').exists():
        raise SystemExit('never change the holdback after confirmation was accessed')
    lookup={r['id']:r for r in pilot.read_rows(Path('data/mix/v3-gold3.dev.jsonl'))}
    encoded=rows(out/'confirmation.jsonl')
    key_ids=defaultdict(set)
    for row in encoded:
        for key in group_keys(lookup[row['id']]):key_ids[key].add(row['id'])
    matches=set();sources=Counter();types=Counter();count=0
    with source.open() as f:
        for line in f:
            row=json.loads(line);count+=1
            overlap=set(group_keys(row)).intersection(key_ids)
            if overlap:
                sources[row['source']]+=1;types.update(k[0] for k in overlap)
                for key in overlap:matches.update(key_ids[key])
            if count%100000==0:print(f'audit {count} rows',flush=True)
    audit={'full_training_rows':count,'confirmation_n':len(encoded),
           'confirmation_rows_with_shared_keys':len(matches),'matched_training_rows_by_source':dict(sources),
           'shared_key_types':dict(types),'source_sha256':manifest['sha256'][str(source)],
           'scope':'upstream id, full state, whitespace-normalized state, item, explicit document keys; not semantic near-duplicate detection',
           'base_previously_used_full_gold3_dev_for_early_stopping':True}
    train.atomic_json(out/'full-training-audit.json',audit)
    bad_groups={r['group'] for r in encoded if r['id'] in matches}
    kept=[r for r in encoded if r['group'] not in bad_groups]
    if not kept:raise ValueError('no usable confirmation rows')
    before=train.sha256_file(out/'confirmation.jsonl')
    (out/'confirmation-before-full-audit.jsonl').write_bytes((out/'confirmation.jsonl').read_bytes())
    (out/'confirmation.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in kept))
    after=train.sha256_file(out/'confirmation.jsonl')
    amendment={'reason':'full original gold3 training audit; remove shared-key document groups before any confirmation scores',
               'scores_not_accessed':True,'before_n':len(encoded),'after_n':len(kept),
               'removed_ids':[r['id'] for r in encoded if r['group'] in bad_groups],
               'removed_groups':sorted(bad_groups),'before_sha256':before,'after_sha256':after,
               'remaining_confirmation_rows_with_original_training_shared_keys':0}
    train.atomic_json(out/'confirmation-amendment.json',amendment)
    manifest['counts']['confirmation']=len(kept)
    manifest['confirmation_groups']=len({r['group'] for r in kept})
    manifest['sources']['confirmation']=len({r['source'] for r in kept})
    manifest['encoded_sha256']['confirmation']=after
    manifest['confirmation_amendment']=amendment
    train.atomic_json(out/'manifest.json',manifest)
    print(json.dumps(amendment),flush=True)


if __name__=='__main__':
    main()
