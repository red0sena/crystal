"""Merge correctness, semantic option alignment and real decoder gradients."""
from copy import deepcopy
from types import SimpleNamespace
import unittest

import torch
from torch import nn

from qwen_decision.v3 import robust_adaptation as robust
from qwen_decision.v3 import architecture_search as search
from qwen_decision.v3.shortlist_experiment import distribute, shortlist_row
from qwen_decision.v3.memory_experiment import schema, make_memory, probabilities
from qwen_decision.v3.reference_comparison import as_example, skill
from qwen_decision.v3.robust_experiment import paired_interval
from qwen_decision.v3.fresh_holdback import state_key
from tests.test_v3_pointer import tiny_backbone


class RobustTests(unittest.TestCase):
    def test_new_holdback_excludes_normalized_and_serialized_state_siblings(self):
        self.assertEqual(state_key('An important FACT!'),state_key('an important\n fact'))
        self.assertEqual(state_key({'b':2,'a':'a value'}),state_key('{"a":"a value", "b":2}'))
        self.assertNotEqual(state_key('a different fact'),state_key('an important fact'))

    def test_paired_interval_uses_source_macro_and_keeps_document_groups(self):
        base=[{'id':str(i),'source':'large','group':'one-document','correct':False} for i in range(8)]
        base += [{'id':'small','source':'small','group':'another-document','correct':True}]
        candidate=[{**r,'correct':not r['correct']} for r in base]
        result=paired_interval(base,candidate,draws=20)
        self.assertEqual(result['macro_delta'],0.)
        self.assertEqual(result['paired_document_bootstrap_95ci'],[0.,0.])
        self.assertFalse(result['confirmed_improvement'])
        self.assertEqual((result['wins'],result['losses']),(8,1))

    def test_reference_labels_and_variable_chance_are_scored_consistently(self):
        row={'id':'x','source':'toy','state':'fact','question':{'type':'choice','criteria':{'opt_1':'one','opt_2':'two'}},'label':'opt_2'}
        self.assertEqual(as_example(row)['target'],[0.,1.])
        self.assertEqual(as_example({**row,'question':{'type':'noul'},'label':False})['target'],[1.,0.])
        examples=[as_example(row),{'source':'toy','target':[1.,0.,0.,0.]}]
        result=skill(examples,[{'correct':True},{'correct':False}])
        self.assertAlmostEqual(result['macro_accuracy'],.5)
        self.assertAlmostEqual(result['macro_skill'],.2)

    def test_memory_labels_align_only_with_exact_question_schema(self):
        question={'type':'choice','instructions':'Classify.','criteria':{'left':None,'right':None}}
        raw=[{'id':str(i),'ordered':False,'question':question,'target':[float(i<16),float(i>=16)]} for i in range(32)]
        x=torch.zeros(32,4);x[:16,0]=-1;x[16:,0]=1
        target=torch.tensor([r['target'] for r in raw])
        memory=make_memory(raw,x,target)
        query={**raw[0],'id':'query'}
        p=probabilities(memory,[query],x[:1],8)
        self.assertGreater(p['query'][0],.99)
        # Changing the query gold has no effect on neighbor selection or vote.
        changed={**query,'target':[0.,1.]}
        self.assertEqual(p,probabilities(memory,[changed],x[:1],8))
        swapped={**query,'question':{**question,'criteria':{'right':None,'left':None}}}
        self.assertNotEqual(schema(query),schema(swapped))
        self.assertEqual(probabilities(memory,[swapped],x[:1],8),{})

    def test_shortlist_preserves_mass_and_does_not_force_gold_inside(self):
        original=[.4,.3,.2,.1]
        p=distribute(original,[0,2],[.25,.75],.5)
        self.assertAlmostEqual(sum(p),1.)
        self.assertEqual(p[1],original[1]);self.assertEqual(p[3],original[3])
        self.assertAlmostEqual(p[0],.275);self.assertAlmostEqual(p[2],.325)
        row={'question':{'type':'choice','criteria':{'a':None,'b':None,'c':None}},'target':[0.,1.,0.]}
        smaller=shortlist_row(row,[0,2])
        self.assertEqual(list(smaller['question']['criteria']),['a','c'])
        self.assertEqual(smaller['target'],[0.,0.])

    def test_initial_identity_and_exact_weight_restore(self):
        module = nn.Sequential(nn.Linear(5,7),nn.GELU(),nn.Linear(7,3))
        original = deepcopy(module.state_dict())
        x = torch.randn(4,5)
        expected = module(x)
        robust.install(module,rank=2)
        torch.testing.assert_close(module(x),expected,atol=0,rtol=0)
        with torch.no_grad():
            for m in module.modules():
                if isinstance(m,robust.LowRankLinear):
                    m.b.normal_(std=.1)
        unmerged = module(x)
        with robust.merged(module):
            self.assertTrue(all(not isinstance(m,robust.LowRankLinear) for m in module.modules()))
            torch.testing.assert_close(module(x),unmerged,atol=2e-6,rtol=2e-6)
        with robust.merged(module,0):
            for name,value in module.state_dict().items():
                torch.testing.assert_close(value,original[name],atol=0,rtol=0)
        torch.testing.assert_close(module(x),unmerged,atol=0,rtol=0)

    def test_soup_averages_updates_and_reload_matches(self):
        module = nn.Sequential(nn.Linear(3,4,bias=False))
        base = module[0].weight.detach().clone()
        robust.install(module,rank=2)
        first = robust.adapter_state(module)
        # adapter_state returns snapshots: clone because CPU tests can share storage.
        first = {k:v.clone() for k,v in first.items()}
        first['0.a'].fill_(1); first['0.b'].fill_(2)
        second = {k:v.clone() for k,v in first.items()}
        second['0.a'].fill_(3); second['0.b'].fill_(-1)
        expected = base+.5*module[0].scale*(first['0.b']@first['0.a']+second['0.b']@second['0.a'])/2
        with robust.merged(module,.5,[first,second]):
            torch.testing.assert_close(module[0].weight,expected)
        robust.restore(module,first)
        clone = nn.Sequential(nn.Linear(3,4,bias=False))
        with torch.no_grad():clone[0].weight.copy_(base)
        robust.install(clone,rank=2); robust.restore(clone,first)
        x = torch.randn(2,3)
        torch.testing.assert_close(module(x),clone(x),atol=0,rtol=0)

    def test_alignment_and_divergence_mask(self):
        z = torch.tensor([[30.,10.,20.],[4.,8.,-1e9]],requires_grad=True)
        rows = [{'order':[2,0,1]},{'order':[1,0]}]
        aligned = robust.align(z,rows)
        torch.testing.assert_close(aligned,torch.tensor([[10.,20.,30.],[8.,4.,-1e9]]))
        torch.testing.assert_close(robust.jensen_shannon(aligned,aligned),torch.zeros(2))
        other = aligned.detach().clone(); other[:,0]+=1
        loss = robust.jensen_shannon(aligned,other).sum()
        self.assertGreater(loss.item(),0)
        loss.backward()
        self.assertTrue(torch.isfinite(z.grad).all())
        self.assertEqual(z.grad[1,2],0)

    def test_real_renderer_matches_base_and_protects_order(self):
        renderer,_ = search.renderer_for(search.BASE)
        row = {'id':'toy','state':'A useful fact.','source':'toy','kind':'choice','ordered':False,
               'question':{'type':'choice','criteria':{'red':None,'green':'a color','blue':None}},
               'target':[0.,1.,0.]}
        a,b = renderer(row),robust.view(renderer,row)
        self.assertEqual(a['input_ids'],b['input_ids'])
        changed = robust.view(renderer,row,[2,0,1],[1,2,0])
        self.assertEqual(changed['target'],[0.,0.,1.])
        self.assertEqual(changed['code_ids'],[renderer.code_ids[i] for i in [1,2,0]])
        self.assertEqual(len(robust.cyclic_views(renderer,row)),3)
        row['ordered']=True
        self.assertEqual(len(robust.cyclic_views(renderer,row)),1)
        with self.assertRaises(ValueError):robust.view(renderer,row,[2,0,1])
        row.update(ordered=False,target=[1.,0.],question={'type':'noul'})
        self.assertEqual(len(robust.cyclic_views(renderer,row)),1)

    def test_actual_decoder_adapters_train_across_depth(self):
        text = tiny_backbone()
        original = {n:p.detach().clone() for n,p in text.named_parameters()}
        names = robust.install(text,rank=2)
        ids = torch.tensor([[1,2,3,4],[4,3,2,0]])
        mask = ids!=0
        out = text(input_ids=ids,attention_mask=mask,use_cache=False).last_hidden_state
        out.square().mean().backward()
        self.assertTrue(any(name.startswith('layers.0.') for name in names))
        self.assertTrue(any(name.startswith('layers.1.') for name in names))
        for name,m in text.named_modules():
            if isinstance(m,robust.LowRankLinear):
                self.assertIsNotNone(m.b.grad)
                self.assertTrue(torch.isfinite(m.b.grad).all())
                self.assertIsNone(m.base.weight.grad)
        for name,value in original.items():
            # Embeddings, norms and original projection weights stay bit-identical.
            parts=name.split('.')
            if '.'.join(parts[:-1]) in names:parts.insert(-1,'base')
            torch.testing.assert_close(dict(text.named_parameters())['.'.join(parts)],value,atol=0,rtol=0)


if __name__=='__main__':
    unittest.main()
