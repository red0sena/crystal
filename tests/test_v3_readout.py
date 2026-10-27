"""Contracts for vocabulary-independent, nonlinear, and option-level readouts."""
from types import SimpleNamespace
import unittest

import torch
from torch import nn
from torch.nn import functional as F

from qwen_decision.v3 import pointer, readout
from qwen_decision.v3.readout_experiment import document_groups
from tests.test_v3_pointer import row, tiny_backbone


class ReadoutTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(11)
        self.weight=torch.randn(7,16)
        self.final=torch.randn(2,16)
        self.options=torch.randn(2,3,16)
        self.columns=torch.tensor([[5,0,2],[4,6,0]])
        self.valid=torch.tensor([[True,True,True],[True,True,False]])

    def test_copied_readout_and_warm_mlp_preserve_lm_code_logits(self):
        expected=F.linear(self.final,self.weight).gather(1,self.columns).masked_fill(~self.valid,pointer.MASK)
        for kind in ('code','warm_mlp'):
            head=readout.DecisionReadout(kind,self.weight,inner=8)
            torch.testing.assert_close(head(self.final,self.options,self.columns,self.valid),expected,rtol=0,atol=0)
            self.assertIsNot(head.output.weight,self.weight)
            self.assertTrue(head.output.weight.requires_grad)

    def test_every_readout_trains_and_masks_unused_options(self):
        for kind in readout.KINDS:
            head=readout.DecisionReadout(kind,self.weight,inner=8)
            z=head(self.final,self.options,self.columns,self.valid)
            self.assertEqual(z.shape,(2,3))
            self.assertEqual(z[1,2],pointer.MASK)
            pointer.soft_ce(z,torch.tensor([[0.,1.,0.],[1.,0.,0.]])).mean().backward()
            self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for p in head.parameters()))
            if kind=='warm_mlp':
                self.assertGreater(head.down.weight.grad.abs().sum(),0)
                self.assertGreater(head.output.weight.grad.abs().sum(),0)

    def test_shared_option_readout_ignores_code_identity_and_permutates_with_vectors(self):
        valid=torch.ones_like(self.valid)
        perm=[2,0,1]
        for kind in ('option_linear','option_mlp'):
            head=readout.DecisionReadout(kind,self.weight,inner=8)
            a=head(self.final,self.options,self.columns,valid)
            b=head(self.final,self.options[:,perm],self.columns,valid)
            c=head(self.final,self.options,self.columns.flip(1),valid)
            torch.testing.assert_close(a[:,perm],b)
            torch.testing.assert_close(a,c)
        # Full model invariance is not implied: the prompt itself still contains codes.

    def test_ragged_batch_handles_noncontiguous_indices_and_invalid_slots(self):
        cache={'base':torch.randn(3,16),'options':torch.randn(9,16),
               'lengths':torch.tensor([3,2,4]),'offsets':torch.tensor([0,3,5]),
               'columns':torch.tensor([[0,2,3,0],[1,4,0,0],[6,2,0,1]]),
               'targets':torch.tensor([[1.,0.,0.,0.],[0.,1.,0.,0.],[0.,0.,1.,0.]])}
        batch=readout.cache_batch(cache,[2,1],True)
        torch.testing.assert_close(batch['options'][0],cache['options'][5:9])
        torch.testing.assert_close(batch['options'][1,:2],cache['options'][3:5])
        self.assertEqual(batch['valid'].tolist(),[[True]*4,[True,True,False,False]])
        self.assertIsNone(readout.cache_batch(cache,[0],False)['options'])

    def test_real_backbone_one_forward_padding_and_tail_gradient(self):
        text=tiny_backbone()
        text.requires_grad_(False)
        text.layers[-1].requires_grad_(True)
        model=SimpleNamespace(model=SimpleNamespace(language_model=text))
        batch=pointer.collate([row(),row('short',6,2)],0,'cpu')
        weight=torch.randn(4,32)
        head=readout.DecisionReadout('warm_mlp',weight,inner=16)
        calls=[]
        hook=text.register_forward_hook(lambda *unused:calls.append(1))
        z=readout.live_logits(model,head,batch)
        self.assertEqual(len(calls),1)
        pointer.soft_ce(z,batch['targets']).mean().backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for p in text.layers[-1].parameters()))
        self.assertTrue(all(p.grad is None for p in text.layers[0].parameters()))
        hook.remove()
        text.eval()
        with torch.no_grad():
            both=readout.live_logits(model,head,batch)[1,:2]
            alone=readout.live_logits(model,head,pointer.collate([row('short',6,2)],0,'cpu'))[0]
        torch.testing.assert_close(both,alone,rtol=1e-4,atol=1e-5)

    def test_holdback_groups_join_upstream_documents_and_whitespace(self):
        data=[{'id':'a','state':'one  fact','question':{}},
              {'id':'b','state':'one fact','question':{},'documents':['shared']},
              {'id':'c','state':'different','question':{},'documents':['shared']},
              {'id':'d','state':'else','question':{}}]
        groups=document_groups(data)
        self.assertEqual(groups['a'],groups['c'])
        self.assertNotEqual(groups['a'],groups['d'])


if __name__=='__main__':
    unittest.main()
