"""Architecture identity, pooling, gradient and selection/confirmation contracts."""
from copy import deepcopy
import unittest

import torch

from qwen_decision.v3 import architecture as arch, pointer
from qwen_decision.v3 import architecture_search as search
from tests.test_v3_pointer import tiny_backbone, row


class ArchitectureTests(unittest.TestCase):
    def test_readouts_preserve_original_prediction_at_initialization(self):
        torch.manual_seed(8)
        weight, base = torch.randn(5, 32), torch.randn(3, 32)
        features = torch.randn(3, 4, 3, 32)
        features[:, -1, 0] = base
        columns = torch.tensor([[0, 1, 2], [1, 3, 4], [4, 0, 2]])
        valid = torch.tensor([[1, 1, 1], [1, 1, 0], [1, 1, 1]], dtype=torch.bool)
        expected = arch.Readout('base', weight, 4)(base, features, columns, valid)
        for kind in ['code', 'depth', 'pool', 'scoring', 'class_attention','ffn']:
            actual = arch.Readout(kind, weight, 4, inner=8)(base, features, columns, valid)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_zero_initialized_class_attention_can_learn(self):
        torch.manual_seed(9)
        head = arch.Readout('class_attention', torch.randn(3, 16), 2, inner=8)
        base, features = torch.randn(4, 16), torch.randn(4, 2, 3, 16)
        cols, valid = torch.arange(3).expand(4, -1), torch.ones(4, 3, dtype=torch.bool)
        optimizer = torch.optim.AdamW(head.parameters(), lr=.001)
        initial = head(base, features, cols, valid).detach().clone()
        loss = torch.nn.functional.cross_entropy(head(base, features, cols, valid), torch.tensor([0,1,2,0]))
        loss.backward()
        self.assertGreater(head.out.weight.grad.abs().sum().item(), 0.)
        optimizer.step()
        self.assertGreater((initial-head(base, features, cols, valid)).abs().max().item(), 0.)

    def test_layer_contrast_zero_is_identity_and_padding_is_finite(self):
        weight,base,features=torch.randn(4,16),torch.randn(2,16),torch.randn(2,4,3,16)
        cols=torch.tensor([[0,1,2],[3,2,0]])
        valid=torch.tensor([[True,True,True],[True,True,False]])
        expected=arch.Readout('base',weight,4)(base,features,cols,valid)
        for mode in ['mean','js']:
            head=arch.ContrastReadout(weight,8,0.,mode)
            torch.testing.assert_close(head(base,features,cols,valid),expected,atol=0,rtol=0)
            head.alpha=.2
            actual=head(base,features,cols,valid)
            self.assertTrue(torch.isfinite(actual).all())
            self.assertEqual(actual[1,2].item(),-1e9)

    def test_residual_option_heads_identity_gradients_and_equivariance(self):
        torch.manual_seed(31)
        weight,base,options=torch.randn(4,16),torch.randn(2,16),torch.randn(2,3,16)
        columns=torch.tensor([[0,1,2],[3,2,0]])
        valid=torch.tensor([[True,True,True],[True,True,False]])
        expected=torch.nn.functional.linear(base,weight).gather(1,columns).masked_fill(~valid,-1e9)
        for kind in ['pointer-residual','set-residual']:
            head=arch.OptionReadout(kind,weight)
            actual=head(base,options,columns,valid)
            torch.testing.assert_close(actual,expected,atol=0,rtol=0)
            torch.nn.functional.cross_entropy(actual,torch.tensor([1,0])).backward()
            grads=[p.grad for p in head.parameters() if p.grad is not None]
            self.assertTrue(all(torch.isfinite(g).all() for g in grads))
            self.assertGreater(sum(g.abs().sum().item() for g in grads),0.)
            if kind=='pointer-residual':
                torch.nn.init.normal_(head.query.weight,std=.05)
            else:
                torch.nn.init.normal_(head.out.weight,std=.05)
            head.eval()
            permutation=[2,0,1]
            a=head(base,options,columns,valid)
            b=head(base,options[:,permutation],columns[:,permutation],valid[:,permutation])
            torch.testing.assert_close(a[:,permutation],b,atol=1e-5,rtol=1e-5)

    def test_summary_collection_ignores_padding_and_keeps_exact_base(self):
        torch.manual_seed(10)
        text = tiny_backbone().eval().requires_grad_(False)
        collector = arch.SummaryCollector(text)
        batch = pointer.collate([row(), row('b', 6, 2)], 0, 'cpu')
        base, features = collector(batch)
        torch.testing.assert_close(base, features[:, -1, 0], atol=0, rtol=0)
        solo_base, solo_features = collector(pointer.collate([row('b', 6, 2)], 0, 'cpu'))
        torch.testing.assert_close(features[1], solo_features[0], atol=2e-5, rtol=2e-5)
        torch.testing.assert_close(base[1], solo_base[0], atol=2e-5, rtol=2e-5)
        collector.close()
        self.assertTrue(all(not layer._forward_hooks for layer in text.layers))

    def test_branch_scales_identity_checkpointed_gradient_and_cleanup(self):
        torch.manual_seed(12)
        for channels in (False, True):
            text = tiny_backbone().eval().requires_grad_(False)
            batch = pointer.collate([row()], 0, 'cpu')
            inputs = {k:batch[k] for k in ['input_ids','attention_mask']}
            with torch.no_grad():
                expected = text(**inputs, use_cache=False).last_hidden_state
            stock = deepcopy(text.state_dict())
            parameters, handles = arch.install_branch_scales(text, channels=channels)
            actual = text(**inputs, use_cache=False).last_hidden_state
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            text.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
            text.train()
            h = text(**inputs, use_cache=False).last_hidden_state[:, -1]
            loss = torch.nn.functional.cross_entropy(h @ torch.randn(32,3), torch.tensor([1]))
            loss.backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in parameters.values()))
            self.assertGreater(sum(p.grad.abs().sum().item() for p in parameters.values()),0.)
            for k,v in stock.items():
                torch.testing.assert_close(v,text.state_dict()[k],rtol=0,atol=0)
            arch.remove_branch_scales(text,handles)
            self.assertEqual(set(stock),set(text.state_dict()))

    def test_partition_keeps_upstream_document_groups_together(self):
        def record(i, text):
            return {'id':str(i),'state':text,'question':{'type':'noul'}}
        rows = [record(1,'first'),record(2,'second'),record(3,'third'),record(4,'first')]
        rows[1]['documents'], rows[2]['documents'] = ['doc-x'], ['doc-x']
        partitions = search.partition_dev(rows)
        self.assertEqual(partitions[0],partitions[3])
        self.assertEqual(partitions[1],partitions[2])
        self.assertEqual(partitions, search.partition_dev(rows))

    def test_inner_scales_preserve_output_and_receive_gradients(self):
        text=tiny_backbone().eval().requires_grad_(False)
        batch=pointer.collate([row()],0,'cpu')
        inputs={k:batch[k] for k in ['input_ids','attention_mask']}
        with torch.no_grad():
            expected=text(**inputs,use_cache=False).last_hidden_state
        params,handles=arch.install_inner_scales(text)
        actual=text(**inputs,use_cache=False).last_hidden_state
        torch.testing.assert_close(actual,expected,atol=0,rtol=0)
        (actual[:,-1]@torch.randn(32,3)).square().mean().backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in params.values()))
        self.assertGreater(sum(p.grad.abs().sum().item() for p in params.values()),0.)
        arch.remove_branch_scales(text,handles)
        self.assertFalse(any('decision_scale' in name for name,_ in text.named_parameters()))

    def test_surgery_is_explicit_and_preserves_layer_identities(self):
        self.assertEqual(arch.layer_order(6,'skip',3),[0,1,2,4,5])
        self.assertEqual(arch.layer_order(6,'swap',3),[0,1,2,4,3,5])
        self.assertEqual(arch.layer_order(6),list(range(6)))

    @unittest.skipUnless(torch.cuda.is_available(), 'hybrid FLA layers need CUDA')
    def test_hybrid_surgery_moves_mask_dispatch_with_layers(self):
        text = tiny_backbone('cuda').to(torch.bfloat16).eval()
        original = list(text.layers)
        batch = pointer.collate([row(), row('b',6,2)],0,'cuda')
        for order in ([0,1],[1],[1,0],[0,1]):
            arch.apply_layer_order(text,original,order)
            self.assertEqual(text.config.layer_types,[original[i].block_type for i in order])
            with torch.no_grad():
                h=text(input_ids=batch['input_ids'],attention_mask=batch['attention_mask'],use_cache=False).last_hidden_state
            self.assertTrue(torch.isfinite(h).all())

    def test_paired_bootstrap_uses_matching_ids(self):
        a = [{'id':str(i),'source':str(i%2),'group':str(i//2),'correct':False} for i in range(8)]
        b = [{**r,'correct':True} for r in a]
        result = search.paired_delta(a,b,iterations=20)
        self.assertEqual(result['macro_delta'],1.)
        self.assertEqual(result['paired_document_bootstrap_95ci'],[1.,1.])
        self.assertEqual(result['wins'],8)
        with self.assertRaises(ValueError):
            search.paired_delta(a,b[:-1],iterations=20)


if __name__ == '__main__':
    unittest.main()
