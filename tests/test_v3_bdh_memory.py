"""Associative factorization, replacement execution, gradients and persistence."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from qwen_decision.v3 import bdh_memory as bdh,modeling
from tests.test_v3_hypothesis_workspace import Model,inputs


class MemoryTests(unittest.TestCase):
    def test_linear_memory_equals_dense_noncausal_no_self_and_gradients(self):
        torch.manual_seed(54)
        phi = torch.rand(2,2,5,4,requires_grad=True)
        values = torch.randn(2,1,5,6,requires_grad=True)
        mask = torch.tensor([[1,1,1,1,1],[1,1,1,0,0]],dtype=torch.bool)
        dense = phi@phi.transpose(-1,-2)
        dense = dense*mask[:,None,None,:]*(1-torch.eye(5))[None,None]
        expected = (dense@values)/(mask.sum(-1)-1)[:,None,None,None]
        actual = bdh.associate(phi,values,mask)
        torch.testing.assert_close(actual,expected,atol=2e-6,rtol=2e-6)
        a = torch.autograd.grad(actual.square().sum(),(phi,values),retain_graph=True)
        e = torch.autograd.grad(expected.square().sum(),(phi,values))
        for x,y in zip(a,e):
            torch.testing.assert_close(x,y,atol=3e-5,rtol=3e-5)

    def test_replaces_upper_layers_padding_and_memory_affects_output(self):
        torch.manual_seed(55)
        model = Model().eval();text = model.model.language_model
        bdh.install(text,start=2,heads=2,cycles=3);model.eval()
        counts = [0]*4
        handles = [l.register_forward_hook(lambda m,a,o,i=i:counts.__setitem__(i,counts[i]+1)) for i,l in enumerate(text.layers)]
        x = inputs();active = bdh.logits(model,*x[:4])
        for h in handles:h.remove()
        self.assertEqual(counts,[1,1,0,0])
        solo = bdh.logits(model,x[0][1:,:4],x[1][1:,:4],x[2][1:,:2],x[3][1:,:2])
        torch.testing.assert_close(solo,active[1:,:2],atol=3e-6,rtol=3e-6)
        removed = bdh.logits(model,*x[:4],memory_enabled=False)
        self.assertGreater((active[:,:2]-removed[:,:2]).abs().max().item(),1e-4)

    def test_checkpoint_replay_and_only_used_parameters_train(self):
        torch.manual_seed(56)
        model = Model();text = model.model.language_model
        bdh.install(text,start=2,heads=2,cycles=3)
        for layer in text.layers[2:]:layer.requires_grad_(False)
        model.train();x = inputs()
        a = bdh.logits(model,*x[:4]);a[:,:2].square().mean().backward()
        expected = {k:p.grad.clone() for k,p in model.named_parameters() if p.requires_grad}
        model.zero_grad(set_to_none=True)
        text.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        z = bdh.logits(model,*x[:4])
        bdh.logits(model,*(v[:1] for v in x[:4]))
        z[:,:2].square().mean().backward()
        for k,p in model.named_parameters():
            if p.requires_grad:
                self.assertIsNotNone(p.grad,k)
                self.assertTrue(torch.isfinite(p.grad).all(),k)
                torch.testing.assert_close(p.grad,expected[k],atol=3e-5,rtol=3e-5)
        self.assertGreater(text.bdh_memory.encoder.grad.norm().item(),0)

    def test_export_preserves_native_keys_and_restores_memory(self):
        model = Model().eval();text = model.model.language_model
        bdh.install(text,start=2,heads=2,cycles=2);model.eval()
        expected = bdh.logits(model,*inputs()[:4])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            def save_stock(m,tokenizer,out,config):
                self.assertFalse(hasattr(m.model.language_model,'bdh_memory'))
                torch.save(m.state_dict(),out/'stock.pt')
            with patch.object(modeling,'save',side_effect=save_stock):bdh.save(model,None,path,{})
            other = Model();other.load_state_dict(torch.load(path/'stock.pt',weights_only=True),strict=True)
            bdh.restore_architecture(other,path);other.eval()
            torch.testing.assert_close(bdh.logits(other,*inputs()[:4]),expected,atol=0,rtol=0)


if __name__=='__main__':unittest.main()
