"""Sparse final-query equivalence, candidate conditioning and gradients."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn

from qwen_decision.v3 import decision_tail as dt, modeling
from tests.test_v3_recurrent import tiny


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.language_model = tiny()
        self.head = nn.Linear(32,64,bias=False)

    def get_output_embeddings(self): return self.head


class TailTests(unittest.TestCase):
    def test_export_restores_nonzero_adapter_and_native_weight_names(self):
        model = Model().eval()
        tail = dt.install(model.model.language_model,rank=8)
        nn.init.normal_(tail.down.weight,std=.03)
        model.eval()
        ids=torch.tensor([[1,2,3,4,5,6]])
        codes=torch.tensor([[2,3,4]])
        positions=torch.tensor([[1,3,4]])
        expected=dt.logits(model,ids,ids.ne(0),codes,codes.ne(-1),positions)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)
            def save_stock(m,tokenizer,out,config):
                self.assertNotIsInstance(m.model.language_model.layers[-1],dt.DecisionTail)
                torch.save(m.state_dict(),out/'stock.pt')
            with patch.object(dt.modeling,'save',side_effect=save_stock):
                dt.save(model,None,path,{},mode='candidates')
            self.assertIs(model.model.language_model.layers[-1],tail)
            other=Model()
            other.load_state_dict(torch.load(path/'stock.pt',weights_only=True),strict=True)
            dt.restore_architecture(other,path)
            other.eval()
            actual=dt.logits(other,ids,ids.ne(0),codes,codes.ne(-1),positions)
            torch.testing.assert_close(actual,expected,atol=0,rtol=0)
            self.assertEqual(other.model.language_model.layers[-1].mode,'candidates')

    def test_native_single_and_identical_candidate_queries_agree_with_padding(self):
        torch.manual_seed(12)
        model = Model().eval()
        ids = torch.tensor([[1,2,3,4,5,6],[7,8,9,0,0,0]])
        mask = ids.ne(0)
        codes = torch.tensor([[2,3,4],[3,5,3]])
        valid = torch.tensor([[True,True,True],[True,True,False]])
        positions = torch.tensor([[1,2,3],[0,1,0]])
        expected = modeling.code_logits(model,ids,mask,codes,valid)
        dt.install(model.model.language_model,rank=8)
        model.eval()
        for mode in ("single","candidates"):
            actual = dt.logits(model,ids,mask,codes,valid,positions,mode)
            torch.testing.assert_close(actual,expected,atol=2e-6,rtol=2e-6)
        with dt.native(model.model.language_model):
            actual = modeling.code_logits(model,ids,mask,codes,valid)
            torch.testing.assert_close(actual,expected,atol=0,rtol=0)

    def test_adapter_learns_and_training_reaches_shared_backbone(self):
        torch.manual_seed(13)
        model = Model()
        tail = dt.install(model.model.language_model,rank=8)
        model.model.language_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant":False})
        ids = torch.tensor([[1,2,3,4,5,6]])
        codes = torch.tensor([[2,3,4]])
        valid = torch.ones_like(codes,dtype=torch.bool)
        positions = torch.tensor([[1,2,3]])
        actual = dt.logits(model,ids,ids.ne(0),codes,valid,positions)
        nn.functional.cross_entropy(actual,torch.tensor([1])).backward()
        self.assertIsNone(tail.result)
        self.assertIsNone(tail.request)
        self.assertGreater(tail.down.weight.grad.abs().sum().item(),0.)
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
        # Distinct candidate states change the attention computation, rather
        # than merely adding a shared scalar shift to final logits.
        before = actual.detach()
        nn.init.normal_(tail.down.weight,std=.03)
        after = dt.logits(model,ids,ids.ne(0),codes,valid,positions)
        self.assertFalse(torch.allclose(before.softmax(-1),after.softmax(-1)))

    def test_query_specific_work_has_k_rows_and_memory_keeps_t_rows(self):
        model = Model().eval()
        tail = dt.install(model.model.language_model,rank=8)
        model.eval()
        shapes = {}
        handles = [module.register_forward_pre_hook(lambda m,a,key=key:shapes.update({key:a[0].shape[1]}))
                   for key,module in [('q',tail.native.self_attn.q_proj),('k',tail.native.self_attn.k_proj),('ffn',tail.native.mlp)]]
        ids=torch.tensor([[1,2,3,4,5,6,7,8,9]])
        codes=torch.tensor([[2,3,4]])
        dt.logits(model,ids,ids.ne(0),codes,codes.ne(-1),torch.tensor([[1,3,5]]))
        for h in handles:h.remove()
        self.assertEqual(shapes,{'q':3,'k':9,'ffn':3})


if __name__=='__main__':unittest.main()
