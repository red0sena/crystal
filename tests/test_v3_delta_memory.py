"""Depth-memory identity, cross-layer gradients, replay and stock-weight export."""
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import torch

from qwen_decision.v3 import delta_memory as dm, modeling
from tests.test_v3_decision_tail import Model
from tests.test_v3_recurrent import tiny


class DeltaTests(unittest.TestCase):
    def test_zero_gates_preserve_outputs_and_backbone_gradients(self):
        torch.manual_seed(51)
        a=Model().eval(); b=Model().eval(); b.load_state_dict(a.state_dict())
        originals=list(b.model.language_model.layers)
        dm.install(b.model.language_model,2); b.eval()
        ids=torch.tensor([[1,2,3,4,5,6],[7,8,9,0,0,0]])
        codes=torch.tensor([[2,3,4],[3,5,3]])
        valid=torch.tensor([[True,True,True],[True,True,False]])
        za=modeling.code_logits(a,ids,ids.ne(0),codes,valid)
        zb=modeling.code_logits(b,ids,ids.ne(0),codes,valid)
        torch.testing.assert_close(za,zb,atol=0,rtol=0)
        loss_a=za[:,:2].square().mean();loss_b=zb[:,:2].square().mean()
        loss_a.backward();loss_b.backward()
        for la,lb in zip(a.model.language_model.layers,originals):
            for pa,pb in zip(la.parameters(),lb.parameters()):
                torch.testing.assert_close(pa.grad,pb.grad,atol=2e-5,rtol=2e-5)
        self.assertGreater(b.model.language_model.layers[0].gate.grad.abs().sum().item(),0)

    def test_checkpoint_replay_is_stateless_and_router_is_active(self):
        torch.manual_seed(52)
        model=Model(); text=model.model.language_model
        dm.install(text,1)
        text.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant":False})
        dm.disable_native_checkpointing(text)
        for block in text.layers[:-1]:
            block.gate.data.fill_(.1);block.query.data.normal_(std=.02)
        ids=torch.tensor([[1,2,3,4,5,6],[7,8,9,0,0,0]])
        codes=torch.tensor([[2,3],[3,5]]);valid=codes.ne(-1)
        model.eval()
        expected=modeling.code_logits(model,ids,ids.ne(0),codes,valid)
        # An intervening different request must not contaminate the delta bank.
        modeling.code_logits(model,ids[:1,:3],ids[:1,:3].ne(0),codes[:1],valid[:1])
        torch.testing.assert_close(modeling.code_logits(model,ids,ids.ne(0),codes,valid),expected,atol=0,rtol=0)
        with dm.native(text): native=modeling.code_logits(model,ids,ids.ne(0),codes,valid)
        self.assertFalse(torch.allclose(native,expected))
        model.train()
        actual=modeling.code_logits(model,ids,ids.ne(0),codes,valid)
        torch.testing.assert_close(actual,expected,atol=0,rtol=0)
        actual.square().mean().backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
        self.assertTrue(all(p.grad.abs().sum()>0 for p in dm.routing_parameters(text).values()))

    def test_export_restores_native_names_and_nonzero_routing(self):
        model=Model().eval();text=model.model.language_model
        dm.install(text,2);text.layers[0].gate.data.fill_(.2)
        model.eval()
        ids=torch.tensor([[1,2,3,4]])
        codes=torch.tensor([[2,3]]);valid=codes.ne(-1)
        expected=modeling.code_logits(model,ids,ids.ne(0),codes,valid)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)
            def save_stock(m,tokenizer,out,config):
                self.assertEqual(len(m.model.language_model.layers),4)
                torch.save(m.state_dict(),out/'stock.pt')
            with patch.object(dm.modeling,'save',side_effect=save_stock):dm.save(model,None,path,{})
            other=Model();other.load_state_dict(torch.load(path/'stock.pt',weights_only=True))
            dm.restore_architecture(other,path);other.eval()
            actual=modeling.code_logits(other,ids,ids.ne(0),codes,valid)
            torch.testing.assert_close(actual,expected,atol=0,rtol=0)

    @unittest.skipUnless(torch.cuda.is_available(),"hybrid DeltaNet requires CUDA")
    def test_hybrid_native_masks_and_checkpointed_gradients(self):
        modeling.use_fla_conv()
        text=tiny('cuda').to(torch.bfloat16).eval()
        ids=torch.tensor([[1,2,3,4,5,6],[7,8,9,0,0,0]],device='cuda')
        expected=text(input_ids=ids,attention_mask=ids.ne(0),use_cache=False).last_hidden_state
        dm.install(text,2);text.eval()
        actual=text(input_ids=ids,attention_mask=ids.ne(0),use_cache=False).last_hidden_state
        torch.testing.assert_close(actual,expected,atol=0,rtol=0)
        text.layers[0].gate.data.fill_(.1)
        text.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant":False})
        dm.disable_native_checkpointing(text);text.train()
        text(input_ids=ids,attention_mask=ids.ne(0),use_cache=False).last_hidden_state.float().square().mean().backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in text.parameters()))


if __name__=='__main__':unittest.main()
