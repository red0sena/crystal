"""Workspace write-back, permutation, checkpoint replay, padding and export."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn

from qwen_decision.v3 import hypothesis_workspace as ws, modeling
from tests.test_v3_recurrent import tiny


class Model(nn.Module):
    def __init__(self, device='cpu'):
        super().__init__()
        self.model = nn.Module()
        self.model.language_model = tiny(device)
        self.head = nn.Linear(32,64,bias=False,device=device)

    def get_output_embeddings(self):
        return self.head


def inputs(device='cpu'):
    ids = torch.tensor([[1,2,3,4,5,6],[7,8,9,10,0,0]], device=device)
    codes = torch.tensor([[2,3,4],[3,5,3]], device=device)
    valid = torch.tensor([[True,True,True],[True,True,False]], device=device)
    positions = torch.tensor([[1,2,3],[0,1,0]], device=device)
    return ids, ids.ne(0), codes, valid, positions


class WorkspaceTests(unittest.TestCase):
    def test_disabled_route_matches_native_and_write_back_is_active(self):
        torch.manual_seed(79)
        model = Model().eval()
        x = inputs()
        native = modeling.code_logits(model,*x[:4])
        ws.install(model.model.language_model,start=2)
        model.eval()
        disabled = ws.logits(model,*x,enabled=False)
        blocked = ws.logits(model,*x,write_back=False)
        active = ws.logits(model,*x)
        torch.testing.assert_close(disabled,native,atol=0,rtol=0)
        torch.testing.assert_close(blocked,native,atol=3e-6,rtol=3e-6)
        self.assertGreater((active[:,:2]-native[:,:2]).abs().max().item(),1e-5)
        self.assertEqual(active[1,2].item(),modeling.MASK)

    def test_slot_order_and_padded_candidate_do_not_affect_context(self):
        torch.manual_seed(80)
        model = Model().eval()
        ws.install(model.model.language_model,start=2)
        model.eval()
        ids,mask,codes,valid,positions = inputs()
        base = ws.logits(model,ids,mask,codes,valid,positions)
        p = torch.tensor([2,0,1])
        perm = ws.logits(model,ids,mask,codes[:,p],valid[:,p],positions[:,p])
        torch.testing.assert_close(perm,base[:,p],atol=3e-6,rtol=3e-6)
        solo = ws.logits(model,ids[1:,:4],mask[1:,:4],codes[1:,:2],valid[1:,:2],positions[1:,:2])
        torch.testing.assert_close(solo,base[1:,:2],atol=3e-6,rtol=3e-6)

    def test_checkpoint_replay_gradients_and_request_independence(self):
        torch.manual_seed(81)
        model = Model()
        ws.install(model.model.language_model,start=2)
        x = inputs()
        model.train()
        a = ws.logits(model,*x)
        a[:,:2].square().mean().backward()
        reference = {k:p.grad.clone() for k,p in model.named_parameters()}
        model.zero_grad(set_to_none=True)
        model.model.language_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        b = ws.logits(model,*x)
        # An unrelated forward before backward must not overwrite saved state.
        ws.logits(model,*(item[:1] for item in x))
        b[:,:2].square().mean().backward()
        torch.testing.assert_close(a,b,atol=0,rtol=0)
        for k,p in model.named_parameters():
            self.assertIsNotNone(p.grad,k)
            self.assertTrue(torch.isfinite(p.grad).all(),k)
            torch.testing.assert_close(p.grad,reference[k],atol=3e-6,rtol=3e-6)
        self.assertGreater(model.model.language_model.hypothesis_workspace.projection.weight.grad.abs().sum().item(),0)

    def test_stock_and_workspace_export_restore(self):
        torch.manual_seed(82)
        model = Model().eval()
        ws.install(model.model.language_model,start=2)
        with torch.no_grad():
            model.model.language_model.hypothesis_workspace.projection.weight.add_(.01)
        model.eval()
        expected = ws.logits(model,*inputs())
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            def save_stock(m,tokenizer,out,config):
                self.assertFalse(hasattr(m.model.language_model,'hypothesis_workspace'))
                torch.save(m.state_dict(),out/'stock.pt')
            with patch.object(ws.modeling,'save',side_effect=save_stock):
                ws.save(model,None,path,{})
            other = Model()
            other.load_state_dict(torch.load(path/'stock.pt',weights_only=True),strict=True)
            ws.restore_architecture(other,path)
            other.eval()
            torch.testing.assert_close(ws.logits(other,*inputs()),expected,atol=0,rtol=0)

    @unittest.skipUnless(torch.cuda.is_available(),'hybrid DeltaNet requires CUDA')
    def test_hybrid_context_mixer_is_not_run_over_ordered_slots(self):
        modeling.use_fla_conv()
        model = Model('cuda').to(torch.bfloat16).eval()
        ws.install(model.model.language_model,start=2)
        model.eval()
        x = inputs('cuda')
        native = modeling.code_logits(model,*x[:4])
        torch.testing.assert_close(ws.logits(model,*x,enabled=False),native,atol=0,rtol=0)
        shapes = []
        handle = model.model.language_model.layers[2].linear_attn.register_forward_pre_hook(
            lambda m,a,kw:shapes.append(kw['hidden_states'].shape[1]),with_kwargs=True)
        active = ws.logits(model,*x)
        handle.remove()
        self.assertEqual(shapes,[x[0].shape[1]])
        solo = ws.logits(model,*(v[1:,:4] if i<2 else v[1:,:2] for i,v in enumerate(x)))
        torch.testing.assert_close(solo,active[1:,:2],atol=.02,rtol=.02)
        model.model.language_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        model.train()
        ws.logits(model,*x)[:,:2].square().mean().backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))


if __name__ == '__main__':
    unittest.main()
