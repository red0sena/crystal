"""Enforce the candidate exit and preserve it across checkpoint loading."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from qwen_decision.v3 import hypothesis_workspace as ws, workspace_bottleneck as bottleneck
from tests.test_v3_hypothesis_workspace import Model,inputs


class BottleneckTests(unittest.TestCase):
    def test_context_readout_is_not_an_available_exit(self):
        model = Model().eval()
        context = torch.randn(2,6,32,requires_grad=True)
        slots = torch.randn(2,3,32,requires_grad=True)
        with patch.object(ws,'hidden',return_value=(context,slots)):
            z = bottleneck.logits(model,*inputs())
            z[:,:2].sum().backward()
        self.assertIsNone(context.grad)
        self.assertGreater(slots.grad.abs().sum().item(),0)
        self.assertTrue(torch.isfinite(slots.grad).all())

    def test_checkpoint_replay_reaches_workspace_and_all_backbone_parameters(self):
        model = Model()
        bottleneck.install(model.model.language_model,start=2)
        model.model.language_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        model.train()
        bottleneck.logits(model,*inputs())[:,:2].square().mean().backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
        self.assertGreater(model.model.language_model.hypothesis_workspace.projection.weight.grad.abs().sum().item(),0)

    def test_export_restores_the_candidate_exit_and_rejects_context_loader(self):
        model = Model().eval()
        bottleneck.install(model.model.language_model,start=2)
        model.eval()
        expected = bottleneck.logits(model,*inputs())
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            def save_stock(m,tokenizer,out,config):
                self.assertFalse(hasattr(m.model.language_model,'hypothesis_workspace'))
                torch.save(m.state_dict(),out/'stock.pt')
            with patch.object(ws.modeling,'save',side_effect=save_stock):
                bottleneck.save(model,None,path,{})
            other = Model()
            other.load_state_dict(torch.load(path/'stock.pt',weights_only=True),strict=True)
            with patch.object(ws.modeling,'read_config',return_value={'attention':'noncausal','deltanet':'causal'}),patch.object(ws.modeling,'load',return_value=other):
                loaded = bottleneck.load(path,device='cpu').eval()
            torch.testing.assert_close(bottleneck.logits(loaded,*inputs()),expected,atol=0,rtol=0)
            with self.assertRaises(ValueError):
                ws.restore_architecture(Model(),path)


if __name__=='__main__':
    unittest.main()
