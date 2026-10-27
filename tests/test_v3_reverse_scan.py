"""Directionality, identity initialization, padding isolation and gradients."""
import copy
from types import SimpleNamespace
import unittest

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from qwen_decision.v3 import reverse_scan


class CausalScan(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(4, 4, bias=False)

    def forward(self, x, attention_mask=None, **kwargs):
        if attention_mask is not None:
            x = x * attention_mask[..., None]
        return self.projection(x).cumsum(1)


def tiny():
    text = nn.Module()
    text.config = SimpleNamespace(hidden_size=4)
    layer = nn.Module()
    layer.block_type = "linear_attention"
    layer.linear_attn = CausalScan()
    text.layers = nn.ModuleList([layer])
    return text


class ReverseScanTests(unittest.TestCase):
    def test_initial_identity_original_gradients_and_new_gate_signal(self):
        torch.manual_seed(8)
        native = tiny()
        text = copy.deepcopy(native)
        before = sum(p.numel() for p in text.parameters())
        reverse_scan.install(text)
        self.assertEqual(sum(p.numel() for p in text.parameters()), before + 4)
        x = torch.randn(2, 5, 4)
        a = native.layers[0].linear_attn(x)
        b = text.layers[0].linear_attn(x)
        torch.testing.assert_close(a, b, atol=0, rtol=0)
        a.square().mean().backward()
        b.square().mean().backward()
        torch.testing.assert_close(native.layers[0].linear_attn.projection.weight.grad,
                                   text.layers[0].linear_attn.projection.weight.grad, atol=0, rtol=0)
        self.assertGreater(text.layers[0].linear_attn.backward_gate.grad.abs().sum().item(), 0)
        with self.assertRaises(ValueError):
            reverse_scan.install(text)

    def test_future_information_padding_and_branch_disable(self):
        torch.manual_seed(12)
        text = tiny()
        reverse_scan.install(text)
        module = text.layers[0].linear_attn
        module.backward_gate.data.fill_(.3)
        x = torch.randn(2, 6, 4)
        mask = torch.tensor([[1, 1, 1, 1, 1, 1], [1, 1, 1, 0, 0, 0]])
        batch = module(x, attention_mask=mask)
        solo = module(x[1:2, :3])
        # Different GEMM row counts can round FP32 products differently.
        torch.testing.assert_close(batch[1, :3], solo[0], atol=3e-7, rtol=3e-7)
        changed = x.clone()
        changed[0, -1] += 1
        self.assertFalse(torch.equal(batch[0, 0], module(changed, attention_mask=mask)[0, 0]))
        with reverse_scan.disabled(text):
            a = module(x, attention_mask=mask)
            b = module(changed, attention_mask=mask)
            torch.testing.assert_close(a[0, 0], b[0, 0], atol=0, rtol=0)
        self.assertTrue(text._reverse_scan_enabled)
        with self.assertRaises(ValueError):
            module(x, attention_mask=mask.flip(1))
        with self.assertRaises(ValueError):
            module(x, use_cache=True)
        with self.assertRaises(ValueError):
            module(x, cu_seq_lens_q=torch.tensor([0, 12]))

    def test_nonzero_gate_checkpointed_backward_and_state_restore(self):
        torch.manual_seed(18)
        text = tiny()
        reverse_scan.install(text)
        module = text.layers[0].linear_attn
        module.backward_gate.data.fill_(.1)
        other = tiny()
        reverse_scan.install(other)
        other.load_state_dict(text.state_dict(), strict=True)
        x = torch.randn(2, 7, 4, requires_grad=True)
        a = checkpoint(module, x, use_reentrant=False)
        b = other.layers[0].linear_attn(x)
        torch.testing.assert_close(a, b, atol=0, rtol=0)
        a.square().mean().backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in text.parameters()))


if __name__ == "__main__":
    unittest.main()
