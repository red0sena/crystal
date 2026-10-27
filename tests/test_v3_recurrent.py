"""Shared-depth execution, padding, gradients and checkpoint persistence."""
import unittest

import torch
from transformers import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel

from qwen_decision.v3 import modeling, recurrent


def tiny(device="cpu"):
    config = Qwen3_5TextConfig(vocab_size=64, hidden_size=32, intermediate_size=64,
        num_hidden_layers=4, num_attention_heads=2, num_key_value_heads=1, head_dim=16,
        linear_key_head_dim=8, linear_value_head_dim=8, linear_num_key_heads=2, linear_num_value_heads=2,
        layer_types=(["full_attention"] * 4 if device == "cpu" else
                     ["linear_attention", "full_attention"] * 2), pad_token_id=0)
    config._attn_implementation = "sdpa"
    model = Qwen3_5TextModel(config).to(device)
    modeling.enable_noncausal(model)
    return model


class RecurrentTests(unittest.TestCase):
    def test_identity_execution_parameter_sharing_and_gate_gradient(self):
        torch.manual_seed(11)
        text = tiny().eval()
        ids = torch.tensor([[1, 2, 3, 4]])
        mask = torch.ones_like(ids)
        expected = text(input_ids=ids, attention_mask=mask, use_cache=False).last_hidden_state.detach()
        original = list(text.layers)
        count = sum(p.numel() for p in text.parameters())
        loop = recurrent.install(text, 0, 2)
        calls = [0] * 4
        handles = [layer.register_forward_hook(lambda m, a, o, i=i: calls.__setitem__(i, calls[i]+1))
                   for i, layer in enumerate(original)]
        actual = text(input_ids=ids, attention_mask=mask, use_cache=False).last_hidden_state
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        self.assertEqual(calls, [2, 2, 1, 1])
        self.assertEqual(sum(p.numel() for p in text.parameters()), count+32)
        self.assertEqual(len(text.layers), text.config.num_hidden_layers)
        self.assertEqual([x.block_type for x in text.layers], text.config.layer_types)
        (actual[:, -1] @ torch.randn(32, 3)).square().mean().backward()
        self.assertTrue(torch.isfinite(loop.recurrence_gate.grad).all())
        self.assertGreater(loop.recurrence_gate.grad.abs().sum().item(), 0.)
        self.assertFalse(any('_layers' in key for key in text.state_dict()))
        for h in handles: h.remove()

    def test_checkpointed_nonzero_gate_gradients_and_padding(self):
        torch.manual_seed(20)
        text = tiny().eval()
        loop = recurrent.install(text, 0, 2)
        loop.recurrence_gate.data.fill_(.2)
        ids = torch.tensor([[1, 2, 3, 4], [5, 6, 0, 0]])
        mask = ids.ne(0)
        batched = text(input_ids=ids, attention_mask=mask, use_cache=False).last_hidden_state
        solo = text(input_ids=ids[1:2, :2], attention_mask=mask[1:2, :2], use_cache=False).last_hidden_state
        torch.testing.assert_close(batched[1, :2], solo[0], atol=3e-6, rtol=3e-6)
        text.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        text.train()
        output = text(input_ids=ids, attention_mask=mask, use_cache=False).last_hidden_state
        torch.testing.assert_close(output, batched, atol=0, rtol=0)
        output.square().mean().backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in text.parameters()))
        with self.assertRaises(ValueError):
            loop(output, use_cache=True)
        with self.assertRaises(ValueError):
            recurrent.install(text, 0, 2)

    def test_state_reload_into_fresh_shared_network(self):
        torch.manual_seed(23)
        text = tiny().eval()
        recurrent.install(text, 0, 2).recurrence_gate.data.fill_(.4)
        other = tiny().eval()
        recurrent.install(other, 0, 2)
        other.load_state_dict(text.state_dict(), strict=True)
        ids = torch.tensor([[1, 4, 6]])
        a = text(input_ids=ids, use_cache=False).last_hidden_state
        b = other(input_ids=ids, use_cache=False).last_hidden_state
        torch.testing.assert_close(a, b, atol=0, rtol=0)

    @unittest.skipUnless(torch.cuda.is_available(), "hybrid DeltaNet requires CUDA")
    def test_hybrid_masks_backward_and_padded_solo_equivalence(self):
        modeling.use_fla_conv()
        text = tiny("cuda").to(torch.bfloat16).eval()
        recurrent.install(text, 0, 2).recurrence_gate.data.fill_(.1)
        ids = torch.tensor([[1, 2, 3, 4, 5, 6], [5, 6, 7, 0, 0, 0]], device="cuda")
        mask = ids.ne(0)
        output = text(input_ids=ids, attention_mask=mask, use_cache=False).last_hidden_state
        solo = text(input_ids=ids[1:2, :3], attention_mask=mask[1:2, :3], use_cache=False).last_hidden_state
        torch.testing.assert_close(output[1, :3], solo[0], atol=.012, rtol=.012)
        text.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant":False})
        text.train()
        text(input_ids=ids, attention_mask=mask, use_cache=False).last_hidden_state.float().square().mean().backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in text.parameters()))


if __name__ == "__main__":
    unittest.main()
