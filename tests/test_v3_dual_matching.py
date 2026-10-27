"""Matching alignment, padding, two-way flow, checkpoint replay and export."""
import copy
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from qwen_decision.v3 import dual_matching as dual, format as fmt, modeling
from qwen_decision.v3 import progressive_replacement as progressive, train
from tests.test_v3_hypothesis_workspace import Model, inputs


class Characters:
    all_special_ids = []

    def apply_chat_template(self, messages, **kwargs):
        return "<system>" + messages[0]["content"] + "</system><user>" + messages[1]["content"] + "</user><assistant>"

    def __call__(self, text, **kwargs):
        return {"input_ids": list(map(ord, text)), "offset_mapping": [(i, i + 1) for i in range(len(text))]}


def rows():
    return [{"passage_positions": [1, 2, 3], "question_positions": [4, 5],
             "candidate_positions": [[6, 7], [8]], "target": [1., 0.]},
            {"passage_positions": [1, 2], "question_positions": [3],
             "candidate_positions": [[4], [5, 6], [7]], "target": [0., 1., 0.]}]


class MatchingTests(unittest.TestCase):
    def test_cached_permutation_and_codes_resolve_exact_content(self):
        tokenizer = Characters()
        renderer = SimpleNamespace(tokenizer=tokenizer, args=SimpleNamespace(layout="d3"),
                                   codes=["A", "B", "C"], code_ids=[1, 2, 3])
        raw = {"id": "x", "state": "Question:\nSAME\nOptions:\nB: distractor", "question": {
            "type": "choice", "instructions": "SAME", "criteria": {"left": "SAME", "right": "SAME"}}}
        q = {**raw["question"], "criteria": {"right": "SAME", "left": "SAME"}}
        text = fmt.render(tokenizer, raw["state"], q, ["C", "A"], "d3")
        encoded = {"input_ids": list(map(ord, text)), "order": [1, 0], "code_ids": [3, 1]}
        spans = dual.content_spans(renderer, raw, encoded)
        extract = lambda indices: "".join(text[i] for i in indices)
        self.assertEqual(extract(spans["passage_positions"]), raw["state"])
        self.assertEqual(extract(spans["question_positions"]), "SAME")
        self.assertEqual(list(map(extract, spans["candidate_positions"])), ["right: SAME", "left: SAME"])
        with self.assertRaisesRegex(ValueError, "changed the matched prompt"):
            dual.content_spans(renderer, raw, {**encoded, "input_ids": encoded["input_ids"][:-1]})

    def test_padding_candidate_order_and_bidirectional_token_flow(self):
        torch.manual_seed(13)
        module = dual.DualMatching(16, heads=4)
        hidden = torch.randn(2, 12, 16, requires_grad=True)
        rr = rows()
        base = module(hidden, rr)
        self.assertEqual(base[0, 2].item(), modeling.MASK)
        solo = module(hidden[:1, :9], rr[:1])
        torch.testing.assert_close(solo, base[:1, :2])
        permuted = copy.deepcopy(rr)
        permuted[0]["candidate_positions"].reverse()
        torch.testing.assert_close(module(hidden, permuted)[0, :2], base[0, :2].flip(0))
        base[:, :2].sum().backward()
        self.assertGreater(hidden.grad[0, 1:4].abs().sum().item(), 0.)
        self.assertGreater(hidden.grad[0, 6:9].abs().sum().item(), 0.)
        self.assertEqual(hidden.grad[:, 9:].abs().sum().item(), 0.)
        self.assertTrue(all(p.grad is not None and p.grad.norm() > 0 for p in module.parameters()))
        removed = module(hidden, rr, disable_evidence=True)
        self.assertGreater((removed[:, :2] - base[:, :2]).abs().max().item(), 1e-7)

    def test_checkpoint_replay_and_parameter_matched_control(self):
        torch.manual_seed(14)
        module = dual.DualMatching(16, heads=4)
        control = copy.deepcopy(module)
        control.mode = "self"
        hidden = torch.randn(2, 12, 16, requires_grad=True)
        a = module(hidden, rows())
        a[:, :2].square().sum().backward()
        expected = [p.grad.clone() for p in module.parameters()]
        xgrad = hidden.grad.clone()
        module.zero_grad(set_to_none=True)
        hidden.grad = None
        b = module(hidden, rows(), checkpointed=True)
        module(hidden[:1], rows()[:1], checkpointed=True)
        b[:, :2].square().sum().backward()
        torch.testing.assert_close(a, b, atol=0, rtol=0)
        for p, grad in zip(module.parameters(), expected):
            torch.testing.assert_close(p.grad, grad, atol=0, rtol=0)
        torch.testing.assert_close(hidden.grad, xgrad, atol=0, rtol=0)
        self.assertEqual(sum(p.numel() for p in module.parameters()), sum(p.numel() for p in control.parameters()))
        self.assertGreater((control(hidden, rows())[:, :2] - a[:, :2]).abs().max().item(), 1e-7)

    def test_stock_export_preserves_module_objects_and_exact_weights(self):
        model = Model().eval()
        module = dual.install(model.model.language_model, heads=4)
        expected = copy.deepcopy(module.state_dict())
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            def save_stock(m, tokenizer, out, cfg):
                self.assertFalse(hasattr(m.model.language_model, "dual_matching"))
                torch.save(m.state_dict(), out / "stock.pt")
            with patch.object(modeling, "save", side_effect=save_stock):
                dual.save(model, None, path, {})
            self.assertIs(model.model.language_model.dual_matching, module)
            other = Model()
            other.load_state_dict(torch.load(path / "stock.pt", weights_only=True), strict=True)
            dual.restore_architecture(other, path)
            for name, value in other.model.language_model.dual_matching.state_dict().items():
                self.assertTrue(torch.equal(value, expected[name]))


class ProgressiveTests(unittest.TestCase):
    def test_probability_schedule_and_stateless_branch(self):
        p = [progressive.replacement_probability(i) for i in range(1, 4097)]
        self.assertEqual(p[0], .1)
        self.assertEqual(p[1023:], [1.] * (4096 - 1023))
        self.assertEqual(p, sorted(p))
        before = random.getstate()
        routes = [progressive.use_successor(41, i) for i in range(1, 4097)]
        self.assertEqual(before, random.getstate())
        self.assertTrue(all(routes[1023:]))
        self.assertTrue(any(routes[:1023]) and not all(routes[:1023]))
        self.assertEqual(routes, [progressive.use_successor(41, i) for i in range(1, 4097)])

    def test_predecessor_executes_frozen_blocks_and_dormant_core_has_no_adam_decay(self):
        from qwen_decision.v3 import bdh_memory as bdh
        torch.manual_seed(16)
        model = Model().eval()
        text = model.model.language_model
        bdh.install(text, start=2, heads=2, cycles=3)
        for layer in text.layers[2:]:
            layer.requires_grad_(False)
        x = inputs()
        native = modeling.code_logits(model, *x[:4])
        old = progressive.logits(model, *x[:4], successor=False)
        torch.testing.assert_close(native, old, atol=0, rtol=0)
        counts = [0] * 4
        handles = [l.register_forward_hook(lambda m, a, o, i=i: counts.__setitem__(i, counts[i] + 1))
                   for i, l in enumerate(text.layers)]
        new = progressive.logits(model, *x[:4], successor=True)
        for handle in handles:
            handle.remove()
        self.assertEqual(counts, [1, 1, 0, 0])
        self.assertGreater((new[:, :2] - old[:, :2]).abs().max().item(), 1e-4)
        opt = train.MasterAdamW(model, 1e-4)
        old[:, :2].square().mean().backward()
        self.assertTrue(all(p.grad is None for p in text.bdh_memory.parameters()))
        self.assertTrue(all(p.grad is None for l in text.layers[2:] for p in l.parameters()))
        self.assertTrue(all(p.grad is not None for l in text.layers[:2] for p in l.parameters()))
        before = [p.detach().clone() for p in text.bdh_memory.parameters()]
        opt.clip_grad_norm(1.)
        opt.step()
        for p, value in zip(text.bdh_memory.parameters(), before):
            self.assertTrue(torch.equal(p, value))


if __name__ == "__main__":
    unittest.main()
