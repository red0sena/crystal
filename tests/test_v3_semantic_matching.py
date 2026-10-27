"""Semantic role alignment, initial native identity, one-call scoring and controls."""
import copy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn

from qwen_decision.v3 import semantic_matching as sem, format as fmt, modeling
from qwen_decision.v3 import single_forward_experiment as experiment
from tests.test_v3_dual_matching import Characters


class Text(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(64, 16)
        self.layers = nn.ModuleList([nn.Linear(16, 16, bias=False)])
        self.calls = 0

    def forward(self, input_ids, **kwargs):
        self.calls += 1
        value = self.embed_tokens(input_ids)
        return SimpleNamespace(last_hidden_state=self.layers[0](value))


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module(); self.model.language_model = Text()
        self.head = nn.Linear(16, 64, bias=False)

    def get_output_embeddings(self):
        return self.head


def rows():
    return [{"generic_passage_positions": [0, 1, 2], "generic_question_positions": [3],
             "semantic_evidence_positions": [0, 1], "semantic_query_positions": [2],
             "candidate_positions": [[4], [5]], "target": [1., 0.]},
            {"generic_passage_positions": [0, 1], "generic_question_positions": [2],
             "semantic_evidence_positions": [0], "semantic_query_positions": [1],
             "candidate_positions": [[2], [3]], "target": [0., 1.]}]


def inputs():
    ids = torch.tensor([[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 0, 0]])
    codes = torch.tensor([[2, 3], [3, 5]])
    return ids, ids.ne(0), codes, torch.ones_like(codes, dtype=torch.bool)


class SemanticMatchingTests(unittest.TestCase):
    def test_actual_query_and_evidence_are_selected_even_with_escaped_duplicate_text(self):
        tok = Characters()
        renderer = SimpleNamespace(tokenizer=tok, args=SimpleNamespace(layout="d3"), codes=["A", "B", "C"], code_ids=[1, 2, 3])
        for p, q in (("passage", "question"), ("evidence", "claim"), ("premise", "hypothesis")):
            state = {p: 'The text quotes "is café open?". The claim is elsewhere.\nYes.', q: 'is café open?'}
            raw = {"id": "x", "state": state, "question": {"type": "choice", "instructions": "Use the evidence.",
                   "criteria": {"yes": None, "no": None}}}
            question = {**raw["question"], "criteria": {"no": None, "yes": None}}
            text = fmt.render(tok, state, question, ["C", "A"], "d3")
            encoded = {"input_ids": list(map(ord, text)), "order": [1, 0], "code_ids": [3, 1]}
            sp = sem.content_spans(renderer, raw, encoded)
            extract = lambda ix: "".join(text[i] for i in ix)
            self.assertEqual(extract(sp["semantic_query_positions"]), 'is café open?')
            self.assertEqual(extract(sp["generic_question_positions"]), "Use the evidence.")
            self.assertIn('\\"is café open?\\"', extract(sp["semantic_evidence_positions"]))
            self.assertFalse(set(sp["semantic_query_positions"]) & set(sp["semantic_evidence_positions"]))
            self.assertEqual(list(map(extract, sp["candidate_positions"])), ["no", "yes"])
            self.assertEqual(sp["semantic_role"], f"{p}:{q}")
            # Target changes cannot change routing.
            self.assertEqual(sp, sem.content_spans(renderer, raw | {"target": [0., 1.]}, encoded))

    def test_plain_state_fallback_and_token_mismatch_rejection(self):
        tok = Characters()
        renderer = SimpleNamespace(tokenizer=tok, args=SimpleNamespace(layout="d3"), codes=["A", "B"], code_ids=[1, 2])
        raw = {"id": "x", "state": "A passage.", "question": {"type": "noul", "instructions": "Was it open?"}}
        text = fmt.render(tok, raw["state"], raw["question"], ["A", "B"], "d3")
        e = {"input_ids": list(map(ord, text)), "order": [0, 1], "code_ids": [1, 2]}
        sp = sem.content_spans(renderer, raw, e)
        self.assertEqual(sp["generic_question_positions"], sp["semantic_query_positions"])
        self.assertEqual(sp["generic_passage_positions"], sp["semantic_evidence_positions"])
        with self.assertRaisesRegex(ValueError, "changed input tokens"):
            sem.content_spans(renderer, raw, e | {"input_ids": e["input_ids"][:-1]})

    def test_initial_native_function_and_parameter_matched_initialization(self):
        torch.manual_seed(44)
        base = Model().eval(); args = inputs()
        expected = modeling.code_logits(base, *args)
        states = []
        for arm in sem.POLICIES:
            model = copy.deepcopy(base)
            torch.manual_seed(17)
            module = sem.install(model.model.language_model, arm, heads=4)
            model.eval()
            self.assertTrue(torch.equal(sem.logits(model, *args, rows()), expected))
            states.append(copy.deepcopy(module.state_dict()))
        for other in states[1:]:
            self.assertEqual(other.keys(), states[0].keys())
            for name, value in other.items():
                self.assertTrue(torch.equal(value, states[0][name]))

    def test_multi_interventions_use_one_encoding_and_do_not_modify_hidden_or_rows(self):
        torch.manual_seed(18)
        model = Model().eval(); module = sem.install(model.model.language_model, "semantic", heads=4)
        nn.init.normal_(module.scorer.weight, std=.2)
        rr = rows(); original = copy.deepcopy(rr); args = inputs()
        calls = model.model.language_model.calls
        many = sem.multi_logits(model, *args, rr)
        self.assertEqual(model.model.language_model.calls - calls, 1)
        self.assertEqual(rr, original)
        native = modeling.code_logits(model, *args)
        self.assertTrue(torch.equal(many["branch_off"], native))
        self.assertGreater((many["active"] - many["question_zero"]).abs().max().item(), 0.)
        self.assertGreater((many["active"] - many["evidence_zero"]).abs().max().item(), 0.)
        self.assertGreater((many["active"] - many["generic_route"]).abs().max().item(), 0.)
        for mode, value in many.items():
            torch.testing.assert_close(value, sem.logits(model, *args, rr, intervention=mode), atol=0, rtol=0)

    def test_candidate_permutation_padding_and_checkpoint_gradients(self):
        torch.manual_seed(19)
        module = sem.ResidualMatching(16, heads=4).train()
        nn.init.normal_(module.scorer.weight, std=.2)
        x = torch.randn(2, 9, 16, requires_grad=True); rr = rows()
        value = module(x, rr); value.sum().backward()
        reference = {n: p.grad.clone() for n, p in module.named_parameters()}; xgrad = x.grad.clone()
        self.assertGreater(xgrad[0, :3].abs().sum().item(), 0.)
        self.assertEqual(xgrad[:, 6:].abs().sum().item(), 0.)
        module.zero_grad(set_to_none=True); x.grad = None
        replay = module(x, rr, checkpointed=True)
        module(x[:1], rr[:1], checkpointed=True)
        replay.sum().backward()
        torch.testing.assert_close(value, replay, atol=0, rtol=0)
        torch.testing.assert_close(x.grad, xgrad, atol=0, rtol=0)
        for n, p in module.named_parameters():
            torch.testing.assert_close(p.grad, reference[n], atol=0, rtol=0)
        permuted = copy.deepcopy(rr)
        for r in permuted: r["candidate_positions"].reverse()
        torch.testing.assert_close(module(x, permuted), value.flip(1))
        torch.testing.assert_close(module(x[1:, :4], rr[1:]), value[1:])

    def test_stock_export_and_architecture_reload_preserve_scores(self):
        torch.manual_seed(20)
        model = Model().eval(); module = sem.install(model.model.language_model, "semantic", heads=4)
        nn.init.normal_(module.scorer.weight, std=.2)
        expected = sem.multi_logits(model, *inputs(), rows())
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            def save(m, tokenizer, out, cfg):
                self.assertFalse(hasattr(m.model.language_model, "semantic_matching"))
                torch.save(m.state_dict(), out / "stock.pt")
            def load(*args, **kwargs):
                other = Model()
                other.load_state_dict(torch.load(path / "stock.pt", weights_only=True), strict=True)
                return other
            with patch.object(modeling, "save", side_effect=save):
                sem.save(model, None, path, {})
            self.assertIs(model.model.language_model.semantic_matching, module)
            with patch.object(modeling, "load", side_effect=load), patch.object(modeling, "read_config", return_value={"attention": "noncausal", "deltanet": "causal"}):
                other = sem.load(path, device="cpu").eval()
            actual = sem.multi_logits(other, *inputs(), rows())
            for key in expected: torch.testing.assert_close(expected[key], actual[key], atol=0, rtol=0)

    def test_architecture_and_path_effects_are_distinct_and_seed_aligned(self):
        def predictions(active):
            return {"active": [{"id": f"x{i}", "source": "s", "correct": active} for i in range(4)],
                    "branch_off": [{"id": f"x{i}", "source": "s", "correct": False} for i in range(4)],
                    "question_zero": [{"id": f"x{i}", "source": "s", "correct": False} for i in range(4)]}
        pp = {s: {a: predictions(a == "semantic") for a in experiment.ARMS} for s in experiment.SEEDS}
        r = experiment.summarize_contrasts(pp, draws=32)
        self.assertEqual(len(r), 5)
        for v in r.values():
            self.assertEqual(v["delta"], 1.)
            self.assertEqual(v["familywise_hierarchical_95ci"], [1., 1.])
        for s in pp:
            pp[s]["generic"] = predictions(True)
        r = experiment.summarize_contrasts(pp, draws=32)
        self.assertEqual(r["semantic_minus_generic"]["delta"], 0.)
        self.assertEqual(r["semantic_minus_self_question_path_dependence"]["delta"], 1.)
        # Identical harm from zeroing any extra reader is not specific evidence
        # that semantic cross-attention relieved a question-routing bottleneck.
        for s in pp:
            pp[s]["self"] = predictions(True)
        r = experiment.summarize_contrasts(pp, draws=32)
        self.assertEqual(r["semantic_minus_self_question_path_dependence"]["delta"], 0.)


if __name__ == "__main__":
    unittest.main()
