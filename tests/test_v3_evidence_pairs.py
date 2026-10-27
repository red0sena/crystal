"""Gold pair validity, alignment invariances, mechanism metrics and interaction."""
import copy
import unittest

import torch
from qwen_decision.v3 import evidence_pairs as ep, evidence_pair_data as data
from qwen_decision.v3.evidence_pair_experiment import factorial_bootstrap, ARMS


def row(ident, passage, label, *, question="Did the project receive support?"):
    return {"id": ident, "item": "doc1", "group": "doc1", "source": "condaqa-train",
            "state": {"passage": passage, "question": question}, "label_origin": "human",
            "question": {"type": "choice", "instructions": "Answer the question.",
                         "criteria": dict.fromkeys(ep.LABELS["condaqa-train"])},
            "target": [float(i == label) for i in range(3)]}


class GoldPairTests(unittest.TestCase):
    def test_only_evidence_changes_and_original_gold_is_preserved(self):
        a = row("a", "The project received support from the local government yesterday.", 0)
        b = row("b", "The project did not receive support from the local government yesterday.", 1)
        c = row("c", "The project received support from the local government last week.", 0)
        original = copy.deepcopy([a, b, c])
        pairs, _ = ep.make_pairs([a, b, c])
        self.assertEqual({p["kind"] for p in pairs}, {"flip", "invariant"})
        self.assertEqual([a, b, c], original)
        for p in pairs:
            self.assertEqual(p["a"]["state"]["question"], p["b"]["state"]["question"])
            self.assertEqual(p["a"]["question"], p["b"]["question"])
            self.assertNotEqual(ep.evidence(p["a"]), ep.evidence(p["b"]))
        shuffled = copy.deepcopy(a)
        shuffled["question"]["criteria"] = {"no": None, "the passage does not say": None, "yes": None}
        shuffled["target"] = [0., 0., 1.]
        self.assertEqual(ep.canonical(shuffled)["target"], [1., 0., 0.])

    def test_conflicting_duplicate_and_different_question_not_a_flip_pair(self):
        a = row("a", "The project received support from the local government yesterday.", 0)
        b = row("b", "The project did not receive support from the local government yesterday.", 1)
        b["state"]["question"] = "Who paid for the project?"
        pairs, _ = ep.make_pairs([a, b])
        self.assertEqual(pairs, [])
        b = {**a, "id": "b", "target": [0., 1., 0.]}
        pairs, rejected = ep.make_pairs([a, b])
        self.assertEqual(pairs, [])
        self.assertEqual(rejected["conflicting_identical_evidence_group"], 1)

    def test_article_and_original_passage_keys_bind_distinct_variants(self):
        a = row("a", "The project received support from the local government yesterday.", 0)
        a["item"] = "condaqa-train:test1"
        b = row("b", "The project did not receive support from the local government yesterday.", 1)
        b["item"] = "condaqa-train:test2"
        meta = ({}, {a["item"]: "same-original", b["item"]: "same-original"})
        ka, kb = set(data.keys(a, meta)), set(data.keys(b, meta))
        self.assertIn(("condaqa_original_passage", "same-original"), ka & kb)

    def test_short_structured_context_is_not_claimed_as_a_new_document(self):
        self.assertFalse(data.independent_retention({"state": {"premise": "A woman walks.", "hypothesis": "A person moves."}}))
        self.assertTrue(data.independent_retention({"state": {"premise": "A detailed and sufficiently long original passage. " * 3,
                                                               "hypothesis": "A person moves."}}))
        self.assertTrue(data.independent_retention({"state": "A woman walks."}))

    def test_distant_edits_and_punctuation_only_changes_are_excluded(self):
        a = row("a", "The project received support from the local government yesterday.", 0)
        b = row("b", "The project did not receive support from the federal government yesterday.", 1)
        self.assertEqual(ep.make_pairs([a, b])[0], [])
        b = row("b", "The project received support from the local government yesterday!", 1)
        self.assertEqual(ep.make_pairs([a, b])[0], [])


class PairedLossTests(unittest.TestCase):
    def test_gold_preference_change_and_offsets_swaps_code_permutations(self):
        target = torch.tensor([[1., 0., 0.], [0., 1., 0.]])
        wrong = torch.tensor([[2., 0., 0.], [2., 0., 0.]], requires_grad=True)
        right = torch.tensor([[2., 0., 0.], [0., 2., 0.]], requires_grad=True)
        loss, delta = ep.margin_loss(right, target)
        bad, _ = ep.margin_loss(wrong, target)
        self.assertLess(loss.item(), bad.item())
        self.assertEqual(delta.item(), 4.)
        for z, y in [(right.flip(0), target.flip(0)),
                     (right[:, [2, 0, 1]], target[:, [2, 0, 1]]),
                     (right + torch.tensor([[100.], [-10.]]), target)]:
            other, other_delta = ep.margin_loss(z, y)
            torch.testing.assert_close(other, loss)
            torch.testing.assert_close(other_delta, delta)
        loss.backward()
        self.assertTrue(torch.isfinite(right.grad).all())
        self.assertLess(right.grad[0, 0].item(), 0.)
        self.assertLess(right.grad[1, 1].item(), 0.)
        with self.assertRaises(ValueError):
            ep.margin_loss(right, target[[0, 0]])

    def test_both_correct_metric_cannot_be_gamed_by_always_changing(self):
        rr = [{"id": "1", "source": "s", "group": "g1", "kind": "flip", "correct": True, "prediction_changed": True},
              {"id": "2", "source": "s", "group": "g2", "kind": "flip", "correct": False, "prediction_changed": True},
              {"id": "3", "source": "s", "group": "g3", "kind": "invariant", "correct": False, "prediction_changed": True}]
        m = ep.response_metrics(rr)
        self.assertEqual(m["flip"]["prediction_change_rate"], 1.)
        self.assertEqual(m["flip"]["macro_both_correct"], .5)
        self.assertEqual(m["invariant"]["both_correct"], 0.)

    def test_equal_learning_gain_is_not_a_structural_interaction(self):
        pp = {}
        for seed in (41, 73):
            pp[seed] = {}
            for arm, values in zip(ARMS, [[0, 0, 1, 1], [0, 1, 1, 1], [0, 0, 1, 1], [0, 1, 1, 1]]):
                pp[seed][arm] = [{"id": str(i), "source": "s", "group": str(i), "kind": "flip", "correct": bool(v)}
                                 for i, v in enumerate(values)]
        result = factorial_bootstrap(pp, draws=100)
        self.assertEqual(result["native_pair_effect"]["delta"], .25)
        self.assertEqual(result["interaction"]["delta"], 0.)
        self.assertEqual(result["interaction"]["paired_document_95ci"], [0., 0.])
        for seed in (41, 73):
            pp[seed]["dual_pair"][0]["correct"] = True
        self.assertEqual(factorial_bootstrap(pp, draws=100)["interaction"]["delta"], .25)


if __name__ == "__main__":
    unittest.main()
