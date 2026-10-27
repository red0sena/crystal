"""Leakage, next-token alignment, chunked gradients and factorial contrasts."""
import copy
import unittest

import torch
import torch.nn.functional as F

from qwen_decision.v3 import masked_adaptation as masked
from qwen_decision.v3.mntp_experiment import ARMS, factorial_bootstrap


class MaskedTests(unittest.TestCase):
    def test_every_target_hidden_and_read_from_previous_position(self):
        row = {"input_ids": [10, 11, 12, 13, 14, 15, 16], "eligible_positions": [2, 3, 4, 5]}
        view = masked.masked_view(row, mask_id=14, seed=41, rate=1)
        self.assertEqual(view["target_positions"], [2, 3, 5])
        self.assertEqual(view["read_positions"], [1, 2, 4])
        self.assertEqual(view["labels"], [12, 13, 15])
        self.assertEqual(view["input_ids"], [10, 11, 14, 14, 14, 14, 16])
        self.assertEqual(row["input_ids"], [10, 11, 12, 13, 14, 15, 16])
        self.assertEqual(view, masked.masked_view(row, mask_id=14, seed=41, rate=1))
        for pos, label in zip(view["target_positions"], view["labels"]):
            self.assertNotEqual(view["input_ids"][pos], label)
        for eligible in [[0], [7], [2, 2]]:
            with self.assertRaises(ValueError):
                masked.masked_view({**row, "eligible_positions": eligible}, mask_id=14, seed=1)

    def test_padding_and_per_example_weighting(self):
        views = [masked.masked_view({"input_ids": [1, 2, 3, 4], "eligible_positions": [1, 2]},
                                    mask_id=9, seed=1, rate=1),
                 masked.masked_view({"input_ids": [5, 6], "eligible_positions": [1]},
                                    mask_id=9, seed=1, rate=1)]
        ids, mask, row, pos, labels, weights = masked.collate(views, 0, "cpu")
        self.assertEqual(ids.tolist(), [[1, 9, 9, 4], [5, 9, 0, 0]])
        self.assertEqual(mask.tolist(), [[1, 1, 1, 1], [1, 1, 0, 0]])
        self.assertEqual(row.tolist(), [0, 0, 1])
        self.assertEqual(pos.tolist(), [0, 1, 0])
        self.assertEqual(labels.tolist(), [2, 3, 6])
        self.assertEqual(weights.tolist(), [.5, .5, 1.])

    def test_full_vocab_chunked_loss_matches_dense_backbone_and_head_gradients(self):
        torch.manual_seed(14)
        source = torch.randn(7, 6)
        a = torch.nn.Linear(6, 5, bias=False)
        b = copy.deepcopy(a)
        weight = torch.randn(19, 5, requires_grad=True)
        other_weight = weight.detach().clone().requires_grad_()
        labels = torch.tensor([0, 1, 4, 6, 9, 15, 18])
        scales = torch.tensor([.5, .5, .2, .2, .2, .2, .2])
        dense = (F.cross_entropy(F.linear(a(source), weight), labels, reduction="none") * scales).sum()
        chunked = masked.chunked_loss(b(source), other_weight, labels, scales, chunk_size=2)
        dense.backward()
        chunked.backward()
        torch.testing.assert_close(chunked, dense)
        torch.testing.assert_close(a.weight.grad, b.weight.grad)
        torch.testing.assert_close(weight.grad, other_weight.grad)

    def test_factorial_interaction_is_not_confused_with_general_mntp_gain(self):
        predictions = {}
        for seed in [41, 73]:
            predictions[seed] = {}
            # MNTP gains equally under both architectures: zero interaction.
            outcomes = [[1, 0, 0, 0], [1, 1, 0, 0], [1, 0, 0, 0], [1, 1, 0, 0]]
            for arm, correct in zip(ARMS, outcomes):
                predictions[seed][arm] = [{"id": str(i), "source": "s", "group": str(i),
                                           "correct": bool(v)} for i, v in enumerate(correct)]
        result = factorial_bootstrap(predictions, draws=100)
        self.assertEqual(result["native_mntp_effect"]["macro_delta"], .25)
        self.assertEqual(result["reverse_mntp_effect"]["macro_delta"], .25)
        self.assertEqual(result["interaction"]["macro_delta"], 0)
        self.assertEqual(result["interaction"]["paired_document_bootstrap_95ci"], [0., 0.])
        predictions[73][ARMS[0]][0]["id"] = "wrong"
        with self.assertRaises(ValueError):
            factorial_bootstrap(predictions, draws=10)


if __name__ == "__main__":
    unittest.main()
