"""Evaluation trainability is canonical, gradient-free, and fully reversible."""
import unittest

import torch
from qwen_decision.v3 import retrofit_recovery as recovery


class EvaluationContextTests(unittest.TestCase):
    def model(self):
        model = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Linear(4, 2))
        model[1].requires_grad_(False)
        model._retrofit_evaluation_flags = {n: p.requires_grad for n, p in model.named_parameters()}
        return model

    def test_frozen_caller_uses_original_mixed_flags_without_gradients(self):
        model = self.model().requires_grad_(False)
        expected_weights = {n: p.detach().clone() for n, p in model.named_parameters()}
        def evaluate(current):
            self.assertFalse(torch.is_grad_enabled())
            self.assertEqual({n: p.requires_grad for n, p in current.named_parameters()}, current._retrofit_evaluation_flags)
            return current(torch.ones(2, 3))
        result = recovery.canonical_evaluation(model, evaluate)
        self.assertFalse(result.requires_grad)
        self.assertTrue(all(not p.requires_grad and p.grad is None for p in model.parameters()))
        for n, p in model.named_parameters():
            self.assertTrue(torch.equal(p, expected_weights[n]))

    def test_exception_restores_caller_flags_and_training_caller_is_unchanged(self):
        model = self.model()
        flags = {n: p.requires_grad for n, p in model.named_parameters()}
        def fail(current):
            self.assertFalse(torch.is_grad_enabled())
            raise ValueError("evaluation failed")
        with self.assertRaisesRegex(ValueError, "evaluation failed"):
            recovery.canonical_evaluation(model, fail)
        self.assertEqual({n: p.requires_grad for n, p in model.named_parameters()}, flags)
        model.requires_grad_(False)
        with self.assertRaises(ValueError):
            recovery.canonical_evaluation(model, fail)
        self.assertTrue(all(not p.requires_grad for p in model.parameters()))


if __name__ == "__main__":
    unittest.main()
