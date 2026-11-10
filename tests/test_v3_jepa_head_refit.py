import unittest

import numpy as np
import torch
import torch.nn.functional as F

from qwen_decision.v3 import jepa_decision as exp, jepa_head_refit as refit


class HeadRefitTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_vectorized_probabilities_loss_and_gradients_match_original(self):
        torch.manual_seed(3)
        raw = torch.randn(2, 12, 32, dtype=torch.float64, requires_grad=True)
        specs = exp.sr.query_specs()
        original = torch.stack([exp.FactoredDecision.answer_logits(x, specs) for x in raw])
        vector = refit.answer_logits(raw)
        torch.testing.assert_close(vector, original, atol=1e-12, rtol=1e-12)
        rows = exp.sr.make_scenes()[:12]
        gold = refit.gold_table(rows, specs, "cpu")
        task = torch.tensor([exp.sr.TASKS.index(x['task']) for x in specs])
        seen = torch.tensor([x['key'] not in exp.sr.HOLDOUT for x in specs])
        old_loss = torch.stack([exp.score(x, gold, task, seen)[0] for x in original])
        new_loss = refit.losses(vector, gold, specs)
        torch.testing.assert_close(new_loss, old_loss, atol=1e-12, rtol=1e-12)
        old_grad = torch.autograd.grad(old_loss.sum(), raw, retain_graph=True)[0]
        new_grad = torch.autograd.grad(new_loss.sum(), raw)[0]
        torch.testing.assert_close(new_grad, old_grad, atol=1e-12, rtol=1e-12)

    def test_batched_adamw_and_individual_clip_equal_separate_optimizers(self):
        batch = refit.Heads(seeds=(701, 719), conditions=("raw_clean",)).double()
        separate = [exp.FactoredDecision().double() for _ in range(2)]
        for i, head in enumerate(separate): head.load_state_dict(batch.single_state(i))
        inputs = torch.randn(2, 12, exp.DIM, dtype=torch.float64)
        specs = exp.sr.query_specs(); gold = refit.gold_table(exp.sr.make_scenes()[:12], specs, "cpu")
        opt = torch.optim.AdamW(batch.parameters(), lr=.001, weight_decay=.0001)
        others = [torch.optim.AdamW(h.parameters(), lr=.001, weight_decay=.0001) for h in separate]
        for step in range(3):
            opt.zero_grad(); refit.losses(batch(inputs), gold, specs).sum().mul(100).backward()
            refit.clip_heads(batch); opt.step()
            for i, (head, optimizer) in enumerate(zip(separate, others)):
                optimizer.zero_grad()
                refit.losses(head(inputs[i], specs)[None], gold, specs).sum().mul(100).backward()
                torch.nn.utils.clip_grad_norm_(head.parameters(), 1.); optimizer.step()
                for key, value in batch.single_state(i).items():
                    torch.testing.assert_close(value, head.state_dict()[key], atol=1e-12, rtol=1e-12)

    def test_noise_relative_rms_and_clean_controls(self):
        raw = torch.randn(20000, exp.DIM)
        unit = F.normalize(raw, dim=-1)
        ids = torch.arange(len(raw)); sigma = 1.7
        inputs = refit.training_inputs(raw, unit, ids, 1, sigma, "cpu")
        for i, (condition, seed) in enumerate(refit.COMBINATIONS):
            if condition == "raw_clean": torch.testing.assert_close(inputs[i], raw)
            elif condition == "unit_clean": torch.testing.assert_close(inputs[i], unit)
            else:
                rms = float((inputs[i] - unit).square().sum(-1).mean().sqrt())
                self.assertAlmostEqual(rms, sigma, delta=.01)

    def test_scene_records_match_original_and_crossed_weights(self):
        specs = exp.sr.query_specs(); rows = exp.sr.make_scenes()
        gold = refit.gold_table(rows[1280:], specs, "cpu")
        logits = refit.answer_logits(torch.randn(2, 256, 32))
        items = [x for x in exp.sr.examples(rows, specs) if x['scene'] >= 1280]
        actual = refit.records_from_logits(logits, gold, specs)
        for i in range(2): np.testing.assert_array_equal(actual[i], exp.scene_records(logits[i], items, gold))
        wh, ws = refit.bootstrap_weights(17)
        np.testing.assert_array_equal(wh.sum((1, 2)), np.full(17, 30))
        np.testing.assert_array_equal(ws.sum(1), np.full(17, 256))
        synthetic = np.tile(actual[0], (10, 3, 1, 1, 1, 1))
        scores = refit.bootstrap_scores(synthetic, wh, ws)
        for b in range(len(ws)):
            counts = (actual[0] * ws[b, :, None, None, None]).sum(0)
            expected = (counts[..., :2] / counts[..., 2:3]).mean(1)
            np.testing.assert_allclose(scores[b], expected, rtol=1e-12, atol=1e-12)
        # Vary encoder and head scores to check the actual crossed weighting.
        for encoder in range(10):
            for head in range(3): synthetic[encoder, head, ..., 0] *= (encoder + head + 1) / 13
        varied = refit.bootstrap_scores(synthetic, wh, ws)
        for b in range(3):
            counts = np.zeros((2, 3, 3))
            for encoder in range(10):
                for head in range(3):
                    counts += (synthetic[encoder, head] * ws[b, :, None, None, None]).sum(0) * wh[b, encoder, head]
            np.testing.assert_allclose(varied[b], (counts[..., :2] / counts[..., 2:3]).mean(1), rtol=1e-12, atol=1e-12)


if __name__ == "__main__": unittest.main()
