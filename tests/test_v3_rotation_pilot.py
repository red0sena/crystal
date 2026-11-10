"""Check cross-modal map direction, null-space behavior, and scene isolation."""
import json
import unittest

import torch

from qwen_decision.v3.rotation_pilot import fit_rotation, scenes, shuffled_targets


class RotationPilotTest(unittest.TestCase):
    def test_recovers_known_rotation_in_correct_direction(self):
        generator = torch.Generator().manual_seed(31)
        x = torch.randn(128, 8, generator=generator)
        q, _ = torch.linalg.qr(torch.randn(8, 8, generator=generator))
        fitted = fit_rotation(x, x @ q, 1e-8)
        torch.testing.assert_close(fitted, q, atol=2e-5, rtol=2e-5)
        torch.testing.assert_close(fitted.T @ fitted, torch.eye(8), atol=1e-6, rtol=1e-6)

    def test_rank_deficient_identity_pairs_leave_unobserved_space_alone(self):
        x = torch.tensor([[1., 0., 0., 0.], [0., 1., 0., 0.]])
        torch.testing.assert_close(fit_rotation(x, x, .01), torch.eye(4))

    def test_scenes_disjoint_and_negative_control_preserves_prompt(self):
        rows = scenes()
        self.assertEqual(len(rows), 672)
        self.assertEqual(len({json.dumps(r['world']) for r in rows}), len(rows))
        train = [r for r in rows if r['split'] == 'train']
        shuffled = shuffled_targets(train)
        self.assertEqual(sorted(shuffled), list(range(len(train))))
        self.assertGreater(sum(i != j for i, j in enumerate(shuffled)), .95 * len(train))
        for i, j in enumerate(shuffled):
            self.assertEqual(train[i]['question'], train[j]['question'])


if __name__ == '__main__':
    unittest.main()
