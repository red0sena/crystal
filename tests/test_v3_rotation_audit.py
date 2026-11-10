"""Analytical checks for the post-hoc controls, independent of the pilot fit."""
import unittest

import torch

from qwen_decision.v3.rotation_audit import mean_direction_rotation, linear_cka, knn_overlap


class RotationAuditTest(unittest.TestCase):
    def test_mean_direction_rotates_plane_and_preserves_complement(self):
        x = torch.tensor([[1., 0., 0.], [3., 0., 0.]])
        y = torch.tensor([[0., 2., 0.], [0., 4., 0.]])
        rotation = mean_direction_rotation(x, y)
        expected = torch.tensor([[0., 1., 0.], [-1., 0., 0.], [0., 0., 1.]], dtype=torch.float64)
        torch.testing.assert_close(rotation, expected)
        torch.testing.assert_close(rotation.T @ rotation, torch.eye(3, dtype=torch.float64))

    def test_geometry_and_effective_head_equivalence(self):
        generator = torch.Generator().manual_seed(49)
        x, y = (torch.randn(24, 5, generator=generator, dtype=torch.float64) for _ in range(2))
        rotation, _ = torch.linalg.qr(torch.randn(5, 5, generator=generator, dtype=torch.float64))
        head = torch.randn(3, 5, generator=generator, dtype=torch.float64)
        torch.testing.assert_close((x @ rotation) @ (x @ rotation).T, x @ x.T)
        self.assertAlmostEqual(linear_cka(x @ rotation, y), linear_cka(x, y), places=12)
        self.assertEqual(knn_overlap(x @ rotation, y), knn_overlap(x, y))
        torch.testing.assert_close((x @ rotation) @ head.T, x @ (head @ rotation.T).T)


if __name__ == '__main__':
    unittest.main()
