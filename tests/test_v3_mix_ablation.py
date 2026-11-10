import math
import unittest
from collections import Counter

import numpy as np

from qwen_decision.v3 import mix_ablation as m


class MixtureTests(unittest.TestCase):
    full = {"a": 900, "b": 50, "c": 50, m.URL: 100}
    old = {"a": 10, "b": 30, "c": 60}

    def test_r2_is_natural_share(self):
        q = m.target_shares("r2", self.full, self.old)
        self.assertAlmostEqual(q["a"], 900/1100)
        self.assertAlmostEqual(sum(q.values()), 1.)

    def test_r1_uses_old_shares_and_keeps_url_natural(self):
        q1 = m.target_shares("r1", self.full, self.old)
        q2 = m.target_shares("r2", self.full, self.old)
        self.assertAlmostEqual(q1[m.URL], q2[m.URL])
        self.assertAlmostEqual(sum(q1.values()), 1.)
        rest = 1 - q1[m.URL]
        self.assertAlmostEqual(q1["a"], rest*.1)
        self.assertAlmostEqual(q1["c"], rest*.6)

    def test_r1_rejects_old_source_missing_from_pool(self):
        with self.assertRaises(AssertionError):
            m.target_shares("r1", {"a": 5}, {"a": 1, "gone": 1})

    def test_quotas_sum_exactly_and_follow_shares(self):
        need = m.quotas(1000, {"a": 1/3, "b": 1/3, "c": 1/3})
        self.assertEqual(sum(need.values()), 1000)
        self.assertTrue(all(abs(v - 1000/3) < 1 for v in need.values()))

    def test_draws_cover_source_before_repeating_and_are_deterministic(self):
        sources = np.array([0]*5 + [1]*20, dtype=np.int16)
        names = ["small", "big"]
        draws = m.text_draws(sources, names, {"small": 12, "big": 4}, seed=3)
        self.assertEqual(len(draws), 16)
        small = Counter(int(i) for i in draws if i < 5)
        self.assertEqual(sorted(small.values()), [2, 2, 2, 3, 3])
        self.assertEqual(len({int(i) for i in draws if i >= 5}), 4)
        np.testing.assert_array_equal(draws, m.text_draws(sources, names, {"small": 12, "big": 4}, seed=3))

    def test_schedule_shape_resume_and_images(self):
        texts = [{"id": f"t{i}", "modality": "text"} for i in range(10)]
        images = [{"id": f"i{i}", "modality": "image"} for i in range(3)]
        draws = np.arange(8) % 10
        s = m.MixSchedule(texts, images, draws, steps=2, seed=1, text_take=4, image_take=2)
        self.assertEqual(len(s), 2)
        rows = [r for i in range(2) for r, _ in s[i]]
        self.assertEqual(sum(r["modality"] == "text" for r in rows), 8)
        self.assertEqual(Counter(r["id"] for r in rows if r["modality"] == "image").most_common(1)[0][1], 2)
        again = m.MixSchedule(texts, images, draws, steps=2, seed=1, text_take=4, image_take=2)
        self.assertEqual(s[1], again[1])
        with self.assertRaises(IndexError):
            s[2]

    def test_lr_matches_v2_formula(self):
        self.assertAlmostEqual(m.lr_factor(256), .5)
        self.assertAlmostEqual(m.lr_factor(m.WARMUP), 1.)
        self.assertAlmostEqual(m.lr_factor(m.STEPS), .1)
        mid = m.WARMUP + (m.STEPS - m.WARMUP)/2
        self.assertAlmostEqual(m.lr_factor(mid), .1 + .45*(1+math.cos(math.pi/2)))


if __name__ == "__main__":
    unittest.main()
