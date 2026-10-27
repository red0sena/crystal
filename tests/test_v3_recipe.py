"""Check the FP32 weight averages of the recipe experiment."""
import unittest

import torch

from qwen_decision.v3.recipe_experiment import Averages


class AveragesTest(unittest.TestCase):
    def test_ema_and_tail_mean(self):
        master = torch.zeros(3)
        averages = Averages([master], decay=.5, tail_start=3)
        history = []
        for step in range(1, 6):
            master.fill_(float(step))
            history.append(master.clone())
            averages.update(step)
        ema = torch.zeros(3)
        for value in history:
            ema = .5 * ema + .5 * value
        torch.testing.assert_close(averages.ema[0], ema)
        torch.testing.assert_close(averages.tail[0], torch.stack(history[2:]).mean(0))
        self.assertEqual(averages.count, 3)

    def test_averages_are_copies(self):
        master = torch.ones(2)
        averages = Averages([master], decay=.9, tail_start=1)
        averages.update(1)
        master.fill_(5.)
        torch.testing.assert_close(averages.tail[0], torch.ones(2))


if __name__ == "__main__":
    unittest.main()
