"""GPU integration checks using real weights, including vision and no-generation invariants."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image
import torch

from scorer import ChoiceScorer


ROOT = Path(__file__).resolve().parent


class SmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scorer = ChoiceScorer(str(ROOT / "model"), device="cuda")
        cls.results = {}

    @classmethod
    def tearDownClass(cls):
        (ROOT / "smoke-results.json").write_text(
            json.dumps(cls.results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
        )

    def check_result(self, result):
        self.assertEqual(result["usage"]["output_tokens"], 0)
        self.assertAlmostEqual(sum(result["probabilities"].values()), 1.0, places=6)
        self.assertTrue(all(0 <= p <= 1 for p in result["probabilities"].values()))

    def test_text_one_forward_no_generation(self):
        request = json.loads((ROOT / "examples/text.json").read_text())
        calls = []
        hook = self.scorer.model.model.register_forward_hook(lambda *args: calls.append(1))
        try:
            with patch.object(self.scorer.model, "generate", side_effect=AssertionError("generation forbidden")):
                result = self.scorer.score(request)
        finally:
            hook.remove()
        self.assertEqual(len(calls), 1)
        self.check_result(result)
        self.assertEqual(result["choice"], "billing")
        self.results["text"] = result

    def test_image_counterfactual(self):
        request = json.loads((ROOT / "examples/image.json").read_text())
        red = self.scorer.score(request, image_base=ROOT / "examples")
        self.check_result(red)
        self.assertEqual(red["choice"], "red")
        with tempfile.TemporaryDirectory() as temporary:
            image_path = Path(temporary) / "blue.png"
            Image.new("RGB", (256, 256), (0, 0, 255)).save(image_path)
            request["images"] = [str(image_path)]
            blue = self.scorer.score(request)
        self.check_result(blue)
        self.assertEqual(blue["choice"], "blue")
        self.results["red_image"] = red
        self.results["blue_image"] = blue

    def test_selected_head_matches_full_head(self):
        request = json.loads((ROOT / "examples/image.json").read_text())
        inputs, labels, _ = self.scorer.prepare(request, image_base=ROOT / "examples")
        ids = [item["token_id"] for item in labels]
        selected = self.scorer.option_logits(inputs, ids, head_mode="selected")
        full = self.scorer.option_logits(inputs, ids, head_mode="full")
        # Allow a BF16 rounding step for different GEMM shapes.
        torch.testing.assert_close(selected, full, atol=0.25, rtol=0)
        torch.testing.assert_close(selected.softmax(0), full.softmax(0), atol=0.03, rtol=0)
        self.results["head_equivalence"] = {
            "selected_logits": selected.cpu().tolist(), "full_logits": full.cpu().tolist(),
            "max_abs_logit_difference": float((selected - full).abs().max()),
        }

    def test_long_text(self):
        request = json.loads((ROOT / "examples/text.json").read_text())
        filler = "Background record: the office is open and the order has been processed.\n"
        request["state"] = filler * 300 + request["state"] + "\n" + filler * 300
        with patch.object(self.scorer.model, "generate", side_effect=AssertionError("generation forbidden")):
            result = self.scorer.score(request)
        self.check_result(result)
        self.assertGreater(result["usage"]["input_tokens"], 8000)
        # Long-context task accuracy is reported, not assumed from context support.
        self.results["long_text"] = result

    def test_overflow_rejected_before_forward(self):
        request = json.loads((ROOT / "examples/text.json").read_text())
        original_limit = self.scorer.max_input_tokens
        self.scorer.max_input_tokens = 1
        try:
            with patch.object(self.scorer.model.model, "forward", side_effect=AssertionError("forward forbidden")):
                with self.assertRaisesRegex(ValueError, "Input was not truncated"):
                    self.scorer.score(request)
        finally:
            self.scorer.max_input_tokens = original_limit


if __name__ == "__main__":
    unittest.main(verbosity=2)
