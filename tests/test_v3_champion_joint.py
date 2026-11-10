import unittest
from unittest.mock import patch
import numpy as np
from qwen_decision.v3 import champion_joint_data as data, format as fmt, vision_train as vt
from qwen_decision.v3.champion_joint_train import rotation_diagnostic
from qwen_decision.v3 import train


class JointDataTest(unittest.TestCase):
    def test_text_diagnostic_reaches_native_renderer_with_unchanged_prompt_and_target(self):
        class Tokenizer:
            def apply_chat_template(self, messages, **kwargs): return repr(messages) + repr(kwargs)
            def encode(self, text, **kwargs): return list(text.encode())
        raw = {"id": "diagnostic-1", "source": "fresh", "state": "a real situation", "gold": 1,
               "question": {"type": "choice", "instructions": "Pick?", "criteria": {"no": None, "yes": None}}}
        row = data.text_example(raw); tok = Tokenizer()
        with patch.object(fmt, "answer_codes", return_value=(["A", "B"], [65, 66])):
            out = train.Renderer(tok, vt.TEXT_ARGS, [])(row, None)
        native = fmt.render(tok, raw["state"], raw["question"], ["A", "B"], "d3")
        self.assertEqual(out["input_ids"], tok.encode(native))
        self.assertEqual(out["target"], [0., 1.])
        self.assertEqual(out["hard_code"], "B")
        self.assertEqual(out["kind"], "choice")

    def test_subsetting_keeps_the_gold_and_only_existing_alternatives(self):
        row = {"options": ["red", "blue", "green", "white", "black"], "gold": 2, "ordered": False}
        sizes = set()
        for seed in range(20):
            out = data.subset_options(row, seed)
            self.assertEqual(out["options"][out["gold"]], "green")
            self.assertTrue(set(out["options"]) <= set(row["options"]))
            sizes.add(len(out["options"]))
        self.assertEqual(sizes, {3, 4, 5})

    def test_ordered_options_are_preserved(self):
        row = {"options": ["1", "2", "3", "4"], "gold": 2, "ordered": True}
        self.assertEqual(data.subset_options(row, 1), row)

    def test_diagnostic_vision_prompt_preserves_native_options(self):
        row = {"id": "x", "source": "s", "question": {"type": "choice", "instructions": "Count?",
               "criteria": {"A": "1", "B": "2"}}, "images": [], "gold": 1, "group": "g"}
        converted = data.vision_example(row)
        q, assigned, target = vt.item_view(converted, None)
        self.assertEqual(fmt.options(q), fmt.options(row["question"]))
        self.assertEqual(target, [0., 1.])

    def test_balanced_sampling_respects_small_sources_without_repeating_rows(self):
        result = data.take_balanced({"a": ["a1"], "b": ["b1", "b2", "b3"]}, 4)
        self.assertEqual(result, ["a1", "b1", "b2", "b3"])
        with self.assertRaises(ValueError): data.take_balanced({"a": [1]}, 2)


class RotationTest(unittest.TestCase):
    def test_recovers_rotation_on_unseen_pairs_against_shuffled_control(self):
        rng = np.random.default_rng(41)
        image = rng.normal(size=(192, 16))
        orthogonal, _ = np.linalg.qr(rng.normal(size=(16, 16)))
        text = image @ orthogonal + rng.normal(size=16)
        fit = np.arange(192) < 128
        report = rotation_diagnostic(image, text, fit)
        self.assertTrue(report["eligible"])
        self.assertGreater(report["orthogonal"]["paired_cosine"], .999999)
        self.assertEqual(report["orthogonal"]["image_to_text_r1"], 1.)
        self.assertLess(report["shuffled_fit"]["paired_cosine"], .3)
        self.assertLess(report["orthogonality_max_error"], 1e-12)

    def test_does_not_fit_the_heldout_relationship(self):
        rng = np.random.default_rng(12)
        image = rng.normal(size=(192, 16)); text = image.copy()
        fit = np.arange(192) < 128
        # Evaluation has the opposite, much larger relationship. Fitting on it would falsely align it.
        image[~fit] *= 1e6; text[~fit] = -image[~fit]
        shifted = rotation_diagnostic(image, text, fit)
        self.assertLess(shifted["orthogonal"]["paired_cosine"], -.99999)

    def test_declines_insufficient_samples(self):
        result = rotation_diagnostic(np.ones((70,16)), np.ones((70,16)), np.arange(70)<60)
        self.assertFalse(result["eligible"])


if __name__ == "__main__": unittest.main()
