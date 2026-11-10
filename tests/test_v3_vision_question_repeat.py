import unittest

import numpy as np

from qwen_decision.v3 import format as fmt, inference_policy as old, vision_question_repeat as exp


class Tokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return messages[1]["content"]


class Processor:
    def __call__(self, *, text, images, return_tensors):
        self.text, self.images = text[0], images
        return {"image_grid_thw": np.ones((len(images), 3))}


class QuestionRepeatTest(unittest.TestCase):
    def setUp(self):
        self.question = {"type": "choice", "instructions": "Count the objects.", "criteria": {"two": "2", "three": "3"}}
        self.tokenizer = Tokenizer()

    def test_base_is_exact_native_format_and_disabled_conditions_are_base(self):
        base = fmt.user_prompt(old.IMAGE, self.question, ["A", "B"], "d3")
        for policy in exp.POLICIES:
            actual, repeated = exp.render(self.tokenizer, old.IMAGE, self.question, ["A", "B"], policy, False)
            self.assertEqual(actual, base); self.assertFalse(repeated)

    def test_primary_removes_only_second_state_image_block(self):
        state = old.IMAGE * 2; codes = ["A", "B"]
        primary, repeated = exp.render(self.tokenizer, state, self.question, codes, "q_repeat", True)
        full, _ = old.render(self.tokenizer, state, self.question, codes, "repeat", True)
        first, second = full.split(old.READ_AGAIN)
        state_prefix = "State:\n" + state + "\n\n"
        self.assertTrue(second.startswith(state_prefix))
        self.assertEqual(primary, first + old.READ_AGAIN + second[len(state_prefix):])
        self.assertTrue(repeated); self.assertEqual(primary.count(old.IMAGE), 2)
        self.assertEqual(primary.count("Count the objects."), 2)

    def test_echo_repeats_options_before_and_after_the_single_image(self):
        for policy in ("q_repeat", "q_echo"):
            text, repeated = exp.render(self.tokenizer, old.IMAGE, self.question, ["A", "B"], policy, True)
            self.assertEqual(text.count(old.IMAGE), 1); self.assertTrue(repeated)
            self.assertEqual(text.count("A: two: 2"), 2); self.assertEqual(text.count("B: three: 3"), 2)
            if policy == "q_echo": self.assertLess(text.index("Count the objects."), text.index(old.IMAGE))
            else: self.assertGreater(text.index("Count the objects."), text.index(old.IMAGE))

    def test_processor_receives_original_images_once_for_every_policy(self):
        runner = exp.Runner.__new__(exp.Runner)
        runner.vision_tok = self.tokenizer; runner.codes = {"vision": (["A", "B"], [11, 12])}; runner.processor = Processor()
        row = {"modality": "vision", "question": self.question}; pictures = [object(), object()]
        for policy in exp.POLICIES:
            encoded, ids, repeated, digest = runner.inputs(row, pictures, policy, True)
            self.assertIs(runner.processor.images, pictures)
            self.assertEqual(len(encoded["image_grid_thw"]), 2)
            self.assertEqual(runner.processor.text.count(old.IMAGE), 2)
            self.assertEqual(ids, [11, 12])


if __name__ == "__main__": unittest.main()
