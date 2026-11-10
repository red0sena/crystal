import unittest

import numpy as np

from qwen_decision.v3 import format as fmt, inference_policy as exp


class Tokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return repr(messages) + repr(kwargs)


class InferencePolicyTest(unittest.TestCase):
    def test_base_is_native_prompt_and_header_preserves_options(self):
        tok = Tokenizer(); q = {"type": "choice", "instructions": "Which shape?", "criteria": {"first": "circle", "second": "square"}}
        native = fmt.render(tok, "Scene", q, ["A", "B"], "d3")
        actual, repeated = exp.render(tok, "Scene", q, ["A", "B"], "base", True)
        self.assertEqual(native, actual); self.assertFalse(repeated)
        header, _ = exp.render(tok, "Scene", q, ["A", "B"], "header", False)
        self.assertLess(header.index("Which shape?"), header.index("Scene"))
        self.assertIn("A: first: circle", header); self.assertIn("B: second: square", header)

    def test_repeat_duplicates_entire_image_user_turn_and_respects_gate(self):
        q = {"type": "choice", "instructions": "Count?", "criteria": {"two": None, "three": None}}
        for policy in exp.POLICIES:
            text, repeated = exp.render(Tokenizer(), exp.IMAGE * 2, q, ["A", "B"], policy, True)
            self.assertEqual(text.count(exp.IMAGE), 4 if repeated else 2)
            once, flag = exp.render(Tokenizer(), exp.IMAGE * 2, q, ["A", "B"], policy, False)
            self.assertFalse(flag); self.assertEqual(once.count(exp.IMAGE), 2)

    def test_shared_images_and_mmvp_pairs_never_cross_calibration_split(self):
        rows = [{"id": f"vision:{i}", "upstream_id": f"heldout_mmvp:{i}", "modality": "vision", "source": "mmvp", "image_sha256": [str(i)]} for i in (1, 2, 3)]
        rows.append({"id": "vision:4", "upstream_id": "other:4", "modality": "vision", "source": "chart", "image_sha256": ["2"]})
        exp.group_assign(rows)
        self.assertEqual(rows[0]["group"], rows[1]["group"])
        self.assertEqual(rows[1]["group"], rows[3]["group"])
        self.assertNotEqual(rows[1]["group"], rows[2]["group"])

    def test_temperature_fit_and_metrics_do_not_change_argmax(self):
        records = [{"id": str(i), "source": "s", "gold": int(i % 5 == 0), "logits": [8., 0.], "prediction": 0,
                    "question_type": "choice", "milliseconds": 1., "input_tokens": 100, "repeat_applied": False} for i in range(100)]
        temperature = exp.fit_temperature(records)
        self.assertGreater(temperature, 1.)
        raw, calibrated = exp.metric_rows(records), exp.metric_rows(records, {"choice": temperature})
        self.assertEqual([r["correct"] for r in raw], [r["correct"] for r in calibrated])
        self.assertLess(exp.summary(calibrated)["macro_nll"], exp.summary(raw)["macro_nll"])

    def test_group_bootstrap_is_paired_and_uses_group_multiplicity(self):
        records = [{"id": str(i), "source": "s", "group": "same" if i < 3 else "other", "correct": i == 0,
                    "nll": float(i), "brier": float(i) / 3} for i in range(4)]
        actual = exp.group_bootstrap(records, np.random.default_rng(5), draws=13)
        again = exp.group_bootstrap(records, np.random.default_rng(5), draws=13)
        np.testing.assert_array_equal(actual, again)
        # Two three-row/one-row groups: counts, not a mean of group means.
        weights = np.random.default_rng(5).multinomial(2, [.5, .5], size=13)
        # Sorted group order is other, same.
        sums = weights @ np.array([[0, 3, 1, 1], [1, 3, 1, 3]])
        np.testing.assert_allclose(actual, sums[:, :3] / sums[:, 3:4])


if __name__ == "__main__": unittest.main()
