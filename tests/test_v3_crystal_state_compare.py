"""Guard the actual experimental controls rather than a particular learned result."""
import unittest
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

from qwen_decision.v3 import crystal_state_compare as experiment
from qwen_decision.v3 import state_rotation as sr


class CrystalMatchedDataTest(unittest.TestCase):
    def test_same_worlds_labels_and_semantic_holdout(self):
        rows, specs, splits = experiment.dataset()
        self.assertEqual(len(rows), 1536)
        self.assertEqual([len(splits[s]) for s in splits], [8922, 2251, 3072])
        for name in ("train", "validation"):
            self.assertFalse(any(e["unseen"] for e in splits[name]))
        for item in splits["test"][::13]:
            example = experiment.training_example(item, rows, specs)
            answers = list(example["question"]["criteria"])
            self.assertEqual(answers[example["target"].index(1.)], sr.ANSWER_NAMES[item["gold"]])
            self.assertEqual(example["state"], rows[item["scene"]]["caption"])

    def test_schedule_has_exactly_planned_passes_no_drops(self):
        seen = {e: [] for e in range(1, experiment.EPOCHS + 1)}
        first = list(experiment.schedule(33, 271))
        self.assertEqual(first, list(experiment.schedule(33, 271)))
        self.assertNotEqual(first, list(experiment.schedule(33, 283)))
        for epoch, indices in first:
            seen[epoch].extend(indices)
        for indices in seen.values():
            self.assertEqual(sorted(indices), list(range(33)))

    def test_fresh_control_freeze_does_not_require_amendment_history(self):
        from qwen_decision.v3 import crystal_state_attention_control as control
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "plan.json").write_text("{}")
            with patch.object(control, "OUT", root / "noncausal-control"), patch.object(control.exp, "OUT", root):
                control.freeze()
                plan = json.loads((root / "noncausal-control/plan.json").read_text())
                self.assertIsNone(plan["prior_plan_sha256"])
                self.assertTrue(plan["created_before_any_evaluation_complete"])


if __name__ == "__main__":
    unittest.main()
