import math

import unittest

from qwen_decision.v3.champion_epoch1_eval import source_summary, validate_predictions


def cases():
    rows = [dict(id="shared-upstream-id", source="source", modality="text", target=target)
            for target in ([1., 0.], [0., 1.])]
    predictions = [dict(id=row["id"], source=row["source"], index=i, k=2, pred=i,
                        correct=True, ce=-math.log(.8), kl=-math.log(.8),
                        logits=[math.log(.8 if j == i else .2) for j in range(2)])
                   for i, row in enumerate(rows)]
    return rows, predictions


class RecoveryTests(unittest.TestCase):
    def test_repeated_upstream_ids_keep_both_distinct_rows(self):
        rows, predictions = cases()
        validate_predictions(rows, predictions)
        summary = source_summary(predictions)
        self.assertEqual(summary["rows"], 2)
        self.assertEqual(summary["accuracy"], 1.)


    def test_duplicate_row_index_and_tampered_cached_loss_are_rejected(self):
        rows, predictions = cases()
        predictions[1]["index"] = 0
        with self.assertRaises(AssertionError):
            validate_predictions(rows, predictions)
        rows, predictions = cases()
        predictions[1]["ce"] = 0.
        with self.assertRaises(AssertionError):
            validate_predictions(rows, predictions)


if __name__ == "__main__":
    unittest.main()
