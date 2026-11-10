import unittest
from concurrent.futures import ThreadPoolExecutor

from qwen_decision.v3 import champion_enriched_data as data, format as fmt


class EnrichedDataTests(unittest.TestCase):
    def test_three_views_agree_on_fact_for_both_gold_labels(self):
        for positive in (False, True):
            base = {"id": "isarcasm-train:A:En:17", "source": "isarcasm-train", "family": "sarcasm",
                    "state": {"text": "An author's actual train text"}, "question": {}, "target": [1., 0.],
                    "split": "train", "ordered": False, "kind": "choice"}
            views = list(data.binary_views(base, positive, "sarcastic", "not sarcastic",
                                          "This text is", "Classify the text."))
            self.assertEqual(len(views), 3)
            choices = fmt.options(views[0]["question"])[0]
            gold = views[0]["target"].index(1.)
            self.assertEqual(choices[gold], "sarcastic" if positive else "not sarcastic")
            # Renderer noul order is false, true: inversion must swap probabilities.
            self.assertEqual(views[1]["target"], [float(not positive), float(positive)])
            self.assertEqual(views[2]["target"], [float(positive), float(not positive)])
            self.assertEqual({r["item"] for r in views}, {"isarcasm-train:En:17"})
            self.assertEqual({r["label_origin"] for r in views}, {"human"})
            for r in views:
                self.assertEqual(r["state"], base["state"])
            self.assertEqual(base["question"], {})

    def test_bounded_parallel_map_keeps_complete_ordered_stream(self):
        calls = []
        def source():
            for i in range(100):
                calls.append(i)
                yield i
        with ThreadPoolExecutor(3) as pool:
            results = data.bounded_map(pool, lambda x: x * x, source(), limit=4)
            self.assertEqual(next(results), 0)
            self.assertEqual(len(calls), 4)
            self.assertEqual(list(results), [i * i for i in range(1, 100)])
        self.assertEqual(calls, list(range(100)))


if __name__ == "__main__":
    unittest.main()
