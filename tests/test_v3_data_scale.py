import itertools
import json
import unittest

from qwen_decision.v3 import data_scale as ds

MIX = ds.ROOT / "mix"
HELD_OUT = ("piqa", "swag", "copa", "qasc", "quartz", "wic", "ethics", "timedial", "epistemic", "presupposition", "anli-train", "trec", "cb")


@unittest.skipUnless((MIX / "manifest.json").exists(), "data-scale mix not built")
class DataScaleMixTests(unittest.TestCase):
    def setUp(self):
        self.manifest = json.loads((MIX / "manifest.json").read_text())

    def test_size_and_option_ratios_track_pool(self):
        self.assertGreater(self.manifest["rows"], 1_900_000)
        pool = {"yes/no": 22.4, "2-way": 18.6, "3": 21.6, "4": 14.9, "5-10": 14.3, "11-50": 5.1, "51+": 3.1}
        for b, pct in pool.items():
            self.assertLess(abs(self.manifest["buckets_pct"][b] - pct), 1.5, b)

    def test_replaced_and_held_out_sources_absent(self):
        sources = set(self.manifest["per_source"])
        self.assertFalse(sources & set(ds.FORMAT_REPLACES))
        self.assertFalse({s for s in sources if any(s == h or s.startswith(h + "-") for h in HELD_OUT)})

    def test_new_rows_are_gold_one_hot(self):
        with (MIX / "text-train.jsonl").open() as f:
            for line in itertools.islice(f, 50000):
                r = json.loads(line)
                if r.get("training_component") != "data-scale-v1":
                    continue
                self.assertEqual(sum(r["target"]), 1.)
                self.assertIn(r["label_origin"], ("human", "programmatic"))
                n = 2 if r["question"]["type"] == "noul" else len(r["question"]["criteria"])
                self.assertEqual(len(r["target"]), n)

    def test_ocnli_label_order(self):
        # HF ClassLabel for clue/ocnli is [neutral, entailment, contradiction]; label 1 must become "entailment"
        import random
        uid, state, (q, target, _), _ = next(ds.ocnli(random.Random(0)))  # raw row 0 has label 1
        self.assertEqual(uid, 0)
        self.assertEqual(list(q["criteria"])[target.index(1.)], "entailment")


if __name__ == "__main__":
    unittest.main()
