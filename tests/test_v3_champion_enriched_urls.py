import unittest
from qwen_decision.v3.champion_enriched_urls import views
from qwen_decision.v3 import format as fmt


class URLDataTests(unittest.TestCase):
    def test_all_views_preserve_the_annotation_without_external_features(self):
        for label in ("phishing", "legitimate"):
            rows = list(views("https://sub.example.org/a?x=1", label))
            self.assertEqual(len(rows), 3)
            self.assertEqual(set(rows[0]["state"]), {"url", "hostname"})
            options = fmt.options(rows[0]["question"])[0]
            self.assertEqual(options[rows[0]["target"].index(1.)], label)
            self.assertEqual(rows[1]["target"], [float(label != "phishing"), float(label == "phishing")])
            self.assertEqual(rows[2]["target"], [float(label != "legitimate"), float(label == "legitimate")])
            self.assertEqual(len({r["item"] for r in rows}), 1)
            self.assertEqual(len({tuple(r["documents"]) for r in rows}), 1)
            for r in rows:
                self.assertNotIn("upstream_label", r["state"])


if __name__ == "__main__":
    unittest.main()
