import json
from pathlib import Path
import unittest

from qwen_decision.v3 import format_rebuild as fr


def rows(name):
    return [json.loads(l) for l in (fr.ROOT / f"{name}.jsonl").open()]


def public(prefix):
    for line in fr.PUBLIC.open():
        if line.startswith('{"id":"' + prefix):
            yield json.loads(line)


@unittest.skipUnless((fr.ROOT / "manifest.json").exists(), "format rebuild not built")
class FormatRebuildTests(unittest.TestCase):
    def test_acos_question_matches_evaluator_text(self):
        q = next(public("ACOS:"))["questions"]
        key, ev = next(iter(q.items()))
        cat, sent = key.split("__sentiment_")
        code = {v: k for k, v in fr.SENTIMENTS.items()}[sent]
        self.assertEqual(fr.acos_question(cat, code), ev["instructions"])
        self.assertEqual(fr.ACOS_CRITERIA, ev["criteria"])

    def test_bfcl_question_matches_evaluator_text(self):
        ev = next(iter(next(public("BFCL")) ["questions"].values()))
        mine = rows("bfcl-format-train")[0]["question"]
        self.assertEqual(mine["instructions"].split("Candidate tool:")[0], ev["instructions"].split("Candidate tool:")[0])
        self.assertEqual(mine["criteria"], ev["criteria"])

    def test_acos_train_dev_disjoint_and_base_rates(self):
        tr, dev = rows("acos-format-train"), rows("acos-format-dev")
        self.assertFalse({r["state"]["review"] for r in tr} & {r["state"]["review"] for r in dev})
        yes = sum(r["target"][0] for r in tr) / len(tr)
        self.assertLess(abs(yes - fr.BASE_RATE), .01)
        self.assertLess(sum(r["target"][0] for r in dev) / len(dev), .02)

    def test_no_public_state_reused(self):
        blocked = fr.public_states(("ACOS:", "iSarcasmEval"))
        texts = [r["state"]["review"] for r in rows("acos-format-train")]
        texts += [r["state"] for r in rows("isarcasm-format-train") if r["state"]]
        self.assertFalse({fr.norm(t) for t in texts} & blocked)

    def test_targets_one_hot(self):
        for name in ("acos-format-train", "isarcasm-format-train", "bfcl-format-train"):
            for r in rows(name)[:3000]:
                self.assertEqual(sum(r["target"]), 1.)
                self.assertEqual(len(r["target"]), len(r["question"]["criteria"]))


if __name__ == "__main__":
    unittest.main()
