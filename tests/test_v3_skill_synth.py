import random
import unittest

from qwen_decision.v3 import skill_synth as s


def home(policy="all_or_nothing"):
    h = s.make_home(random.Random(5))
    h["policy"]["multiple_devices"] = policy
    return h


def powered(h):
    return [d for d in h["devices"] if "power" in d and d["kind"] != "thermostat"]


class HomePolicyTests(unittest.TestCase):
    def test_single_unavailable_is_blocked_and_changes_nothing(self):
        h = home()
        d = powered(h)[0]
        d["available"] = False
        r = s.resolve(h, {"kind": "execute", "ops": [[d["id"], "power", not d["power"]]]})
        self.assertEqual((r["status"], r["changed"]), ("blocked", []))
        self.assertEqual(r["referenced"], [d["id"]])

    def test_multi_unavailable_depends_on_policy(self):
        for policy, status in (("all_or_nothing", "blocked"), ("best_effort", "execute_partial")):
            h = home(policy)
            a, b = powered(h)[:2]
            a["available"], b["available"] = True, False
            r = s.resolve(h, {"kind": "execute", "ops": [[a["id"], "power", not a["power"]], [b["id"], "power", True]]})
            self.assertEqual(r["status"], status)
            if status == "execute_partial":
                self.assertEqual(r["changed"], [[a["id"], "power", not a["power"]]])

    def test_last_operation_wins_and_noop_is_no_change(self):
        h = home()
        d = powered(h)[0]
        d["available"] = True
        r = s.resolve(h, {"kind": "execute", "ops": [[d["id"], "power", not d["power"]], [d["id"], "power", d["power"]]]})
        self.assertEqual(r["status"], "no_change")

    def test_clarify_references_nothing(self):
        r = s.resolve(home(), {"kind": "clarify", "candidates": ["x", "y"]})
        self.assertEqual(r["referenced"], [])

    def test_rows_are_valid_one_hot_and_deterministic(self):
        rows = s.home_rows(30)
        self.assertEqual(rows, s.home_rows(30))
        for r in rows:
            self.assertEqual(sum(r["target"]), 1.)
            self.assertEqual(len(r["target"]), len(r["question"]["criteria"]))


class ChordTests(unittest.TestCase):
    def test_vocabulary_merges_equivalent_sets(self):
        self.assertIn("Caug", s.VOCAB[(0, 4, 8)])
        self.assertIn("Eaug", s.VOCAB[(0, 4, 8)])
        self.assertEqual(len(s.CHORD_SETS), len(set(s.CHORD_SETS)))

    def test_rows_answer_matches_label(self):
        for r in s.chord_rows(200):
            answer = next(o for o, t in zip(r["question"]["criteria"], r["target"]) if t)
            label = r["chord_label"]
            if label == "Other":
                self.assertEqual(answer, "a pitch-class set not in this list")
            elif label == "NoChord":
                self.assertEqual(answer, "no chord is sounding")
                t = r["state"]["target_beat"]
                self.assertFalse(any(n["start_beat"] <= t < n["end_beat"] for n in r["state"]["notes"]))
            else:
                self.assertEqual(answer, s.option_text(tuple(label)))


if __name__ == "__main__":
    unittest.main()
