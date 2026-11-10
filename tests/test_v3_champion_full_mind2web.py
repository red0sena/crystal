import unittest

from qwen_decision.v3.champion_full_mind2web import partition


def row(action, task, visual):
    return {"id": "mind2web:" + action, "action_uid": action,
            "annotation_id": task, "raw_sha256": visual, "raw_dhash": int(visual)}


class Mind2WebSplitTests(unittest.TestCase):
    def test_tasks_and_shared_raw_screens_cannot_cross_partitions(self):
        rows = [row("a", "task-a", "1"), row("b", "task-a", "2"),
                row("c", "task-b", "2"), row("d", "task-b", "3")]
        train, dev, dropped = partition(rows)
        self.assertEqual(sorted(r["action_uid"] for r in train + dev), list("abcd"))
        self.assertTrue(bool(train) != bool(dev))
        self.assertEqual(len({r["group"] for r in train + dev}), 1)
        self.assertEqual(dropped, {})

    def test_one_evaluation_action_excludes_its_entire_connected_task_group(self):
        rows = [row("a", "task-a", "1"), row("b", "task-a", "2"), row("c", "task-b", "2")]
        train, dev, dropped = partition(rows, forbidden_actions={"b"})
        self.assertEqual(train + dev, [])
        self.assertEqual(dropped, {"evaluation_action_or_task_group": 3})

    def test_visual_match_excludes_related_task_actions_and_shared_screens(self):
        rows = [row("a", "task-a", "1"), row("b", "task-a", "2"), row("c", "task-b", "2")]
        train, dev, dropped = partition(rows, forbidden_visuals={"mind2web:a"})
        self.assertEqual(train + dev, [])
        self.assertEqual(dropped, {"evaluation_or_prior_dev_visual_group": 3})


if __name__ == "__main__":
    unittest.main()
