import unittest

from qwen_decision.v3 import champion_audit as audit, mix


class CampaignAuditTest(unittest.TestCase):
    def example(self, source="boolq-train", origin="human"):
        row = {"id": f"{source}:42", "source": source, "state": "A passage.",
               "question": {"type": "noul", "instructions": "Is this true?"},
               "target": [0., 1.], "target_kind": "single", "label_origin": origin}
        row["item"] = mix.item_key(row)
        return row

    def test_unknown_sizes_and_names_do_not_enter_parameter_bucket(self):
        def model(name, parameters, score):
            return {"engine": name, "meta": {"served_params": parameters}, "scores": {"balanced_skill": score}}
        models = [model("unknown-0.8B", None, 100), model("too-large-1B", 1000000001, 99),
                  model("valid", 1000000000, 20), model("smaller", 100, 30)]
        self.assertEqual([r["engine"] for r in audit.small_text(models)], ["smaller", "valid"])

    def test_human_and_prior_tool_exception_are_explicit(self):
        audit.validate_row(self.example())
        audit.validate_row(self.example("tools-percall", "model"))
        with self.assertRaises(AssertionError):
            audit.validate_row(self.example(origin="model"))

    def test_bad_target_and_wrong_item_cannot_pass_audit(self):
        row = self.example(); row["target"] = [.4, .4]
        with self.assertRaises(AssertionError): audit.validate_row(row)
        row = self.example(); row["item"] = "wrong"
        with self.assertRaises(AssertionError): audit.validate_row(row)

    def test_state_hash_preserves_content_but_ignores_json_key_order(self):
        self.assertEqual(audit.state_key({"a": "YES", "b": "No"}), audit.state_key({"b": "No", "a": "yes"}))
        self.assertNotEqual(audit.state_key("The answer is 2"), audit.state_key("The answer is 3"))

    def test_same_id_with_changed_label_is_not_an_identical_repeat(self):
        seen = {}; row = self.example()
        audit.identity_kind(row, seen)
        self.assertEqual(audit.identity_kind(dict(row), seen), "identical_row")
        row["target"] = [1., 0.]
        self.assertEqual(audit.identity_kind(row, seen), "different_row_content")

    def test_empty_state_does_not_make_different_questions_identical(self):
        a = self.example(); a["state"] = ""
        b = self.example(); b["state"] = ""; b["question"] = {"type": "noul", "instructions": "Another question?"}
        self.assertFalse(audit.has_context(a))
        self.assertNotEqual(audit.input_key(a), audit.input_key(b))

    def test_reordered_choices_keep_input_and_gold_identity(self):
        a = {"state": "s", "question": {"type": "choice", "criteria": {"x": "one", "y": "two"}}, "target": [1., 0.]}
        b = {"state": "s", "question": {"type": "choice", "criteria": {"y": "two", "x": "one"}}, "target": [0., 1.]}
        self.assertEqual(audit.input_key(a), audit.input_key(b))
        self.assertEqual(audit.target_signature(a), audit.target_signature(b))


if __name__ == "__main__": unittest.main()
