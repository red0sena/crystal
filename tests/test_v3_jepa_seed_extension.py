import ast
import tempfile
import unittest
from pathlib import Path

import numpy as np

from qwen_decision.v3 import jepa_decision_seed_extension as extension


class SeedExtensionTest(unittest.TestCase):
    def test_evaluator_changes_only_bootstrap_seed_count(self):
        original = ast.parse(Path(extension.exp.__file__).read_text())
        original = next(n for n in original.body if isinstance(n, ast.FunctionDef) and n.name == "analyze")
        generated = ast.parse(extension.evaluator_source(extension.SEEDS))
        generated = next(n for n in generated.body if isinstance(n, ast.FunctionDef) and n.name == "analyze")
        calls = [n for n in ast.walk(generated) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                 and n.func.attr == "integers" and isinstance(n.args[0], ast.Call)]
        self.assertEqual(len(calls), 1)
        self.assertEqual(ast.unparse(calls[0]), "rng.integers(len(SEEDS), size=len(SEEDS))")
        calls[0].args = [ast.Constant(3)]
        calls[0].keywords = [ast.keyword(arg="size", value=ast.Constant(3))]
        self.assertEqual(ast.dump(original), ast.dump(generated))
        self.assertEqual(len(set(extension.SEEDS)), 10)
        self.assertFalse(set(extension.SEEDS[:3]) & set(extension.ADDED))

    def test_cohort_averages_only_requested_seeds(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for arm in extension.exp.ARMS:
                for seed, correct in ((1, 0), (2, 10), (3, 20)):
                    directory = root / f"{arm}-{seed}"; directory.mkdir()
                    records = np.zeros((2, 2, 3, 3))
                    records[..., 0], records[..., 1], records[..., 2] = correct, 30, 20
                    np.savez(directory / "test-records.npz", **{k: records for k in ("text", "raw", "mean", "rotation", "similarity")},
                             all_shuffles=np.stack((records, records)))
            cohort = extension.cohort_scores(root, (2, 3))
            for arm in cohort.values():
                for variant in arm.values():
                    for group in variant.values():
                        self.assertAlmostEqual(group["macro_accuracy"], .75)
                        self.assertAlmostEqual(group["macro_nll"], 1.5)


if __name__ == "__main__": unittest.main()
