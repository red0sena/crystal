"""Guard question-free extraction, semantic holdouts, and train-only geometry."""
import json
import unittest

import torch

from qwen_decision.v3.state_rotation import (ANSWER_NAMES, COUNTS, HOLDOUT, examples, make_scenes,
                                              orthogonal, prompt, query_specs, state_space)


class StateRotationTest(unittest.TestCase):
    def test_worlds_are_disjoint_and_each_has_twelve_valid_questions(self):
        rows, specs = make_scenes(), query_specs()
        self.assertEqual(len(rows), sum(COUNTS))
        self.assertEqual(len({json.dumps(r['world']) for r in rows}), len(rows))
        items = examples(rows[:8], specs)
        self.assertEqual(len(items), 8 * 12)
        self.assertEqual({q['key'] for q in specs if q['key'] in HOLDOUT}, HOLDOUT)
        for r in rows:
            self.assertEqual(len(r['world']['objects']), 3)
            self.assertNotIn('question', r)
            self.assertNotIn('gold', r)

    def test_state_prompt_does_not_take_decision_inputs(self):
        class Tokenizer:
            def apply_chat_template(self, messages, **kwargs):
                return '\n'.join(m['content'] for m in messages)
        rendered = prompt(Tokenizer(), 'A red circle and a blue square.')
        self.assertNotIn('Question:', rendered)
        self.assertNotIn('Options:', rendered)
        self.assertNotIn('code', rendered)

    def test_gold_is_consistent_with_known_three_object_world(self):
        row = {'world': {'axis': 'horizontal', 'objects': [['red', 'circle'], ['blue', 'triangle'], ['green', 'diamond']]}}
        specs = query_specs()
        actual = {specs[e['query']]['key']: ANSWER_NAMES[e['gold']] for e in examples([row], specs)}
        self.assertEqual(actual['color:circle'], 'red')
        self.assertEqual(actual['shape:green'], 'diamond')
        self.assertEqual(actual['relation:diamond:circle'], 'right')
        self.assertEqual(actual['relation:circle:triangle'], 'left')

    def test_geometry_fit_never_reads_test_states_and_recovers_rotation(self):
        generator = torch.Generator().manual_seed(17)
        x, y = [torch.randn(32, 8, generator=generator) for _ in range(2)]
        train = list(range(24))
        first = state_space(x, y, train, dim=4)
        x[24:] *= 1000; y[24:] -= 1000
        second = state_space(x, y, train, dim=4)
        torch.testing.assert_close(first['basis'], second['basis'])
        torch.testing.assert_close(first['image'][train], second['image'][train])
        a = torch.randn(128, 4, generator=generator)
        expected, _ = torch.linalg.qr(torch.randn(4, 4, generator=generator))
        torch.testing.assert_close(orthogonal(a, a @ expected), expected, atol=1e-6, rtol=1e-6)


if __name__ == '__main__':
    unittest.main()
