"""Check the decision semantics of experimental input and prior transformations."""
import copy
import unittest

import torch
from transformers import AutoTokenizer

from qwen_decision.v3 import format as fmt
from qwen_decision.v3.input_experiment import calibrated, encode, prompt


class InputSemanticsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = AutoTokenizer.from_pretrained('models/pointer-test-tokenizer', local_files_only=True)
        cls.codes, _ = fmt.answer_codes(cls.tokenizer)

    def example(self):
        return {'id': 'example', 'source': 'unit', 'state': {'message': 'payment failure'},
                'question': {'type': 'choice', 'instructions': 'Pick a route.',
                             'criteria': {'payments': 'needs review', 'technical': 'needs review',
                                          'other': None, 'fallback': ''}},
                'target': [0., 1., 0., 0.]}

    def test_base_matches_serving_and_no_mutation(self):
        e = self.example()
        original = copy.deepcopy(e)
        for kind in ('choice', 'noul'):
            if kind == 'noul':
                e['question'] = {'type': 'noul', 'instructions': 'Did payment fail?'}
            k = 4 if kind == 'choice' else 2
            self.assertEqual(prompt(self.tokenizer, e, self.codes[:k], 'base'),
                             fmt.render(self.tokenizer, e['state'], e['question'], self.codes[:k]))
        e['question'] = original['question']
        self.assertEqual(e, original)

    def test_equal_definitions_keep_distinct_answer_slots(self):
        e = self.example()
        row = encode(self.tokenizer, [e], {'example': 'doc'}, 'definitions')[0]
        text = self.tokenizer.decode(row['input_ids'])
        self.assertIn('A: needs review\nB: needs review\nC: other\nD: fallback', text)
        self.assertNotIn('payments:', text)
        self.assertEqual(row['target'], [0., 1., 0., 0.])
        self.assertEqual(len(row['code_ids']), 4)

    def test_empty_state_override_does_not_use_original_state(self):
        e = self.example()
        text = prompt(self.tokenizer, e, self.codes[:4], 'base', state_override='')
        self.assertIn('State:\n(empty)', text)
        self.assertNotIn('payment failure', text)

    def test_repetition_keeps_single_chat_and_duplicates_user_data(self):
        text = prompt(self.tokenizer, self.example(), self.codes[:4], 'repeat')
        self.assertEqual(text.count('State:\n'), 2)
        self.assertEqual(text.count(fmt.SYSTEM_PROMPT), 1)
        self.assertEqual(text.count('<|im_start|>assistant'), 1)

    def test_prefill_preserves_single_token_answer_boundary(self):
        row = encode(self.tokenizer, [self.example()], {'example': 'doc'}, 'prefill')[0]
        self.assertEqual(len(row['target']), len(row['code_ids']))

    def test_calibration_averages_probabilities_and_ignores_logit_offsets(self):
        logits = [[1., 2.]]
        priors = [[[0., 2.]], [[5., 4.]], [[-1., 0.]]]
        p = torch.stack([torch.tensor(z[0], dtype=torch.float64).softmax(0) for z in priors]).mean(0)
        expected = torch.tensor(logits[0]) - p.log()
        actual = torch.tensor(calibrated(logits, priors, 1.)[0], dtype=torch.float64)
        torch.testing.assert_close(actual, expected)
        shifted = [[[v + 20. for v in r[0]]] for r in priors]
        torch.testing.assert_close(actual, torch.tensor(calibrated(logits, shifted, 1.)[0], dtype=torch.float64))
        self.assertEqual(calibrated(logits, priors, 0.), logits)


if __name__ == '__main__':
    unittest.main()
