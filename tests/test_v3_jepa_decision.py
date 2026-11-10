import unittest

import torch

from qwen_decision.v3 import jepa_decision as exp, state_rotation as sr


class JEPADecisionTest(unittest.TestCase):
    def test_factored_decoder_answers_all_questions_including_holdouts(self):
        rows = sr.make_scenes()
        logits = torch.full((len(rows), 32), -30.)
        for s, row in enumerate(rows):
            for i, (color, shape) in enumerate(row["world"]["objects"]):
                logits[s, i * 6 + list(sr.COLORS).index(color)] = 30.
                logits[s, 18 + i * 4 + list(sr.SHAPES).index(shape)] = 30.
            logits[s, 30 + int(row["world"]["axis"] == "vertical")] = 30.
        specs = sr.query_specs(); answers = exp.FactoredDecision.answer_logits(logits, specs)
        valid = torch.zeros(len(rows), len(specs), dtype=torch.bool)
        for item in sr.examples(rows, specs):
            self.assertEqual(int(answers[item["scene"], item["query"]].argmax()), item["gold"])
            valid[item["scene"], item["query"]] = True
        torch.testing.assert_close(answers.exp().sum(-1)[valid], torch.ones(int(valid.sum())))

    def test_both_losses_update_the_actual_state_encoder(self):
        torch.manual_seed(1)
        model = exp.Model()
        tokens = torch.randn(4, exp.LENGTH, 1024)
        lengths = torch.tensor([16, 20, 24, 28])
        logits, z = model(tokens, lengths, sr.query_specs())
        ce = -logits[:, 0, 0].mean()
        ce.backward(retain_graph=True)
        self.assertGreater(float(model.encoder.out[-1].weight.grad.norm()), 0.)
        model.zero_grad()
        loss, _, _ = exp.jepa(z.reshape(2, 2, exp.DIM), 1)
        loss.backward()
        self.assertGreater(float(model.encoder.out[-1].weight.grad.norm()), 0.)

    def test_positive_pairs_have_expected_observation_and_marginals(self):
        for arm in ("good", "bad"):
            i, j = exp.pair_indices(401, 8, 10000, arm, "cpu")
            self.assertTrue(bool(((i // 2 != j // 2) if arm == "good" else (i // 2 == j // 2)).all()))
            self.assertTrue(bool((i != j).all()))
            for x in (i, j):
                self.assertLess(float((torch.bincount(x, minlength=4).float() / len(x) - .25).abs().max()), .02)

    def test_sigreg_rejects_collapse_and_matches_numerical_quadrature(self):
        torch.manual_seed(5)
        x = torch.randn(512, exp.DIM)
        self.assertLess(float(exp.sigreg(x, 11)), float(exp.sigreg(torch.zeros_like(x), 11)))
        generator = torch.Generator().manual_seed(11)
        a = torch.randn(exp.DIM, 128, generator=generator); a /= a.norm(dim=0)
        t = torch.linspace(0, 3, 17); phi = (-t.square() / 2).exp()
        projections = x @ a
        error = ((projections[:, :, None] * t).cos().mean(0) - phi).square() + (projections[:, :, None] * t).sin().mean(0).square()
        manual = torch.trapezoid(error * phi, t, dim=-1).mean() * len(x) * 2
        torch.testing.assert_close(exp.sigreg(x, 11), manual)

    def test_views_preserve_all_color_shape_facts(self):
        row = sr.make_scenes()[0]
        for obs in range(2):
            for aug in range(2):
                text = exp.text_view(row, obs, aug)
                for color, shape in row["world"]["objects"]:
                    self.assertIn(color + " " + shape, text)
        self.assertNotEqual(exp.image_view(row, 0, 0).tobytes(), exp.image_view(row, 0, 1).tobytes())

    def test_padding_cannot_change_the_actual_state(self):
        torch.manual_seed(23)
        model = exp.StateEncoder().eval()
        tokens = torch.randn(2, exp.LENGTH, 1024)
        lengths = torch.tensor([13, 21])
        changed = tokens.clone()
        for i, length in enumerate(lengths):
            changed[i, length:] = torch.randn_like(changed[i, length:]) * 100
        with torch.no_grad():
            torch.testing.assert_close(model(tokens, lengths), model(changed, lengths))

    def test_geometry_is_invariant_to_independent_rotations(self):
        torch.manual_seed(31)
        x, y = torch.randn(64, 16), torch.randn(64, 16)
        a, _ = torch.linalg.qr(torch.randn(16, 16))
        b, _ = torch.linalg.qr(torch.randn(16, 16))
        before, after = exp.geometry(x, y), exp.geometry(x @ a + 3, y @ b - 2)
        self.assertAlmostEqual(before["biased_linear_CKA"], after["biased_linear_CKA"], places=6)
        self.assertAlmostEqual(before["centered_cosine_mutual10NN"], after["centered_cosine_mutual10NN"])


if __name__ == "__main__":
    torch.set_num_threads(8)
    unittest.main()
