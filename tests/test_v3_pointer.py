"""Offline readout contracts and tiny random-Qwen integration, not accuracy benchmarks.

Run: .venv/bin/python -m unittest tests.test_v3_pointer -v
Real-tokenizer checks also run if models/pointer-test-tokenizer has been restored.
"""
from copy import deepcopy
import json
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest

import torch
from transformers import AutoTokenizer, Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel

from qwen_decision.v3 import format as fmt, modeling, pointer
from qwen_decision.v3 import pointer_experiment as experiment


def tiny_backbone(device="cpu"):
    config = Qwen3_5TextConfig(vocab_size=64, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1, head_dim=16,
        linear_key_head_dim=8, linear_value_head_dim=8, linear_num_key_heads=2, linear_num_value_heads=2,
        # FLA is CUDA-only; CPU checks cover the noncausal softmax path, CUDA covers the hybrid.
        layer_types=(["full_attention"] * 2 if device == "cpu" else ["linear_attention", "full_attention"]),
        pad_token_id=0)
    config._attn_implementation = "sdpa"
    model = Qwen3_5TextModel(config).to(device)
    modeling.enable_noncausal(model)
    return model


def row(name="a", length=9, k=3):
    return {"id": name, "source": "toy", "input_ids": list(range(1, length + 1)),
            "option_positions": list(range(2, 2 + k)), "code_columns": list(range(k)),
            "target": [1.] + [0.] * (k - 1), "variant": "original", "order": list(range(k))}


class CharTokenizer:
    is_fast = True

    def apply_chat_template(self, messages, **kwargs):
        return "".join(f"<{m['role']}>\n{m['content']}\n" for m in messages) + "<assistant>\n"

    def __call__(self, text, **kwargs):
        return {"input_ids": list(map(ord, text)), "offset_mapping": [(i, i + 1) for i in range(len(text))]}


class PointerTests(unittest.TestCase):
    def test_prompt_compatibility_and_offsets(self):
        tokenizer = CharTokenizer()
        for question in (
            {"type": "choice", "instructions": "Options:\nA: duplicate", "criteria": {
                "한글": "선택지\nOptions:\nA: red", "日本語": {"x": "かな"}, "English": "red"}},
            {"type": "noul", "criteria": {"false": "아니오", "true": "네"}},
        ):
            codes = ["A", "B", "C"][:len(fmt.options(question)[0])]
            for layout in fmt.LAYOUTS:
                state = "State:\nOptions:\nA: red"
                asked = ["Question:", question.get("instructions") or fmt.DEFAULT_INSTRUCTIONS, "", "Options:"]
                asked += [f"{c}: {t}" for c, t in zip(codes, fmt.options(question)[1])]
                lines = (asked + [""] if layout == "echo" else []) + ["State:", state, ""] + asked + ["", fmt.CLOSING_LINE]
                self.assertEqual(fmt.user_prompt(state, question, codes, layout), "\n".join(lines))
                encoded = fmt.encode_options(tokenizer, state, question, codes, layout)
                text = fmt.render(tokenizer, state, question, codes, layout)
                self.assertEqual(encoded["input_ids"], list(map(ord, text)))
                for pos, content in zip(encoded["option_positions"], fmt.options(question)[1]):
                    self.assertEqual(text[pos], content[-1])
                    self.assertGreater(pos, text.rfind("Options:\n"))

    def test_actual_qwen_tokenizer(self):
        path = Path(__file__).resolve().parents[1] / "models/pointer-test-tokenizer"
        if not (path / "tokenizer.json").exists():
            self.skipTest("optional pinned Qwen tokenizer not downloaded")
        t = AutoTokenizer.from_pretrained(path, local_files_only=True)
        args = SimpleNamespace(layout="echo", pointer_positions=True, pad_probability=0.,
            noul_choice_probability=.2, code_shuffle_probability=.2, code_sample_probability=.2)
        renderer = experiment.Renderer(t, args, [])
        e = {"id": "toy", "kind": "choice", "source": "toy", "state": "Options:\nA: 同じ",
             "question": {"type": "choice", "criteria": {"韓国語": "선택지", "日本語": "同じ", "English": "same"}},
             "target": [0., 1., 0.], "ordered": False}
        for seed in range(15):
            encoded = renderer(e, random.Random(seed))
            # Map each selected vector back to the displayed option text, including sampled answer codes.
            ids = encoded["input_ids"]
            self.assertEqual(len(encoded["option_positions"]), 3)
            text = t.decode(ids)
            self.assertIn("Options:", text)
            self.assertEqual([renderer.code_ids[c] for c in encoded["code_columns"]], encoded["code_ids"])
            starts = [t.decode(ids[:p + 1]) for p in encoded["option_positions"]]
            gold = encoded["target"].index(1.)
            self.assertTrue(starts[gold].endswith("同じ"), starts[gold])
            for code, prefix in zip(encoded["code_columns"], starts):
                self.assertIn(f"{renderer.codes[code]}: ", prefix.split("Options:\n")[-1])

    def test_pointer_is_equivariant_over_gathered_options(self):
        torch.manual_seed(7)
        head = pointer.PointerHead(16, 8)
        decision, options = torch.randn(2, 16), torch.randn(2, 4, 16)
        columns, valid = torch.zeros(2, 4, dtype=torch.long), torch.ones(2, 4, dtype=torch.bool)
        permutation = [3, 1, 0, 2]
        a = head(decision, options, columns, valid)
        b = head(decision, options[:, permutation], columns, valid)
        torch.testing.assert_close(a[:, permutation], b)
        # This is a head property; the backbone may still depend on prompt order.

    def test_padding_soft_targets_and_gradients(self):
        batch = pointer.collate([row(), row("b", 6, 2)], 0, "cpu")
        self.assertFalse(batch["valid"][1, 2])
        head = pointer.PointerHead(8, 4)
        decision = torch.randn(2, 8, requires_grad=True)
        options = torch.randn(2, 3, 8, requires_grad=True)
        logits = head(decision, options, batch["columns"], batch["valid"])
        loss = pointer.soft_ce(logits, batch["targets"]).mean()
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertEqual(logits[1, 2].item(), pointer.MASK)
        self.assertEqual(options.grad[1, 2].abs().sum().item(), 0.)
        self.assertGreater(options.grad[0].abs().sum().item(), 0.)
        for p in head.parameters():
            self.assertTrue(torch.isfinite(p.grad).all())

    def test_code_control_matches_original_lm_rows(self):
        torch.manual_seed(1)
        head = pointer.CodeHead(torch.randn(5, 12))
        h, columns = torch.randn(2, 12), torch.tensor([[0, 4, 1], [3, 2, 0]])
        valid = torch.tensor([[True] * 3, [True, True, False]])
        actual = head(h, None, columns, valid)
        expected = torch.nn.functional.linear(h, head.weight).gather(1, columns).masked_fill(~valid, pointer.MASK)
        torch.testing.assert_close(actual, expected)

    def test_invalid_positions_fail_before_forward(self):
        bad = row()
        bad["option_positions"][0] = 99
        with self.assertRaisesRegex(ValueError, "outside real input"):
            pointer.collate([bad], 0, "cpu")

    def test_tiny_qwen_one_forward_noncausal_and_frozen_update(self):
        torch.manual_seed(11)
        backbone = tiny_backbone().eval().requires_grad_(False)
        head = pointer.PointerHead(32, 8)
        batch = pointer.collate([row(), row("b", 6, 2)], 0, "cpu")
        calls = []
        hook = backbone.register_forward_hook(lambda *a: calls.append(1))
        before = {k: v.clone() for k, v in backbone.state_dict().items()}
        head_before = head.query.weight.detach().clone()
        optimizer = torch.optim.AdamW(head.parameters(), lr=.01)
        loss = pointer.soft_ce(pointer.forward(backbone, head, batch), batch["targets"]).mean()
        loss.backward()
        optimizer.step()
        hook.remove()
        self.assertEqual(len(calls), 1)
        self.assertFalse(torch.equal(head_before, head.query.weight))
        for k, v in backbone.state_dict().items():
            self.assertTrue(torch.equal(before[k], v), k)
        changed = deepcopy(batch)
        changed["input_ids"][0, -1] = 23
        with torch.no_grad():
            a = backbone(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], use_cache=False).last_hidden_state
            b = backbone(input_ids=changed["input_ids"], attention_mask=batch["attention_mask"], use_cache=False).last_hidden_state
        self.assertGreater((a[0, 2] - b[0, 2]).abs().max().item(), 0.)

    def test_split_overlap_rejected(self):
        def example(name, text):
            return {"id": name, "source": "toy", "state": text,
                    "question": {"type": "choice", "criteria": {"yes": None, "no": None}}}
        with self.assertRaisesRegex(ValueError, "overlap"):
            experiment.check_disjoint({"train": [example("a", "Repeated document about a river crossing and bridges in Seoul.")],
                                       "dev": [example("b", "Repeated document about a river crossing and bridges in Seoul.")]})
        experiment.check_disjoint({"train": [example("a", "Completely different train passage about a river and mountains.")],
                                   "dev": [example("b", "Independent evaluation text on financial research in Europe.")]})
        # Short classification states with the same options are independent examples.
        experiment.check_disjoint({"train": [example("a", "refund")], "dev": [example("b", "transfer")]})
        left, right = example("a", "refund"), example("b", "transfer")
        left["item"] = right["item"] = "shared-upstream-document"
        with self.assertRaisesRegex(ValueError, "overlap"):
            experiment.check_disjoint({"train": [left], "dev": [right]})

    def test_variants_keep_distinct_evaluation_ids(self):
        e = {"id": "same-upstream-id", "source": "toy", "state": "A document.", "ordered": False,
             "kind": "choice", "question": {"type": "choice", "criteria": {"yes": None, "no": None}},
             "target": [1., 0.], "label_origin": "programmatic", "target_kind": "single"}
        variant = {**e, "question": {"type": "noul"}, "target": [0., 1.]}
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "mix.jsonl"
            p.write_text(json.dumps(e) + "\n" + json.dumps(variant) + "\n")
            rows = experiment.read_rows(p)
            self.assertNotEqual(rows[0]["id"], rows[1]["id"])
            self.assertEqual(rows[0]["upstream_id"], rows[1]["upstream_id"])
            with self.assertRaisesRegex(ValueError, "overlap"):
                experiment.check_disjoint({"train": rows[:1], "dev": rows[1:]})
            p.write_text(json.dumps(e) + "\n" + json.dumps(e) + "\n")
            with self.assertRaisesRegex(ValueError, "same prompt"):
                experiment.read_rows(p)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA optimizer integration")
    def test_cuda_frozen_and_limited_tuning_phases(self):
        class Wrapper(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.model = torch.nn.Module()
                self.model.language_model = tiny_backbone("cuda").to(torch.bfloat16)

            def gradient_checkpointing_enable(self, **kwargs):
                self.model.language_model.gradient_checkpointing_enable(**kwargs)

        args = SimpleNamespace(backbone_lr=.003, head_lr=.01, seed=11, effective_batch=2, micro_batch=2,
                               token_budget=128, pack=False, max_tokens=32, eval_every=1)
        renderer = SimpleNamespace(tokenizer=SimpleNamespace(pad_token_id=0))

        class Render:
            tokenizer = renderer.tokenizer

            def __call__(self, r, rng):
                return r

        for arm in ("code", "pointer"):
            model = Wrapper()
            head = (pointer.PointerHead(32, 8) if arm == "pointer" else pointer.CodeHead(torch.randn(3, 32))).cuda()
            rows = [row(), row("b", 6, 2)]
            before = {k: v.clone() for k, v in model.model.language_model.state_dict().items()}
            with tempfile.TemporaryDirectory() as d:
                out = Path(d)
                frozen = experiment.train_phase(model, head, Render(), rows, rows, args, out / "frozen", 2, 0)
                self.assertEqual(frozen["rows_seen"], 4)
                for k, v in model.model.language_model.state_dict().items():
                    self.assertTrue(torch.equal(before[k], v), k)
                tuned = experiment.train_phase(model, head, Render(), rows, rows, args, out / "tuned", 2, 1)
                self.assertEqual(tuned["rows_seen"], 4)
                # Tuning permits only the last layer; selection can legitimately choose step 0.
                self.assertFalse(any(p.requires_grad for p in model.model.language_model.layers[0].parameters()))
                self.assertTrue(all(p.requires_grad for p in model.model.language_model.layers[-1].parameters()))
                self.assertTrue((out / "tuned/language.safetensors").exists())
                self.assertEqual(json.loads((out / "tuned/result.json").read_text())["updates"], 2)


if __name__ == "__main__":
    unittest.main()
