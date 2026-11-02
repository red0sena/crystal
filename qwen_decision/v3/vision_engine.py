"""Decision Index vision engine for v3 checkpoints and the untouched base model (transformers, one row per forward).

  PYTHONPATH=external/decision-index .venv/bin/python -m decision_index.cli vision run --dir data/vision-work/vision \
      --engine qwen_decision.v3.vision_engine:VisionEngine --option model=<dir> --option prompt=chat --workers 1 \
      --out runs/vision/<name>

prompt=chat follows the kit's vllm-vl reference path for stock VLMs: one user turn (the images, then the
question, `A. option` lines and the letter instruction), the model's default chat template, the row's readout
prefix continued as an assistant turn, and the next-token logits of the option letters.
prompt=decide is the v3 text format (format.render) with the images as the state; last-token readout over the
answer codes. The row's readout prefix is not used there: the v3 prompt always ends at the assistant prefix.

Attention comes from the checkpoint's v3_config.json (causal for the base model). Images are capped at 1.6 MP
as the kit's engines do (decision_index.vision.images.pil); the processor's own budget is above that, so every
image goes in at its native resolution up to the cap.
"""
from __future__ import annotations

import time
from pathlib import Path

import torch
from decision_index.engines import Engine, Unsupported
from decision_index.vision import images as img
from decision_index.vision.engines import answer, prompt_text, single_choice
from transformers import AutoProcessor

from qwen_decision.v3 import format as fmt
from qwen_decision.v3 import modeling

PROCESSOR = Path("models/qwen3.5-0.8b")  # v3 checkpoints carry the tokenizer but not the image processor
IMAGE = "<|vision_start|><|image_pad|><|vision_end|>"
PROMPTS = ("chat", "decide")


class VisionEngine(Engine):
    name = "qwen-decision-v3-vision"
    latency = ("Device-synchronized in-process request wall time including image loading and preprocessing; "
               "one row per forward; excludes model loading.")

    def __init__(self, model: str | None = None, prompt: str = "chat", device: str = "cuda:0",
                 attention: str | None = None, max_pixels: int = img.MAX_PIXELS, processor: str | None = None,
                 image_root: str | None = None, **options):
        if options:
            raise TypeError(f"unknown engine options {sorted(options)}")
        if not model:
            raise ValueError("pass --option model=<checkpoint dir>")
        if prompt not in PROMPTS:
            raise ValueError(f"prompt is one of {PROMPTS}")
        super().__init__(model=model, prompt=prompt, device=device, attention=attention, max_pixels=max_pixels)
        started = time.perf_counter()
        config = modeling.read_config(Path(model))
        self.prompt, self.device, self.max_pixels, self.image_root = prompt, device, max_pixels, image_root
        self.attention, self.layout = attention or config["attention"], config["layout"]
        self.model = modeling.load(Path(model), attention=self.attention, deltanet=config["deltanet"],
                                   dtype=torch.bfloat16, device=device)
        self.model.model.visual.to(device)
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(processor or PROCESSOR)
        tok = self.processor.tokenizer
        self.letter_ids = [tok.convert_tokens_to_ids(letter) for letter in fmt.candidate_codes()[:26]]
        self.codes, self.code_ids = fmt.answer_codes(tok)
        self.image_token_id = self.model.config.image_token_id
        self.provenance = {"kind": "qwen-decision-v3-vision", "model": str(model), "prompt": prompt,
                           "attention": self.attention, "deltanet": config["deltanet"], "layout": self.layout,
                           "backbone_dtype": "bfloat16", "readout_dtype": "float32", "readout": "tied LM head",
                           "processor": str(processor or PROCESSOR), "max_pixels": max_pixels,
                           "image_resize": "images over max_pixels are downscaled to fit (aspect kept, LANCZOS) by the "
                                           "kit's images.pil; then the model's own processor",
                           "processor_image_budget": dict(self.processor.image_processor.size),
                           "load_seconds": time.perf_counter() - started,
                           "policy": ("chat: one user turn (images, then question, lettered options and the letter "
                                      "instruction), default chat template, the readout prefix continued as an "
                                      "assistant turn where the row asks for it, next-token logits of the option "
                                      "letters. decide: v3 format.render with the images as the state, last-token "
                                      "logits of the answer codes, no readout prefix.")}

    def synchronize(self):
        torch.cuda.synchronize(self.device)

    def runtime(self):
        return {"torch": torch.__version__, "gpu": torch.cuda.get_device_name(self.device)}

    def text(self, questions, n_images, prompt):
        key, q, keys = single_choice(questions)
        if self.prompt == "chat":
            content = [{"type": "image"} for _ in range(n_images)] + [{"type": "text", "text": prompt_text(q, prompt)}]
            msgs = [{"role": "user", "content": content}]
            readout = (prompt or {}).get("readout")
            if readout:
                msgs.append({"role": "assistant", "content": readout})
            # The base processor ships no chat template; the tokenizer's renders image parts as IMAGE.
            text = self.processor.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=not readout,
                                                                continue_final_message=bool(readout))
            return key, keys, text, self.letter_ids[:len(keys)]
        state = IMAGE * n_images if n_images else {}
        text = fmt.render(self.processor.tokenizer, state, q, self.codes[:len(keys)], self.layout)
        return key, keys, text, self.code_ids[:len(keys)]

    @torch.no_grad()
    def __call__(self, state, questions, images=(), prompt=None):
        if state not in ({}, None, ""):
            raise Unsupported("vision rows carry their context in the images; this engine takes an empty state")
        pictures = [img.pil(p, self.image_root, self.max_pixels).convert("RGB") for p in images or []]
        key, keys, text, ids = self.text(questions, len(pictures), prompt)
        inputs = self.processor(text=[text], images=pictures or None, return_tensors="pt").to(self.device)
        hidden = self.model.model(**inputs, use_cache=False).last_hidden_state[0, -1]
        weight = self.model.get_output_embeddings().weight[torch.tensor(ids, device=self.device)]
        logits = (weight.float() @ hidden.float()).tolist()
        choice = keys[max(range(len(keys)), key=logits.__getitem__)]
        probs = torch.tensor(logits).softmax(-1).tolist()
        raw = {"logits": logits, "image_tokens": int((inputs["input_ids"] == self.image_token_id).sum()),
               "image_pixels": [p.width * p.height for p in pictures]}
        return answer(key, keys, choice, probs, self.name, {"input_tokens": int(inputs["input_ids"].shape[1])}), raw
