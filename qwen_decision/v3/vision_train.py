"""Image fine-tuning path for v3 checkpoints: packed image rows, LoRA on the LM and optionally the ViT, speed bench.

The text path's speed work carries over unchanged: BF16 compute with FP32 masters (train.MasterAdamW), padding-free
packing through varlen attention and FLA with cu_seqlens, compiled element-wise blocks, token-budget microbatches.
Images add four things:
  positions   M-RoPE (t, h, w) ids per row, built on the CPU as Qwen3_5Model.get_rope_index builds them, then
              concatenated, so every packed row restarts at 0 as packed text rows do.
  ViT         one varlen_attn call over all images of a microbatch instead of transformers' per-image SDPA loop
              (which reads the lengths back to the host in every layer). The grid helpers (cu_seqlens, rotary
              ids, position-embedding interpolation) arrive precomputed from the CPU; transformers.vision_utils
              pops them from the forward's kwargs.
  frozen ViT  with no adapter in it, the ViT runs under no_grad: no activations kept, no backward.
  loading     decoding, the kit's 1.6 MP cap (decision_index.vision.images.pil, as VisionEngine) and the image
              processor run in DataLoader workers; one item is one optimizer step, split and packed there.
`check` compares the packed path with transformers' own forward row by row; `bench` times training steps on rows
built from The Cauldron (a training source), and nothing it trains is saved.

`fit` trains LoRA on a vision_mix training file, one pass, 16 rows a step, warmup then cosine to 10% (the text
screens' schedule). Image items render as VisionEngine's decide prompt (the images as the state, the kit's option
letters), with the options shuffled and the codes reassigned as train.Renderer does; text rows go through
train.Renderer with train.py's defaults. Arms: frozen (LoRA in the LM) and vit (LoRA in the ViT too). After
training it evaluates through the same packed forward without augmentation: the held-out benchmarks (selection),
the mix's image dev items, a v3-gold6 dev sample (text retention) and optionally the board's public vision rows
(kit scorer). `eval-base` runs those on the untrained base, `report` compares the screen arms, `merge` folds a
fit's adapters into a full checkpoint for the kit runner.

  PYTHONPATH=.:external/decision-index .venv/bin/python -m qwen_decision.v3.vision_train check
  PYTHONPATH=.:external/decision-index .venv/bin/python -m qwen_decision.v3.vision_train bench --preset fast|fast-vit|hf
  PYTHONPATH=.:external/decision-index .venv/bin/python -m qwen_decision.v3.vision_train fit --name screen-frozen-s1 \
      --arm frozen --mix screen-16k --seed 1
  PYTHONPATH=.:external/decision-index .venv/bin/python -m qwen_decision.v3.vision_train eval-base --name base-packed
  PYTHONPATH=.:external/decision-index .venv/bin/python -m qwen_decision.v3.vision_train report --prefix screen
  PYTHONPATH=.:external/decision-index .venv/bin/python -m qwen_decision.v3.vision_train merge --name <fit>
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import random
import re
import statistics
import time
from pathlib import Path
from types import SimpleNamespace

import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from decision_index.vision import images as img
from decision_index.vision import rows as kit_rows
from transformers import AutoProcessor
from transformers import vision_utils as vu
from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen

from qwen_decision.v3 import format as fmt
from qwen_decision.v3 import lora_experiment as lora
from qwen_decision.v3 import modeling, train

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "runs/v3/wise80-g6-g7e2"
PROCESSOR = ROOT / "models/qwen3.5-0.8b"  # v3 checkpoints carry the tokenizer but not the image processor
OUT = ROOT / "runs/vision-train"
IMAGE = "<|vision_start|><|image_pad|><|vision_end|>"
# lora_placement_experiment's attn-mlp arm (q/k/v/o of the 6 attention layers and all 24 MLPs, 6.39M at rank 16)
LM_TARGETS = (("self_attn", ("q_proj", "k_proj", "v_proj", "o_proj")), ("mlp", ("gate_proj", "up_proj", "down_proj")))
VIT_TARGETS = (("attn", ("qkv", "proj")), ("mlp", ("linear_fc1", "linear_fc2")))
BENCH_SUBSETS = ("ai2d", "vqav2", "textvqa", "docvqa", "infographic_vqa", "nlvr2")
STEP_ROWS = 16  # lora_experiment's effective batch
NEVER = 1 << 40
MIX = ROOT / "data/vision-mix"
HELDOUT = ROOT / "data/vision-heldout"
BOARD = ROOT / "data/vision-work/vision"
TEXT_DEV = ROOT / "data/mix/v3-gold6.dev.jsonl"
TEXT_DEV_PER_SOURCE = 24
# train.py's defaults, which trained the base: pad with pool options, noul as choice, shuffle or sample the codes
TEXT_ARGS = SimpleNamespace(layout="d3", pointer_positions=False, pad_probability=.35, noul_choice_probability=.2,
                            code_shuffle_probability=.2, code_sample_probability=.2)
ARMS = {"frozen": "fast", "vit": "fast-vit"}  # screen arm -> speed preset
PRESETS = {
    # what plain transformers gives: padded rows, per-image ViT loop, every layer checkpointed, loading in the loop
    "hf": {"pack": False, "vit_attention": "loop", "compile": False, "workers": 0, "checkpoint_above": 0,
           "micro_rows": 4, "token_budget": 16384, "kernel": "sdpa", "vit_lora": False, "vit_checkpoint_above": 0},
    # 8k-token microbatches with no checkpointing: the fastest that fits the 3090 with room (bench, 2026-11-01:
    # 1.30 s/step, 15.2 GB; 16k checkpointed 1.61 s, 12k uncheckpointed 1.34 s at 21.9 GB, 6k 1.33 s)
    "fast": {"pack": True, "vit_attention": "varlen", "compile": True, "workers": 8, "checkpoint_above": 8192,
             "token_budget": 8192, "kernel": "varlen", "vit_lora": False, "vit_checkpoint_above": NEVER},
    # LoRA in the ViT too: 4k microbatches, nothing checkpointed (2.23 s/step, 13.5 GB; at 8k the uncheckpointed
    # ViT ran out of memory and a checkpointed one took 2.48 s)
    "fast-vit": {"pack": True, "vit_attention": "varlen", "compile": True, "workers": 8, "checkpoint_above": 4096,
                 "token_budget": 4096, "kernel": "varlen", "vit_lora": True, "vit_checkpoint_above": NEVER},
}


def vision_attention(self, hidden_states, cu_seqlens, position_embeddings=None, max_seqlen=None, **kwargs):
    """Qwen3_5VisionAttention.forward with every image of the call in one varlen_attn launch (no mask, no loop)."""
    from torch.nn.attention.varlen import varlen_attn

    n = hidden_states.shape[0]
    q, k, v = self.qkv(hidden_states).reshape(n, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
    cos, sin = position_embeddings
    q, k = qwen.apply_rotary_pos_emb_vision(q, k, cos, sin)
    if max_seqlen is None:
        max_seqlen = int((cu_seqlens[1:] - cu_seqlens[:-1]).max())
    cu = cu_seqlens.to(torch.int32)
    out = varlen_attn(*(x.contiguous().to(torch.bfloat16) for x in (q, k, v)), cu, cu, max_seqlen, max_seqlen,
                      is_causal=False)
    return self.proj(out.reshape(n, -1).to(hidden_states.dtype))


STOCK_VISION_ATTENTION = qwen.Qwen3_5VisionAttention.forward


def use_vision_attention(kind: str) -> None:
    qwen.Qwen3_5VisionAttention.forward = vision_attention if kind == "varlen" else STOCK_VISION_ATTENTION


def rope_positions(mm_types: list[int], grid: list[list[int]], merge: int) -> torch.Tensor:
    """[3, L] M-RoPE ids of one unpadded row, as Qwen3_5Model.get_rope_index computes them (check compares)."""
    pieces, position, grids = [], 0, iter(grid)
    for kind, group in itertools.groupby(mm_types):
        n = len(list(group))
        if kind == 0:
            pieces.append(torch.arange(n).view(1, -1).expand(3, -1) + position)
            position += n
            continue
        t, h, w = next(grids)
        tt, hh, ww = torch.meshgrid(torch.arange(t), torch.arange(h // merge) + position,
                                    torch.arange(w // merge) + position, indexing="ij")
        ids = torch.stack([tt, hh, ww]).reshape(3, -1)
        ids[0] += position
        pieces.append(ids)
        position += max(h, w) // merge
    return torch.cat(pieces, 1)


class Encoder:
    """Row -> v3 prompt ids with the image tokens expanded, pixel patches, grid and M-RoPE ids (CPU only)."""

    def __init__(self, layout: str = "d3", max_pixels: int = img.MAX_PIXELS, text_pool: list[str] | None = None):
        self.processor = AutoProcessor.from_pretrained(PROCESSOR)
        self.tokenizer = self.processor.tokenizer
        self.codes, self.code_ids = fmt.answer_codes(self.tokenizer)
        self.layout, self.max_pixels = layout, max_pixels
        self.merge = self.processor.image_processor.merge_size
        self.renderer = train.Renderer(self.tokenizer, TEXT_ARGS, text_pool or [])

    def text(self, row) -> str:
        return fmt.render(self.tokenizer, IMAGE * len(row["images"]), row["question"], self.codes[:len(row["target"])],
                          self.layout)

    def pictures(self, row, root=None) -> list:
        return [img.pil(ref, root, self.max_pixels).convert("RGB") for ref in row["images"]]

    def __call__(self, row) -> dict:
        return self.encode(self.text(row), self.pictures(row), self.code_ids[:len(row["target"])], row["target"])

    def encode(self, text: str, pictures: list, code_ids: list[int], target: list[float]) -> dict:
        out = self.processor(text=[text], images=pictures or None, return_tensors="pt")
        grid = out["image_grid_thw"].tolist() if "image_grid_thw" in out else []
        mm = out["mm_token_type_ids"][0].tolist()
        return {"input_ids": out["input_ids"][0].tolist(),
                "pixel_values": out["pixel_values"].to(torch.bfloat16) if grid else None,
                "grid": grid, "positions": rope_positions(mm, grid, self.merge), "code_ids": code_ids, "target": target}

    def kit(self, question: dict, images: list, target: list[float], assigned: list[int] | None = None,
            root=None) -> dict:
        """A kit-format question as VisionEngine's decide prompt renders it: the images as the state."""
        assigned = list(range(len(target))) if assigned is None else assigned
        pictures = [img.pil(ref, root, self.max_pixels).convert("RGB") for ref in images]
        text = fmt.render(self.tokenizer, IMAGE * len(pictures) if pictures else {}, question,
                          [self.codes[c] for c in assigned], self.layout)
        return self.encode(text, pictures, [self.code_ids[c] for c in assigned], target)

    def row(self, row: dict, rng: random.Random | None) -> dict:
        """A training-file row (vision_mix: modality image or text); rng=None renders it without augmentation."""
        if row["modality"] == "text":
            r = self.renderer(row, rng)
            n = len(r["input_ids"])
            return {"input_ids": r["input_ids"], "pixel_values": None, "grid": [],
                    "positions": torch.arange(n).view(1, -1).expand(3, -1), "code_ids": r["code_ids"],
                    "target": r["target"]}
        question, assigned, target = item_view(row, rng)
        return self.kit(question, row["images"], target, assigned, root=MIX)


def item_view(it: dict, rng: random.Random | None) -> tuple[dict, list[int], list[float]]:
    """A mix item as a kit request (rows.request: the options under A, B, C, ...), its codes and target.

    With an rng, an unordered item gets the text renderer's augmentation (train.Renderer): the options shuffled,
    then the codes kept in order, shuffled, or drawn from all 255 (TEXT_ARGS' probabilities); ordered items keep
    both. rng=None keeps the stored order and the codes A, B, C, ... as on the board.
    """
    k = len(it["options"])
    order, assigned = list(range(k)), list(range(k))
    if rng is not None and not it["ordered"]:
        rng.shuffle(order)
        draw = rng.random()
        if draw < TEXT_ARGS.code_shuffle_probability:
            rng.shuffle(assigned)
        elif draw < TEXT_ARGS.code_shuffle_probability + TEXT_ARGS.code_sample_probability:
            assigned = sorted(rng.sample(range(fmt.MAX_OPTIONS), k))
    question = {"type": "choice", "instructions": it["question"],
                "criteria": {kit_rows.OPTION_LETTERS[i]: it["options"][j] for i, j in enumerate(order)}}
    return question, assigned, [float(j == it["gold"]) for j in order]


def codes_and_targets(rows) -> dict:
    union = sorted({c for r in rows for c in r["code_ids"]})
    where = {c: i for i, c in enumerate(union)}
    kmax = max(len(r["code_ids"]) for r in rows)
    return {"union": torch.tensor(union),
            "columns": torch.tensor([[where[c] for c in r["code_ids"]] + [0] * (kmax - len(r["code_ids"])) for r in rows]),
            "code_mask": torch.tensor([[True] * len(r["code_ids"]) + [False] * (kmax - len(r["code_ids"])) for r in rows]),
            "targets": torch.tensor([r["target"] + [0.] * (kmax - len(r["target"])) for r in rows])}


def vision_inputs(grids: list[list[int]], pixel_values: list[torch.Tensor], config) -> dict:
    """Pixel patches of a microbatch and the ViT's grid helpers, computed here instead of on the GPU."""
    grid = torch.tensor(grids)
    cu = vu.get_vision_cu_seqlens(grid)
    indices, weights = vu.get_vision_interpolation_indices_and_weights(
        grid, num_grid_per_side=int(config["num_position_embeddings"] ** .5), mode="bilinear", align_corners=True,
        spatial_merge_size=config["spatial_merge_size"])
    return {"pixel_values": torch.cat(pixel_values), "grid": grid,
            "vision": {"cu_seqlens": cu, "max_seqlen": int((cu[1:] - cu[:-1]).max()),
                       "position_ids": vu.get_vision_position_ids(grid, config["spatial_merge_size"]),
                       "interp_indices": indices, "interp_weights": weights}}


def pack(rows, image_token: int, vision_config) -> dict:
    """One padding-free microbatch, everything shape-dependent known on the CPU (as modeling.pack_rows)."""
    lengths = [len(r["input_ids"]) for r in rows]
    starts = [0]
    for n in lengths:
        starts.append(starts[-1] + n)
    ids = torch.tensor([t for r in rows for t in r["input_ids"]])
    batch = {"ids": ids, "positions": torch.cat([r["positions"] for r in rows], 1), "cu": torch.tensor(starts),
             "cu_cpu": torch.tensor(starts), "longest": max(lengths), "last": torch.tensor(starts[1:]) - 1,
             "image_slots": (ids == image_token).nonzero().squeeze(1), **codes_and_targets(rows)}
    with_images = [r for r in rows if r["grid"]]
    if with_images:
        batch |= vision_inputs([g for r in with_images for g in r["grid"]], [r["pixel_values"] for r in with_images],
                               vision_config)
    return batch


class Steps(torch.utils.data.Dataset):
    """Item i: optimizer step i's microbatches, encoded and packed (or padded, for the hf preset) in the worker."""

    def __init__(self, steps: list[list[dict]], cfg: SimpleNamespace):
        self.steps, self.cfg, self.encoder = steps, cfg, None
        self.vision_config = json.loads((PROCESSOR / "config.json").read_text())["vision_config"]

    def __len__(self):
        return len(self.steps)

    def __getitem__(self, i):
        if self.encoder is None:
            self.encoder = Encoder(max_pixels=self.cfg.max_pixels)
        enc, rows = self.encoder, self.steps[i]
        if not self.cfg.pack:
            return [padded(enc, rows[j:j + self.cfg.micro_rows]) for j in range(0, len(rows), self.cfg.micro_rows)]
        image_token = enc.tokenizer.convert_tokens_to_ids("<|image_pad|>")
        encoded = [enc(r) for r in rows]
        budget = SimpleNamespace(pack=True, micro_batch=len(rows), token_budget=self.cfg.token_budget)
        return [pack(g, image_token, self.vision_config) for g in train.microbatches(encoded, budget)]


def padded(enc: Encoder, rows) -> dict:
    """The processor's own padded batch (right padding) and the rows' codes, as a stock training loop builds it."""
    pictures = [p for r in rows for p in enc.pictures(r)]
    out = enc.processor(text=[enc.text(r) for r in rows], images=pictures or None, padding=True, return_tensors="pt")
    codes = [{"code_ids": enc.code_ids[:len(r["target"])], "target": r["target"]} for r in rows]
    return {"hf": dict(out), **codes_and_targets(codes)}


def to_device(x, device="cuda"):
    if isinstance(x, torch.Tensor):
        return x.to(device, non_blocking=True)
    if isinstance(x, dict):
        return {k: (v if k == "cu_cpu" else to_device(v, device)) for k, v in x.items()}
    return x


def readout(model, h, batch) -> torch.Tensor:
    weight = model.get_output_embeddings().weight.index_select(0, batch["union"])
    scores = F.linear(h.float(), weight.float())
    return scores.gather(1, batch["columns"]).masked_fill(~batch["code_mask"], modeling.MASK)


def packed_logits(model, batch, vit_grad: bool = False, embeds_hook=None) -> torch.Tensor:
    """[B, K] code logits of a pack() microbatch on the device."""
    text = model.model.language_model
    embeds = text.embed_tokens(batch["ids"])
    if "grid" in batch:
        with torch.set_grad_enabled(vit_grad and torch.is_grad_enabled()):
            features = model.model.visual(batch["pixel_values"], grid_thw=batch["grid"], **batch["vision"]).pooler_output
        embeds = embeds.index_copy(0, batch["image_slots"], features.to(embeds.dtype))
    if embeds_hook is not None:
        embeds = embeds_hook(embeds)
    hidden = text(inputs_embeds=embeds[None], position_ids=batch["positions"][:, None], cu_seq_lens_q=batch["cu"],
                  cu_seq_lens_k=batch["cu"], max_length_q=batch["longest"], max_length_k=batch["longest"],
                  cu_seqlens_cpu=batch["cu_cpu"], use_cache=False).last_hidden_state
    return readout(model, hidden[0, batch["last"]], batch)


def hf_logits(model, batch) -> torch.Tensor:
    """[B, K] code logits through transformers' multimodal forward on a padded batch."""
    inputs = batch["hf"]
    hidden = model.model(**inputs, use_cache=False).last_hidden_state
    mask = inputs["attention_mask"]
    return readout(model, hidden[torch.arange(len(mask), device=mask.device), modeling.last_positions(mask)], batch)


def add_lora(model, vit: bool) -> None:
    model.requires_grad_(False)
    blocks = [(layer, LM_TARGETS) for layer in model.model.language_model.layers]
    blocks += [(block, VIT_TARGETS) for block in model.model.visual.blocks] if vit else []
    for parent, targets in blocks:
        for name, names in targets:
            if hasattr(parent, name):
                module = getattr(parent, name)
                for n in names:
                    setattr(module, n, lora.LoRALinear(getattr(module, n)))


def load_model(cfg: SimpleNamespace):
    if cfg.compile:
        modeling.compile_elementwise()
    use_vision_attention(cfg.vit_attention)
    config = modeling.read_config(BASE)
    model = modeling.load(BASE, attention=config["attention"], deltanet=config["deltanet"], dtype=torch.bfloat16,
                          kernel=cfg.kernel)
    model.model.visual.to("cuda")
    add_lora(model, cfg.vit_lora)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    set_vit_checkpointing(model, False)
    return model


def set_vit_checkpointing(model, enabled: bool) -> None:
    for block in model.model.visual.blocks:
        block.gradient_checkpointing = enabled


def cauldron() -> Path:
    return next((ROOT / "data/raw").glob("HuggingFaceM4__the_cauldron@*"))


def bench_rows(per: int = 48, seed: int = 0) -> list[dict]:
    """4-option rows (2 for NLVR2) from the first rows of six Cauldron subsets; open answers get 3 other answers."""
    rng = random.Random(seed)
    rows = []
    for subset in BENCH_SUBSETS:
        raw = []
        for batch in pq.ParquetFile(min((cauldron() / subset).glob("*.parquet"))).iter_batches(batch_size=64):
            raw += batch.to_pylist()
            if len(raw) >= 4 * per:
                break
        answers = sorted({r["texts"][0]["assistant"].strip() for r in raw})
        kept = 0
        for r in raw:
            t = r["texts"][0]
            gold = t["assistant"].strip()
            if subset == "ai2d":
                m = re.match(r"Question: (.*?)\nChoices:\n(.*)\nAnswer with the letter\.", t["user"], re.DOTALL)
                options = dict(re.findall(r"^([A-Z])\. (.*)$", m.group(2), re.MULTILINE)) if m else {}
                question, letter = (m.group(1) if m else ""), gold.removeprefix("Answer: ")
                if letter not in options or len(set(options.values())) != len(options):
                    continue
                gold, options = options[letter], list(options.values())
            elif subset == "nlvr2":
                question, gold, options = t["user"], gold.rstrip(".").capitalize(), ["Yes", "No"]
            else:
                question = t["user"]
                options = [gold] + rng.sample([a for a in answers if a.lower() != gold.lower()], 3)
                rng.shuffle(options)
            rows.append({"source": subset, "images": [i["bytes"] for i in r["images"]],
                         "question": {"type": "choice", "instructions": question, "criteria": {o: None for o in options}},
                         "target": [float(o == gold) for o in options]})
            kept += 1
            if kept == per:
                break
    rng.shuffle(rows)
    return rows


def check(out: Path) -> None:
    """Packed path vs transformers' forward on single rows, M-RoPE ids, the varlen ViT and row independence."""
    torch.manual_seed(0)
    rows = bench_rows(2, seed=1)  # 12 rows, 2 per subset; the NLVR2 rows carry two images
    cfg = SimpleNamespace(compile=False, vit_attention="loop", kernel="varlen", vit_lora=False)
    model = load_model(cfg)
    model.eval()
    enc = Encoder()
    vision_config = json.loads((PROCESSOR / "config.json").read_text())["vision_config"]
    image_token = enc.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    report = {"rows": len(rows), "images": sum(len(r["images"]) for r in rows), "rope_mismatches": 0}
    reference = []
    with torch.no_grad():
        for r in rows:  # stock path: transformers' own positions and per-image SDPA ViT, one row per forward
            inputs = enc.processor(text=[enc.text(r)], images=enc.pictures(r), return_tensors="pt").to("cuda")
            stock, _ = model.model.get_rope_index(inputs["input_ids"], inputs["mm_token_type_ids"],
                                                  inputs["image_grid_thw"])
            mine = enc(r)["positions"]
            report["rope_mismatches"] += int(not torch.equal(stock[:, 0].cpu(), mine))
            hidden = model.model(**inputs, use_cache=False).last_hidden_state[0, -1]
            reference.append(readout(model, hidden[None], to_device(codes_and_targets([enc(r)])))[0])
        encoded = [enc(r) for r in rows]
        batch = to_device(pack(encoded, image_token, vision_config))
        stock_features = model.model.visual(batch["pixel_values"], grid_thw=batch["grid"]).pooler_output
        use_vision_attention("varlen")
        fast_features = model.model.visual(batch["pixel_values"], grid_thw=batch["grid"], **batch["vision"]).pooler_output
        packed = packed_logits(model, batch)
    ref = torch.nn.utils.rnn.pad_sequence(reference, batch_first=True, padding_value=modeling.MASK)
    real = batch["code_mask"]
    # Both ViT paths compute in BF16; against an FP32 ViT their mean token cosines were 0.996967 (varlen) and
    # 0.996966 (stock) with the same worst token (0.625), so the varlen-vs-stock gap is BF16 rounding.
    token_cosine = F.cosine_similarity(stock_features.float(), fast_features.float(), dim=-1)
    report["vit_max_abs_diff"] = (stock_features - fast_features).abs().max().item()
    report["vit_cosine_min"], report["vit_cosine_mean"] = token_cosine.min().item(), token_cosine.mean().item()
    report["logit_max_abs_diff"] = (ref - packed)[real].abs().max().item()
    report["logit_scale"] = ref[real].abs().max().item()
    top2 = ref.masked_fill(~real, -1e30).topk(2, -1).values
    disagree = (ref.masked_fill(~real, -1e30).argmax(-1) != packed.argmax(-1)).nonzero().squeeze(1).tolist()
    report["argmax_agree"] = len(rows) - len(disagree)
    # a flip is accepted only on a near-tie: the stock top-2 margin under twice the largest logit difference
    report["disagreement_margins"] = [(top2[i, 0] - top2[i, 1]).item() for i in disagree]
    # row independence: row 0's logits must not depend on any other row's input embeddings (image tokens included);
    # the 4 shortest rows, since an eval-mode backward keeps every activation
    small = to_device(pack(sorted(encoded, key=lambda r: len(r["input_ids"]))[:4], image_token, vision_config))
    leaf = {}

    def hook(embeds):
        leaf["x"] = embeds.detach().requires_grad_(True)
        return leaf["x"]
    with torch.enable_grad():
        packed_logits(model, small, embeds_hook=hook)[0][small["code_mask"][0]].sum().backward()
    grad = leaf["x"].grad
    end = int(small["cu_cpu"][1])
    report["independence_rows"] = 4
    report["row0_grad_norm"] = grad[:end].norm().item()
    report["other_rows_grad_max"] = grad[end:].abs().max().item()
    report["passed"] = (report["rope_mismatches"] == 0 and report["vit_cosine_mean"] > .9999
                        and all(m < 2 * report["logit_max_abs_diff"] for m in report["disagreement_margins"])
                        and report["other_rows_grad_max"] == 0. and report["row0_grad_norm"] > 0)
    out.mkdir(parents=True, exist_ok=True)
    (out / "check.json").write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report, indent=1))
    if not report["passed"]:
        raise SystemExit(1)


def worker_init(_):
    torch.set_num_threads(2)


def bench(args) -> None:
    cfg = SimpleNamespace(**(PRESETS[args.preset] | {k: v for k, v in vars(args).items() if v is not None}))
    torch.manual_seed(0)
    model = load_model(cfg)
    model.train()
    opt = train.MasterAdamW(model, lora.LORA_LR)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    rows = bench_rows()
    n = cfg.warmup + cfg.steps
    steps = [rows[i * STEP_ROWS:(i + 1) * STEP_ROWS] for i in range(n)]
    loader = torch.utils.data.DataLoader(Steps(steps, cfg), batch_size=None, num_workers=cfg.workers, pin_memory=True,
                                         prefetch_factor=4 if cfg.workers else None, worker_init_fn=worker_init)
    timing = {k: [] for k in ("data", "fwd_bwd", "optimizer", "step")}
    counts = {k: [] for k in ("tokens", "computed_tokens", "image_tokens", "patches", "images")}
    torch.cuda.reset_peak_memory_stats()
    batches = iter(loader)
    for i in range(n):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        groups = next(batches)
        t1 = time.perf_counter()
        opt.zero_grad()
        tokens = padded_tokens = image_tokens = patches = images = 0
        for cpu in groups:
            g = to_device(cpu)
            if cfg.pack:
                size = len(cpu["ids"])
                train.set_checkpointing(model, size > cfg.checkpoint_above)
                if cfg.vit_lora and "grid" in cpu:
                    set_vit_checkpointing(model, len(cpu["pixel_values"]) > cfg.vit_checkpoint_above)
                z = packed_logits(model, g, vit_grad=cfg.vit_lora)
                tokens += size
                image_tokens += len(cpu["image_slots"])
                if "grid" in cpu:
                    patches, images = patches + len(cpu["pixel_values"]), images + len(cpu["grid"])
            else:
                train.set_checkpointing(model, True)
                set_vit_checkpointing(model, cfg.vit_lora)
                z = hf_logits(model, g)
                tokens += int(cpu["hf"]["attention_mask"].sum())
                padded_tokens += int(cpu["hf"]["attention_mask"].numel())  # padded width x rows: what the GPU computes
                image_tokens += int((cpu["hf"]["input_ids"] == model.config.image_token_id).sum())
                patches += len(cpu["hf"]["pixel_values"])
                images += len(cpu["hf"]["image_grid_thw"])
            (train.soft_ce(z, g["targets"]).sum() / STEP_ROWS).backward()
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        opt.clip_grad_norm(1.)
        opt.step()
        torch.cuda.synchronize()
        t3 = time.perf_counter()
        if i >= cfg.warmup:
            for k, v in zip(timing, (t1 - t0, t2 - t1, t3 - t2, t3 - t0)):
                timing[k].append(v)
            for k, v in zip(counts, (tokens, padded_tokens or tokens, image_tokens, patches, images)):
                counts[k].append(v)
    total = sum(timing["step"])
    result = {"preset": args.preset, **{k: getattr(cfg, k) for k in
                                         ("pack", "vit_attention", "compile", "workers", "checkpoint_above", "token_budget",
                                          "vit_lora", "max_pixels", "kernel")},
              "trainable": trainable, "steps": cfg.steps, "rows_per_step": STEP_ROWS,
              **{f"{k}_median_s": round(statistics.median(v), 4) for k, v in timing.items()},
              "rows_per_s": round(STEP_ROWS * cfg.steps / total, 2),
              "lm_tokens_per_step": round(statistics.mean(counts["tokens"])),
              "lm_tokens_computed_per_step": round(statistics.mean(counts["computed_tokens"])),
              "lm_tokens_per_s": round(sum(counts["tokens"]) / total),
              "image_tokens_per_step": round(statistics.mean(counts["image_tokens"])),
              "patches_per_step": round(statistics.mean(counts["patches"])),
              "images_per_step": round(statistics.mean(counts["images"]), 1),
              "peak_gb": round(torch.cuda.max_memory_allocated() / 2**30, 1)}
    print(json.dumps(result), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "bench.jsonl").open("a") as f:
        f.write(json.dumps(result | {"label": args.label}) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").split("\n") if line.strip()]


def text_pool(rows: list[dict]) -> list[str]:
    """train.py's padding pool: every option of the unordered text choice rows."""
    return sorted({o for r in rows if r["modality"] == "text" and r["question"]["type"] == "choice" and not r["ordered"]
                   for o in r["question"]["criteria"]})


def gold_text(rows: list[dict]) -> list[dict]:
    """Text rows get their target from the gold (train.gold_targets), as the base's training did."""
    text = iter(train.gold_targets([r for r in rows if r["modality"] == "text"]))
    return [next(text) if r["modality"] == "text" else r for r in rows]


class MixSteps(torch.utils.data.Dataset):
    """Item i: training step i's rows (each with its own augmentation seed), encoded and packed in the worker."""

    def __init__(self, steps: list[list[tuple[dict, str]]], cfg: SimpleNamespace, pool: list[str]):
        self.steps, self.cfg, self.pool, self.encoder = steps, cfg, pool, None
        self.vision_config = json.loads((PROCESSOR / "config.json").read_text())["vision_config"]

    def __len__(self):
        return len(self.steps)

    def __getitem__(self, i):
        if self.encoder is None:
            self.encoder = Encoder(max_pixels=self.cfg.max_pixels, text_pool=self.pool)
        enc = self.encoder
        image_token = enc.tokenizer.convert_tokens_to_ids("<|image_pad|>")
        encoded = [enc.row(r, random.Random(seed)) for r, seed in self.steps[i]]
        budget = SimpleNamespace(pack=True, micro_batch=len(encoded), token_budget=self.cfg.token_budget)
        return [pack(g, image_token, self.vision_config) for g in train.microbatches(encoded, budget)]


class EvalChunks(torch.utils.data.Dataset):
    """Evaluation rows in chunks, encoded without augmentation and packed in the worker; rows keep their index.

    An entry is ("kit", row, image root) for a kit-format row (held-out, board) or ("mix", row) for a vision_mix
    row (image dev items, text dev rows).
    """

    def __init__(self, entries: list[tuple], token_budget: int, chunk: int = 32):
        self.entries, self.token_budget, self.chunk, self.encoder = entries, token_budget, chunk, None
        self.vision_config = json.loads((PROCESSOR / "config.json").read_text())["vision_config"]

    def __len__(self):
        return (len(self.entries) + self.chunk - 1) // self.chunk

    def __getitem__(self, i):
        if self.encoder is None:
            self.encoder = Encoder()
        enc = self.encoder
        image_token = enc.tokenizer.convert_tokens_to_ids("<|image_pad|>")
        encoded = []
        for index in range(i * self.chunk, min(len(self.entries), (i + 1) * self.chunk)):
            entry = self.entries[index]
            if entry[0] == "kit":
                _, row, root = entry
                question = row["questions"]["answer"]
                keys = list(question["criteria"])
                target = [float(k == row["expected"]["answer"]) for k in keys]
                e = enc.kit(question, row["images"], target, root=root)
            else:
                e = enc.row(entry[1], None)
            encoded.append(e | {"index": index})
        budget = SimpleNamespace(pack=True, micro_batch=len(encoded), token_budget=self.token_budget)
        return [pack(g, image_token, self.vision_config) | {"index": [r["index"] for r in g]}
                for g in train.microbatches(encoded, budget)]


@torch.no_grad()
def predict(model, entries: list[tuple], workers: int = 8, token_budget: int = 16384) -> list[dict]:
    """Per entry: the predicted option (position), whether it holds the target's top probability, and the logits."""
    was_training = model.training
    model.eval()
    train.set_checkpointing(model, False)
    out = [None] * len(entries)
    # spawn, not fork: after training, torch.compile's threads are live and forked workers deadlocked (smoke fit)
    loader = torch.utils.data.DataLoader(EvalChunks(entries, token_budget), batch_size=None, num_workers=workers,
                                         pin_memory=True, prefetch_factor=4 if workers else None,
                                         worker_init_fn=worker_init,
                                         multiprocessing_context="spawn" if workers else None)
    for groups in loader:
        for cpu in groups:
            g = to_device(cpu)
            z = packed_logits(model, g).float().cpu()
            targets, mask = cpu["targets"], cpu["code_mask"]
            pred = z.argmax(-1)
            for j, index in enumerate(cpu["index"]):
                k = int(mask[j].sum())
                t = targets[j, :k]
                out[index] = {"pred": int(pred[j]), "correct": bool(t[pred[j]] >= t.max() - 1e-6), "k": k,
                              "logits": [round(x, 4) for x in z[j, :k].tolist()]}
    model.train(was_training)
    return out


def heldout_entries() -> list[tuple]:
    return [("kit", r, HELDOUT) for r in read_jsonl(HELDOUT / "rows.jsonl")]


def board_entries() -> list[tuple]:
    return [("kit", r, BOARD) for r in read_jsonl(BOARD / "rows-built.jsonl")]


def text_dev_rows() -> list[dict]:
    """TEXT_DEV_PER_SOURCE rows per source of v3-gold6's dev split (by id hash), gold targets."""
    import hashlib

    by = {}
    for r in read_jsonl(TEXT_DEV):
        by.setdefault(r["source"], []).append(r)
    rows = []
    for source in sorted(by):
        ranked = sorted(by[source], key=lambda r: hashlib.sha256(f"text-dev:{r['id']}".encode()).hexdigest())
        rows += [{"modality": "text", **r} for r in ranked[:TEXT_DEV_PER_SOURCE]]
    return gold_text(rows)


def chance_skill(results: list[dict], groups: list[str]) -> dict:
    """Per group: accuracy, chance (mean 1/k) and skill (accuracy - chance) / (1 - chance), as the kit scores."""
    by = {}
    for r, g in zip(results, groups):
        by.setdefault(g, []).append(r)
    out = {}
    for g, rs in sorted(by.items()):
        acc = sum(r["correct"] for r in rs) / len(rs)
        ch = sum(1 / r["k"] for r in rs) / len(rs)
        out[g] = {"rows": len(rs), "accuracy": round(acc, 4), "chance": round(ch, 4),
                  "skill": round((acc - ch) / (1 - ch), 4)}
    return out


def evaluate(model, dest: Path, sets: tuple[str, ...]) -> dict:
    """Held-out index (17-benchmark mean of clipped skills), image dev and text dev (source means) and the board."""
    summary = {}
    for name in sets:
        started = time.time()
        if name == "heldout":
            entries = heldout_entries()
            groups = [e[1]["benchmark"] for e in entries]
        elif name == "board":
            entries = board_entries()
            groups = [e[1]["benchmark"] for e in entries]
        elif name == "image-dev":
            entries = [("mix", r) for r in read_jsonl(MIX / "dev.jsonl")]
            groups = [e[1]["source"] for e in entries]
        else:
            entries = [("mix", r) for r in text_dev_rows()]
            groups = [e[1]["source"] for e in entries]
        results = predict(model, entries)
        with (dest / f"eval-{name}.jsonl").open("w") as f:
            for e, g, r in zip(entries, groups, results):
                f.write(json.dumps({"id": e[1]["id"], "group": g, **r}) + "\n")
        per = chance_skill(results, groups)
        if name == "board":
            from decision_index.vision import score as kit_score

            letters = {e[1]["id"]: list(e[1]["questions"]["answer"]["criteria"])[r["pred"]]
                       for e, r in zip(entries, results)}
            board = kit_score.score([e[1] for e in entries], letters)
            summary[name] = {"index": round(board["public_vision_index"], 2),
                             "benchmarks": {b: round(v["skill"], 4) for b, v in board["benchmarks"].items()}}
        elif name == "heldout":
            summary[name] = {"index": round(100 * statistics.mean(min(1., max(0., v["skill"])) for v in per.values()), 2),
                             "benchmarks": {b: v["skill"] for b, v in per.items()}}
        else:
            summary[name] = {"macro_accuracy": round(statistics.mean(v["accuracy"] for v in per.values()), 4),
                             "sources": {s: v["accuracy"] for s, v in per.items()}}
        summary[name]["rows"] = len(entries)
        summary[name]["seconds"] = round(time.time() - started, 1)
        print("EVAL", name, json.dumps({k: v for k, v in summary[name].items() if k in ("index", "macro_accuracy",
                                                                                     "rows", "seconds")}), flush=True)
    return summary


def adapter_state(model, opt=None) -> dict:
    masters = {id(p): m for p, m in opt.pairs} if opt else {}
    out = {}
    for n, p in model.named_parameters():
        if ".lora_" in n:
            out[n] = p.detach().contiguous()
            if id(p) in masters:
                out[f"master.{n}"] = masters[id(p)].contiguous()
    return out


def fit(args) -> None:
    from safetensors.torch import save_file

    cfg = SimpleNamespace(**(PRESETS[ARMS[args.arm]] | {"max_pixels": args.max_pixels}))
    dest = OUT / args.name
    if (dest / "result.json").exists():
        print("SKIP_COMPLETE", args.name, flush=True)
        return
    if dest.exists():
        dest.rename(OUT / f"{args.name}.incomplete-{time.time_ns()}")
    dest.mkdir(parents=True)
    mix_path = MIX / f"{args.mix}.jsonl"
    rows = gold_text(read_jsonl(mix_path))
    order = list(range(len(rows)))
    random.Random(f"order:{args.seed}").shuffle(order)
    n_steps = (len(rows) + STEP_ROWS - 1) // STEP_ROWS
    steps = [[(rows[i], f"view:{args.seed}:{s}:{j}") for j, i in enumerate(order[s * STEP_ROWS:(s + 1) * STEP_ROWS])]
             for s in range(n_steps)]
    warmup = max(1, n_steps // 16)  # 64 of 1,024 steps, as the text screens (training_method_experiment)
    config = {"arm": args.arm, "preset": ARMS[args.arm], "seed": args.seed, "mix": str(mix_path.relative_to(ROOT)),
              "mix_sha256": train.sha256_file(mix_path), "rows": len(rows), "steps": n_steps, "warmup": warmup,
              "step_rows": STEP_ROWS, "lr": lora.LORA_LR, "lora": {"rank": lora.RANK, "alpha": lora.ALPHA,
                                                                   "dropout": lora.DROPOUT},
              "lm_targets": LM_TARGETS, "vit_targets": VIT_TARGETS if cfg.vit_lora else None, "base": str(BASE),
              "text_args": vars(TEXT_ARGS), **{k: getattr(cfg, k) for k in ("token_budget", "checkpoint_above",
                                                                            "max_pixels", "workers")}}
    (dest / "config.json").write_text(json.dumps(config, indent=1) + "\n")
    train.seed_all(args.seed)
    model = load_model(cfg)
    model.train()
    opt = train.MasterAdamW(model, lora.LORA_LR)
    loader = torch.utils.data.DataLoader(MixSteps(steps, cfg, text_pool(rows)), batch_size=None,
                                         num_workers=cfg.workers, pin_memory=True, prefetch_factor=4,
                                         worker_init_fn=worker_init)
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    with (dest / "train.jsonl").open("w") as log:
        for step, groups in enumerate(loader, 1):
            opt.zero_grad()
            n_rows = sum(len(cpu["last"]) for cpu in groups)
            loss_sum = torch.zeros((), device="cuda")
            for cpu in groups:
                g = to_device(cpu)
                train.set_checkpointing(model, len(cpu["ids"]) > cfg.checkpoint_above)
                if cfg.vit_lora and "grid" in cpu:
                    set_vit_checkpointing(model, len(cpu["pixel_values"]) > cfg.vit_checkpoint_above)
                loss = train.soft_ce(packed_logits(model, g, vit_grad=cfg.vit_lora), g["targets"]).sum() / n_rows
                loss.backward()
                loss_sum += loss.detach()
            norm = opt.clip_grad_norm(1.)
            loss_value = loss_sum.item()
            if not (math.isfinite(loss_value) and math.isfinite(norm)):
                raise ValueError(f"nonfinite loss or gradients at step {step}")
            scale = min(step / warmup, 1.) * (.1 + .45 * (1 + math.cos(math.pi * max(0, step - warmup)
                                                                         / max(1, n_steps - warmup))))
            for group in opt.param_groups:
                group["lr"] = lora.LORA_LR * group["lr_scale"] * scale
            opt.step()
            entry = {"step": step, "loss": round(loss_value, 5), "grad_norm": round(norm, 4), "rows": n_rows,
                     "tokens": sum(len(cpu["ids"]) for cpu in groups), "seconds": round(time.monotonic() - started, 1)}
            log.write(json.dumps(entry) + "\n")
            if step % 32 == 0 or step == n_steps:
                log.flush()
                print("FIT", args.name, step, n_steps, entry["loss"], entry["seconds"], flush=True)
                train.atomic_json(dest / "status.json", {"state": "training", "step": step, "steps": n_steps,
                                                         "seconds": entry["seconds"]})
    trained = time.monotonic() - started
    save_file(adapter_state(model, opt), str(dest / "adapter.safetensors"))
    peak = torch.cuda.max_memory_allocated() / 2**30
    del opt
    torch.cuda.empty_cache()
    train.atomic_json(dest / "status.json", {"state": "evaluating"})
    summary = evaluate(model, dest, tuple(args.eval))
    train.atomic_json(dest / "result.json", {"name": args.name, "train_seconds": round(trained, 1),
                                             "peak_gb": round(peak, 1), "eval": summary, "config": config})


def evaluate_base(args) -> None:
    """The base checkpoint with zero-initialized adapters (identical outputs), through the same eval path."""
    dest = OUT / args.name
    dest.mkdir(parents=True, exist_ok=True)
    cfg = SimpleNamespace(**PRESETS["fast"])
    model = load_model(cfg)
    summary = evaluate(model, dest, tuple(args.eval))
    train.atomic_json(dest / "result.json", {"name": args.name, "eval": summary, "base": str(BASE)})


def merge(args) -> None:
    """A full checkpoint of a fit (adapters folded into the weights) for the kit runner and VisionEngine."""
    import shutil

    from safetensors import safe_open
    from safetensors.torch import load_file, save_file

    src = OUT / args.name
    config = json.loads((src / "config.json").read_text())
    adapters = load_file(str(src / "adapter.safetensors"))
    dest = src / "model"
    dest.mkdir(exist_ok=True)
    for f in BASE.iterdir():
        if f.name != "model.safetensors" and f.is_file():
            shutil.copy2(f, dest / f.name)
    scale = lora.ALPHA / lora.RANK
    merged, folded = {}, 0
    with safe_open(str(BASE / "model.safetensors"), "pt") as base:
        names = base.keys()  # a safe_open handle is not a mapping
        for key in names:
            w = base.get_tensor(key)
            stem = key.removesuffix(".weight")
            a, b = adapters.get(f"master.{stem}.lora_a"), adapters.get(f"master.{stem}.lora_b")
            if a is None:
                a, b = adapters.get(f"{stem}.lora_a"), adapters.get(f"{stem}.lora_b")
            if a is not None and key.endswith(".weight"):
                w = (w.float() + scale * (b.float() @ a.float())).to(w.dtype)
                folded += 1
            merged[key] = w.contiguous()
    pairs = sum(k.endswith(".lora_a") and not k.startswith("master.") for k in adapters)
    if folded != pairs:
        raise ValueError(f"folded {folded} of {pairs} adapters: adapter names do not match the checkpoint keys")
    save_file(merged, str(dest / "model.safetensors"), metadata={"format": "pt"})
    (dest / "merge.json").write_text(json.dumps({"base": str(BASE), "adapters": str(src / "adapter.safetensors"),
                                                 "folded": folded, "fit": config}, indent=1) + "\n")
    print(json.dumps({"merged": str(dest), "folded": folded}))


def heldout_index(correct: dict[str, float], groups: dict[str, str], chance: dict[str, float], ids) -> float:
    by = {}
    for i in ids:
        by.setdefault(groups[i], []).append(i)
    skills = []
    for members in by.values():
        acc = sum(correct[i] for i in members) / len(members)
        ch = sum(chance[i] for i in members) / len(members)
        skills.append(min(1., max(0., (acc - ch) / (1 - ch))))
    return 100 * sum(skills) / len(skills)


def report(args) -> None:
    """Screen arms: per-seed and mean held-out index, image and text dev; held-out difference with a paired
    bootstrap (rows resampled within each benchmark, correctness averaged over seeds)."""
    runs = {}
    for path in sorted(OUT.glob(f"{args.prefix}-*-s*/result.json")):
        arm = path.parent.name.removeprefix(f"{args.prefix}-").rsplit("-s", 1)[0]
        runs.setdefault(arm, []).append(path.parent)
    base = OUT / "base-packed"
    table, per_row = {}, {}
    for arm, dirs in sorted(runs.items()) + ([("base", [base])] if (base / "result.json").exists() else []):
        results = [json.loads((d / "result.json").read_text())["eval"] for d in dirs]
        table[arm] = {"seeds": [d.name for d in dirs],
                      **{f"{k}_{m}": [r[k][m] for r in results if k in r] for k, m in
                         (("heldout", "index"), ("image-dev", "macro_accuracy"), ("text-dev", "macro_accuracy"))}}
        for key in list(table[arm]):
            if isinstance(table[arm][key], list) and table[arm][key] and key != "seeds":
                table[arm][f"{key}_mean"] = round(statistics.mean(table[arm][key]), 4)
        rows = [read_jsonl(d / "eval-heldout.jsonl") for d in dirs]
        per_row[arm] = {r["id"]: statistics.mean(float(rs[j]["correct"]) for rs in rows) for j, r in enumerate(rows[0])}
        groups = {r["id"]: r["group"] for r in rows[0]}
        chance = {r["id"]: 1 / r["k"] for r in rows[0]}
    comparisons = {}
    arms = sorted(per_row)
    rng = random.Random(0)
    by_group = {}
    for i, g in groups.items():
        by_group.setdefault(g, []).append(i)
    for a, b in itertools.combinations(arms, 2):
        point = heldout_index(per_row[b], groups, chance, groups) - heldout_index(per_row[a], groups, chance, groups)
        draws = []
        for _ in range(2000):
            ids = [rng.choice(members) for members in by_group.values() for _ in members]
            draws.append(heldout_index(per_row[b], groups, chance, ids) - heldout_index(per_row[a], groups, chance, ids))
        draws.sort()
        comparisons[f"{b} - {a}"] = {"heldout_index": round(point, 2),
                                     "ci95": [round(draws[49], 2), round(draws[1949], 2)]}
    out = {"arms": table, "comparisons": comparisons}
    (OUT / f"{args.prefix}-report.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["check", "bench", "fit", "eval-base", "merge", "report"])
    p.add_argument("--prefix", default="screen")
    p.add_argument("--name")
    p.add_argument("--arm", choices=sorted(ARMS), default="frozen")
    p.add_argument("--mix", default="screen-16k")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eval", nargs="*", default=["heldout", "image-dev", "text-dev"],
                   choices=["heldout", "image-dev", "text-dev", "board"])
    p.add_argument("--preset", choices=sorted(PRESETS), default="fast")
    p.add_argument("--label", default="")
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--steps", type=int, default=12)
    p.add_argument("--max-pixels", type=int, default=img.MAX_PIXELS)
    # preset overrides
    p.add_argument("--vit-lora", type=int, choices=[0, 1])
    p.add_argument("--vit-checkpoint-above", type=int, help="ViT patches per microbatch (vit-lora only)")
    p.add_argument("--workers", type=int)
    p.add_argument("--checkpoint-above", type=int)
    p.add_argument("--token-budget", type=int)
    p.add_argument("--vit-attention", choices=["loop", "varlen"])
    p.add_argument("--compile", type=int, choices=[0, 1])
    args = p.parse_args()
    for flag in ("compile", "vit_lora"):
        if getattr(args, flag) is not None:
            setattr(args, flag, bool(getattr(args, flag)))
    if args.command == "check":
        check(OUT)
    elif args.command == "bench":
        bench(args)
    elif args.command == "fit":
        fit(args)
    elif args.command == "eval-base":
        evaluate_base(args)
    elif args.command == "report":
        report(args)
    else:
        merge(args)


if __name__ == "__main__":
    main()
