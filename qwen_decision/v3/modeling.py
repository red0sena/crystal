"""Qwen3.5 backbone with the v3 attention modes and the answer-code readout.

The readout is the tied LM head restricted to the question's answer codes,
read at the last real prompt token. Nothing is generated.

Attention modes:
  causal     the stock model.
  noncausal  the 6 full-attention layers see the whole prompt (padding keys
             masked); the 18 Gated DeltaNet layers keep their causal scan.

DeltaNet modes:
  causal         the stock left-to-right scan.
  bidirectional  each Gated DeltaNet layer also scans right to left with its own
                 weights, added through a zero-initialized per-channel gate.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file
from transformers import AttentionInterface, Qwen3_5ForConditionalGeneration
from transformers.masking_utils import create_recurrent_attention_mask

ATTENTION_MODES = ("causal", "noncausal")
DELTANET_MODES = ("causal", "bidirectional")
CONFIG_NAME = "v3_config.json"
GATES_NAME = "backward_gates.safetensors"
READOUT_NAME = "readout.safetensors"
MASK = -1e9  # finite, so a zero soft target times a padded column stays 0 instead of NaN


VARLEN = "v3_varlen_noncausal"


def varlen_noncausal_attention(module, query, key, value, attention_mask, dropout=0.0, scaling=None, **kwargs):
    """Bidirectional attention over the real tokens only, with no mask tensor.

    SDPA with a padding mask cannot use the flash kernel; with Qwen3.5's head_dim 256 the memory-
    efficient kernel it falls back to was 5.6x (1k tokens) to 8.4x (4k) slower than causal flash on an
    A6000 for the attention op alone, while packing the real tokens for torch's varlen_attn was 2.7x-5.8x
    faster than the masked call with the same outputs (max |diff| 0.002 in BF16). Only 6 of 24 layers
    are softmax attention, so the whole model gains much less (see enable_noncausal). `attention_mask` is
    the [B, 1, 1, L] boolean key mask that enable_noncausal passes; each row's real tokens must be
    contiguous (left or right padding).
    """
    from torch.nn.attention.varlen import varlen_attn

    if attention_mask is None and kwargs.get("cu_seq_lens_q") is not None:
        # Packed rows (code_logits(packed=True)): one [1, H, N, D] sequence split by cu_seq_lens_q.
        if dropout or query.shape[0] != 1:
            raise ValueError("packed attention takes one packed row and no dropout")
        # No K/V repeat: varlen_attn's flash kernel takes 2 K/V heads for 8 query heads (GQA) directly;
        # measured identical outputs and ~3% faster than repeating K/V 4x, without the 4x K/V copies.
        starts = kwargs["cu_seq_lens_q"].to(torch.int32)
        longest = int(kwargs["max_length_q"])
        q, k, v = (x[0].transpose(0, 1).to(torch.bfloat16) for x in (query, key, value))
        out = varlen_attn(q, k, v, starts, starts, longest, longest, is_causal=False)
        return out[None].to(query.dtype), None
    if attention_mask is None or attention_mask.dtype != torch.bool or tuple(attention_mask.shape[1:3]) != (1, 1):
        raise ValueError("expected the [B, 1, 1, L] boolean key mask from enable_noncausal")
    if dropout:
        raise ValueError("varlen noncausal attention has no dropout")
    batch, heads, length, dim = query.shape
    if scaling is not None and abs(scaling - dim ** -0.5) > 1e-9:
        raise ValueError("varlen_attn uses the default 1/sqrt(head_dim) scale")
    valid = attention_mask[:, 0, 0, :]
    if (valid.int().diff(dim=-1).abs().sum(-1) > 2).any():
        raise ValueError("padding must be on the left or right of each row")
    groups = heads // key.shape[1]
    if groups > 1:
        key, value = key.repeat_interleave(groups, 1), value.repeat_interleave(groups, 1)
    dtype = query.dtype if query.dtype in (torch.float16, torch.bfloat16) else torch.bfloat16
    index = valid.reshape(-1).nonzero().squeeze(1)
    lengths = valid.sum(-1)
    starts = F.pad(lengths.cumsum(0), (1, 0)).to(torch.int32)
    longest = int(lengths.max())

    def pack(x):
        return x.transpose(1, 2).reshape(batch * length, heads, dim).index_select(0, index).to(dtype)

    out = varlen_attn(pack(query), pack(key), pack(value), starts, starts, longest, longest, is_causal=False)
    full = out.new_zeros(batch * length, heads, dim).index_copy(0, index, out)
    return full.view(batch, length, heads, dim).to(query.dtype), None


AttentionInterface.register(VARLEN, varlen_noncausal_attention)


def fla_causal_conv1d(hidden_states, weight, bias=None, activation=None, **kwargs):
    """Qwen3.5's DeltaNet short convolution through FLA's Triton kernel ([B, D, T] in and out).

    Without the `causal-conv1d` package transformers falls back to an FP32 depthwise F.conv1d, which took
    ~15% of a training step's GPU time (torch profiler, m2 recipe on the A6000). FLA, already required for
    the DeltaNet scan, ships the same causal convolution; it also takes the packed-sequence boundaries.
    """
    from fla.modules.convolution import causal_conv1d

    out, _ = causal_conv1d(hidden_states.transpose(1, 2), weight.to(hidden_states.dtype),
                           None if bias is None else bias.to(hidden_states.dtype), activation=activation,
                           cu_seqlens=kwargs.get("cu_seq_lens_q"), cu_seqlens_cpu=kwargs.get("cu_seqlens_cpu"))
    return out.transpose(1, 2)


def compile_elementwise() -> None:
    """Fuse Qwen3.5's element-wise blocks (RMSNorm, gated RMSNorm, SwiGLU product) with torch.compile.

    After packing, unfused multiply/add/copy kernels from these blocks took ~35% of a training step's
    GPU time (torch profiler), against ~31% for the matmuls. Only these small functions are compiled,
    with dynamic shapes, so packed lengths that change every microbatch do not recompile them; the
    matmuls, attention and DeltaNet kernels run as before.
    """
    from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen

    if getattr(qwen, "_v3_compiled", False):
        return

    # Free functions of tensors, not methods: a compiled method guards on its module instance, so the 49
    # RMSNorm modules each recompiled and hit dynamo's cache limit (8), leaving most of them eager.
    @torch.compile(dynamic=True)
    def rms_norm(x, weight, eps: float):
        y = x.float()
        y = y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + eps)
        return (y * (1.0 + weight.float())).type_as(x)

    @torch.compile(dynamic=True)
    def gated_rms_norm(x, gate, weight, eps: float):
        dtype = x.dtype
        y = x.to(torch.float32)
        y = y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + eps)
        y = weight * y.to(dtype)
        return (y * F.silu(gate.to(torch.float32))).to(dtype)

    silu_mul = torch.compile(lambda gate, up: F.silu(gate) * up, dynamic=True)

    # Flattened to [rows, dim]: the attention q/k norms see strided 4-D head tensors and the layer norms
    # 3-D hidden states, and each layout and stride recompiled the same function.
    def flat(fn, x, *rest):
        return fn(x.reshape(-1, x.shape[-1]), *rest).view(x.shape)

    qwen.Qwen3_5RMSNorm.forward = lambda self, x: flat(rms_norm, x, self.weight, self.eps)
    qwen.Qwen3_5RMSNormGated.forward = lambda self, hidden_states, gate: gated_rms_norm(
        hidden_states.reshape(-1, hidden_states.shape[-1]), gate.reshape(-1, gate.shape[-1]),
        self.weight, self.variance_epsilon).view(hidden_states.shape)
    torch._dynamo.config.recompile_limit = max(torch._dynamo.config.recompile_limit, 32)

    def mlp_forward(self, x):
        return self.down_proj(silu_mul(self.gate_proj(x), self.up_proj(x)))
    qwen.Qwen3_5MLP.forward = mlp_forward
    qwen._v3_compiled = True


class _FixedNB:
    """A Triton kernel launched with NB pinned to 1.

    FLA's conv kernels take NB = cdiv(B*T, 1024) as a tl.constexpr and autotune key, though the kernel
    body never reads it (fla 0.5.2 modules/conv/triton). Every new token count therefore compiled and
    autotuned ~20 configs again: with packed rows, whose total length changes every microbatch, a
    training step took 23 s instead of ~1 s.
    """

    def __init__(self, kernel):
        self.kernel = kernel

    def __getitem__(self, grid):
        launch = self.kernel[grid]

        def call(*args, **kwargs):
            kwargs["NB"] = 1
            return launch(*args, **kwargs)
        return call


def use_fla_conv() -> None:
    from fla.modules.conv.triton import ops
    from transformers.models.qwen3_5 import modeling_qwen3_5

    for name in ("causal_conv1d_fwd_kernel", "causal_conv1d_bwd_kernel"):
        if not isinstance(getattr(ops, name), _FixedNB):
            setattr(ops, name, _FixedNB(getattr(ops, name)))
    modeling_qwen3_5.causal_conv1d_fn = fla_causal_conv1d


def enable_noncausal(text_model, kernel: str = "sdpa") -> None:
    """Replace the causal mask of the softmax-attention layers with a padding key mask.

    kernel="sdpa" (default) passes the mask to SDPA as d3_runtime does; kernel="varlen" runs those layers
    through varlen_noncausal_attention. Measured on the model (4 padded rows, checkpointed fwd+bwd):
    varlen is ~18% faster with BF16-level differences (logits 0.125, gradient cosine 0.9988), so the
    structure comparison keeps sdpa for every variant and varlen is left for later runs.

    Adapted from vllm-sr/d3-lite d3_runtime.py, itself adapted from
    perplexity-ai/pplx-decider-v1.1-27b (Copyright Perplexity AI, Apache License 2.0).
    """
    if text_model.config._attn_implementation not in ("sdpa", VARLEN):
        raise ValueError("noncausal attention requires SDPA")
    if kernel not in ("varlen", "sdpa"):
        raise ValueError(f"unknown kernel {kernel!r}")
    text_model.config._attn_implementation = VARLEN if kernel == "varlen" else "sdpa"

    def mask_inputs(module, args, kwargs):
        if args:
            raise ValueError("noncausal attention requires keyword inputs")
        if kwargs.get("past_key_values") is not None or kwargs.get("use_cache"):
            raise ValueError("noncausal attention does not support a KV cache")
        if kwargs.get("cu_seq_lens_q") is not None:  # packed rows: the varlen kernels take the boundaries
            if module.config._attn_implementation != VARLEN or kwargs.get("attention_mask") is not None:
                raise ValueError("packed rows need kernel='varlen' and no attention mask")
            kwargs["attention_mask"] = {"full_attention": None, "linear_attention": None}
            return args, kwargs
        embeddings = kwargs.get("inputs_embeds")
        if embeddings is None:
            embeddings = module.embed_tokens(kwargs["input_ids"])
            kwargs["inputs_embeds"], kwargs["input_ids"] = embeddings, None
        padding = kwargs.get("attention_mask")
        if padding is None:
            padding = torch.ones(embeddings.shape[:2], device=embeddings.device, dtype=torch.bool)
        if padding.ndim != 2 or padding.shape != embeddings.shape[:2] or not padding.bool().any(-1).all():
            raise ValueError("expected a 2D padding mask over a nonempty input")
        kwargs["attention_mask"] = {
            "full_attention": padding[:, None, None, :].bool(),
            "linear_attention": create_recurrent_attention_mask(config=module.config, inputs_embeds=embeddings,
                                                                attention_mask=padding),
        }
        return args, kwargs

    text_model.register_forward_pre_hook(mask_inputs, with_kwargs=True)


def enable_bidirectional_deltanet(text_model) -> None:
    """Add a right-to-left pass to every Gated DeltaNet layer through a zero-initialized gate.

    The reverse pass reuses the layer's own weights on the flipped sequence, so the only new
    parameters are one per-channel gate per layer (`backward_gate`). At gate 0 the model is
    exactly the input model, and the shared weights get no gradient from the reverse pass until
    the gate moves.
    """
    for layer in text_model.layers:
        if layer.block_type != "linear_attention":
            continue
        module = layer.linear_attn
        weight = module.out_proj.weight
        module.register_parameter("backward_gate", torch.nn.Parameter(
            torch.zeros(text_model.config.hidden_size, device=weight.device, dtype=weight.dtype)))
        original = module.forward

        def forward(hidden_states, cache_params=None, attention_mask=None, _original=original, _module=module, **kwargs):
            if cache_params is not None:
                raise ValueError("bidirectional DeltaNet does not support a cache")
            out = _original(hidden_states, attention_mask=attention_mask, **kwargs)
            flipped = attention_mask.flip(-1) if attention_mask is not None else None
            back = _original(hidden_states.flip(1), attention_mask=flipped, **kwargs).flip(1)
            return out + _module.backward_gate.to(out.dtype) * back

        module.forward = forward


def gate_state(text_model) -> dict[str, torch.Tensor]:
    return {name: p.detach().cpu().contiguous() for name, p in text_model.named_parameters() if name.endswith("backward_gate")}


def load(path: Path, *, attention: str, deltanet: str = "causal", dtype=torch.bfloat16, device: str = "cuda",
         kernel: str = "sdpa"):
    """Text backbone on `device`, vision tower on CPU (text requests only)."""
    if attention not in ATTENTION_MODES:
        raise ValueError(f"unknown attention mode {attention!r}")
    if deltanet not in DELTANET_MODES:
        raise ValueError(f"unknown DeltaNet mode {deltanet!r}")
    use_fla_conv()
    model = Qwen3_5ForConditionalGeneration.from_pretrained(path, dtype=dtype, attn_implementation="sdpa")
    model.model.language_model.to(device)
    model.lm_head.to(device)
    model.model.visual.to("cpu")
    model.config.use_cache = False
    model.config.text_config.use_cache = False
    if attention == "noncausal":
        enable_noncausal(model.model.language_model, kernel)
    if deltanet == "bidirectional":
        enable_bidirectional_deltanet(model.model.language_model)
        gates = Path(path) / GATES_NAME
        if gates.exists():
            missing, unexpected = model.model.language_model.load_state_dict(load_file(str(gates)), strict=False)
            assert not unexpected and all(not k.endswith("backward_gate") for k in missing), unexpected

    def forbidden(*args, **kwargs):
        raise AssertionError("v3 decisions never generate text")
    model.generate = forbidden
    return model


def load_readout(path: Path, token_ids: list[int], device: str = "cuda"):
    """A d3 checkpoint's own code readout ([255, hidden] in code order) as rows looked up by token id, or None.

    d3 checkpoints score codes with readout.safetensors rather than the LM head: in d3-lite the rows are
    the tied embedding rows lightly trained, and d3-flash (untied, backbone-only checkpoint) has no LM
    head at all. Our own checkpoints have no such file and read the tied LM head.
    """
    path = Path(path)
    if not (path / READOUT_NAME).exists():
        return None
    config = json.loads((path / "decision_config.json").read_text())
    if config.get("token_ids") != list(token_ids):
        raise ValueError("readout codes differ from the tokenizer's answer codes")
    weight = load_file(str(path / READOUT_NAME))["weight"].to(device=device, dtype=torch.float32)
    return {"weight": weight, "row": {t: i for i, t in enumerate(token_ids)}}


def last_positions(attention_mask: torch.Tensor) -> torch.Tensor:
    """Index of the last real token per row, for left or right padding."""
    width = attention_mask.shape[1]
    positions = torch.arange(width, device=attention_mask.device).expand_as(attention_mask)
    return torch.where(attention_mask.bool(), positions, -1).max(-1).values


def packed_hidden(text_model, input_ids, attention_mask):
    """Last-token hidden states [B, H] with the real tokens of all rows packed into one sequence.

    No compute is spent on padding: the varlen attention kernel, FLA's DeltaNet scan and FLA's short
    convolution all take the row boundaries (cu_seq_lens_q), and positions restart at 0 in every row.
    Needs right padding, noncausal attention with kernel="varlen" and causal DeltaNet layers.
    """
    if (attention_mask[:, 1:] > attention_mask[:, :-1]).any():
        raise ValueError("packing needs right padding")
    if gate_state(text_model):
        raise ValueError("the bidirectional DeltaNet pass would scan across packed rows")
    lengths = attention_mask.sum(-1)
    starts = F.pad(lengths.cumsum(0), (1, 0))
    ids = input_ids[attention_mask.bool()]
    positions = torch.arange(len(ids), device=ids.device) - torch.repeat_interleave(starts[:-1], lengths)
    longest = int(lengths.max())
    hidden = text_model(input_ids=ids[None], position_ids=positions[None], cu_seq_lens_q=starts, cu_seq_lens_k=starts,
                        max_length_q=longest, max_length_k=longest, use_cache=False).last_hidden_state
    return hidden[0, starts[1:] - 1]


def pack_rows(input_ids: list[list[int]], code_ids: list[list[int]], device) -> dict:
    """A packed batch built on the CPU and sent in one host-to-device copy.

    The GPU path (packed_hidden on padded tensors) synchronized several times per microbatch: a boolean
    index, `.any()`, `int(lengths.max())`, a data-dependent repeat_interleave and torch.unique. Here every
    shape-dependent value (boundaries, positions, last positions, the codes' union and columns) is known
    on the CPU; the boundaries also go to FLA as `cu_seqlens_cpu`, so its chunk indexing needs no sync.
    """
    lengths = [len(x) for x in input_ids]
    starts = [0]
    for n in lengths:
        starts.append(starts[-1] + n)
    union = sorted({c for row in code_ids for c in row})
    where = {c: i for i, c in enumerate(union)}
    kmax = max(len(c) for c in code_ids)
    columns = [[where[c] for c in row] + [0] * (kmax - len(row)) for row in code_ids]
    mask = [[1] * len(row) + [0] * (kmax - len(row)) for row in code_ids]
    flat = ([t for x in input_ids for t in x] + [i for n in lengths for i in range(n)] + starts
            + [s - 1 for s in starts[1:]] + union + [c for row in columns for c in row] + [m for row in mask for m in row])
    host = torch.tensor(flat, dtype=torch.long).pin_memory()
    gpu = host.to(device, non_blocking=True)
    total, b = starts[-1], len(input_ids)
    sizes = [total, total, b + 1, b, len(union), b * kmax, b * kmax]
    ids, positions, cu, last, union_t, columns_t, mask_t = torch.split(gpu, sizes)
    return {"ids": ids, "positions": positions, "cu": cu, "cu_cpu": torch.tensor(starts), "longest": max(lengths),
            "last": last, "union": union_t, "columns": columns_t.view(b, kmax), "code_mask": mask_t.view(b, kmax).bool()}


def packed_code_logits(model, batch: dict, *, readout_dtype=torch.float32) -> torch.Tensor:
    """code_logits for a pack_rows batch: [B, K], padded code columns MASK."""
    text = model.model.language_model
    hidden = text(input_ids=batch["ids"][None], position_ids=batch["positions"][None], cu_seq_lens_q=batch["cu"],
                  cu_seq_lens_k=batch["cu"], max_length_q=batch["longest"], max_length_k=batch["longest"],
                  cu_seqlens_cpu=batch["cu_cpu"], use_cache=False).last_hidden_state
    h = hidden[0, batch["last"]]
    weight = model.get_output_embeddings().weight.index_select(0, batch["union"])
    scores = F.linear(h.to(readout_dtype), weight.to(readout_dtype))
    return scores.gather(1, batch["columns"]).masked_fill(~batch["code_mask"], MASK)


def code_logits(model, input_ids, attention_mask, code_ids, code_mask, *, readout_dtype=torch.float32, readout=None,
                packed=False):
    """[B, K] logits of each row's answer codes; padded code columns are MASK.

    code_ids/code_mask: [B, K] token ids of the codes shown in each prompt and
    which columns are real. Only the needed LM-head rows are multiplied, or the
    rows of a d3 readout (load_readout) when one is given. packed=True runs the
    rows as one padding-free sequence (packed_hidden).
    """
    if packed:
        h = packed_hidden(model.model.language_model, input_ids, attention_mask)
    else:
        hidden = model.model.language_model(input_ids=input_ids, attention_mask=attention_mask,
                                            use_cache=False).last_hidden_state
        h = hidden[torch.arange(len(input_ids), device=hidden.device), last_positions(attention_mask)]
    union, columns = torch.unique(code_ids, return_inverse=True)
    if readout is None:
        weight = model.get_output_embeddings().weight.index_select(0, union)
    else:
        rows = torch.tensor([readout["row"][t] for t in union.tolist()], device=union.device)
        weight = readout["weight"].index_select(0, rows)
    scores = F.linear(h.to(readout_dtype), weight.to(readout_dtype))
    return scores.gather(1, columns).masked_fill(~code_mask, MASK)


def save(model, tokenizer, out: Path, config: dict) -> None:
    """Stock checkpoint files (loadable by plain transformers) plus the v3 config and any DeltaNet gates."""
    out.mkdir(parents=True, exist_ok=True)
    text = model.model.language_model
    gates = gate_state(text)
    stock = {k: v for k, v in model.state_dict().items() if not k.endswith("backward_gate")}
    model.save_pretrained(out, state_dict=stock, safe_serialization=True, max_shard_size="4GB")
    if gates:
        save_file(gates, str(out / GATES_NAME))
    tokenizer.save_pretrained(out)
    (out / CONFIG_NAME).write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n")


def read_config(path: Path) -> dict:
    file = Path(path) / CONFIG_NAME
    config = json.loads(file.read_text()) if file.exists() else {}
    return {"attention": "causal", "layout": "d3", "deltanet": "causal", **config}
