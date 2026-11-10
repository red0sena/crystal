"""Public-suite adapters. Native decision policies; no evaluation-dependent tuning."""
from pathlib import Path
import time

import torch
from transformers import AutoTokenizer
from decision_index.engines import Engine
from qwen_decision.v3 import format as fmt, modeling
from qwen_decision.v3.engine import Scorer, V3Engine
from qwen_decision.v3.leader_engines import GLiClassNative, qev_vision


class PackedScorer(Scorer):
    """Identical prompts/readout with independent sequences in the varlen kernel."""
    def __init__(self, model, device="cuda:0", batch_tokens=8192, max_batch=8):
        config = modeling.read_config(Path(model))
        self.attention, self.deltanet, self.layout = (config[k] for k in ("attention", "deltanet", "layout"))
        if (self.attention, self.deltanet) != ("noncausal", "causal"):
            raise ValueError("packing requires noncausal attention and causal DeltaNet")
        if (Path(model)/"decision_architecture.json").exists():
            raise ValueError("architecture overrides require their native scorer")
        self.device = device
        self.model = modeling.load(Path(model), attention=self.attention, deltanet=self.deltanet,
                                   dtype=torch.bfloat16, device=device, kernel="varlen").eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
        self.codes, self.code_ids = fmt.answer_codes(self.tokenizer)
        if not self.model.config.get_text_config().tie_word_embeddings or (Path(model)/modeling.READOUT_NAME).exists():
            raise ValueError("this packed scorer is for the pilot's tied embedding readout")
        self.readout = self.decision_tail_mode = None
        self.batch_tokens, self.max_batch, self.orders = int(batch_tokens), int(max_batch), 1

    @torch.inference_mode()
    def forward(self, batch):
        packed = modeling.pack_rows([item[1] for item in batch], [item[2] for item in batch], self.device)
        logits = modeling.packed_code_logits(self.model, packed)
        return [row[:len(item[2])].softmax(-1).tolist() for row,item in zip(logits,batch)]


class JointText(V3Engine):
    def __init__(self, model, device="cuda:0", batch_tokens=8192, max_batch=8):
        super().__init__(model=model,device=device,batch_tokens=batch_tokens,max_batch=max_batch,orders=1)
        self.provenance.update(kernel="native padded sdpa",optimization="packed path rejected after 23/24 native decision parity")


class GLiClassText(GLiClassNative):
    def synchronize(self):
        torch.cuda.synchronize()

    def runtime(self):
        return {"torch":torch.__version__, "gpu":torch.cuda.get_device_name()}


def QEVVision(image_root=None, **options):
    engine = qev_vision(**options)
    engine.root = Path(image_root) if image_root else None
    return engine
