"""Native author inference adapters for frozen leader comparisons, never training teachers."""
from pathlib import Path
import json

import torch
from decision_index.engines import Engine, Unsupported


def as_text(value):
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


def gliclass_state(state):
    # Exactly the GLiClass model card's SystemOne render_state convention.
    if state in (None, "", {}, []): return ""
    if isinstance(state, str): return state
    if isinstance(state, list): return "\n".join(f"- {as_text(item)}" for item in state)
    lines = []
    for key,value in state.items():
        if isinstance(value, list) and all(isinstance(item,str) for item in value):
            lines += [f"{key}:"] + [f"- {item}" for item in value]
        else: lines.append(f"{key}: {as_text(value)}")
    return "\n".join(lines)


def gliclass_options(question):
    if question["type"] == "choice":
        criteria = question["criteria"]
        return list(criteria), [as_text(v) if v is not None else k for k,v in criteria.items()]
    if question["type"] == "noul":
        criteria = question.get("criteria") or {}
        return ["true","false"], [as_text(criteria.get("true") or "yes"),as_text(criteria.get("false") or "no")]
    raise Unsupported(f"unsupported question type {question['type']!r}")


class GLiClassNative(Engine):
    """Model-card SystemOne mapping, calibrated native logits, no prompt truncation."""
    name = "gliclass-native"

    def __init__(self, model="models/leader-gliclass-v4-edge", device="cuda", **options):
        super().__init__(**options)
        from gliclass import GLiClassModel,ZeroShotClassificationPipeline
        self.m = GLiClassModel.from_pretrained(model,torch_dtype=torch.bfloat16,local_files_only=True).to(device).eval()
        self.pipe = ZeroShotClassificationPipeline(self.m,model,max_length=10**9,
                      classification_type="single-label",device=device,progress_bar=False).pipe
        self.provenance = {"kind":"native GLiClass model-card SystemOne adapter", "model":model,
                           "max_tokens":16384,"calibrator":True,"truncation":False}

    @torch.inference_mode()
    def __call__(self,state,questions,**unused):
        answers = {}; tokens = 0
        for name,q in questions.items():
            keys,labels = gliclass_options(q)
            prompt = as_text(q.get("instructions") or "") or None
            inputs = self.pipe.prepare_inputs([gliclass_state(state)],labels,same_labels=True,prompt=prompt)
            n = inputs["input_ids"].shape[-1]
            if n > 16384: raise Unsupported(f"GLiClass input has {n} tokens; limit 16384; no truncation")
            logits = self.m(**inputs,max_num_classes=len(labels)).logits[0,:len(labels)].float()
            assert logits.shape == (len(labels),) and torch.isfinite(logits).all()
            probabilities = dict(zip(keys,logits.softmax(-1).tolist()))
            if q["type"] == "choice":
                answers[name] = {"type":"choice","choice":max(keys,key=probabilities.get),"probabilities":probabilities}
            else: answers[name] = {"type":"noul","noul":probabilities["true"]}
            tokens += n
        return {"model":self.name,"answers":answers,"usage":{"input_tokens":tokens}},None


def qev_vision(model="models/leader-qev-0.8b", cache="runs/champion-v1/leader-compare-v1/hub-cache"):
    """Use the author's unchanged kit engine call, with a verified offline cache in its loader."""
    from qev_decision_index import QEVEngine
    from veyra.option_model import OptionModel
    # The released adapter does not forward cache_dir. Initialize its documented members explicitly;
    # __call__, preprocessing, limit handling, probability conversion and pixel accounting remain native.
    engine = QEVEngine.__new__(QEVEngine)
    Engine.__init__(engine)
    engine.root = None
    engine.m = OptionModel.load(Path(model),device="cuda",mode="accurate",cache_dir=cache,local_files_only=True)
    engine.m.eval();engine.model = model
    engine.provenance = {"kind":"author qev_decision_index.QEVEngine", "model":model,"mode":"accurate",
                         "max_pixels":1638400,"max_tokens":8192,"offline_pinned_backbone":True}
    return engine
