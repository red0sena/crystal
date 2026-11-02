"""Kit vision engine for a d3 checkpoint (vllm-sr/d3-lite): d3's own D3Engine on the board rows.

The kit's vision runner passes image_root and each row's prompt hint (meant for stock VLMs); D3Engine takes
neither. Images are read as VisionEngine reads them (decision_index.vision.images.pil, the kit's 1.6 MP cap) and
handed to d3 as PIL images; the prompt hint is dropped, since d3 renders its own prompt and reads its code readout.

    PYTHONPATH=.:external/decision-index:models/d3-lite python -c "
    from decision_index.vision.run import run
    run('qwen_decision.v3.vision_d3:D3VisionEngine', {'model': 'models/d3-lite'}, 'data/vision-work/vision',
        'runs/vision/d3-lite', rows_path='data/vision-work/vision/rows-built.jsonl')"
"""

from __future__ import annotations

from d3_engine import D3Engine
from decision_index.vision import images as img

MAX_PIXELS = 1_638_400


class D3VisionEngine(D3Engine):
    def __init__(self, image_root: str | None = None, max_pixels: int = MAX_PIXELS, **options):
        super().__init__(**options)
        self.image_root, self.max_pixels = image_root, int(max_pixels)
        if isinstance(self.provenance, dict):
            self.provenance = {**self.provenance, "max_pixels": self.max_pixels,
                               "image_resize": "images over max_pixels are downscaled by the kit's images.pil "
                                               "(as VisionEngine), then passed to d3 as PIL images",
                               "prompt_hint": "ignored"}

    def __call__(self, state, questions, images=(), prompt=None, videos=None):
        pictures = [img.pil(p, self.image_root, self.max_pixels).convert("RGB") for p in images or []]
        return super().__call__(state, questions, images=pictures or None, videos=videos)
