"""real_models.py -- the GPU-box implementation of the `models` interface (CLIP + YOLOv8)."""
from __future__ import annotations

from clip_encoder import ClipEncoder
from yolo_tools import YoloProposer, YoloWorldProposer


class RealModels:
    def __init__(self, clip_pretrained: str = "openai", yolo_weights: str = None, device: str = None, clip_model: str = "ViT-B-16",
                 conf: float = 0.01, iou: float = 0.7, max_det: int = 100, fp16: bool = True,
                 proposer: str = "yolo", world_weights: str = "yolov8s-worldv2.pt", world_prompts=("object",)):
        self.clip = ClipEncoder(pretrained=clip_pretrained, model_name=clip_model, device=device, fp16=fp16)
        self.dim = self.clip.dim
        if proposer == "yoloworld":
            self.proposer = YoloWorldProposer(world_weights, prompts=world_prompts, conf=conf, iou=iou, max_det=max_det,
                                              device=device)
        elif proposer == "yolo":
            self.proposer = (YoloProposer(yolo_weights, conf=conf, iou=iou, max_det=max_det, device=device)
                             if yolo_weights else None)
        else:
            raise ValueError(f"proposer must be 'yolo' or 'yoloworld', got {proposer!r}")

    def encode_text(self, texts):
        return self.clip.encode_text(list(texts))

    def encode_images_u8(self, crops):
        return self.clip.encode_images_u8(crops)

    def propose(self, img):
        if self.proposer is None:
            raise RuntimeError("no YOLO weights given (--yolo_weights); proposals unavailable")
        return self.proposer(img)
