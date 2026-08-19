from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from PIL import Image


class Predictor(Protocol):
    model_id: str

    def logits(self, image: Image.Image) -> np.ndarray:
        """Return a one-dimensional logit vector."""


def target_margin(logits: np.ndarray, target: int) -> float:
    logits = np.asarray(logits, dtype=np.float64)
    if logits.ndim != 1 or not 0 <= target < logits.shape[0]:
        raise ValueError("invalid logits or target")
    competitor = np.max(np.delete(logits, target))
    return float(logits[target] - competitor)


def predicted_label(logits: np.ndarray) -> int:
    logits = np.asarray(logits)
    if logits.ndim != 1:
        raise ValueError("logits must be one-dimensional")
    return int(np.argmax(logits))


@dataclass
class TorchvisionPredictor:
    """Lazy optional adapter; importing this class does not require torch."""

    model_name: str
    weights_name: str = "DEFAULT"
    device: str = "cpu"

    def __post_init__(self) -> None:
        try:
            import torch
            import torchvision.models as models
        except ImportError as exc:
            raise RuntimeError("install edittrace-xai[torch] to use TorchvisionPredictor") from exc
        if self.model_name not in {"resnet50", "vit_b_16"}:
            raise ValueError("supported models: resnet50, vit_b_16")
        constructor = getattr(models, self.model_name)
        weights_enum = models.get_model_weights(constructor)
        weights = weights_enum.DEFAULT if self.weights_name == "DEFAULT" else weights_enum[self.weights_name]
        self._torch = torch
        self._weights = weights
        self._preprocess = weights.transforms()
        self._model = constructor(weights=weights).eval().to(self.device)
        self.model_id = f"torchvision:{self.model_name}:{weights.name}"

    def logits(self, image: Image.Image) -> np.ndarray:
        tensor = self._preprocess(image.convert("RGB")).unsqueeze(0).to(self.device)
        with self._torch.inference_mode():
            output = self._model(tensor)[0].detach().cpu().numpy()
        return output

