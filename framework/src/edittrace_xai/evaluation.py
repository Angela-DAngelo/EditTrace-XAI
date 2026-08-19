from __future__ import annotations

import hashlib
import json
import math
import statistics
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
from PIL import Image, ImageChops, ImageStat

from .modeling import Predictor, TorchvisionPredictor
from .schema import ActionRecord, GeneratorOutcome, OperationName, TraceRecord


FROZEN_TORCHVISION_WEIGHTS = {
    "resnet50": "IMAGENET1K_V2",
    "vit_b_16": "IMAGENET1K_V1",
}


def build_frozen_predictor(model_name: str, device: str) -> TorchvisionPredictor:
    try:
        weights_name = FROZEN_TORCHVISION_WEIGHTS[model_name]
    except KeyError as exc:
        raise ValueError(f"unsupported frozen model: {model_name}") from exc
    return TorchvisionPredictor(model_name, weights_name=weights_name, device=device)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON in {path}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"expected object in {path}:{line_number}")
            rows.append(value)
    return rows


def write_jsonl(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True) + "\n")


def write_json(path: str | Path, value: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(value), indent=2, sort_keys=True), encoding="utf-8")


def load_traces(path: str | Path, *, accepted_only: bool = False) -> list[TraceRecord]:
    traces: list[TraceRecord] = []
    for row in read_jsonl(path):
        if "accepted" in row:
            outcome = GeneratorOutcome.model_validate(row)
            if accepted_only and not outcome.accepted:
                continue
            if outcome.trace is not None:
                traces.append(outcome.trace)
        else:
            traces.append(TraceRecord.model_validate(row))
    return traces


def resolve_source_path(trace: TraceRecord, image_root: str | Path | None = None) -> Path:
    candidates: list[Path] = []
    if image_root is not None:
        candidates.append(Path(image_root).expanduser() / trace.source_id)
    if trace.source_path:
        candidates.append(Path(trace.source_path).expanduser())
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    rendered = ", ".join(str(path) for path in candidates) or "<none>"
    raise FileNotFoundError(
        f"source image for {trace.trace_id} was not found; checked: {rendered}. "
        "Pass --image-root to remap the stored source paths."
    )


def load_source(trace: TraceRecord, image_root: str | Path | None = None) -> Image.Image:
    path = resolve_source_path(trace, image_root)
    with Image.open(path) as image:
        return image.convert("RGB").copy()


def stable_seed(*parts: object) -> int:
    digest = hashlib.sha256("|".join(map(str, parts)).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") & 0x7FFF_FFFF


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rank_ids(scores: Mapping[str, float], order: Mapping[str, int]) -> list[str]:
    return sorted(scores, key=lambda key: (-float(scores[key]), int(order[key]), key))


def reciprocal_rank(ranking: Sequence[str], target: str) -> float:
    try:
        return 1.0 / (ranking.index(target) + 1)
    except ValueError:
        return 0.0


def pair_key(first: str, second: str, action_order: Mapping[str, int]) -> str:
    ordered = sorted((first, second), key=lambda item: (action_order[item], item))
    return f"{ordered[0]}||{ordered[1]}"


def percentile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise ValueError("cannot compute a percentile of an empty sequence")
    if not 0 <= probability <= 1:
        raise ValueError("probability must be in [0,1]")
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def descriptive_stats(values: Sequence[float]) -> dict[str, Any]:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return {"n": 0, "available": False}
    return {
        "n": len(finite),
        "available": True,
        "mean": statistics.fmean(finite),
        "median": statistics.median(finite),
        "p05": percentile(finite, 0.05),
        "p95": percentile(finite, 0.95),
        "min": min(finite),
        "max": max(finite),
    }


def clustered_bootstrap_mean(
    rows: Sequence[Mapping[str, Any]],
    value_key: str,
    *,
    cluster_key: str = "source_id",
    resamples: int = 10_000,
    seed: int = 20260802,
) -> dict[str, Any]:
    if resamples < 1:
        raise ValueError("resamples must be positive")
    clusters: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        value = row.get(value_key)
        cluster = row.get(cluster_key)
        if cluster is None or value is None:
            continue
        value = float(value)
        if math.isfinite(value):
            clusters[str(cluster)].append(value)
    if not clusters:
        return {"n_clusters": 0, "available": False}
    cluster_ids = sorted(clusters)
    cluster_means = np.asarray(
        [statistics.fmean(clusters[cluster]) for cluster in cluster_ids], dtype=np.float64
    )
    rng = np.random.default_rng(seed)
    samples = np.empty(resamples, dtype=np.float64)
    for index in range(resamples):
        selected = rng.integers(0, len(cluster_means), size=len(cluster_means))
        samples[index] = float(cluster_means[selected].mean())
    return {
        "available": True,
        "n_clusters": len(cluster_ids),
        "estimate": float(cluster_means.mean()),
        "ci95_low": float(np.quantile(samples, 0.025)),
        "ci95_high": float(np.quantile(samples, 0.975)),
        "resamples": resamples,
        "seed": seed,
    }


def paired_clustered_bootstrap_difference(
    rows: Sequence[Mapping[str, Any]],
    method: str,
    baseline: str,
    value_key: str,
    *,
    cluster_key: str = "source_id",
    method_key: str = "method",
    resamples: int = 10_000,
    seed: int = 20260802,
) -> dict[str, Any]:
    by_cluster: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        name = str(row.get(method_key, ""))
        if name not in {method, baseline} or row.get(value_key) is None:
            continue
        by_cluster[str(row[cluster_key])][name].append(float(row[value_key]))
    differences: list[float] = []
    for cluster in sorted(by_cluster):
        values = by_cluster[cluster]
        if method in values and baseline in values:
            differences.append(statistics.fmean(values[method]) - statistics.fmean(values[baseline]))
    if not differences:
        return {"available": False, "n_clusters": 0, "method": method, "baseline": baseline}
    array = np.asarray(differences, dtype=np.float64)
    rng = np.random.default_rng(seed)
    samples = np.empty(resamples, dtype=np.float64)
    for index in range(resamples):
        selected = rng.integers(0, len(array), size=len(array))
        samples[index] = float(array[selected].mean())
    return {
        "available": True,
        "method": method,
        "baseline": baseline,
        "n_clusters": len(array),
        "estimate": float(array.mean()),
        "ci95_low": float(np.quantile(samples, 0.025)),
        "ci95_high": float(np.quantile(samples, 0.975)),
        "resamples": resamples,
        "seed": seed,
    }


@dataclass
class TimedPredictor:
    predictor: Predictor
    synchronize: Callable[[], None] | None = None
    calls: int = 0
    elapsed_ms: float = 0.0
    durations_ms: list[float] = field(default_factory=list)

    @property
    def model_id(self) -> str:
        return self.predictor.model_id

    def logits(self, image: Image.Image) -> np.ndarray:
        if self.synchronize:
            self.synchronize()
        start = time.perf_counter_ns()
        result = self.predictor.logits(image)
        if self.synchronize:
            self.synchronize()
        duration_ms = (time.perf_counter_ns() - start) / 1_000_000
        self.calls += 1
        self.elapsed_ms += duration_ms
        self.durations_ms.append(duration_ms)
        return result

    def snapshot(self) -> tuple[int, float]:
        return self.calls, self.elapsed_ms

    def delta(self, snapshot: tuple[int, float]) -> dict[str, float | int]:
        calls, elapsed = snapshot
        return {"model_calls": self.calls - calls, "model_inference_ms": self.elapsed_ms - elapsed}


def synchronize_for_predictor(predictor: Predictor) -> Callable[[], None] | None:
    torch_module = getattr(predictor, "_torch", None)
    device = str(getattr(predictor, "device", "cpu"))
    if torch_module is not None and device.startswith("cuda"):
        return torch_module.cuda.synchronize
    return None


def warmup_predictor(predictor: Predictor, image: Image.Image, iterations: int) -> None:
    synchronize = synchronize_for_predictor(predictor)
    for _ in range(iterations):
        predictor.logits(image)
    if synchronize:
        synchronize()


def normalized_parameter_magnitude(action: ActionRecord, config: Mapping[str, Any]) -> float:
    operations = config["operations"]
    op = action.op_name
    if op in {OperationName.BRIGHTNESS, OperationName.CONTRAST, OperationName.SATURATION}:
        values = [float(value) for value in operations[op.value]["factors"]]
        denominator = max(abs(value - 1.0) for value in values)
        return abs(float(action.params["factor"]) - 1.0) / denominator
    if op == OperationName.GAUSSIAN_BLUR:
        maximum = max(float(value) for value in operations[op.value]["sigmas"])
        return float(action.params["sigma"]) / maximum
    if op == OperationName.CROP_RESIZE:
        minimum_area = min(float(value) ** 2 for value in operations[op.value]["side_fractions"])
        left, top, right, bottom = map(float, action.params["box"])
        area = (right - left) * (bottom - top)
        return (1.0 - area) / (1.0 - minimum_area)
    if op == OperationName.JPEG:
        minimum_quality = min(int(value) for value in operations[op.value]["qualities"])
        return (100.0 - float(action.params["quality"])) / (100.0 - minimum_quality)
    raise ValueError(f"unsupported operation: {op}")


def normalized_l1(first: Image.Image, second: Image.Image) -> float:
    if first.size != second.size:
        raise ValueError("images must have the same size")
    difference = ImageChops.difference(first.convert("RGB"), second.convert("RGB"))
    means = ImageStat.Stat(difference).mean
    return float(sum(means) / (len(means) * 255.0))


class PerceptualMetric:
    def __init__(self, name: str, device: str = "cpu"):
        self.name = name
        self.device = device
        self.elapsed_ms = 0.0
        self.calls = 0
        self._metric: Any = None
        if name == "lpips":
            try:
                import lpips
                import torch
            except ImportError as exc:
                raise RuntimeError(
                    "LPIPS is required for paper evaluation; install edittrace-xai[evaluation]"
                ) from exc
            self._torch = torch
            self._metric = lpips.LPIPS(net="alex").eval().to(device)
        elif name == "l1":
            self._torch = None
        else:
            raise ValueError("perceptual metric must be 'lpips' or 'l1'")

    def __call__(self, first: Image.Image, second: Image.Image) -> float:
        start = time.perf_counter_ns()
        if self.name == "l1":
            value = normalized_l1(first, second)
        else:
            arrays = []
            for image in (first, second):
                array = np.asarray(image.convert("RGB"), dtype=np.float32) / 127.5 - 1.0
                arrays.append(self._torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(self.device))
            if self.device.startswith("cuda"):
                self._torch.cuda.synchronize()
            with self._torch.inference_mode():
                value = float(self._metric(arrays[0], arrays[1]).item())
            if self.device.startswith("cuda"):
                self._torch.cuda.synchronize()
        self.calls += 1
        self.elapsed_ms += (time.perf_counter_ns() - start) / 1_000_000
        return value


def structural_similarity(first: Image.Image, second: Image.Image) -> float | None:
    try:
        from skimage.metrics import structural_similarity
    except ImportError:
        return None
    first_array = np.asarray(first.convert("RGB"), dtype=np.uint8)
    second_array = np.asarray(second.convert("RGB"), dtype=np.uint8)
    return float(structural_similarity(first_array, second_array, channel_axis=2, data_range=255))


def average_precision(labels: Sequence[int], scores: Sequence[float]) -> float | None:
    if len(labels) != len(scores):
        raise ValueError("labels and scores must have equal length")
    positives = sum(int(label) for label in labels)
    if positives == 0:
        return None
    order = sorted(range(len(scores)), key=lambda index: (-float(scores[index]), index))
    hits = 0
    precision_sum = 0.0
    for rank, index in enumerate(order, 1):
        if int(labels[index]) == 1:
            hits += 1
            precision_sum += hits / rank
    return precision_sum / positives
