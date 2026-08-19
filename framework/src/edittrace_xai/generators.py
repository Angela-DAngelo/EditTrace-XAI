from __future__ import annotations

import hashlib
import itertools
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import yaml
from PIL import Image, __version__ as pillow_version

from .engine import ReplayEngine, ReplayError, image_hash, image_spec
from .modeling import Predictor, target_margin
from .operations import apply_operation, identity_params
from .schema import (
    ActionRecord,
    CodecRecord,
    GeneratorOutcome,
    GeometryRecord,
    NormalizedBox,
    OperationName,
    SupportRecord,
    TraceRecord,
)


ANCHORS = {"center", "top_left", "top_right", "bottom_left", "bottom_right"}


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if config.get("schema_version") != "1.0":
        raise ValueError("unsupported configuration schema_version")
    return config


def derive_seed(*parts: object) -> int:
    digest = hashlib.sha256("|".join(map(str, parts)).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") & 0x7FFF_FFFF


def crop_box(side_fraction: float, anchor: str) -> list[float]:
    if not 0 < side_fraction <= 1:
        raise ValueError("side_fraction must be in (0,1]")
    if anchor not in ANCHORS:
        raise ValueError(f"unsupported anchor: {anchor}")
    gap = 1.0 - side_fraction
    offsets = {
        "center": (gap / 2, gap / 2),
        "top_left": (0.0, 0.0),
        "top_right": (gap, 0.0),
        "bottom_left": (0.0, gap),
        "bottom_right": (gap, gap),
    }
    left, top = offsets[anchor]
    return [left, top, left + side_fraction, top + side_fraction]


@dataclass
class TraceGenerator:
    config: Mapping[str, Any]

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TraceGenerator":
        return cls(load_config(path))

    def sample_params(self, op_name: OperationName, rng: np.random.Generator) -> dict[str, Any]:
        operations = self.config["operations"]
        if op_name in {OperationName.BRIGHTNESS, OperationName.CONTRAST, OperationName.SATURATION}:
            factor = float(rng.choice(operations[op_name.value]["factors"]))
            return {"factor": factor}
        if op_name == OperationName.GAUSSIAN_BLUR:
            sigma = float(rng.choice(operations[op_name.value]["sigmas"]))
            return {"sigma": sigma, "bypass": False}
        if op_name == OperationName.CROP_RESIZE:
            cfg = operations[op_name.value]
            side = float(rng.choice(cfg["side_fractions"]))
            anchor = str(rng.choice(cfg["anchors"]))
            return {"box": crop_box(side, anchor), "interpolation": str(cfg["interpolation"])}
        if op_name == OperationName.JPEG:
            cfg = operations[op_name.value]
            return {
                "quality": int(rng.choice(cfg["qualities"])),
                "subsampling": int(cfg["subsampling"]),
                "optimize": bool(cfg["optimize"]),
                "progressive": bool(cfg["progressive"]),
                "bypass": False,
            }
        raise ValueError(f"unsupported operation: {op_name}")

    def _action_metadata(
        self,
        op_name: OperationName,
        params: Mapping[str, Any],
    ) -> tuple[GeometryRecord, SupportRecord, CodecRecord | None]:
        if op_name == OperationName.CROP_RESIZE:
            box = NormalizedBox(
                left=params["box"][0], top=params["box"][1],
                right=params["box"][2], bottom=params["box"][3],
            )
            return (
                GeometryRecord(kind="crop_resize", crop_box=box, interpolation=params["interpolation"]),
                SupportRecord(kind="box", box=box),
                None,
            )
        if op_name == OperationName.JPEG:
            codec = CodecRecord(
                library_version=pillow_version,
                quality=int(params["quality"]),
                subsampling=int(params["subsampling"]),
                optimize=bool(params["optimize"]),
                progressive=bool(params["progressive"]),
                bypass=bool(params.get("bypass", False)),
            )
            return GeometryRecord(), SupportRecord(kind="global"), codec
        return GeometryRecord(), SupportRecord(kind="global"), None

    def build_trace(
        self,
        source: Image.Image,
        *,
        source_id: str,
        source_path: str | None,
        ground_truth_label: int | None,
        benchmark: str,
        seed: int,
        length: int,
        model_id: str | None = None,
        designated_index: int | None = None,
        designated_pair_indices: tuple[int, int] | None = None,
    ) -> TraceRecord:
        if length not in self.config["history_lengths"]:
            raise ValueError("history length is outside the frozen configuration")
        rng = np.random.default_rng(seed)
        available = list(OperationName)
        sampled = list(rng.choice(available, size=length, replace=False))
        trace_id = f"{benchmark.lower()}-{source_id}-{seed:08x}"
        current = source.convert("RGB")
        actions: list[ActionRecord] = []
        for index, op_name in enumerate(sampled):
            op_name = OperationName(op_name)
            params = self.sample_params(op_name, rng)
            predecessor_hash = image_hash(current)
            input_spec = image_spec(current)
            output = apply_operation(current, op_name, params)
            current = output.image
            geometry, support, codec = self._action_metadata(op_name, params)
            action = ActionRecord(
                trace_id=trace_id,
                action_id=f"{trace_id}:a{index}",
                index=index,
                op_name=op_name,
                params=params,
                identity_params=identity_params(op_name, params),
                input_spec=input_spec,
                output_spec=image_spec(current),
                geometry=geometry,
                support=support,
                codec=codec,
                predecessor_hash=predecessor_hash,
                output_hash=image_hash(current),
                seed=derive_seed(seed, index, op_name.value),
            )
            actions.append(action)
        designated_action_id = actions[designated_index].action_id if designated_index is not None else None
        designated_pair = None
        if designated_pair_indices is not None:
            designated_pair = (
                actions[designated_pair_indices[0]].action_id,
                actions[designated_pair_indices[1]].action_id,
            )
        return TraceRecord(
            trace_id=trace_id,
            source_id=source_id,
            source_path=source_path,
            ground_truth_label=ground_truth_label,
            model_id=model_id,
            benchmark=benchmark,
            seed=seed,
            source_spec=image_spec(source),
            source_hash=image_hash(source),
            final_hash=image_hash(current),
            actions=actions,
            designated_action_id=designated_action_id,
            designated_pair=designated_pair,
        )

    def natural(
        self,
        source: Image.Image,
        *,
        source_id: str,
        source_path: str | None = None,
        ground_truth_label: int | None = None,
        model_id: str | None = None,
        ordinal: int = 0,
    ) -> TraceRecord:
        seed = derive_seed(self.config["global_seed"], "B-Natural", source_id, model_id, ordinal)
        lengths = self.config["history_lengths"]
        length = int(lengths[ordinal % len(lengths)])
        return self.build_trace(
            source,
            source_id=source_id,
            source_path=source_path,
            ground_truth_label=ground_truth_label,
            benchmark="B-Natural",
            seed=seed,
            length=length,
            model_id=model_id,
        )


def trace_margins(
    source: Image.Image,
    trace: TraceRecord,
    predictor: Predictor,
    target: int,
    engine: ReplayEngine | None = None,
) -> dict[str, float]:
    engine = engine or ReplayEngine()
    final = engine.replay(source, trace)
    margins = {"source": target_margin(predictor.logits(source), target),
               "final": target_margin(predictor.logits(final.image), target)}
    for action in trace.actions:
        try:
            replay = engine.remove_action(source, trace, action.action_id)
            margins[f"remove:{action.action_id}"] = target_margin(predictor.logits(replay.image), target)
        except ReplayError:
            continue
    return margins


def validate_single_trigger(
    source: Image.Image,
    trace: TraceRecord,
    predictor: Predictor,
    target: int,
    engine: ReplayEngine | None = None,
) -> tuple[bool, dict[str, float], str]:
    if trace.designated_action_id is None:
        raise ValueError("B-Single trace requires designated_action_id")
    margins = trace_margins(source, trace, predictor, target, engine)
    if margins["source"] <= 0:
        return False, margins, "source_not_target"
    if margins["final"] > 0:
        return False, margins, "final_not_shifted"
    designated = margins.get(f"remove:{trace.designated_action_id}")
    if designated is None or designated <= 0:
        return False, margins, "designated_not_restoring"
    other_success = any(
        value > 0
        for key, value in margins.items()
        if key.startswith("remove:") and key != f"remove:{trace.designated_action_id}"
    )
    if other_success:
        return False, margins, "non_designated_restores"
    return True, margins, "accepted"


def validate_interaction(
    source: Image.Image,
    trace: TraceRecord,
    predictor: Predictor,
    target: int,
    engine: ReplayEngine | None = None,
) -> tuple[bool, dict[str, float], str]:
    if trace.designated_pair is None:
        raise ValueError("B-Interaction trace requires designated_pair")
    engine = engine or ReplayEngine()
    margins = trace_margins(source, trace, predictor, target, engine)
    if margins["source"] <= 0:
        return False, margins, "source_not_target"
    if margins["final"] > 0:
        return False, margins, "final_not_shifted"
    first, second = trace.designated_pair
    if margins.get(f"remove:{first}", float("inf")) > 0:
        return False, margins, "first_single_restores"
    if margins.get(f"remove:{second}", float("inf")) > 0:
        return False, margins, "second_single_restores"
    try:
        joint = engine.remove_pair(source, trace, first, second)
    except ReplayError:
        return False, margins, "joint_invalid"
    joint_margin = target_margin(predictor.logits(joint.image), target)
    margins[f"remove_pair:{first},{second}"] = joint_margin
    if joint_margin <= 0:
        return False, margins, "joint_not_restoring"
    return True, margins, "accepted"


def generate_single(
    generator: TraceGenerator,
    source: Image.Image,
    *,
    source_id: str,
    target: int,
    predictor: Predictor,
    source_path: str | None = None,
    max_attempts: int | None = None,
) -> GeneratorOutcome:
    max_attempts = max_attempts or int(generator.config["generators"]["single_max_attempts"])
    rejections: Counter[str] = Counter()
    for attempt in range(1, max_attempts + 1):
        seed = derive_seed(generator.config["global_seed"], "B-Single", source_id, predictor.model_id, attempt)
        length = int(generator.config["history_lengths"][(attempt - 1) % 4])
        rng = np.random.default_rng(seed)
        designated = int(rng.integers(0, length))
        trace = generator.build_trace(
            source,
            source_id=source_id,
            source_path=source_path,
            ground_truth_label=target,
            benchmark="B-Single",
            seed=seed,
            length=length,
            model_id=predictor.model_id,
            designated_index=designated,
        )
        accepted, margins, reason = validate_single_trigger(source, trace, predictor, target)
        if accepted:
            return GeneratorOutcome(
                benchmark="B-Single", accepted=True, attempts=attempt,
                trace=trace, rejection_counts=dict(rejections), metrics=margins,
            )
        rejections[reason] += 1
    return GeneratorOutcome(
        benchmark="B-Single", accepted=False, attempts=max_attempts,
        rejection_counts=dict(rejections),
    )


def generate_interaction(
    generator: TraceGenerator,
    source: Image.Image,
    *,
    source_id: str,
    target: int,
    predictor: Predictor,
    source_path: str | None = None,
    max_attempts: int | None = None,
) -> GeneratorOutcome:
    max_attempts = max_attempts or int(generator.config["generators"]["interaction_max_attempts"])
    rejections: Counter[str] = Counter()
    for attempt in range(1, max_attempts + 1):
        seed = derive_seed(generator.config["global_seed"], "B-Interaction", source_id, predictor.model_id, attempt)
        length = max(3, int(generator.config["history_lengths"][(attempt - 1) % 4]))
        rng = np.random.default_rng(seed)
        pair = tuple(sorted(rng.choice(length, size=2, replace=False).tolist()))
        trace = generator.build_trace(
            source,
            source_id=source_id,
            source_path=source_path,
            ground_truth_label=target,
            benchmark="B-Interaction",
            seed=seed,
            length=length,
            model_id=predictor.model_id,
            designated_pair_indices=pair,
        )
        accepted, margins, reason = validate_interaction(source, trace, predictor, target)
        if accepted:
            return GeneratorOutcome(
                benchmark="B-Interaction", accepted=True, attempts=attempt,
                trace=trace, rejection_counts=dict(rejections), metrics=margins,
            )
        rejections[reason] += 1
    return GeneratorOutcome(
        benchmark="B-Interaction", accepted=False, attempts=max_attempts,
        rejection_counts=dict(rejections),
    )


def oracle_subset(traces: Iterable[TraceRecord], count: int = 500) -> list[TraceRecord]:
    ordered = sorted(traces, key=lambda item: hashlib.sha256(item.trace_id.encode()).hexdigest())
    selected = ordered[:count]
    return [trace.model_copy(update={"benchmark": "B-Oracle"}) for trace in selected]


def write_jsonl(path: str | Path, records: Iterable[Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            if hasattr(record, "model_dump"):
                payload = record.model_dump(mode="json")
            else:
                payload = record
            handle.write(json.dumps(payload, sort_keys=True) + "\n")


def read_trace_jsonl(path: str | Path) -> list[TraceRecord]:
    traces: list[TraceRecord] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                traces.append(TraceRecord.model_validate(json.loads(line)))
    return traces
