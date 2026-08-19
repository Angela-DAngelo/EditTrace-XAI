from __future__ import annotations

import json
import os
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

from PIL import Image

from .core import Action, Session, apply_action, create_session, image_hash


OP_MAP = {
    "brightness": "brightness",
    "contrast": "contrast",
    "saturation": "saturation",
    "gaussian_blur": "blur",
    "crop_resize": "crop",
    "jpeg": "jpeg",
}


class TraceCatalog:
    def __init__(self, images_root: Path, traces_path: Path):
        self.images_root = images_root.resolve()
        self.traces_path = traces_path.resolve()
        self.records: list[dict[str, Any]] = []
        self.by_id: dict[str, dict[str, Any]] = {}
        self.errors: list[str] = []
        self.load()

    @classmethod
    def discover(cls, project_root: Path) -> "TraceCatalog | None":
        data_root = Path(os.getenv("EDITTRACE_DATA_ROOT", project_root / "data"))
        images = Path(os.getenv("EDITTRACE_IMAGES", data_root / "images"))
        explicit = os.getenv("EDITTRACE_TRACES")
        if explicit:
            traces = Path(explicit)
        else:
            candidates = sorted(data_root.glob("*.jsonl")) if data_root.is_dir() else []
            traces = candidates[0] if candidates else data_root / "traces.jsonl"
        if not traces.is_file():
            return None
        return cls(images, traces)

    def load(self) -> None:
        self.records = []
        self.by_id = {}
        self.errors = []
        with self.traces_path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                    self._validate_record(record)
                    if record["trace_id"] in self.by_id:
                        raise ValueError("duplicate trace_id")
                    self.records.append(record)
                    self.by_id[record["trace_id"]] = record
                except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                    self.errors.append(f"line {line_number}: {exc}")

    def _validate_record(self, record: dict[str, Any]) -> None:
        for key in ("trace_id", "source_id", "ground_truth_label", "benchmark", "actions"):
            if key not in record:
                raise ValueError(f"missing {key}")
        if not 0 <= int(record["ground_truth_label"]) <= 999:
            raise ValueError("ground_truth_label outside [0, 999]")
        if not 3 <= len(record["actions"]) <= 6:
            raise ValueError("trace length outside [3, 6]")
        operations = [action.get("op_name") for action in record["actions"]]
        if any(operation not in OP_MAP for operation in operations):
            raise ValueError("unsupported operation")
        if len(set(operations)) != len(operations):
            raise ValueError("operation types are not unique")

    def image_path(self, source_id: str) -> Path:
        relative = Path(source_id)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("unsafe source_id")
        candidates = [self.images_root / relative, self.images_root / relative.name]
        for candidate in candidates:
            resolved = candidate.resolve()
            if self.images_root in resolved.parents and resolved.is_file():
                return resolved
        return candidates[0]

    def status(self) -> dict[str, Any]:
        available = sum(self.image_path(record["source_id"]).is_file() for record in self.records)
        return {
            "configured": True,
            "images_root": str(self.images_root),
            "traces_path": str(self.traces_path),
            "trace_count": len(self.records),
            "available_images": available,
            "missing_images": len(self.records) - available,
            "invalid_lines": len(self.errors),
            "errors": self.errors[:10],
            "benchmarks": dict(Counter(record["benchmark"] for record in self.records)),
            "lengths": dict(sorted(Counter(len(record["actions"]) for record in self.records).items())),
        }

    def listing(self, query: str = "", offset: int = 0, limit: int = 100) -> dict[str, Any]:
        query = query.strip().lower()
        selected = [record for record in self.records if not query or query in record["trace_id"].lower() or query in record["source_id"].lower()]
        rows = []
        for record in selected[offset:offset + limit]:
            rows.append({
                "trace_id": record["trace_id"],
                "source_id": record["source_id"],
                "ground_truth_label": record["ground_truth_label"],
                "benchmark": record["benchmark"],
                "length": len(record["actions"]),
                "operations": [action["op_name"] for action in record["actions"]],
                "image_available": self.image_path(record["source_id"]).is_file(),
            })
        return {"total": len(selected), "offset": offset, "items": rows}

    def open_session(self, trace_id: str) -> Session:
        if trace_id not in self.by_id:
            raise KeyError("Trace not found.")
        record = self.by_id[trace_id]
        path = self.image_path(record["source_id"])
        if not path.is_file():
            raise FileNotFoundError(f"Image not found: {path}")
        with Image.open(path) as opened:
            image = opened.convert("RGB")
        expected_spec = record.get("source_spec", {})
        if expected_spec and image.size != (expected_spec.get("width"), expected_spec.get("height")):
            raise ValueError(f"Source dimensions {image.size} do not match frozen spec.")
        source_hash = image_hash(image)
        session = create_session(
            image=image,
            filename=Path(record["source_id"]).name,
            ground_truth_label=int(record["ground_truth_label"]),
            trace_id=record["trace_id"], benchmark=record["benchmark"], source_id=record["source_id"],
            expected_source_hash=record.get("source_hash"), computed_source_hash=source_hash,
            source_hash_match=source_hash == record.get("source_hash"), dataset_mode=True,
        )
        current = session.source.copy()
        for raw in sorted(record["actions"], key=lambda action: action["index"]):
            edit_type, params = self._map_action(raw)
            action = Action(
                id=raw["action_id"], type=edit_type, params=params,
                input_size=(raw["input_spec"]["width"], raw["input_spec"]["height"]),
                output_size=(raw["output_spec"]["width"], raw["output_spec"]["height"]),
                expected_output_hash=raw.get("output_hash"), seed=raw.get("seed"), op_version=raw.get("op_version", "1.0"),
            )
            current = apply_action(current, action)
            action.computed_output_hash = image_hash(current)
            action.hash_match = action.computed_output_hash == action.expected_output_hash
            session.actions.append(action)
        session.frozen_hashes_checked = len(session.actions)
        session.frozen_hashes_matching = sum(action.hash_match is True for action in session.actions)
        session.replay_valid = bool(session.source_hash_match and session.frozen_hashes_matching == session.frozen_hashes_checked)
        session.replay_reason = "source and action hashes match the frozen trace" if session.replay_valid else "source or action replay hashes do not match the frozen trace"
        return session

    def _map_action(self, raw: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        kind = OP_MAP[raw["op_name"]]
        params = dict(raw["params"])
        if kind == "blur":
            params = {"radius": float(params["sigma"]), "bypass": bool(params.get("bypass", False))}
        elif kind == "crop":
            left, top, right, bottom = params["box"]
            params = {"x": left, "y": top, "width": right - left, "height": bottom - top, "interpolation": params.get("interpolation", "bicubic")}
        return kind, params
