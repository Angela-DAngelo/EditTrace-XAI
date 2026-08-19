from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image

from .modeling import Predictor, predicted_label
from .schema import DatasetEntry


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def build_imagefolder_manifest(root: str | Path, limit: int | None = None) -> list[DatasetEntry]:
    root = Path(root).resolve()
    if not root.is_dir():
        raise ValueError(f"dataset root is not a directory: {root}")
    class_dirs = sorted(path for path in root.iterdir() if path.is_dir())
    class_to_label = {path.name: index for index, path in enumerate(class_dirs)}
    entries: list[DatasetEntry] = []
    for class_dir in class_dirs:
        for image_path in sorted(class_dir.rglob("*")):
            if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            relative = image_path.relative_to(root)
            entries.append(DatasetEntry(
                source_id=relative.as_posix(),
                path=str(image_path),
                ground_truth_label=class_to_label[class_dir.name],
                class_name=class_dir.name,
                sha256=file_sha256(image_path),
            ))
            if limit is not None and len(entries) >= limit:
                return entries
    return entries


def build_flat_manifest(
    root: str | Path,
    labels_file: str | Path,
    *,
    labels_one_indexed: bool = True,
    limit: int | None = None,
) -> list[DatasetEntry]:
    root = Path(root).resolve()
    labels_file = Path(labels_file).resolve()
    if not root.is_dir():
        raise ValueError(f"dataset root is not a directory: {root}")
    if not labels_file.is_file():
        raise ValueError(f"labels file does not exist: {labels_file}")
    image_paths = sorted(
        path for path in root.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    labels = [int(line.strip()) for line in labels_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(labels) != len(image_paths):
        raise ValueError(
            f"flat labels/image mismatch: labels={len(labels)}, images={len(image_paths)}"
        )
    entries: list[DatasetEntry] = []
    for image_path, label in zip(image_paths, labels, strict=True):
        zero_based = label - 1 if labels_one_indexed else label
        if zero_based < 0:
            raise ValueError("labels must be non-negative after index conversion")
        entries.append(DatasetEntry(
            source_id=image_path.name,
            path=str(image_path),
            ground_truth_label=zero_based,
            sha256=file_sha256(image_path),
        ))
        if limit is not None and len(entries) >= limit:
            break
    return entries


def read_jsonl(path: str | Path) -> list[DatasetEntry]:
    records: list[DatasetEntry] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(DatasetEntry.model_validate(json.loads(line)))
    return records


def write_jsonl(path: str | Path, records: Iterable[DatasetEntry]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(record.model_dump_json() + "\n")


def select_model_eligible(
    records: list[DatasetEntry],
    predictor: Predictor,
    *,
    seed: int,
    development_candidate_pool: int,
    development_count: int,
    test_count: int,
) -> tuple[list[DatasetEntry], dict[str, int | str]]:
    if any(record.ground_truth_label is None for record in records):
        raise ValueError("eligibility selection requires ground_truth_label for every record")
    if development_candidate_pool <= 0 or development_candidate_pool >= len(records):
        raise ValueError("development_candidate_pool must split the manifest into two non-empty pools")
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(records)).tolist()
    development_indices = order[:development_candidate_pool]
    test_indices = order[development_candidate_pool:]

    def collect(indices: list[int], split: str, target_count: int) -> tuple[list[DatasetEntry], int]:
        selected: list[DatasetEntry] = []
        evaluated = 0
        for index in indices:
            record = records[index]
            with Image.open(record.path) as image:
                label = predicted_label(predictor.logits(image.convert("RGB")))
            evaluated += 1
            if label == record.ground_truth_label:
                selected.append(record.model_copy(update={"split": split, "model_id": predictor.model_id}))
                if len(selected) == target_count:
                    break
        return selected, evaluated

    development, development_evaluated = collect(
        development_indices, "development", development_count
    )
    test, test_evaluated = collect(test_indices, "test", test_count)
    if len(development) < development_count or len(test) < test_count:
        raise RuntimeError(
            "insufficient correctly classified images: "
            f"development={len(development)}/{development_count}, test={len(test)}/{test_count}"
        )
    summary: dict[str, int | str] = {
        "model_id": predictor.model_id,
        "seed": seed,
        "development_candidate_pool": development_candidate_pool,
        "development_selected": len(development),
        "development_evaluated": development_evaluated,
        "test_selected": len(test),
        "test_evaluated": test_evaluated,
    }
    return development + test, summary


def audit_eligible_manifests(
    manifests: dict[str, list[DatasetEntry]],
    *,
    expected_development: int,
    expected_test: int,
) -> dict[str, object]:
    report: dict[str, object] = {"manifests": {}, "cross_split_leakage": []}
    all_development: set[str] = set()
    all_test: set[str] = set()
    for name, records in manifests.items():
        development = {record.source_id for record in records if record.split == "development"}
        test = {record.source_id for record in records if record.split == "test"}
        unassigned = [record.source_id for record in records if record.split == "unassigned"]
        overlap = sorted(development & test)
        model_ids = sorted({record.model_id for record in records if record.model_id is not None})
        report["manifests"][name] = {
            "development": len(development),
            "test": len(test),
            "unassigned": len(unassigned),
            "within_manifest_overlap": overlap,
            "model_ids": model_ids,
            "counts_match_protocol": (
                len(development) == expected_development and len(test) == expected_test
            ),
        }
        all_development.update(development)
        all_test.update(test)
    report["cross_split_leakage"] = sorted(all_development & all_test)
    report["passed"] = (
        not report["cross_split_leakage"]
        and all(
            item["counts_match_protocol"]
            and not item["within_manifest_overlap"]
            and item["unassigned"] == 0
            for item in report["manifests"].values()
        )
    )
    return report
