from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from edittrace_xai.engine import ReplayEngine, image_hash
from edittrace_xai.generators import (
    TraceGenerator,
    derive_seed,
    generate_interaction,
    generate_single,
)
from edittrace_xai.manifest import (
    audit_eligible_manifests,
    build_flat_manifest,
    file_sha256,
    select_model_eligible,
)
from edittrace_xai.schema import DatasetEntry


CONFIG = Path(__file__).parents[1] / "configs" / "development.yaml"


def fixture_image(size: int = 64, offset: int = 0) -> Image.Image:
    y, x = np.mgrid[0:size, 0:size]
    array = np.stack([
        (x * 3 + offset) % 256,
        (y * 4 + offset) % 256,
        ((x + y) * 2 + offset) % 256,
    ], axis=-1).astype(np.uint8)
    return Image.fromarray(array, mode="RGB")


class HashPredictor:
    model_id = "fixture:hash"

    def __init__(self, positive_hashes: set[str]):
        self.positive_hashes = positive_hashes

    def logits(self, image: Image.Image) -> np.ndarray:
        margin = 1.0 if image_hash(image) in self.positive_hashes else -1.0
        return np.array([margin, 0.0], dtype=np.float64)


class AlwaysCorrectPredictor:
    model_id = "fixture:always-correct"

    def logits(self, image: Image.Image) -> np.ndarray:
        return np.array([1.0, 0.0], dtype=np.float64)


class ModelGeneratorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.generator = TraceGenerator.from_yaml(CONFIG)
        self.source = fixture_image()
        self.engine = ReplayEngine()

    def test_single_generator_accepts_predeclared_trigger(self) -> None:
        source_id = "fixture/single"
        seed = derive_seed(
            self.generator.config["global_seed"], "B-Single", source_id,
            HashPredictor.model_id, 1,
        )
        length = 3
        rng = np.random.default_rng(seed)
        designated = int(rng.integers(0, length))
        trace = self.generator.build_trace(
            self.source,
            source_id=source_id,
            source_path=None,
            ground_truth_label=0,
            benchmark="B-Single",
            seed=seed,
            length=length,
            model_id=HashPredictor.model_id,
            designated_index=designated,
        )
        designated_hash = self.engine.remove_action(
            self.source, trace, trace.designated_action_id
        ).output_hash
        predictor = HashPredictor({image_hash(self.source), designated_hash})
        outcome = generate_single(
            self.generator,
            self.source,
            source_id=source_id,
            target=0,
            predictor=predictor,
            max_attempts=1,
        )
        self.assertTrue(outcome.accepted)
        self.assertEqual(outcome.attempts, 1)

    def test_interaction_generator_accepts_pair_only_case(self) -> None:
        source_id = "fixture/interaction"
        seed = derive_seed(
            self.generator.config["global_seed"], "B-Interaction", source_id,
            HashPredictor.model_id, 1,
        )
        length = 3
        rng = np.random.default_rng(seed)
        pair_indices = tuple(sorted(rng.choice(length, size=2, replace=False).tolist()))
        trace = self.generator.build_trace(
            self.source,
            source_id=source_id,
            source_path=None,
            ground_truth_label=0,
            benchmark="B-Interaction",
            seed=seed,
            length=length,
            model_id=HashPredictor.model_id,
            designated_pair_indices=pair_indices,
        )
        first, second = trace.designated_pair
        pair_hash = self.engine.remove_pair(self.source, trace, first, second).output_hash
        predictor = HashPredictor({image_hash(self.source), pair_hash})
        outcome = generate_interaction(
            self.generator,
            self.source,
            source_id=source_id,
            target=0,
            predictor=predictor,
            max_attempts=1,
        )
        self.assertTrue(outcome.accepted)
        self.assertEqual(outcome.attempts, 1)

    def test_global_candidate_pools_are_disjoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = []
            records = []
            for index in range(10):
                path = Path(directory) / f"{index}.png"
                fixture_image(offset=index).save(path)
                paths.append(path)
                records.append(DatasetEntry(
                    source_id=str(index),
                    path=str(path),
                    ground_truth_label=0,
                    sha256=file_sha256(path),
                ))
            selected, summary = select_model_eligible(
                records,
                AlwaysCorrectPredictor(),
                seed=7,
                development_candidate_pool=4,
                development_count=3,
                test_count=4,
            )
            development = {r.source_id for r in selected if r.split == "development"}
            test = {r.source_id for r in selected if r.split == "test"}
            self.assertFalse(development & test)
            self.assertEqual(summary["development_selected"], 3)
            self.assertEqual(summary["test_selected"], 4)

    def test_flat_manifest_and_cross_model_audit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for index in range(6):
                fixture_image(offset=index).save(root / f"ILSVRC2012_val_{index:08d}.JPEG")
            labels = root / "labels.txt"
            labels.write_text("1\n2\n3\n4\n5\n6\n", encoding="utf-8")
            records = build_flat_manifest(root, labels)
            self.assertEqual([record.ground_truth_label for record in records], list(range(6)))
            model_a = [
                record.model_copy(update={
                    "split": "development" if index < 2 else "test",
                    "model_id": "a",
                })
                for index, record in enumerate(records[:4])
            ]
            model_b = [
                records[index].model_copy(update={
                    "split": "development" if index < 2 else "test",
                    "model_id": "b",
                })
                for index in range(4)
            ]
            report = audit_eligible_manifests(
                {"a": model_a, "b": model_b}, expected_development=2, expected_test=2
            )
            self.assertTrue(report["passed"])


if __name__ == "__main__":
    unittest.main()
