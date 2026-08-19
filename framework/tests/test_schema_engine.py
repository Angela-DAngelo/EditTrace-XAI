from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from edittrace_xai.engine import ReplayEngine, ReplayError, image_hash
from edittrace_xai.generators import TraceGenerator
from edittrace_xai.operations import OperationValidationError, apply_operation, attenuate_params
from edittrace_xai.schema import OperationName, TraceRecord


CONFIG = Path(__file__).parents[1] / "configs" / "development.yaml"


def fixture_image(size: int = 64) -> Image.Image:
    y, x = np.mgrid[0:size, 0:size]
    array = np.stack([
        (x * 4) % 256,
        (y * 4) % 256,
        ((x + y) * 2) % 256,
    ], axis=-1).astype(np.uint8)
    return Image.fromarray(array, mode="RGB")


class SchemaEngineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.source = fixture_image()
        self.generator = TraceGenerator.from_yaml(CONFIG)
        self.trace = self.generator.natural(
            self.source, source_id="fixture/gradient", ordinal=0
        )

    def test_schema_round_trip(self) -> None:
        payload = self.trace.model_dump_json()
        restored = TraceRecord.model_validate_json(payload)
        self.assertEqual(restored, self.trace)

    def test_deterministic_original_replay(self) -> None:
        engine = ReplayEngine()
        first = engine.replay(self.source, self.trace, verify_original_lineage=True)
        second = engine.replay(self.source, self.trace, verify_original_lineage=True)
        self.assertEqual(first.output_hash, self.trace.final_hash)
        self.assertEqual(first.output_hash, second.output_hash)
        self.assertEqual(first.image.tobytes(), second.image.tobytes())

    def test_remove_and_attenuate(self) -> None:
        engine = ReplayEngine()
        action = self.trace.actions[0]
        removed = engine.remove_action(self.source, self.trace, action.action_id)
        identity = engine.attenuate_action(self.source, self.trace, action.action_id, 1.0)
        self.assertEqual(removed.output_hash, identity.output_hash)
        original = engine.attenuate_action(self.source, self.trace, action.action_id, 0.0)
        self.assertEqual(original.output_hash, self.trace.final_hash)
        half = engine.attenuate_action(self.source, self.trace, action.action_id, 0.5)
        self.assertEqual(half.override_action_ids, (action.action_id,))

    def test_unknown_action_fails_typed(self) -> None:
        with self.assertRaises(ReplayError) as context:
            ReplayEngine().remove_action(self.source, self.trace, "missing")
        self.assertEqual(context.exception.code, "unknown_action")

    def test_unique_operation_types(self) -> None:
        names = [action.op_name for action in self.trace.actions]
        self.assertEqual(len(names), len(set(names)))

    def test_identity_operations(self) -> None:
        for op, params in [
            (OperationName.BRIGHTNESS, {"factor": 1.0}),
            (OperationName.CONTRAST, {"factor": 1.0}),
            (OperationName.SATURATION, {"factor": 1.0}),
            (OperationName.GAUSSIAN_BLUR, {"sigma": 0.0, "bypass": True}),
            (OperationName.CROP_RESIZE, {"box": [0, 0, 1, 1], "interpolation": "bicubic"}),
            (OperationName.JPEG, {"quality": 100, "subsampling": 0, "bypass": True}),
        ]:
            output = apply_operation(self.source, op, params).image
            self.assertEqual(image_hash(output), image_hash(self.source), op)

    def test_invalid_crop_rejected(self) -> None:
        with self.assertRaises(OperationValidationError):
            apply_operation(
                self.source,
                OperationName.CROP_RESIZE,
                {"box": [0.8, 0.2, 0.2, 0.9], "interpolation": "bicubic"},
            )

    def test_attenuation_endpoints(self) -> None:
        params = {"box": [0.1, 0.2, 0.8, 0.9], "interpolation": "bicubic"}
        self.assertEqual(attenuate_params(OperationName.CROP_RESIZE, params, 0), params)
        self.assertEqual(
            attenuate_params(OperationName.CROP_RESIZE, params, 1)["box"],
            [0.0, 0.0, 1.0, 1.0],
        )


if __name__ == "__main__":
    unittest.main()
