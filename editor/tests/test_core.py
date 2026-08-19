import unittest
import json
import tempfile
from pathlib import Path

import numpy as np

from backend.core import (
    EDIT_SPECS,
    MODEL,
    add_action,
    apply_repair,
    create_session,
    diagnose,
    replay,
    image_hash,
)
from backend.dataset import TraceCatalog


class EditEngineTests(unittest.TestCase):
    def test_all_edits_are_deterministic_and_shape_preserving(self):
        cases = {
            "brightness": {"factor": 0.7},
            "contrast": {"factor": 1.4},
            "saturation": {"factor": 0.4},
            "blur": {"radius": 3.0},
            "crop": {"x": 0.1, "y": 0.15, "width": 0.8, "height": 0.7},
            "jpeg": {"quality": 35},
        }
        for kind, params in cases.items():
            with self.subTest(kind=kind):
                session = create_session()
                add_action(session, kind, params)
                first, _ = replay(session)
                second, _ = replay(session)
                self.assertEqual(first.size, session.source.size)
                np.testing.assert_array_equal(np.asarray(first), np.asarray(second))

    def test_removal_replays_valid_suffix(self):
        session = create_session()
        first = add_action(session, "brightness", {"factor": 0.55})
        add_action(session, "crop", {"x": 0.17, "y": 0.08, "width": 0.7, "height": 0.82})
        add_action(session, "blur", {"radius": 2.5})
        removed_a, _ = replay(session, {first.id})
        removed_b, _ = replay(session, {first.id})
        self.assertEqual(removed_a.size, session.source.size)
        np.testing.assert_array_equal(np.asarray(removed_a), np.asarray(removed_b))

    def test_history_length_is_frozen_at_six(self):
        session = create_session()
        for edit_type in EDIT_SPECS:
            add_action(session, edit_type, {})
        with self.assertRaisesRegex(ValueError, "six actions"):
            add_action(session, "blur", {"radius": 1})

    def test_edit_types_are_unique(self):
        session = create_session()
        add_action(session, "blur", {"radius": 1})
        with self.assertRaisesRegex(ValueError, "without replacement"):
            add_action(session, "blur", {"radius": 2})


class DiagnosisTests(unittest.TestCase):
    def test_single_effect_matches_declared_intervention(self):
        session = create_session()
        action = add_action(session, "blur", {"radius": 8})
        report = diagnose(session)
        current, _ = replay(session)
        removed, _ = replay(session, {action.id})
        expected = MODEL.prediction(removed, session.target_class)["margin"] - MODEL.prediction(current, session.target_class)["margin"]
        self.assertAlmostEqual(report["ranking"][0]["effect"], expected, places=6)

    def test_pair_interaction_is_difference_in_differences(self):
        session = create_session()
        first = add_action(session, "blur", {"radius": 5})
        second = add_action(session, "contrast", {"factor": 0.5})
        report = diagnose(session)
        full = MODEL.prediction(replay(session)[0], session.target_class)["margin"]
        only_first_removed = MODEL.prediction(replay(session, {first.id})[0], session.target_class)["margin"]
        only_second_removed = MODEL.prediction(replay(session, {second.id})[0], session.target_class)["margin"]
        both_removed = MODEL.prediction(replay(session, {first.id, second.id})[0], session.target_class)["margin"]
        expected = both_removed - only_first_removed - only_second_removed + full
        self.assertAlmostEqual(report["interactions"][0]["effect"], expected, places=6)

    def test_feasible_repair_is_executable_and_undoable(self):
        session = create_session()
        add_action(session, "blur", {"radius": 8})
        before = len(session.actions)
        report = diagnose(session)
        self.assertTrue(report["repairs"])
        apply_repair(session, report["repairs"][0])
        repaired = MODEL.prediction(replay(session)[0], session.target_class)
        self.assertTrue(repaired["recovered"])
        self.assertLessEqual(len(session.actions), before)
        self.assertTrue(session.undo)


class DatasetTests(unittest.TestCase):
    def test_frozen_trace_schema_maps_codec_and_geometry(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            images = root / "images" / "00007"
            images.mkdir(parents=True)
            source = create_session().source.resize((64, 48))
            image_path = images / "sample.jpg"
            source.save(image_path, quality=95)
            from PIL import Image
            with Image.open(image_path) as opened:
                canonical = opened.convert("RGB")
            spec = {"width": 64, "height": 48}
            actions = [
                {"action_id": "trace:a0", "index": 0, "op_name": "gaussian_blur", "op_version": "1.0", "params": {"sigma": 1.0, "bypass": False}, "input_spec": spec, "output_spec": spec, "output_hash": "0" * 64, "seed": 1},
                {"action_id": "trace:a1", "index": 1, "op_name": "jpeg", "op_version": "1.0", "params": {"quality": 40, "subsampling": 0, "optimize": False, "progressive": False, "bypass": False}, "input_spec": spec, "output_spec": spec, "output_hash": "1" * 64, "seed": 2},
                {"action_id": "trace:a2", "index": 2, "op_name": "crop_resize", "op_version": "1.0", "params": {"box": [0.1, 0.2, 0.9, 1.0], "interpolation": "bicubic"}, "input_spec": spec, "output_spec": spec, "output_hash": "2" * 64, "seed": 3},
            ]
            record = {"trace_id": "trace", "source_id": "00007/sample.jpg", "ground_truth_label": 7, "benchmark": "B-Natural", "source_hash": image_hash(canonical), "source_spec": {**spec, "mode": "RGB"}, "actions": actions}
            traces = root / "traces.jsonl"
            traces.write_text(json.dumps(record) + "\n")
            catalog = TraceCatalog(root / "images", traces)
            session = catalog.open_session("trace")
            self.assertTrue(session.source_hash_match)
            self.assertEqual(session.actions[0].params["radius"], 1.0)
            self.assertEqual(session.actions[1].params["subsampling"], 0)
            self.assertEqual(session.actions[2].params["interpolation"], "bicubic")
            self.assertAlmostEqual(session.actions[2].params["width"], 0.8)
            self.assertFalse(session.eligible)


if __name__ == "__main__":
    unittest.main()
