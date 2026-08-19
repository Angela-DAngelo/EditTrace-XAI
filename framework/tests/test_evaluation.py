from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from edittrace_xai.engine import ReplayEngine, image_hash
from edittrace_xai.evaluation import (
    PerceptualMetric,
    TimedPredictor,
    clustered_bootstrap_mean,
    normalized_parameter_magnitude,
    paired_clustered_bootstrap_difference,
)
from edittrace_xai.generators import TraceGenerator, derive_seed, generate_interaction, generate_single
from scripts.evaluate_rq1 import evaluate_trace as evaluate_rq1_trace
from scripts.evaluate_rq2 import evaluate_trace as evaluate_rq2_trace
from scripts.evaluate_rq3 import evaluate_trace as evaluate_rq3_trace
from scripts.evaluate_rq4 import aggregate as aggregate_rq4
from scripts.build_paper_tables import write_simple_latex


ROOT = Path(__file__).parents[1]
CONFIG_PATH = ROOT / "configs" / "development.yaml"


def fixture_image(size: int = 64) -> Image.Image:
    y, x = np.mgrid[0:size, 0:size]
    array = np.stack([
        (x * 3) % 256,
        (y * 4) % 256,
        ((x + y) * 2) % 256,
    ], axis=-1).astype(np.uint8)
    return Image.fromarray(array, mode="RGB")


class HashPredictor:
    model_id = "fixture:hash"

    def __init__(self, positive_hashes: set[str]):
        self.positive_hashes = positive_hashes

    def logits(self, image: Image.Image) -> np.ndarray:
        margin = 1.0 if image_hash(image) in self.positive_hashes else -1.0
        return np.asarray([margin, 0.0], dtype=np.float64)


class EvaluationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.generator = TraceGenerator.from_yaml(CONFIG_PATH)
        self.config = self.generator.config
        self.source = fixture_image()
        self.engine = ReplayEngine()

    def _saved_source(self, directory: str) -> Path:
        path = Path(directory) / "source.png"
        self.source.save(path)
        return path

    def test_clustered_bootstrap_and_paired_difference(self) -> None:
        rows = [
            {"source_id": "a", "method": "m", "value": 1.0},
            {"source_id": "a", "method": "b", "value": 0.0},
            {"source_id": "b", "method": "m", "value": 1.0},
            {"source_id": "b", "method": "b", "value": 0.0},
        ]
        estimate = clustered_bootstrap_mean(
            [row for row in rows if row["method"] == "m"], "value", resamples=100, seed=1
        )
        difference = paired_clustered_bootstrap_difference(
            rows, "m", "b", "value", resamples=100, seed=1
        )
        self.assertEqual(estimate["estimate"], 1.0)
        self.assertEqual(difference["estimate"], 1.0)
        self.assertEqual(difference["ci95_low"], 1.0)

    def test_rq1_designated_trigger_is_ranked_first(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source_path = self._saved_source(directory)
            source_id = "fixture/single"
            seed = derive_seed(self.config["global_seed"], "B-Single", source_id, HashPredictor.model_id, 1)
            length = 3
            rng = np.random.default_rng(seed)
            designated_index = int(rng.integers(0, length))
            trace = self.generator.build_trace(
                self.source,
                source_id=source_id,
                source_path=str(source_path),
                ground_truth_label=0,
                benchmark="B-Single",
                seed=seed,
                length=length,
                model_id=HashPredictor.model_id,
                designated_index=designated_index,
            )
            designated_hash = self.engine.remove_action(
                self.source, trace, trace.designated_action_id
            ).output_hash
            predictor = HashPredictor({image_hash(self.source), designated_hash})
            outcome = generate_single(
                self.generator,
                self.source,
                source_id=source_id,
                source_path=str(source_path),
                target=0,
                predictor=predictor,
                max_attempts=1,
            )
            rows, errors = evaluate_rq1_trace(
                outcome.trace,
                TimedPredictor(predictor),
                self.config,
                None,
                include_ig=False,
                ig_steps=8,
            )
            proposed = next(row for row in rows if row["method"] == "counterfactual_replay")
            self.assertFalse(errors)
            self.assertEqual(proposed["top1"], 1)

    def test_rq2_designated_pair_is_ranked_first(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source_path = self._saved_source(directory)
            source_id = "fixture/interaction"
            seed = derive_seed(
                self.config["global_seed"], "B-Interaction", source_id, HashPredictor.model_id, 1
            )
            length = 3
            rng = np.random.default_rng(seed)
            pair_indices = tuple(sorted(rng.choice(length, size=2, replace=False).tolist()))
            trace = self.generator.build_trace(
                self.source,
                source_id=source_id,
                source_path=str(source_path),
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
                source_path=str(source_path),
                target=0,
                predictor=predictor,
                max_attempts=1,
            )
            rows, errors = evaluate_rq2_trace(outcome.trace, TimedPredictor(predictor), None)
            proposed = next(row for row in rows if row["method"] == "difference_in_differences")
            self.assertFalse(errors)
            self.assertEqual(proposed["top1"], 1)
            self.assertEqual(proposed["selected_pair_only_recovery"], 1)

    def test_rq3_single_rollback_restores_target(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source_path = self._saved_source(directory)
            trace = self.generator.natural(
                self.source,
                source_id="fixture/natural",
                source_path=str(source_path),
                ground_truth_label=0,
                model_id=HashPredictor.model_id,
            )
            restored_hash = self.engine.remove_action(
                self.source, trace, trace.actions[0].action_id
            ).output_hash
            predictor = HashPredictor({image_hash(self.source), restored_hash})
            rows, error = evaluate_rq3_trace(
                trace,
                TimedPredictor(predictor),
                self.config,
                PerceptualMetric("l1"),
                None,
                exhaustive_verification=False,
            )
            proposed = next(row for row in rows if row["method"] == "constrained_repair")
            self.assertIsNone(error)
            self.assertEqual(proposed["success"], 1)
            self.assertEqual(proposed["tier"], "1A-single-rollback")

    def test_operation_magnitude_is_bounded(self) -> None:
        trace = self.generator.natural(self.source, source_id="fixture/magnitude")
        for action in trace.actions:
            value = normalized_parameter_magnitude(action, self.config)
            self.assertGreaterEqual(value, 0.0)
            self.assertLessEqual(value, 1.0)

    def test_rq4_uses_only_primary_methods(self) -> None:
        groups = aggregate_rq4([
            {
                "rq": "RQ1", "model": "fixture", "method": "counterfactual_replay",
                "history_length": 3, "latency_ms": 5.0, "model_calls": 4,
            },
            {
                "rq": "RQ1", "model": "fixture", "method": "random",
                "history_length": 3, "latency_ms": 0.1, "model_calls": 0,
            },
        ])
        overall = next(group for group in groups if group["history_length"] is None)
        self.assertEqual(overall["component"], "diagnosis")
        self.assertEqual(overall["n"], 1)
        self.assertEqual(overall["latency_ms"]["median"], 5.0)

    def test_latex_table_writer_escapes_method_names(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "table.tex"
            write_simple_latex(
                path,
                "RQ1 results",
                "tab:rq1",
                ["Method", "Top-1"],
                [["counterfactual_replay", "0.900 [0.800, 0.950]"]],
            )
            text = path.read_text(encoding="utf-8")
            self.assertIn(r"counterfactual\_replay", text)
            self.assertIn(r"\begin{table}", text)


if __name__ == "__main__":
    unittest.main()
