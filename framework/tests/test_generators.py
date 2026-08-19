from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from edittrace_xai.engine import ReplayEngine
from edittrace_xai.generators import TraceGenerator, oracle_subset


CONFIG = Path(__file__).parents[1] / "configs" / "development.yaml"


def fixture_image(size: int = 72) -> Image.Image:
    rng = np.random.default_rng(42)
    return Image.fromarray(rng.integers(0, 256, size=(size, size, 3), dtype=np.uint8), mode="RGB")


class GeneratorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.generator = TraceGenerator.from_yaml(CONFIG)
        self.source = fixture_image()

    def test_natural_is_reproducible(self) -> None:
        first = self.generator.natural(self.source, source_id="fixture/noise", ordinal=2)
        second = self.generator.natural(self.source, source_id="fixture/noise", ordinal=2)
        self.assertEqual(first, second)
        self.assertEqual(len(first.actions), 5)

    def test_lengths_are_balanced_by_ordinal(self) -> None:
        lengths = [
            len(self.generator.natural(self.source, source_id=f"fixture/{i}", ordinal=i).actions)
            for i in range(8)
        ]
        self.assertEqual(lengths, [3, 4, 5, 6, 3, 4, 5, 6])

    def test_every_counterfactual_executes(self) -> None:
        trace = self.generator.natural(self.source, source_id="fixture/replay", ordinal=3)
        engine = ReplayEngine()
        for action in trace.actions:
            result = engine.remove_action(self.source, trace, action.action_id)
            self.assertEqual(result.image.size, self.source.size)
        for first, second in zip(trace.actions, trace.actions[1:]):
            result = engine.remove_pair(self.source, trace, first.action_id, second.action_id)
            self.assertEqual(result.image.size, self.source.size)

    def test_oracle_subset_is_stable(self) -> None:
        traces = [
            self.generator.natural(self.source, source_id=f"fixture/{i}", ordinal=i)
            for i in range(6)
        ]
        first = oracle_subset(traces, count=3)
        second = oracle_subset(reversed(traces), count=3)
        self.assertEqual([x.trace_id for x in first], [x.trace_id for x in second])
        self.assertTrue(all(x.benchmark == "B-Oracle" for x in first))


if __name__ == "__main__":
    unittest.main()
