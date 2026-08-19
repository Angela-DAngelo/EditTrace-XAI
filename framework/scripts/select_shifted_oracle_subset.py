#!/usr/bin/env python3
"""Freeze a deterministic shifted B-Oracle subset before repair evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from edittrace_xai.engine import ReplayEngine
from edittrace_xai.evaluation import (
    TimedPredictor,
    build_frozen_predictor,
    load_source,
    load_traces,
    stable_seed,
    synchronize_for_predictor,
    warmup_predictor,
    write_json,
    write_jsonl,
)
from edittrace_xai.modeling import target_margin


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--model", required=True, choices=("resnet50", "vit_b_16"))
    parser.add_argument("--image-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--warmup", type=int, default=10)
    args = parser.parse_args()

    traces = load_traces(args.input)
    if not traces:
        raise RuntimeError("no B-Oracle traces found")
    raw = build_frozen_predictor(args.model, args.device)
    warmup_predictor(raw, load_source(traces[0], args.image_root), args.warmup)
    predictor = TimedPredictor(raw, synchronize_for_predictor(raw))
    ordered = sorted(
        traces,
        key=lambda trace: (stable_seed(args.seed, "B-Oracle-shifted", trace.trace_id), trace.trace_id),
    )
    selected = []
    shifted_total = 0
    for trace in ordered:
        if trace.ground_truth_label is None:
            continue
        source = load_source(trace, args.image_root)
        final = ReplayEngine().replay(source, trace).image
        source_margin = target_margin(predictor.logits(source), trace.ground_truth_label)
        final_margin = target_margin(predictor.logits(final), trace.ground_truth_label)
        if source_margin > 0 and final_margin <= 0:
            shifted_total += 1
            if len(selected) < args.count:
                selected.append(trace.model_copy(update={"model_id": raw.model_id}))
    if len(selected) < args.count:
        raise RuntimeError(f"only {len(selected)} shifted traces available; requested {args.count}")

    write_jsonl(args.output, [trace.model_dump(mode="json") for trace in selected])
    manifest_path = args.output.with_suffix(args.output.suffix + ".manifest.json")
    manifest = {
        "schema_version": "1.0",
        "purpose": "DEV-001 deterministic shifted B-Oracle subset",
        "model": raw.model_id,
        "input": str(args.input.resolve()),
        "input_sha256": sha256(args.input),
        "selection_seed": args.seed,
        "requested_count": args.count,
        "shifted_available": shifted_total,
        "selected_trace_ids": [trace.trace_id for trace in selected],
        "output": str(args.output.resolve()),
    }
    write_json(manifest_path, manifest)
    manifest["output_sha256"] = sha256(args.output)
    write_json(manifest_path, manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
