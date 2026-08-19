#!/usr/bin/env python3
"""Evaluate RQ2 pairwise interaction diagnosis on accepted B-Interaction outcomes."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from edittrace_xai.engine import ReplayEngine, ReplayError
from edittrace_xai.evaluation import (
    TimedPredictor,
    average_precision,
    build_frozen_predictor,
    clustered_bootstrap_mean,
    file_sha256,
    load_source,
    load_traces,
    pair_key,
    paired_clustered_bootstrap_difference,
    reciprocal_rank,
    synchronize_for_predictor,
    warmup_predictor,
    write_json,
    write_jsonl,
)
from edittrace_xai.modeling import TorchvisionPredictor, target_margin
from edittrace_xai.schema import TraceRecord


def evaluate_trace(
    trace: TraceRecord,
    predictor: TimedPredictor,
    image_root: Path | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if trace.benchmark != "B-Interaction" or trace.designated_pair is None:
        raise ValueError(f"RQ2 requires accepted B-Interaction traces: {trace.trace_id}")
    if trace.ground_truth_label is None:
        raise ValueError(f"trace {trace.trace_id} has no ground-truth label")
    source = load_source(trace, image_root)
    engine = ReplayEngine()
    order = {action.action_id: action.index for action in trace.actions}
    errors: list[dict[str, Any]] = []
    snapshot = predictor.snapshot()
    start = time.perf_counter_ns()

    full = engine.replay(source, trace)
    full_margin = target_margin(predictor.logits(full.image), trace.ground_truth_label)
    single_margins: dict[str, float] = {}
    for action in trace.actions:
        try:
            replay = engine.remove_action(source, trace, action.action_id)
            single_margins[action.action_id] = target_margin(
                predictor.logits(replay.image), trace.ground_truth_label
            )
        except ReplayError as exc:
            errors.append({
                "trace_id": trace.trace_id,
                "action_id": action.action_id,
                "code": exc.code,
            })

    pair_margins: dict[str, float] = {}
    pair_members: dict[str, tuple[str, str]] = {}
    for first, second in itertools.combinations(trace.actions, 2):
        key = pair_key(first.action_id, second.action_id, order)
        try:
            replay = engine.remove_pair(source, trace, first.action_id, second.action_id)
            pair_margins[key] = target_margin(predictor.logits(replay.image), trace.ground_truth_label)
            pair_members[key] = (first.action_id, second.action_id)
        except ReplayError as exc:
            errors.append({
                "trace_id": trace.trace_id,
                "pair": [first.action_id, second.action_id],
                "code": exc.code,
            })
    elapsed_ms = (time.perf_counter_ns() - start) / 1_000_000
    calls = predictor.calls - snapshot[0]

    interaction_scores: dict[str, float] = {}
    independent_scores: dict[str, float] = {}
    pair_only: dict[str, bool] = {}
    for key, joint_margin in pair_margins.items():
        first, second = pair_members[key]
        if first not in single_margins or second not in single_margins:
            continue
        interaction_scores[key] = (
            joint_margin - single_margins[first] - single_margins[second] + full_margin
        )
        independent_scores[key] = (
            single_margins[first] - full_margin + single_margins[second] - full_margin
        )
        pair_only[key] = (
            single_margins[first] <= 0
            and single_margins[second] <= 0
            and joint_margin > 0
        )

    designated = pair_key(trace.designated_pair[0], trace.designated_pair[1], order)
    pair_order = {
        key: order[pair_members[key][0]] * len(trace.actions) + order[pair_members[key][1]]
        for key in pair_members
    }
    rows: list[dict[str, Any]] = []
    for method, scores, method_elapsed, method_calls in (
        ("difference_in_differences", interaction_scores, elapsed_ms, calls),
        ("independent_action", independent_scores, 0.0, 0),
    ):
        ranking = sorted(scores, key=lambda key: (-scores[key], pair_order[key], key))
        labels = [int(key == designated) for key in scores]
        score_values = [scores[key] for key in scores]
        selected = ranking[0] if ranking else None
        rows.append({
            "rq": "RQ2",
            "model": trace.model_id,
            "trace_id": trace.trace_id,
            "source_id": trace.source_id,
            "history_length": len(trace.actions),
            "designated_pair": designated,
            "method": method,
            "predicted_pair": selected,
            "top1": int(selected == designated),
            "reciprocal_rank": reciprocal_rank(ranking, designated),
            "average_precision": average_precision(labels, score_values),
            "selected_pair_only_recovery": int(bool(selected and pair_only[selected])),
            "designated_pair_only_recovery": int(bool(pair_only.get(designated, False))),
            "valid_pairs": len(scores),
            "ranking": ranking,
            "scores": scores,
            "full_margin": full_margin,
            "single_margins": single_margins,
            "pair_margins": pair_margins,
            "latency_ms": method_elapsed,
            "model_calls": method_calls,
        })
    return rows, errors


def summarize(rows: list[dict[str, Any]], resamples: int, seed: int) -> dict[str, Any]:
    methods = sorted({row["method"] for row in rows})
    results: dict[str, Any] = {}
    for method in methods:
        selected = [row for row in rows if row["method"] == method]
        results[method] = {
            metric: clustered_bootstrap_mean(selected, metric, resamples=resamples, seed=seed)
            for metric in ("top1", "reciprocal_rank", "average_precision", "selected_pair_only_recovery")
        }
        results[method]["mean_latency_ms"] = float(np.mean([row["latency_ms"] for row in selected]))
        results[method]["total_model_calls"] = int(sum(row["model_calls"] for row in selected))
    differences = []
    for metric in ("top1", "reciprocal_rank", "selected_pair_only_recovery"):
        differences.append(
            paired_clustered_bootstrap_difference(
                rows,
                "difference_in_differences",
                "independent_action",
                metric,
                resamples=resamples,
                seed=seed,
            )
        )
    return {"methods": results, "paired_differences": differences}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "model", "trace_id", "source_id", "history_length", "method", "top1",
        "reciprocal_rank", "average_precision", "selected_pair_only_recovery",
        "valid_pairs", "latency_ms", "model_calls",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in fields})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--model", required=True, choices=("resnet50", "vit_b_16"))
    parser.add_argument("--image-root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260802)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    traces = load_traces(args.input, accepted_only=True)
    if args.limit is not None:
        traces = traces[: args.limit]
    if not traces:
        raise RuntimeError("no accepted B-Interaction traces found")
    raw_predictor = build_frozen_predictor(args.model, args.device)
    mismatched = sorted({trace.model_id for trace in traces if trace.model_id != raw_predictor.model_id})
    if mismatched:
        raise RuntimeError(f"trace model identifiers do not match evaluator: {mismatched}")
    warmup_predictor(raw_predictor, load_source(traces[0], args.image_root), args.warmup)
    predictor = TimedPredictor(raw_predictor, synchronize_for_predictor(raw_predictor))
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for trace in traces:
        trace_rows, trace_errors = evaluate_trace(trace, predictor, args.image_root)
        rows.extend(trace_rows)
        errors.extend(trace_errors)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"rq2_{args.model}"
    write_jsonl(args.output_dir / f"{stem}_per_trace.jsonl", rows)
    write_csv(args.output_dir / f"{stem}_per_trace.csv", rows)
    summary = {
        "schema_version": "1.0",
        "rq": "RQ2",
        "scope": "development" if "dev" in args.input.name.lower() else "evaluation",
        "model": raw_predictor.model_id,
        "input": str(args.input.resolve()),
        "input_sha256": file_sha256(args.input),
        "accepted_traces": len(traces),
        "warmup_iterations": args.warmup,
        "bootstrap_resamples": args.bootstrap_resamples,
        "bootstrap_seed": args.bootstrap_seed,
        "errors": errors,
        "results": summarize(rows, args.bootstrap_resamples, args.bootstrap_seed),
    }
    write_json(args.output_dir / f"{stem}_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
