#!/usr/bin/env python3
"""Evaluate RQ1 action diagnosis on accepted B-Single outcomes.

The designated trigger is fixed by the controlled generator and is never derived
from the proposed leave-one-action-out score. Outputs include one record per
trace/method, clustered-bootstrap intervals, paired differences, and timings.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from edittrace_xai.engine import ReplayEngine, ReplayError
from edittrace_xai.evaluation import (
    TimedPredictor,
    build_frozen_predictor,
    clustered_bootstrap_mean,
    file_sha256,
    load_source,
    load_traces,
    normalized_parameter_magnitude,
    paired_clustered_bootstrap_difference,
    rank_ids,
    reciprocal_rank,
    stable_seed,
    synchronize_for_predictor,
    warmup_predictor,
    write_json,
    write_jsonl,
)
from edittrace_xai.generators import load_config
from edittrace_xai.modeling import TorchvisionPredictor, target_margin
from edittrace_xai.operations import apply_operation
from edittrace_xai.schema import TraceRecord


def adjacent_step_scores(source, trace: TraceRecord, predictor: TimedPredictor, target: int) -> dict[str, float]:
    current = source.copy()
    previous_margin = target_margin(predictor.logits(current), target)
    scores: dict[str, float] = {}
    for action in trace.actions:
        current = apply_operation(current, action.op_name, action.params).image
        current_margin = target_margin(predictor.logits(current), target)
        scores[action.action_id] = previous_margin - current_margin
        previous_margin = current_margin
    return scores


def integrated_gradients_support_scores(
    predictor: TorchvisionPredictor,
    image,
    trace: TraceRecord,
    target: int,
    steps: int,
) -> dict[str, float]:
    if steps < 2:
        raise ValueError("IG requires at least two integration steps")
    torch = predictor._torch
    model = predictor._model
    tensor = predictor._preprocess(image.convert("RGB")).unsqueeze(0).to(predictor.device)
    baseline = torch.zeros_like(tensor)
    gradients = torch.zeros_like(tensor)
    alphas = torch.linspace(1.0 / steps, 1.0, steps, device=tensor.device)
    for alpha in alphas:
        point = (baseline + alpha * (tensor - baseline)).detach().requires_grad_(True)
        output = model(point)
        mask = torch.ones(output.shape[1], dtype=torch.bool, device=output.device)
        mask[target] = False
        margin = output[0, target] - output[0, mask].max()
        model.zero_grad(set_to_none=True)
        margin.backward()
        gradients += point.grad.detach()
    attribution = ((tensor - baseline) * gradients / steps).abs().sum(dim=1)[0]
    attribution = attribution.detach().cpu().numpy()
    height, width = attribution.shape
    global_score = float(attribution.mean())
    scores: dict[str, float] = {}
    for action in trace.actions:
        if action.support.kind == "global" or action.support.box is None:
            scores[action.action_id] = global_score
            continue
        box = action.support.box
        left = max(0, min(width - 1, round(box.left * width)))
        top = max(0, min(height - 1, round(box.top * height)))
        right = max(left + 1, min(width, round(box.right * width)))
        bottom = max(top + 1, min(height, round(box.bottom * height)))
        scores[action.action_id] = float(attribution[top:bottom, left:right].mean())
    return scores


def method_record(
    trace: TraceRecord,
    method: str,
    scores: Mapping[str, float],
    elapsed_ms: float,
    model_calls: int,
) -> dict[str, Any]:
    if trace.designated_action_id is None:
        raise ValueError(f"trace {trace.trace_id} has no designated action")
    order = {action.action_id: action.index for action in trace.actions}
    ranking = rank_ids(scores, order)
    target = trace.designated_action_id
    return {
        "rq": "RQ1",
        "model": trace.model_id,
        "trace_id": trace.trace_id,
        "source_id": trace.source_id,
        "history_length": len(trace.actions),
        "designated_action_id": target,
        "designated_operation": next(a.op_name.value for a in trace.actions if a.action_id == target),
        "method": method,
        "predicted_action_id": ranking[0],
        "top1": int(ranking[0] == target),
        "reciprocal_rank": reciprocal_rank(ranking, target),
        "ranking": ranking,
        "scores": {key: float(value) for key, value in scores.items()},
        "latency_ms": elapsed_ms,
        "model_calls": model_calls,
    }


def evaluate_trace(
    trace: TraceRecord,
    predictor: TimedPredictor,
    config: Mapping[str, Any],
    image_root: Path | None,
    *,
    include_ig: bool,
    ig_steps: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if trace.benchmark != "B-Single" or trace.designated_action_id is None:
        raise ValueError(f"RQ1 requires accepted B-Single traces: {trace.trace_id}")
    if trace.ground_truth_label is None:
        raise ValueError(f"trace {trace.trace_id} has no ground-truth label")
    source = load_source(trace, image_root)
    engine = ReplayEngine()
    order = {action.action_id: action.index for action in trace.actions}
    errors: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []

    start_calls, start_elapsed = predictor.snapshot()
    start = time.perf_counter_ns()
    final = engine.replay(source, trace)
    final_margin = target_margin(predictor.logits(final.image), trace.ground_truth_label)
    replay_scores: dict[str, float] = {}
    for action in trace.actions:
        try:
            counterfactual = engine.remove_action(source, trace, action.action_id)
            margin = target_margin(predictor.logits(counterfactual.image), trace.ground_truth_label)
            replay_scores[action.action_id] = margin - final_margin
        except ReplayError as exc:
            errors.append({"trace_id": trace.trace_id, "action_id": action.action_id, "code": exc.code})
    proposed_ms = (time.perf_counter_ns() - start) / 1_000_000
    proposed_calls = predictor.calls - start_calls
    if len(replay_scores) == len(trace.actions):
        records.append(method_record(trace, "counterfactual_replay", replay_scores, proposed_ms, proposed_calls))

    baseline_specs: list[tuple[str, dict[str, float], float, int]] = []
    start = time.perf_counter_ns()
    last_scores = {action.action_id: float(action.index == len(trace.actions) - 1) for action in trace.actions}
    baseline_specs.append(("last_action", last_scores, (time.perf_counter_ns() - start) / 1_000_000, 0))

    start = time.perf_counter_ns()
    magnitude = {action.action_id: normalized_parameter_magnitude(action, config) for action in trace.actions}
    baseline_specs.append(("edit_magnitude", magnitude, (time.perf_counter_ns() - start) / 1_000_000, 0))

    start = time.perf_counter_ns()
    rng = np.random.default_rng(stable_seed(20260802, "RQ1", trace.trace_id))
    random_scores = {action.action_id: float(rng.random()) for action in trace.actions}
    baseline_specs.append(("random", random_scores, (time.perf_counter_ns() - start) / 1_000_000, 0))

    adjacent_snapshot = predictor.snapshot()
    start = time.perf_counter_ns()
    adjacent = adjacent_step_scores(source, trace, predictor, trace.ground_truth_label)
    baseline_specs.append(
        ("adjacent_score_change", adjacent, (time.perf_counter_ns() - start) / 1_000_000, predictor.calls - adjacent_snapshot[0])
    )

    for name, scores, elapsed, calls in baseline_specs:
        records.append(method_record(trace, name, scores, elapsed, calls))

    if include_ig:
        raw = predictor.predictor
        if not isinstance(raw, TorchvisionPredictor):
            raise TypeError("IG baseline requires TorchvisionPredictor")
        sync = synchronize_for_predictor(raw)
        if sync:
            sync()
        start = time.perf_counter_ns()
        ig_scores = integrated_gradients_support_scores(
            raw, final.image, trace, trace.ground_truth_label, ig_steps
        )
        if sync:
            sync()
        elapsed = (time.perf_counter_ns() - start) / 1_000_000
        records.append(method_record(trace, "integrated_gradients_support", ig_scores, elapsed, ig_steps))

    return records, errors


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "model", "trace_id", "source_id", "history_length", "designated_operation",
        "method", "top1", "reciprocal_rank", "latency_ms", "model_calls",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in fields})


def summarize(rows: list[dict[str, Any]], resamples: int, seed: int) -> dict[str, Any]:
    methods = sorted({str(row["method"]) for row in rows})
    metrics: dict[str, Any] = {}
    for method in methods:
        selected = [row for row in rows if row["method"] == method]
        metrics[method] = {
            "top1": clustered_bootstrap_mean(selected, "top1", resamples=resamples, seed=seed),
            "mrr": clustered_bootstrap_mean(selected, "reciprocal_rank", resamples=resamples, seed=seed),
            "mean_latency_ms": float(np.mean([row["latency_ms"] for row in selected])),
            "total_model_calls": int(sum(row["model_calls"] for row in selected)),
        }
    comparisons = []
    for baseline in methods:
        if baseline == "counterfactual_replay":
            continue
        for metric in ("top1", "reciprocal_rank"):
            comparisons.append(
                paired_clustered_bootstrap_difference(
                    rows,
                    "counterfactual_replay",
                    baseline,
                    metric,
                    resamples=resamples,
                    seed=seed,
                )
            )
    return {"methods": metrics, "paired_differences": comparisons}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="B-Single outcomes JSONL")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model", required=True, choices=("resnet50", "vit_b_16"))
    parser.add_argument("--image-root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260802)
    parser.add_argument("--include-ig", action="store_true")
    parser.add_argument("--ig-steps", type=int, default=32)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    traces = load_traces(args.input, accepted_only=True)
    if args.limit is not None:
        traces = traces[: args.limit]
    if not traces:
        raise RuntimeError("no accepted B-Single traces found")
    raw_predictor = build_frozen_predictor(args.model, args.device)
    mismatched = sorted({trace.model_id for trace in traces if trace.model_id != raw_predictor.model_id})
    if mismatched:
        raise RuntimeError(f"trace model identifiers do not match evaluator: {mismatched}")
    first_source = load_source(traces[0], args.image_root)
    warmup_predictor(raw_predictor, first_source, args.warmup)
    predictor = TimedPredictor(raw_predictor, synchronize_for_predictor(raw_predictor))
    config = load_config(args.config)

    rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for trace in traces:
        trace_rows, trace_errors = evaluate_trace(
            trace,
            predictor,
            config,
            args.image_root,
            include_ig=args.include_ig,
            ig_steps=args.ig_steps,
        )
        rows.extend(trace_rows)
        errors.extend(trace_errors)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"rq1_{args.model}"
    write_jsonl(args.output_dir / f"{stem}_per_trace.jsonl", rows)
    write_csv(args.output_dir / f"{stem}_per_trace.csv", rows)
    summary = {
        "schema_version": "1.0",
        "rq": "RQ1",
        "scope": "development" if "dev" in args.input.name.lower() else "evaluation",
        "model": raw_predictor.model_id,
        "input": str(args.input.resolve()),
        "input_sha256": file_sha256(args.input),
        "config": str(args.config.resolve()),
        "config_sha256": file_sha256(args.config),
        "accepted_traces": len(traces),
        "include_ig": args.include_ig,
        "ig_steps": args.ig_steps if args.include_ig else None,
        "warmup_iterations": args.warmup,
        "random_baseline_seed": 20260802,
        "bootstrap_resamples": args.bootstrap_resamples,
        "bootstrap_seed": args.bootstrap_seed,
        "errors": errors,
        "results": summarize(rows, args.bootstrap_resamples, args.bootstrap_seed),
    }
    write_json(args.output_dir / f"{stem}_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
