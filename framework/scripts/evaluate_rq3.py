#!/usr/bin/env python3
"""Evaluate RQ3 constrained repair on shifted B-Natural or B-Oracle traces."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from PIL import Image

from edittrace_xai.engine import ReplayEngine, ReplayError
from edittrace_xai.evaluation import (
    PerceptualMetric,
    TimedPredictor,
    build_frozen_predictor,
    clustered_bootstrap_mean,
    file_sha256,
    load_source,
    load_traces,
    normalized_parameter_magnitude,
    stable_seed,
    structural_similarity,
    synchronize_for_predictor,
    warmup_predictor,
    write_json,
    write_jsonl,
)
from edittrace_xai.generators import load_config
from edittrace_xai.modeling import TorchvisionPredictor, target_margin
from edittrace_xai.schema import TraceRecord


TIER_ORDER = {"1A-single-rollback": 0, "1B-attenuation": 1, "2-pair-rollback": 2}


@dataclass
class Candidate:
    candidate_id: str
    tier: str
    image: Image.Image
    margin: float
    action_ids: tuple[str, ...]
    normalized_movement: float
    evaluation_ms: float
    perceptual_distance: float | None = None

    @property
    def feasible(self) -> bool:
        return self.margin > 0


def evaluate_candidate(
    candidate_id: str,
    tier: str,
    image: Image.Image,
    action_ids: tuple[str, ...],
    movement: float,
    predictor: TimedPredictor,
    target: int,
    started_ns: int,
) -> Candidate:
    margin = target_margin(predictor.logits(image), target)
    return Candidate(
        candidate_id=candidate_id,
        tier=tier,
        image=image,
        margin=margin,
        action_ids=action_ids,
        normalized_movement=movement,
        evaluation_ms=(time.perf_counter_ns() - started_ns) / 1_000_000,
    )


def choose_candidate(
    candidates: list[Candidate],
    final_image: Image.Image,
    perceptual: PerceptualMetric,
) -> Candidate | None:
    feasible = [candidate for candidate in candidates if candidate.feasible]
    if not feasible:
        return None
    lowest_tier = min(TIER_ORDER[candidate.tier] for candidate in feasible)
    tier_candidates = [candidate for candidate in feasible if TIER_ORDER[candidate.tier] == lowest_tier]
    for candidate in tier_candidates:
        if candidate.perceptual_distance is None:
            candidate.perceptual_distance = perceptual(final_image, candidate.image)
    return min(
        tier_candidates,
        key=lambda candidate: (
            float(candidate.perceptual_distance),
            candidate.normalized_movement,
            -candidate.margin,
            candidate.candidate_id,
        ),
    )


def candidate_record(
    trace: TraceRecord,
    method: str,
    candidate: Candidate | None,
    final_image: Image.Image,
    perceptual: PerceptualMetric,
    *,
    latency_ms: float,
    model_calls: int,
    oracle_candidate: Candidate | None = None,
) -> dict[str, Any]:
    success = candidate is not None and candidate.feasible
    if success and candidate.perceptual_distance is None:
        candidate.perceptual_distance = perceptual(final_image, candidate.image)
    oracle_regret = None
    if success and oracle_candidate is not None and oracle_candidate.perceptual_distance is not None:
        oracle_regret = float(candidate.perceptual_distance) - float(oracle_candidate.perceptual_distance)
    return {
        "rq": "RQ3",
        "model": trace.model_id,
        "trace_id": trace.trace_id,
        "source_id": trace.source_id,
        "history_length": len(trace.actions),
        "method": method,
        "success": int(success),
        "candidate_id": candidate.candidate_id if candidate else None,
        "tier": candidate.tier if candidate else None,
        "restored_margin": candidate.margin if candidate else None,
        "perceptual_metric": perceptual.name,
        "perceptual_distance": candidate.perceptual_distance if success else None,
        "ssim": structural_similarity(final_image, candidate.image) if success else None,
        "action_count": len(candidate.action_ids) if candidate else None,
        "normalized_parameter_movement": candidate.normalized_movement if candidate else None,
        "oracle_cost_regret": oracle_regret,
        "latency_ms": latency_ms,
        "model_calls": model_calls,
    }


def evaluate_trace(
    trace: TraceRecord,
    predictor: TimedPredictor,
    config: Mapping[str, Any],
    perceptual: PerceptualMetric,
    image_root: Path | None,
    *,
    exhaustive_verification: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    if trace.ground_truth_label is None:
        raise ValueError(f"trace {trace.trace_id} has no ground-truth label")
    source = load_source(trace, image_root)
    engine = ReplayEngine()
    try:
        full = engine.replay(source, trace)
    except ReplayError as exc:
        return [], {"trace_id": trace.trace_id, "stage": "full_replay", "code": exc.code}
    source_margin = target_margin(predictor.logits(source), trace.ground_truth_label)
    final_margin = target_margin(predictor.logits(full.image), trace.ground_truth_label)
    if not (source_margin > 0 and final_margin <= 0):
        return [], None

    trace_snapshot = predictor.snapshot()
    perceptual_start = perceptual.elapsed_ms
    search_start = time.perf_counter_ns()
    singles: dict[str, Candidate] = {}
    candidates: list[Candidate] = []
    errors: list[dict[str, Any]] = []

    for action in trace.actions:
        started = time.perf_counter_ns()
        try:
            replay = engine.remove_action(source, trace, action.action_id)
            candidate = evaluate_candidate(
                f"rollback:{action.action_id}",
                "1A-single-rollback",
                replay.image,
                (action.action_id,),
                1.0,
                predictor,
                trace.ground_truth_label,
                started,
            )
            singles[action.action_id] = candidate
            candidates.append(candidate)
        except ReplayError as exc:
            errors.append({"action_id": action.action_id, "code": exc.code})

    has_single = any(candidate.feasible for candidate in singles.values())
    attenuation_candidates: list[Candidate] = []
    steps = int(config["repair"]["attenuation_steps"])
    if exhaustive_verification or not has_single:
        for action in trace.actions:
            for step in range(1, steps + 1):
                alpha = step / steps
                started = time.perf_counter_ns()
                try:
                    replay = engine.attenuate_action(source, trace, action.action_id, alpha)
                    candidate = evaluate_candidate(
                        f"attenuate:{action.action_id}:{step}/{steps}",
                        "1B-attenuation",
                        replay.image,
                        (action.action_id,),
                        alpha,
                        predictor,
                        trace.ground_truth_label,
                        started,
                    )
                    attenuation_candidates.append(candidate)
                    candidates.append(candidate)
                except ReplayError as exc:
                    errors.append({"action_id": action.action_id, "alpha": alpha, "code": exc.code})

    has_attenuation = any(candidate.feasible for candidate in attenuation_candidates)
    pair_candidates: list[Candidate] = []
    if exhaustive_verification or (not has_single and not has_attenuation):
        for first, second in itertools.combinations(trace.actions, 2):
            started = time.perf_counter_ns()
            try:
                replay = engine.remove_pair(source, trace, first.action_id, second.action_id)
                candidate = evaluate_candidate(
                    f"pair:{first.action_id}+{second.action_id}",
                    "2-pair-rollback",
                    replay.image,
                    (first.action_id, second.action_id),
                    2.0,
                    predictor,
                    trace.ground_truth_label,
                    started,
                )
                pair_candidates.append(candidate)
                candidates.append(candidate)
            except ReplayError as exc:
                errors.append({"pair": [first.action_id, second.action_id], "code": exc.code})

    proposed = choose_candidate(candidates, full.image, perceptual)
    oracle = choose_candidate(candidates, full.image, perceptual) if exhaustive_verification else None
    search_ms = (time.perf_counter_ns() - search_start) / 1_000_000
    search_calls = predictor.calls - trace_snapshot[0]

    last = singles.get(trace.actions[-1].action_id)
    magnitude_action = max(
        trace.actions,
        key=lambda action: (normalized_parameter_magnitude(action, config), -action.index),
    )
    magnitude = singles.get(magnitude_action.action_id)
    rows = [
        candidate_record(
            trace,
            "constrained_repair",
            proposed,
            full.image,
            perceptual,
            latency_ms=search_ms,
            model_calls=search_calls,
            oracle_candidate=oracle,
        ),
        candidate_record(
            trace,
            "last_action_rollback",
            last,
            full.image,
            perceptual,
            latency_ms=last.evaluation_ms if last else 0.0,
            model_calls=1 if last else 0,
            oracle_candidate=oracle,
        ),
        candidate_record(
            trace,
            "max_magnitude_rollback",
            magnitude,
            full.image,
            perceptual,
            latency_ms=magnitude.evaluation_ms if magnitude else 0.0,
            model_calls=1 if magnitude else 0,
            oracle_candidate=oracle,
        ),
    ]
    if exhaustive_verification:
        rows.append(
            candidate_record(
                trace,
                "exhaustive_oracle",
                oracle,
                full.image,
                perceptual,
                latency_ms=search_ms,
                model_calls=search_calls,
                oracle_candidate=oracle,
            )
        )
    for row in rows:
        row.update({
            "source_margin": source_margin,
            "final_margin": final_margin,
            "valid_candidates": len(candidates),
            "candidate_errors": len(errors),
            "perceptual_time_ms": perceptual.elapsed_ms - perceptual_start,
        })
    error = {"trace_id": trace.trace_id, "candidate_errors": errors} if errors else None
    return rows, error


def is_shifted_trace(
    trace: TraceRecord,
    predictor: TimedPredictor,
    image_root: Path | None,
) -> bool:
    if trace.ground_truth_label is None:
        return False
    source = load_source(trace, image_root)
    final = ReplayEngine().replay(source, trace).image
    return (
        target_margin(predictor.logits(source), trace.ground_truth_label) > 0
        and target_margin(predictor.logits(final), trace.ground_truth_label) <= 0
    )


def deterministic_shifted_oracle_subset(
    traces: list[TraceRecord],
    limit: int,
    predictor: TimedPredictor,
    image_root: Path | None,
) -> list[TraceRecord]:
    ordered = sorted(
        traces,
        key=lambda trace: (stable_seed(20260801, "B-Oracle-shifted", trace.trace_id), trace.trace_id),
    )
    selected: list[TraceRecord] = []
    for trace in ordered:
        if is_shifted_trace(trace, predictor, image_root):
            selected.append(trace)
            if len(selected) == limit:
                break
    if len(selected) < limit:
        raise RuntimeError(f"only {len(selected)} shifted B-Oracle traces are available; requested {limit}")
    return selected


def summarize(rows: list[dict[str, Any]], resamples: int, seed: int) -> dict[str, Any]:
    methods = sorted({row["method"] for row in rows})
    results: dict[str, Any] = {}
    for method in methods:
        selected = [row for row in rows if row["method"] == method]
        result: dict[str, Any] = {
            "success": clustered_bootstrap_mean(selected, "success", resamples=resamples, seed=seed),
            "mean_latency_ms": float(np.mean([row["latency_ms"] for row in selected])),
            "total_model_calls": int(sum(row["model_calls"] for row in selected)),
        }
        for metric in ("perceptual_distance", "ssim", "action_count", "normalized_parameter_movement", "oracle_cost_regret"):
            available = [row for row in selected if row.get(metric) is not None]
            result[metric] = (
                clustered_bootstrap_mean(available, metric, resamples=resamples, seed=seed)
                if available else {"available": False, "n_clusters": 0}
            )
        results[method] = result
    return results


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "model", "trace_id", "source_id", "history_length", "method", "success", "tier",
        "restored_margin", "perceptual_metric", "perceptual_distance", "ssim", "action_count",
        "normalized_parameter_movement", "oracle_cost_regret", "latency_ms", "model_calls",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fields})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="B-Natural or B-Oracle JSONL")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model", required=True, choices=("resnet50", "vit_b_16"))
    parser.add_argument("--image-root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--perceptual-metric", choices=("lpips", "l1"), default="lpips")
    parser.add_argument("--exhaustive-verification", action="store_true")
    parser.add_argument("--oracle-limit", type=int, default=50)
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260802)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    traces = load_traces(args.input)
    if not traces:
        raise RuntimeError("no traces found")
    raw_predictor = build_frozen_predictor(args.model, args.device)
    warmup_predictor(raw_predictor, load_source(traces[0], args.image_root), args.warmup)
    predictor = TimedPredictor(raw_predictor, synchronize_for_predictor(raw_predictor))
    traces = [trace.model_copy(update={"model_id": raw_predictor.model_id}) for trace in traces]
    if args.exhaustive_verification:
        traces = deterministic_shifted_oracle_subset(
            traces, args.oracle_limit, predictor, args.image_root
        )
    if args.limit is not None:
        traces = traces[: args.limit]
    perceptual = PerceptualMetric(args.perceptual_metric, args.device)
    config = load_config(args.config)

    rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for trace in traces:
        trace_rows, trace_error = evaluate_trace(
            trace,
            predictor,
            config,
            perceptual,
            args.image_root,
            exhaustive_verification=args.exhaustive_verification,
        )
        rows.extend(trace_rows)
        if trace_error:
            errors.append(trace_error)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    mode = "oracle" if args.exhaustive_verification else "natural"
    stem = f"rq3_{args.model}_{mode}"
    write_jsonl(args.output_dir / f"{stem}_per_trace.jsonl", rows)
    write_csv(args.output_dir / f"{stem}_per_trace.csv", rows)
    shifted = len({row["trace_id"] for row in rows})
    summary = {
        "schema_version": "1.0",
        "rq": "RQ3",
        "scope": "development" if "dev" in args.input.name.lower() else "evaluation",
        "model": raw_predictor.model_id,
        "input": str(args.input.resolve()),
        "input_sha256": file_sha256(args.input),
        "config": str(args.config.resolve()),
        "config_sha256": file_sha256(args.config),
        "input_traces": len(traces),
        "shifted_traces_evaluated": shifted,
        "exhaustive_verification": args.exhaustive_verification,
        "oracle_limit": args.oracle_limit if args.exhaustive_verification else None,
        "perceptual_metric": args.perceptual_metric,
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
