#!/usr/bin/env python3
"""Aggregate RQ4 latency and model-call results emitted by RQ1--RQ3 evaluators."""

from __future__ import annotations

import argparse
import csv
import json
import platform
from collections import defaultdict
from pathlib import Path
from typing import Any

from edittrace_xai.evaluation import descriptive_stats, read_jsonl, write_json


PRIMARY_METHODS = {
    "RQ1": "counterfactual_replay",
    "RQ2": "difference_in_differences",
    "RQ3": "constrained_repair",
}
COMPONENT_NAMES = {"RQ1": "diagnosis", "RQ2": "interaction", "RQ3": "repair"}


def hardware_metadata() -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "processor": platform.processor(),
    }
    try:
        import torch

        metadata.update({
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version() if torch.cuda.is_available() else None,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        })
    except ImportError:
        metadata["torch"] = None
    return metadata


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str, int | None], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        rq = str(row.get("rq", ""))
        if row.get("method") != PRIMARY_METHODS.get(rq):
            continue
        model = str(row.get("model", "unknown"))
        component = COMPONENT_NAMES[rq]
        length = int(row["history_length"]) if row.get("history_length") is not None else None
        groups[(component, model, str(row["method"]), None)].append(row)
        groups[(component, model, str(row["method"]), length)].append(row)

    output: list[dict[str, Any]] = []
    for (component, model, method, length), selected in sorted(groups.items(), key=str):
        latency = descriptive_stats([float(row["latency_ms"]) for row in selected])
        calls = descriptive_stats([float(row["model_calls"]) for row in selected])
        output.append({
            "component": component,
            "model": model,
            "method": method,
            "history_length": length,
            "n": len(selected),
            "latency_ms": latency,
            "model_calls": calls,
        })
    return output


def write_csv(path: Path, groups: list[dict[str, Any]]) -> None:
    fields = [
        "component", "model", "method", "history_length", "n",
        "latency_median_ms", "latency_p95_ms", "latency_mean_ms",
        "model_calls_median", "model_calls_p95", "model_calls_mean",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for group in groups:
            writer.writerow({
                "component": group["component"],
                "model": group["model"],
                "method": group["method"],
                "history_length": group["history_length"],
                "n": group["n"],
                "latency_median_ms": group["latency_ms"]["median"],
                "latency_p95_ms": group["latency_ms"]["p95"],
                "latency_mean_ms": group["latency_ms"]["mean"],
                "model_calls_median": group["model_calls"]["median"],
                "model_calls_p95": group["model_calls"]["p95"],
                "model_calls_mean": group["model_calls"]["mean"],
            })


def write_markdown(path: Path, groups: list[dict[str, Any]], hardware: dict[str, Any]) -> None:
    lines = [
        "# RQ4 latency summary",
        "",
        "Times exclude model construction and warm-up. GPU evaluators synchronize before and after timed inference.",
        "",
        "| Component | Model | History | n | Median (ms) | p95 (ms) | Calls median | Calls p95 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for group in groups:
        if group["history_length"] is not None:
            continue
        lines.append(
            f"| {group['component']} | {group['model']} | all | {group['n']} | "
            f"{group['latency_ms']['median']:.3f} | {group['latency_ms']['p95']:.3f} | "
            f"{group['model_calls']['median']:.1f} | {group['model_calls']['p95']:.1f} |"
        )
    lines.extend(["", "## Hardware", "", "```json", json.dumps(hardware, indent=2, sort_keys=True), "```", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", type=Path, required=True, help="RQ1--RQ3 per-trace JSONL files")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    for path in args.inputs:
        rows.extend(read_jsonl(path))
    groups = aggregate(rows)
    if not groups:
        raise RuntimeError("no primary RQ1--RQ3 method records found")
    hardware = hardware_metadata()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "rq4_latency.csv", groups)
    write_markdown(args.output_dir / "rq4_latency.md", groups, hardware)
    payload = {
        "schema_version": "1.0",
        "rq": "RQ4",
        "inputs": [str(path.resolve()) for path in args.inputs],
        "timing_contract": {
            "model_initialization_excluded": True,
            "warmup_excluded": True,
            "gpu_synchronization": True,
            "repair_source_and_final_state_assumed_cached": True,
        },
        "hardware": hardware,
        "groups": groups,
    }
    write_json(args.output_dir / "rq4_latency.json", payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
