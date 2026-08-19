from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .engine import ReplayEngine
from .generators import (
    TraceGenerator,
    generate_interaction,
    generate_single,
    load_config,
    oracle_subset,
    read_trace_jsonl,
    write_jsonl,
)
from .manifest import (
    audit_eligible_manifests,
    build_flat_manifest,
    build_imagefolder_manifest,
    read_jsonl,
    select_model_eligible,
    write_jsonl as write_manifest,
)
from .modeling import TorchvisionPredictor
from .preflight import preflight_report


def synthetic_image(size: int = 96) -> Image.Image:
    y, x = np.mgrid[0:size, 0:size]
    checker = ((x // 12 + y // 12) % 2) * 70
    array = np.stack([
        np.clip(80 + x + checker, 0, 255),
        np.clip(60 + y + checker, 0, 255),
        np.clip(180 - x // 2 + checker, 0, 255),
    ], axis=-1).astype(np.uint8)
    return Image.fromarray(array, mode="RGB")


def command_smoke(args: argparse.Namespace) -> None:
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    source = synthetic_image()
    source_path = output / "source.png"
    source.save(source_path)
    generator = TraceGenerator.from_yaml(args.config)
    traces = []
    engine = ReplayEngine()
    for ordinal in range(4):
        trace = generator.natural(
            source,
            source_id="synthetic/checker",
            source_path=str(source_path),
            ground_truth_label=0,
            model_id="fixture:none",
            ordinal=ordinal,
        )
        result = engine.replay(source, trace, verify_original_lineage=True)
        result.image.save(output / f"trace-{ordinal}-final.png")
        engine.remove_action(source, trace, trace.actions[0].action_id).image.save(
            output / f"trace-{ordinal}-remove-a0.png"
        )
        engine.attenuate_action(source, trace, trace.actions[0].action_id, 0.5).image.save(
            output / f"trace-{ordinal}-attenuate-a0.png"
        )
        traces.append(trace)
    write_jsonl(output / "b_natural.jsonl", traces)
    write_jsonl(output / "b_oracle.jsonl", oracle_subset(traces, count=2))
    summary = {
        "synthetic_only": True,
        "trace_count": len(traces),
        "history_lengths": [len(trace.actions) for trace in traces],
        "schema_version": "1.0",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


def command_build_manifest(args: argparse.Namespace) -> None:
    if args.layout == "imagefolder":
        entries = build_imagefolder_manifest(args.root, args.limit)
    else:
        if args.labels is None:
            raise ValueError("--labels is required for --layout flat")
        entries = build_flat_manifest(
            args.root,
            args.labels,
            labels_one_indexed=not args.labels_zero_indexed,
            limit=args.limit,
        )
    write_manifest(args.output, entries)
    print(json.dumps({"entries": len(entries), "output": str(Path(args.output))}, indent=2))


def command_generate_natural(args: argparse.Namespace) -> None:
    entries = read_jsonl(args.manifest)
    if args.split != "all":
        entries = [entry for entry in entries if entry.split == args.split]
    if not entries:
        raise ValueError(f"manifest contains no records for split={args.split!r}")
    generator = TraceGenerator.from_yaml(args.config)
    traces = []
    for ordinal, entry in enumerate(entries[: args.count]):
        with Image.open(entry.path) as image:
            source = image.convert("RGB")
        traces.append(generator.natural(
            source,
            source_id=entry.source_id,
            source_path=entry.path,
            ground_truth_label=entry.ground_truth_label,
            ordinal=ordinal,
        ))
    write_jsonl(args.output, traces)
    print(json.dumps({"traces": len(traces), "output": str(Path(args.output))}, indent=2))


def command_select_eligible(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    predictor = TorchvisionPredictor(args.model, args.weights, args.device)
    records = read_jsonl(args.manifest)
    data = config["data"]
    selected, summary = select_model_eligible(
        records,
        predictor,
        seed=int(config["global_seed"]),
        development_candidate_pool=int(data["development_candidate_pool"]),
        development_count=int(data["development_eligible_per_model"]),
        test_count=int(data["test_eligible_per_model"]),
    )
    write_manifest(args.output, selected)
    meta_path = Path(args.output).with_suffix(Path(args.output).suffix + ".meta.json")
    meta_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({**summary, "output": str(Path(args.output))}, indent=2))


def command_generate_controlled(args: argparse.Namespace) -> None:
    entries = [entry for entry in read_jsonl(args.manifest) if entry.split == args.split]
    if not entries:
        raise ValueError(f"manifest contains no records for split={args.split!r}")
    generator = TraceGenerator.from_yaml(args.config)
    predictor = TorchvisionPredictor(args.model, args.weights, args.device)
    outcomes = []
    for entry in entries[: args.count]:
        if entry.ground_truth_label is None:
            raise ValueError("controlled generators require ground_truth_label")
        if entry.model_id is not None and entry.model_id != predictor.model_id:
            raise ValueError(
                f"manifest model_id {entry.model_id!r} does not match predictor {predictor.model_id!r}"
            )
        with Image.open(entry.path) as image:
            source = image.convert("RGB")
        kwargs = {
            "source_id": entry.source_id,
            "target": entry.ground_truth_label,
            "predictor": predictor,
            "source_path": entry.path,
        }
        if args.benchmark == "B-Single":
            outcome = generate_single(generator, source, **kwargs)
        else:
            outcome = generate_interaction(generator, source, **kwargs)
        outcomes.append(outcome)
    write_jsonl(args.output, outcomes)
    accepted = sum(outcome.accepted for outcome in outcomes)
    print(json.dumps({
        "benchmark": args.benchmark,
        "requested": len(outcomes),
        "accepted": accepted,
        "failed": len(outcomes) - accepted,
        "output": str(Path(args.output)),
    }, indent=2))


def command_generate_oracle(args: argparse.Namespace) -> None:
    traces = read_trace_jsonl(args.input)
    selected = oracle_subset(traces, count=args.count)
    write_jsonl(args.output, selected)
    print(json.dumps({"traces": len(selected), "output": str(Path(args.output))}, indent=2))


def command_preflight(args: argparse.Namespace) -> None:
    report = preflight_report(
        config_path=args.config,
        data_root=args.data_root,
        layout=args.layout,
        labels_file=args.labels,
    )
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(2)


def command_audit_eligible(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    manifests = {
        Path(path).stem: read_jsonl(path)
        for path in args.manifests
    }
    data = config["data"]
    report = audit_eligible_manifests(
        manifests,
        expected_development=int(data["development_eligible_per_model"]),
        expected_test=int(data["test_eligible_per_model"]),
    )
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(2)


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="edittrace")
    subparsers = root.add_subparsers(dest="command", required=True)

    smoke = subparsers.add_parser("smoke", help="run the synthetic development smoke pipeline")
    smoke.add_argument("--output-dir", required=True)
    smoke.add_argument("--config", default=str(Path(__file__).parents[2] / "configs" / "development.yaml"))
    smoke.set_defaults(func=command_smoke)

    manifest = subparsers.add_parser("build-manifest", help="build an ImageFolder JSONL manifest")
    manifest.add_argument("--root", required=True)
    manifest.add_argument("--layout", choices=["imagefolder", "flat"], default="imagefolder")
    manifest.add_argument("--labels", help="one label per sorted image; required for flat layout")
    manifest.add_argument("--labels-zero-indexed", action="store_true")
    manifest.add_argument("--output", required=True)
    manifest.add_argument("--limit", type=int)
    manifest.set_defaults(func=command_build_manifest)

    natural = subparsers.add_parser("generate-natural", help="generate B-Natural traces")
    natural.add_argument("--manifest", required=True)
    natural.add_argument("--config", required=True)
    natural.add_argument("--output", required=True)
    natural.add_argument("--count", type=int, default=500)
    natural.add_argument(
        "--split", choices=["development", "test", "unassigned", "all"], default="development"
    )
    natural.set_defaults(func=command_generate_natural)

    eligible = subparsers.add_parser(
        "select-eligible", help="create model-specific development/test eligibility manifests"
    )
    eligible.add_argument("--manifest", required=True)
    eligible.add_argument("--config", required=True)
    eligible.add_argument("--model", choices=["resnet50", "vit_b_16"], required=True)
    eligible.add_argument("--weights", default="DEFAULT")
    eligible.add_argument("--device", default="cpu")
    eligible.add_argument("--output", required=True)
    eligible.set_defaults(func=command_select_eligible)

    controlled = subparsers.add_parser(
        "generate-controlled", help="generate B-Single or B-Interaction with full failure accounting"
    )
    controlled.add_argument("--manifest", required=True)
    controlled.add_argument("--config", required=True)
    controlled.add_argument("--benchmark", choices=["B-Single", "B-Interaction"], required=True)
    controlled.add_argument("--model", choices=["resnet50", "vit_b_16"], required=True)
    controlled.add_argument("--weights", default="DEFAULT")
    controlled.add_argument("--device", default="cpu")
    controlled.add_argument("--split", choices=["development", "test"], default="development")
    controlled.add_argument("--count", type=int, default=500)
    controlled.add_argument("--output", required=True)
    controlled.set_defaults(func=command_generate_controlled)

    oracle = subparsers.add_parser("generate-oracle", help="select stable B-Oracle traces")
    oracle.add_argument("--input", required=True, help="B-Natural trace JSONL")
    oracle.add_argument("--output", required=True)
    oracle.add_argument("--count", type=int, default=500)
    oracle.set_defaults(func=command_generate_oracle)

    preflight = subparsers.add_parser("preflight", help="check environment and dataset readiness")
    preflight.add_argument("--config", required=True)
    preflight.add_argument("--data-root")
    preflight.add_argument("--layout", choices=["imagefolder", "flat"], default="imagefolder")
    preflight.add_argument("--labels")
    preflight.set_defaults(func=command_preflight)

    audit = subparsers.add_parser(
        "audit-eligible", help="verify counts and cross-model development/test isolation"
    )
    audit.add_argument("--config", required=True)
    audit.add_argument("--manifests", nargs="+", required=True)
    audit.set_defaults(func=command_audit_eligible)
    return root


def main() -> None:
    args = parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
