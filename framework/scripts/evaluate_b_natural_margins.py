#!/usr/bin/env python3
"""Evaluate source/final target-class margins for EditTrace-XAI B-Natural traces.

The script reads the development B-Natural JSONL files, deterministically replays
logged edit actions, evaluates the matching frozen torchvision classifier, and
writes per-trace and aggregate audit artifacts.

Outputs (under artifacts/logs by default):
  - b_natural_margin_audit.json
  - b_natural_margin_audit.md
  - b_natural_margin_audit.csv
  - b_natural_margins_<model>.jsonl

This is a development-only audit. It does not evaluate repair performance and it
does not access the held-out test split.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from PIL import Image, ImageEnhance, ImageFilter


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Invalid JSON in {path}:{lineno}: {exc}") from exc
            if not isinstance(obj, dict):
                raise RuntimeError(f"Expected a JSON object in {path}:{lineno}")
            rows.append(obj)
    return rows


def trace_from_row(row: dict[str, Any]) -> dict[str, Any]:
    trace = row.get("trace")
    if isinstance(trace, dict):
        return trace
    return row


def actions_from_trace(trace: dict[str, Any]) -> list[dict[str, Any]]:
    actions = trace.get("actions", [])
    if not isinstance(actions, list):
        return []
    return [a for a in actions if isinstance(a, dict)]


def op_name(action: dict[str, Any]) -> str:
    for key in ("op_name", "operation", "op", "name", "type"):
        value = action.get(key)
        if isinstance(value, str) and value:
            return value.lower().replace("-", "_")
    raise ValueError(f"Action has no operation name: {action.get('action_id', '<unknown>')}")


def params(action: dict[str, Any]) -> dict[str, Any]:
    for key in ("params", "theta", "parameters"):
        value = action.get(key)
        if isinstance(value, dict):
            return value
    return {}


def finite_float(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Expected numeric {name}, got {value!r}")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"Non-finite {name}: {value!r}")
    return value


def first_value(mapping: dict[str, Any], keys: Iterable[str]) -> Any:
    for key in keys:
        if key in mapping:
            return mapping[key]
    return None


def normalized_crop_box(action: dict[str, Any], p: dict[str, Any]) -> tuple[float, float, float, float]:
    candidates: list[Any] = []
    for key in ("box", "crop_box", "normalized_box", "bbox"):
        if key in p:
            candidates.append(p[key])
    geometry = action.get("geometry")
    if isinstance(geometry, dict):
        for key in ("box", "crop_box", "normalized_box", "source_box"):
            if key in geometry:
                candidates.append(geometry[key])

    for box in candidates:
        if isinstance(box, dict):
            vals = [box.get(k) for k in ("left", "top", "right", "bottom")]
            if all(isinstance(v, (int, float)) for v in vals):
                return tuple(float(v) for v in vals)  # type: ignore[return-value]
            vals = [box.get(k) for k in ("x0", "y0", "x1", "y1")]
            if all(isinstance(v, (int, float)) for v in vals):
                return tuple(float(v) for v in vals)  # type: ignore[return-value]
        if isinstance(box, (list, tuple)) and len(box) == 4:
            return tuple(float(v) for v in box)  # type: ignore[return-value]

    # Fallback for side-fraction + anchor representation.
    frac = first_value(p, ("side_fraction", "fraction", "scale", "size_fraction"))
    anchor = first_value(p, ("anchor", "position"))
    if frac is not None:
        f = finite_float(frac, "crop side fraction")
        anchor = str(anchor or "center").lower().replace("-", "_")
        offsets = {
            "top_left": (0.0, 0.0),
            "top_right": (1.0 - f, 0.0),
            "bottom_left": (0.0, 1.0 - f),
            "bottom_right": (1.0 - f, 1.0 - f),
            "center": ((1.0 - f) / 2.0, (1.0 - f) / 2.0),
        }
        if anchor not in offsets:
            raise ValueError(f"Unknown crop anchor {anchor!r}")
        x0, y0 = offsets[anchor]
        return x0, y0, x0 + f, y0 + f

    raise ValueError("Crop action does not contain a supported normalized crop box")


def replay_action(image: Image.Image, action: dict[str, Any]) -> Image.Image:
    op = op_name(action)
    p = params(action)
    image = image.convert("RGB")

    if op in {"brightness", "adjust_brightness"}:
        factor = finite_float(first_value(p, ("factor", "brightness_factor", "value")), "brightness factor")
        return ImageEnhance.Brightness(image).enhance(factor).convert("RGB")

    if op in {"contrast", "adjust_contrast"}:
        factor = finite_float(first_value(p, ("factor", "contrast_factor", "value")), "contrast factor")
        return ImageEnhance.Contrast(image).enhance(factor).convert("RGB")

    if op in {"saturation", "color", "adjust_saturation"}:
        factor = finite_float(first_value(p, ("factor", "saturation_factor", "value")), "saturation factor")
        return ImageEnhance.Color(image).enhance(factor).convert("RGB")

    if op in {"gaussian_blur", "gaussianblur", "blur"}:
        sigma = finite_float(first_value(p, ("sigma", "radius", "std", "value")), "blur sigma")
        if sigma <= 0:
            return image.copy()
        return image.filter(ImageFilter.GaussianBlur(radius=sigma)).convert("RGB")

    if op in {"crop_resize", "crop", "crop_and_resize"}:
        x0, y0, x1, y1 = normalized_crop_box(action, p)
        if max(abs(x0), abs(y0), abs(x1), abs(y1)) > 1.000001:
            # Pixel coordinates are accepted as a fallback.
            left, top, right, bottom = x0, y0, x1, y1
        else:
            w, h = image.size
            left, top, right, bottom = x0 * w, y0 * h, x1 * w, y1 * h
        left = max(0, min(image.width - 1, round(left)))
        top = max(0, min(image.height - 1, round(top)))
        right = max(left + 1, min(image.width, round(right)))
        bottom = max(top + 1, min(image.height, round(bottom)))
        cropped = image.crop((left, top, right, bottom))
        resampling = getattr(Image, "Resampling", Image).BICUBIC
        return cropped.resize(image.size, resample=resampling).convert("RGB")

    if op in {"jpeg", "jpeg_compression", "compression"}:
        quality = int(round(finite_float(first_value(p, ("quality", "q", "value")), "JPEG quality")))
        quality = max(1, min(100, quality))
        subsampling = first_value(p, ("subsampling", "chroma_subsampling"))
        save_kwargs: dict[str, Any] = {"format": "JPEG", "quality": quality, "optimize": False, "progressive": False}
        if subsampling is not None:
            save_kwargs["subsampling"] = subsampling
        buffer = io.BytesIO()
        image.save(buffer, **save_kwargs)
        buffer.seek(0)
        with Image.open(buffer) as decoded:
            return decoded.convert("RGB").copy()

    raise ValueError(f"Unsupported operation {op!r}")


def replay_trace(trace: dict[str, Any]) -> tuple[Image.Image, Image.Image]:
    source_path = trace.get("source_path")
    if not isinstance(source_path, str) or not source_path:
        raise ValueError("Trace has no source_path")
    with Image.open(source_path) as img:
        source = img.convert("RGB").copy()
    final = source.copy()
    for action in actions_from_trace(trace):
        final = replay_action(final, action)
    return source, final


def model_bundle(model_name: str, device: str):
    import torch
    from torchvision.models import ResNet50_Weights, ViT_B_16_Weights, resnet50, vit_b_16

    if model_name == "resnet50":
        weights = ResNet50_Weights.IMAGENET1K_V2
        model = resnet50(weights=weights)
    elif model_name == "vit_b_16":
        weights = ViT_B_16_Weights.IMAGENET1K_V1
        model = vit_b_16(weights=weights)
    else:
        raise ValueError(f"Unsupported model {model_name!r}")
    model.eval().to(device)
    return model, weights.transforms(), torch


def batch_logits(model, preprocess, torch_mod, images: list[Image.Image], device: str, batch_size: int):
    outputs = []
    with torch_mod.inference_mode():
        for start in range(0, len(images), batch_size):
            batch = torch_mod.stack([preprocess(im) for im in images[start:start + batch_size]]).to(device)
            logits = model(batch).detach().cpu()
            outputs.extend(logits)
    return outputs


def target_margin(logits, target: int) -> float:
    import torch
    if target < 0 or target >= int(logits.numel()):
        raise ValueError(f"Target label {target} outside logit range")
    target_value = float(logits[target].item())
    mask = torch.ones_like(logits, dtype=torch.bool)
    mask[target] = False
    competitor = float(logits[mask].max().item())
    return target_value - competitor


def percentile(values: list[float], p: float) -> float:
    values = sorted(values)
    if len(values) == 1:
        return values[0]
    rank = (len(values) - 1) * p
    lo = math.floor(rank)
    hi = math.ceil(rank)
    if lo == hi:
        return values[lo]
    frac = rank - lo
    return values[lo] * (1 - frac) + values[hi] * frac


def stats(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"n": 0, "available": False}
    return {
        "n": len(values),
        "available": True,
        "min": min(values),
        "p05": percentile(values, 0.05),
        "p25": percentile(values, 0.25),
        "median": statistics.median(values),
        "mean": statistics.fmean(values),
        "p75": percentile(values, 0.75),
        "p95": percentile(values, 0.95),
        "max": max(values),
    }


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def evaluate_model(project_root: Path, logs_dir: Path, model_name: str, device: str, batch_size: int) -> dict[str, Any]:
    input_path = project_root / "artifacts" / "traces" / f"{model_name}_b_natural_dev.jsonl"
    rows = read_jsonl(input_path)
    model, preprocess, torch_mod = model_bundle(model_name, device)

    sources: list[Image.Image] = []
    finals: list[Image.Image] = []
    valid_meta: list[tuple[int, dict[str, Any], dict[str, Any]]] = []
    errors: list[dict[str, Any]] = []

    for index, row in enumerate(rows):
        trace = trace_from_row(row)
        try:
            source, final = replay_trace(trace)
            sources.append(source)
            finals.append(final)
            valid_meta.append((index, row, trace))
        except Exception as exc:
            errors.append({
                "index": index,
                "trace_id": trace.get("trace_id"),
                "source_id": trace.get("source_id"),
                "error_type": type(exc).__name__,
                "error": str(exc),
            })

    source_logits = batch_logits(model, preprocess, torch_mod, sources, device, batch_size)
    final_logits = batch_logits(model, preprocess, torch_mod, finals, device, batch_size)

    out_path = logs_dir / f"b_natural_margins_{model_name}.jsonl"
    per_trace: list[dict[str, Any]] = []
    initial_values: list[float] = []
    final_values: list[float] = []
    delta_values: list[float] = []
    shifted = 0
    source_not_positive = 0
    final_positive = 0
    final_nonpositive = 0
    predicted_source_correct = 0

    with out_path.open("w", encoding="utf-8") as fh:
        for (index, row, trace), src_logit, fin_logit in zip(valid_meta, source_logits, final_logits):
            label = trace.get("ground_truth_label", row.get("ground_truth_label"))
            if not isinstance(label, int):
                raise RuntimeError(f"Trace {trace.get('trace_id')} has no integer ground_truth_label")
            initial = target_margin(src_logit, label)
            final = target_margin(fin_logit, label)
            delta = final - initial
            is_shift = initial > 0 and final <= 0
            src_pred = int(src_logit.argmax().item())
            fin_pred = int(fin_logit.argmax().item())

            initial_values.append(initial)
            final_values.append(final)
            delta_values.append(delta)
            shifted += int(is_shift)
            source_not_positive += int(initial <= 0)
            final_positive += int(final > 0)
            final_nonpositive += int(final <= 0)
            predicted_source_correct += int(src_pred == label)

            result = {
                "model": model_name,
                "index": index,
                "trace_id": trace.get("trace_id"),
                "source_id": trace.get("source_id"),
                "source_path": trace.get("source_path"),
                "ground_truth_label": label,
                "source_prediction": src_pred,
                "final_prediction": fin_pred,
                "initial_margin": initial,
                "final_margin": final,
                "margin_change": delta,
                "shifted": is_shift,
                "history_length": len(actions_from_trace(trace)),
            }
            per_trace.append(result)
            fh.write(json.dumps(result, sort_keys=True) + "\n")

    total = len(rows)
    valid = len(per_trace)
    return {
        "model": model_name,
        "input": str(input_path),
        "input_sha256": sha256(input_path),
        "per_trace_output": str(out_path),
        "total_rows": total,
        "replay_evaluated": valid,
        "replay_errors": len(errors),
        "error_details": errors,
        "source_prediction_correct": predicted_source_correct,
        "source_margin_nonpositive": source_not_positive,
        "final_margin_positive": final_positive,
        "final_margin_nonpositive": final_nonpositive,
        "shifted": shifted,
        "not_shifted": valid - shifted,
        "shift_rate": shifted / valid if valid else None,
        "initial_margin": stats(initial_values),
        "final_margin": stats(final_values),
        "margin_change": stats(delta_values),
    }


def write_csv(path: Path, reports: list[dict[str, Any]]) -> None:
    fields = [
        "model", "total_rows", "replay_evaluated", "replay_errors",
        "source_prediction_correct", "source_margin_nonpositive",
        "shifted", "not_shifted", "shift_rate",
        "initial_margin_mean", "initial_margin_median",
        "final_margin_mean", "final_margin_median",
        "margin_change_mean", "margin_change_median",
    ]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for r in reports:
            writer.writerow({
                "model": r["model"],
                "total_rows": r["total_rows"],
                "replay_evaluated": r["replay_evaluated"],
                "replay_errors": r["replay_errors"],
                "source_prediction_correct": r["source_prediction_correct"],
                "source_margin_nonpositive": r["source_margin_nonpositive"],
                "shifted": r["shifted"],
                "not_shifted": r["not_shifted"],
                "shift_rate": r["shift_rate"],
                "initial_margin_mean": r["initial_margin"].get("mean"),
                "initial_margin_median": r["initial_margin"].get("median"),
                "final_margin_mean": r["final_margin"].get("mean"),
                "final_margin_median": r["final_margin"].get("median"),
                "margin_change_mean": r["margin_change"].get("mean"),
                "margin_change_median": r["margin_change"].get("median"),
            })


def write_markdown(path: Path, reports: list[dict[str, Any]], device: str, batch_size: int) -> None:
    lines = [
        "# B-Natural Margin Audit", "",
        "Development-only audit. The held-out test split was not accessed.", "",
        f"- Device: `{device}`",
        f"- Batch size: `{batch_size}`", "",
        "## Summary", "",
        "| Model | Traces | Replay errors | Source-correct | Shifted | Shift rate | Non-shifted |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in reports:
        rate = "N/A" if r["shift_rate"] is None else f"{100*r['shift_rate']:.2f}%"
        lines.append(
            f"| {r['model']} | {r['replay_evaluated']} | {r['replay_errors']} | "
            f"{r['source_prediction_correct']} | {r['shifted']} | {rate} | {r['not_shifted']} |"
        )
    lines.extend(["", "## Margin distributions", ""])
    for r in reports:
        lines.extend([
            f"### {r['model']}", "",
            f"- Initial margin: `{json.dumps(r['initial_margin'], sort_keys=True)}`",
            f"- Final margin: `{json.dumps(r['final_margin'], sort_keys=True)}`",
            f"- Margin change: `{json.dumps(r['margin_change'], sort_keys=True)}`",
            f"- Source margins <= 0: **{r['source_margin_nonpositive']}**",
            f"- Final margins <= 0: **{r['final_margin_nonpositive']}**",
            "",
        ])
        if r["replay_errors"]:
            lines.append("Replay errors were recorded; inspect the JSON report before closing the audit.")
            lines.append("")
    lines.extend([
        "## Interpretation boundary", "",
        "These quantities determine the development B-Natural population available for repair evaluation. "
        "They are generation/audit statistics, not RQ1--RQ4 test results.", "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--models", nargs="+", default=["resnet50", "vit_b_16"], choices=("resnet50", "vit_b_16"))
    args = parser.parse_args()

    project_root = args.project_root.expanduser().resolve()
    logs_dir = project_root / "artifacts" / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    reports = [evaluate_model(project_root, logs_dir, model, args.device, args.batch_size) for model in args.models]

    json_path = logs_dir / "b_natural_margin_audit.json"
    csv_path = logs_dir / "b_natural_margin_audit.csv"
    md_path = logs_dir / "b_natural_margin_audit.md"

    payload = {
        "schema_version": "1.0",
        "scope": "development-only B-Natural source/final margin audit",
        "device": args.device,
        "batch_size": args.batch_size,
        "models": reports,
        "status": "COMPLETE" if all(r["replay_errors"] == 0 and r["source_margin_nonpositive"] == 0 for r in reports) else "REVIEW_REQUIRED",
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    write_csv(csv_path, reports)
    write_markdown(md_path, reports, args.device, args.batch_size)

    print(json.dumps({
        "json": str(json_path),
        "markdown": str(md_path),
        "csv": str(csv_path),
        "status": payload["status"],
        "summary": {
            r["model"]: {
                "traces": r["replay_evaluated"],
                "shifted": r["shifted"],
                "shift_rate": r["shift_rate"],
                "replay_errors": r["replay_errors"],
                "source_margin_nonpositive": r["source_margin_nonpositive"],
            }
            for r in reports
        },
    }, indent=2))


if __name__ == "__main__":
    main()
