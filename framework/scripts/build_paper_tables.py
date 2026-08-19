#!/usr/bin/env python3
"""Build compact CSV and LaTeX result tables from RQ1--RQ4 summary JSON files."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected object in {path}")
    return value


def short_model(value: str) -> str:
    if "resnet50" in value:
        return "ResNet-50"
    if "vit_b_16" in value:
        return "ViT-B/16"
    return value


def estimate(value: dict[str, Any]) -> str:
    if not value.get("available"):
        return "--"
    return f"{value['estimate']:.3f} [{value['ci95_low']:.3f}, {value['ci95_high']:.3f}]"


def latex_escape(value: object) -> str:
    return str(value).replace("_", r"\_").replace("%", r"\%")


def write_simple_latex(path: Path, caption: str, label: str, headers: list[str], rows: list[list[object]]) -> None:
    columns = "l" * len(headers)
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{{latex_escape(caption)}}}",
        rf"\label{{{label}}}",
        rf"\begin{{tabular}}{{{columns}}}",
        r"\toprule",
        " & ".join(latex_escape(item) for item in headers) + r" \\",
        r"\midrule",
    ]
    lines.extend(" & ".join(latex_escape(item) for item in row) + r" \\" for row in rows)
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summaries", nargs="+", type=Path, required=True)
    parser.add_argument("--rq4", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    payloads = [load(path) for path in args.summaries]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    flat_rows: list[dict[str, Any]] = []
    latex_tables: dict[str, tuple[list[str], list[list[object]]]] = {}
    for rq in ("RQ1", "RQ2", "RQ3"):
        rq_payloads = [payload for payload in payloads if payload.get("rq") == rq]
        table_rows: list[list[object]] = []
        for payload in rq_payloads:
            model = short_model(str(payload.get("model", "unknown")))
            methods = payload.get("results", {}).get("methods", {}) if rq in {"RQ1", "RQ2"} else payload.get("results", {})
            for method, metrics in methods.items():
                if rq == "RQ1":
                    values = [model, method, estimate(metrics["top1"]), estimate(metrics["mrr"])]
                    headers = ["Model", "Method", "Top-1 (95% CI)", "MRR (95% CI)"]
                elif rq == "RQ2":
                    values = [
                        model, method, estimate(metrics["top1"]),
                        estimate(metrics["selected_pair_only_recovery"]),
                    ]
                    headers = ["Model", "Method", "Pair Top-1 (95% CI)", "Pair-only recovery (95% CI)"]
                else:
                    values = [
                        model, method, estimate(metrics["success"]),
                        estimate(metrics["perceptual_distance"]),
                    ]
                    headers = ["Model", "Method", "Recovery (95% CI)", "Perceptual distance (95% CI)"]
                table_rows.append(values)
                flat_rows.append({
                    "rq": rq,
                    "model": model,
                    "method": method,
                    "primary": values[2],
                    "secondary": values[3],
                })
        if table_rows:
            latex_tables[rq] = (headers, table_rows)

    if args.rq4:
        rq4 = load(args.rq4)
        rows = []
        for group in rq4.get("groups", []):
            if group.get("history_length") is not None:
                continue
            rows.append([
                group["component"], short_model(group["model"]), group["n"],
                f"{group['latency_ms']['median']:.2f}", f"{group['latency_ms']['p95']:.2f}",
                f"{group['model_calls']['median']:.1f}",
            ])
        if rows:
            latex_tables["RQ4"] = (
                ["Component", "Model", "n", "Median ms", "p95 ms", "Calls"], rows
            )

    with (args.output_dir / "paper_results_index.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["rq", "model", "method", "primary", "secondary"])
        writer.writeheader()
        writer.writerows(flat_rows)
    for rq, (headers, rows) in latex_tables.items():
        write_simple_latex(
            args.output_dir / f"{rq.lower()}_table.tex",
            f"{rq} development results. Replace development with held-out results for the paper.",
            f"tab:{rq.lower()}",
            headers,
            rows,
        )
    print(json.dumps({"output_dir": str(args.output_dir.resolve()), "tables": sorted(latex_tables)}, indent=2))


if __name__ == "__main__":
    main()
