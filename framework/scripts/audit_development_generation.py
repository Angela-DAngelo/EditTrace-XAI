#!/usr/bin/env python3
"""Audit EditTrace-XAI development benchmark generation artifacts.

Produces:
  - development_generation_audit.json
  - development_generation_audit.md
  - development_generation_audit.csv

The script is intentionally schema-tolerant: it inspects known EditTrace fields but
keeps missing quantities explicit instead of inventing values.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

try:
    import yaml  # type: ignore
except Exception:
    yaml = None


CONTROLLED_FILES = {
    ("resnet50", "B-Single"): "resnet50_B-Single_dev_outcomes.jsonl",
    ("resnet50", "B-Interaction"): "resnet50_B-Interaction_dev_outcomes.jsonl",
    ("vit_b_16", "B-Single"): "vit_b_16_B-Single_dev_outcomes.jsonl",
    ("vit_b_16", "B-Interaction"): "vit_b_16_B-Interaction_dev_outcomes.jsonl",
}

NATURAL_FILES = {
    "resnet50": "resnet50_b_natural_dev.jsonl",
    "vit_b_16": "vit_b_16_b_natural_dev.jsonl",
}

ORACLE_FILES = {
    "resnet50": "resnet50_b_oracle_dev.jsonl",
    "vit_b_16": "vit_b_16_b_oracle_dev.jsonl",
}

ELIGIBLE_FILES = {
    "resnet50": "resnet50_eligible.jsonl",
    "vit_b_16": "vit_b_16_eligible.jsonl",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Invalid JSON in {path}:{lineno}: {exc}") from exc
            if not isinstance(value, dict):
                raise RuntimeError(f"Expected JSON object in {path}:{lineno}")
            rows.append(value)
    return rows


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def finite_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def get_nested(obj: dict[str, Any], paths: Iterable[tuple[str, ...]]) -> Any:
    for path in paths:
        cur: Any = obj
        ok = True
        for key in path:
            if not isinstance(cur, dict) or key not in cur:
                ok = False
                break
            cur = cur[key]
        if ok:
            return cur
    return None


def trace_from_row(row: dict[str, Any]) -> dict[str, Any] | None:
    if isinstance(row.get("trace"), dict):
        return row["trace"]
    # B-Natural and B-Oracle may already be bare traces.
    if isinstance(row.get("actions"), list) or "trace_id" in row:
        return row
    return None


def actions_from_trace(trace: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not trace:
        return []
    actions = trace.get("actions", [])
    return [a for a in actions if isinstance(a, dict)] if isinstance(actions, list) else []


def op_name(action: dict[str, Any]) -> str:
    for key in ("op_name", "operation", "op", "name", "type"):
        value = action.get(key)
        if isinstance(value, str) and value:
            return value
    return "UNKNOWN"


def action_id(action: dict[str, Any]) -> str | None:
    value = action.get("action_id")
    return value if isinstance(value, str) else None


def index_for_action(trace: dict[str, Any], wanted_id: str | None) -> int | None:
    if not wanted_id:
        return None
    for idx, action in enumerate(actions_from_trace(trace)):
        if action_id(action) == wanted_id:
            return idx
    return None


def basic_stats(values: list[float]) -> dict[str, Any]:
    values = [v for v in values if math.isfinite(v)]
    if not values:
        return {"n": 0, "available": False}
    values.sort()

    def percentile(p: float) -> float:
        if len(values) == 1:
            return values[0]
        rank = (len(values) - 1) * p
        lo = math.floor(rank)
        hi = math.ceil(rank)
        if lo == hi:
            return values[lo]
        frac = rank - lo
        return values[lo] * (1 - frac) + values[hi] * frac

    return {
        "n": len(values),
        "available": True,
        "min": min(values),
        "p05": percentile(0.05),
        "p25": percentile(0.25),
        "median": statistics.median(values),
        "mean": statistics.fmean(values),
        "p75": percentile(0.75),
        "p95": percentile(0.95),
        "max": max(values),
    }


def stringify_param(value: Any) -> str:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return json.dumps(value, sort_keys=True)
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def extract_margins(row: dict[str, Any]) -> tuple[float | None, float | None]:
    # Known and plausible key variants. Missing values stay missing.
    start = get_nested(
        row,
        [
            ("metrics", "source_margin"),
            ("metrics", "initial_margin"),
            ("metrics", "start_margin"),
            ("metrics", "original_margin"),
            ("metrics", "source"),
            ("source_margin",),
            ("initial_margin",),
        ],
    )
    final = get_nested(
        row,
        [
            ("metrics", "final_margin"),
            ("metrics", "final"),
            ("metrics", "end_margin"),
            ("final_margin",),
        ],
    )
    return finite_number(start), finite_number(final)


def collect_trace_distributions(rows: list[dict[str, Any]]) -> dict[str, Any]:
    length_counts: Counter[str] = Counter()
    op_counts: Counter[str] = Counter()
    op_position_counts: Counter[str] = Counter()
    designated_ops: Counter[str] = Counter()
    designated_positions: Counter[str] = Counter()
    pair_ops: Counter[str] = Counter()
    pair_positions: Counter[str] = Counter()
    params: dict[str, dict[str, Counter[str]]] = defaultdict(lambda: defaultdict(Counter))
    support_kinds: Counter[str] = Counter()
    replay_invalid_reasons: Counter[str] = Counter()
    invalid_rows = 0

    trace_ids: list[str] = []
    source_ids: list[str] = []
    source_hashes: list[str] = []
    final_hashes: list[str] = []

    initial_margins: list[float] = []
    final_margins: list[float] = []
    deltas: list[float] = []
    shifted = 0
    margin_pairs_available = 0

    for row in rows:
        trace = trace_from_row(row)
        if not trace:
            continue
        actions = actions_from_trace(trace)
        length_counts[str(len(actions))] += 1

        tid = trace.get("trace_id")
        sid = trace.get("source_id")
        sh = trace.get("source_hash")
        fh = trace.get("final_hash")
        if isinstance(tid, str): trace_ids.append(tid)
        if isinstance(sid, str): source_ids.append(sid)
        if isinstance(sh, str): source_hashes.append(sh)
        if isinstance(fh, str): final_hashes.append(fh)

        for idx, action in enumerate(actions):
            op = op_name(action)
            op_counts[op] += 1
            op_position_counts[f"{op}@{idx}"] += 1
            support = action.get("support")
            if isinstance(support, dict):
                kind = support.get("kind")
                if isinstance(kind, str):
                    support_kinds[kind] += 1
            action_params = action.get("params")
            if not isinstance(action_params, dict):
                action_params = action.get("theta") if isinstance(action.get("theta"), dict) else {}
            for key, value in action_params.items():
                params[op][str(key)][stringify_param(value)] += 1

        designated_id = trace.get("designated_action_id")
        if isinstance(designated_id, str):
            pos = index_for_action(trace, designated_id)
            if pos is not None:
                designated_positions[str(pos)] += 1
                designated_ops[op_name(actions[pos])] += 1

        pair = trace.get("designated_pair")
        if isinstance(pair, (list, tuple)) and len(pair) == 2:
            pos_a = index_for_action(trace, pair[0] if isinstance(pair[0], str) else None)
            pos_b = index_for_action(trace, pair[1] if isinstance(pair[1], str) else None)
            if pos_a is not None and pos_b is not None:
                p1, p2 = sorted((pos_a, pos_b))
                pair_positions[f"{p1}-{p2}"] += 1
                op_pair = sorted((op_name(actions[p1]), op_name(actions[p2])))
                pair_ops[" + ".join(op_pair)] += 1

        # Typed invalidity can appear in several places.
        invalid = False
        candidates = [
            row.get("replay_invalid_reason"),
            row.get("invalid_reason"),
            get_nested(row, [("metrics", "replay_invalid_reason"), ("metrics", "invalid_reason")]),
            trace.get("replay_invalid_reason"),
            trace.get("invalid_reason"),
        ]
        for candidate in candidates:
            if isinstance(candidate, str) and candidate:
                replay_invalid_reasons[candidate] += 1
                invalid = True
        if row.get("replay_valid") is False or trace.get("replay_valid") is False:
            replay_invalid_reasons["explicit_replay_valid_false"] += 1
            invalid = True
        if invalid:
            invalid_rows += 1

        initial, final = extract_margins(row)
        if initial is not None:
            initial_margins.append(initial)
        if final is not None:
            final_margins.append(final)
        if initial is not None and final is not None:
            margin_pairs_available += 1
            deltas.append(final - initial)
            if initial > 0 and final <= 0:
                shifted += 1

    def duplicates(values: list[str]) -> dict[str, Any]:
        counter = Counter(values)
        dup = {k: v for k, v in counter.items() if v > 1}
        return {
            "available": bool(values),
            "total_values": len(values),
            "unique_values": len(counter),
            "duplicate_keys": len(dup),
            "duplicate_occurrences_beyond_first": sum(v - 1 for v in dup.values()),
            "examples": dict(list(sorted(dup.items(), key=lambda kv: (-kv[1], kv[0])))[:20]),
        }

    param_out: dict[str, Any] = {}
    for op, fields in params.items():
        param_out[op] = {}
        for field, counter in fields.items():
            total = sum(counter.values())
            ordered = dict(counter.most_common())
            # Data-driven saturation indicator: concentration at the observed min/max numeric values.
            numeric_values: list[tuple[float, int, str]] = []
            for raw, count in counter.items():
                try:
                    decoded = json.loads(raw)
                except Exception:
                    decoded = None
                num = finite_number(decoded)
                if num is not None:
                    numeric_values.append((num, count, raw))
            extremes = None
            if numeric_values:
                min_v = min(x[0] for x in numeric_values)
                max_v = max(x[0] for x in numeric_values)
                min_count = sum(x[1] for x in numeric_values if x[0] == min_v)
                max_count = sum(x[1] for x in numeric_values if x[0] == max_v)
                extremes = {
                    "observed_min": min_v,
                    "observed_max": max_v,
                    "observed_min_count": min_count,
                    "observed_max_count": max_count,
                    "observed_extreme_fraction": (min_count + max_count) / total if total else None,
                    "note": "Observed-grid extreme concentration; not a pixel-saturation measurement.",
                }
            param_out[op][field] = {
                "n": total,
                "values": ordered,
                "observed_extremes": extremes,
            }

    return {
        "trace_count": sum(length_counts.values()),
        "history_length": dict(sorted(length_counts.items())),
        "operations": dict(op_counts.most_common()),
        "operation_by_position": dict(op_position_counts.most_common()),
        "designated_operation": dict(designated_ops.most_common()),
        "designated_position": dict(sorted(designated_positions.items())),
        "designated_pair_operations": dict(pair_ops.most_common()),
        "designated_pair_positions": dict(pair_positions.most_common()),
        "parameter_distributions": param_out,
        "support_kinds": dict(support_kinds.most_common()),
        "margins": {
            "initial": basic_stats(initial_margins),
            "final": basic_stats(final_margins),
            "final_minus_initial": basic_stats(deltas),
            "paired_available": margin_pairs_available,
            "shifted_count": shifted if margin_pairs_available else None,
            "shifted_fraction": (shifted / margin_pairs_available) if margin_pairs_available else None,
            "definition": "initial_margin > 0 and final_margin <= 0",
        },
        "replay_invalidity": {
            "invalid_trace_rows": invalid_rows,
            "typed_reasons": dict(replay_invalid_reasons.most_common()),
            "note": "Zero means no typed invalidity was found in inspected fields, not proof that invalidity is impossible.",
        },
        "duplicates": {
            "trace_id": duplicates(trace_ids),
            "source_id": duplicates(source_ids),
            "source_hash": duplicates(source_hashes),
            "final_hash": duplicates(final_hashes),
        },
    }


def controlled_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    accepted = sum(1 for r in rows if r.get("accepted") is True)
    rejected = sum(1 for r in rows if r.get("accepted") is not True)
    attempts = [int(r.get("attempts", 0) or 0) for r in rows]
    rejection_attempt_counts: Counter[str] = Counter()
    rejected_outcomes_with_reason: Counter[str] = Counter()
    max_attempt_rows = 0

    for row in rows:
        counts = row.get("rejection_counts")
        if isinstance(counts, dict):
            for reason, count in counts.items():
                if isinstance(reason, str) and isinstance(count, int):
                    rejection_attempt_counts[reason] += count
                    if count > 0:
                        rejected_outcomes_with_reason[reason] += 1
        # B-Single bound 50, B-Interaction bound 100 is inferred from observed maximum only.
        if attempts and row.get("attempts") == max(attempts):
            max_attempt_rows += 1

    total_rejection_attempts = sum(rejection_attempt_counts.values())
    reason_table = {}
    for reason, count in rejection_attempt_counts.most_common():
        reason_table[reason] = {
            "attempt_count": count,
            "fraction_of_rejected_attempts": count / total_rejection_attempts if total_rejection_attempts else None,
            "outcomes_with_reason": rejected_outcomes_with_reason[reason],
            "fraction_of_rejected_outcomes": rejected_outcomes_with_reason[reason] / rejected if rejected else None,
        }

    accepted_rows = [r for r in rows if r.get("accepted") is True]
    return {
        "outcomes": len(rows),
        "accepted": accepted,
        "rejected": rejected,
        "acceptance_yield": accepted / len(rows) if rows else None,
        "attempts": {
            **basic_stats([float(v) for v in attempts]),
            "total": sum(attempts),
            "observed_max": max(attempts) if attempts else None,
            "rows_at_observed_max": sum(1 for value in attempts if attempts and value == max(attempts)),
        },
        "rejection_reasons": reason_table,
        "accepted_trace_distributions": collect_trace_distributions(accepted_rows),
    }


def overlap_report(groups: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    fields = ("source_id", "source_hash", "trace_id", "final_hash")
    out: dict[str, Any] = {}
    for field in fields:
        group_values: dict[str, set[str]] = {}
        for name, rows in groups.items():
            vals: set[str] = set()
            for row in rows:
                trace = trace_from_row(row)
                if trace and isinstance(trace.get(field), str):
                    vals.add(trace[field])
                elif isinstance(row.get(field), str):
                    vals.add(row[field])
            group_values[name] = vals
        pairs = {}
        names = sorted(group_values)
        for i, left in enumerate(names):
            for right in names[i + 1 :]:
                intersection = group_values[left] & group_values[right]
                pairs[f"{left}__VS__{right}"] = {
                    "count": len(intersection),
                    "examples": sorted(intersection)[:20],
                }
        out[field] = pairs
    return out


def oracle_containment(natural: list[dict[str, Any]], oracle: list[dict[str, Any]]) -> dict[str, Any]:
    def keyset(rows: list[dict[str, Any]], field: str) -> set[str]:
        vals = set()
        for row in rows:
            trace = trace_from_row(row)
            if trace and isinstance(trace.get(field), str):
                vals.add(trace[field])
        return vals

    result = {}
    for field in ("trace_id", "source_id", "source_hash", "final_hash"):
        natural_set = keyset(natural, field)
        oracle_set = keyset(oracle, field)
        missing = oracle_set - natural_set
        result[field] = {
            "natural_unique": len(natural_set),
            "oracle_unique": len(oracle_set),
            "oracle_not_in_natural": len(missing),
            "examples": sorted(missing)[:20],
            "contained": len(missing) == 0 if oracle_set else None,
        }
    return result


def eligible_split_audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    split_counts: Counter[str] = Counter()
    source_by_split: dict[str, set[str]] = defaultdict(set)
    hashes_by_split: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        split = row.get("split") or row.get("partition") or row.get("subset") or "unassigned"
        split = str(split)
        split_counts[split] += 1
        sid = row.get("source_id") or row.get("id")
        sh = row.get("source_hash")
        if isinstance(sid, str): source_by_split[split].add(sid)
        if isinstance(sh, str): hashes_by_split[split].add(sh)

    overlap = {}
    splits = sorted(source_by_split)
    for i, a in enumerate(splits):
        for b in splits[i + 1 :]:
            overlap[f"{a}__VS__{b}"] = {
                "source_id_overlap": len(source_by_split[a] & source_by_split[b]),
                "source_hash_overlap": len(hashes_by_split[a] & hashes_by_split[b]),
            }
    return {"split_counts": dict(split_counts), "within_manifest_overlap": overlap}


def markdown_counter(counter: dict[str, Any], limit: int = 30) -> str:
    if not counter:
        return "_Not available_\n"
    lines = ["| Value | Count |", "|---|---:|"]
    for key, value in list(counter.items())[:limit]:
        lines.append(f"| `{key}` | {value} |")
    if len(counter) > limit:
        lines.append(f"\n_Only first {limit} rows shown; full data are in JSON._")
    return "\n".join(lines) + "\n"


def render_markdown(report: dict[str, Any]) -> str:
    lines: list[str] = []
    lines += ["# EditTrace-XAI Development Generation Audit", ""]
    lines += [f"Generated from: `{report['paths']['project_root']}`", ""]

    lines += ["## 1. Controlled benchmark generation", ""]
    lines += ["| Model | Benchmark | Outcomes | Accepted | Rejected | Yield | Attempts |", "|---|---|---:|---:|---:|---:|---:|"]
    for key, value in report["controlled"].items():
        model, benchmark = key.split("::", 1)
        y = value.get("acceptance_yield")
        lines.append(
            f"| {model} | {benchmark} | {value['outcomes']} | {value['accepted']} | {value['rejected']} | "
            f"{(100*y):.1f}% | {value['attempts']['total']} |"
        )
    lines.append("")

    for key, value in report["controlled"].items():
        lines += [f"### {key.replace('::', ' — ')} rejection reasons", ""]
        reasons = value["rejection_reasons"]
        if not reasons:
            lines.append("_No rejection reasons found._\n")
            continue
        lines += ["| Reason | Attempt count | % rejected attempts | Outcomes with reason | % rejected outcomes |", "|---|---:|---:|---:|---:|"]
        for reason, item in reasons.items():
            fa = item["fraction_of_rejected_attempts"]
            fo = item["fraction_of_rejected_outcomes"]
            lines.append(
                f"| `{reason}` | {item['attempt_count']} | {100*fa:.2f}% | {item['outcomes_with_reason']} | {100*fo:.2f}% |"
            )
        lines.append("")

    lines += ["## 2. B-Natural shift availability", ""]
    lines += ["| Model | Traces | Margin pairs available | Shifted | Shift fraction |", "|---|---:|---:|---:|---:|"]
    for model, value in report["natural"].items():
        margins = value["distributions"]["margins"]
        shifted = margins.get("shifted_count")
        frac = margins.get("shifted_fraction")
        shifted_text = str(shifted) if shifted is not None else "N/A"
        frac_text = f"{100*frac:.2f}%" if frac is not None else "N/A"
        lines.append(f"| {model} | {value['rows']} | {margins['paired_available']} | {shifted_text} | {frac_text} |")
    lines += ["", "If margin fields are unavailable in the JSONL, this section remains explicitly `N/A` and must be populated from model evaluation outputs rather than inferred.", ""]

    lines += ["## 3. Trace structure", ""]
    for group_name in ("natural", "oracle"):
        lines += [f"### {group_name.title()}", ""]
        for model, value in report[group_name].items():
            d = value["distributions"]
            lines += [f"#### {model}", "", "**History length**", "", markdown_counter(d["history_length"]), "**Operations**", "", markdown_counter(d["operations"])]

    lines += ["## 4. Oracle containment", ""]
    for model, value in report["oracle_containment"].items():
        lines += [f"### {model}", "", "| Key | Oracle not in B-Natural | Contained |", "|---|---:|---|"]
        for field, item in value.items():
            lines.append(f"| `{field}` | {item['oracle_not_in_natural']} | {item['contained']} |")
        lines.append("")

    lines += ["## 5. Replay invalidity", ""]
    for group_name in ("natural", "oracle"):
        for model, value in report[group_name].items():
            inv = value["distributions"]["replay_invalidity"]
            lines.append(f"- **{group_name}/{model}:** {inv['invalid_trace_rows']} trace rows with typed invalidity; reasons: `{json.dumps(inv['typed_reasons'], sort_keys=True)}`")
    for key, value in report["controlled"].items():
        inv = value["accepted_trace_distributions"]["replay_invalidity"]
        lines.append(f"- **controlled/{key}:** {inv['invalid_trace_rows']} accepted trace rows with typed invalidity; reasons: `{json.dumps(inv['typed_reasons'], sort_keys=True)}`")
    lines += ["", "A zero count means that no typed invalidity was found in the inspected fields; it is not automatically a proof of complete replay validity.", ""]

    lines += ["## 6. Eligible-manifest split audit", ""]
    for model, value in report["eligible"].items():
        lines += [f"### {model}", "", f"Split counts: `{json.dumps(value['split_counts'], sort_keys=True)}`", "", f"Overlap checks: `{json.dumps(value['within_manifest_overlap'], sort_keys=True)}`", ""]

    lines += ["## 7. Parameter severity and observed-grid extremes", ""]
    lines.append("Full parameter distributions and observed min/max concentration are stored in the JSON report. These indicators describe concentration at observed parameter-grid extremes; they do not measure pixel clipping or visual saturation unless such fields are explicitly logged.\n")

    lines += ["## 8. Cross-benchmark overlaps", ""]
    lines.append("The JSON report contains pairwise overlaps for `source_id`, `source_hash`, `trace_id`, and `final_hash`. Some source overlap across development benchmark families may be expected; interpretation must follow the frozen protocol. Development/test overlap should remain zero in eligible manifests.\n")

    lines += ["## 9. Audit status", ""]
    for item in report["status"]:
        lines.append(f"- {item}")
    lines.append("")
    return "\n".join(lines)


def write_summary_csv(path: Path, report: dict[str, Any]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "model", "benchmark", "outcomes", "accepted", "rejected", "acceptance_yield",
                "attempts_total", "attempts_mean", "attempts_median", "attempts_max",
            ],
        )
        writer.writeheader()
        for key, value in report["controlled"].items():
            model, benchmark = key.split("::", 1)
            writer.writerow({
                "model": model,
                "benchmark": benchmark,
                "outcomes": value["outcomes"],
                "accepted": value["accepted"],
                "rejected": value["rejected"],
                "acceptance_yield": value["acceptance_yield"],
                "attempts_total": value["attempts"]["total"],
                "attempts_mean": value["attempts"].get("mean"),
                "attempts_median": value["attempts"].get("median"),
                "attempts_max": value["attempts"].get("observed_max"),
            })


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--traces-dir", type=Path, default=None)
    parser.add_argument("--manifests-dir", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    root = args.project_root.resolve()
    traces_dir = (args.traces_dir or root / "artifacts" / "traces").resolve()
    manifests_dir = (args.manifests_dir or root / "artifacts" / "manifests").resolve()
    output_dir = (args.output_dir or root / "artifacts" / "logs").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    missing: list[str] = []
    controlled: dict[str, Any] = {}
    natural: dict[str, Any] = {}
    oracle: dict[str, Any] = {}
    eligible: dict[str, Any] = {}
    loaded_groups: dict[str, list[dict[str, Any]]] = {}
    file_inventory: dict[str, Any] = {}

    for (model, benchmark), filename in CONTROLLED_FILES.items():
        path = traces_dir / filename
        key = f"{model}::{benchmark}"
        if not path.exists():
            missing.append(str(path))
            continue
        rows = read_jsonl(path)
        controlled[key] = controlled_summary(rows)
        loaded_groups[f"controlled::{key}"] = rows
        file_inventory[str(path)] = {"rows": len(rows), "bytes": path.stat().st_size, "sha256": sha256(path)}

    natural_rows_by_model: dict[str, list[dict[str, Any]]] = {}
    oracle_rows_by_model: dict[str, list[dict[str, Any]]] = {}

    for model, filename in NATURAL_FILES.items():
        path = traces_dir / filename
        if not path.exists():
            missing.append(str(path))
            continue
        rows = read_jsonl(path)
        natural_rows_by_model[model] = rows
        natural[model] = {"rows": len(rows), "distributions": collect_trace_distributions(rows)}
        loaded_groups[f"natural::{model}"] = rows
        file_inventory[str(path)] = {"rows": len(rows), "bytes": path.stat().st_size, "sha256": sha256(path)}

    for model, filename in ORACLE_FILES.items():
        path = traces_dir / filename
        if not path.exists():
            missing.append(str(path))
            continue
        rows = read_jsonl(path)
        oracle_rows_by_model[model] = rows
        oracle[model] = {"rows": len(rows), "distributions": collect_trace_distributions(rows)}
        loaded_groups[f"oracle::{model}"] = rows
        file_inventory[str(path)] = {"rows": len(rows), "bytes": path.stat().st_size, "sha256": sha256(path)}

    for model, filename in ELIGIBLE_FILES.items():
        path = manifests_dir / filename
        if not path.exists():
            missing.append(str(path))
            continue
        rows = read_jsonl(path)
        eligible[model] = eligible_split_audit(rows)
        file_inventory[str(path)] = {"rows": len(rows), "bytes": path.stat().st_size, "sha256": sha256(path)}

    containment = {}
    for model in sorted(set(natural_rows_by_model) & set(oracle_rows_by_model)):
        containment[model] = oracle_containment(natural_rows_by_model[model], oracle_rows_by_model[model])

    config_data = None
    if args.config:
        config_path = args.config.resolve()
        if config_path.exists() and yaml is not None:
            with config_path.open("r", encoding="utf-8") as fh:
                config_data = yaml.safe_load(fh)
            file_inventory[str(config_path)] = {"bytes": config_path.stat().st_size, "sha256": sha256(config_path)}
        elif config_path.exists():
            config_data = {"note": "PyYAML unavailable; config not parsed."}
        else:
            missing.append(str(config_path))

    status = []
    status.append("Controlled benchmark accepted/rejected counts and attempts: COMPLETE" if len(controlled) == 4 else "Controlled benchmark summary: INCOMPLETE")
    status.append("B-Oracle files: COMPLETE" if len(oracle) == 2 else "B-Oracle files: INCOMPLETE")
    margin_available = all(v["distributions"]["margins"]["paired_available"] > 0 for v in natural.values()) if natural else False
    status.append("B-Natural shifted-count audit: COMPLETE" if margin_available else "B-Natural shifted-count audit: OPEN — margin fields not found in one or more B-Natural JSONL files")
    status.append("Eligible split overlap audit: COMPLETE" if len(eligible) == 2 else "Eligible split overlap audit: INCOMPLETE")
    status.append("Parameter-grid distribution audit: COMPLETE for logged parameters; pixel-level saturation remains OPEN unless explicitly logged")
    status.append("Replay-invalidity audit: COMPLETE for typed fields present in JSONL; zero counts are not an implementation proof")
    status.append("TEST SET MUST REMAIN LOCKED until all OPEN items are resolved or formally documented.")

    report = {
        "schema_version": "1.0",
        "paths": {
            "project_root": str(root),
            "traces_dir": str(traces_dir),
            "manifests_dir": str(manifests_dir),
            "output_dir": str(output_dir),
        },
        "file_inventory": file_inventory,
        "missing_files": missing,
        "controlled": controlled,
        "natural": natural,
        "oracle": oracle,
        "oracle_containment": containment,
        "eligible": eligible,
        "cross_benchmark_overlap": overlap_report(loaded_groups),
        "config": config_data,
        "status": status,
    }

    json_path = output_dir / "development_generation_audit.json"
    md_path = output_dir / "development_generation_audit.md"
    csv_path = output_dir / "development_generation_audit.csv"

    json_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")
    write_summary_csv(csv_path, report)

    print(json.dumps({
        "json": str(json_path),
        "markdown": str(md_path),
        "csv": str(csv_path),
        "missing_files": missing,
        "status": status,
    }, indent=2))
    return 0 if not missing else 2


if __name__ == "__main__":
    raise SystemExit(main())
