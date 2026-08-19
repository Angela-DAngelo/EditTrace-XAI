"""Remove machine-specific source_path fields before publishing a JSONL catalog."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def sanitize(path: Path, in_place: bool) -> Path:
    destination = path if in_place else path.with_name(f"{path.stem}.public{path.suffix}")
    temporary = path.with_suffix(path.suffix + ".tmp") if in_place else destination
    count = 0
    with path.open("r", encoding="utf-8") as source, temporary.open("w", encoding="utf-8", newline="\n") as target:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number}: {exc}") from exc
            count += int(record.pop("source_path", None) is not None)
            target.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
    if in_place:
        backup = path.with_suffix(path.suffix + ".bak")
        shutil.copy2(path, backup)
        temporary.replace(path)
        print(f"Sanitized {count} records in {path}; backup written to {backup}")
    else:
        print(f"Sanitized {count} records; public catalog written to {destination}")
    return destination


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("catalog", type=Path, help="Input JSONL trace catalog")
    parser.add_argument("--in-place", action="store_true", help="Replace the catalog after creating a .bak copy")
    args = parser.parse_args()
    if not args.catalog.is_file():
        parser.error(f"Catalog not found: {args.catalog}")
    sanitize(args.catalog, args.in_place)


if __name__ == "__main__":
    main()
