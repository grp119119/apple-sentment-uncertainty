"""Validate normalized official-dataset exports without inventing labels."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from affective_v03.models import AffectLabel, ScenarioEmotion


TURN_LABELS = {item.value for item in AffectLabel}
SCENARIO_LABELS = {item.value for item in ScenarioEmotion}


def validate_row(row: dict, index: int) -> list[str]:
    errors = []
    if row.get("affect_label") not in (None, "") and row["affect_label"] not in TURN_LABELS:
        errors.append(f"row {index}: 非法 affect_label={row['affect_label']!r}")
    if row.get("scenario_emotion") not in (None, "") and row["scenario_emotion"] not in SCENARIO_LABELS:
        errors.append(f"row {index}: 非法 scenario_emotion={row['scenario_emotion']!r}")
    distribution = row.get("emotion_distribution")
    if isinstance(distribution, str) and distribution.strip():
        try:
            distribution = json.loads(distribution)
        except json.JSONDecodeError:
            errors.append(f"row {index}: emotion_distribution 不是有效 JSON")
            distribution = None
    if distribution is not None:
        if not isinstance(distribution, dict):
            errors.append(f"row {index}: emotion_distribution 必须是对象")
        else:
            unknown = set(distribution) - TURN_LABELS
            if unknown:
                errors.append(f"row {index}: 概率分布含未知标签 {sorted(unknown)}")
            values = list(distribution.values())
            if not values or any(isinstance(v, bool) or not isinstance(v, (int, float)) or v < 0 for v in values) or sum(values) <= 0:
                errors.append(f"row {index}: 概率分布必须包含非负且总和大于 0 的数值")
    return errors


def load_rows(path: Path) -> list[dict]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with path.open(encoding="utf-8-sig", newline="") as handle:
            return list(csv.DictReader(handle))
    if suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, list) else data.get("records", [])


def main() -> None:
    parser = argparse.ArgumentParser(description="校验 v0.3 标准化数据标签")
    parser.add_argument("input", type=Path)
    args = parser.parse_args()
    rows = load_rows(args.input)
    errors = [error for index, row in enumerate(rows, 1) for error in validate_row(row, index)]
    print(json.dumps({"valid": not errors, "rows": len(rows), "errors": errors}, ensure_ascii=False, indent=2))
    raise SystemExit(1 if errors else 0)


if __name__ == "__main__":
    main()
