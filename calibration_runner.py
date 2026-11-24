"""Generate a descriptive pilot report; never mutates policy thresholds."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean


RATINGS = ("appropriateness", "intrusiveness", "perceived_understanding", "willingness_to_continue", "helpfulness")


def build_report(payload: dict) -> dict:
    if payload.get("schema_version") != 3:
        raise ValueError("仅支持 schema_version 3")
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for record in payload.get("records", []):
        condition = record.get("metadata", {}).get("condition", "unknown")
        mode = record.get("result", {}).get("decision", {}).get("mode", "unknown")
        groups[(condition, mode)].append(record)
    summary = []
    for (condition, mode), rows in sorted(groups.items()):
        feedback_rows = [row["feedback"] for row in rows if row.get("feedback")]
        uncertainty_rows = [row.get("result", {}).get("uncertainty", {}) for row in rows]
        item = {"condition": condition, "mode": mode, "n": len(rows), "feedback_n": len(feedback_rows)}
        for metric in RATINGS:
            values = [row[metric] for row in feedback_rows if isinstance(row.get(metric), (int, float))]
            item[f"mean_{metric}"] = round(mean(values), 3) if values else None
        entropies = [row["predictive_entropy"] for row in uncertainty_rows if isinstance(row.get("predictive_entropy"), (int, float))]
        item["mean_predictive_entropy"] = round(mean(entropies), 3) if entropies else None
        summary.append(item)
    return {
        "report_type": "DESCRIPTIVE_PILOT_REPORT",
        "thresholds_changed": False,
        "warning": "该报告只描述数据；阈值必须经 Study 1 审查后另行冻结。",
        "total_records": len(payload.get("records", [])),
        "groups": summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="生成 v0.3 pilot 描述性报告")
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = build_report(json.loads(args.input.read_text(encoding="utf-8")))
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    else:
        print(text)


if __name__ == "__main__":
    main()
