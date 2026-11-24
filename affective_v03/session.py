"""Versioned experiment logging; feedback never changes a frozen policy."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from .models import PolicyContext, ScenarioEmotion, StudyFeedback
from .pipeline import PipelineResult


@dataclass(frozen=True)
class StudyMetadata:
    participant_id: str
    condition: str
    scenario_id: str
    scenario_emotion: ScenarioEmotion | None = None
    pre_emotion_intensity: int | None = None

    def __post_init__(self) -> None:
        if self.condition not in {"emotion_only", "uncertainty_aware"}:
            raise ValueError("condition must be emotion_only or uncertainty_aware")
        if self.pre_emotion_intensity is not None and not 1 <= self.pre_emotion_intensity <= 5:
            raise ValueError("pre_emotion_intensity must be 1..5")


@dataclass
class ExperimentRecord:
    timestamp: str
    metadata: dict[str, Any]
    context: dict[str, Any]
    result: dict[str, Any]
    feedback: dict[str, Any] | None = None


class ExperimentSession:
    schema_version = 3

    def __init__(self) -> None:
        self.records: list[ExperimentRecord] = []

    def add_round(
        self,
        metadata: StudyMetadata,
        context: PolicyContext,
        result: PipelineResult,
        feedback: StudyFeedback | None = None,
    ) -> ExperimentRecord:
        metadata_dict = asdict(metadata)
        if metadata.scenario_emotion is not None:
            metadata_dict["scenario_emotion"] = metadata.scenario_emotion.value
        context_dict = context.to_dict()
        feedback_dict = asdict(feedback) if feedback is not None else None
        if feedback is not None and feedback.corrected_emotion is not None:
            feedback_dict["corrected_emotion"] = feedback.corrected_emotion.value
        record = ExperimentRecord(
            timestamp=datetime.now().astimezone().isoformat(timespec="seconds"),
            metadata=metadata_dict,
            context=context_dict,
            result=result.to_dict(),
            feedback=feedback_dict,
        )
        self.records.append(record)
        return record

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        payload = {
            "schema_version": self.schema_version,
            "records": [asdict(record) for record in self.records],
        }
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(path)

    @classmethod
    def load(cls, path: Path) -> "ExperimentSession":
        session = cls()
        if not path.exists():
            return session
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("schema_version") != cls.schema_version:
            raise ValueError("unsupported experiment log schema")
        session.records = [ExperimentRecord(**item) for item in data.get("records", [])]
        return session
