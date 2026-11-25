"""Run both study conditions on demonstration or supplied observations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from affective.config import load_policy_configuration
from affective.models import InteractionStage, PolicyContext, Privacy, TimePressure
from affective.multimodal import Modality, ModalityEvidence, MultimodalObservation
from affective.pipeline import AffectiveInterventionPipeline
from affective.policy import EmotionOnlyPolicy
from affective.scenarios import DEMONSTRATION_ONLY, demonstration_scenarios


ROOT = Path(__file__).resolve().parent

UNCERTAINTY_SUMMARY_FIELDS = (
    "predictive_entropy",
    "top_two_margin",
    "signal_quality",
    "data_coverage",
    "missingness",
    "cross_modal_disagreement",
)


def observation_from_dict(data: dict) -> MultimodalObservation:
    items = []
    for raw in data["evidence"]:
        items.append(ModalityEvidence(
            modality=Modality(raw["modality"]),
            emotion_distribution=raw.get("emotion_distribution"),
            valence=raw.get("valence"),
            arousal=raw.get("arousal"),
            dominance=raw.get("dominance"),
            quality=raw.get("quality", 1.0),
            available=raw.get("available", True),
            prediction_samples=raw.get("prediction_samples", ()),
        ))
    return MultimodalObservation(tuple(items))


def summarize_condition(result: dict) -> dict:
    """Return the decision and main uncertainty values used for comparison."""
    decision = result["decision"]
    uncertainty = result["uncertainty"]
    return {
        "mode": decision["mode"],
        "intensity": decision["intensity"],
        "support_action": decision["support_action"],
        "uncertainty": {
            field: uncertainty[field] for field in UNCERTAINTY_SUMMARY_FIELDS
        },
    }


def run_comparison(
    observation: MultimodalObservation,
    user_initiated: bool = True,
    receptive: bool = True,
) -> dict:
    context = PolicyContext(
        privacy=Privacy.PRIVATE,
        time_pressure=TimePressure.LOW,
        interaction_stage=InteractionStage.EXPLORING,
        receptive=receptive,
        user_initiated=user_initiated,
    )
    aware = load_policy_configuration(ROOT / "config" / "policy_thresholds.json").build_policy()
    policies = {"emotion_only": EmotionOnlyPolicy(), "uncertainty_aware": aware}
    conditions = {
        name: AffectiveInterventionPipeline(policy).run(observation, context).to_dict()
        for name, policy in policies.items()
    }
    return {
        "status": DEMONSTRATION_ONLY,
        "context": {
            "privacy": context.privacy.value,
            "time_pressure": context.time_pressure.value,
            "interaction_stage": context.interaction_stage.value,
            "receptive": context.receptive,
            "user_initiated": context.user_initiated,
        },
        "conditions": conditions,
        "summary": {
            name: summarize_condition(result) for name, result in conditions.items()
        },
    }


def run_all_comparisons(
    user_initiated: bool = True,
    receptive: bool = True,
) -> dict:
    """Run both policies for every built-in demonstration scenario."""
    return {
        "status": DEMONSTRATION_ONLY,
        "run_type": "all_scenarios",
        "context": {
            "user_initiated": user_initiated,
            "receptive": receptive,
        },
        "scenarios": {
            name: run_comparison(
                observation,
                user_initiated=user_initiated,
                receptive=receptive,
            )
            for name, observation in demonstration_scenarios().items()
        },
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="v0.3 双条件管线运行器")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--input", type=Path, help="包含 evidence 数组的 JSON 文件")
    source.add_argument("--scenario", help="内置演示场景名称")
    source.add_argument("--all", action="store_true", help="运行全部内置演示场景")
    initiated = parser.add_mutually_exclusive_group()
    initiated.add_argument(
        "--user-initiated",
        dest="user_initiated",
        action="store_true",
        help="用户主动发起交互（默认）",
    )
    initiated.add_argument(
        "--no-user-initiated",
        dest="user_initiated",
        action="store_false",
        help="用户未主动发起交互",
    )
    reception = parser.add_mutually_exclusive_group()
    reception.add_argument(
        "--receptive",
        dest="receptive",
        action="store_true",
        help="用户愿意接受介入（默认）",
    )
    reception.add_argument(
        "--not-receptive",
        dest="receptive",
        action="store_false",
        help="用户不愿意接受介入",
    )
    parser.add_argument("--output", type=Path, help="可选输出 JSON 文件")
    parser.set_defaults(user_initiated=True, receptive=True)
    args = parser.parse_args(argv)
    if args.all:
        result = run_all_comparisons(
            user_initiated=args.user_initiated,
            receptive=args.receptive,
        )
    elif args.input:
        observation = observation_from_dict(json.loads(args.input.read_text(encoding="utf-8")))
        result = run_comparison(
            observation,
            user_initiated=args.user_initiated,
            receptive=args.receptive,
        )
    else:
        scenarios = demonstration_scenarios()
        name = args.scenario or next(iter(scenarios))
        if name not in scenarios:
            raise SystemExit(f"未知场景：{name}；可选：{', '.join(scenarios)}")
        result = run_comparison(
            scenarios[name],
            user_initiated=args.user_initiated,
            receptive=args.receptive,
        )
        result["scenario"] = name
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    else:
        print(text)


if __name__ == "__main__":
    main()
