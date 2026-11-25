"""Literature-backed uncertainty and offline calibration metrics."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence

from .models import AffectLabel, UncertaintyProfile, normalize_distribution
from .multimodal import ModalityEvidence


def normalized_entropy(distribution: Mapping[AffectLabel | str, float]) -> float:
    values = normalize_distribution(distribution).values()
    entropy = -sum(value * math.log2(value) for value in values if value > 0)
    return entropy / math.log2(len(AffectLabel))


def top_two_margin(distribution: Mapping[AffectLabel | str, float]) -> float:
    ordered = sorted(normalize_distribution(distribution).values(), reverse=True)
    return ordered[0] - ordered[1]


def total_variation(
    first: Mapping[AffectLabel | str, float],
    second: Mapping[AffectLabel | str, float],
) -> float:
    left, right = normalize_distribution(first), normalize_distribution(second)
    return 0.5 * sum(abs(left[label] - right[label]) for label in AffectLabel)


def _entropy_raw(distribution: Mapping[AffectLabel, float]) -> float:
    return -sum(value * math.log2(value) for value in distribution.values() if value > 0)


def generalized_js_disagreement(distributions: Sequence[Mapping[AffectLabel, float]]) -> float:
    """Experimental cross-modal disagreement; no final affect threshold is claimed."""
    if len(distributions) < 2:
        return 0.0
    mean = {
        label: sum(item[label] for item in distributions) / len(distributions)
        for label in AffectLabel
    }
    divergence = _entropy_raw(mean) - sum(_entropy_raw(item) for item in distributions) / len(distributions)
    maximum = math.log2(len(distributions))
    return max(0.0, min(1.0, divergence / maximum))


def ensemble_variance(evidence: Iterable[ModalityEvidence]) -> float:
    variances: list[float] = []
    for item in evidence:
        samples = item.prediction_samples
        if len(samples) < 2:
            continue
        for label in AffectLabel:
            values = [sample[label] for sample in samples]
            mean = sum(values) / len(values)
            variances.append(sum((value - mean) ** 2 for value in values) / len(values))
    if not variances:
        return 0.0
    # Bernoulli probability variance is bounded by 0.25; normalize to 0..1.
    return min(1.0, sum(variances) / len(variances) / 0.25)


def build_uncertainty_profile(
    fused_distribution: Mapping[AffectLabel | str, float],
    evidence: Sequence[ModalityEvidence],
    previous_distribution: Mapping[AffectLabel | str, float] | None = None,
) -> UncertaintyProfile:
    active = [item for item in evidence if item.available and item.quality > 0]
    categorical = [item for item in active if item.emotion_distribution is not None]
    total_modalities = len(evidence)
    coverage = len(active) / total_modalities
    quality = sum(item.quality for item in active) / len(active) if active else 0.0
    disagreement = generalized_js_disagreement(
        [item.emotion_distribution for item in categorical if item.emotion_distribution is not None]
    )
    instability = (
        total_variation(previous_distribution, fused_distribution)
        if previous_distribution is not None
        else 0.0
    )
    entropy = normalized_entropy(fused_distribution)
    margin = top_two_margin(fused_distribution)
    reasons = (
        f"predictive entropy={entropy:.3f}",
        f"top-two margin={margin:.3f}",
        f"signal quality={quality:.3f}",
        f"data coverage={coverage:.3f}",
        "cross-modal disagreement is experimental pending affect-specific calibration",
    )
    return UncertaintyProfile(
        predictive_entropy=entropy,
        top_two_margin=margin,
        signal_quality=quality,
        data_coverage=coverage,
        missingness=1.0 - coverage,
        ensemble_variance=ensemble_variance(evidence),
        temporal_instability=instability,
        cross_modal_disagreement=disagreement,
        reasons=reasons,
    )


def brier_score(
    predictions: Sequence[Mapping[AffectLabel | str, float]],
    labels: Sequence[AffectLabel | str],
) -> float:
    if not predictions or len(predictions) != len(labels):
        raise ValueError("predictions and labels must be non-empty and equal in length")
    total = 0.0
    for prediction, raw_label in zip(predictions, labels):
        distribution = normalize_distribution(prediction)
        label = raw_label if isinstance(raw_label, AffectLabel) else AffectLabel(raw_label)
        total += sum(
            (distribution[candidate] - (1.0 if candidate == label else 0.0)) ** 2
            for candidate in AffectLabel
        )
    return total / len(predictions)


def expected_calibration_error(
    predictions: Sequence[Mapping[AffectLabel | str, float]],
    labels: Sequence[AffectLabel | str],
    bins: int = 10,
) -> float:
    if not predictions or len(predictions) != len(labels):
        raise ValueError("predictions and labels must be non-empty and equal in length")
    if bins <= 0:
        raise ValueError("bins must be positive")
    buckets: list[list[tuple[float, bool]]] = [[] for _ in range(bins)]
    for prediction, raw_label in zip(predictions, labels):
        distribution = normalize_distribution(prediction)
        predicted = max(distribution, key=distribution.get)
        confidence = distribution[predicted]
        actual = raw_label if isinstance(raw_label, AffectLabel) else AffectLabel(raw_label)
        index = min(bins - 1, int(confidence * bins))
        buckets[index].append((confidence, predicted == actual))
    total = len(predictions)
    error = 0.0
    for bucket in buckets:
        if not bucket:
            continue
        mean_confidence = sum(item[0] for item in bucket) / len(bucket)
        accuracy = sum(item[1] for item in bucket) / len(bucket)
        error += len(bucket) / total * abs(accuracy - mean_confidence)
    return error
