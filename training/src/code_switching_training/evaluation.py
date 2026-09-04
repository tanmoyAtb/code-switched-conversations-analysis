"""Reusable classification metrics for all evaluated models."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

from .data import LABELS, Message


@dataclass(frozen=True, slots=True)
class LabelMetrics:
    """Precision, recall, F1, and support for one language label."""

    precision: float
    recall: float
    f1: float
    support: int

    def to_dict(self) -> dict[str, float | int]:
        return {
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "support": self.support,
        }


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    """Comparable metrics for a fixed-label classification experiment."""

    accuracy: float
    macro_f1: float
    per_label: dict[str, LabelMetrics]
    confusion_matrix: list[list[int]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "accuracy": self.accuracy,
            "macro_f1": self.macro_f1,
            "per_label": {
                label: metrics.to_dict() for label, metrics in self.per_label.items()
            },
            "confusion_matrix": self.confusion_matrix,
            "confusion_matrix_labels": list(LABELS),
        }


def evaluate_predictions(y_true: Sequence[str], y_pred: Sequence[str]) -> EvaluationResult:
    """Calculate the common metrics used for every model comparison."""
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must contain the same number of labels")
    if not y_true:
        raise ValueError("Cannot evaluate an empty set of predictions")

    observed_labels = set(y_true) | set(y_pred)
    unknown_labels = observed_labels - set(LABELS)
    if unknown_labels:
        unknown = ", ".join(sorted(unknown_labels))
        raise ValueError(f"Predictions contain unknown labels: {unknown}")

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=LABELS,
        zero_division=0,
    )
    per_label = {
        label: LabelMetrics(
            precision=float(precision[index]),
            recall=float(recall[index]),
            f1=float(f1[index]),
            support=int(support[index]),
        )
        for index, label in enumerate(LABELS)
    }
    matrix = confusion_matrix(y_true, y_pred, labels=LABELS).tolist()

    return EvaluationResult(
        accuracy=float(accuracy_score(y_true, y_pred)),
        macro_f1=float(sum(metrics.f1 for metrics in per_label.values()) / len(LABELS)),
        per_label=per_label,
        confusion_matrix=matrix,
    )


def write_predictions(
    messages: Sequence[Message],
    predictions: Sequence[str],
    output_path: Path,
) -> None:
    """Write auditable labels without duplicating any message text in result artifacts."""
    if len(messages) != len(predictions):
        raise ValueError("messages and predictions must contain the same number of records")

    unknown_labels = set(predictions) - set(LABELS)
    if unknown_labels:
        unknown = ", ".join(sorted(unknown_labels))
        raise ValueError(f"Predictions contain unknown labels: {unknown}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as destination:
        for message, prediction in zip(messages, predictions, strict=True):
            record = {
                "message_id": message.message_id,
                "true_label": message.label,
                "predicted_label": prediction,
            }
            destination.write(json.dumps(record, ensure_ascii=False) + "\n")
