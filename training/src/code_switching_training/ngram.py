"""Character n-gram logistic-regression baseline experiment."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from .data import DEFAULT_DATA_DIR, Message, load_split, validate_dataset_splits
from .evaluation import EvaluationResult, evaluate_predictions, write_predictions

DEFAULT_RESULT_DIR = Path(__file__).resolve().parents[2] / "results" / "ngram_logistic_regression"
DEFAULT_RESULT_PATH = DEFAULT_RESULT_DIR / "evaluation.json"
DEFAULT_PREDICTION_DIR = DEFAULT_RESULT_DIR / "predictions"
SELECTION_METRIC = "macro_f1"
RANDOM_SEED = 42


@dataclass(frozen=True, slots=True)
class NgramConfig:
    """One fully specified character n-gram logistic-regression configuration."""

    name: str
    analyzer: str
    ngram_range: tuple[int, int]
    min_df: int
    c: float

    def to_dict(self) -> dict[str, str | int | float | list[int]]:
        return {
            "name": self.name,
            "analyzer": self.analyzer,
            "ngram_range": list(self.ngram_range),
            "min_df": self.min_df,
            "C": self.c,
        }


@dataclass(frozen=True, slots=True)
class CandidateResult:
    """Validation performance for one candidate configuration."""

    config: NgramConfig
    validation_metrics: EvaluationResult

    def to_dict(self) -> dict[str, object]:
        return {
            "config": self.config.to_dict(),
            "validation_metrics": self.validation_metrics.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class ExperimentResult:
    """Selection record and final held-out test metrics for the baseline."""

    candidates: list[CandidateResult]
    selected_config: NgramConfig
    test_metrics: EvaluationResult
    split_message_counts: dict[str, int]

    def to_dict(self) -> dict[str, object]:
        return {
            "model": "character_tfidf_logistic_regression",
            "selection_metric": SELECTION_METRIC,
            "random_seed": RANDOM_SEED,
            "split_message_counts": self.split_message_counts,
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "selected_config": self.selected_config.to_dict(),
            "test_metrics": self.test_metrics.to_dict(),
        }


CANDIDATE_CONFIGS = (
    NgramConfig("char_wb_2_5_c1", "char_wb", (2, 5), 1, 1.0),
    NgramConfig("char_wb_3_6_c1", "char_wb", (3, 6), 1, 1.0),
    NgramConfig("char_2_5_c1", "char", (2, 5), 1, 1.0),
    NgramConfig("char_2_5_c2", "char", (2, 5), 1, 2.0),
    NgramConfig("char_3_6_c2", "char", (3, 6), 1, 2.0),
)


def build_pipeline(config: NgramConfig) -> Pipeline:
    """Create an unfitted classifier pipeline from a fixed configuration."""
    return Pipeline(
        [
            (
                "vectorizer",
                TfidfVectorizer(
                    analyzer=config.analyzer,
                    ngram_range=config.ngram_range,
                    min_df=config.min_df,
                    lowercase=True,
                    sublinear_tf=True,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    C=config.c,
                    max_iter=1_000,
                    random_state=RANDOM_SEED,
                ),
            ),
        ]
    )


def evaluate_configuration(
    config: NgramConfig,
    training_messages: Sequence[Message],
    evaluation_messages: Sequence[Message],
) -> EvaluationResult:
    """Fit a candidate on training messages and evaluate it on another fixed split."""
    predictions = predict_configuration(config, training_messages, evaluation_messages)
    return evaluate_predictions(_labels(evaluation_messages), predictions)


def predict_configuration(
    config: NgramConfig,
    training_messages: Sequence[Message],
    evaluation_messages: Sequence[Message],
) -> list[str]:
    """Fit a configuration and return one prediction for every evaluation message."""
    model = build_pipeline(config)
    model.fit(_texts(training_messages), _labels(training_messages))
    return model.predict(_texts(evaluation_messages)).tolist()


def select_configuration(
    training_messages: Sequence[Message],
    validation_messages: Sequence[Message],
    candidate_configs: Sequence[NgramConfig] = CANDIDATE_CONFIGS,
) -> list[CandidateResult]:
    """Evaluate all candidates on validation and retain their comparable metrics."""
    if not candidate_configs:
        raise ValueError("At least one candidate configuration is required")

    return [
        CandidateResult(
            config=config,
            validation_metrics=evaluate_configuration(
                config,
                training_messages,
                validation_messages,
            ),
        )
        for config in candidate_configs
    ]


def run_experiment(data_dir: Path = DEFAULT_DATA_DIR) -> ExperimentResult:
    """Select on validation, then evaluate the selected baseline once on the test split."""
    summaries = validate_dataset_splits(data_dir)
    training_messages = load_split("training", data_dir)
    validation_messages = load_split("validation", data_dir)
    test_messages = load_split("test", data_dir)

    candidates = select_configuration(training_messages, validation_messages)
    selected = max(candidates, key=lambda candidate: candidate.validation_metrics.macro_f1)

    validation_predictions = predict_configuration(
        selected.config,
        training_messages,
        validation_messages,
    )
    write_predictions(
        validation_messages,
        validation_predictions,
        DEFAULT_PREDICTION_DIR / "validation.jsonl",
    )

    final_training_messages = [*training_messages, *validation_messages]
    test_predictions = predict_configuration(
        selected.config,
        final_training_messages,
        test_messages,
    )
    test_metrics = evaluate_predictions(_labels(test_messages), test_predictions)
    write_predictions(test_messages, test_predictions, DEFAULT_PREDICTION_DIR / "test.jsonl")
    return ExperimentResult(
        candidates=candidates,
        selected_config=selected.config,
        test_metrics=test_metrics,
        split_message_counts={split: summary.message_count for split, summary in summaries.items()},
    )


def write_result(result: ExperimentResult, output_path: Path = DEFAULT_RESULT_PATH) -> None:
    """Write a compact, tracked experiment record without predictions or checkpoints."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _texts(messages: Sequence[Message]) -> list[str]:
    return [message.text for message in messages]


def _labels(messages: Sequence[Message]) -> list[str]:
    return [message.label for message in messages]


def main() -> None:
    result = run_experiment()
    write_result(result)
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
