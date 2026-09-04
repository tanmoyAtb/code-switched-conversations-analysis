"""Zero-shot GPT-4o mini evaluation with validation-only prompt selection."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from openai import OpenAI

from .data import DEFAULT_DATA_DIR, LABELS, Message, load_split, validate_dataset_splits
from .evaluation import EvaluationResult, evaluate_predictions, write_predictions

DEFAULT_MODEL_NAME = "gpt-4o-mini"
DEFAULT_RESULT_DIR = Path(__file__).resolve().parents[2] / "results" / "gpt_4o_mini"
DEFAULT_RESULT_PATH = DEFAULT_RESULT_DIR / "evaluation.json"
DEFAULT_PREDICTION_DIR = DEFAULT_RESULT_DIR / "predictions"
SELECTION_METRIC = "macro_f1"
PROGRESS_INTERVAL_MESSAGES = 25

LABEL_SCHEMA = {
    "type": "object",
    "properties": {
        "label": {
            "type": "string",
            "enum": list(LABELS),
        }
    },
    "required": ["label"],
    "additionalProperties": False,
}


@dataclass(frozen=True, slots=True)
class PromptConfig:
    """A fixed classification prompt considered during validation selection."""

    name: str
    instructions: str

    def to_dict(self) -> dict[str, str]:
        return {"name": self.name, "instructions": self.instructions}


@dataclass(frozen=True, slots=True)
class CandidateResult:
    """Validation metrics for one fixed zero-shot prompt."""

    prompt: PromptConfig
    validation_metrics: EvaluationResult

    def to_dict(self) -> dict[str, object]:
        return {
            "prompt": self.prompt.to_dict(),
            "validation_metrics": self.validation_metrics.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class ExperimentResult:
    """Selection record and one held-out test evaluation for GPT-4o mini."""

    model_name: str
    candidates: list[CandidateResult]
    selected_prompt: PromptConfig
    test_metrics: EvaluationResult
    split_message_counts: dict[str, int]

    def to_dict(self) -> dict[str, object]:
        return {
            "model": self.model_name,
            "method": "zero_shot_structured_output",
            "selection_metric": SELECTION_METRIC,
            "split_message_counts": self.split_message_counts,
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "selected_prompt": self.selected_prompt.to_dict(),
            "test_metrics": self.test_metrics.to_dict(),
        }


PROMPT_CANDIDATES = (
    PromptConfig(
        name="definitions_v1",
        instructions=(
            "Classify the language of one anonymized message from a Bangla-English "
            "business conversation. Return one label only. "
            "english: the message is primarily English. "
            "bangla: the message is primarily Bangla/Bengali written in Bengali script. "
            "romanized_bangla: the message is primarily Bangla written in the Latin alphabet. "
            "Ignore placeholders, numbers, punctuation, and emoji when deciding."
        ),
    ),
    PromptConfig(
        name="decision_rules_v1",
        instructions=(
            "Identify the predominant language of one anonymized Bangla-English business "
            "chat message. Choose bangla when its main Bangla content uses Bengali script. "
            "Choose romanized_bangla when its main Bangla content is spelled with Latin "
            "letters, even if it contains a few English words. Choose english when its "
            "main content is English. Ignore placeholders, numbers, punctuation, and emoji."
        ),
    ),
)


def predict_label(
    client: Any,
    text: str,
    prompt: PromptConfig,
    model_name: str = DEFAULT_MODEL_NAME,
) -> str:
    """Request one schema-constrained language label and validate the returned value."""
    response = client.responses.create(
        model=model_name,
        input=[
            {"role": "developer", "content": prompt.instructions},
            {"role": "user", "content": f"Message:\n{text}"},
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": "language_label",
                "strict": True,
                "schema": LABEL_SCHEMA,
            }
        },
        temperature=0,
    )
    try:
        payload = json.loads(response.output_text)
    except (AttributeError, TypeError, json.JSONDecodeError) as error:
        raise ValueError("GPT-4o mini did not return a valid JSON label") from error

    if not isinstance(payload, dict) or set(payload) != {"label"}:
        raise ValueError("GPT-4o mini response must contain only a 'label' field")
    label = payload["label"]
    if label not in LABELS:
        raise ValueError(f"GPT-4o mini returned an unknown label: {label!r}")
    return label


def predict_messages(
    client: Any,
    messages: Sequence[Message],
    prompt: PromptConfig,
    model_name: str = DEFAULT_MODEL_NAME,
    split_name: str = "evaluation",
) -> list[str]:
    """Predict messages in order and print bounded progress for a user-run API job."""
    predictions: list[str] = []
    for index, message in enumerate(messages, start=1):
        predictions.append(predict_label(client, message.text, prompt, model_name))
        if index % PROGRESS_INTERVAL_MESSAGES == 0 or index == len(messages):
            print(
                f"{prompt.name} {split_name}: {index}/{len(messages)} messages completed",
                flush=True,
            )
    return predictions


def select_prompt(
    client: Any,
    validation_messages: Sequence[Message],
    prompt_candidates: Sequence[PromptConfig] = PROMPT_CANDIDATES,
    model_name: str = DEFAULT_MODEL_NAME,
    prediction_dir: Path | None = None,
) -> list[CandidateResult]:
    """Evaluate all fixed prompt candidates exclusively on validation messages."""
    if not prompt_candidates:
        raise ValueError("At least one prompt candidate is required")

    labels = [message.label for message in validation_messages]
    candidates: list[CandidateResult] = []
    for prompt in prompt_candidates:
        predictions = predict_messages(
            client,
            validation_messages,
            prompt,
            model_name,
            "validation",
        )
        if prediction_dir is not None:
            write_predictions(
                validation_messages,
                predictions,
                prediction_dir / f"validation_{prompt.name}.jsonl",
            )
        candidates.append(
            CandidateResult(
                prompt=prompt,
                validation_metrics=evaluate_predictions(labels, predictions),
            )
        )
    return candidates


def run_experiment(
    data_dir: Path = DEFAULT_DATA_DIR,
    model_name: str = DEFAULT_MODEL_NAME,
    client: Any | None = None,
) -> ExperimentResult:
    """Select a prompt on validation, then call the held-out test split exactly once."""
    summaries = validate_dataset_splits(data_dir)
    validation_messages = load_split("validation", data_dir)
    test_messages = load_split("test", data_dir)
    api_client = client or _create_client()

    candidates = select_prompt(
        api_client,
        validation_messages,
        model_name=model_name,
        prediction_dir=DEFAULT_PREDICTION_DIR,
    )
    selected = max(candidates, key=lambda candidate: candidate.validation_metrics.macro_f1)
    test_predictions = predict_messages(
        api_client,
        test_messages,
        selected.prompt,
        model_name,
        "test",
    )
    write_predictions(test_messages, test_predictions, DEFAULT_PREDICTION_DIR / "test.jsonl")
    return ExperimentResult(
        model_name=model_name,
        candidates=candidates,
        selected_prompt=selected.prompt,
        test_metrics=evaluate_predictions(
            [message.label for message in test_messages],
            test_predictions,
        ),
        split_message_counts={split: summary.message_count for split, summary in summaries.items()},
    )


def write_result(result: ExperimentResult, output_path: Path = DEFAULT_RESULT_PATH) -> None:
    """Write the reproducible run record alongside separate prediction artifacts."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _create_client() -> OpenAI:
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Export it in your terminal before running this evaluator."
        )
    return OpenAI()


def main() -> None:
    result = run_experiment()
    write_result(result)
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
