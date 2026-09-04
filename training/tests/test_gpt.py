import json
from pathlib import Path

import pytest

from code_switching_training.data import Message
from code_switching_training.evaluation import evaluate_predictions
from code_switching_training.gpt import (
    DEFAULT_MODEL_NAME,
    DEFAULT_PREDICTION_DIR,
    DEFAULT_RESULT_PATH,
    LABEL_SCHEMA,
    CandidateResult,
    ExperimentResult,
    PromptConfig,
    predict_label,
    select_prompt,
    write_result,
)


class _FakeResponses:
    def __init__(self, labels: list[str]) -> None:
        self.labels = iter(labels)
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        return type("Response", (), {"output_text": json.dumps({"label": next(self.labels)})})()


class _FakeClient:
    def __init__(self, labels: list[str]) -> None:
        self.responses = _FakeResponses(labels)


def _message(message_id: int, text: str, label: str) -> Message:
    return Message(
        message_id=message_id,
        thread_id=message_id,
        turn=1,
        business_id=1,
        role="customer",
        is_bot=False,
        text=text,
        label=label,
    )


def test_predict_label_uses_the_schema_constrained_responses_api() -> None:
    client = _FakeClient(["bangla"])
    prompt = PromptConfig("test", "Classify the message.")

    label = predict_label(client, "আমি ভালো আছি", prompt)

    assert label == "bangla"
    request = client.responses.calls[0]
    assert request["model"] == DEFAULT_MODEL_NAME
    assert request["temperature"] == 0
    assert request["text"] == {
        "format": {
            "type": "json_schema",
            "name": "language_label",
            "strict": True,
            "schema": LABEL_SCHEMA,
        }
    }


def test_predict_label_rejects_an_unknown_api_label() -> None:
    client = _FakeClient(["mixed"])

    with pytest.raises(ValueError, match="unknown label"):
        predict_label(client, "hello", PromptConfig("test", "Classify the message."))


def test_select_prompt_evaluates_each_candidate_on_validation(tmp_path: Path) -> None:
    messages = [
        _message(1, "hello", "english"),
        _message(2, "আমি", "bangla"),
    ]
    candidates = [PromptConfig("first", "Prompt one"), PromptConfig("second", "Prompt two")]
    client = _FakeClient(["english", "bangla", "english", "english"])

    results = select_prompt(client, messages, candidates, prediction_dir=tmp_path)

    assert [result.prompt for result in results] == candidates
    assert results[0].validation_metrics.accuracy == 1.0
    assert results[1].validation_metrics.accuracy == 0.5
    assert len(client.responses.calls) == 4
    assert (tmp_path / "validation_first.jsonl").is_file()
    assert "hello" not in (tmp_path / "validation_first.jsonl").read_text(encoding="utf-8")


def test_write_result_creates_a_compact_json_record(tmp_path: Path) -> None:
    metrics = evaluate_predictions(["english"], ["english"])
    prompt = PromptConfig("test", "Classify the message.")
    result = ExperimentResult(
        model_name=DEFAULT_MODEL_NAME,
        candidates=[CandidateResult(prompt, metrics)],
        selected_prompt=prompt,
        test_metrics=metrics,
        split_message_counts={"training": 1, "validation": 1, "test": 1},
    )
    output_path = tmp_path / "result.json"

    write_result(result, output_path)

    written = json.loads(output_path.read_text(encoding="utf-8"))
    assert written["model"] == DEFAULT_MODEL_NAME
    assert written["method"] == "zero_shot_structured_output"
    assert written["selected_prompt"]["name"] == "test"
    assert DEFAULT_RESULT_PATH == DEFAULT_PREDICTION_DIR.parent / "evaluation.json"
