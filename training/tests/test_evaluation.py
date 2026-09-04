import json
from pathlib import Path

import pytest

from code_switching_training.data import Message
from code_switching_training.evaluation import evaluate_predictions, write_predictions


def test_evaluate_predictions_returns_all_shared_metrics() -> None:
    result = evaluate_predictions(
        ["english", "bangla", "romanized_bangla"],
        ["english", "english", "romanized_bangla"],
    )

    assert result.accuracy == pytest.approx(2 / 3)
    assert result.per_label["bangla"].support == 1
    assert result.per_label["bangla"].f1 == 0.0
    assert result.confusion_matrix == [[1, 0, 0], [1, 0, 0], [0, 0, 1]]
    assert result.to_dict()["confusion_matrix_labels"] == [
        "english",
        "bangla",
        "romanized_bangla",
    ]


def test_evaluate_predictions_rejects_unknown_labels() -> None:
    with pytest.raises(ValueError, match="unknown labels"):
        evaluate_predictions(["english"], ["other"])


def test_write_predictions_keeps_ids_and_labels_without_message_text(tmp_path: Path) -> None:
    messages = [
        Message(1, 1, 1, 1, "customer", False, "private message", "english"),
    ]
    output_path = tmp_path / "predictions.jsonl"

    write_predictions(messages, ["bangla"], output_path)

    written = json.loads(output_path.read_text(encoding="utf-8"))
    assert written == {
        "message_id": 1,
        "true_label": "english",
        "predicted_label": "bangla",
    }
