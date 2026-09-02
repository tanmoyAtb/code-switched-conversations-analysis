import pytest

from code_switching_training.evaluation import evaluate_predictions


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
