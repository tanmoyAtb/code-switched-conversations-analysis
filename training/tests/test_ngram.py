import json
from pathlib import Path

from code_switching_training.data import Message
from code_switching_training.ngram import (
    CANDIDATE_CONFIGS,
    ExperimentResult,
    evaluate_configuration,
    select_configuration,
    write_result,
)


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


def test_candidate_configuration_evaluates_with_shared_metrics() -> None:
    training_messages = [
        _message(1, "hello thanks", "english"),
        _message(2, "good morning", "english"),
        _message(3, "আমি ভালো আছি", "bangla"),
        _message(4, "আজকে আসবেন", "bangla"),
        _message(5, "ami bhalo achi", "romanized_bangla"),
        _message(6, "ajke ashben", "romanized_bangla"),
    ]
    validation_messages = [
        _message(7, "thanks", "english"),
        _message(8, "আমি আসবো", "bangla"),
        _message(9, "ami ashbo", "romanized_bangla"),
    ]

    metrics = evaluate_configuration(CANDIDATE_CONFIGS[0], training_messages, validation_messages)

    assert 0.0 <= metrics.accuracy <= 1.0
    assert 0.0 <= metrics.macro_f1 <= 1.0
    assert len(metrics.confusion_matrix) == 3


def test_selection_returns_validation_results_for_every_candidate() -> None:
    training_messages = [
        _message(1, "hello", "english"),
        _message(2, "আমি", "bangla"),
        _message(3, "ami", "romanized_bangla"),
    ]
    validation_messages = [
        _message(4, "hello", "english"),
        _message(5, "আমি", "bangla"),
        _message(6, "ami", "romanized_bangla"),
    ]

    candidates = select_configuration(
        training_messages,
        validation_messages,
        candidate_configs=CANDIDATE_CONFIGS[:2],
    )

    assert [candidate.config for candidate in candidates] == list(CANDIDATE_CONFIGS[:2])
    assert all(0.0 <= candidate.validation_metrics.macro_f1 <= 1.0 for candidate in candidates)


def test_write_result_creates_compact_json_record(tmp_path: Path) -> None:
    metrics = evaluate_configuration(
        CANDIDATE_CONFIGS[0],
        [_message(1, "hello", "english"), _message(2, "আমি", "bangla")],
        [_message(3, "hello", "english"), _message(4, "আমি", "bangla")],
    )
    result = ExperimentResult(
        candidates=[],
        selected_config=CANDIDATE_CONFIGS[0],
        test_metrics=metrics,
        split_message_counts={"training": 2, "validation": 0, "test": 2},
    )
    output_path = tmp_path / "result.json"

    write_result(result, output_path)

    written = json.loads(output_path.read_text(encoding="utf-8"))
    assert written["model"] == "character_tfidf_logistic_regression"
    assert written["selected_config"]["name"] == "char_wb_2_5_c1"
