import json
from pathlib import Path

from code_switching_training.evaluation import evaluate_predictions
from code_switching_training.xlm import (
    DEFAULT_MODEL_NAME,
    DEFAULT_MODEL_PATH,
    PROGRESS_INTERVAL_STEPS,
    EpochResult,
    ExperimentResult,
    XLMConfig,
    save_model,
    write_result,
)


def test_xlm_config_has_expected_default_training_settings() -> None:
    config = XLMConfig()

    assert config.model_name == DEFAULT_MODEL_NAME
    assert config.max_length == 96
    assert config.train_batch_size == 4
    assert config.max_epochs == 5
    assert config.learning_rate == 2e-5
    assert PROGRESS_INTERVAL_STEPS == 25


def test_xlm_result_serializes_validation_and_test_metrics(tmp_path: Path) -> None:
    metrics = evaluate_predictions(["english"], ["english"])
    result = ExperimentResult(
        config=XLMConfig(),
        device="mps",
        validation_history=[EpochResult(epoch=1, validation_metrics=metrics)],
        selected_epoch=1,
        test_metrics=metrics,
        split_message_counts={"training": 1, "validation": 1, "test": 1},
    )
    output_path = tmp_path / "result.json"

    write_result(result, output_path)

    written = json.loads(output_path.read_text(encoding="utf-8"))
    assert written["model"] == "xlm_roberta_base_sequence_classification"
    assert written["model_directory"] == "models/xlm_roberta_base"
    assert written["selected_epoch"] == 1
    assert written["validation_history"][0]["validation_metrics"]["accuracy"] == 1.0


def test_save_model_saves_weights_and_tokenizer_to_requested_directory(tmp_path: Path) -> None:
    saved_paths: list[Path] = []

    class Saveable:
        def save_pretrained(self, path: Path, **_kwargs: object) -> None:
            saved_paths.append(path)

    output_path = tmp_path / "xlm_roberta_base"
    save_model(Saveable(), Saveable(), output_path)

    assert output_path.is_dir()
    assert saved_paths == [output_path, output_path]
    assert DEFAULT_MODEL_PATH.name == "xlm_roberta_base"
