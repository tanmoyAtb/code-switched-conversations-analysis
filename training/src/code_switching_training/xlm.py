"""Fine-tuning experiment for FacebookAI/xlm-roberta-base."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding

from .data import DEFAULT_DATA_DIR, LABELS, Message, load_split, validate_dataset_splits
from .evaluation import EvaluationResult, evaluate_predictions

DEFAULT_MODEL_NAME = "FacebookAI/xlm-roberta-base"
DEFAULT_RESULT_PATH = Path(__file__).resolve().parents[2] / "results" / "xlm_roberta_base.json"
DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "xlm_roberta_base"
RANDOM_SEED = 42
LABEL_TO_ID = {label: index for index, label in enumerate(LABELS)}
ID_TO_LABEL = {index: label for label, index in LABEL_TO_ID.items()}
PROGRESS_INTERVAL_STEPS = 25


@dataclass(frozen=True, slots=True)
class XLMConfig:
    """Fixed fine-tuning settings; the validation split selects the best epoch."""

    model_name: str = DEFAULT_MODEL_NAME
    max_length: int = 96
    train_batch_size: int = 4
    evaluation_batch_size: int = 16
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    max_epochs: int = 5
    early_stopping_patience: int = 2

    def to_dict(self) -> dict[str, str | int | float]:
        return {
            "model_name": self.model_name,
            "max_length": self.max_length,
            "train_batch_size": self.train_batch_size,
            "evaluation_batch_size": self.evaluation_batch_size,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "max_epochs": self.max_epochs,
            "early_stopping_patience": self.early_stopping_patience,
        }


@dataclass(frozen=True, slots=True)
class EpochResult:
    """Validation result after one training epoch."""

    epoch: int
    validation_metrics: EvaluationResult

    def to_dict(self) -> dict[str, object]:
        return {
            "epoch": self.epoch,
            "validation_metrics": self.validation_metrics.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class ExperimentResult:
    """Validation history and final test metrics for XLM-R fine-tuning."""

    config: XLMConfig
    device: str
    validation_history: list[EpochResult]
    selected_epoch: int
    test_metrics: EvaluationResult
    split_message_counts: dict[str, int]

    def to_dict(self) -> dict[str, object]:
        return {
            "model": "xlm_roberta_base_sequence_classification",
            "model_directory": "models/xlm_roberta_base",
            "config": self.config.to_dict(),
            "random_seed": RANDOM_SEED,
            "device": self.device,
            "selection_metric": "validation_macro_f1",
            "validation_history": [result.to_dict() for result in self.validation_history],
            "selected_epoch": self.selected_epoch,
            "split_message_counts": self.split_message_counts,
            "test_metrics": self.test_metrics.to_dict(),
        }


class TokenizedMessageDataset(Dataset[dict[str, Any]]):
    """Tokenized message-level labels for a PyTorch data loader."""

    def __init__(self, tokenizer: Any, messages: Sequence[Message], max_length: int) -> None:
        self.encodings = tokenizer(
            [message.text for message in messages],
            truncation=True,
            max_length=max_length,
        )
        self.labels = [LABEL_TO_ID[message.label] for message in messages]

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict[str, Any]:
        encoding = {key: value[index] for key, value in self.encodings.items()}
        return {**encoding, "labels": self.labels[index]}


def select_device() -> torch.device:
    """Prefer the local accelerator while retaining portable CUDA and CPU fallbacks."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def run_experiment(
    data_dir: Path = DEFAULT_DATA_DIR,
    config: XLMConfig = XLMConfig(),
) -> ExperimentResult:
    """Select an epoch on validation, refit on training plus validation, and test once."""
    summaries = validate_dataset_splits(data_dir)
    training_messages = load_split("training", data_dir)
    validation_messages = load_split("validation", data_dir)
    test_messages = load_split("test", data_dir)
    device = select_device()

    _set_seed()
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
        use_fast=True,
        local_files_only=True,
    )
    model = _new_model(config)
    validation_history, selected_epoch = _train_with_validation(
        model,
        tokenizer,
        training_messages,
        validation_messages,
        config,
        device,
    )

    del model
    _clear_accelerator_cache(device)

    _set_seed()
    final_model = _new_model(config)
    _train_for_epochs(
        final_model,
        tokenizer,
        [*training_messages, *validation_messages],
        config,
        device,
        selected_epoch,
    )
    save_model(final_model, tokenizer)
    test_metrics = _evaluate(final_model, tokenizer, test_messages, config, device)

    return ExperimentResult(
        config=config,
        device=device.type,
        validation_history=validation_history,
        selected_epoch=selected_epoch,
        test_metrics=test_metrics,
        split_message_counts={split: summary.message_count for split, summary in summaries.items()},
    )


def write_result(result: ExperimentResult, output_path: Path = DEFAULT_RESULT_PATH) -> None:
    """Write compact metrics and settings without storing model weights or predictions."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def save_model(model: Any, tokenizer: Any, model_path: Path = DEFAULT_MODEL_PATH) -> None:
    """Save the final fine-tuned model and tokenizer for local reuse or later publishing."""
    model_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(model_path, safe_serialization=True)
    tokenizer.save_pretrained(model_path)


def _new_model(config: XLMConfig) -> Any:
    return AutoModelForSequenceClassification.from_pretrained(
        config.model_name,
        num_labels=len(LABELS),
        label2id=LABEL_TO_ID,
        id2label=ID_TO_LABEL,
        local_files_only=True,
    )


def _train_with_validation(
    model: Any,
    tokenizer: Any,
    training_messages: Sequence[Message],
    validation_messages: Sequence[Message],
    config: XLMConfig,
    device: torch.device,
) -> tuple[list[EpochResult], int]:
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    validation_history: list[EpochResult] = []
    best_epoch = 0
    best_score = float("-inf")
    epochs_without_improvement = 0

    for epoch in range(1, config.max_epochs + 1):
        _train_one_epoch(model, tokenizer, training_messages, config, device, optimizer, epoch)
        validation_metrics = _evaluate(model, tokenizer, validation_messages, config, device)
        validation_history.append(EpochResult(epoch, validation_metrics))
        print(f"Epoch {epoch}: validation macro F1 = {validation_metrics.macro_f1:.4f}")

        if validation_metrics.macro_f1 > best_score:
            best_score = validation_metrics.macro_f1
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.early_stopping_patience:
                break

    if best_epoch == 0:
        raise RuntimeError("No validation epoch completed")
    return validation_history, best_epoch


def _train_for_epochs(
    model: Any,
    tokenizer: Any,
    training_messages: Sequence[Message],
    config: XLMConfig,
    device: torch.device,
    epochs: int,
) -> None:
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    for epoch in range(1, epochs + 1):
        _train_one_epoch(model, tokenizer, training_messages, config, device, optimizer, epoch)
        print(f"Final fit: completed epoch {epoch} of {epochs}")


def _train_one_epoch(
    model: Any,
    tokenizer: Any,
    messages: Sequence[Message],
    config: XLMConfig,
    device: torch.device,
    optimizer: AdamW,
    epoch: int,
) -> None:
    model.train()
    loader = _make_loader(
        tokenizer,
        messages,
        config.max_length,
        config.train_batch_size,
        shuffle=True,
        seed=RANDOM_SEED + epoch,
    )
    total_steps = len(loader)
    for step, batch in enumerate(loader, start=1):
        optimizer.zero_grad(set_to_none=True)
        output = model(**_move_to_device(batch, device))
        output.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if step % PROGRESS_INTERVAL_STEPS == 0 or step == total_steps:
            print(f"Epoch {epoch}: completed batch {step}/{total_steps}", flush=True)


def _evaluate(
    model: Any,
    tokenizer: Any,
    messages: Sequence[Message],
    config: XLMConfig,
    device: torch.device,
) -> EvaluationResult:
    model.eval()
    loader = _make_loader(
        tokenizer,
        messages,
        config.max_length,
        config.evaluation_batch_size,
        shuffle=False,
        seed=RANDOM_SEED,
    )
    prediction_ids: list[int] = []
    with torch.inference_mode():
        for batch in loader:
            output = model(**_move_to_device(batch, device))
            prediction_ids.extend(output.logits.argmax(dim=-1).cpu().tolist())

    predictions = [ID_TO_LABEL[prediction_id] for prediction_id in prediction_ids]
    return evaluate_predictions([message.label for message in messages], predictions)


def _make_loader(
    tokenizer: Any,
    messages: Sequence[Message],
    max_length: int,
    batch_size: int,
    *,
    shuffle: bool,
    seed: int,
) -> DataLoader[dict[str, torch.Tensor]]:
    dataset = TokenizedMessageDataset(tokenizer, messages, max_length)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt"),
        generator=torch.Generator().manual_seed(seed),
    )


def _move_to_device(
    batch: dict[str, torch.Tensor], device: torch.device
) -> dict[str, torch.Tensor]:
    return {name: value.to(device) for name, value in batch.items()}


def _set_seed() -> None:
    random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)


def _clear_accelerator_cache(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()


def main() -> None:
    result = run_experiment()
    write_result(result)
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
