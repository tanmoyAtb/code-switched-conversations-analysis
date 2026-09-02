"""Loading and validation for the fixed resolved data splits."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

LABELS = ("english", "bangla", "romanized_bangla")
SPLIT_NAMES = ("training", "validation", "test")
EXPECTED_CONVERSATION_COUNTS = {
    "training": 1200,
    "validation": 150,
    "test": 150,
}
EXPECTED_MESSAGE_COUNT = 10_004
DEFAULT_DATA_DIR = Path(__file__).resolve().parents[3] / "data" / "resolved"


@dataclass(frozen=True, slots=True)
class Message:
    """One annotated message used by all language-identification experiments."""

    message_id: int
    thread_id: int
    turn: int
    business_id: int
    role: str
    is_bot: bool
    text: str
    label: str


@dataclass(frozen=True, slots=True)
class SplitSummary:
    """Basic provenance information for one fixed data split."""

    split: str
    message_count: int
    conversation_count: int
    label_counts: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "split": self.split,
            "message_count": self.message_count,
            "conversation_count": self.conversation_count,
            "label_counts": self.label_counts,
        }


def load_split(split: str, data_dir: Path = DEFAULT_DATA_DIR) -> list[Message]:
    """Load one split and validate every record against the shared data contract."""
    if split not in SPLIT_NAMES:
        expected = ", ".join(SPLIT_NAMES)
        raise ValueError(f"Unknown split {split!r}; expected one of: {expected}")

    path = data_dir / f"{split}.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"Expected split file does not exist: {path}")

    messages: list[Message] = []
    seen_message_ids: set[int] = set()
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                raise ValueError(f"{path}:{line_number} is blank; JSONL records cannot be blank")
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number} is not valid JSON") from error

            message = _parse_message(record, path, line_number)
            if message.message_id in seen_message_ids:
                raise ValueError(f"{path}:{line_number} duplicates message_id {message.message_id}")
            seen_message_ids.add(message.message_id)
            messages.append(message)

    if not messages:
        raise ValueError(f"{path} contains no messages")
    return messages


def validate_dataset_splits(data_dir: Path = DEFAULT_DATA_DIR) -> dict[str, SplitSummary]:
    """Verify the frozen split sizes, label validity, and conversation isolation."""
    split_messages = {split: load_split(split, data_dir) for split in SPLIT_NAMES}
    thread_ids_by_split = {
        split: {message.thread_id for message in messages}
        for split, messages in split_messages.items()
    }
    message_ids_by_split = {
        split: {message.message_id for message in messages}
        for split, messages in split_messages.items()
    }

    for split, expected_count in EXPECTED_CONVERSATION_COUNTS.items():
        actual_count = len(thread_ids_by_split[split])
        if actual_count != expected_count:
            raise ValueError(
                f"{split} contains {actual_count} conversations; expected {expected_count}"
            )

    _assert_disjoint(thread_ids_by_split, "thread_id")
    _assert_disjoint(message_ids_by_split, "message_id")

    total_messages = sum(len(messages) for messages in split_messages.values())
    if total_messages != EXPECTED_MESSAGE_COUNT:
        raise ValueError(
            f"Splits contain {total_messages} messages; expected {EXPECTED_MESSAGE_COUNT}"
        )

    return {
        split: SplitSummary(
            split=split,
            message_count=len(messages),
            conversation_count=len(thread_ids_by_split[split]),
            label_counts=dict(Counter(message.label for message in messages)),
        )
        for split, messages in split_messages.items()
    }


def _parse_message(record: object, path: Path, line_number: int) -> Message:
    if not isinstance(record, dict):
        raise ValueError(f"{path}:{line_number} must contain a JSON object")

    expected_fields = {
        "message_id": int,
        "thread_id": int,
        "turn": int,
        "business_id": int,
        "role": str,
        "is_bot": bool,
        "text": str,
        "label": str,
    }
    missing_fields = expected_fields.keys() - record.keys()
    if missing_fields:
        missing = ", ".join(sorted(missing_fields))
        raise ValueError(f"{path}:{line_number} is missing required fields: {missing}")

    for field, expected_type in expected_fields.items():
        value = record[field]
        if expected_type is int:
            is_expected_type = isinstance(value, int) and not isinstance(value, bool)
        else:
            is_expected_type = isinstance(value, expected_type)
        if not is_expected_type:
            expected_name = expected_type.__name__
            raise ValueError(f"{path}:{line_number} field {field!r} must be a {expected_name}")

    if not record["text"].strip():
        raise ValueError(f"{path}:{line_number} field 'text' cannot be empty")
    if record["label"] not in LABELS:
        expected = ", ".join(LABELS)
        raise ValueError(
            f"{path}:{line_number} has invalid label {record['label']!r}; "
            f"expected one of: {expected}"
        )

    return Message(
        message_id=record["message_id"],
        thread_id=record["thread_id"],
        turn=record["turn"],
        business_id=record["business_id"],
        role=record["role"],
        is_bot=record["is_bot"],
        text=record["text"],
        label=record["label"],
    )


def _assert_disjoint(ids_by_split: dict[str, set[int]], identifier: str) -> None:
    for index, left_split in enumerate(SPLIT_NAMES):
        for right_split in SPLIT_NAMES[index + 1 :]:
            overlap = ids_by_split[left_split] & ids_by_split[right_split]
            if overlap:
                examples = ", ".join(str(value) for value in sorted(overlap)[:5])
                raise ValueError(
                    f"{identifier} values appear in both {left_split} and {right_split}: {examples}"
                )
