import json
from pathlib import Path

import pytest

from code_switching_training.data import (
    EXPECTED_CONVERSATION_COUNTS,
    EXPECTED_MESSAGE_COUNT,
    LABELS,
    load_split,
    validate_dataset_splits,
)


def test_resolved_splits_match_the_fixed_data_contract() -> None:
    summaries = validate_dataset_splits()

    assert set(summaries) == set(EXPECTED_CONVERSATION_COUNTS)
    assert sum(summary.message_count for summary in summaries.values()) == EXPECTED_MESSAGE_COUNT
    assert {
        split: summary.conversation_count for split, summary in summaries.items()
    } == EXPECTED_CONVERSATION_COUNTS
    for summary in summaries.values():
        assert set(summary.label_counts) == set(LABELS)


def test_load_split_rejects_an_invalid_label(tmp_path: Path) -> None:
    record = {
        "message_id": 1,
        "thread_id": 1,
        "turn": 1,
        "business_id": 1,
        "role": "customer",
        "is_bot": False,
        "text": "Example message",
        "label": "unsupported_label",
    }
    (tmp_path / "training.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="invalid label"):
        load_split("training", tmp_path)
