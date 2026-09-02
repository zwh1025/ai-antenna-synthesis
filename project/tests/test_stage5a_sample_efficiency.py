"""Artifact-independent contract tests for the portable Stage 5A reader."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "project"))

import run_stage5a_sample_efficiency as stage5a  # noqa: E402


def test_portable_summary_loads_without_model_artifacts():
    payload = stage5a.load_summary()
    assert payload["study"]["train_parent_counts"] == [2, 4, 6, 8]
    assert payload["study"]["d4_rows_by_parent_count"] == {"2": 16, "4": 32, "6": 48, "8": 64}
    assert payload["directly_comparable_to_upstream_v3"] is False


def test_nested_parent_counts_and_correlated_d4_rows():
    payload = stage5a.load_summary()
    previous = 0
    for count in (2, 4, 6, 8):
        item = payload["levels"][str(count)]
        assert item["independent_train_parents"] > previous
        assert item["correlated_d4_rows"] == 8 * count
        previous = item["independent_train_parents"]


def test_selection_and_test_isolation_are_frozen():
    payload = stage5a.load_summary()
    assert payload["study"]["selection"] == "validation_only"
    assert all(payload["levels"][str(count)]["test_used_for_selection"] is False for count in (2, 4, 6, 8))


def test_teacher_gap_detector_keeps_the_fixed_threshold():
    history = [
        {"epoch": 1, "validation_sll_evaluated": True, "validation_teacher_gap_db_by_row": {"a": 0.7, "b": 0.4}},
        {"epoch": 5, "validation_sll_evaluated": True, "validation_teacher_gap_db_by_row": {"a": 0.49, "b": -0.51}},
        {"epoch": 10, "validation_sll_evaluated": True, "validation_teacher_gap_db_by_row": {"a": 0.2, "b": -0.2}},
    ]
    assert stage5a.first_epoch_within_teacher_gap(history, require_all=False) == 1
    assert stage5a.first_epoch_within_teacher_gap(history, require_all=True) == 10


def test_ninety_percent_gain_detector():
    history = [
        {"epoch": 1, "validation_sll_evaluated": True, "validation_gain_vs_taylor_db": 0.2},
        {"epoch": 5, "validation_sll_evaluated": True, "validation_gain_vs_taylor_db": 0.9},
        {"epoch": 10, "validation_sll_evaluated": True, "validation_gain_vs_taylor_db": 1.0},
    ]
    assert stage5a.first_epoch_to_gain(history, final_gain_db=1.0) == 5
    assert stage5a.first_epoch_to_gain(history, final_gain_db=0.0) == "NOT_REACHED"


def test_missing_summary_is_an_explicit_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="compact summary not found"):
        stage5a.load_summary(tmp_path / "missing.json")


def test_summary_is_finite_and_json_serializable():
    payload = stage5a.load_summary()
    json.dumps(payload, ensure_ascii=False)
    assert payload["complexity"]["parameter_count"] == 597250
