import pandas as pd
import pytest

from data_preprocessing.op2_data_evaluation import (
    run_evaluation,
    EvaluationOutputs,
)

def test_run_evaluation_returns_namedtuple():
    df = pd.DataFrame({
        "a": [1, 2, 3],
        "b": [4, 5, 6]
    })

    result = run_evaluation(df)

    assert isinstance(result, EvaluationOutputs)


def test_no_nulls_detected():
    df = pd.DataFrame({
        "num": [1, 2, 3],
        "cat": ["a", "b", "c"]
    })

    result = run_evaluation(df)

    assert result.columns_with_nulls_output.empty


def test_detects_null_columns():
    df = pd.DataFrame({
        "num": [1, None, 3],
        "cat": ["a", "b", None]
    })

    result = run_evaluation(df)

    assert "num" in result.columns_with_nulls_output
    assert "cat" in result.columns_with_nulls_output
    assert result.columns_with_nulls_output["num"] == 1
    assert result.columns_with_nulls_output["cat"] == 1


def test_detects_object_columns():
    df = pd.DataFrame({
        "num": [1, 2, 3],
        "cat": ["x", "y", "z"],
        "cat2": ["a", "b", "c"]
    })

    result = run_evaluation(df)

    assert list(result.object_columns_output) == ["cat", "cat2"]


def test_no_object_columns():
    df = pd.DataFrame({
        "a": [1, 2, 3],
        "b": [4.0, 5.0, 6.0]
    })

    result = run_evaluation(df)

    assert result.object_columns_output.empty
