import pytest

from casmi26.evaluation.metrics import (
    mean_reciprocal_rank,
    reciprocal_rank,
    smiles_to_inchikey14,
)


def test_correct_prediction_at_rank_one():
    assert reciprocal_rank(
        "CCO",
        ["CCO"],
    ) == 1.0


def test_correct_prediction_at_rank_two():
    assert reciprocal_rank(
        "CCO",
        ["CC", "CCO"],
    ) == 0.5


def test_missing_prediction_scores_zero():
    assert reciprocal_rank(
        "CCO",
        ["CC", "CCC", "CCCC"],
    ) == 0.0


def test_predictions_after_k_are_ignored():
    predictions = ["CC"] * 25 + ["CCO"]

    assert reciprocal_rank(
        "CCO",
        predictions,
        k=25,
    ) == 0.0


def test_stereochemistry_is_ignored():
    first = smiles_to_inchikey14(
        "OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O"
    )
    second = smiles_to_inchikey14(
        "OCC1OC(O)C(O)C(O)C1O"
    )

    assert first == second


def test_invalid_prediction_does_not_match():
    assert reciprocal_rank(
        "CCO",
        ["this-is-not-smiles", "CCO"],
    ) == 0.5


def test_mean_reciprocal_rank():
    score = mean_reciprocal_rank(
        truths=[
            "CCO",
            "CCC",
            "CCCC",
        ],
        predictions=[
            ["CCO"],
            ["CC", "CCC"],
            ["CC"],
        ],
    )

    assert score == pytest.approx(
        (1.0 + 0.5 + 0.0) / 3
    )