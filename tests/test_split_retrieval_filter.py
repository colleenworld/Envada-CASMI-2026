import pandas as pd
import pytest

from casmi26.retrieval.split_index import (
    filter_to_training_structures,
)


def test_filters_validation_structures():
    frame = pd.DataFrame(
        {
            "inchikey14": [
                "TRAIN-A",
                "VALIDATION-A",
                "TRAIN-B",
                "VALIDATION-B",
            ],
            "value": [
                1,
                2,
                3,
                4,
            ],
        }
    )

    result = filter_to_training_structures(
        frame,
        {
            "TRAIN-A",
            "TRAIN-B",
        },
    )

    assert result[
        "inchikey14"
    ].tolist() == [
        "TRAIN-A",
        "TRAIN-B",
    ]

    assert result[
        "value"
    ].tolist() == [
        1,
        3,
    ]


def test_output_contains_only_training_structures():
    frame = pd.DataFrame(
        {
            "inchikey14": [
                "A",
                "B",
                "C",
            ],
        }
    )

    training_keys = {
        "A",
        "C",
    }

    result = filter_to_training_structures(
        frame,
        training_keys,
    )

    assert set(
        result[
            "inchikey14"
        ]
    ) <= training_keys


def test_rejects_missing_inchikey_column():
    frame = pd.DataFrame(
        {
            "value": [
                1,
            ],
        }
    )

    with pytest.raises(
        ValueError,
        match="inchikey14",
    ):
        filter_to_training_structures(
            frame,
            {
                "A",
            },
        )


def test_rejects_empty_training_key_set():
    frame = pd.DataFrame(
        {
            "inchikey14": [
                "A",
            ],
        }
    )

    with pytest.raises(
        ValueError,
        match="must not be empty",
    ):
        filter_to_training_structures(
            frame,
            set(),
        )
