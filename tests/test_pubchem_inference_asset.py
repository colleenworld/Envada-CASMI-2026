from pathlib import Path

import pandas as pd

from casmi26.pubchem.inference_asset import (
    load_pubchem_candidates_for_formulas,
)


def test_load_pubchem_candidates_reads_only_requested_formulas(
    tmp_path: Path,
):
    root = (
        tmp_path
        / "pubchem_formula_index"
    )

    candidate_dir = (
        root
        / "candidates"
    )

    candidate_dir.mkdir(
        parents=True
    )

    pd.DataFrame(
        {
            "molecular_formula": [
                "C2H6O",
                "C3H8O",
            ],
            "bucket": [
                0,
                1,
            ],
        }
    ).to_parquet(
        root
        / "formula_buckets.parquet",
        index=False,
    )

    bucket0 = pd.DataFrame(
        {
            "molecular_formula": [
                "C2H6O",
            ],
            "inchikey14": [
                "LFQSCWFLJHTTHZ",
            ],
            "smiles": [
                "CCO",
            ],
        }
    )

    bucket1 = pd.DataFrame(
        {
            "molecular_formula": [
                "C3H8O",
            ],
            "inchikey14": [
                "BDERNNFJNOPAEC",
            ],
            "smiles": [
                "CCCO",
            ],
        }
    )

    bucket0.to_parquet(
        candidate_dir
        / "bucket-000.parquet",
        index=False,
    )

    bucket1.to_parquet(
        candidate_dir
        / "bucket-001.parquet",
        index=False,
    )

    result = (
        load_pubchem_candidates_for_formulas(
            root,
            {
                "C3H8O",
            },
        )
    )

    assert (
        result[
            "molecular_formula"
        ].tolist()
        == [
            "C3H8O",
        ]
    )

    assert (
        result[
            "inchikey14"
        ].tolist()
        == [
            "BDERNNFJNOPAEC",
        ]
    )


def test_load_pubchem_candidates_rejects_missing_formula(
    tmp_path: Path,
):
    root = (
        tmp_path
        / "pubchem_formula_index"
    )

    root.mkdir(
        parents=True
    )

    pd.DataFrame(
        {
            "molecular_formula": [
                "C2H6O",
            ],
            "bucket": [
                0,
            ],
        }
    ).to_parquet(
        root
        / "formula_buckets.parquet",
        index=False,
    )

    try:
        load_pubchem_candidates_for_formulas(
            root,
            {
                "C9H9NO",
            },
        )
    except KeyError as exc:
        assert (
            "C9H9NO"
            in str(exc)
        )
    else:
        raise AssertionError(
            "Expected missing formula to raise KeyError"
        )
