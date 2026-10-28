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


def test_load_pubchem_candidates_from_assets_combines_and_deduplicates(
    tmp_path: Path,
):
    from casmi26.pubchem.inference_asset import (
        load_pubchem_candidates_from_assets,
    )

    def write_asset(
        root: Path,
        rows: list[tuple[str, int, str, str]],
    ) -> None:
        candidate_dir = root / "candidates"
        candidate_dir.mkdir(parents=True)

        formula_buckets = pd.DataFrame(
            {
                "molecular_formula": [
                    row[0]
                    for row in rows
                ],
                "bucket": [
                    row[1]
                    for row in rows
                ],
            }
        ).drop_duplicates()

        formula_buckets.to_parquet(
            root / "formula_buckets.parquet",
            index=False,
        )

        for bucket, group in pd.DataFrame(
            rows,
            columns=[
                "molecular_formula",
                "bucket",
                "inchikey14",
                "smiles",
            ],
        ).groupby("bucket"):
            group[
                [
                    "molecular_formula",
                    "inchikey14",
                    "smiles",
                ]
            ].to_parquet(
                candidate_dir
                / f"bucket-{int(bucket):03d}.parquet",
                index=False,
            )

    base = tmp_path / "base"
    expansion = tmp_path / "expansion"

    write_asset(
        base,
        [
            (
                "C2H6O",
                0,
                "LFQSCWFLJHTTHZ",
                "CCO",
            ),
            (
                "C3H8O",
                1,
                "BDERNNFJNOPAEC",
                "CCCO",
            ),
        ],
    )

    write_asset(
        expansion,
        [
            (
                "C3H8O",
                0,
                "BDERNNFJNOPAEC",
                "CCCO",
            ),
            (
                "C4H10O",
                1,
                "LRHPLDYGYMQRHN",
                "CCCCO",
            ),
        ],
    )

    result = load_pubchem_candidates_from_assets(
        [
            base,
            expansion,
        ],
        {
            "C2H6O",
            "C3H8O",
            "C4H10O",
        },
    )

    assert set(
        result[
            "molecular_formula"
        ]
    ) == {
        "C2H6O",
        "C3H8O",
        "C4H10O",
    }

    duplicate = result.loc[
        result[
            "inchikey14"
        ]
        == "BDERNNFJNOPAEC"
    ]

    assert len(
        duplicate
    ) == 1


def test_load_pubchem_candidates_from_assets_rejects_uncovered_formula(
    tmp_path: Path,
):
    from casmi26.pubchem.inference_asset import (
        load_pubchem_candidates_from_assets,
    )

    root = tmp_path / "asset"
    root.mkdir(parents=True)

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
        root / "formula_buckets.parquet",
        index=False,
    )

    candidate_dir = root / "candidates"
    candidate_dir.mkdir()

    pd.DataFrame(
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
    ).to_parquet(
        candidate_dir / "bucket-000.parquet",
        index=False,
    )

    try:
        load_pubchem_candidates_from_assets(
            [
                root,
            ],
            {
                "C9H9NO",
            },
        )
    except KeyError as exc:
        assert "C9H9NO" in str(exc)
    else:
        raise AssertionError(
            "Expected uncovered formula to raise KeyError"
        )
