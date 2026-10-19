from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_pubchem_candidates_for_formulas(
    asset_root: Path,
    formulas: set[str] | list[str],
) -> pd.DataFrame:
    requested = {
        str(formula)
        for formula in formulas
    }

    columns = [
        "molecular_formula",
        "pubchem_cid",
        "full_inchikey",
        "inchikey14",
        "smiles",
    ]

    if not requested:
        return pd.DataFrame(
            columns=columns
        )

    formula_buckets_path = (
        asset_root
        / "formula_buckets.parquet"
    )

    if not formula_buckets_path.exists():
        raise FileNotFoundError(
            "PubChem inference formula-bucket index not found: "
            f"{formula_buckets_path}"
        )

    formula_buckets = pd.read_parquet(
        formula_buckets_path
    )

    required = {
        "molecular_formula",
        "bucket",
    }

    missing = (
        required
        - set(formula_buckets.columns)
    )

    if missing:
        raise ValueError(
            "PubChem formula-bucket index is missing columns: "
            f"{sorted(missing)}"
        )

    selected = formula_buckets.loc[
        formula_buckets[
            "molecular_formula"
        ]
        .astype(str)
        .isin(
            requested
        )
    ].copy()

    missing_formulas = (
        requested
        - set(
            selected[
                "molecular_formula"
            ].astype(str)
        )
    )

    if missing_formulas:
        raise KeyError(
            "Requested formulas are missing from the offline "
            "PubChem asset. Examples: "
            f"{sorted(missing_formulas)[:20]}"
        )

    frames = []

    for bucket in sorted(
        selected[
            "bucket"
        ]
        .astype(int)
        .unique()
    ):
        path = (
            asset_root
            / "candidates"
            / f"bucket-{bucket:03d}.parquet"
        )

        if not path.exists():
            raise FileNotFoundError(
                f"PubChem candidate bucket not found: {path}"
            )

        frame = pd.read_parquet(
            path
        )

        frame = frame.loc[
            frame[
                "molecular_formula"
            ]
            .astype(str)
            .isin(
                requested
            )
        ]

        if not frame.empty:
            frames.append(
                frame
            )

    if not frames:
        return pd.DataFrame(
            columns=columns
        )

    result = pd.concat(
        frames,
        ignore_index=True,
    )

    return (
        result
        .drop_duplicates(
            subset=[
                "molecular_formula",
                "inchikey14",
            ]
        )
        .reset_index(
            drop=True
        )
    )
