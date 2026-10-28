from __future__ import annotations

from pathlib import Path

import pandas as pd


CANDIDATE_COLUMNS = [
    "molecular_formula",
    "inchikey14",
    "smiles",
]


def available_formulas(
    asset_root: Path,
) -> set[str]:
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
        formula_buckets_path,
        columns=[
            "molecular_formula",
        ],
    )

    return set(
        formula_buckets[
            "molecular_formula"
        ]
        .dropna()
        .astype(str)
    )


def load_pubchem_candidates_for_formulas(
    asset_root: Path,
    formulas: set[str] | list[str],
) -> pd.DataFrame:
    requested = {
        str(formula)
        for formula in formulas
    }

    if not requested:
        return pd.DataFrame(
            columns=CANDIDATE_COLUMNS
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
            columns=CANDIDATE_COLUMNS
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


def load_pubchem_candidates_from_assets(
    asset_roots: list[Path] | tuple[Path, ...],
    formulas: set[str] | list[str],
) -> pd.DataFrame:
    requested = {
        str(formula)
        for formula in formulas
    }

    if not requested:
        return pd.DataFrame(
            columns=CANDIDATE_COLUMNS
        )

    roots = [
        Path(root)
        for root in asset_roots
    ]

    if not roots:
        raise ValueError(
            "At least one PubChem asset root is required"
        )

    available_by_root = [
        available_formulas(
            root
        )
        for root in roots
    ]

    covered = set().union(
        *available_by_root
    )

    missing = (
        requested
        - covered
    )

    if missing:
        raise KeyError(
            "Requested formulas are missing from all offline "
            "PubChem assets. Examples: "
            f"{sorted(missing)[:20]}"
        )

    frames = []

    for root, available in zip(
        roots,
        available_by_root,
        strict=True,
    ):
        formulas_for_root = (
            requested
            & available
        )

        if not formulas_for_root:
            continue

        frame = (
            load_pubchem_candidates_for_formulas(
                root,
                formulas_for_root,
            )
        )

        if not frame.empty:
            frames.append(
                frame
            )

    if not frames:
        return pd.DataFrame(
            columns=CANDIDATE_COLUMNS
        )

    combined = pd.concat(
        frames,
        ignore_index=True,
    )

    return (
        combined
        .drop_duplicates(
            subset=[
                "molecular_formula",
                "inchikey14",
            ],
            keep="first",
        )
        .reset_index(
            drop=True
        )
    )
