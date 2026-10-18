from __future__ import annotations

import pandas as pd


CANDIDATE_COLUMNS = [
    "molecular_formula",
    "pubchem_cid",
    "full_inchikey",
    "inchikey14",
    "smiles",
]


def parse_formula_properties(
    molecular_formula: str,
    properties: list[dict],
) -> pd.DataFrame:
    """
    Normalize a PubChem property-table response for one exact formula.

    Candidates are collapsed to InChIKey14 because the competition's
    structure identity ignores stereochemistry at that level.
    """
    rows = []

    for item in properties:
        inchikey = item.get(
            "InChIKey"
        )

        if not inchikey:
            continue

        returned_formula = item.get(
            "MolecularFormula"
        )

        if (
            returned_formula is not None
            and str(returned_formula)
            != str(molecular_formula)
        ):
            continue

        smiles = (
            item.get(
                "ConnectivitySMILES"
            )
            or item.get(
                "CanonicalSMILES"
            )
            or item.get(
                "SMILES"
            )
            or item.get(
                "IsomericSMILES"
            )
        )

        if not smiles:
            continue

        rows.append(
            {
                "molecular_formula":
                    str(molecular_formula),
                "pubchem_cid":
                    item.get("CID"),
                "full_inchikey":
                    str(inchikey),
                "inchikey14":
                    str(inchikey)[:14],
                "smiles":
                    str(smiles),
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=CANDIDATE_COLUMNS
        )

    frame = pd.DataFrame(
        rows,
        columns=CANDIDATE_COLUMNS,
    )

    frame[
        "pubchem_cid"
    ] = pd.to_numeric(
        frame[
            "pubchem_cid"
        ],
        errors="coerce",
    ).astype(
        "Int64"
    )

    frame = (
        frame
        .sort_values(
            [
                "molecular_formula",
                "inchikey14",
                "pubchem_cid",
                "full_inchikey",
            ],
            na_position="last",
        )
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

    return frame


def candidate_burden(
    candidates: pd.DataFrame,
    formulas: list[str],
) -> pd.DataFrame:
    """
    Return one row per requested formula with its unique-structure burden.
    """
    requested = pd.DataFrame(
        {
            "molecular_formula":
                sorted(
                    {
                        str(value)
                        for value in formulas
                    }
                ),
        }
    )

    if candidates.empty:
        requested[
            "candidate_count"
        ] = 0

        return requested

    counts = (
        candidates.groupby(
            "molecular_formula"
        )[
            "inchikey14"
        ]
        .nunique()
        .rename(
            "candidate_count"
        )
        .reset_index()
    )

    result = requested.merge(
        counts,
        on="molecular_formula",
        how="left",
        validate="one_to_one",
    )

    result[
        "candidate_count"
    ] = (
        result[
            "candidate_count"
        ]
        .fillna(0)
        .astype(int)
    )

    return result
