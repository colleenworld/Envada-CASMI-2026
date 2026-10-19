from __future__ import annotations

import gzip
from pathlib import Path

import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

from casmi26.pubchem.formula_candidates import (
    CANDIDATE_COLUMNS,
)


CID_PROPERTIES = (
    "PUBCHEM_COMPOUND_CID",
    "PUBCHEM_CID",
)

FORMULA_PROPERTIES = (
    "PUBCHEM_MOLECULAR_FORMULA",
)

INCHIKEY_PROPERTIES = (
    "PUBCHEM_IUPAC_INCHIKEY",
    "PUBCHEM_INCHIKEY",
)


def first_property(
    mol: Chem.Mol,
    names: tuple[str, ...],
) -> str | None:
    for name in names:
        if mol.HasProp(name):
            value = mol.GetProp(name).strip()

            if value:
                return value

    return None


def pubchem_record(
    mol: Chem.Mol,
    target_formulas: set[str],
) -> dict | None:
    formula = first_property(
        mol,
        FORMULA_PROPERTIES,
    )

    if not formula:
        try:
            formula = (
                rdMolDescriptors.CalcMolFormula(
                    mol
                )
            )
        except Exception:
            return None

    formula = str(formula)

    if formula not in target_formulas:
        return None

    inchikey = first_property(
        mol,
        INCHIKEY_PROPERTIES,
    )

    if not inchikey:
        try:
            inchikey = Chem.MolToInchiKey(
                mol
            )
        except Exception:
            return None

    if not inchikey:
        return None

    try:
        smiles = Chem.MolToSmiles(
            mol,
            canonical=True,
            isomericSmiles=False,
        )
    except Exception:
        return None

    if not smiles:
        return None

    cid_value = first_property(
        mol,
        CID_PROPERTIES,
    )

    try:
        cid = (
            int(cid_value)
            if cid_value
            else None
        )
    except ValueError:
        cid = None

    return {
        "molecular_formula":
            formula,
        "pubchem_cid":
            cid,
        "full_inchikey":
            str(inchikey),
        "inchikey14":
            str(inchikey)[:14],
        "smiles":
            str(smiles),
    }


def extract_matching_candidates(
    sdf_gz_path: Path,
    target_formulas: set[str],
) -> tuple[
    pd.DataFrame,
    dict[str, int],
]:
    rows = []

    records_seen = 0
    invalid_records = 0
    matched_records = 0

    with gzip.open(
        sdf_gz_path,
        "rb",
    ) as handle:
        supplier = (
            Chem.ForwardSDMolSupplier(
                handle,
                sanitize=True,
                removeHs=True,
            )
        )

        for mol in supplier:
            records_seen += 1

            if mol is None:
                invalid_records += 1
                continue

            record = pubchem_record(
                mol,
                target_formulas,
            )

            if record is None:
                continue

            matched_records += 1
            rows.append(
                record
            )

    if rows:
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

    else:
        frame = pd.DataFrame(
            columns=CANDIDATE_COLUMNS
        )

    stats = {
        "records_seen":
            records_seen,
        "invalid_records":
            invalid_records,
        "matched_records":
            matched_records,
        "unique_candidates":
            len(frame),
    }

    return frame, stats
