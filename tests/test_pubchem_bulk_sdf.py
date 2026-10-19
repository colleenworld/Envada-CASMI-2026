from __future__ import annotations

import gzip
from pathlib import Path

from rdkit import Chem

from casmi26.pubchem.bulk_sdf import (
    extract_matching_candidates,
)


def write_pubchem_sdf(
    path: Path,
) -> None:
    ethanol = Chem.MolFromSmiles(
        "CCO"
    )

    ethanol.SetProp(
        "PUBCHEM_COMPOUND_CID",
        "702",
    )

    ethanol.SetProp(
        "PUBCHEM_MOLECULAR_FORMULA",
        "C2H6O",
    )

    ethanol.SetProp(
        "PUBCHEM_IUPAC_INCHIKEY",
        "LFQSCWFLJHTTHZ-UHFFFAOYSA-N",
    )

    propane = Chem.MolFromSmiles(
        "CCC"
    )

    propane.SetProp(
        "PUBCHEM_COMPOUND_CID",
        "6334",
    )

    propane.SetProp(
        "PUBCHEM_MOLECULAR_FORMULA",
        "C3H8",
    )

    propane.SetProp(
        "PUBCHEM_IUPAC_INCHIKEY",
        "ATUOYWHBWRKTHZ-UHFFFAOYSA-N",
    )

    raw_path = (
        path
        .with_suffix("")
    )

    writer = Chem.SDWriter(
        str(raw_path)
    )

    writer.write(
        ethanol
    )

    writer.write(
        propane
    )

    writer.close()

    with raw_path.open(
        "rb"
    ) as source:
        with gzip.open(
            path,
            "wb",
        ) as target:
            target.write(
                source.read()
            )

    raw_path.unlink()


def test_extract_matching_candidates_filters_formula_set(
    tmp_path: Path,
):
    path = (
        tmp_path
        / "Compound_000000001_000500000.sdf.gz"
    )

    write_pubchem_sdf(
        path
    )

    frame, stats = (
        extract_matching_candidates(
            path,
            {
                "C2H6O",
            },
        )
    )

    assert len(frame) == 1

    row = frame.iloc[0]

    assert (
        row.molecular_formula
        == "C2H6O"
    )

    assert (
        row.inchikey14
        == "LFQSCWFLJHTTHZ"
    )

    assert row.smiles == "CCO"

    assert stats == {
        "records_seen": 2,
        "invalid_records": 0,
        "matched_records": 1,
        "unique_candidates": 1,
    }
