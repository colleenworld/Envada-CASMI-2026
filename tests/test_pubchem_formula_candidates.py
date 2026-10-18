import pandas as pd

from casmi26.pubchem.formula_candidates import (
    candidate_burden,
    parse_formula_properties,
)


def test_parse_formula_properties_dedupes_connectivity():
    properties = [
        {
            "CID": 10,
            "InChIKey":
                "ABCDEFGHIJKLMN-AAAAAAAASA-N",
            "ConnectivitySMILES":
                "CCO",
            "MolecularFormula":
                "C2H6O",
        },
        {
            "CID": 11,
            "InChIKey":
                "ABCDEFGHIJKLMN-BBBBBBBBSA-N",
            "ConnectivitySMILES":
                "CCO",
            "MolecularFormula":
                "C2H6O",
        },
        {
            "CID": 12,
            "InChIKey":
                "ZZZZZZZZZZZZZZ-AAAAAAAASA-N",
            "ConnectivitySMILES":
                "COC",
            "MolecularFormula":
                "C2H6O",
        },
    ]

    result = parse_formula_properties(
        "C2H6O",
        properties,
    )

    assert len(result) == 2

    assert set(
        result[
            "inchikey14"
        ]
    ) == {
        "ABCDEFGHIJKLMN",
        "ZZZZZZZZZZZZZZ",
    }


def test_parse_formula_properties_rejects_wrong_formula():
    properties = [
        {
            "CID": 1,
            "InChIKey":
                "ABCDEFGHIJKLMN-AAAAAAAASA-N",
            "ConnectivitySMILES":
                "CC",
            "MolecularFormula":
                "C2H6",
        },
    ]

    result = parse_formula_properties(
        "C2H6O",
        properties,
    )

    assert result.empty


def test_candidate_burden_includes_zero_candidate_formulas():
    candidates = pd.DataFrame(
        {
            "molecular_formula": [
                "A",
                "A",
                "B",
            ],
            "inchikey14": [
                "X",
                "Y",
                "Z",
            ],
        }
    )

    result = candidate_burden(
        candidates,
        [
            "A",
            "B",
            "C",
        ],
    )

    counts = dict(
        zip(
            result[
                "molecular_formula"
            ],
            result[
                "candidate_count"
            ],
            strict=True,
        )
    )

    assert counts == {
        "A": 2,
        "B": 1,
        "C": 0,
    }
