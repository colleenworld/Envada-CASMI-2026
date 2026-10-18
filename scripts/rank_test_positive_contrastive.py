from __future__ import annotations

import os
import pickle
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(
    os.environ.get(
        "CASMI26_PROJECT_ROOT",
        Path(__file__).resolve().parent.parent,
    )
)

TEST_ROOT = (
    PROJECT_ROOT
    / "data/processed/test"
)

HYPOTHESES_PATH = (
    TEST_ROOT
    / "multispectrum_formula_hypotheses.parquet"
)

PREDICTIONS_PATH = (
    TEST_ROOT
    / "contrastive_predictions_positive"
    / (
        "retrieval_contrastive_"
        "casmi_test_morgan4096_"
        "with_morgan4096_retrieval_db_"
        "casmi_test_positive_contrastive_"
        "cosine.p"
    )
)

OUTPUT_PATH = (
    TEST_ROOT
    / "positive_contrastive_ranked_candidates.parquet"
)


#
# Frozen on v1, validated once on v2.
#
MASS_PENALTY = 0.020


OUTPUT_COLUMNS = [
    "molecule_id",
    "hypothesis_spec",
    "candidate_formula",
    "candidate_inchikey14",
    "candidate_smiles",
    "selected_adduct",
    "selected_mass_error_ppm",
    "contrastive_distance",
    "final_score",
    "rank",
]


def to_str(value):
    if isinstance(
        value,
        bytes,
    ):
        return value.decode()

    return str(value)


def write_empty_output(
    message: str,
) -> None:
    print()
    print(message)

    empty = pd.DataFrame(
        columns=OUTPUT_COLUMNS
    )

    empty.to_parquet(
        OUTPUT_PATH,
        index=False,
    )

    print(
        "Wrote empty positive ranking."
    )

    print(
        "These molecules will be handled by "
        "the same-polarity retrieval fallback."
    )

    print()
    print(
        f"Wrote {OUTPUT_PATH}"
    )


def main():
    hypotheses = pd.read_parquet(
        HYPOTHESES_PATH
    )

    #
    # Only hypotheses supported by the positive MIST checkpoint.
    #
    supported_adducts = {
        "[M+H]+",
        "[M+Na]+",
        "[M+NH4]+",
        "[M+K]+",
    }

    hypotheses = hypotheses[
        hypotheses[
            "selected_adduct"
        ].isin(
            supported_adducts
        )
    ].copy()

    print(
        "Supported hypotheses:",
        len(hypotheses),
    )

    print(
        "Supported molecules:",
        hypotheses[
            "molecule_id"
        ].nunique(),
    )

    #
    # A hidden test may legitimately have no positive hypotheses.
    # That is not fatal: the fallback branch will handle all molecules.
    #
    if hypotheses.empty:
        write_empty_output(
            "No MIST-supported positive hypotheses."
        )
        return

    if hypotheses[
        "hypothesis_spec"
    ].duplicated().any():
        raise ValueError(
            "Duplicate hypothesis_spec values."
        )

    hypothesis_lookup = (
        hypotheses
        .set_index(
            "hypothesis_spec"
        )
    )

    #
    # If we have positive hypotheses, the MIST retrieval artifact
    # should exist. Missing it means the upstream MIST step failed.
    #
    if not PREDICTIONS_PATH.exists():
        raise FileNotFoundError(
            "Expected positive MIST prediction artifact was not found: "
            f"{PREDICTIONS_PATH}"
        )

    with PREDICTIONS_PATH.open(
        "rb"
    ) as f:
        retrieval = pickle.load(f)

    print(
        "Retrieval keys:",
        list(
            retrieval.keys()
        ),
    )

    required_retrieval_keys = {
        "names",
        "ikeys",
        "smiles",
        "dists",
    }

    missing_keys = (
        required_retrieval_keys
        - set(
            retrieval.keys()
        )
    )

    if missing_keys:
        raise ValueError(
            "Retrieval artifact is missing keys: "
            f"{sorted(missing_keys)}"
        )

    names = [
        to_str(x)
        for x in retrieval[
            "names"
        ]
    ]

    ikey_lists = retrieval[
        "ikeys"
    ]

    smiles_lists = retrieval[
        "smiles"
    ]

    dist_lists = retrieval[
        "dists"
    ]

    print(
        "Retrieved hypotheses:",
        len(names),
    )

    if not (
        len(names)
        == len(ikey_lists)
        == len(smiles_lists)
        == len(dist_lists)
    ):
        raise ValueError(
            "Retrieval arrays have inconsistent lengths."
        )

    #
    # Empty retrieval output is legal. The fallback branch will handle
    # any molecules that do not receive a positive MIST ranking.
    #
    if not names:
        write_empty_output(
            "MIST returned no positive retrieval hypotheses."
        )
        return

    #
    # Every retrieval hypothesis should correspond to one of our
    # supported test hypotheses.
    #
    unexpected = (
        set(names)
        - set(
            hypothesis_lookup.index
        )
    )

    if unexpected:
        raise ValueError(
            "Retrieval returned unknown hypotheses. "
            f"Examples: {sorted(unexpected)[:10]}"
        )

    missing = (
        set(
            hypothesis_lookup.index
        )
        - set(names)
    )

    print(
        "Supported hypotheses missing retrieval:",
        len(missing),
    )

    if missing:
        print(
            "First missing:",
            sorted(missing)[:20],
        )

    rows = []

    #
    # Expand each hypothesis's ranked structure list and attach the
    # frozen cross-formula score.
    #
    for (
        hypothesis_spec,
        ikeys,
        smiles,
        dists,
    ) in zip(
        names,
        ikey_lists,
        smiles_lists,
        dist_lists,
    ):
        meta = hypothesis_lookup.loc[
            hypothesis_spec
        ]

        molecule_id = (
            meta[
                "molecule_id"
            ]
        )

        formula = (
            meta[
                "candidate_formula"
            ]
        )

        mass_error_ppm = float(
            meta[
                "selected_mass_error_ppm"
            ]
        )

        adduct = (
            meta[
                "selected_adduct"
            ]
        )

        if not (
            len(ikeys)
            == len(smiles)
            == len(dists)
        ):
            raise ValueError(
                f"Candidate array length mismatch "
                f"for {hypothesis_spec}"
            )

        for (
            candidate_ikey,
            candidate_smiles,
            distance,
        ) in zip(
            ikeys,
            smiles,
            dists,
        ):
            candidate_ikey = to_str(
                candidate_ikey
            )

            candidate_smiles = to_str(
                candidate_smiles
            )

            distance = float(
                distance
            )

            final_score = (
                -distance
                - MASS_PENALTY
                * abs(
                    mass_error_ppm
                )
            )

            rows.append(
                {
                    "molecule_id":
                        molecule_id,

                    "hypothesis_spec":
                        hypothesis_spec,

                    "candidate_formula":
                        formula,

                    "candidate_inchikey14":
                        candidate_ikey,

                    "candidate_smiles":
                        candidate_smiles,

                    "selected_adduct":
                        adduct,

                    "selected_mass_error_ppm":
                        mass_error_ppm,

                    "contrastive_distance":
                        distance,

                    "final_score":
                        final_score,
                }
            )

    #
    # MIST may return hypothesis names but no candidate structures.
    # That is also a valid fallback case.
    #
    if not rows:
        write_empty_output(
            "MIST returned no positive candidate structures."
        )
        return

    ranked = pd.DataFrame(
        rows
    )

    print()
    print(
        "Expanded retrieval rows:",
        len(ranked),
    )

    print(
        "Molecules before collapse:",
        ranked[
            "molecule_id"
        ].nunique(),
    )

    #
    # A structure can theoretically occur via more than one formula
    # hypothesis. Competition ranking is by structure, not hypothesis.
    #
    # Keep whichever occurrence has the highest frozen final score.
    #
    ranked = ranked.sort_values(
        [
            "molecule_id",
            "candidate_inchikey14",
            "final_score",
        ],
        ascending=[
            True,
            True,
            False,
        ],
    )

    ranked = ranked.drop_duplicates(
        subset=[
            "molecule_id",
            "candidate_inchikey14",
        ],
        keep="first",
    )

    ranked = ranked.sort_values(
        [
            "molecule_id",
            "final_score",
            "candidate_inchikey14",
        ],
        ascending=[
            True,
            False,
            True,
        ],
    ).reset_index(
        drop=True
    )

    ranked[
        "rank"
    ] = (
        ranked.groupby(
            "molecule_id"
        )
        .cumcount()
        + 1
    )

    #
    # Keep the output schema stable even if pandas column ordering
    # changes during earlier operations.
    #
    ranked = ranked[
        OUTPUT_COLUMNS
    ]

    ranked.to_parquet(
        OUTPUT_PATH,
        index=False,
    )

    print()
    print(
        "Positive contrastive ranking"
    )

    print(
        "----------------------------"
    )

    print(
        "Ranked rows:",
        len(ranked),
    )

    print(
        "Ranked molecules:",
        ranked[
            "molecule_id"
        ].nunique(),
    )

    counts = (
        ranked.groupby(
            "molecule_id"
        )
        .size()
    )

    print(
        "Candidates/molecule mean:",
        float(
            counts.mean()
        ),
    )

    print(
        "Candidates/molecule median:",
        float(
            counts.median()
        ),
    )

    print(
        "Candidates/molecule min:",
        int(
            counts.min()
        ),
    )

    print(
        "Candidates/molecule max:",
        int(
            counts.max()
        ),
    )

    top25 = ranked[
        ranked[
            "rank"
        ]
        <= 25
    ]

    print(
        "Top-25 rows:",
        len(top25),
    )

    print(
        "Molecules with >=25 candidates:",
        int(
            (
                counts
                >= 25
            ).sum()
        ),
    )

    print()
    print(
        "Top-ranked adducts:"
    )

    print(
        ranked[
            ranked[
                "rank"
            ]
            == 1
        ][
            "selected_adduct"
        ]
        .value_counts(
            dropna=False
        )
        .to_string()
    )

    print()
    print(
        f"Wrote {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()