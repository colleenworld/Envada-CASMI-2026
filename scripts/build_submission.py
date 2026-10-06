from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(
    "/home/colleen/PycharmProjects/Envada-CASMI-2026"
)

SAMPLE_PATH = (
    PROJECT_ROOT
    / "data/raw/sample_submission.csv"
)

POSITIVE_PATH = (
    PROJECT_ROOT
    / "data/processed/test/"
      "positive_contrastive_ranked_candidates.parquet"
)

NEGATIVE_PATH = (
    PROJECT_ROOT
    / "data/processed/test/"
      "negative_retrieval_ranked_candidates.parquet"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "submission.csv"
)

MAX_CANDIDATES = 25


def find_smiles_column(df, label):
    """
    Locate the candidate SMILES column without silently guessing
    between multiple possibilities.
    """
    candidates = [
        col
        for col in [
            "candidate_smiles",
            "normalized_smiles",
            "smiles",
        ]
        if col in df.columns
    ]

    if len(candidates) != 1:
        raise ValueError(
            f"{label}: expected exactly one SMILES column "
            f"from candidate_smiles/normalized_smiles/smiles; "
            f"found {candidates}"
        )

    return candidates[0]


def prepare_rankings(
    df,
    *,
    label,
):
    required = {
        "molecule_id",
        "rank",
    }

    missing = (
        required
        - set(df.columns)
    )

    if missing:
        raise ValueError(
            f"{label}: missing columns "
            f"{sorted(missing)}"
        )

    smiles_col = find_smiles_column(
        df,
        label,
    )

    work = df[
        [
            "molecule_id",
            smiles_col,
            "rank",
        ]
    ].copy()

    work = work.rename(
        columns={
            smiles_col: "smiles",
        }
    )

    work["molecule_id"] = (
        work["molecule_id"]
        .astype(str)
    )

    #
    # A submission prediction must have a usable SMILES.
    #
    missing_smiles = (
        work["smiles"]
        .isna()
        .sum()
    )

    if missing_smiles:
        raise ValueError(
            f"{label}: {missing_smiles} candidate rows "
            "have missing SMILES."
        )

    work["smiles"] = (
        work["smiles"]
        .astype(str)
        .str.strip()
    )

    empty_smiles = (
        work["smiles"]
        == ""
    ).sum()

    if empty_smiles:
        raise ValueError(
            f"{label}: {empty_smiles} candidate rows "
            "have empty SMILES."
        )

    #
    # Preserve ranking order.
    #
    work = work.sort_values(
        by=[
            "molecule_id",
            "rank",
        ],
        ascending=[
            True,
            True,
        ],
    )

    #
    # Protect against the same structure/SMILES appearing more
    # than once for a molecule.
    #
    work = work.drop_duplicates(
        subset=[
            "molecule_id",
            "smiles",
        ],
        keep="first",
    )

    #
    # Re-rank after deduplication and keep at most 25.
    #
    work["submission_rank"] = (
        work.groupby(
            "molecule_id"
        )
        .cumcount()
        + 1
    )

    work = work[
        work["submission_rank"]
        <= MAX_CANDIDATES
    ].copy()

    return work


def main():
    sample = pd.read_csv(
        SAMPLE_PATH
    )

    positive_raw = pd.read_parquet(
        POSITIVE_PATH
    )

    negative_raw = pd.read_parquet(
        NEGATIVE_PATH
    )

    print("Input rankings")
    print("--------------")
    print(
        "Positive molecules:",
        positive_raw[
            "molecule_id"
        ].nunique(),
    )
    print(
        "Negative molecules:",
        negative_raw[
            "molecule_id"
        ].nunique(),
    )

    positive = prepare_rankings(
        positive_raw,
        label="positive",
    )

    negative = prepare_rankings(
        negative_raw,
        label="negative",
    )

    positive_ids = set(
        positive[
            "molecule_id"
        ]
    )

    negative_ids = set(
        negative[
            "molecule_id"
        ]
    )

    overlap = (
        positive_ids
        & negative_ids
    )

    if overlap:
        raise ValueError(
            "Positive and negative ranking branches overlap: "
            f"{sorted(overlap)[:20]}"
        )

    combined = pd.concat(
        [
            positive,
            negative,
        ],
        ignore_index=True,
    )

    #
    # Check molecule coverage against the official sample.
    #
    expected_ids = set(
        sample[
            "molecule_id"
        ].astype(str)
    )

    actual_ids = set(
        combined[
            "molecule_id"
        ]
    )

    missing_ids = sorted(
        expected_ids
        - actual_ids
    )

    unexpected_ids = sorted(
        actual_ids
        - expected_ids
    )

    if missing_ids:
        raise ValueError(
            "Submission is missing molecule IDs: "
            f"{missing_ids}"
        )

    if unexpected_ids:
        raise ValueError(
            "Submission contains unexpected molecule IDs: "
            f"{unexpected_ids}"
        )

    #
    # Convert the ranked rows into Kaggle's required
    # semicolon-separated format.
    #
    prediction = (
        combined
        .sort_values(
            [
                "molecule_id",
                "submission_rank",
            ]
        )
        .groupby(
            "molecule_id",
            sort=False,
        )["smiles"]
        .apply(
            lambda values:
                ";".join(values)
        )
        .rename(
            "smiles"
        )
        .reset_index()
    )

    #
    # Preserve the exact molecule ordering of sample_submission.csv.
    #
    submission = (
        sample[
            [
                "molecule_id",
            ]
        ]
        .copy()
    )

    submission[
        "molecule_id"
    ] = submission[
        "molecule_id"
    ].astype(str)

    submission = submission.merge(
        prediction,
        on="molecule_id",
        how="left",
        validate="one_to_one",
    )

    if submission[
        "smiles"
    ].isna().any():
        bad = submission.loc[
            submission[
                "smiles"
            ].isna(),
            "molecule_id",
        ].tolist()

        raise ValueError(
            "Missing final predictions for: "
            f"{bad}"
        )

    #
    # Final validation of list lengths.
    #
    submission[
        "candidate_count"
    ] = (
        submission[
            "smiles"
        ]
        .str.split(";")
        .str.len()
    )

    if (
        submission[
            "candidate_count"
        ]
        > MAX_CANDIDATES
    ).any():
        raise ValueError(
            "At least one molecule has more than "
            f"{MAX_CANDIDATES} predictions."
        )

    if (
        submission[
            "candidate_count"
        ]
        < 1
    ).any():
        raise ValueError(
            "At least one molecule has no predictions."
        )

    print()
    print("Final submission")
    print("----------------")
    print(
        "Rows:",
        len(submission),
    )
    print(
        "Unique molecules:",
        submission[
            "molecule_id"
        ].nunique(),
    )

    print()
    print(
        "Candidate count distribution:"
    )

    print(
        submission[
            "candidate_count"
        ]
        .describe()
        .to_string()
    )

    print()
    print(
        "Molecules with 25 candidates:",
        int(
            (
                submission[
                    "candidate_count"
                ]
                == 25
            ).sum()
        ),
    )

    print(
        "Molecules with fewer than 25:",
        int(
            (
                submission[
                    "candidate_count"
                ]
                < 25
            ).sum()
        ),
    )

    print()
    print(
        "Candidate-count frequencies:"
    )

    print(
        submission[
            "candidate_count"
        ]
        .value_counts()
        .sort_index()
        .to_string()
    )

    #
    # Do not include our diagnostic column in the Kaggle file.
    #
    submission[
        [
            "molecule_id",
            "smiles",
        ]
    ].to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print()
    print("First five rows:")
    print(
        submission[
            [
                "molecule_id",
                "smiles",
            ]
        ]
        .head()
        .to_string(
            index=False
        )
    )

    print()
    print(
        f"Wrote {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()