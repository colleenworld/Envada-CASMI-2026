from __future__ import annotations

import os
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(
    os.environ.get(
        "CASMI26_PROJECT_ROOT",
        Path(__file__).resolve().parent.parent,
    )
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


def find_smiles_column(
    df,
    label,
):
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


def empty_prepared_rankings():
    return pd.DataFrame(
        columns=[
            "molecule_id",
            "smiles",
            "rank",
            "submission_rank",
        ]
    )


def prepare_rankings(
    df,
    *,
    label,
):
    #
    # An empty ranking branch is legitimate.
    #
    # For example, the positive MIST branch may be empty and all
    # molecules may be handled by same-polarity fallback retrieval.
    #
    if df.empty:
        print(
            f"{label}: ranking branch is empty."
        )

        return empty_prepared_rankings()

    required = {
        "molecule_id",
        "rank",
    }

    missing = (
        required
        - set(
            df.columns
        )
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
            smiles_col:
                "smiles",
        }
    )

    work[
        "molecule_id"
    ] = (
        work[
            "molecule_id"
        ]
        .astype(str)
    )

    #
    # A submission prediction must have a usable SMILES.
    #
    missing_smiles = (
        work[
            "smiles"
        ]
        .isna()
        .sum()
    )

    if missing_smiles:
        raise ValueError(
            f"{label}: {missing_smiles} candidate rows "
            "have missing SMILES."
        )

    work[
        "smiles"
    ] = (
        work[
            "smiles"
        ]
        .astype(str)
        .str.strip()
    )

    empty_smiles = (
        work[
            "smiles"
        ]
        == ""
    ).sum()

    if empty_smiles:
        raise ValueError(
            f"{label}: {empty_smiles} candidate rows "
            "have empty SMILES."
        )

    #
    # Rank must be numeric and positive.
    #
    work[
        "rank"
    ] = pd.to_numeric(
        work[
            "rank"
        ],
        errors="coerce",
    )

    invalid_rank = (
        work[
            "rank"
        ]
        .isna()
        | (
            work[
                "rank"
            ]
            < 1
        )
    )

    if invalid_rank.any():
        raise ValueError(
            f"{label}: {int(invalid_rank.sum())} candidate rows "
            "have invalid rank values."
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
    work[
        "submission_rank"
    ] = (
        work.groupby(
            "molecule_id"
        )
        .cumcount()
        + 1
    )

    work = work[
        work[
            "submission_rank"
        ]
        <= MAX_CANDIDATES
    ].copy()

    return work


def main():
    sample = pd.read_csv(
        SAMPLE_PATH
    )

    sample[
        "molecule_id"
    ] = (
        sample[
            "molecule_id"
        ]
        .astype(str)
    )

    positive_raw = pd.read_parquet(
        POSITIVE_PATH
    )

    negative_raw = pd.read_parquet(
        NEGATIVE_PATH
    )

    print(
        "Input rankings"
    )

    print(
        "--------------"
    )

    print(
        "Positive rows:",
        len(
            positive_raw
        ),
    )

    print(
        "Positive molecules:",
        (
            positive_raw[
                "molecule_id"
            ].nunique()
            if (
                "molecule_id"
                in positive_raw.columns
            )
            else 0
        ),
    )

    print(
        "Fallback rows:",
        len(
            negative_raw
        ),
    )

    print(
        "Fallback molecules:",
        (
            negative_raw[
                "molecule_id"
            ].nunique()
            if (
                "molecule_id"
                in negative_raw.columns
            )
            else 0
        ),
    )

    positive = prepare_rankings(
        positive_raw,
        label="positive",
    )

    negative = prepare_rankings(
        negative_raw,
        label="fallback",
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

    #
    # This remains strict: a molecule should be owned by exactly one
    # ranking branch.
    #
    if overlap:
        raise ValueError(
            "Positive and fallback ranking branches overlap: "
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
        ]
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

    print()
    print(
        "Coverage"
    )

    print(
        "--------"
    )

    print(
        "Expected molecules:",
        len(
            expected_ids
        ),
    )

    print(
        "Ranked molecules:",
        len(
            actual_ids
        ),
    )

    print(
        "Missing molecules:",
        len(
            missing_ids
        ),
    )

    print(
        "Unexpected molecules:",
        len(
            unexpected_ids
        ),
    )

    #
    # Keep this strict. If fallback fails to cover a hidden molecule,
    # that is a real inference failure and should not be silently hidden.
    #
    if missing_ids:
        raise ValueError(
            "Submission is missing molecule IDs: "
            f"{missing_ids[:50]}"
        )

    if unexpected_ids:
        raise ValueError(
            "Submission contains unexpected molecule IDs: "
            f"{unexpected_ids[:50]}"
        )

    #
    # Convert ranked rows into Kaggle's required semicolon-separated
    # candidate list.
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
        )[
            "smiles"
        ]
        .apply(
            lambda values:
                ";".join(
                    values
                )
        )
        .rename(
            "smiles"
        )
        .reset_index()
    )

    #
    # Preserve exact sample_submission ordering.
    #
    submission = (
        sample[
            [
                "molecule_id",
            ]
        ]
        .copy()
    )

    submission = submission.merge(
        prediction,
        on="molecule_id",
        how="left",
        validate="one_to_one",
    )

    if submission[
        "smiles"
    ].isna().any():
        bad = (
            submission.loc[
                submission[
                    "smiles"
                ].isna(),
                "molecule_id",
            ]
            .tolist()
        )

        raise ValueError(
            "Missing final predictions for: "
            f"{bad[:50]}"
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
    print(
        "Final submission"
    )

    print(
        "----------------"
    )

    print(
        "Rows:",
        len(
            submission
        ),
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
    # Do not include diagnostic candidate_count in the Kaggle file.
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
    print(
        "First five rows:"
    )

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