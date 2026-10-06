from pathlib import Path

import numpy as np
import pandas as pd

from casmi26.chemistry.adducts import neutral_mass
from casmi26.retrieval.index import load_retrieval_index
from casmi26.spectra.binning import bin_spectrum


PROJECT_ROOT = Path(
    "/home/colleen/PycharmProjects/Envada-CASMI-2026"
)

TEST_PATH = (
    PROJECT_ROOT
    / "data/raw/test.parquet"
)

INDEX_DIR = (
    PROJECT_ROOT
    / "data/processed/retrieval_index"
)

POSITIVE_RANKING_PATH = (
    PROJECT_ROOT
    / "data/processed/test/"
      "positive_contrastive_ranked_candidates.parquet"
)

STRUCTURES_PATH = (
    PROJECT_ROOT
    / "data/processed/structure_index/structures.parquet"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data/processed/test/"
      "negative_retrieval_ranked_candidates.parquet"
)

BIN_WIDTH = 1.0
MAX_MZ = 1500.0

#
# Frozen retrieval setting.
#
NEUTRAL_MASS_TOLERANCE_DA = 0.01


def infer_neutral_mass(
    precursor_mz,
    adduct,
):
    mass = neutral_mass(
        float(precursor_mz),
        str(adduct),
    )

    if mass is None:
        return np.nan

    return float(mass)


def aggregate_max(
    candidate_keys,
    scores,
):
    """
    Collapse spectrum-level reference hits to one score per
    candidate InChIKey14, retaining the maximum cosine score.
    """

    if len(candidate_keys) == 0:
        return {}

    frame = pd.DataFrame(
        {
            "candidate_inchikey14":
                candidate_keys,

            "score":
                np.asarray(
                    scores,
                    dtype=np.float32,
                ),
        }
    )

    result = (
        frame.groupby(
            "candidate_inchikey14",
            sort=False,
        )["score"]
        .max()
    )

    return result.to_dict()


def merge_scores(
    destination,
    source,
):
    """
    Merge scores across multiple query spectra for the same
    test molecule using max aggregation, matching the frozen
    validation retrieval behavior.
    """

    for candidate, score in source.items():
        previous = destination.get(
            candidate,
            -np.inf,
        )

        if score > previous:
            destination[
                candidate
            ] = float(score)


def main():
    #
    # Determine which test molecules were NOT handled by the
    # positive contrastive branch.
    #
    test = pd.read_parquet(
        TEST_PATH
    ).reset_index(
        drop=True
    )

    positive = pd.read_parquet(
        POSITIVE_RANKING_PATH,
        columns=[
            "molecule_id",
        ],
    )

    positive_molecules = set(
        positive[
            "molecule_id"
        ].astype(str)
    )

    all_molecules = set(
        test[
            "molecule_id"
        ].astype(str)
    )

    negative_molecules = (
        all_molecules
        - positive_molecules
    )

    print(
        "Total test molecules:",
        len(all_molecules),
    )

    print(
        "Positive contrastive molecules:",
        len(positive_molecules),
    )

    print(
        "Fallback molecules:",
        len(negative_molecules),
    )

    queries = test[
        test[
            "molecule_id"
        ]
        .astype(str)
        .isin(
            negative_molecules
        )
    ].copy()

    print(
        "Fallback query spectra:",
        len(queries),
    )

    print()
    print(
        "Fallback ionization modes:"
    )

    print(
        queries[
            "ionization_mode"
        ]
        .value_counts(
            dropna=False
        )
        .to_string()
    )

    print()
    print(
        "Fallback adducts:"
    )

    print(
        queries[
            "adduct"
        ]
        .value_counts(
            dropna=False
        )
        .to_string()
    )

    #
    # These should genuinely be our negative-only molecules.
    #
    non_negative = queries[
        queries[
            "ionization_mode"
        ] != "negative"
    ]

    if not non_negative.empty:
        raise ValueError(
            "Fallback set unexpectedly contains "
            "non-negative spectra."
        )

    #
    # Load frozen retrieval index.
    #
    print()
    print(
        "Loading retrieval index..."
    )

    metadata, vectors = (
        load_retrieval_index(
            INDEX_DIR
        )
    )

    print(
        "Reference spectra:",
        len(metadata),
    )

    print(
        "Vector shape:",
        vectors.shape,
    )

    #
    # Restrict to negative reference spectra only.
    #
    negative_mask = (
        metadata[
            "ionization_mode"
        ]
        .astype(str)
        .to_numpy()
        == "negative"
    )

    negative_indices = np.flatnonzero(
        negative_mask
    )

    ref_meta = (
        metadata.iloc[
            negative_indices
        ]
        .reset_index(
            drop=True
        )
    )

    ref_vectors = vectors[
        negative_indices
    ]

    print(
        "Negative reference spectra:",
        len(ref_meta),
    )

    #
    # Compute neutral masses for the negative reference library.
    #
    print(
        "Computing negative reference neutral masses..."
    )

    ref_neutral_masses = np.empty(
        len(ref_meta),
        dtype=np.float64,
    )

    for i, row in enumerate(
        ref_meta.itertuples(
            index=False
        )
    ):
        ref_neutral_masses[i] = (
            infer_neutral_mass(
                row.precursor_mz,
                row.adduct,
            )
        )

    finite_reference_mass = (
        np.isfinite(
            ref_neutral_masses
        )
    )

    print(
        "Negative references with supported adduct:",
        int(
            finite_reference_mass.sum()
        ),
    )

    print(
        "Negative references with unsupported adduct:",
        int(
            (
                ~finite_reference_mass
            ).sum()
        ),
    )

    ref_keys = (
        ref_meta[
            "inchikey14"
        ]
        .astype(str)
        .to_numpy()
    )

    #
    # Aggregate retrieval scores across every spectrum belonging
    # to each fallback molecule.
    #
    molecule_scores = {
        molecule_id: {}
        for molecule_id
        in sorted(
            negative_molecules
        )
    }

    filtered_query_spectra = 0
    fallback_query_spectra = 0

    print()
    print(
        "Searching negative spectra..."
    )

    for query_number, row in enumerate(
        queries.itertuples(
            index=False
        ),
        start=1,
    ):
        molecule_id = str(
            row.molecule_id
        )

        query_vector = bin_spectrum(
            row.ms2_mzs,
            row.ms2_normalized_intensities,
            max_mz=MAX_MZ,
            bin_width=BIN_WIDTH,
        )

        query_mass = infer_neutral_mass(
            row.precursor_mz,
            row.adduct,
        )

        #
        # Frozen Baseline-2 behavior:
        #
        # Use same-polarity candidates within ±0.01 Da whenever
        # at least one such candidate exists.
        #
        use_filtered = False
        candidate_indices = None

        if np.isfinite(
            query_mass
        ):
            mass_mask = (
                finite_reference_mass
                & (
                    np.abs(
                        ref_neutral_masses
                        - query_mass
                    )
                    <= NEUTRAL_MASS_TOLERANCE_DA
                )
            )

            candidate_indices = (
                np.flatnonzero(
                    mass_mask
                )
            )

            if len(
                candidate_indices
            ):
                use_filtered = True

        if use_filtered:
            filtered_query_spectra += 1

            candidate_vectors = (
                ref_vectors[
                    candidate_indices
                ]
            )

            candidate_keys = (
                ref_keys[
                    candidate_indices
                ]
            )

        else:
            #
            # Same-negative-mode cosine fallback.
            #
            fallback_query_spectra += 1

            candidate_vectors = (
                ref_vectors
            )

            candidate_keys = (
                ref_keys
            )

        #
        # bin_spectrum() L2 normalizes the query and the retrieval
        # index was built with the same normalized representation,
        # so dot product is cosine similarity.
        #
        spectrum_scores = (
            candidate_vectors
            .dot(
                query_vector
            )
        )

        spectrum_scores = (
            np.asarray(
                spectrum_scores
            )
            .reshape(-1)
        )

        candidate_scores = (
            aggregate_max(
                candidate_keys,
                spectrum_scores,
            )
        )

        merge_scores(
            molecule_scores[
                molecule_id
            ],
            candidate_scores,
        )

        if (
            query_number % 10 == 0
            or query_number == len(
                queries
            )
        ):
            print(
                "  processed "
                f"{query_number}/"
                f"{len(queries)} query spectra"
            )

    print()
    print(
        "Neutral-mass-filtered query spectra:",
        filtered_query_spectra,
    )

    print(
        "Same-mode fallback query spectra:",
        fallback_query_spectra,
    )

    #
    # Turn final molecule-level scores into ranked rows.
    #
    result_rows = []

    for molecule_id in sorted(
        molecule_scores
    ):
        scores = molecule_scores[
            molecule_id
        ]

        ordered = sorted(
            scores.items(),
            key=lambda item: (
                -item[1],
                item[0],
            ),
        )

        for rank, (
            candidate_ikey,
            cosine_score,
        ) in enumerate(
            ordered,
            start=1,
        ):
            result_rows.append(
                {
                    "molecule_id":
                        molecule_id,

                    "candidate_inchikey14":
                        candidate_ikey,

                    "retrieval_cosine":
                        float(
                            cosine_score
                        ),

                    "rank":
                        rank,

                    "ranking_method":
                        "negative_neutral_mass_retrieval",
                }
            )

    ranked = pd.DataFrame(
        result_rows
    )

    if ranked.empty:
        raise RuntimeError(
            "Negative retrieval produced no rankings."
        )

    #
    # Attach training SMILES where available.
    #
    structures = pd.read_parquet(
        STRUCTURES_PATH,
        columns=[
            "inchikey14",
            "normalized_smiles",
        ],
    )

    structures = (
        structures.drop_duplicates(
            subset=[
                "inchikey14",
            ],
            keep="first",
        )
        .rename(
            columns={
                "inchikey14":
                    "candidate_inchikey14",

                "normalized_smiles":
                    "candidate_smiles",
            }
        )
    )

    ranked = ranked.merge(
        structures,
        on="candidate_inchikey14",
        how="left",
        validate="many_to_one",
    )

    #
    # Keep useful ranking columns together.
    #
    ranked = ranked[
        [
            "molecule_id",
            "candidate_inchikey14",
            "candidate_smiles",
            "retrieval_cosine",
            "rank",
            "ranking_method",
        ]
    ]

    ranked.to_parquet(
        OUTPUT_PATH,
        index=False,
    )

    counts = (
        ranked.groupby(
            "molecule_id"
        )
        .size()
    )

    print()
    print(
        "Negative retrieval ranking"
    )
    print(
        "--------------------------"
    )

    print(
        "Ranked molecules:",
        ranked[
            "molecule_id"
        ].nunique(),
    )

    print(
        "Ranked candidate rows:",
        len(ranked),
    )

    print(
        "Candidates/molecule min:",
        int(
            counts.min()
        ),
    )

    print(
        "Candidates/molecule median:",
        float(
            counts.median()
        ),
    )

    print(
        "Candidates/molecule max:",
        int(
            counts.max()
        ),
    )

    print(
        "Molecules with >=25 candidates:",
        int(
            (
                counts >= 25
            ).sum()
        ),
    )

    missing_smiles = (
        ranked[
            "candidate_smiles"
        ]
        .isna()
        .sum()
    )

    print(
        "Candidate rows missing SMILES:",
        int(
            missing_smiles
        ),
    )

    print()
    print(
        "Top candidate per fallback molecule:"
    )

    print(
        ranked[
            ranked[
                "rank"
            ]
            == 1
        ][
            [
                "molecule_id",
                "candidate_inchikey14",
                "retrieval_cosine",
            ]
        ]
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