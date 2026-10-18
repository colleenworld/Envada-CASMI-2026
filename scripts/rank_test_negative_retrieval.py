from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

from casmi26.chemistry.adducts import neutral_mass
from casmi26.retrieval.index import load_retrieval_index
from casmi26.spectra.binning import bin_spectrum


PROJECT_ROOT = Path(
    os.environ.get(
        "CASMI26_PROJECT_ROOT",
        Path(__file__).resolve().parent.parent,
    )
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

#
# Keep the existing output filename so build_submission.py does not
# need to change. The contents are now a generic same-polarity
# retrieval fallback, not a negative-only fallback.
#
OUTPUT_PATH = (
    PROJECT_ROOT
    / "data/processed/test/"
      "negative_retrieval_ranked_candidates.parquet"
)

BIN_WIDTH = 1.0
MAX_MZ = 1500.0

#
# Frozen retrieval setting from the validated retrieval baseline.
#
NEUTRAL_MASS_TOLERANCE_DA = 0.01


OUTPUT_COLUMNS = [
    "molecule_id",
    "candidate_inchikey14",
    "candidate_smiles",
    "retrieval_cosine",
    "rank",
    "ranking_method",
]


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


def write_empty_output():
    """
    A zero-sized fallback branch is valid: all test molecules may
    already have been ranked by the positive contrastive branch.
    Write the expected schema and return successfully.
    """

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    pd.DataFrame(
        columns=OUTPUT_COLUMNS
    ).to_parquet(
        OUTPUT_PATH,
        index=False,
    )

    print()
    print(
        "No fallback molecules require spectral retrieval."
    )
    print(
        f"Wrote empty fallback ranking to {OUTPUT_PATH}"
    )


def build_reference_cache(
    metadata,
    vectors,
    ionization_mode,
):
    """
    Build the same-polarity retrieval view for one ionization mode.

    Neutral masses are computed once for this mode and reused for
    every fallback query spectrum of the same polarity.
    """

    mode_mask = (
        metadata[
            "ionization_mode"
        ]
        .astype(str)
        .to_numpy()
        == ionization_mode
    )

    mode_indices = np.flatnonzero(
        mode_mask
    )

    if len(mode_indices) == 0:
        raise RuntimeError(
            "Retrieval index contains no reference spectra for "
            f"ionization mode {ionization_mode!r}."
        )

    ref_meta = (
        metadata.iloc[
            mode_indices
        ]
        .reset_index(
            drop=True
        )
    )

    ref_vectors = vectors[
        mode_indices
    ]

    print()
    print(
        f"{ionization_mode.capitalize()} reference spectra:",
        len(ref_meta),
    )

    print(
        f"Computing {ionization_mode} reference neutral masses..."
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
        f"{ionization_mode.capitalize()} references "
        "with supported adduct:",
        int(
            finite_reference_mass.sum()
        ),
    )

    print(
        f"{ionization_mode.capitalize()} references "
        "with unsupported adduct:",
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

    return {
        "vectors": ref_vectors,
        "keys": ref_keys,
        "neutral_masses": ref_neutral_masses,
        "finite_mass": finite_reference_mass,
    }


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

    test[
        "molecule_id"
    ] = test[
        "molecule_id"
    ].astype(str)

    all_molecules = set(
        test[
            "molecule_id"
        ]
    )

    if POSITIVE_RANKING_PATH.exists():
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
    else:
        #
        # This also makes the fallback script robust if a hidden
        # test set produces no positive ranking file.
        #
        positive_molecules = set()

        print(
            "Positive ranking file does not exist; "
            "all test molecules will use retrieval fallback."
        )

    fallback_molecules = (
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
        len(fallback_molecules),
    )

    queries = test[
        test[
            "molecule_id"
        ]
        .isin(
            fallback_molecules
        )
    ].copy()

    print(
        "Fallback query spectra:",
        len(queries),
    )

    if not queries.empty:
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
    # An empty fallback branch is valid.
    #
    if not fallback_molecules:
        write_empty_output()
        return

    if queries.empty:
        raise RuntimeError(
            "Fallback molecules exist, but no query spectra "
            "were found for them."
        )

    #
    # Hidden-test robustness:
    #
    # Do NOT assume every molecule missed by positive MIST is
    # negative-only. A positive molecule can miss the MIST branch
    # because none of its formula hypotheses has a structure
    # candidate. Retrieval therefore operates on the query's own
    # ionization mode.
    #
    query_modes = sorted(
        set(
            queries[
                "ionization_mode"
            ]
            .dropna()
            .astype(str)
        )
    )

    unsupported_modes = (
        set(query_modes)
        - {
            "positive",
            "negative",
        }
    )

    if unsupported_modes:
        raise ValueError(
            "Fallback contains unsupported ionization modes: "
            f"{sorted(unsupported_modes)}"
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
    # Build retrieval views only for polarities actually needed by
    # the fallback set. On the visible test this remains negative-only,
    # preserving the previous computation.
    #
    reference_cache = {}

    for mode in query_modes:
        reference_cache[
            mode
        ] = build_reference_cache(
            metadata,
            vectors,
            mode,
        )

    #
    # Aggregate retrieval scores across every spectrum belonging
    # to each fallback molecule.
    #
    molecule_scores = {
        molecule_id: {}
        for molecule_id
        in sorted(
            fallback_molecules
        )
    }

    filtered_query_spectra = 0
    unfiltered_query_spectra = 0

    filtered_by_mode = {
        mode: 0
        for mode in query_modes
    }

    unfiltered_by_mode = {
        mode: 0
        for mode in query_modes
    }

    print()
    print(
        "Searching fallback spectra..."
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

        ionization_mode = str(
            row.ionization_mode
        )

        ref = reference_cache.get(
            ionization_mode
        )

        if ref is None:
            raise RuntimeError(
                "No retrieval reference cache for "
                f"ionization mode {ionization_mode!r}."
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
        # at least one such candidate exists. Otherwise use all
        # same-polarity reference spectra.
        #
        use_filtered = False
        candidate_indices = None

        if np.isfinite(
            query_mass
        ):
            mass_mask = (
                ref[
                    "finite_mass"
                ]
                & (
                    np.abs(
                        ref[
                            "neutral_masses"
                        ]
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
            filtered_by_mode[
                ionization_mode
            ] += 1

            candidate_vectors = (
                ref[
                    "vectors"
                ][
                    candidate_indices
                ]
            )

            candidate_keys = (
                ref[
                    "keys"
                ][
                    candidate_indices
                ]
            )

        else:
            #
            # Same-polarity cosine fallback.
            #
            unfiltered_query_spectra += 1
            unfiltered_by_mode[
                ionization_mode
            ] += 1

            candidate_vectors = (
                ref[
                    "vectors"
                ]
            )

            candidate_keys = (
                ref[
                    "keys"
                ]
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
        "Same-polarity unfiltered query spectra:",
        unfiltered_query_spectra,
    )

    for mode in query_modes:
        print(
            f"  {mode}: "
            f"{filtered_by_mode[mode]} mass-filtered, "
            f"{unfiltered_by_mode[mode]} unfiltered"
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
                        "same_polarity_neutral_mass_retrieval",
                }
            )

    ranked = pd.DataFrame(
        result_rows,
        columns=[
            "molecule_id",
            "candidate_inchikey14",
            "retrieval_cosine",
            "rank",
            "ranking_method",
        ],
    )

    #
    # At this point a non-empty fallback set really should have
    # produced rankings. Report exactly which molecules failed.
    #
    ranked_molecules = set(
        ranked[
            "molecule_id"
        ].astype(str)
    )

    missing_rankings = sorted(
        fallback_molecules
        - ranked_molecules
    )

    if missing_rankings:
        print()
        print(
            "Fallback molecules with no retrieval ranking:",
            len(missing_rankings),
        )

        print(
            test.loc[
                test[
                    "molecule_id"
                ].isin(
                    missing_rankings
                ),
                [
                    "molecule_id",
                    "ionization_mode",
                    "adduct",
                ],
            ]
            .drop_duplicates()
            .sort_values(
                [
                    "molecule_id",
                    "ionization_mode",
                    "adduct",
                ]
            )
            .to_string(
                index=False
            )
        )

        raise RuntimeError(
            f"{len(missing_rankings)} fallback molecules "
            "produced no retrieval ranking."
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
    # A retrieval hit without a usable SMILES cannot appear in a
    # Kaggle submission. Drop those hits and promote the next valid
    # candidate for the molecule.
    #
    missing_smiles = (
            ranked["candidate_smiles"].isna()
            | (
                    ranked["candidate_smiles"]
                    .astype(str)
                    .str.strip()
                    == ""
            )
    )

    if missing_smiles.any():
        print()
        print(
            "Dropping candidate rows without usable SMILES:",
            int(missing_smiles.sum()),
        )

        ranked = (
            ranked.loc[
                ~missing_smiles
            ]
            .copy()
        )

    #
    # Re-rank after removing unusable structures.
    #
    ranked = ranked.sort_values(
        [
            "molecule_id",
            "retrieval_cosine",
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

    ranked["rank"] = (
            ranked.groupby(
                "molecule_id"
            )
            .cumcount()
            + 1
    )

    #
    # Verify every fallback molecule still has at least one usable
    # submission candidate.
    #
    ranked_molecules = set(
        ranked["molecule_id"]
        .astype(str)
    )

    missing_after_smiles = sorted(
        fallback_molecules
        - ranked_molecules
    )

    if missing_after_smiles:
        raise RuntimeError(
            f"{len(missing_after_smiles)} fallback molecules "
            "have no retrieval candidates with usable SMILES. "
            f"Examples: {missing_after_smiles[:20]}"
        )

    ranked = ranked[
        OUTPUT_COLUMNS
    ]

    #
    # Keep useful ranking columns together.
    #
    ranked = ranked[
        OUTPUT_COLUMNS
    ]

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

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
        "Fallback retrieval ranking"
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

    if not counts.empty:
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
