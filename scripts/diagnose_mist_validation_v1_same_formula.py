from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem


ROOT = Path("data/processed/mist/validation_500")

RANKED_PATH = (
    ROOT / "mist_validation_ranked_candidates.parquet"
)

TRAINING_STRUCTURES_PATH = Path(
    "data/processed/structure_index/training_structures.parquet"
)

COCONUT_PATH = Path(
    "data/external/coconut/coconut_approved.parquet"
)

OUTPUT_PATH = (
    ROOT / "mist_same_formula_error_diagnostics.parquet"
)

QUERY_OUTPUT_PATH = (
    ROOT / "mist_same_formula_query_diagnostics.parquet"
)

RADIUS = 2
N_BITS = 4096


def make_fp(smiles: str):
    if not isinstance(smiles, str) or not smiles:
        return None

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return None

    return AllChem.GetMorganFingerprintAsBitVect(
        mol,
        radius=RADIUS,
        nBits=N_BITS,
    )


def build_structure_catalog():
    """
    Rebuild the same leakage-safe structure universe used for
    validation:

      - training_structures.parquet
      - approved COCONUT
      - remove cross-catalog formula conflicts
      - prefer training representation when present in both
    """
    training = pd.read_parquet(
        TRAINING_STRUCTURES_PATH
    )[
        [
            "inchikey14",
            "normalized_smiles",
            "molecular_formula",
        ]
    ].copy()

    training["in_training"] = True
    training["in_coconut"] = False

    coconut = pd.read_parquet(
        COCONUT_PATH
    )[
        [
            "inchikey14",
            "smiles",
            "molecular_formula",
        ]
    ].copy()

    coconut = coconut.rename(
        columns={
            "smiles": "normalized_smiles",
        }
    )

    coconut["in_training"] = False
    coconut["in_coconut"] = True

    #
    # Exclude cross-catalog formula conflicts, exactly as in the
    # candidate-generation stage.
    #
    overlap = training[
        [
            "inchikey14",
            "molecular_formula",
        ]
    ].merge(
        coconut[
            [
                "inchikey14",
                "molecular_formula",
            ]
        ],
        on="inchikey14",
        how="inner",
        suffixes=(
            "_training",
            "_coconut",
        ),
    )

    conflicts = set(
        overlap.loc[
            overlap[
                "molecular_formula_training"
            ]
            != overlap[
                "molecular_formula_coconut"
            ],
            "inchikey14",
        ]
    )

    print(
        "Cross-catalog formula conflicts:",
        len(conflicts),
    )

    training = training[
        ~training["inchikey14"].isin(
            conflicts
        )
    ].copy()

    coconut = coconut[
        ~coconut["inchikey14"].isin(
            conflicts
        )
    ].copy()

    combined = pd.concat(
        [
            training,
            coconut,
        ],
        ignore_index=True,
    )

    #
    # Preserve membership in both catalogs.
    #
    membership = (
        combined.groupby(
            "inchikey14",
            as_index=False,
        )
        .agg(
            in_training=(
                "in_training",
                "max",
            ),
            in_coconut=(
                "in_coconut",
                "max",
            ),
        )
    )

    #
    # Prefer training SMILES where both catalogs contain the
    # structure.
    #
    combined = combined.sort_values(
        by=[
            "inchikey14",
            "in_training",
        ],
        ascending=[
            True,
            False,
        ],
    )

    catalog = (
        combined.drop_duplicates(
            subset=["inchikey14"],
            keep="first",
        )
        .drop(
            columns=[
                "in_training",
                "in_coconut",
            ]
        )
        .merge(
            membership,
            on="inchikey14",
            how="left",
            validate="one_to_one",
        )
    )

    print(
        "Combined unique structures:",
        len(catalog),
    )

    return catalog


def main():
    ranked = pd.read_parquet(
        RANKED_PATH
    )

    print(
        "Ranked candidate rows:",
        len(ranked),
    )

    print(
        "Queries represented:",
        ranked[
            "query_inchikey14"
        ].nunique(),
    )

    required = {
        "query_inchikey14",
        "hypothesis_spec",
        "candidate_formula",
        "candidate_inchikey14",
        "is_truth",
        "mist_cosine",
        "formula_mass_error_ppm",
        "final_score",
        "rank",
    }

    missing = (
        required
        - set(ranked.columns)
    )

    if missing:
        raise ValueError(
            "Missing required columns: "
            f"{sorted(missing)}"
        )

    #
    # Attach SMILES and provenance to the ranked v1 artifact.
    #
    catalog = build_structure_catalog()

    catalog_lookup = catalog[
        [
            "inchikey14",
            "normalized_smiles",
            "in_training",
            "in_coconut",
        ]
    ].rename(
        columns={
            "inchikey14":
                "candidate_inchikey14",
            "normalized_smiles":
                "candidate_smiles",
        }
    )

    ranked = ranked.merge(
        catalog_lookup,
        on="candidate_inchikey14",
        how="left",
        validate="many_to_one",
    )

    missing_smiles = ranked[
        "candidate_smiles"
    ].isna()

    print(
        "Candidate rows missing SMILES:",
        int(
            missing_smiles.sum()
        ),
    )

    if missing_smiles.any():
        examples = (
            ranked.loc[
                missing_smiles,
                "candidate_inchikey14",
            ]
            .drop_duplicates()
            .head(20)
            .tolist()
        )

        raise ValueError(
            "Some ranked candidates could not be "
            "mapped back to the leakage-safe catalog. "
            f"Examples: {examples}"
        )

    #
    # Cache fingerprints because structures recur across queries.
    #
    fingerprint_cache = {}

    def fingerprint(inchikey14, smiles):
        if inchikey14 not in fingerprint_cache:
            fingerprint_cache[
                inchikey14
            ] = make_fp(
                smiles
            )

        return fingerprint_cache[
            inchikey14
        ]

    diagnostic_rows = []

    #
    # Analyze only queries for which the truth structure is
    # actually rankable.
    #
    for query, group in ranked.groupby(
        "query_inchikey14",
        sort=False,
    ):
        truth_rows = group[
            group["is_truth"]
        ]

        if truth_rows.empty:
            continue

        if len(truth_rows) != 1:
            raise ValueError(
                f"{query}: expected exactly one truth "
                f"row, found {len(truth_rows)}"
            )

        truth = truth_rows.iloc[0]

        truth_formula = (
            truth["candidate_formula"]
        )

        #
        # Oracle-formula subset. Within a formula, the mass
        # penalty is identical for all structures, so structure
        # ordering is determined by MIST cosine.
        #
        same_formula = group[
            group[
                "candidate_formula"
            ]
            == truth_formula
        ].copy()

        same_formula = (
            same_formula.sort_values(
                by=[
                    "mist_cosine",
                    "candidate_inchikey14",
                ],
                ascending=[
                    False,
                    True,
                ],
            )
            .reset_index(
                drop=True
            )
        )

        same_formula[
            "same_formula_rank"
        ] = (
            np.arange(
                len(same_formula)
            )
            + 1
        )

        truth_rows_sf = same_formula[
            same_formula[
                "is_truth"
            ]
        ]

        if len(truth_rows_sf) != 1:
            raise ValueError(
                f"{query}: expected one truth row "
                "inside true-formula subset."
            )

        truth_sf = (
            truth_rows_sf.iloc[0]
        )

        truth_rank = int(
            truth_sf[
                "same_formula_rank"
            ]
        )

        truth_cosine = float(
            truth_sf[
                "mist_cosine"
            ]
        )

        truth_fp = fingerprint(
            truth_sf[
                "candidate_inchikey14"
            ],
            truth_sf[
                "candidate_smiles"
            ],
        )

        if truth_fp is None:
            raise ValueError(
                "Could not fingerprint truth "
                f"structure for {query}"
            )

        #
        # Structures from the CORRECT formula that MIST ranks
        # above the truth.
        #
        outrankers = same_formula[
            same_formula[
                "same_formula_rank"
            ]
            < truth_rank
        ]

        #
        # Keep one query row even when the truth is rank 1.
        #
        if outrankers.empty:
            diagnostic_rows.append(
                {
                    "query_inchikey14":
                        query,

                    "truth_formula":
                        truth_formula,

                    "truth_same_formula_rank":
                        truth_rank,

                    "same_formula_candidate_count":
                        len(
                            same_formula
                        ),

                    "truth_mist_cosine":
                        truth_cosine,

                    "truth_final_score":
                        float(
                            truth_sf[
                                "final_score"
                            ]
                        ),

                    "truth_formula_mass_error_ppm":
                        float(
                            truth_sf[
                                "formula_mass_error_ppm"
                            ]
                        ),

                    "truth_in_training":
                        bool(
                            truth_sf[
                                "in_training"
                            ]
                        ),

                    "truth_in_coconut":
                        bool(
                            truth_sf[
                                "in_coconut"
                            ]
                        ),

                    "competitor_inchikey14":
                        None,

                    "competitor_mist_cosine":
                        np.nan,

                    "competitor_same_formula_rank":
                        np.nan,

                    "competitor_in_training":
                        np.nan,

                    "competitor_in_coconut":
                        np.nan,

                    "mist_cosine_advantage":
                        np.nan,

                    "truth_competitor_tanimoto":
                        np.nan,

                    "is_same_formula_error":
                        False,
                }
            )

            continue

        for competitor in (
            outrankers.itertuples(
                index=False
            )
        ):
            competitor_fp = fingerprint(
                competitor.candidate_inchikey14,
                competitor.candidate_smiles,
            )

            if competitor_fp is None:
                tanimoto = np.nan
            else:
                tanimoto = float(
                    DataStructs.TanimotoSimilarity(
                        truth_fp,
                        competitor_fp,
                    )
                )

            diagnostic_rows.append(
                {
                    "query_inchikey14":
                        query,

                    "truth_formula":
                        truth_formula,

                    "truth_same_formula_rank":
                        truth_rank,

                    "same_formula_candidate_count":
                        len(
                            same_formula
                        ),

                    "truth_mist_cosine":
                        truth_cosine,

                    "truth_final_score":
                        float(
                            truth_sf[
                                "final_score"
                            ]
                        ),

                    "truth_formula_mass_error_ppm":
                        float(
                            truth_sf[
                                "formula_mass_error_ppm"
                            ]
                        ),

                    "truth_in_training":
                        bool(
                            truth_sf[
                                "in_training"
                            ]
                        ),

                    "truth_in_coconut":
                        bool(
                            truth_sf[
                                "in_coconut"
                            ]
                        ),

                    "competitor_inchikey14":
                        competitor.candidate_inchikey14,

                    "competitor_mist_cosine":
                        float(
                            competitor.mist_cosine
                        ),

                    "competitor_same_formula_rank":
                        int(
                            competitor.same_formula_rank
                        ),

                    "competitor_in_training":
                        bool(
                            competitor.in_training
                        ),

                    "competitor_in_coconut":
                        bool(
                            competitor.in_coconut
                        ),

                    #
                    # Positive means the wrong structure beat
                    # the truth by this much.
                    #
                    "mist_cosine_advantage":
                        float(
                            competitor.mist_cosine
                            - truth_cosine
                        ),

                    "truth_competitor_tanimoto":
                        tanimoto,

                    "is_same_formula_error":
                        True,
                }
            )

    diagnostics = pd.DataFrame(
        diagnostic_rows
    )

    diagnostics.to_parquet(
        OUTPUT_PATH,
        index=False,
    )

    #
    # Collapse to one row per query for query-level statistics.
    #
    query_summary = (
        diagnostics.groupby(
            "query_inchikey14",
            as_index=False,
        )
        .agg(
            truth_same_formula_rank=(
                "truth_same_formula_rank",
                "first",
            ),

            same_formula_candidate_count=(
                "same_formula_candidate_count",
                "first",
            ),

            num_outrankers=(
                "is_same_formula_error",
                "sum",
            ),

            max_competitor_advantage=(
                "mist_cosine_advantage",
                "max",
            ),

            max_tanimoto_to_outranker=(
                "truth_competitor_tanimoto",
                "max",
            ),

            median_tanimoto_to_outrankers=(
                "truth_competitor_tanimoto",
                "median",
            ),

            truth_in_training=(
                "truth_in_training",
                "first",
            ),

            truth_in_coconut=(
                "truth_in_coconut",
                "first",
            ),
        )
    )

    query_summary.to_parquet(
        QUERY_OUTPUT_PATH,
        index=False,
    )

    errors = query_summary[
        query_summary[
            "num_outrankers"
        ]
        > 0
    ]

    correct = query_summary[
        query_summary[
            "num_outrankers"
        ]
        == 0
    ]

    print(
        "\nSame-formula ranking"
    )
    print(
        "--------------------"
    )

    print(
        "Rankable queries:",
        len(
            query_summary
        ),
    )

    print(
        "Truth rank 1 within formula:",
        len(
            correct
        ),
    )

    print(
        "Same-formula errors:",
        len(
            errors
        ),
    )

    print(
        "Same-formula Top-1:",
        f"{len(correct) / len(query_summary) * 100:.2f}%",
    )

    print(
        "\nTruth same-formula rank"
    )
    print(
        "-----------------------"
    )

    print(
        query_summary[
            "truth_same_formula_rank"
        ]
        .describe(
            percentiles=[
                0.5,
                0.75,
                0.9,
                0.95,
                0.99,
            ]
        )
        .to_string()
    )

    print(
        "\nTrue-formula candidate counts"
    )
    print(
        "-----------------------------"
    )

    print(
        query_summary[
            "same_formula_candidate_count"
        ]
        .describe(
            percentiles=[
                0.5,
                0.75,
                0.9,
                0.95,
                0.99,
            ]
        )
        .to_string()
    )

    wrong_rows = diagnostics[
        diagnostics[
            "is_same_formula_error"
        ]
    ]

    if not wrong_rows.empty:
        similarities = (
            wrong_rows[
                "truth_competitor_tanimoto"
            ]
            .dropna()
        )

        print(
            "\nTanimoto similarity of structures "
            "that outrank truth"
        )
        print(
            "-----------------------------------"
        )

        print(
            similarities.describe(
                percentiles=[
                    0.25,
                    0.5,
                    0.75,
                    0.9,
                    0.95,
                    0.99,
                ]
            ).to_string()
        )

        print(
            "\nFraction of outrankers above "
            "Tanimoto thresholds"
        )
        print(
            "--------------------------------"
        )

        for threshold in [
            0.50,
            0.60,
            0.70,
            0.80,
            0.90,
            0.95,
        ]:
            fraction = (
                similarities
                >= threshold
            ).mean()

            print(
                f">= {threshold:.2f}: "
                f"{fraction * 100:.2f}%"
            )

        print(
            "\nMIST cosine advantage of outrankers"
        )
        print(
            "----------------------------------"
        )

        print(
            wrong_rows[
                "mist_cosine_advantage"
            ]
            .describe(
                percentiles=[
                    0.25,
                    0.5,
                    0.75,
                    0.9,
                    0.95,
                    0.99,
                ]
            )
            .to_string()
        )

        #
        # Does provenance correlate with errors?
        #
        error_truth = diagnostics[
            diagnostics[
                "is_same_formula_error"
            ]
        ].drop_duplicates(
            "query_inchikey14"
        )

        correct_truth = diagnostics[
            ~diagnostics[
                "is_same_formula_error"
            ]
        ]

        correct_query_ids = set(
            correct[
                "query_inchikey14"
            ]
        )

        correct_truth = (
            correct_truth[
                correct_truth[
                    "query_inchikey14"
                ].isin(
                    correct_query_ids
                )
            ]
            .drop_duplicates(
                "query_inchikey14"
            )
        )

        print(
            "\nTruth provenance"
        )
        print(
            "----------------"
        )

        print(
            "Error queries truth in training:",
            f"{error_truth['truth_in_training'].mean() * 100:.2f}%",
        )

        print(
            "Correct queries truth in training:",
            f"{correct_truth['truth_in_training'].mean() * 100:.2f}%",
        )

        print(
            "Error queries truth in COCONUT:",
            f"{error_truth['truth_in_coconut'].mean() * 100:.2f}%",
        )

        print(
            "Correct queries truth in COCONUT:",
            f"{correct_truth['truth_in_coconut'].mean() * 100:.2f}%",
        )

        print(
            "\nWorst same-formula queries"
        )
        print(
            "--------------------------"
        )

        worst = (
            errors.sort_values(
                by=[
                    "truth_same_formula_rank",
                    "same_formula_candidate_count",
                ],
                ascending=[
                    False,
                    False,
                ],
            )
            .head(20)
        )

        print(
            worst.to_string(
                index=False
            )
        )

    print(
        f"\nWrote {OUTPUT_PATH}"
    )

    print(
        f"Wrote {QUERY_OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()