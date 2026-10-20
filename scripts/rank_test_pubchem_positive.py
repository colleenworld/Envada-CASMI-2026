from __future__ import annotations

import os
import pickle
from pathlib import Path

import numpy as np
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

CANDIDATES_PATH = (
    TEST_ROOT
    / "pubchem_pruned_structure_candidates.parquet"
)

PRIOR_PATH = (
    TEST_ROOT
    / "pubchem_neighbor_prior.parquet"
)

PREDICTION_DIR = (
    TEST_ROOT
    / "pubchem_mist/predictions"
)

OUTPUT_PATH = (
    TEST_ROOT
    / "positive_contrastive_ranked_candidates.parquet"
)

MASS_PENALTY = 0.020
RRF_K = 60.0

OUTPUT_COLUMNS = [
    "molecule_id",
    "hypothesis_spec",
    "candidate_formula",
    "candidate_inchikey14",
    "candidate_smiles",
    "selected_adduct",
    "selected_mass_error_ppm",
    "contrastive_distance",
    "neighbor_prior_score",
    "mist_rank",
    "prior_rank",
    "ranking_method",
    "final_score",
    "rank",
]


def to_str(value):
    if isinstance(value, bytes):
        return value.decode()

    return str(value)


def main() -> None:
    predictions = sorted(
        PREDICTION_DIR.glob(
            "retrieval_contrastive_*.p"
        )
    )

    if len(predictions) != 1:
        raise RuntimeError(
            "Expected exactly one PubChem MIST prediction artifact, "
            f"found {len(predictions)}: {predictions}"
        )

    hypotheses = pd.read_parquet(
        HYPOTHESES_PATH
    )

    candidates = pd.read_parquet(
        CANDIDATES_PATH
    )

    prior = pd.read_parquet(
        PRIOR_PATH
    )

    with predictions[0].open(
        "rb"
    ) as handle:
        retrieval = pickle.load(
            handle
        )

    required = {
        "names",
        "ikeys",
        "smiles",
        "dists",
    }

    missing = required - set(
        retrieval.keys()
    )

    if missing:
        raise ValueError(
            "Prediction artifact is missing keys: "
            f"{sorted(missing)}"
        )

    supported = hypotheses.loc[
        hypotheses[
            "hypothesis_spec"
        ]
        .astype(str)
        .isin(
            set(
                candidates[
                    "hypothesis_spec"
                ].astype(str)
            )
        )
    ].copy()

    hypothesis_lookup = (
        supported
        .set_index(
            "hypothesis_spec"
        )
    )

    shortlist = (
        candidates[
            [
                "hypothesis_spec",
                "candidate_inchikey14",
                "candidate_smiles",
                "candidate_count_before_pruning",
            ]
        ]
        .copy()
    )

    shortlist[
        "hypothesis_spec"
    ] = shortlist[
        "hypothesis_spec"
    ].astype(str)

    shortlist[
        "candidate_inchikey14"
    ] = shortlist[
        "candidate_inchikey14"
    ].astype(str)

    shortlist_keys = {
        spec: set(
            group[
                "candidate_inchikey14"
            ]
        )
        for spec, group in shortlist.groupby(
            "hypothesis_spec",
            sort=False,
        )
    }

    shortlist_smiles = {
        (
            str(row.hypothesis_spec),
            str(row.candidate_inchikey14),
        ):
            str(row.candidate_smiles)
        for row in shortlist.itertuples(
            index=False
        )
    }

    candidate_count = (
        shortlist[
            [
                "hypothesis_spec",
                "candidate_count_before_pruning",
            ]
        ]
        .drop_duplicates(
            subset=[
                "hypothesis_spec",
            ]
        )
        .set_index(
            "hypothesis_spec"
        )[
            "candidate_count_before_pruning"
        ]
        .astype(int)
        .to_dict()
    )

    prior_lookup = (
        prior[
            [
                "hypothesis_spec",
                "candidate_inchikey14",
                "neighbor_prior_score",
                "prior_rank",
            ]
        ]
        .copy()
    )

    prior_lookup[
        "hypothesis_spec"
    ] = prior_lookup[
        "hypothesis_spec"
    ].astype(str)

    prior_lookup[
        "candidate_inchikey14"
    ] = prior_lookup[
        "candidate_inchikey14"
    ].astype(str)

    prior_lookup = prior_lookup.set_index(
        [
            "hypothesis_spec",
            "candidate_inchikey14",
        ]
    )

    rows = []

    for (
        spec,
        ikeys,
        smiles,
        dists,
    ) in zip(
        retrieval[
            "names"
        ],
        retrieval[
            "ikeys"
        ],
        retrieval[
            "smiles"
        ],
        retrieval[
            "dists"
        ],
        strict=True,
    ):
        spec = to_str(
            spec
        )

        if spec not in hypothesis_lookup.index:
            raise ValueError(
                f"Unexpected PubChem MIST hypothesis: {spec}"
            )

        allowed = shortlist_keys.get(
            spec,
            set(),
        )

        if not allowed:
            continue

        meta = hypothesis_lookup.loc[
            spec
        ]

        ordered = sorted(
            zip(
                ikeys,
                smiles,
                dists,
                strict=True,
            ),
            key=lambda item: (
                float(
                    item[2]
                ),
                to_str(
                    item[0]
                ),
            ),
        )

        filtered = []

        for (
            ikey,
            smiles_value,
            distance,
        ) in ordered:
            key = to_str(
                ikey
            )

            if key not in allowed:
                continue

            filtered.append(
                (
                    key,
                    to_str(
                        smiles_value
                    ),
                    float(
                        distance
                    ),
                )
            )

        for mist_rank, (
            key,
            smiles_value,
            distance,
        ) in enumerate(
            filtered,
            start=1,
        ):
            prior_meta = prior_lookup.loc[
                (
                    spec,
                    key,
                )
            ]

            prior_rank = int(
                prior_meta[
                    "prior_rank"
                ]
            )

            prior_score = float(
                prior_meta[
                    "neighbor_prior_score"
                ]
            )

            total_candidates = int(
                candidate_count[
                    spec
                ]
            )

            if total_candidates <= 100:
                ranking_method = "mist"

                within_hypothesis_score = (
                    -float(
                        mist_rank
                    )
                )

            else:
                ranking_method = "rrf"

                within_hypothesis_score = (
                    1.0
                    / (
                        RRF_K
                        + float(
                            mist_rank
                        )
                    )
                    + 1.0
                    / (
                        RRF_K
                        + float(
                            prior_rank
                        )
                    )
                )

            mass_error_ppm = float(
                meta[
                    "selected_mass_error_ppm"
                ]
            )

            final_score = (
                within_hypothesis_score
                - MASS_PENALTY
                * abs(
                    mass_error_ppm
                )
            )

            rows.append(
                {
                    "molecule_id":
                        str(
                            meta[
                                "molecule_id"
                            ]
                        ),
                    "hypothesis_spec":
                        spec,
                    "candidate_formula":
                        str(
                            meta[
                                "candidate_formula"
                            ]
                        ),
                    "candidate_inchikey14":
                        key,
                    "candidate_smiles":
                        shortlist_smiles[
                            (
                                spec,
                                key,
                            )
                        ],
                    "selected_adduct":
                        meta[
                            "selected_adduct"
                        ],
                    "selected_mass_error_ppm":
                        mass_error_ppm,
                    "contrastive_distance":
                        distance,
                    "neighbor_prior_score":
                        prior_score,
                    "mist_rank":
                        mist_rank,
                    "prior_rank":
                        prior_rank,
                    "ranking_method":
                        ranking_method,
                    "final_score":
                        final_score,
                }
            )

    if not rows:
        raise RuntimeError(
            "PubChem MIST produced no usable shortlisted candidates"
        )

    ranked = pd.DataFrame(
        rows
    )

    ranked = (
        ranked
        .sort_values(
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
        .drop_duplicates(
            subset=[
                "molecule_id",
                "candidate_inchikey14",
            ],
            keep="first",
        )
    )

    ranked = (
        ranked
        .sort_values(
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
        )
        .reset_index(
            drop=True
        )
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

    ranked = ranked[
        OUTPUT_COLUMNS
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
        "PubChem adaptive positive ranking"
    )
    print(
        "================================="
    )
    print(
        "Ranked rows:",
        f"{len(ranked):,}",
    )
    print(
        "Ranked molecules:",
        f"{ranked['molecule_id'].nunique():,}",
    )
    print(
        "Candidates/molecule mean:",
        f"{counts.mean():.1f}",
    )
    print(
        "Candidates/molecule median:",
        f"{counts.median():.1f}",
    )
    print(
        "Candidates/molecule max:",
        f"{counts.max():,}",
    )
    print()
    print(
        "Winning ranking methods:"
    )
    print(
        ranked.loc[
            ranked[
                "rank"
            ]
            == 1,
            "ranking_method",
        ]
        .value_counts()
        .to_string()
    )
    print()
    print(
        f"Wrote {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()
