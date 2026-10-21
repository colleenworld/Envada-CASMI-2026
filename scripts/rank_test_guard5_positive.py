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

TEST_ROOT = PROJECT_ROOT / "data/processed/test"

HYPOTHESES_PATH = (
    TEST_ROOT
    / "multispectrum_formula_hypotheses.parquet"
)

NEIGHBOR_TOP500_PATH = (
    TEST_ROOT
    / "guard5_neighbor_top500.parquet"
)

PREDICTION_DIR = (
    TEST_ROOT
    / "guard5_mist/predictions"
)

OUTPUT_PATH = (
    TEST_ROOT
    / "positive_contrastive_ranked_candidates.parquet"
)

RRF_K = 60.0
GUARD_DEPTH = 5
MASS_PENALTY = 0.020


def to_str(value) -> str:
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
            "Expected exactly one guard-5 MIST prediction artifact, "
            f"found {predictions}"
        )

    hypotheses = pd.read_parquet(
        HYPOTHESES_PATH
    )

    neighbor = pd.read_parquet(
        NEIGHBOR_TOP500_PATH
    )

    selected_specs = set(
        neighbor[
            "hypothesis_spec"
        ].astype(str)
    )

    hypothesis_meta = (
        hypotheses.loc[
            hypotheses[
                "hypothesis_spec"
            ]
            .astype(str)
            .isin(
                selected_specs
            )
        ]
        .copy()
    )

    hypothesis_meta[
        "molecule_id"
    ] = hypothesis_meta[
        "molecule_id"
    ].astype(str)

    hypothesis_meta = (
        hypothesis_meta
        .sort_values(
            [
                "molecule_id",
                "min_abs_mass_error_ppm",
                "candidate_formula",
            ],
            ascending=[
                True,
                True,
                True,
            ],
            kind="stable",
        )
        .groupby(
            "molecule_id",
            sort=False,
        )
        .head(3)
        .reset_index(drop=True)
    )

    selected_specs = set(
        hypothesis_meta[
            "hypothesis_spec"
        ].astype(str)
    )

    neighbor = neighbor.loc[
        neighbor[
            "hypothesis_spec"
        ]
        .astype(str)
        .isin(
            selected_specs
        )
    ].copy()

    neighbor[
        "hypothesis_spec"
    ] = neighbor[
        "hypothesis_spec"
    ].astype(str)

    neighbor[
        "candidate_inchikey14"
    ] = neighbor[
        "candidate_inchikey14"
    ].astype(str)

    neighbor_lookup = {
        spec: group.copy()
        for spec, group
        in neighbor.groupby(
            "hypothesis_spec",
            sort=False,
        )
    }

    meta_lookup = (
        hypothesis_meta
        .set_index(
            "hypothesis_spec"
        )
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

    missing = (
        required
        - set(
            retrieval.keys()
        )
    )

    if missing:
        raise ValueError(
            "MIST prediction artifact missing keys: "
            f"{sorted(missing)}"
        )

    mist_by_spec = {}

    for spec, ikeys, smiles, dists in zip(
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
        spec = to_str(spec)

        if spec not in selected_specs:
            continue

        ordered = sorted(
            [
                (
                    to_str(ikey),
                    to_str(smiles_value),
                    float(distance),
                )
                for (
                    ikey,
                    smiles_value,
                    distance,
                )
                in zip(
                    ikeys,
                    smiles,
                    dists,
                    strict=True,
                )
            ],
            key=lambda item: (
                item[2],
                item[0],
            ),
        )

        mist_by_spec[
            spec
        ] = ordered

    missing_predictions = (
        selected_specs
        - set(
            mist_by_spec
        )
    )

    if missing_predictions:
        raise RuntimeError(
            "Missing MIST predictions for selected hypotheses. "
            f"Examples: {sorted(missing_predictions)[:20]}"
        )

    ranked_parts = []

    for spec in sorted(
        selected_specs
    ):
        meta = meta_lookup.loc[
            spec
        ]

        neighbor_group = (
            neighbor_lookup.get(
                spec
            )
        )

        if (
            neighbor_group is None
            or neighbor_group.empty
        ):
            continue

        neighbor_rows = (
            neighbor_group[
                [
                    "candidate_inchikey14",
                    "candidate_smiles",
                    "prior_rank",
                    "neighbor_prior_score",
                ]
            ]
            .drop_duplicates(
                subset=[
                    "candidate_inchikey14",
                ]
            )
            .copy()
        )

        mist_rows = []

        for mist_rank, (
            key,
            smiles_value,
            distance,
        ) in enumerate(
            mist_by_spec[
                spec
            ],
            start=1,
        ):
            mist_rows.append(
                {
                    "candidate_inchikey14":
                        key,
                    "mist_smiles":
                        smiles_value,
                    "mist_rank":
                        mist_rank,
                    "contrastive_distance":
                        distance,
                }
            )

        mist_frame = pd.DataFrame(
            mist_rows
        )

        union = neighbor_rows.merge(
            mist_frame,
            on="candidate_inchikey14",
            how="outer",
            validate="one_to_one",
        )

        union[
            "candidate_smiles"
        ] = (
            union[
                "candidate_smiles"
            ]
            .fillna(
                union[
                    "mist_smiles"
                ]
            )
        )

        if union[
            "candidate_smiles"
        ].isna().any():
            raise RuntimeError(
                f"Missing candidate SMILES in {spec}"
            )

        protected = (
            union.loc[
                union[
                    "prior_rank"
                ].notna()
                & (
                    union[
                        "prior_rank"
                    ]
                    <= GUARD_DEPTH
                )
            ]
            .sort_values(
                [
                    "prior_rank",
                    "candidate_inchikey14",
                ],
                ascending=[
                    True,
                    True,
                ],
                kind="stable",
            )
            .copy()
        )

        protected[
            "adaptive_rank"
        ] = np.arange(
            1,
            len(protected) + 1,
            dtype=np.int64,
        )

        protected[
            "ranking_method"
        ] = "neighbor_guard5"

        protected_keys = set(
            protected[
                "candidate_inchikey14"
            ].astype(str)
        )

        remainder = union.loc[
            ~union[
                "candidate_inchikey14"
            ]
            .astype(str)
            .isin(
                protected_keys
            )
        ].copy()

        remainder[
            "_rrf_score"
        ] = 0.0

        has_neighbor = (
            remainder[
                "prior_rank"
            ].notna()
        )

        has_mist = (
            remainder[
                "mist_rank"
            ].notna()
        )

        remainder.loc[
            has_neighbor,
            "_rrf_score",
        ] += (
            1.0
            / (
                RRF_K
                + remainder.loc[
                    has_neighbor,
                    "prior_rank",
                ].astype(float)
            )
        )

        remainder.loc[
            has_mist,
            "_rrf_score",
        ] += (
            1.0
            / (
                RRF_K
                + remainder.loc[
                    has_mist,
                    "mist_rank",
                ].astype(float)
            )
        )

        remainder = (
            remainder
            .sort_values(
                [
                    "_rrf_score",
                    "candidate_inchikey14",
                ],
                ascending=[
                    False,
                    True,
                ],
                kind="stable",
            )
            .reset_index(drop=True)
        )

        remainder[
            "adaptive_rank"
        ] = (
            np.arange(
                len(remainder),
                dtype=np.int64,
            )
            + len(protected)
            + 1
        )

        remainder[
            "ranking_method"
        ] = "guard5_rrf"

        combined = pd.concat(
            [
                protected,
                remainder,
            ],
            ignore_index=True,
        )

        mist_distances = np.sort(
            mist_frame[
                "contrastive_distance"
            ]
            .to_numpy(
                dtype=float
            )
        )

        if len(
            mist_distances
        ) == 0:
            raise RuntimeError(
                f"No MIST distances for {spec}"
            )

        ranks = (
            combined[
                "adaptive_rank"
            ]
            .to_numpy(
                dtype=int
            )
        )

        clipped = np.minimum(
            ranks,
            len(
                mist_distances
            ),
        )

        combined[
            "calibrated_distance"
        ] = (
            mist_distances[
                clipped - 1
            ]
        )

        combined[
            "molecule_id"
        ] = str(
            meta[
                "molecule_id"
            ]
        )

        combined[
            "hypothesis_spec"
        ] = spec

        combined[
            "candidate_formula"
        ] = str(
            meta[
                "candidate_formula"
            ]
        )

        combined[
            "selected_adduct"
        ] = meta[
            "selected_adduct"
        ]

        combined[
            "selected_mass_error_ppm"
        ] = float(
            meta[
                "selected_mass_error_ppm"
            ]
        )

        combined[
            "final_score"
        ] = (
            -combined[
                "calibrated_distance"
            ]
            - MASS_PENALTY
            * abs(
                float(
                    meta[
                        "selected_mass_error_ppm"
                    ]
                )
            )
        )

        ranked_parts.append(
            combined
        )

    if not ranked_parts:
        raise RuntimeError(
            "No guard-5 positive rankings were produced"
        )

    ranked = pd.concat(
        ranked_parts,
        ignore_index=True,
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
            kind="stable",
        )
        .drop_duplicates(
            subset=[
                "molecule_id",
                "candidate_inchikey14",
            ],
            keep="first",
        )
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
            kind="stable",
        )
        .reset_index(drop=True)
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

    keep_columns = [
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
        "adaptive_rank",
        "ranking_method",
        "calibrated_distance",
        "final_score",
        "rank",
    ]

    ranked[
        keep_columns
    ].to_parquet(
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
        "Guard-5 positive ranking"
    )
    print(
        "========================"
    )
    print(
        "Ranked molecules:",
        f"{ranked['molecule_id'].nunique():,}",
    )
    print(
        "Ranked rows:",
        f"{len(ranked):,}",
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
        "Top-ranked methods:"
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
