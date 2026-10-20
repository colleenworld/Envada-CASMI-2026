from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

from casmi26.pubchem.inference_asset import (
    load_pubchem_candidates_for_formulas,
)


PROJECT_ROOT = Path(
    os.environ.get(
        "CASMI26_PROJECT_ROOT",
        Path(__file__).resolve().parent.parent,
    )
)

ROOT = (
    PROJECT_ROOT
    / "data/processed/mist/validation_500_v2"
)

BASELINE_PATH = (
    ROOT
    / "contrastive_multiformula_ranked_candidates.parquet"
)

HYPOTHESES_PATH = (
    ROOT
    / "mist_multispectrum_hypotheses.parquet"
)

MANIFEST_PATH = (
    ROOT
    / "manifest.parquet"
)

PUBCHEM_ASSET_ROOT = (
    PROJECT_ROOT
    / "data/processed/pubchem_formula_index"
)

OUTPUT_PATH = (
    ROOT
    / "pubchem_append_rescue_opportunity.parquet"
)

TOP_K = 25


def rr_at_25(
    rank: int | float | None,
) -> float:
    if rank is None or pd.isna(rank):
        return 0.0

    rank = int(rank)

    if rank < 1 or rank > TOP_K:
        return 0.0

    return 1.0 / rank


def main() -> None:
    baseline = pd.read_parquet(
        BASELINE_PATH
    )

    hypotheses = pd.read_parquet(
        HYPOTHESES_PATH
    )

    manifest = pd.read_parquet(
        MANIFEST_PATH
    )

    required_baseline = {
        "query_inchikey14",
        "candidate_inchikey14",
        "rank",
        "is_truth",
    }

    missing = (
        required_baseline
        - set(baseline.columns)
    )

    if missing:
        raise ValueError(
            "Baseline ranking is missing columns: "
            f"{sorted(missing)}"
        )

    required_hypotheses = {
        "query_inchikey14",
        "hypothesis_spec",
        "candidate_formula",
    }

    missing = (
        required_hypotheses
        - set(hypotheses.columns)
    )

    if missing:
        raise ValueError(
            "Validation hypotheses are missing columns: "
            f"{sorted(missing)}"
        )

    truth_column = None

    for candidate in [
        "query_inchikey14",
        "inchikey14",
    ]:
        if candidate in manifest.columns:
            truth_column = candidate
            break

    if truth_column is None:
        raise ValueError(
            "Validation manifest has no recognizable truth structure column"
        )

    universe = (
        manifest[
            [
                truth_column,
            ]
        ]
        .rename(
            columns={
                truth_column:
                    "query_inchikey14",
            }
        )
        .drop_duplicates()
        .copy()
    )

    if len(universe) != 500:
        raise ValueError(
            f"Expected 500 validation queries, found {len(universe)}"
        )

    baseline_counts = (
        baseline.groupby(
            "query_inchikey14"
        )[
            "candidate_inchikey14"
        ]
        .nunique()
        .rename(
            "baseline_candidate_count"
        )
    )

    truth_rows = (
        baseline.loc[
            baseline[
                "is_truth"
            ].astype(bool),
            [
                "query_inchikey14",
                "rank",
            ],
        ]
        .sort_values(
            [
                "query_inchikey14",
                "rank",
            ]
        )
        .drop_duplicates(
            subset=[
                "query_inchikey14",
            ],
            keep="first",
        )
        .rename(
            columns={
                "rank":
                    "baseline_truth_rank",
            }
        )
    )

    diagnostics = (
        universe
        .merge(
            baseline_counts,
            on="query_inchikey14",
            how="left",
        )
        .merge(
            truth_rows,
            on="query_inchikey14",
            how="left",
            validate="one_to_one",
        )
    )

    diagnostics[
        "baseline_candidate_count"
    ] = (
        diagnostics[
            "baseline_candidate_count"
        ]
        .fillna(0)
        .astype(int)
    )

    diagnostics[
        "needs_rescue"
    ] = (
        diagnostics[
            "baseline_candidate_count"
        ]
        < TOP_K
    )

    rescue_queries = set(
        diagnostics.loc[
            diagnostics[
                "needs_rescue"
            ],
            "query_inchikey14",
        ].astype(str)
    )

    print(
        "Validation queries:",
        len(diagnostics),
    )

    print(
        "Queries with fewer than 25 baseline candidates:",
        len(rescue_queries),
    )

    print(
        "Baseline candidate-count distribution "
        "for rescue queries:"
    )

    if rescue_queries:
        print(
            diagnostics.loc[
                diagnostics[
                    "needs_rescue"
                ],
                "baseline_candidate_count",
            ]
            .describe()
            .to_string()
        )

    rescue_hypotheses = hypotheses.loc[
        hypotheses[
            "query_inchikey14"
        ]
        .astype(str)
        .isin(
            rescue_queries
        )
    ].copy()

    formulas = set(
        rescue_hypotheses[
            "candidate_formula"
        ]
        .dropna()
        .astype(str)
    )

    print()
    print(
        "Rescue formula hypotheses:",
        len(
            rescue_hypotheses
        ),
    )

    print(
        "Unique rescue formulas:",
        len(
            formulas
        ),
    )

    if formulas:
        pubchem = (
            load_pubchem_candidates_for_formulas(
                PUBCHEM_ASSET_ROOT,
                formulas,
            )
        )
    else:
        pubchem = pd.DataFrame(
            columns=[
                "molecular_formula",
                "inchikey14",
                "smiles",
            ]
        )

    print(
        "PubChem candidate rows loaded:",
        f"{len(pubchem):,}",
    )

    formula_to_queries = (
        rescue_hypotheses[
            [
                "query_inchikey14",
                "candidate_formula",
            ]
        ]
        .drop_duplicates()
        .rename(
            columns={
                "candidate_formula":
                    "molecular_formula",
            }
        )
    )

    rescue_pairs = (
        formula_to_queries.merge(
            pubchem[
                [
                    "molecular_formula",
                    "inchikey14",
                ]
            ],
            on="molecular_formula",
            how="inner",
            validate="many_to_many",
        )
    )

    truth_in_pubchem = set(
        rescue_pairs.loc[
            rescue_pairs[
                "query_inchikey14"
            ].astype(str)
            == rescue_pairs[
                "inchikey14"
            ].astype(str),
            "query_inchikey14",
        ].astype(str)
    )

    diagnostics[
        "truth_in_pubchem_rescue"
    ] = (
        diagnostics[
            "query_inchikey14"
        ]
        .astype(str)
        .isin(
            truth_in_pubchem
        )
    )

    #
    # Append-only means existing baseline candidates never move.
    # If the truth is absent from the baseline but present in PubChem,
    # the absolute best possible rescue rank is the first vacant slot.
    #
    diagnostics[
        "oracle_append_truth_rank"
    ] = diagnostics[
        "baseline_truth_rank"
    ]

    rescueable = (
        diagnostics[
            "needs_rescue"
        ]
        & diagnostics[
            "baseline_truth_rank"
        ].isna()
        & diagnostics[
            "truth_in_pubchem_rescue"
        ]
    )

    diagnostics.loc[
        rescueable,
        "oracle_append_truth_rank",
    ] = (
        diagnostics.loc[
            rescueable,
            "baseline_candidate_count",
        ]
        + 1
    )

    diagnostics[
        "baseline_rr25"
    ] = diagnostics[
        "baseline_truth_rank"
    ].map(
        rr_at_25
    )

    diagnostics[
        "oracle_append_rr25"
    ] = diagnostics[
        "oracle_append_truth_rank"
    ].map(
        rr_at_25
    )

    diagnostics.to_parquet(
        OUTPUT_PATH,
        index=False,
    )

    baseline_mrr = float(
        diagnostics[
            "baseline_rr25"
        ].mean()
    )

    oracle_mrr = float(
        diagnostics[
            "oracle_append_rr25"
        ].mean()
    )

    baseline_top1 = float(
        (
            diagnostics[
                "baseline_truth_rank"
            ]
            == 1
        ).mean()
    )

    oracle_top1 = float(
        (
            diagnostics[
                "oracle_append_truth_rank"
            ]
            == 1
        ).mean()
    )

    baseline_top25 = float(
        diagnostics[
            "baseline_truth_rank"
        ]
        .between(
            1,
            TOP_K,
            inclusive="both",
        )
        .mean()
    )

    oracle_top25 = float(
        diagnostics[
            "oracle_append_truth_rank"
        ]
        .between(
            1,
            TOP_K,
            inclusive="both",
        )
        .mean()
    )

    print()
    print(
        "Append-only PubChem rescue opportunity"
    )
    print(
        "======================================"
    )

    print(
        "Rescue queries:",
        len(
            rescue_queries
        ),
    )

    print(
        "Rescue queries whose truth is in PubChem:",
        int(
            diagnostics.loc[
                diagnostics[
                    "needs_rescue"
                ],
                "truth_in_pubchem_rescue",
            ].sum()
        ),
    )

    print(
        "Queries newly rescueable "
        "(baseline truth absent, PubChem truth present):",
        int(
            rescueable.sum()
        ),
    )

    print()
    print(
        "Baseline MRR@25:",
        f"{baseline_mrr:.4f}",
    )

    print(
        "Oracle append-only MRR@25:",
        f"{oracle_mrr:.4f}",
    )

    print(
        "Maximum MRR@25 lift:",
        f"{oracle_mrr - baseline_mrr:+.4f}",
    )

    print()
    print(
        "Baseline Top-1:",
        f"{baseline_top1:.2%}",
    )

    print(
        "Oracle append-only Top-1:",
        f"{oracle_top1:.2%}",
    )

    print()
    print(
        "Baseline Top-25:",
        f"{baseline_top25:.2%}",
    )

    print(
        "Oracle append-only Top-25:",
        f"{oracle_top25:.2%}",
    )

    print(
        "Maximum Top-25 lift:",
        f"{oracle_top25 - baseline_top25:+.2%}",
    )

    print()
    print(
        f"Wrote {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()
