from __future__ import annotations

import hashlib
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent

ROOT = (
    PROJECT_ROOT
    / "data/processed/blind_pseudotest"
)

TRUTH_PATH = ROOT / "truth.parquet"

PRODUCTION_PATH = (
    PROJECT_ROOT
    / "data/processed/test/"
      "multispectrum_formula_hypotheses.parquet"
)

MSBUDDY_PATH = (
    ROOT
    / "msbuddy_formula_gate.parquet"
)

OUTPUT_PATH = (
    ROOT
    / "production_msbuddy_fusion_holdout.csv"
)

RANKING_OUTPUT_PATH = (
    ROOT
    / "production_msbuddy_fused_formulas.parquet"
)

TOP_K_SOURCE = 5
TOP_K_FINAL = 3

PRODUCTION_WEIGHTS = [
    0.5,
    1.0,
    1.5,
    2.0,
]

MSBUDDY_WEIGHTS = [
    0.5,
    1.0,
    1.5,
    2.0,
]

AGREEMENT_BONUSES = [
    0.0,
    0.5,
    1.0,
    2.0,
]

PRODUCTION_SUPPORT_WEIGHTS = [
    0.0,
    0.25,
    0.5,
]

MSBUDDY_SUPPORT_WEIGHTS = [
    0.0,
    0.25,
    0.5,
]


def split_bucket(
    molecule_id: str,
) -> int:
    digest = hashlib.sha256(
        molecule_id.encode(
            "utf-8"
        )
    ).digest()

    return int.from_bytes(
        digest[:8],
        "big",
    ) % 10


def prepare_production(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    required = {
        "molecule_id",
        "candidate_formula",
        "min_abs_mass_error_ppm",
        "spectrum_support_count",
        "spectrum_support_fraction",
    }

    missing = required - set(
        frame.columns
    )

    if missing:
        raise ValueError(
            "Production hypotheses missing columns: "
            f"{sorted(missing)}"
        )

    work = frame[
        [
            "molecule_id",
            "candidate_formula",
            "min_abs_mass_error_ppm",
            "spectrum_support_count",
            "spectrum_support_fraction",
        ]
    ].copy()

    work["molecule_id"] = (
        work["molecule_id"]
        .astype(str)
    )

    work["candidate_formula"] = (
        work["candidate_formula"]
        .astype(str)
    )

    work = (
        work.sort_values(
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
        .drop_duplicates(
            subset=[
                "molecule_id",
                "candidate_formula",
            ],
            keep="first",
        )
        .reset_index(
            drop=True
        )
    )

    work["production_rank"] = (
        work.groupby(
            "molecule_id",
            sort=False,
        )
        .cumcount()
        + 1
    )

    return work.loc[
        work[
            "production_rank"
        ]
        <= TOP_K_SOURCE
    ].copy()


def prepare_msbuddy(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    required = {
        "molecule_id",
        "candidate_formula",
        "molecule_rank",
        "supporting_spectra",
    }

    missing = required - set(
        frame.columns
    )

    if missing:
        raise ValueError(
            "MSBuddy benchmark missing columns: "
            f"{sorted(missing)}"
        )

    work = frame[
        [
            "molecule_id",
            "candidate_formula",
            "molecule_rank",
            "supporting_spectra",
        ]
    ].copy()

    work["molecule_id"] = (
        work["molecule_id"]
        .astype(str)
    )

    work["candidate_formula"] = (
        work["candidate_formula"]
        .astype(str)
    )

    work = work.rename(
        columns={
            "molecule_rank":
                "msbuddy_rank",
            "supporting_spectra":
                "msbuddy_supporting_spectra",
        }
    )

    return work.loc[
        work[
            "msbuddy_rank"
        ]
        <= TOP_K_SOURCE
    ].copy()


def candidate_table(
    production: pd.DataFrame,
    msbuddy: pd.DataFrame,
) -> pd.DataFrame:
    keys = [
        "molecule_id",
        "candidate_formula",
    ]

    merged = production.merge(
        msbuddy,
        on=keys,
        how="outer",
        validate="one_to_one",
    )

    merged[
        "in_production"
    ] = (
        merged[
            "production_rank"
        ]
        .notna()
    )

    merged[
        "in_msbuddy"
    ] = (
        merged[
            "msbuddy_rank"
        ]
        .notna()
    )

    merged[
        "agreement"
    ] = (
        merged[
            "in_production"
        ]
        & merged[
            "in_msbuddy"
        ]
    )

    merged[
        "production_support_fraction"
    ] = pd.to_numeric(
        merged[
            "spectrum_support_fraction"
        ],
        errors="coerce",
    ).fillna(
        0.0
    )

    # Normalize MSBuddy support within molecule because the number of
    # selected spectra varies between queries.
    max_msbuddy_support = (
        merged.groupby(
            "molecule_id"
        )[
            "msbuddy_supporting_spectra"
        ]
        .transform(
            "max"
        )
    )

    merged[
        "msbuddy_support_fraction"
    ] = (
        pd.to_numeric(
            merged[
                "msbuddy_supporting_spectra"
            ],
            errors="coerce",
        )
        .fillna(
            0.0
        )
        / max_msbuddy_support
        .fillna(
            1.0
        )
        .clip(
            lower=1.0
        )
    )

    return merged


def score_candidates(
    candidates: pd.DataFrame,
    *,
    production_weight: float,
    msbuddy_weight: float,
    agreement_bonus: float,
    production_support_weight: float,
    msbuddy_support_weight: float,
) -> pd.DataFrame:
    scored = candidates.copy()

    production_rr = np.where(
        scored[
            "production_rank"
        ].notna(),
        1.0
        / scored[
            "production_rank"
        ].fillna(
            np.inf
        ).astype(float),
        0.0,
    )

    msbuddy_rr = np.where(
        scored[
            "msbuddy_rank"
        ].notna(),
        1.0
        / scored[
            "msbuddy_rank"
        ].fillna(
            np.inf
        ).astype(float),
        0.0,
    )

    scored[
        "fusion_score"
    ] = (
        production_weight
        * production_rr
        + msbuddy_weight
        * msbuddy_rr
        + agreement_bonus
        * scored[
            "agreement"
        ].astype(float)
        + production_support_weight
        * scored[
            "production_support_fraction"
        ].astype(float)
        + msbuddy_support_weight
        * scored[
            "msbuddy_support_fraction"
        ].astype(float)
    )

    scored = (
        scored.sort_values(
            [
                "molecule_id",
                "fusion_score",
                "agreement",
                "production_rank",
                "msbuddy_rank",
                "candidate_formula",
            ],
            ascending=[
                True,
                False,
                False,
                True,
                True,
                True,
            ],
            kind="stable",
            na_position="last",
        )
        .reset_index(
            drop=True
        )
    )

    scored[
        "fusion_rank"
    ] = (
        scored.groupby(
            "molecule_id",
            sort=False,
        )
        .cumcount()
        + 1
    )

    return scored


def recall_at_3(
    ranked: pd.DataFrame,
    truth: pd.DataFrame,
    molecule_ids: set[str],
) -> tuple[int, int, float]:
    selected = truth.loc[
        truth[
            "molecule_id"
        ].isin(
            molecule_ids
        )
    ].copy()

    top3 = (
        ranked.loc[
            ranked[
                "fusion_rank"
            ]
            <= TOP_K_FINAL,
            [
                "molecule_id",
                "candidate_formula",
            ],
        ]
        .groupby(
            "molecule_id"
        )[
            "candidate_formula"
        ]
        .agg(
            set
        )
        .to_dict()
    )

    hits = sum(
        1
        for row
        in selected.itertuples(
            index=False
        )
        if str(
            row.molecular_formula
        )
        in top3.get(
            str(
                row.molecule_id
            ),
            set(),
        )
    )

    total = len(
        selected
    )

    return (
        hits,
        total,
        hits / total
        if total
        else 0.0,
    )


def main() -> None:
    for path in [
        TRUTH_PATH,
        PRODUCTION_PATH,
        MSBUDDY_PATH,
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"Missing required artifact: {path}"
            )

    truth = pd.read_parquet(
        TRUTH_PATH,
        columns=[
            "molecule_id",
            "molecular_formula",
        ],
    ).copy()

    truth[
        "molecule_id"
    ] = (
        truth[
            "molecule_id"
        ]
        .astype(str)
    )

    truth[
        "molecular_formula"
    ] = (
        truth[
            "molecular_formula"
        ]
        .astype(str)
    )

    production = prepare_production(
        pd.read_parquet(
            PRODUCTION_PATH
        )
    )

    msbuddy = prepare_msbuddy(
        pd.read_parquet(
            MSBUDDY_PATH
        )
    )

    candidates = candidate_table(
        production,
        msbuddy,
    )

    all_ids = set(
        truth[
            "molecule_id"
        ]
    )

    train_ids = {
        molecule_id
        for molecule_id
        in all_ids
        if split_bucket(
            molecule_id
        )
        < 7
    }

    holdout_ids = (
        all_ids
        - train_ids
    )

    print(
        "Fusion tuning queries:",
        len(
            train_ids
        ),
    )

    print(
        "Fusion holdout queries:",
        len(
            holdout_ids
        ),
    )

    results = []

    for (
        production_weight,
        msbuddy_weight,
        agreement_bonus,
        production_support_weight,
        msbuddy_support_weight,
    ) in product(
        PRODUCTION_WEIGHTS,
        MSBUDDY_WEIGHTS,
        AGREEMENT_BONUSES,
        PRODUCTION_SUPPORT_WEIGHTS,
        MSBUDDY_SUPPORT_WEIGHTS,
    ):
        ranked = score_candidates(
            candidates,
            production_weight=(
                production_weight
            ),
            msbuddy_weight=(
                msbuddy_weight
            ),
            agreement_bonus=(
                agreement_bonus
            ),
            production_support_weight=(
                production_support_weight
            ),
            msbuddy_support_weight=(
                msbuddy_support_weight
            ),
        )

        (
            train_hits,
            train_total,
            train_recall,
        ) = recall_at_3(
            ranked,
            truth,
            train_ids,
        )

        results.append(
            {
                "production_weight":
                    production_weight,
                "msbuddy_weight":
                    msbuddy_weight,
                "agreement_bonus":
                    agreement_bonus,
                "production_support_weight":
                    production_support_weight,
                "msbuddy_support_weight":
                    msbuddy_support_weight,
                "train_hits":
                    train_hits,
                "train_total":
                    train_total,
                "train_recall_at3":
                    train_recall,
            }
        )

    search = (
        pd.DataFrame(
            results
        )
        .sort_values(
            [
                "train_recall_at3",
                "agreement_bonus",
                "production_weight",
                "msbuddy_weight",
                "production_support_weight",
                "msbuddy_support_weight",
            ],
            ascending=[
                False,
                True,
                True,
                True,
                True,
                True,
            ],
            kind="stable",
        )
        .reset_index(
            drop=True
        )
    )

    best = search.iloc[
        0
    ]

    fused = score_candidates(
        candidates,
        production_weight=float(
            best[
                "production_weight"
            ]
        ),
        msbuddy_weight=float(
            best[
                "msbuddy_weight"
            ]
        ),
        agreement_bonus=float(
            best[
                "agreement_bonus"
            ]
        ),
        production_support_weight=float(
            best[
                "production_support_weight"
            ]
        ),
        msbuddy_support_weight=float(
            best[
                "msbuddy_support_weight"
            ]
        ),
    )

    (
        holdout_hits,
        holdout_total,
        holdout_recall,
    ) = recall_at_3(
        fused,
        truth,
        holdout_ids,
    )

    (
        all_hits,
        all_total,
        all_recall,
    ) = recall_at_3(
        fused,
        truth,
        all_ids,
    )

    best_summary = pd.DataFrame(
        [
            {
                **best.to_dict(),
                "holdout_hits":
                    holdout_hits,
                "holdout_total":
                    holdout_total,
                "holdout_recall_at3":
                    holdout_recall,
                "all_hits":
                    all_hits,
                "all_total":
                    all_total,
                "all_recall_at3":
                    all_recall,
            }
        ]
    )

    best_summary.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    fused.to_parquet(
        RANKING_OUTPUT_PATH,
        index=False,
        compression="zstd",
    )

    print()
    print(
        "Production + MSBuddy Top-3 fusion"
    )
    print("=" * 100)

    print(
        "Chosen weights:"
    )

    print(
        best_summary[
            [
                "production_weight",
                "msbuddy_weight",
                "agreement_bonus",
                "production_support_weight",
                "msbuddy_support_weight",
            ]
        ].to_string(
            index=False
        )
    )

    print()

    print(
        "Tuning Top-3 recall:",
        (
            f"{int(best['train_hits'])}/"
            f"{int(best['train_total'])} "
            f"({best['train_recall_at3']:.2%})"
        ),
    )

    print(
        "HOLDOUT Top-3 recall:",
        (
            f"{holdout_hits}/"
            f"{holdout_total} "
            f"({holdout_recall:.2%})"
        ),
    )

    print(
        "All-query Top-3 recall:",
        (
            f"{all_hits}/"
            f"{all_total} "
            f"({all_recall:.2%})"
        ),
    )

    print()
    print(
        "Reference single-gate all-query Top-3:"
    )
    print(
        "  production: 85.00%"
    )
    print(
        "  MSBuddy:    81.90%"
    )
    print(
        "  union cap:  92.90%"
    )

    print()
    print(
        "Wrote:",
        OUTPUT_PATH,
    )
    print(
        "Wrote:",
        RANKING_OUTPUT_PATH,
    )


if __name__ == "__main__":
    main()
