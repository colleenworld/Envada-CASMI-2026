from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from msbuddy import Msbuddy, MsbuddyConfig
from msbuddy.base import MetaFeature, Spectrum

from casmi26.chemistry.inference import (
    select_query_spectra,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent

BLIND_TEST_PATH = (
    PROJECT_ROOT
    / "data/processed/blind_pseudotest/test.parquet"
)

BLIND_TRUTH_PATH = (
    PROJECT_ROOT
    / "data/processed/blind_pseudotest/truth.parquet"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data/processed/blind_pseudotest/"
      "msbuddy_formula_gate.parquet"
)

SUMMARY_PATH = (
    PROJECT_ROOT
    / "data/processed/blind_pseudotest/"
      "msbuddy_formula_gate_summary.csv"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark MSBuddy formula annotation on the leakage-safe "
            "blind pseudo-test."
        )
    )

    parser.add_argument(
        "--max-molecules",
        type=int,
        default=250,
    )

    parser.add_argument(
        "--ms1-tol-ppm",
        type=float,
        default=10.0,
    )

    parser.add_argument(
        "--ms2-tol-ppm",
        type=float,
        default=10.0,
    )

    parser.add_argument(
        "--timeout-secs",
        type=float,
        default=120.0,
    )

    args = parser.parse_args()

    if args.max_molecules <= 0:
        parser.error("--max-molecules must be positive")

    return args


def spectrum_from_row(
    row,
) -> Spectrum:
    mz = np.asarray(
        row.ms2_mzs,
        dtype=np.float64,
    )

    intensity = np.asarray(
        row.ms2_normalized_intensities,
        dtype=np.float64,
    )

    finite = (
        np.isfinite(mz)
        & np.isfinite(intensity)
        & (mz > 0)
        & (intensity > 0)
    )

    return Spectrum(
        mz_array=mz[finite],
        int_array=intensity[finite],
    )


def charge_for_mode(
    mode: str,
) -> int:
    if mode == "positive":
        return 1

    if mode == "negative":
        return -1

    raise ValueError(
        f"Unsupported ionization mode: {mode!r}"
    )


def main() -> None:
    args = parse_args()

    for path in [
        BLIND_TEST_PATH,
        BLIND_TRUTH_PATH,
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"Missing required artifact: {path}"
            )

    test = pd.read_parquet(
        BLIND_TEST_PATH
    ).copy()

    test["molecule_id"] = (
        test["molecule_id"]
        .astype(str)
    )

    selected = select_query_spectra(
        test
    )

    molecule_ids = (
        selected["molecule_id"]
        .drop_duplicates()
        .astype(str)
        .head(
            args.max_molecules
        )
        .tolist()
    )

    selected = selected.loc[
        selected["molecule_id"].isin(
            molecule_ids
        )
    ].copy()

    truth = pd.read_parquet(
        BLIND_TRUTH_PATH,
        columns=[
            "molecule_id",
            "molecular_formula",
        ],
    ).copy()

    truth["molecule_id"] = (
        truth["molecule_id"]
        .astype(str)
    )

    truth_lookup = dict(
        zip(
            truth["molecule_id"],
            truth["molecular_formula"].astype(str),
            strict=True,
        )
    )

    config = MsbuddyConfig(
        ms_instr=None,
        ppm=True,
        ms1_tol=args.ms1_tol_ppm,
        ms2_tol=args.ms2_tol_ppm,
        halogen=True,
        parallel=False,
        timeout_secs=args.timeout_secs,
    )

    engine = Msbuddy(
        config
    )

    features = []

    for row_number, row in enumerate(
        selected.itertuples(
            index=False
        )
    ):
        mode = str(
            row.ionization_mode
        )

        identifier = (
            f"{row.molecule_id}"
            f"__row_{row_number:06d}"
        )

        feature = MetaFeature(
            identifier=identifier,
            mz=float(
                row.precursor_mz
            ),
            charge=charge_for_mode(
                mode
            ),
            adduct=str(
                row.adduct
            ),
            ms2=spectrum_from_row(
                row
            ),
        )

        features.append(
            feature
        )

    print(
        "Blind molecules:",
        len(molecule_ids),
    )
    print(
        "MSBuddy feature spectra:",
        len(features),
    )

    engine.add_data(
        features
    )

    print(
        "Running MSBuddy formula annotation..."
    )

    engine.annotate_formula()

    summary = engine.get_summary()

    rows = []

    for item in summary:
        identifier = str(
            item["identifier"]
        )

        molecule_id = identifier.split(
            "__row_",
            maxsplit=1,
        )[0]

        truth_formula = truth_lookup.get(
            molecule_id
        )

        for rank in range(
            1,
            6,
        ):
            formula = item.get(
                f"formula_rank_{rank}"
            )

            if formula is None:
                continue

            formula = str(
                formula
            )

            if (
                not formula
                or formula.lower()
                == "nan"
            ):
                continue

            rows.append(
                {
                    "molecule_id":
                        molecule_id,
                    "identifier":
                        identifier,
                    "rank":
                        rank,
                    "candidate_formula":
                        formula,
                    "truth_formula":
                        truth_formula,
                    "estimated_fdr":
                        item.get(
                            "estimated_fdr"
                        ),
                }
            )

    result = pd.DataFrame(
        rows
    )

    if result.empty:
        raise RuntimeError(
            "MSBuddy produced no ranked formula annotations"
        )

    # Aggregate across spectra belonging to the same molecule by the
    # best rank observed for each formula.
    aggregate = (
        result.groupby(
            [
                "molecule_id",
                "candidate_formula",
                "truth_formula",
            ],
            as_index=False,
            dropna=False,
        )
        .agg(
            best_rank=(
                "rank",
                "min",
            ),
            supporting_spectra=(
                "identifier",
                "nunique",
            ),
            min_estimated_fdr=(
                "estimated_fdr",
                "min",
            ),
        )
    )

    aggregate = (
        aggregate
        .sort_values(
            [
                "molecule_id",
                "best_rank",
                "supporting_spectra",
                "candidate_formula",
            ],
            ascending=[
                True,
                True,
                False,
                True,
            ],
            kind="stable",
        )
        .reset_index(
            drop=True
        )
    )

    aggregate[
        "molecule_rank"
    ] = (
        aggregate.groupby(
            "molecule_id",
            sort=False,
        )
        .cumcount()
        + 1
    )

    truth_ranks = (
        aggregate.loc[
            aggregate[
                "candidate_formula"
            ]
            == aggregate[
                "truth_formula"
            ],
            [
                "molecule_id",
                "molecule_rank",
            ],
        ]
        .drop_duplicates(
            subset=[
                "molecule_id",
            ]
        )
        .set_index(
            "molecule_id"
        )[
            "molecule_rank"
        ]
    )

    eval_frame = pd.DataFrame(
        {
            "molecule_id":
                molecule_ids,
        }
    )

    eval_frame[
        "truth_rank"
    ] = (
        eval_frame[
            "molecule_id"
        ]
        .map(
            truth_ranks
        )
    )

    rows_summary = []

    for cutoff in [
        1,
        3,
        5,
    ]:
        hits = (
            eval_frame[
                "truth_rank"
            ]
            .between(
                1,
                cutoff,
            )
        )

        rows_summary.append(
            {
                "cutoff":
                    cutoff,
                "queries":
                    len(
                        eval_frame
                    ),
                "truth_hits":
                    int(
                        hits.sum()
                    ),
                "truth_recall":
                    float(
                        hits.mean()
                    ),
            }
        )

    output_summary = pd.DataFrame(
        rows_summary
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    aggregate.to_parquet(
        OUTPUT_PATH,
        index=False,
        compression="zstd",
    )

    output_summary.to_csv(
        SUMMARY_PATH,
        index=False,
    )

    print()
    print(
        "MSBuddy blind formula-gate benchmark"
    )
    print("=" * 64)
    print(
        output_summary.to_string(
            index=False,
            formatters={
                "truth_recall":
                    lambda value:
                        f"{value:.2%}",
            },
        )
    )
    print()
    print(
        "Wrote:",
        OUTPUT_PATH,
    )
    print(
        "Wrote:",
        SUMMARY_PATH,
    )


if __name__ == "__main__":
    main()
