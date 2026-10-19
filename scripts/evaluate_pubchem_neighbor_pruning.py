from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

from casmi26.evaluation.strict_validation import (
    resolve_structure_metadata,
)
from casmi26.retrieval.index import (
    load_retrieval_index,
)
from casmi26.pubchem.neighbor_pruning import (
    score_candidates,
)
from casmi26.spectra.binning import (
    bin_spectrum,
)


PROJECT_ROOT = (
    Path(__file__).resolve().parent.parent
)

TRAIN_PATH = (
    PROJECT_ROOT
    / "data/raw/train.parquet"
)

STRICT_INDEX_DIR = (
    PROJECT_ROOT
    / "data/processed/"
    "retrieval_index_strict_validation"
)

BENCHMARK_DIR = (
    PROJECT_ROOT
    / "data/processed/diagnostics/"
    "pubchem_mist_benchmark"
)

SAMPLE_PATH = (
    BENCHMARK_DIR
    / "sample.parquet"
)

CANDIDATES_PATH = (
    BENCHMARK_DIR
    / "candidates.parquet"
)

OUTPUT_PATH = (
    BENCHMARK_DIR
    / "neighbor_pruning_ranked.parquet"
)

SUMMARY_PATH = (
    BENCHMARK_DIR
    / "neighbor_pruning_summary.parquet"
)

BIN_WIDTH = 1.0
MAX_MZ = 1500.0

SHORTLIST_SIZES = [
    25,
    100,
    250,
    500,
]


RDLogger.DisableLog(
    "rdApp.*"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate spectral-neighbor/Morgan pruning of "
            "PubChem formula candidates."
        )
    )

    parser.add_argument(
        "--spectral-neighbors",
        type=int,
        default=25,
        help=(
            "Number of unique training structures used as "
            "spectral-neighbor anchors."
        ),
    )

    parser.add_argument(
        "--min-candidates",
        type=int,
        default=0,
        help=(
            "Only evaluate benchmark molecules with at least "
            "this many PubChem candidates."
        ),
    )

    return parser.parse_args()


def morgan_fp(
    smiles: str,
):
    mol = Chem.MolFromSmiles(
        str(smiles)
    )

    if mol is None:
        return None

    return (
        AllChem.GetMorganFingerprintAsBitVect(
            mol,
            radius=2,
            nBits=4096,
        )
    )


def build_training_smiles(
    allowed_keys: set[str],
) -> dict[str, str]:
    frame = pd.read_parquet(
        TRAIN_PATH,
        columns=[
            "inchikey14",
            "normalized_smiles",
            "molecular_formula",
        ],
    )

    frame[
        "inchikey14"
    ] = (
        frame[
            "inchikey14"
        ].astype(str)
    )

    frame = frame.loc[
        frame[
            "inchikey14"
        ].isin(
            allowed_keys
        )
    ].copy()

    resolved = (
        resolve_structure_metadata(
            frame,
            smiles_column="normalized_smiles",
        )
    )

    return dict(
        zip(
            resolved[
                "inchikey14"
            ].astype(str),
            resolved[
                "normalized_smiles"
            ].astype(str),
            strict=True,
        )
    )


def top_structure_neighbors(
    query_vector: np.ndarray,
    ref_vectors,
    ref_keys: np.ndarray,
    count: int,
) -> list[
    tuple[
        str,
        float,
    ]
]:
    spectrum_scores = (
        ref_vectors.dot(
            query_vector
        )
    )

    spectrum_scores = np.asarray(
        spectrum_scores
    ).reshape(-1)

    if spectrum_scores.size == 0:
        return []

    #
    # Pull a generous spectrum-level pool first, then collapse to
    # structures. This avoids sorting all 2.2M references while still
    # giving duplicate-library spectra a chance to contribute.
    #
    pool_size = min(
        spectrum_scores.size,
        max(
            count * 100,
            5000,
        ),
    )

    partition = np.argpartition(
        spectrum_scores,
        -pool_size,
    )[
        -pool_size:
    ]

    partition = partition[
        np.argsort(
            spectrum_scores[
                partition
            ]
        )[::-1]
    ]

    best_by_structure = {}

    for index in partition:
        key = str(
            ref_keys[
                index
            ]
        )

        score = float(
            spectrum_scores[
                index
            ]
        )

        previous = (
            best_by_structure.get(
                key
            )
        )

        if (
            previous is None
            or score > previous
        ):
            best_by_structure[
                key
            ] = score

    ordered = sorted(
        best_by_structure.items(),
        key=lambda item: (
            -item[1],
            item[0],
        ),
    )

    return ordered[
        :count
    ]


def main() -> None:
    args = parse_args()

    if args.spectral_neighbors <= 0:
        raise ValueError(
            "--spectral-neighbors must be positive"
        )

    sample = pd.read_parquet(
        SAMPLE_PATH
    )

    candidates = pd.read_parquet(
        CANDIDATES_PATH
    )

    required_sample_columns = {
        "spec",
        "inchikey14",
        "truth_formula",
        "candidate_count",
        "burden_bucket",
        "mist_ionization",
        "ms2_mzs",
        "ms2_normalized_intensities",
    }

    missing_sample_columns = (
        required_sample_columns
        - set(sample.columns)
    )

    if missing_sample_columns:
        raise ValueError(
            "Benchmark sample is missing required columns: "
            f"{sorted(missing_sample_columns)}"
        )

    supported_positive_ionizations = {
        "[M+H]+",
        "[M+Na]+",
        "[M+K]+",
        "[M+H3N+H]+",
    }

    unexpected_ionizations = sorted(
        set(
            sample[
                "mist_ionization"
            ]
            .dropna()
            .astype(str)
        )
        - supported_positive_ionizations
    )

    if unexpected_ionizations:
        raise ValueError(
            "Neighbor-pruning benchmark expects the existing "
            "positive-only MIST sample, but found unsupported "
            f"ionizations: {unexpected_ionizations}"
        )

    required_candidate_columns = {
        "spec",
        "inchikey14",
        "smiles",
    }

    missing_candidate_columns = (
        required_candidate_columns
        - set(candidates.columns)
    )

    if missing_candidate_columns:
        raise ValueError(
            "Benchmark candidates are missing required columns: "
            f"{sorted(missing_candidate_columns)}"
        )

    sample = sample.loc[
        sample[
            "candidate_count"
        ]
        >= args.min_candidates
    ].copy()

    if sample.empty:
        raise RuntimeError(
            "No benchmark molecules satisfy --min-candidates"
        )

    selected_specs = set(
        sample[
            "spec"
        ].astype(str)
    )

    candidates = candidates.loc[
        candidates[
            "spec"
        ]
        .astype(str)
        .isin(
            selected_specs
        )
    ].copy()

    print(
        "Benchmark molecules:",
        len(sample),
    )
    print(
        "PubChem candidate rows:",
        f"{len(candidates):,}",
    )
    print(
        "Spectral neighbors:",
        args.spectral_neighbors,
    )

    print()
    print(
        "Loading strict retrieval index..."
    )

    metadata, vectors = (
        load_retrieval_index(
            STRICT_INDEX_DIR
        )
    )

    ref_keys = (
        metadata[
            "inchikey14"
        ]
        .astype(str)
        .to_numpy()
    )

    ref_modes = (
        metadata[
            "ionization_mode"
        ]
        .astype(str)
        .to_numpy()
    )

    indexed_keys = set(
        ref_keys
    )

    print(
        "Reference spectra:",
        f"{len(metadata):,}",
    )
    print(
        "Reference structures:",
        f"{len(indexed_keys):,}",
    )

    print()
    print(
        "Loading training structure metadata..."
    )

    training_smiles = (
        build_training_smiles(
            indexed_keys
        )
    )

    print(
        "Training structures with SMILES:",
        f"{len(training_smiles):,}",
    )

    training_fps = {}

    invalid_training_smiles = 0

    for key, smiles in (
        training_smiles.items()
    ):
        fp = morgan_fp(
            smiles
        )

        if fp is None:
            invalid_training_smiles += 1
            continue

        training_fps[
            key
        ] = fp

    print(
        "Training fingerprints:",
        f"{len(training_fps):,}",
    )
    print(
        "Invalid training SMILES:",
        f"{invalid_training_smiles:,}",
    )

    #
    # Cache candidate fingerprints by structure because the same
    # formula can occur in more than one sampled benchmark row.
    #
    candidate_fp_cache = {}

    ranked_rows = []
    summary_rows = []

    for number, row in enumerate(
        sample.itertuples(
            index=False
        ),
        start=1,
    ):
        spec = str(
            row.spec
        )

        formula = str(
            row.truth_formula
        )

        truth_key = str(
            row.inchikey14
        )

        #
        # The PubChem/MIST benchmark sample is deliberately restricted
        # to checkpoint-supported positive-ion spectra.
        #
        mode = "positive"

        mode_indices = np.flatnonzero(
            ref_modes
            == mode
        )

        if len(
            mode_indices
        ) == 0:
            raise RuntimeError(
                "Strict retrieval index has no references for "
                f"mode {mode!r}"
            )

        query_vector = (
            bin_spectrum(
                row.ms2_mzs,
                row.ms2_normalized_intensities,
                max_mz=MAX_MZ,
                bin_width=BIN_WIDTH,
            )
        )

        neighbors = (
            top_structure_neighbors(
                query_vector,
                vectors[
                    mode_indices
                ],
                ref_keys[
                    mode_indices
                ],
                args.spectral_neighbors,
            )
        )

        spec_candidates = (
            candidates.loc[
                candidates[
                    "spec"
                ].astype(str)
                == spec
            ]
            .drop_duplicates(
                subset=[
                    "inchikey14",
                ]
            )
            .copy()
        )

        valid_rows = []
        valid_fps = []

        for candidate in (
            spec_candidates
            .itertuples(
                index=False
            )
        ):
            key = str(
                candidate.inchikey14
            )

            fp = candidate_fp_cache.get(
                key
            )

            if key not in (
                candidate_fp_cache
            ):
                fp = morgan_fp(
                    candidate.smiles
                )

                candidate_fp_cache[
                    key
                ] = fp

            if fp is None:
                continue

            valid_rows.append(
                candidate
            )

            valid_fps.append(
                fp
            )

        if not valid_rows:
            raise RuntimeError(
                f"No valid PubChem candidate fingerprints for {spec}"
            )

        (
            prior_scores,
            usable_neighbors,
        ) = score_candidates(
            valid_fps,
            neighbors,
            training_fps,
        )

        ranked = pd.DataFrame(
            {
                "candidate_inchikey14": [
                    str(
                        candidate.inchikey14
                    )
                    for candidate
                    in valid_rows
                ],
                "candidate_smiles": [
                    str(
                        candidate.smiles
                    )
                    for candidate
                    in valid_rows
                ],
                "neighbor_prior_score":
                    prior_scores,
            }
        )

        ranked = (
            ranked
            .sort_values(
                [
                    "neighbor_prior_score",
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

        ranked[
            "prior_rank"
        ] = (
            np.arange(
                len(ranked),
                dtype=np.int64,
            )
            + 1
        )

        truth_rows = ranked.loc[
            ranked[
                "candidate_inchikey14"
            ]
            == truth_key
        ]

        if truth_rows.empty:
            truth_rank = None
        else:
            truth_rank = int(
                truth_rows.iloc[
                    0
                ][
                    "prior_rank"
                ]
            )

        top_neighbor_key = (
            neighbors[0][0]
            if neighbors
            else None
        )

        top_neighbor_score = (
            neighbors[0][1]
            if neighbors
            else np.nan
        )

        summary_row = {
            "spec":
                spec,
            "truth_inchikey14":
                truth_key,
            "truth_formula":
                formula,
            "burden_bucket":
                row.burden_bucket,
            "candidate_count":
                int(
                    row.candidate_count
                ),
            "valid_candidate_count":
                len(ranked),
            "spectral_neighbor_count":
                len(neighbors),
            "usable_neighbor_fingerprints":
                usable_neighbors,
            "top_spectral_neighbor":
                top_neighbor_key,
            "top_spectral_cosine":
                top_neighbor_score,
            "truth_prior_rank":
                truth_rank,
        }

        for shortlist_size in (
            SHORTLIST_SIZES
        ):
            summary_row[
                f"truth_in_top_{shortlist_size}"
            ] = (
                truth_rank is not None
                and truth_rank
                <= shortlist_size
            )

        summary_rows.append(
            summary_row
        )

        ranked[
            "spec"
        ] = spec

        ranked[
            "truth_inchikey14"
        ] = truth_key

        ranked_rows.append(
            ranked[
                [
                    "spec",
                    "truth_inchikey14",
                    "candidate_inchikey14",
                    "candidate_smiles",
                    "neighbor_prior_score",
                    "prior_rank",
                ]
            ]
        )

        print(
            f"  {number:>3}/{len(sample)} "
            f"{spec}: "
            f"{len(ranked):,} candidates, "
            f"truth rank "
            f"{truth_rank if truth_rank is not None else 'missing'}"
        )

    result = pd.concat(
        ranked_rows,
        ignore_index=True,
    )

    summary = pd.DataFrame(
        summary_rows
    )

    result.to_parquet(
        OUTPUT_PATH,
        index=False,
    )

    summary.to_parquet(
        SUMMARY_PATH,
        index=False,
    )

    print()
    print(
        "Spectral-neighbor/Morgan pruning"
    )
    print(
        "================================"
    )

    def report(
        label: str,
        frame: pd.DataFrame,
    ) -> None:
        print()
        print(label)
        print(
            "-" * len(label)
        )
        print(
            "Molecules:",
            len(frame),
        )
        print(
            "Truth fingerprinted:",
            f"{frame['truth_prior_rank'].notna().mean():.1%}",
        )

        for shortlist_size in (
            SHORTLIST_SIZES
        ):
            print(
                f"Truth retained @ {shortlist_size:>3}:",
                f"{frame[f'truth_in_top_{shortlist_size}'].mean():.1%}",
            )

        ranks = (
            frame[
                "truth_prior_rank"
            ]
            .dropna()
        )

        if not ranks.empty:
            print(
                "Median truth prior rank:",
                f"{float(ranks.median()):.1f}",
            )

    report(
        "Overall",
        summary,
    )

    for bucket_name in [
        "le100",
        "101to500",
        "501to1000",
        "1001to5000",
    ]:
        bucket = summary.loc[
            summary[
                "burden_bucket"
            ]
            == bucket_name
        ]

        if not bucket.empty:
            report(
                bucket_name,
                bucket,
            )

    large = summary.loc[
        summary[
            "candidate_count"
        ]
        > 500
    ]

    if not large.empty:
        report(
            "Candidate count >500",
            large,
        )

    print()
    print(
        f"Wrote {OUTPUT_PATH}"
    )
    print(
        f"Wrote {SUMMARY_PATH}"
    )


if __name__ == "__main__":
    main()
