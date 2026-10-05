from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


TRAIN_PATH = Path("data/raw/train.parquet")

FEATURES_PATH = Path(
    "data/processed/results/"
    "coconut_ranking_features.parquet"
)

TRAINING_STRUCTURES_PATH = Path(
    "data/processed/structure_index/"
    "training_structures.parquet"
)

COCONUT_PATH = Path(
    "data/external/coconut/"
    "coconut_approved.parquet"
)

OUTPUT_DIR = Path(
    "data/processed/mist/benchmark_100"
)

MGF_PATH = OUTPUT_DIR / "spectra.mgf"
SMILES_PATH = OUTPUT_DIR / "lookup_smiles.txt"
MANIFEST_PATH = OUTPUT_DIR / "manifest.parquet"
CANDIDATES_PATH = OUTPUT_DIR / "candidates.parquet"

BENCHMARK_SIZE = 100
RANDOM_SEED = 42


def select_queries(
    features: pd.DataFrame,
) -> pd.DataFrame:
    """
    Select a deterministic benchmark of queries for
    which:

      * the truth survived candidate generation
      * there are at least 2 same-formula structures
      * there are no more than 100 same-formula
        structures

    The upper bound keeps the first MIST experiment
    reasonably small while retaining meaningful
    structural ambiguity.
    """

    rows: list[dict[str, object]] = []

    for query_key, group in features.groupby(
        "query_inchikey14",
        sort=False,
    ):
        truth = group[group["is_truth"]]

        if len(truth) != 1:
            continue

        truth_row = truth.iloc[0]

        truth_formula = str(
            truth_row["candidate_formula"]
        )

        same_formula = group[
            group["candidate_formula"]
            == truth_formula
        ]

        candidate_count = len(same_formula)

        if candidate_count < 2:
            continue

        if candidate_count > 100:
            continue

        rows.append(
            {
                "query_inchikey14":
                    str(query_key),
                "truth_formula":
                    truth_formula,
                "candidate_count":
                    candidate_count,
            }
        )

    eligible = pd.DataFrame(rows)

    if len(eligible) < BENCHMARK_SIZE:
        raise RuntimeError(
            f"Only {len(eligible):,} eligible "
            f"queries; need {BENCHMARK_SIZE}"
        )

    #
    # Stratify approximately by candidate-set size
    # rather than accidentally selecting only easy
    # cases.
    #

    eligible["ambiguity_bin"] = pd.cut(
        eligible["candidate_count"],
        bins=[
            1,
            5,
            10,
            25,
            50,
            100,
        ],
        labels=[
            "2-5",
            "6-10",
            "11-25",
            "26-50",
            "51-100",
        ],
        include_lowest=True,
    )

    print("Eligible queries by ambiguity")
    print(
        eligible["ambiguity_bin"]
        .value_counts()
        .sort_index()
    )

    #
    # 20 from each ambiguity bucket where possible.
    #

    rng = np.random.default_rng(
        RANDOM_SEED
    )

    selected_parts: list[pd.DataFrame] = []

    per_bin = BENCHMARK_SIZE // 5

    for _, group in eligible.groupby(
        "ambiguity_bin",
        observed=True,
        sort=True,
    ):
        n = min(
            per_bin,
            len(group),
        )

        indices = rng.choice(
            group.index.to_numpy(),
            size=n,
            replace=False,
        )

        selected_parts.append(
            group.loc[indices]
        )

    selected = pd.concat(
        selected_parts,
        ignore_index=True,
    )

    #
    # If a bin contained fewer than 20 examples,
    # fill the remaining slots from the unused pool.
    #

    if len(selected) < BENCHMARK_SIZE:
        selected_keys = set(
            selected["query_inchikey14"]
        )

        remaining = eligible[
            ~eligible["query_inchikey14"]
            .isin(selected_keys)
        ]

        needed = (
            BENCHMARK_SIZE
            - len(selected)
        )

        indices = rng.choice(
            remaining.index.to_numpy(),
            size=needed,
            replace=False,
        )

        selected = pd.concat(
            [
                selected,
                remaining.loc[indices],
            ],
            ignore_index=True,
        )

    selected = selected.sort_values(
        [
            "candidate_count",
            "query_inchikey14",
        ]
    ).reset_index(drop=True)

    return selected


def build_structure_lookup() -> pd.DataFrame:
    """
    Build one SMILES record per candidate structure.

    Prefer COCONUT SMILES when available because all
    benchmark truths are independently represented
    in COCONUT.
    """

    training = pd.read_parquet(
        TRAINING_STRUCTURES_PATH
    ).copy()

    coconut = pd.read_parquet(
        COCONUT_PATH
    ).copy()

    rows: dict[str, dict[str, object]] = {}

    #
    # Training structure index column naming may
    # differ from COCONUT, so tolerate either
    # "smiles" or "normalized_smiles".
    #

    if "smiles" in training.columns:
        training_smiles_col = "smiles"
    elif "normalized_smiles" in training.columns:
        training_smiles_col = (
            "normalized_smiles"
        )
    else:
        raise RuntimeError(
            "Training structure index has no "
            "SMILES column"
        )

    for row in training.itertuples(
        index=False
    ):
        key = str(row.inchikey14)

        smiles = getattr(
            row,
            training_smiles_col,
        )

        if pd.isna(smiles):
            continue

        rows[key] = {
            "inchikey14": key,
            "smiles": str(smiles),
            "smiles_source": "training",
        }

    for row in coconut.itertuples(
        index=False
    ):
        key = str(row.inchikey14)

        if pd.isna(row.smiles):
            continue

        #
        # COCONUT wins if the structure occurs
        # in both catalogs.
        #

        rows[key] = {
            "inchikey14": key,
            "smiles": str(row.smiles),
            "smiles_source": "coconut",
        }

    return pd.DataFrame(
        rows.values()
    )


def collect_spectra(
    selected_keys: set[str],
) -> pd.DataFrame:
    parquet = pq.ParquetFile(
        TRAIN_PATH
    )

    columns = [
        "inchikey14",
        "molecular_formula",
        "ionization_mode",
        "adduct",
        "precursor_mz",
        "ms2_mzs",
        "ms2_normalized_intensities",
        "instrument_type",
    ]

    frames: list[pd.DataFrame] = []

    scanned = 0

    print()
    print("Collecting benchmark spectra...")

    for batch in parquet.iter_batches(
        columns=columns,
        batch_size=100_000,
    ):
        frame = batch.to_pandas()

        scanned += len(frame)

        mask = frame[
            "inchikey14"
        ].astype(str).isin(
            selected_keys
        )

        if mask.any():
            frames.append(
                frame.loc[mask].copy()
            )

        print(
            f"\rRows scanned: {scanned:,}",
            end="",
            flush=True,
        )

    print()

    if not frames:
        raise RuntimeError(
            "No benchmark spectra found"
        )

    return pd.concat(
        frames,
        ignore_index=True,
    )


def write_mgf(
    spectra: pd.DataFrame,
    selected: pd.DataFrame,
) -> pd.DataFrame:
    """
    Export one representative spectrum per query.

    For this first MIST experiment, prefer [M+H]+
    because the public CANOPUS/NPLIB1 MIST model is
    based on H+ spectra.

    If a query has no [M+H]+ spectrum, it is omitted
    from this first benchmark rather than silently
    mixing ionization regimes.
    """

    selected_lookup = (
        selected.set_index(
            "query_inchikey14"
        )
    )

    manifest_rows: list[
        dict[str, object]
    ] = []

    with MGF_PATH.open(
        "w",
        encoding="utf-8",
    ) as handle:
        for query_key, group in spectra.groupby(
            "inchikey14",
            sort=False,
        ):
            query_key = str(query_key)

            if query_key not in (
                selected_lookup.index
            ):
                continue

            #
            # Public MIST quickstart/model is
            # primarily H+ based. Keep this first
            # benchmark chemically compatible.
            #

            preferred = group[
                group["adduct"]
                == "[M+H]+"
            ].copy()

            if preferred.empty:
                continue

            #
            # Prefer the spectrum with the largest
            # number of peaks.
            #

            preferred[
                "_peak_count"
            ] = preferred[
                "ms2_mzs"
            ].apply(
                lambda x: (
                    len(x)
                    if x is not None
                    else 0
                )
            )

            row = preferred.sort_values(
                "_peak_count",
                ascending=False,
                kind="stable",
            ).iloc[0]

            mzs = np.asarray(
                row["ms2_mzs"],
                dtype=np.float64,
            )

            intensities = np.asarray(
                row[
                    "ms2_normalized_intensities"
                ],
                dtype=np.float64,
            )

            if len(mzs) == 0:
                continue

            if len(mzs) != len(intensities):
                raise RuntimeError(
                    f"{query_key}: m/z and "
                    "intensity lengths differ"
                )

            valid = (
                np.isfinite(mzs)
                & np.isfinite(intensities)
                & (mzs > 0)
                & (intensities >= 0)
            )

            mzs = mzs[valid]
            intensities = intensities[
                valid
            ]

            if len(mzs) == 0:
                continue

            spec_id = (
                f"casmi_{query_key}"
            )

            formula = str(
                selected_lookup.loc[
                    query_key,
                    "truth_formula",
                ]
            )

            precursor_mz = float(
                row["precursor_mz"]
            )

            #
            # MGF metadata. MIST's quickstart
            # consumes MGF input; we retain useful
            # metadata even if a particular parser
            # ignores some fields.
            #

            handle.write(
                "BEGIN IONS\n"
            )

            handle.write(
                f"TITLE={spec_id}\n"
            )

            handle.write(
                f"FEATURE_ID={spec_id}\n"
            )

            handle.write(
                f"PEPMASS={precursor_mz:.8f}\n"
            )

            handle.write(
                "CHARGE=1+\n"
            )

            handle.write(
                f"FORMULA={formula}\n"
            )

            handle.write(
                "ADDUCT=[M+H]+\n"
            )

            handle.write(
                "IONMODE=positive\n"
            )

            for mz, intensity in zip(
                mzs,
                intensities,
                strict=True,
            ):
                handle.write(
                    f"{mz:.8f} "
                    f"{intensity:.8f}\n"
                )

            handle.write(
                "END IONS\n\n"
            )

            manifest_rows.append(
                {
                    "spec":
                        spec_id,
                    "query_inchikey14":
                        query_key,
                    "truth_formula":
                        formula,
                    "source_row_peak_count": len(mzs),
                    "precursor_mz":
                        precursor_mz,
                    "adduct":
                        "[M+H]+",
                    "ionization_mode":
                        str(
                            row[
                                "ionization_mode"
                            ]
                        ),
                    "instrument_type":
                        (
                            None
                            if pd.isna(
                                row[
                                    "instrument_type"
                                ]
                            )
                            else str(
                                row[
                                    "instrument_type"
                                ]
                            )
                        ),
                    "peak_count":
                        len(mzs),
                    "candidate_count":
                        int(
                            selected_lookup.loc[
                                query_key,
                                "candidate_count",
                            ]
                        ),
                }
            )

    return pd.DataFrame(
        manifest_rows
    )


def main() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    features = pd.read_parquet(
        FEATURES_PATH
    )

    print("Selecting benchmark queries...")

    selected = select_queries(
        features
    )

    print()
    print(
        f"Initially selected: "
        f"{len(selected):,}"
    )

    print(
        "Candidate counts: "
        f"median="
        f"{selected['candidate_count'].median():.0f} "
        f"min="
        f"{selected['candidate_count'].min()} "
        f"max="
        f"{selected['candidate_count'].max()}"
    )

    selected_keys = set(
        selected[
            "query_inchikey14"
        ]
    )

    spectra = collect_spectra(
        selected_keys
    )

    manifest = write_mgf(
        spectra,
        selected,
    )

    if manifest.empty:
        raise RuntimeError(
            "No compatible [M+H]+ benchmark "
            "spectra were found"
        )

    #
    # Only keep queries successfully exported
    # to MGF.
    #

    exported_keys = set(
        manifest[
            "query_inchikey14"
        ]
    )

    #
    # Candidate table: restrict each query to
    # candidates sharing the truth formula.
    #

    candidates = features[
        features[
            "query_inchikey14"
        ].astype(str).isin(
            exported_keys
        )
    ].copy()

    truth_formula_lookup = dict(
        zip(
            manifest[
                "query_inchikey14"
            ],
            manifest[
                "truth_formula"
            ],
            strict=True,
        )
    )

    candidates = candidates[
        candidates.apply(
            lambda row: (
                str(
                    row[
                        "candidate_formula"
                    ]
                )
                == truth_formula_lookup[
                    str(
                        row[
                            "query_inchikey14"
                        ]
                    )
                ]
            ),
            axis=1,
        )
    ].copy()

    structures = (
        build_structure_lookup()
    )

    candidates = candidates.merge(
        structures,
        left_on=(
            "candidate_inchikey14"
        ),
        right_on="inchikey14",
        how="left",
        validate="many_to_one",
    )

    missing_smiles = candidates[
        "smiles"
    ].isna()

    if missing_smiles.any():
        missing = candidates.loc[
            missing_smiles,
            "candidate_inchikey14",
        ].nunique()

        print(
            f"WARNING: {missing:,} candidate "
            "structures have no SMILES and "
            "will be excluded"
        )

        candidates = candidates[
            ~missing_smiles
        ].copy()

    #
    # Confirm every exported truth still has
    # a candidate SMILES.
    #

    truth_candidates = candidates[
        candidates["is_truth"]
    ]

    truth_keys = set(
        truth_candidates[
            "query_inchikey14"
        ].astype(str)
    )

    lost_truths = (
        exported_keys - truth_keys
    )

    if lost_truths:
        raise RuntimeError(
            f"{len(lost_truths):,} exported "
            "queries lost their truth SMILES"
        )

    #
    # Assign a stable candidate ID. This is useful
    # when mapping MIST output back to our table.
    #

    candidates = candidates.sort_values(
        [
            "query_inchikey14",
            "candidate_inchikey14",
        ]
    ).reset_index(drop=True)

    candidates[
        "mist_candidate_id"
    ] = [
        f"candidate_{i:06d}"
        for i in range(
            len(candidates)
        )
    ]

    #
    # MIST quickstart accepts a reference SMILES
    # list. Write the unique candidate structures
    # participating in this benchmark.
    #

    unique_smiles = (
        candidates[
            [
                "candidate_inchikey14",
                "smiles",
            ]
        ]
        .drop_duplicates(
            subset=[
                "candidate_inchikey14"
            ]
        )
        .sort_values(
            "candidate_inchikey14"
        )
    )

    with SMILES_PATH.open(
        "w",
        encoding="utf-8",
    ) as handle:
        for smiles in (
            unique_smiles["smiles"]
        ):
            handle.write(
                f"{smiles}\n"
            )

    manifest.to_parquet(
        MANIFEST_PATH,
        index=False,
    )

    candidates.to_parquet(
        CANDIDATES_PATH,
        index=False,
    )

    print()
    print("MIST benchmark export")
    print("=" * 70)

    print(
        f"Selected queries:        "
        f"{len(selected):,}"
    )

    print(
        f"Exported H+ queries:     "
        f"{len(manifest):,}"
    )

    print(
        f"Candidate rows:          "
        f"{len(candidates):,}"
    )

    print(
        f"Unique candidate "
        f"structures: {len(unique_smiles):,}"
    )

    counts = (
        candidates.groupby(
            "query_inchikey14"
        )
        .size()
        .to_numpy()
    )

    print()
    print("Exported candidates/query")
    print("-" * 70)

    print(
        f"median={np.median(counts):.0f} "
        f"p90={np.percentile(counts, 90):.0f} "
        f"max={np.max(counts):,}"
    )

    print()
    print("Truth validation")
    print("-" * 70)

    print(
        f"Queries with truth:      "
        f"{truth_candidates['query_inchikey14'].nunique():,}"
    )

    print(
        f"Truth rows:              "
        f"{len(truth_candidates):,}"
    )

    print()
    print(f"MGF:        {MGF_PATH}")
    print(f"SMILES:     {SMILES_PATH}")
    print(f"Manifest:   {MANIFEST_PATH}")
    print(f"Candidates: {CANDIDATES_PATH}")


if __name__ == "__main__":
    main()