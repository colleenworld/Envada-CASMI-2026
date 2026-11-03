from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from casmi26.chemistry.inference import (
    neutral_mass_from_adduct,
)
from casmi26.public_models.fpnet import (
    ADDUCT_INDEX,
    FPNet,
    instrument_family,
    load_checkpoint,
    prep_peaks,
)


ROOT = Path(__file__).resolve().parent.parent

ASSET_ROOT = (
    ROOT
    / "data/external/casmi26-public-assets"
)

POOL_ROOT = (
    ASSET_ROOT
    / "casmi26-v2-pool"
)

MODEL_PATH = (
    ASSET_ROOT
    / "casmi26-fpnet-full1/fpnet_full1.pt"
)

BLIND_TEST_PATH = (
    ROOT
    / "data/processed/blind_pseudotest/test.parquet"
)

BLIND_TRUTH_PATH = (
    ROOT
    / "data/processed/blind_pseudotest/truth.parquet"
)

OUTPUT_PATH = (
    ROOT
    / "data/processed/blind_pseudotest/"
      "public_fpnet_pool_ranks.parquet"
)

SUMMARY_PATH = (
    ROOT
    / "data/processed/blind_pseudotest/"
      "public_fpnet_pool_summary.csv"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "First-pass reproduction of public FPNet scoring over "
            "the full 710k public v2 pool."
        )
    )

    parser.add_argument(
        "--max-molecules",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--ppm",
        type=float,
        default=10.0,
    )

    parser.add_argument(
        "--device",
        default=(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        ),
    )

    return parser.parse_args()


def _first_existing(
    frame: pd.DataFrame,
    names: list[str],
):
    for name in names:
        if name in frame.columns:
            return name
    return None


def collision_energy_value(
    value,
) -> float:
    if value is None:
        return 0.0

    try:
        array = np.atleast_1d(
            np.asarray(
                value,
                dtype=np.float32,
            )
        )
    except Exception:
        return 0.0

    array = array[
        np.isfinite(
            array
        )
    ]

    if array.size == 0:
        return 0.0

    return float(
        array.mean()
    )


def neutral_mass_for_group(
    group: pd.DataFrame,
) -> float | None:
    values = []

    for row in group.itertuples(
        index=False
    ):
        value = neutral_mass_from_adduct(
            getattr(
                row,
                "precursor_mz",
            ),
            getattr(
                row,
                "adduct",
            ),
        )

        if np.isfinite(
            value
        ):
            values.append(
                float(
                    value
                )
            )

    if not values:
        return None

    return float(
        np.median(
            values
        )
    )


def predict_logits(
    model: FPNet,
    group: pd.DataFrame,
    *,
    device: str,
) -> np.ndarray | None:
    instrument_column = _first_existing(
        group,
        [
            "instrument",
            "instrument_type",
            "instrument_name",
        ],
    )

    ce_column = _first_existing(
        group,
        [
            "collision_energy_ev",
            "collision_energy",
        ],
    )

    rows = list(
        group.itertuples(
            index=False
        )
    )[:16]

    prepared = []

    for row in rows:
        precursor = float(
            getattr(
                row,
                "precursor_mz",
            )
        )

        mz, intensity = prep_peaks(
            getattr(
                row,
                "ms2_mzs",
            ),
            getattr(
                row,
                "ms2_normalized_intensities",
            ),
            precursor,
        )

        if mz.size == 0:
            continue

        adduct = str(
            getattr(
                row,
                "adduct",
            )
        )

        mode_value = getattr(
            row,
            "ionization_mode",
            "positive",
        )

        mode = (
            1.0
            if str(
                mode_value
            ).lower()
            == "positive"
            else -1.0
        )

        instrument = (
            getattr(
                row,
                instrument_column,
            )
            if instrument_column
            else None
        )

        collision_energy = (
            collision_energy_value(
                getattr(
                    row,
                    ce_column,
                )
            )
            if ce_column
            else 0.0
        )

        prepared.append(
            {
                "mz": mz,
                "intensity": intensity,
                "precursor": precursor,
                "adduct": ADDUCT_INDEX.get(
                    adduct,
                    ADDUCT_INDEX[
                        "<unk>"
                    ],
                ),
                "instrument": instrument_family(
                    instrument
                ),
                "collision_energy": collision_energy,
                "mode": mode,
            }
        )

    if not prepared:
        return None

    peak_count = max(
        len(
            item[
                "mz"
            ]
        )
        for item in prepared
    )

    batch_size = len(
        prepared
    )

    mz = np.zeros(
        (
            batch_size,
            peak_count,
        ),
        dtype=np.float32,
    )

    intensity = np.zeros_like(
        mz
    )

    padding = np.ones(
        (
            batch_size,
            peak_count,
        ),
        dtype=bool,
    )

    precursor = np.zeros(
        batch_size,
        dtype=np.float32,
    )

    adduct = np.zeros(
        batch_size,
        dtype=np.int64,
    )

    instrument = np.zeros(
        batch_size,
        dtype=np.int64,
    )

    collision_energy = np.zeros(
        batch_size,
        dtype=np.float32,
    )

    mode = np.zeros(
        batch_size,
        dtype=np.float32,
    )

    for index, item in enumerate(
        prepared
    ):
        count = len(
            item[
                "mz"
            ]
        )

        mz[
            index,
            :count,
        ] = item[
            "mz"
        ]

        intensity[
            index,
            :count,
        ] = item[
            "intensity"
        ]

        padding[
            index,
            :count,
        ] = False

        precursor[
            index
        ] = item[
            "precursor"
        ]

        adduct[
            index
        ] = item[
            "adduct"
        ]

        instrument[
            index
        ] = item[
            "instrument"
        ]

        collision_energy[
            index
        ] = item[
            "collision_energy"
        ]

        mode[
            index
        ] = item[
            "mode"
        ]

    tensor = lambda value: torch.as_tensor(
        value,
        device=device,
    )

    with torch.inference_mode():
        logits = model(
            tensor(
                mz
            ),
            tensor(
                intensity
            ),
            tensor(
                padding
            ),
            tensor(
                precursor
            ),
            tensor(
                adduct
            ),
            tensor(
                instrument
            ),
            tensor(
                collision_energy
            ),
            tensor(
                mode
            ),
        )

    return (
        logits
        .float()
        .mean(
            dim=0
        )
        .cpu()
        .numpy()
        .astype(
            np.float32
        )
    )


def main() -> None:
    args = parse_args()

    required = [
        MODEL_PATH,
        POOL_ROOT
        / "pool_meta.parquet",
        POOL_ROOT
        / "pool_fp.npy",
        POOL_ROOT
        / "fp_bits.npy",
        BLIND_TEST_PATH,
        BLIND_TRUTH_PATH,
    ]

    for path in required:
        if not path.exists():
            raise FileNotFoundError(
                f"Missing required artifact: {path}"
            )

    model, checkpoint = load_checkpoint(
        MODEL_PATH,
        device=args.device,
    )

    meta = pd.read_parquet(
        POOL_ROOT
        / "pool_meta.parquet",
        columns=[
            "key",
            "mass",
        ],
    )

    masses = (
        meta[
            "mass"
        ]
        .to_numpy(
            dtype=np.float64,
        )
    )

    keys = (
        meta[
            "key"
        ]
        .astype(str)
        .to_numpy()
    )

    if not np.all(
        np.diff(
            masses
        )
        >= 0
    ):
        raise ValueError(
            "Public pool is not mass-sorted"
        )

    bits = np.load(
        POOL_ROOT
        / "fp_bits.npy"
    )

    packed_fingerprints = np.load(
        POOL_ROOT
        / "pool_fp.npy",
        mmap_mode="r",
    )

    nbits = int(
        checkpoint[
            "nbits"
        ]
    )

    if len(bits) != nbits:
        raise ValueError(
            "FPNet output width does not match fp_bits.npy: "
            f"{nbits} vs {len(bits)}"
        )

    if (
        packed_fingerprints.shape[0]
        != len(meta)
    ):
        raise ValueError(
            "pool_fp.npy row count does not match pool_meta.parquet"
        )

    test = pd.read_parquet(
        BLIND_TEST_PATH
    ).copy()

    truth = pd.read_parquet(
        BLIND_TRUTH_PATH,
        columns=[
            "molecule_id",
            "inchikey14",
        ],
    ).copy()

    test[
        "molecule_id"
    ] = (
        test[
            "molecule_id"
        ]
        .astype(str)
    )

    truth[
        "molecule_id"
    ] = (
        truth[
            "molecule_id"
        ]
        .astype(str)
    )

    truth[
        "inchikey14"
    ] = (
        truth[
            "inchikey14"
        ]
        .astype(str)
    )

    groups = {
        str(
            molecule_id
        ): group.copy()
        for molecule_id, group
        in test.groupby(
            "molecule_id",
            sort=False,
        )
    }

    rows = []

    truth_rows = list(
        truth.itertuples(
            index=False
        )
    )

    if (
        args.max_molecules
        is not None
    ):
        truth_rows = truth_rows[
            :args.max_molecules
        ]

    for number, row in enumerate(
        truth_rows,
        start=1,
    ):
        molecule_id = str(
            row.molecule_id
        )

        truth_key = str(
            row.inchikey14
        )

        group = groups.get(
            molecule_id
        )

        if group is None:
            continue

        target = neutral_mass_for_group(
            group
        )

        if target is None:
            continue

        logits = predict_logits(
            model,
            group,
            device=args.device,
        )

        if logits is None:
            continue

        tolerance = (
            target
            * args.ppm
            / 1_000_000.0
        )

        left = int(
            np.searchsorted(
                masses,
                target
                - tolerance,
                side="left",
            )
        )

        right = int(
            np.searchsorted(
                masses,
                target
                + tolerance,
                side="right",
            )
        )

        candidate_count = (
            right
            - left
        )

        if candidate_count <= 0:
            rows.append(
                {
                    "molecule_id":
                        molecule_id,
                    "candidate_count":
                        0,
                    "truth_present":
                        False,
                    "truth_rank":
                        np.nan,
                }
            )
            continue

        candidate_keys = keys[
            left:right
        ]

        truth_positions = np.flatnonzero(
            candidate_keys
            == truth_key
        )

        if truth_positions.size == 0:
            rows.append(
                {
                    "molecule_id":
                        molecule_id,
                    "candidate_count":
                        candidate_count,
                    "truth_present":
                        False,
                    "truth_rank":
                        np.nan,
                }
            )
            continue

        fingerprints = np.unpackbits(
            np.asarray(
                packed_fingerprints[
                    left:right
                ]
            ),
            axis=1,
        )[
            :,
            :nbits,
        ].astype(
            np.float32,
            copy=False,
        )

        scores = (
            fingerprints
            @ logits
        )

        truth_score = float(
            np.max(
                scores[
                    truth_positions
                ]
            )
        )

        rank = (
            int(
                np.sum(
                    scores
                    > truth_score
                )
            )
            + 1
        )

        rows.append(
            {
                "molecule_id":
                    molecule_id,
                "candidate_count":
                    candidate_count,
                "truth_present":
                    True,
                "truth_rank":
                    rank,
                "truth_score":
                    truth_score,
                "best_score":
                    float(
                        scores.max()
                    ),
            }
        )

        if (
            number % 25 == 0
            or number
            == len(
                truth_rows
            )
        ):
            evaluated = len(
                rows
            )

            present = sum(
                bool(
                    item[
                        "truth_present"
                    ]
                )
                for item in rows
            )

            top25 = sum(
                (
                    bool(
                        item[
                            "truth_present"
                        ]
                    )
                    and float(
                        item[
                            "truth_rank"
                        ]
                    )
                    <= 25
                )
                for item in rows
            )

            print(
                f"{number:,}/{len(truth_rows):,} "
                f"| pool recall {present / evaluated:.2%} "
                f"| FPNet Top25 {top25 / evaluated:.2%}"
            )

    result = pd.DataFrame(
        rows
    )

    cutoffs = [
        1,
        5,
        10,
        25,
        100,
        250,
        500,
        1000,
    ]

    summary_rows = []

    for cutoff in cutoffs:
        hits = int(
            (
                result[
                    "truth_present"
                ]
                & (
                    result[
                        "truth_rank"
                    ]
                    <= cutoff
                )
            ).sum()
        )

        summary_rows.append(
            {
                "cutoff":
                    cutoff,
                "queries":
                    len(result),
                "truth_hits":
                    hits,
                "recall":
                    hits
                    / len(result)
                    if len(result)
                    else 0.0,
            }
        )

    summary = pd.DataFrame(
        summary_rows
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_parquet(
        OUTPUT_PATH,
        index=False,
        compression="zstd",
    )

    summary.to_csv(
        SUMMARY_PATH,
        index=False,
    )

    print()
    print(
        "Public FPNet / v2-pool mechanics benchmark"
    )
    print(
        "=" * 72
    )

    print(
        "NOTE: this uses the released full-training FPNet and full public pool."
    )

    print(
        "It is a Kaggle-production mechanics check, not leakage-safe blind validation."
    )

    print()

    print(
        summary.to_string(
            index=False,
            formatters={
                "recall":
                    lambda value:
                        f"{value:.2%}",
            },
        )
    )

    print()

    present = result.loc[
        result[
            "truth_present"
        ]
    ]

    print(
        "Pool candidate availability:",
        f"{len(present)}/{len(result)} "
        f"({len(present) / len(result):.2%})"
        if len(result)
        else "n/a",
    )

    if len(present):
        print(
            "Candidate count median:",
            f"{present['candidate_count'].median():.0f}",
        )

        print(
            "Candidate count p95:",
            f"{present['candidate_count'].quantile(0.95):.0f}",
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
