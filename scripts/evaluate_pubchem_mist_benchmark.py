from __future__ import annotations

import argparse
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = (
    Path(__file__).resolve().parent.parent
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

OUTPUT_PATH = (
    BENCHMARK_DIR
    / "ranked.parquet"
)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "prediction_path",
        type=Path,
    )

    return parser.parse_args()


def to_str(value) -> str:
    if isinstance(
        value,
        bytes,
    ):
        return value.decode()

    return str(value)


def main() -> None:
    args = parse_args()

    sample = pd.read_parquet(
        SAMPLE_PATH
    )

    with args.prediction_path.open(
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
            "Prediction artifact is missing keys: "
            f"{sorted(missing)}"
        )

    sample_lookup = (
        sample
        .set_index(
            "spec"
        )
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

        if spec not in (
            sample_lookup.index
        ):
            raise ValueError(
                f"Unexpected MIST spec: {spec}"
            )

        meta = sample_lookup.loc[
            spec
        ]

        if not (
            len(ikeys)
            == len(smiles)
            == len(dists)
        ):
            raise ValueError(
                f"Candidate array mismatch for {spec}"
            )

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

        truth_key = str(
            meta.inchikey14
        )

        truth_rank = None

        for rank, (
            ikey,
            smiles_value,
            distance,
        ) in enumerate(
            ordered,
            start=1,
        ):
            ikey = to_str(
                ikey
            )

            if (
                truth_rank is None
                and ikey
                == truth_key
            ):
                truth_rank = rank

            rows.append(
                {
                    "spec":
                        spec,
                    "truth_inchikey14":
                        truth_key,
                    "candidate_inchikey14":
                        ikey,
                    "candidate_smiles":
                        to_str(
                            smiles_value
                        ),
                    "distance":
                        float(
                            distance
                        ),
                    "rank":
                        rank,
                    "burden_bucket":
                        meta.burden_bucket,
                    "candidate_count":
                        int(
                            meta.candidate_count
                        ),
                }
            )

        sample.loc[
            sample[
                "spec"
            ]
            == spec,
            "truth_rank",
        ] = (
            truth_rank
            if truth_rank
            is not None
            else np.nan
        )

    ranked = pd.DataFrame(
        rows
    )

    ranked.to_parquet(
        OUTPUT_PATH,
        index=False,
    )

    sample[
        "truth_rank"
    ] = pd.to_numeric(
        sample.get(
            "truth_rank"
        ),
        errors="coerce",
    )

    sample[
        "reciprocal_rank_at_25"
    ] = np.where(
        sample[
            "truth_rank"
        ].between(
            1,
            25,
            inclusive="both",
        ),
        1.0
        / sample[
            "truth_rank"
        ],
        0.0,
    )

    print(
        "Strict-validation PubChem/MIST ranking"
    )
    print(
        "======================================"
    )

    def report(
        label: str,
        frame: pd.DataFrame,
    ) -> None:
        ranks = frame[
            "truth_rank"
        ]

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
            "MRR@25:",
            f"{frame['reciprocal_rank_at_25'].mean():.4f}",
        )
        print(
            "Top-1:",
            f"{(ranks == 1).mean():.1%}",
        )
        print(
            "Top-25:",
            f"{ranks.between(1, 25).mean():.1%}",
        )
        print(
            "Truth returned:",
            f"{ranks.notna().mean():.1%}",
        )

    report(
        "Overall",
        sample,
    )

    for bucket_name in [
        "le100",
        "101to500",
        "501to1000",
        "1001to5000",
    ]:
        bucket = sample.loc[
            sample[
                "burden_bucket"
            ]
            == bucket_name
        ]

        if not bucket.empty:
            report(
                bucket_name,
                bucket,
            )

    print()
    print(
        f"Wrote {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()
