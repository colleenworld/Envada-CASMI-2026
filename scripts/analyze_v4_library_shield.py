from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_BANDS = [
    (-np.inf, 0.80, "<0.80"),
    (0.80, 0.88, "0.80-<0.88"),
    (0.88, 0.90, "0.88-<0.90"),
    (0.90, np.inf, ">=0.90"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze v4 library-confidence shielding from a Kaggle stage dump."
        )
    )
    parser.add_argument(
        "--base-json",
        type=Path,
        required=True,
        help="stage_cache/base.json from the v4 run",
    )
    parser.add_argument(
        "--submission-csv",
        type=Path,
        required=True,
        help="Final submission.csv produced by the same run",
    )
    parser.add_argument(
        "--run-manifest",
        type=Path,
        default=None,
        help="Optional run_manifest.json for fusion stats/config",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/diagnostics/v4_library_shield.csv"),
    )
    return parser.parse_args()


def first_smiles(value: object) -> str | None:
    text = str(value or "").strip()

    if not text or text == "nan":
        return None

    parts = [
        part.strip()
        for part in text.split(";")
        if part.strip()
    ]

    return parts[0] if parts else None


def band_for(lib_max: float) -> str:
    for low, high, label in DEFAULT_BANDS:
        if low <= lib_max < high:
            return label

    raise AssertionError(lib_max)


def load_base(path: Path) -> pd.DataFrame:
    payload = json.loads(
        path.read_text()
    )

    rows = []

    for molecule_id, value in payload.items():
        if not isinstance(value, list) or len(value) < 3:
            raise ValueError(
                f"Unexpected base entry for {molecule_id}: {type(value)}"
            )

        smiles = list(
            value[0] or []
        )

        lib_max = float(
            value[2]
        )

        rows.append(
            {
                "molecule_id":
                    str(molecule_id),
                "lib_max":
                    lib_max,
                "base_top1":
                    smiles[0]
                    if smiles
                    else None,
                "base_candidates":
                    len(smiles),
            }
        )

    return pd.DataFrame(
        rows
    )


def main() -> None:
    args = parse_args()

    if not args.base_json.exists():
        raise FileNotFoundError(
            args.base_json
        )

    if not args.submission_csv.exists():
        raise FileNotFoundError(
            args.submission_csv
        )

    base = load_base(
        args.base_json
    )

    submission = pd.read_csv(
        args.submission_csv,
        usecols=[
            "molecule_id",
            "smiles",
        ],
    )

    submission[
        "molecule_id"
    ] = (
        submission[
            "molecule_id"
        ]
        .astype(str)
    )

    submission[
        "final_top1"
    ] = submission[
        "smiles"
    ].map(
        first_smiles
    )

    joined = base.merge(
        submission[
            [
                "molecule_id",
                "final_top1",
            ]
        ],
        on="molecule_id",
        how="left",
        validate="one_to_one",
    )

    joined[
        "shield_088"
    ] = (
        joined[
            "lib_max"
        ]
        >= 0.88
    )

    joined[
        "library_band"
    ] = joined[
        "lib_max"
    ].map(
        band_for
    )

    joined[
        "top1_changed"
    ] = (
        joined[
            "base_top1"
        ].fillna("")
        != joined[
            "final_top1"
        ].fillna("")
    )

    joined[
        "top1_preserved"
    ] = ~joined[
        "top1_changed"
    ]

    summary = (
        joined.groupby(
            "library_band",
            observed=False,
        )
        .agg(
            molecules=(
                "molecule_id",
                "size",
            ),
            mean_lib_max=(
                "lib_max",
                "mean",
            ),
            shielded=(
                "shield_088",
                "sum",
            ),
            top1_changed=(
                "top1_changed",
                "sum",
            ),
            top1_change_rate=(
                "top1_changed",
                "mean",
            ),
        )
        .reset_index()
    )

    order = {
        label: i
        for i, (_, _, label)
        in enumerate(
            DEFAULT_BANDS
        )
    }

    summary[
        "_order"
    ] = summary[
        "library_band"
    ].map(
        order
    )

    summary = (
        summary
        .sort_values(
            "_order"
        )
        .drop(
            columns=[
                "_order"
            ]
        )
    )

    print()
    print(
        "V4 library-shield diagnostic"
    )
    print(
        "=" * 76
    )
    print(
        summary.to_string(
            index=False,
            formatters={
                "mean_lib_max":
                    lambda x: f"{x:.4f}",
                "top1_change_rate":
                    lambda x: f"{x:.2%}",
            },
        )
    )

    print()
    print(
        "Overall:",
        f"{len(joined):,} molecules;",
        f"{int(joined['shield_088'].sum()):,} at lib_max >= 0.88;",
        f"{int(joined['top1_changed'].sum()):,} final top-1 changes",
    )

    if args.run_manifest is not None:
        if not args.run_manifest.exists():
            raise FileNotFoundError(
                args.run_manifest
            )

        manifest = json.loads(
            args.run_manifest.read_text()
        )

        stats = manifest.get(
            "stats",
            {}
        )

        cfg = manifest.get(
            "cfg",
            {}
        )

        print()
        print(
            "Run manifest"
        )
        print(
            "-" * 76
        )
        print(
            "version:",
            manifest.get(
                "version"
            ),
        )
        print(
            "LIB_TAU:",
            cfg.get(
                "LIB_TAU"
            ),
            "| hard shield threshold: 0.88",
        )
        print(
            "fusion stats:",
            json.dumps(
                stats,
                sort_keys=True,
            ),
        )

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joined.to_csv(
        args.output,
        index=False,
    )

    print()
    print(
        "Wrote:",
        args.output,
    )


if __name__ == "__main__":
    main()
