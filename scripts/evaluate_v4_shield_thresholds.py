from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger


RDLogger.DisableLog("rdApp.*")

BANDS = [
    (-np.inf, 0.80, "<0.80"),
    (0.80, 0.85, "0.80-<0.85"),
    (0.85, 0.88, "0.85-<0.88"),
    (0.88, 0.90, "0.88-<0.90"),
    (0.90, 0.95, "0.90-<0.95"),
    (0.95, np.inf, ">=0.95"),
]

DEFAULT_THRESHOLDS = [
    None,
    0.80,
    0.85,
    0.88,
    0.90,
    0.95,
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate v4 library-shield counterfactuals using per-molecule "
            "pre-shield rankings recorded in run_manifest.json."
        )
    )
    parser.add_argument(
        "--run-manifest",
        type=Path,
        required=True,
        help="run_manifest.json from an instrumented v4 validation run.",
    )
    parser.add_argument(
        "--truth-file",
        type=Path,
        required=True,
        help="CSV or Parquet containing molecule IDs and truth structures.",
    )
    parser.add_argument(
        "--molecule-id-column",
        default=None,
        help="Truth-table molecule ID column. Auto-detected when omitted.",
    )
    parser.add_argument(
        "--truth-smiles-column",
        default=None,
        help="Truth-table SMILES column. Auto-detected when omitted.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "data/processed/diagnostics/v4_shield_threshold_evaluation.csv"
        ),
    )
    parser.add_argument(
        "--per-molecule-output",
        type=Path,
        default=Path(
            "data/processed/diagnostics/v4_shield_threshold_per_molecule.csv"
        ),
    )
    return parser.parse_args()


def read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()

    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)

    if suffix in {".csv", ".txt"}:
        return pd.read_csv(path)

    raise ValueError(
        f"Unsupported truth-file format: {path.suffix}"
    )


def detect_column(
    frame: pd.DataFrame,
    explicit: str | None,
    candidates: list[str],
    label: str,
) -> str:
    if explicit is not None:
        if explicit not in frame.columns:
            raise ValueError(
                f"{label} column {explicit!r} is not present. "
                f"Columns: {list(frame.columns)}"
            )
        return explicit

    for column in candidates:
        if column in frame.columns:
            return column

    raise ValueError(
        f"Could not auto-detect {label} column. "
        f"Pass it explicitly. Columns: {list(frame.columns)}"
    )


def structure_key(smiles: object) -> str | None:
    text = str(smiles or "").strip()

    if not text or text.lower() == "nan":
        return None

    try:
        mol = Chem.MolFromSmiles(
            text
        )
    except Exception:
        return None

    if mol is None:
        return None

    try:
        return Chem.MolToInchiKey(
            mol
        )[:14]
    except Exception:
        return None


def ranking_keys(
    ranking: object,
) -> list[str]:
    if not isinstance(
        ranking,
        list,
    ):
        return []

    out: list[str] = []
    seen: set[str] = set()

    for smiles in ranking:
        key = structure_key(
            smiles
        )

        if (
            key is None
            or key in seen
        ):
            continue

        seen.add(
            key
        )
        out.append(
            key
        )

    return out


def truth_rank(
    ranking: list[str],
    truth_key: str | None,
) -> int | None:
    if truth_key is None:
        return None

    try:
        return (
            ranking.index(
                truth_key
            )
            + 1
        )
    except ValueError:
        return None


def rr25(
    rank: int | None,
) -> float:
    if (
        rank is None
        or rank < 1
        or rank > 25
    ):
        return 0.0

    return 1.0 / rank


def library_band(
    lib_max: float,
) -> str:
    for low, high, label in BANDS:
        if low <= lib_max < high:
            return label

    raise AssertionError(
        lib_max
    )


def apply_threshold(
    pre_shield: list[str],
    final_shielded: list[str],
    lib_max: float,
    threshold: float | None,
) -> list[str]:
    if threshold is None:
        return pre_shield

    if lib_max >= threshold:
        return final_shielded

    return pre_shield


def threshold_label(
    threshold: float | None,
) -> str:
    if threshold is None:
        return "unshielded"

    return f"shield>={threshold:.2f}"


def main() -> None:
    args = parse_args()

    if not args.run_manifest.exists():
        raise FileNotFoundError(
            args.run_manifest
        )

    if not args.truth_file.exists():
        raise FileNotFoundError(
            args.truth_file
        )

    manifest = json.loads(
        args.run_manifest.read_text()
    )

    stats = manifest.get(
        "stats",
        {}
    )

    molecule_stats = stats.get(
        "molecules",
        {}
    )

    if not molecule_stats:
        raise ValueError(
            "run_manifest.json has no stats.molecules diagnostics. "
            "Run the instrumented v4 kernel first."
        )

    truth = read_table(
        args.truth_file
    )

    molecule_column = detect_column(
        truth,
        args.molecule_id_column,
        [
            "molecule_id",
            "query_id",
            "id",
        ],
        "molecule ID",
    )

    smiles_column = detect_column(
        truth,
        args.truth_smiles_column,
        [
            "normalized_smiles",
            "smiles",
            "canonical_smiles",
            "truth_smiles",
        ],
        "truth SMILES",
    )

    truth = (
        truth[
            [
                molecule_column,
                smiles_column,
            ]
        ]
        .dropna(
            subset=[
                molecule_column,
                smiles_column,
            ]
        )
        .copy()
    )

    truth[
        molecule_column
    ] = (
        truth[
            molecule_column
        ]
        .astype(str)
    )

    truth[
        "truth_key"
    ] = truth[
        smiles_column
    ].map(
        structure_key
    )

    truth_by_molecule = (
        truth.dropna(
            subset=[
                "truth_key"
            ]
        )
        .drop_duplicates(
            subset=[
                molecule_column
            ]
        )
        .set_index(
            molecule_column
        )[
            "truth_key"
        ]
        .to_dict()
    )

    rows = []

    for molecule_id, info in molecule_stats.items():
        molecule_id = str(
            molecule_id
        )

        truth_key = truth_by_molecule.get(
            molecule_id
        )

        if truth_key is None:
            continue

        pre = ranking_keys(
            info.get(
                "pre_shield_top25",
                [],
            )
        )

        final = ranking_keys(
            info.get(
                "final_top25",
                [],
            )
        )

        lib_max = float(
            info.get(
                "lib_max",
                0.0,
            )
        )

        base = {
            "molecule_id":
                molecule_id,
            "truth_key":
                truth_key,
            "lib_max":
                lib_max,
            "library_band":
                library_band(
                    lib_max
                ),
            "recorded_shielded":
                bool(
                    info.get(
                        "shielded",
                        False,
                    )
                ),
            "pre_shield_rank":
                truth_rank(
                    pre,
                    truth_key,
                ),
            "recorded_final_rank":
                truth_rank(
                    final,
                    truth_key,
                ),
        }

        for threshold in DEFAULT_THRESHOLDS:
            ranking = apply_threshold(
                pre,
                final,
                lib_max,
                threshold,
            )

            rank = truth_rank(
                ranking,
                truth_key,
            )

            label = threshold_label(
                threshold
            )

            base[
                f"{label}_rank"
            ] = rank

        rows.append(
            base
        )

    per_molecule = pd.DataFrame(
        rows
    )

    if per_molecule.empty:
        raise ValueError(
            "No truth molecule IDs matched stats.molecules. "
            "Check the validation truth file and molecule ID column."
        )

    summary_rows = []

    scopes = [
        (
            "all",
            per_molecule,
        ),
    ]

    for label, group in per_molecule.groupby(
        "library_band",
        sort=False,
    ):
        scopes.append(
            (
                str(label),
                group,
            )
        )

    for scope, frame in scopes:
        for threshold in DEFAULT_THRESHOLDS:
            label = threshold_label(
                threshold
            )

            ranks = frame[
                f"{label}_rank"
            ]

            summary_rows.append(
                {
                    "scope":
                        scope,
                    "threshold":
                        label,
                    "queries":
                        len(
                            frame
                        ),
                    "mrr25":
                        ranks.map(
                            lambda x: rr25(
                                None
                                if pd.isna(x)
                                else int(x)
                            )
                        ).mean(),
                    "top1":
                        ranks.eq(
                            1
                        ).mean(),
                    "top5":
                        ranks.between(
                            1,
                            5,
                        ).mean(),
                    "top10":
                        ranks.between(
                            1,
                            10,
                        ).mean(),
                    "top25":
                        ranks.between(
                            1,
                            25,
                        ).mean(),
                    "truth_returned":
                        ranks.notna().mean(),
                }
            )

    summary = pd.DataFrame(
        summary_rows
    )

    print()
    print(
        "V4 shield-threshold evaluation"
    )
    print(
        "=" * 88
    )

    overall = summary.loc[
        summary[
            "scope"
        ]
        == "all"
    ].copy()

    print(
        overall.to_string(
            index=False,
            formatters={
                "mrr25":
                    lambda x: f"{x:.4f}",
                "top1":
                    lambda x: f"{x:.2%}",
                "top5":
                    lambda x: f"{x:.2%}",
                "top10":
                    lambda x: f"{x:.2%}",
                "top25":
                    lambda x: f"{x:.2%}",
                "truth_returned":
                    lambda x: f"{x:.2%}",
            },
        )
    )

    print()
    print(
        "Queries by library-confidence band"
    )
    print(
        "-"
        * 88
    )
    print(
        per_molecule[
            "library_band"
        ]
        .value_counts()
        .reindex(
            [
                label
                for _,
                _,
                label
                in BANDS
            ],
            fill_value=0,
        )
        .to_string()
    )

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    args.per_molecule_output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary.to_csv(
        args.output,
        index=False,
    )

    per_molecule.to_csv(
        args.per_molecule_output,
        index=False,
    )

    print()
    print(
        "Wrote:",
        args.output,
    )
    print(
        "Wrote:",
        args.per_molecule_output,
    )


if __name__ == "__main__":
    main()
