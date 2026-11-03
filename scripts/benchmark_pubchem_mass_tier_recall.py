from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import inchi

from casmi26.chemistry.inference import (
    neutral_mass_from_adduct,
    select_query_spectra,
)
from casmi26.pubchem.mass_tier import (
    PubChemMassTier,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_ASSET_ROOT = (
    PROJECT_ROOT
    / "data/external/casmi26-public-assets"
)

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
      "pubchem_mass_tier_recall.parquet"
)

SUMMARY_PATH = (
    PROJECT_ROOT
    / "data/processed/blind_pseudotest/"
      "pubchem_mass_tier_recall_summary.csv"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure truth-structure recall of the public "
            "105.9M PubChem mass-sorted tier on the blind pseudo-test."
        )
    )

    parser.add_argument(
        "--asset-root",
        type=Path,
        default=DEFAULT_ASSET_ROOT,
    )

    parser.add_argument(
        "--ppm",
        type=float,
        default=10.0,
    )

    parser.add_argument(
        "--max-molecules",
        type=int,
        default=None,
    )

    return parser.parse_args()


def score_key(
    smiles: str,
) -> str | None:
    mol = Chem.MolFromSmiles(
        smiles
    )

    if mol is None:
        return None

    try:
        key = inchi.MolToInchiKey(
            mol
        )
    except Exception:
        return None

    if not key:
        return None

    return str(
        key
    ).split(
        "-"
    )[0]


def representative_neutral_mass(
    group: pd.DataFrame,
) -> float | None:
    selected = select_query_spectra(
        group
    )

    if selected.empty:
        return None

    neutral = np.asarray(
        [
            neutral_mass_from_adduct(
                precursor_mz,
                adduct,
            )
            for precursor_mz, adduct
            in zip(
                selected[
                    "precursor_mz"
                ],
                selected[
                    "adduct"
                ],
                strict=True,
            )
        ],
        dtype=np.float64,
    )

    neutral = neutral[
        np.isfinite(
            neutral
        )
    ]

    if neutral.size == 0:
        return None

    return float(
        np.median(
            neutral
        )
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

    tier = PubChemMassTier.load(
        args.asset_root
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

    test["molecule_id"] = (
        test["molecule_id"]
        .astype(str)
    )

    truth["molecule_id"] = (
        truth["molecule_id"]
        .astype(str)
    )

    truth["inchikey14"] = (
        truth["inchikey14"]
        .astype(str)
    )

    molecule_ids = (
        truth[
            "molecule_id"
        ]
        .tolist()
    )

    if (
        args.max_molecules
        is not None
    ):
        molecule_ids = (
            molecule_ids[
                :args.max_molecules
            ]
        )

    truth_lookup = dict(
        zip(
            truth[
                "molecule_id"
            ],
            truth[
                "inchikey14"
            ],
            strict=True,
        )
    )

    test_by_molecule = {
        str(molecule_id):
            group.copy()
        for molecule_id, group
        in test.groupby(
            "molecule_id",
            sort=False,
        )
    }

    rows = []

    for number, molecule_id in enumerate(
        molecule_ids,
        start=1,
    ):
        group = test_by_molecule.get(
            molecule_id
        )

        if group is None:
            rows.append(
                {
                    "molecule_id":
                        molecule_id,
                    "neutral_mass":
                        np.nan,
                    "window_size":
                        0,
                    "truth_present":
                        False,
                    "decoded_candidates":
                        0,
                    "valid_structure_keys":
                        0,
                }
            )
            continue

        neutral_mass = representative_neutral_mass(
            group
        )

        if neutral_mass is None:
            rows.append(
                {
                    "molecule_id":
                        molecule_id,
                    "neutral_mass":
                        np.nan,
                    "window_size":
                        0,
                    "truth_present":
                        False,
                    "decoded_candidates":
                        0,
                    "valid_structure_keys":
                        0,
                }
            )
            continue

        smiles = tier.smiles_window(
            neutral_mass,
            ppm=args.ppm,
        )

        truth_key = truth_lookup[
            molecule_id
        ]

        found = False
        valid_keys = 0

        for candidate in smiles:
            key = score_key(
                candidate
            )

            if key is None:
                continue

            valid_keys += 1

            if key == truth_key:
                found = True
                break

        rows.append(
            {
                "molecule_id":
                    molecule_id,
                "neutral_mass":
                    neutral_mass,
                "window_size":
                    len(
                        smiles
                    ),
                "truth_present":
                    found,
                "decoded_candidates":
                    len(
                        smiles
                    ),
                "valid_structure_keys":
                    valid_keys,
            }
        )

        if (
            number % 25 == 0
            or number
            == len(
                molecule_ids
            )
        ):
            print(
                f"{number:,}/{len(molecule_ids):,} "
                f"| recall so far "
                f"{sum(row['truth_present'] for row in rows) / len(rows):.2%}"
            )

    result = pd.DataFrame(
        rows
    )

    summary = pd.DataFrame(
        [
            {
                "ppm":
                    args.ppm,
                "queries":
                    len(result),
                "truth_hits":
                    int(
                        result[
                            "truth_present"
                        ].sum()
                    ),
                "truth_recall":
                    float(
                        result[
                            "truth_present"
                        ].mean()
                    ),
                "window_mean":
                    float(
                        result[
                            "window_size"
                        ].mean()
                    ),
                "window_median":
                    float(
                        result[
                            "window_size"
                        ].median()
                    ),
                "window_p90":
                    float(
                        result[
                            "window_size"
                        ].quantile(
                            0.90
                        )
                    ),
                "window_p95":
                    float(
                        result[
                            "window_size"
                        ].quantile(
                            0.95
                        )
                    ),
                "window_p99":
                    float(
                        result[
                            "window_size"
                        ].quantile(
                            0.99
                        )
                    ),
                "window_max":
                    int(
                        result[
                            "window_size"
                        ].max()
                    ),
            }
        ]
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
        "PubChem mass-tier blind candidate recall"
    )
    print(
        "=" * 72
    )

    print(
        summary.to_string(
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
