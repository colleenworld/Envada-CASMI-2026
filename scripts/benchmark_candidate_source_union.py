from __future__ import annotations

from pathlib import Path
import pickle

import numpy as np
import pandas as pd

from casmi26.pubchem.inference_asset import (
    available_formulas,
    load_pubchem_candidates_for_formulas,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent

BLIND_ROOT = (
    PROJECT_ROOT
    / "data/processed/blind_pseudotest"
)

TRUTH_PATH = (
    BLIND_ROOT
    / "truth.parquet"
)

MASS_TIER_RESULT_PATH = (
    BLIND_ROOT
    / "pubchem_mass_tier_recall.parquet"
)

TRAINING_STRUCTURES_PATH = (
    PROJECT_ROOT
    / "data/processed/structure_index/training_structures.parquet"
)

COCONUT_PATH = (
    PROJECT_ROOT
    / "data/external/coconut/coconut_approved.parquet"
)

FORMULA_PUBCHEM_ROOT = (
    PROJECT_ROOT
    / "data/processed/pubchem_formula_index"
)

PUBLIC_V2_POOL_PATH = (
    PROJECT_ROOT
    / "data/external/casmi26-public-assets/"
      "casmi26-v2-pool/pool_meta.parquet"
)

BIO_ROOT = (
    PROJECT_ROOT
    / "data/external/casmi26-public-assets/"
      "chebi-lipidmaps-casmi26"
)

PUBCHEMLITE_ROOT = (
    PROJECT_ROOT
    / "data/external/casmi26-public-assets/"
      "casmi26-pubchemlite-cc0-candidate-fingerprints"
)

OUTPUT_PATH = (
    BLIND_ROOT
    / "candidate_source_union_recall.parquet"
)

SUMMARY_PATH = (
    BLIND_ROOT
    / "candidate_source_union_recall_summary.csv"
)


def pair_set(
    frame: pd.DataFrame,
    *,
    formula_column: str,
    key_column: str,
) -> set[tuple[str, str]]:
    return set(
        zip(
            frame[
                formula_column
            ].dropna().astype(str),
            frame.loc[
                frame[
                    formula_column
                ].notna(),
                key_column,
            ].astype(str),
            strict=True,
        )
    )



def load_mass_key_bank(
    root: Path,
    *,
    mass_name: str,
    meta_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    mass = np.load(
        root / mass_name,
        mmap_mode="r",
    )

    with (root / meta_name).open("rb") as handle:
        meta = pickle.load(handle)

    keys = np.asarray(
        meta["keys"],
        dtype=str,
    )

    if len(mass) != len(keys):
        raise ValueError(
            f"Mass/key length mismatch for {root}: "
            f"{len(mass):,} vs {len(keys):,}"
        )

    return mass, keys


def truth_hits_by_mass_window(
    truth: pd.DataFrame,
    *,
    mass: np.ndarray,
    keys: np.ndarray,
    ppm: float = 10.0,
) -> set[str]:
    hits: set[str] = set()

    for row in truth.itertuples(
        index=False
    ):
        neutral_mass = float(
            row.exact_mass
        )

        tolerance = (
            neutral_mass
            * ppm
            / 1_000_000.0
        )

        left = int(
            np.searchsorted(
                mass,
                neutral_mass - tolerance,
                side="left",
            )
        )

        right = int(
            np.searchsorted(
                mass,
                neutral_mass + tolerance,
                side="right",
            )
        )

        if right <= left:
            continue

        truth_key = str(
            row.inchikey14
        )

        if truth_key in set(
            keys[
                left:right
            ].tolist()
        ):
            hits.add(
                str(
                    row.molecule_id
                )
            )

    return hits

def main() -> None:
    for path in [
        TRUTH_PATH,
        MASS_TIER_RESULT_PATH,
        TRAINING_STRUCTURES_PATH,
        COCONUT_PATH,
        PUBLIC_V2_POOL_PATH,
        BIO_ROOT / "bio_mass.npy",
        BIO_ROOT / "bio_meta.pkl",
        PUBCHEMLITE_ROOT / "pubchemlite_mass.npy",
        PUBCHEMLITE_ROOT / "pubchemlite_meta.pkl",
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"Missing required artifact: {path}"
            )

    truth = pd.read_parquet(
        TRUTH_PATH
    ).copy()

    truth["molecule_id"] = (
        truth["molecule_id"]
        .astype(str)
    )
    truth["molecular_formula"] = (
        truth["molecular_formula"]
        .astype(str)
    )
    truth["inchikey14"] = (
        truth["inchikey14"]
        .astype(str)
    )

    mass = pd.read_parquet(
        MASS_TIER_RESULT_PATH,
        columns=[
            "molecule_id",
            "truth_present",
        ],
    ).copy()

    mass["molecule_id"] = (
        mass["molecule_id"]
        .astype(str)
    )

    result = truth.merge(
        mass,
        on="molecule_id",
        how="left",
        validate="one_to_one",
    )

    result[
        "pubchem_mass_tier"
    ] = (
        result[
            "truth_present"
        ]
        .fillna(False)
        .astype(bool)
    )

    training = pd.read_parquet(
        TRAINING_STRUCTURES_PATH,
        columns=[
            "molecular_formula",
            "inchikey14",
        ],
    )

    training_pairs = set(
        zip(
            training[
                "molecular_formula"
            ].dropna().astype(str),
            training.loc[
                training[
                    "molecular_formula"
                ].notna(),
                "inchikey14",
            ].astype(str),
            strict=True,
        )
    )

    coconut = pd.read_parquet(
        COCONUT_PATH,
        columns=[
            "molecular_formula",
            "inchikey14",
        ],
    )

    coconut_pairs = set(
        zip(
            coconut[
                "molecular_formula"
            ].dropna().astype(str),
            coconut.loc[
                coconut[
                    "molecular_formula"
                ].notna(),
                "inchikey14",
            ].astype(str),
            strict=True,
        )
    )

    truth_pairs = list(
        zip(
            result[
                "molecular_formula"
            ],
            result[
                "inchikey14"
            ],
            strict=True,
        )
    )

    result[
        "training_structure"
    ] = [
        pair in training_pairs
        for pair in truth_pairs
    ]

    result[
        "coconut"
    ] = [
        pair in coconut_pairs
        for pair in truth_pairs
    ]

    old_pubchem_pairs: set[
        tuple[str, str]
    ] = set()

    if (
        FORMULA_PUBCHEM_ROOT
        / "formula_buckets.parquet"
    ).exists():
        available = available_formulas(
            FORMULA_PUBCHEM_ROOT
        )

        truth_formulas = set(
            result[
                "molecular_formula"
            ]
        )

        covered_formulas = (
            truth_formulas
            & available
        )

        if covered_formulas:
            old_pubchem = (
                load_pubchem_candidates_for_formulas(
                    FORMULA_PUBCHEM_ROOT,
                    covered_formulas,
                )
            )

            old_pubchem_pairs = set(
                zip(
                    old_pubchem[
                        "molecular_formula"
                    ].astype(str),
                    old_pubchem[
                        "inchikey14"
                    ].astype(str),
                    strict=True,
                )
            )

    result[
        "old_formula_pubchem"
    ] = [
        pair in old_pubchem_pairs
        for pair in truth_pairs
    ]

    public_v2_pool = pd.read_parquet(
        PUBLIC_V2_POOL_PATH,
        columns=[
            "formula",
            "key",
            "src",
            "train_sid",
        ],
    )

    public_v2_all_pairs = set(
        zip(
            public_v2_pool[
                "formula"
            ].dropna().astype(str),
            public_v2_pool.loc[
                public_v2_pool[
                    "formula"
                ].notna(),
                "key",
            ].astype(str),
            strict=True,
        )
    )

    public_v2_external = public_v2_pool.loc[
        public_v2_pool[
            "src"
        ].astype(int)
        .isin([2, 3])
    ].copy()

    public_v2_external_pairs = set(
        zip(
            public_v2_external[
                "formula"
            ].dropna().astype(str),
            public_v2_external.loc[
                public_v2_external[
                    "formula"
                ].notna(),
                "key",
            ].astype(str),
            strict=True,
        )
    )

    result[
        "public_v2_pool_all"
    ] = [
        pair in public_v2_all_pairs
        for pair in truth_pairs
    ]

    result[
        "public_v2_pool_external_only"
    ] = [
        pair in public_v2_external_pairs
        for pair in truth_pairs
    ]

    if "exact_mass" not in truth.columns:
        raise ValueError(
            "Blind truth must contain exact_mass for mass-window bank recall"
        )

    bio_mass, bio_keys = load_mass_key_bank(
        BIO_ROOT,
        mass_name="bio_mass.npy",
        meta_name="bio_meta.pkl",
    )

    pubchemlite_mass, pubchemlite_keys = load_mass_key_bank(
        PUBCHEMLITE_ROOT,
        mass_name="pubchemlite_mass.npy",
        meta_name="pubchemlite_meta.pkl",
    )

    bio_hits = truth_hits_by_mass_window(
        truth,
        mass=bio_mass,
        keys=bio_keys,
        ppm=10.0,
    )

    pubchemlite_hits = truth_hits_by_mass_window(
        truth,
        mass=pubchemlite_mass,
        keys=pubchemlite_keys,
        ppm=10.0,
    )

    result[
        "chebi_lipidmaps"
    ] = (
        result[
            "molecule_id"
        ].isin(
            bio_hits
        )
    )

    result[
        "pubchemlite"
    ] = (
        result[
            "molecule_id"
        ].isin(
            pubchemlite_hits
        )
    )

    # The blind pseudo-test is sampled from the competition training set.
    # Public-v2 provenance uses src=1 for training-only, src=2 for COCONUT,
    # and src=3 for structures present in both. Training-only rows are
    # therefore excluded, while src=2/3 rows remain valid because the
    # structure exists independently in the external COCONUT pool.
    source_columns = [
        "pubchem_mass_tier",
        "training_structure",
        "coconut",
        "old_formula_pubchem",
        "public_v2_pool_external_only",
        "chebi_lipidmaps",
        "pubchemlite",
    ]

    result[
        "any_source"
    ] = (
        result[
            source_columns
        ]
        .any(
            axis=1
        )
    )

    result[
        "rescued_outside_mass_tier"
    ] = (
        ~result[
            "pubchem_mass_tier"
        ]
        & result[
            [
                "training_structure",
                "coconut",
                "old_formula_pubchem",
                "public_v2_pool_external_only",
                "chebi_lipidmaps",
                "pubchemlite",
            ]
        ]
        .any(
            axis=1
        )
    )

    rows = []

    report_columns = [
        "pubchem_mass_tier",
        "training_structure",
        "coconut",
        "old_formula_pubchem",
        "public_v2_pool_all",
        "public_v2_pool_external_only",
        "chebi_lipidmaps",
        "pubchemlite",
        "any_source",
        "rescued_outside_mass_tier",
    ]

    for column in report_columns:
        hits = int(
            result[
                column
            ].sum()
        )

        rows.append(
            {
                "source":
                    column,
                "queries":
                    len(result),
                "truth_hits":
                    hits,
                "truth_recall":
                    hits
                    / len(result),
            }
        )

    mass_misses = result.loc[
        ~result[
            "pubchem_mass_tier"
        ]
    ]

    rescued_misses = int(
        mass_misses[
            "rescued_outside_mass_tier"
        ].sum()
    )

    unresolved = int(
        (
            ~result[
                "any_source"
            ]
        ).sum()
    )

    summary = pd.DataFrame(
        rows
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
        "Blind candidate-source union recall"
    )
    print(
        "=" * 76
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
        "Mass-tier misses:",
        len(
            mass_misses
        ),
    )
    print(
        "Mass-tier misses rescued by existing sources:",
        rescued_misses,
    )
    print(
        "Still absent from every existing source:",
        unresolved,
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
