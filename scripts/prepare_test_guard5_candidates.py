from __future__ import annotations

import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

from casmi26.pubchem.inference_asset import (
    load_pubchem_candidates_for_formulas,
)
from casmi26.pubchem.neighbor_pruning import (
    score_candidates,
)
from casmi26.retrieval.index import (
    load_retrieval_index,
)
from casmi26.spectra.binning import (
    bin_spectrum,
)


PROJECT_ROOT = Path(
    os.environ.get(
        "CASMI26_PROJECT_ROOT",
        Path(__file__).resolve().parent.parent,
    )
)

TEST_PATH = (
    PROJECT_ROOT
    / "data/raw/test.parquet"
)

TEST_ROOT = (
    PROJECT_ROOT
    / "data/processed/test"
)

HYPOTHESES_PATH = (
    TEST_ROOT
    / "multispectrum_formula_hypotheses.parquet"
)

PUBCHEM_ASSET_ROOT = (
    PROJECT_ROOT
    / "data/processed/pubchem_formula_index"
)

RETRIEVAL_INDEX_DIR = Path(
    os.environ.get(
        "CASMI26_RETRIEVAL_INDEX_DIR",
        PROJECT_ROOT / "data/processed/retrieval_index",
    )
)

STRUCTURES_PATH = Path(
    os.environ.get(
        "CASMI26_STRUCTURES_PATH",
        PROJECT_ROOT / "data/processed/structure_index/structures.parquet",
    )
)

COCONUT_PATH = (
    PROJECT_ROOT
    / "data/external/coconut/coconut_approved.parquet"
)

OUTPUT_CANDIDATES_PATH = (
    TEST_ROOT
    / "guard5_neighbor_top500.parquet"
)

OUTPUT_PRIOR_PATH = (
    TEST_ROOT
    / "guard5_neighbor_top500_prior.parquet"
)

OUTPUT_FORM_TO_SMI_PATH = (
    TEST_ROOT
    / "guard5_full_form_to_smi.pkl"
)

SUPPORTED_ADDUCTS = {
    "[M+H]+",
    "[M+Na]+",
    "[M+NH4]+",
    "[M+K]+",
}

BIN_WIDTH = 1.0
MAX_MZ = 1500.0
SPECTRAL_NEIGHBORS = 25

RDLogger.DisableLog(
    "rdApp.*"
)


NEIGHBOR_SAVE_TOP_K = 500


def morgan_fp(
    smiles: str,
):
    mol = Chem.MolFromSmiles(
        str(smiles)
    )

    if mol is None:
        return None

    return AllChem.GetMorganFingerprintAsBitVect(
        mol,
        radius=2,
        nBits=4096,
    )


def top_structure_neighbors(
    query_vector: np.ndarray,
    ref_vectors,
    ref_keys: np.ndarray,
    count: int,
) -> list[tuple[str, float]]:
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
    hypotheses = pd.read_parquet(
        HYPOTHESES_PATH
    )

    test = (
        pd.read_parquet(
            TEST_PATH
        )
        .reset_index(
            drop=True
        )
        .copy()
    )

    test[
        "source_test_row_id"
    ] = np.arange(
        len(test),
        dtype=np.int64,
    )

    required_hypotheses = {
        "molecule_id",
        "source_test_row_id",
        "hypothesis_spec",
        "candidate_formula",
        "selected_adduct",
        "selected_ionization_mode",
    }

    missing = (
        required_hypotheses
        - set(hypotheses.columns)
    )

    if missing:
        raise ValueError(
            "Formula hypotheses are missing columns: "
            f"{sorted(missing)}"
        )

    positive = hypotheses.loc[
        hypotheses[
            "selected_adduct"
        ].isin(
            SUPPORTED_ADDUCTS
        )
        & (
            hypotheses[
                "selected_ionization_mode"
            ]
            .astype(str)
            == "positive"
        )
    ].copy()

    # Keep only the top three formula hypotheses per molecule, matching
    # the large validation result. Hypotheses are ranked by best absolute
    # mass error, with formula as a deterministic tie-breaker.
    positive = (
        positive
        .sort_values(
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
        .groupby(
            "molecule_id",
            sort=False,
        )
        .head(3)
        .reset_index(drop=True)
    )

    if positive.empty:
        raise RuntimeError(
            "No MIST-supported positive formula hypotheses were found"
        )

    if positive[
        "hypothesis_spec"
    ].duplicated().any():
        raise ValueError(
            "Duplicate hypothesis_spec values"
        )

    formulas = set(
        positive[
            "candidate_formula"
        ]
        .dropna()
        .astype(str)
    )

    print(
        "Supported positive hypotheses:",
        f"{len(positive):,}",
    )
    print(
        "Supported positive molecules:",
        f"{positive['molecule_id'].nunique():,}",
    )
    print(
        "Required PubChem formulas:",
        f"{len(formulas):,}",
    )

    print()
    print(
        "Loading formula-specific PubChem candidates..."
    )

    pubchem = (
        load_pubchem_candidates_for_formulas(
            PUBCHEM_ASSET_ROOT,
            formulas,
        )
    )

    pubchem = pubchem.rename(
        columns={
            "molecular_formula":
                "candidate_formula",
            "inchikey14":
                "candidate_inchikey14",
            "smiles":
                "candidate_smiles",
        }
    )

    pubchem[
        "candidate_source"
    ] = "pubchem"

    print(
        "PubChem candidate rows:",
        f"{len(pubchem):,}",
    )

    #
    # Preserve known training structures as preferred representations
    # when they overlap PubChem. They are also the structural anchors
    # used by the spectral-neighbor prior.
    #
    structures_all = pd.read_parquet(
        STRUCTURES_PATH,
        columns=[
            "inchikey14",
            "normalized_smiles",
            "molecular_formula",
        ],
    )

    training = structures_all.loc[
        structures_all[
            "molecular_formula"
        ]
        .astype(str)
        .isin(
            formulas
        )
    ].copy()

    training[
        "candidate_source"
    ] = "training"

    coconut = pd.read_parquet(
        COCONUT_PATH,
        columns=[
            "inchikey14",
            "smiles",
            "molecular_formula",
        ],
    )

    coconut = coconut.loc[
        coconut[
            "molecular_formula"
        ]
        .astype(str)
        .isin(
            formulas
        )
    ].copy()

    overlap = training[
        [
            "inchikey14",
            "molecular_formula",
        ]
    ].merge(
        coconut[
            [
                "inchikey14",
                "molecular_formula",
            ]
        ],
        on="inchikey14",
        how="inner",
        suffixes=(
            "_training",
            "_coconut",
        ),
    )

    conflicts = set(
        overlap.loc[
            overlap[
                "molecular_formula_training"
            ]
            != overlap[
                "molecular_formula_coconut"
            ],
            "inchikey14",
        ].astype(str)
    )

    if conflicts:
        training = training.loc[
            ~training[
                "inchikey14"
            ]
            .astype(str)
            .isin(
                conflicts
            )
        ].copy()

        coconut = coconut.loc[
            ~coconut[
                "inchikey14"
            ]
            .astype(str)
            .isin(
                conflicts
            )
        ].copy()

    known_training = training.rename(
        columns={
            "molecular_formula":
                "candidate_formula",
            "inchikey14":
                "candidate_inchikey14",
            "normalized_smiles":
                "candidate_smiles",
        }
    )[
        [
            "candidate_formula",
            "candidate_inchikey14",
            "candidate_smiles",
            "candidate_source",
        ]
    ]

    known_coconut = coconut.rename(
        columns={
            "molecular_formula":
                "candidate_formula",
            "inchikey14":
                "candidate_inchikey14",
            "smiles":
                "candidate_smiles",
        }
    )[
        [
            "candidate_formula",
            "candidate_inchikey14",
            "candidate_smiles",
        ]
    ]

    known_coconut[
        "candidate_source"
    ] = "coconut"

    catalog = pd.concat(
        [
            known_training,
            known_coconut,
            pubchem[
                [
                    "candidate_formula",
                    "candidate_inchikey14",
                    "candidate_smiles",
                    "candidate_source",
                ]
            ],
        ],
        ignore_index=True,
    )

    catalog[
        "source_priority"
    ] = (
        catalog[
            "candidate_source"
        ]
        .map(
            {
                "training": 0,
                "coconut": 1,
                "pubchem": 2,
            }
        )
    )

    catalog = (
        catalog
        .sort_values(
            [
                "candidate_formula",
                "candidate_inchikey14",
                "source_priority",
            ]
        )
        .drop_duplicates(
            subset=[
                "candidate_formula",
                "candidate_inchikey14",
            ],
            keep="first",
        )
        .drop(
            columns=[
                "source_priority",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    print(
        "Combined unique formula/structure candidates:",
        f"{len(catalog):,}",
    )

    print()
    print(
        "Loading production retrieval index..."
    )

    metadata, vectors = (
        load_retrieval_index(
            RETRIEVAL_INDEX_DIR
        )
    )

    mode_mask = (
        metadata[
            "ionization_mode"
        ]
        .astype(str)
        .to_numpy()
        == "positive"
    )

    mode_indices = np.flatnonzero(
        mode_mask
    )

    if not len(mode_indices):
        raise RuntimeError(
            "Retrieval index contains no positive references"
        )

    ref_vectors = vectors[
        mode_indices
    ]

    ref_keys = (
        metadata.iloc[
            mode_indices
        ][
            "inchikey14"
        ]
        .astype(str)
        .to_numpy()
    )

    source_lookup = (
        test.set_index(
            "source_test_row_id"
        )
    )

    source_ids = sorted(
        set(
            positive[
                "source_test_row_id"
            ]
            .astype(int)
        )
    )

    print(
        "Unique positive source spectra:",
        len(source_ids),
    )

    neighbors_by_source = {}
    neighbor_keys = set()

    for number, source_id in enumerate(
        source_ids,
        start=1,
    ):
        source = source_lookup.loc[
            source_id
        ]

        query_vector = bin_spectrum(
            source.ms2_mzs,
            source.ms2_normalized_intensities,
            max_mz=MAX_MZ,
            bin_width=BIN_WIDTH,
        )

        neighbors = top_structure_neighbors(
            query_vector,
            ref_vectors,
            ref_keys,
            SPECTRAL_NEIGHBORS,
        )

        neighbors_by_source[
            source_id
        ] = neighbors

        neighbor_keys.update(
            key
            for key, _
            in neighbors
        )

        if (
            number % 50 == 0
            or number == len(source_ids)
        ):
            print(
                f"  spectral neighbors "
                f"{number}/{len(source_ids)}"
            )

    print(
        "Unique neighbor structures:",
        f"{len(neighbor_keys):,}",
    )

    anchor_rows = structures_all.loc[
        structures_all[
            "inchikey14"
        ]
        .astype(str)
        .isin(
            neighbor_keys
        )
    ].copy()

    anchor_smiles = dict(
        zip(
            anchor_rows[
                "inchikey14"
            ].astype(str),
            anchor_rows[
                "normalized_smiles"
            ].astype(str),
        )
    )

    training_fps = {}

    for key, smiles in anchor_smiles.items():
        fp = morgan_fp(
            smiles
        )

        if fp is not None:
            training_fps[
                key
            ] = fp

    print(
        "Usable neighbor fingerprints:",
        f"{len(training_fps):,}",
    )

    catalog_groups = {
        formula: group.reset_index(
            drop=True
        )
        for formula, group
        in catalog.groupby(
            "candidate_formula",
            sort=False,
        )
    }

    fp_cache = {}

    selected_rows = []
    prior_rows = []

    for number, hypothesis in enumerate(
        positive.itertuples(
            index=False
        ),
        start=1,
    ):
        spec = str(
            hypothesis.hypothesis_spec
        )

        formula = str(
            hypothesis.candidate_formula
        )

        source_id = int(
            hypothesis.source_test_row_id
        )

        candidates = (
            catalog_groups.get(
                formula
            )
        )

        if (
            candidates is None
            or candidates.empty
        ):
            print(
                f"  WARNING {spec}: "
                f"no candidates for {formula}"
            )
            continue

        valid_indices = []
        candidate_fps = []

        for index, row in candidates.iterrows():
            key = str(
                row[
                    "candidate_inchikey14"
                ]
            )

            cache_key = (
                key,
                str(
                    row[
                        "candidate_smiles"
                    ]
                ),
            )

            if cache_key not in fp_cache:
                fp_cache[
                    cache_key
                ] = morgan_fp(
                    row[
                        "candidate_smiles"
                    ]
                )

            fp = fp_cache[
                cache_key
            ]

            if fp is None:
                continue

            valid_indices.append(
                index
            )

            candidate_fps.append(
                fp
            )

        valid = candidates.loc[
            valid_indices
        ].copy()

        if valid.empty:
            print(
                f"  WARNING {spec}: "
                "no RDKit-valid candidates"
            )
            continue

        scores, usable_neighbors = (
            score_candidates(
                candidate_fps,
                neighbors_by_source[
                    source_id
                ],
                training_fps,
            )
        )

        valid[
            "neighbor_prior_score"
        ] = scores

        valid = (
            valid
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

        valid[
            "prior_rank"
        ] = (
            np.arange(
                len(valid),
                dtype=np.int64,
            )
            + 1
        )

        limit = min(
            NEIGHBOR_SAVE_TOP_K,
            len(valid),
        )

        selected = valid.head(
            limit
        ).copy()

        selected[
            "molecule_id"
        ] = str(
            hypothesis.molecule_id
        )

        selected[
            "hypothesis_spec"
        ] = spec

        selected[
            "source_test_row_id"
        ] = source_id

        selected[
            "candidate_count_before_pruning"
        ] = len(
            valid
        )

        selected[
            "shortlist_size"
        ] = limit

        selected[
            "usable_neighbor_fingerprints"
        ] = usable_neighbors

        selected_rows.append(
            selected
        )

        prior = valid[
            [
                "candidate_formula",
                "candidate_inchikey14",
                "neighbor_prior_score",
                "prior_rank",
            ]
        ].copy()

        prior[
            "molecule_id"
        ] = str(
            hypothesis.molecule_id
        )

        prior[
            "hypothesis_spec"
        ] = spec

        prior_rows.append(
            prior
        )

        if (
            number % 50 == 0
            or number == len(positive)
        ):
            print(
                f"  pruned hypotheses "
                f"{number}/{len(positive)}"
            )

    if not selected_rows:
        raise RuntimeError(
            "No PubChem candidates survived pruning"
        )

    selected = pd.concat(
        selected_rows,
        ignore_index=True,
    )

    prior = pd.concat(
        prior_rows,
        ignore_index=True,
    )

    selected.to_parquet(
        OUTPUT_CANDIDATES_PATH,
        index=False,
        compression="zstd",
    )

    prior.to_parquet(
        OUTPUT_PRIOR_PATH,
        index=False,
        compression="zstd",
    )

    #
    # MIST retrieval databases are keyed by formula. If multiple
    # hypotheses share a formula, use the union of their selected
    # candidates. The downstream ranker will filter returned MIST
    # candidates back to each hypothesis's own shortlist.
    #
    # MIST must see the full formula-specific candidate universe.
    # The neighbor ranking is retained only as a top-500 rescue/prior list.
    formula_structures = (
        catalog[
            [
                "candidate_formula",
                "candidate_inchikey14",
                "candidate_smiles",
            ]
        ]
        .drop_duplicates(
            subset=[
                "candidate_formula",
                "candidate_inchikey14",
            ]
        )
    )

    form_to_smi = {}

    for formula, group in formula_structures.groupby(
        "candidate_formula",
        sort=False,
    ):
        form_to_smi[
            str(formula)
        ] = [
            {
                "smi":
                    str(
                        row.candidate_smiles
                    ),
                "ikey":
                    str(
                        row.candidate_inchikey14
                    ),
            }
            for row in group.itertuples(
                index=False
            )
        ]

    with OUTPUT_FORM_TO_SMI_PATH.open(
        "wb"
    ) as handle:
        pickle.dump(
            form_to_smi,
            handle,
        )

    counts = (
        selected[
            [
                "hypothesis_spec",
                "candidate_inchikey14",
            ]
        ]
        .groupby(
            "hypothesis_spec"
        )
        .size()
    )

    before = (
        selected[
            [
                "hypothesis_spec",
                "candidate_count_before_pruning",
            ]
        ]
        .drop_duplicates(
            subset=[
                "hypothesis_spec",
            ]
        )[
            "candidate_count_before_pruning"
        ]
    )

    print()
    print(
        "Guard-5 full-PubChem positive candidate set"
    )
    print(
        "===================================="
    )
    print(
        "Hypotheses with candidates:",
        f"{counts.size:,}",
    )
    print(
        "Candidate rows before pruning:",
        f"{int(before.sum()):,}",
    )
    print(
        "Neighbor top-500 rows:",
        f"{len(selected):,}",
    )
    print(
        "Full formula-union MIST entries:",
        f"{len(formula_structures):,}",
    )
    print(
        "Selected candidates/hypothesis mean:",
        f"{counts.mean():.1f}",
    )
    print(
        "Selected candidates/hypothesis median:",
        f"{counts.median():.1f}",
    )
    print(
        "Selected candidates/hypothesis max:",
        f"{counts.max():,}",
    )
    print()
    print(
        f"Wrote {OUTPUT_CANDIDATES_PATH}"
    )
    print(
        f"Wrote {OUTPUT_PRIOR_PATH}"
    )
    print(
        f"Wrote {OUTPUT_FORM_TO_SMI_PATH}"
    )


if __name__ == "__main__":
    main()
