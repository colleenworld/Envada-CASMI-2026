from __future__ import annotations

import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem

from casmi26.pubchem.inference_asset import (
    load_pubchem_candidates_from_assets,
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

PUBCHEM_EXPANSION_ASSET_ROOT = (
    PROJECT_ROOT
    / "data/processed/pubchem_formula_index_runs/"
    "generated-test-top25"
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


def load_excluded_structure_keys() -> set[str]:
    path = os.environ.get(
        "CASMI26_EXCLUDED_STRUCTURE_KEYS_PATH",
        "",
    ).strip()

    if not path:
        return set()

    frame = pd.read_parquet(
        Path(path),
        columns=["inchikey14"],
    )

    return set(
        frame["inchikey14"]
        .dropna()
        .astype(str)
    )


def main() -> None:
    excluded_structure_keys = load_excluded_structure_keys()

    if excluded_structure_keys:
        print(
            "Excluded reference structures:",
            f"{len(excluded_structure_keys):,}",
        )

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

    # Generated hypotheses are already ranked by the validated
    # training-derived formula prior. Preserve that order when available;
    # otherwise retain the legacy mass-error ranking.
    if "formula_rank" in positive.columns:
        positive = (
            positive
            .sort_values(
                [
                    "molecule_id",
                    "formula_rank",
                    "candidate_formula",
                ],
                ascending=[
                    True,
                    True,
                    True,
                ],
                kind="stable",
            )
            .reset_index(
                drop=True
            )
        )
    else:
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
            .reset_index(
                drop=True
            )
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

    pubchem_asset_roots = [
        PUBCHEM_ASSET_ROOT,
    ]

    if (
        PUBCHEM_EXPANSION_ASSET_ROOT
        / "formula_buckets.parquet"
    ).exists():
        pubchem_asset_roots.append(
            PUBCHEM_EXPANSION_ASSET_ROOT
        )

    print(
        "PubChem asset roots:"
    )

    for asset_root in pubchem_asset_roots:
        print(
            "  ",
            asset_root,
        )

    pubchem = (
        load_pubchem_candidates_from_assets(
            pubchem_asset_roots,
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

    if excluded_structure_keys:
        structures_all = structures_all.loc[
            ~structures_all["inchikey14"]
            .astype(str)
            .isin(excluded_structure_keys)
        ].copy()

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

    if excluded_structure_keys:
        mode_mask &= ~(
            metadata["inchikey14"]
            .astype(str)
            .isin(excluded_structure_keys)
            .to_numpy()
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

        raw_neighbors = neighbors_by_source[
            source_id
        ]

        scores, usable_neighbors = (
            score_candidates(
                candidate_fps,
                raw_neighbors,
                training_fps,
            )
        )

        valid[
            "neighbor_prior_score"
        ] = scores

        usable = [
            (
                key,
                max(float(spectral_score), 0.0),
                training_fps[key],
            )
            for key, spectral_score in raw_neighbors
            if key in training_fps
        ]

        row_count = len(valid)

        tanimoto_max = np.full(
            row_count,
            np.nan,
            dtype=np.float32,
        )
        tanimoto_mean = np.full(
            row_count,
            np.nan,
            dtype=np.float32,
        )
        tanimoto_top3_mean = np.full(
            row_count,
            np.nan,
            dtype=np.float32,
        )
        tanimoto_top5_mean = np.full(
            row_count,
            np.nan,
            dtype=np.float32,
        )
        weighted_mean = np.full(
            row_count,
            np.nan,
            dtype=np.float32,
        )
        weighted_top3_mean = np.full(
            row_count,
            np.nan,
            dtype=np.float32,
        )
        count_ge_03 = np.zeros(
            row_count,
            dtype=np.int16,
        )
        count_ge_05 = np.zeros(
            row_count,
            dtype=np.int16,
        )
        count_ge_07 = np.zeros(
            row_count,
            dtype=np.int16,
        )
        best_neighbor_spectral_score = np.full(
            row_count,
            np.nan,
            dtype=np.float32,
        )

        if usable and candidate_fps:
            similarities = np.empty(
                (
                    len(usable),
                    len(candidate_fps),
                ),
                dtype=np.float32,
            )

            weights = np.asarray(
                [
                    spectral_score
                    for _, spectral_score, _
                    in usable
                ],
                dtype=np.float32,
            )

            for neighbor_index, (
                _,
                _,
                neighbor_fp,
            ) in enumerate(usable):
                similarities[
                    neighbor_index
                ] = np.asarray(
                    DataStructs.BulkTanimotoSimilarity(
                        neighbor_fp,
                        candidate_fps,
                    ),
                    dtype=np.float32,
                )

            sorted_sim = np.sort(
                similarities,
                axis=0,
            )[::-1]

            weighted = (
                similarities
                * weights[:, None]
            )

            sorted_weighted = np.sort(
                weighted,
                axis=0,
            )[::-1]

            max_neighbor_idx = np.argmax(
                similarities,
                axis=0,
            )

            weight_sum = float(
                weights.sum()
            )

            for local_column in range(
                len(candidate_fps)
            ):
                values = similarities[
                    :,
                    local_column,
                ]

                tanimoto_max[
                    local_column
                ] = float(
                    values.max()
                )

                tanimoto_mean[
                    local_column
                ] = float(
                    values.mean()
                )

                tanimoto_top3_mean[
                    local_column
                ] = float(
                    sorted_sim[
                        :min(3, len(usable)),
                        local_column,
                    ].mean()
                )

                tanimoto_top5_mean[
                    local_column
                ] = float(
                    sorted_sim[
                        :min(5, len(usable)),
                        local_column,
                    ].mean()
                )

                if weight_sum > 0:
                    weighted_mean[
                        local_column
                    ] = float(
                        weighted[
                            :,
                            local_column,
                        ].sum()
                        / weight_sum
                    )

                weighted_top3_mean[
                    local_column
                ] = float(
                    sorted_weighted[
                        :min(3, len(usable)),
                        local_column,
                    ].mean()
                )

                count_ge_03[
                    local_column
                ] = int(
                    np.sum(
                        values >= 0.3
                    )
                )

                count_ge_05[
                    local_column
                ] = int(
                    np.sum(
                        values >= 0.5
                    )
                )

                count_ge_07[
                    local_column
                ] = int(
                    np.sum(
                        values >= 0.7
                    )
                )

                best_neighbor_spectral_score[
                    local_column
                ] = float(
                    weights[
                        max_neighbor_idx[
                            local_column
                        ]
                    ]
                )

        valid[
            "neighbor_tanimoto_max"
        ] = tanimoto_max
        valid[
            "neighbor_tanimoto_mean"
        ] = tanimoto_mean
        valid[
            "neighbor_tanimoto_top3_mean"
        ] = tanimoto_top3_mean
        valid[
            "neighbor_tanimoto_top5_mean"
        ] = tanimoto_top5_mean
        valid[
            "neighbor_weighted_mean"
        ] = weighted_mean
        valid[
            "neighbor_weighted_top3_mean"
        ] = weighted_top3_mean
        valid[
            "neighbor_count_ge_03"
        ] = count_ge_03
        valid[
            "neighbor_count_ge_05"
        ] = count_ge_05
        valid[
            "neighbor_count_ge_07"
        ] = count_ge_07
        valid[
            "best_structural_neighbor_spectral_score"
        ] = best_neighbor_spectral_score
        valid[
            "neighbor_feature_count"
        ] = usable_neighbors

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
