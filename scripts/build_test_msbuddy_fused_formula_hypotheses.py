from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import msbuddy
from msbuddy import Msbuddy, MsbuddyConfig
from msbuddy.base import MetaFeature, Spectrum

from casmi26.chemistry.formula import monoisotopic_mass
from casmi26.chemistry.inference import (
    FORMULA_TOLERANCE_PPM,
    is_mist_processable,
    neutral_mass_from_adduct,
    ppm_error,
    select_query_spectra,
)


PROJECT_ROOT = Path(
    os.environ.get(
        "CASMI26_PROJECT_ROOT",
        Path(__file__).resolve().parent.parent,
    )
)

TEST_PATH = PROJECT_ROOT / "data/raw/test.parquet"

TEST_ROOT = PROJECT_ROOT / "data/processed/test"

PRODUCTION_HYPOTHESES_PATH = (
    TEST_ROOT
    / "multispectrum_formula_hypotheses.parquet"
)

MSBUDDY_OUTPUT_PATH = (
    TEST_ROOT
    / "msbuddy_formula_gate.parquet"
)

FUSED_HYPOTHESES_PATH = (
    TEST_ROOT
    / "multispectrum_formula_hypotheses_fused.parquet"
)

TOP_K_SOURCE = 5
TOP_K_FINAL = 3

MIST_SUPPORTED_ADDUCTS = {
    "[M+H]+",
    "[M+Na]+",
    "[M+NH4]+",
    "[M+K]+",
}

PRODUCTION_WEIGHT = 0.5
MSBUDDY_WEIGHT = 0.5
AGREEMENT_BONUS = 0.0
PRODUCTION_SUPPORT_WEIGHT = 0.0
MSBUDDY_SUPPORT_WEIGHT = 0.25

MSBUDDY_TIMEOUT_SECS = float(
    os.environ.get(
        "CASMI26_MSBUDDY_TIMEOUT_SECS",
        "120.0",
    )
)

MSBUDDY_DATA_FILENAMES = {
    "common_db": "common_db_v0.2.4.joblib",
    "formula_db": "formula_db_v0.2.4.joblib",
    "ml_model": "ml_v0.3.0.joblib",
}


def spectrum_from_row(
    row,
) -> Spectrum:
    mz = np.asarray(
        row.ms2_mzs,
        dtype=np.float64,
    )

    intensity = np.asarray(
        row.ms2_normalized_intensities,
        dtype=np.float64,
    )

    finite = (
        np.isfinite(mz)
        & np.isfinite(intensity)
        & (mz > 0)
        & (intensity > 0)
    )

    return Spectrum(
        mz_array=mz[finite],
        int_array=intensity[finite],
    )


def charge_for_mode(
    mode: str,
) -> int:
    if mode == "positive":
        return 1

    if mode == "negative":
        return -1

    raise ValueError(
        f"Unsupported ionization mode: {mode!r}"
    )


def prepare_production(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    required = {
        "molecule_id",
        "candidate_formula",
        "min_abs_mass_error_ppm",
        "spectrum_support_count",
        "spectrum_support_fraction",
    }

    missing = required - set(
        frame.columns
    )

    if missing:
        raise ValueError(
            "Production hypotheses missing columns: "
            f"{sorted(missing)}"
        )

    work = frame[
        [
            "molecule_id",
            "candidate_formula",
            "min_abs_mass_error_ppm",
            "spectrum_support_count",
            "spectrum_support_fraction",
        ]
    ].copy()

    work["molecule_id"] = (
        work["molecule_id"]
        .astype(str)
    )

    work["candidate_formula"] = (
        work["candidate_formula"]
        .astype(str)
    )

    work["min_abs_mass_error_ppm"] = pd.to_numeric(
        work["min_abs_mass_error_ppm"],
        errors="coerce",
    )

    work = (
        work.sort_values(
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
            na_position="last",
        )
        .drop_duplicates(
            subset=[
                "molecule_id",
                "candidate_formula",
            ],
            keep="first",
        )
        .reset_index(
            drop=True
        )
    )

    work["production_rank"] = (
        work.groupby(
            "molecule_id",
            sort=False,
        )
        .cumcount()
        + 1
    )

    return work.loc[
        work["production_rank"] <= TOP_K_SOURCE
    ].copy()


def run_msbuddy(
    selected: pd.DataFrame,
) -> pd.DataFrame:
    if selected.empty:
        raise RuntimeError(
            "No supported spectra are available for MSBuddy."
        )

    project_data_dir = (
        PROJECT_ROOT / "data/external/msbuddy"
    )
    package_data_dir = (
        Path(msbuddy.__file__).resolve().parent / "data"
    )
    data_dirs = (
        project_data_dir,
        package_data_dir,
    )
    data_files = None
    missing_files = []
    for data_dir in data_dirs:
        candidate_files = {
            key: data_dir / filename
            for key, filename in MSBUDDY_DATA_FILENAMES.items()
        }
        missing_files = [
            path
            for path in candidate_files.values()
            if not path.is_file()
        ]
        if not missing_files:
            data_files = {
                key: str(path)
                for key, path in candidate_files.items()
            }
            break
    if data_files is None:
        raise FileNotFoundError(
            "MSBuddy offline data files are missing:\n  "
            + "\n  ".join(str(path) for path in missing_files)
            + "\nProvide them under data/external/msbuddy or the "
            "installed msbuddy package data directory."
        )

    config = MsbuddyConfig(
        ms_instr=None,
        ppm=True,
        ms1_tol=10.0,
        ms2_tol=10.0,
        halogen=True,
        parallel=False,
        timeout_secs=MSBUDDY_TIMEOUT_SECS,
        data_files=data_files,
    )

    engine = Msbuddy(
        config
    )

    features = []
    feature_to_molecule: dict[str, str] = {}

    for row_number, row in enumerate(
        selected.itertuples(index=False)
    ):
        identifier = (
            f"casmi26_test_row_{int(row.source_test_row_id):09d}"
            f"__{row_number:06d}"
        )

        feature_to_molecule[identifier] = str(
            row.molecule_id
        )

        features.append(
            MetaFeature(
                identifier=identifier,
                mz=float(row.precursor_mz),
                charge=charge_for_mode(
                    str(row.ionization_mode)
                ),
                adduct=str(row.adduct),
                ms2=spectrum_from_row(row),
            )
        )

    print(
        "MSBuddy selected spectra:",
        f"{len(features):,}",
    )
    print(
        "MSBuddy selected molecules:",
        f"{selected['molecule_id'].nunique():,}",
    )
    print(
        "MSBuddy timeout per spectrum:",
        f"{MSBUDDY_TIMEOUT_SECS:g}s",
    )
    print(
        "Running MSBuddy formula annotation..."
    )

    engine.add_data(
        features
    )
    engine.annotate_formula()

    rows = []

    for item in engine.get_summary():
        identifier = str(
            item.get("identifier", "")
        )
        molecule_id = feature_to_molecule.get(
            identifier
        )

        if molecule_id is None:
            continue

        for rank in range(
            1,
            TOP_K_SOURCE + 1,
        ):
            formula = item.get(
                f"formula_rank_{rank}"
            )

            if formula is None:
                continue

            formula = str(formula).strip()

            if (
                not formula
                or formula.lower() == "nan"
            ):
                continue

            rows.append(
                {
                    "molecule_id": molecule_id,
                    "identifier": identifier,
                    "rank": rank,
                    "candidate_formula": formula,
                    "estimated_fdr": item.get(
                        "estimated_fdr"
                    ),
                }
            )

    result = pd.DataFrame(
        rows,
        columns=[
            "molecule_id",
            "identifier",
            "rank",
            "candidate_formula",
            "estimated_fdr",
        ],
    )

    if result.empty:
        raise RuntimeError(
            "MSBuddy produced no ranked formula annotations."
        )

    aggregate = (
        result.groupby(
            [
                "molecule_id",
                "candidate_formula",
            ],
            as_index=False,
        )
        .agg(
            best_rank=(
                "rank",
                "min",
            ),
            supporting_spectra=(
                "identifier",
                "nunique",
            ),
            min_estimated_fdr=(
                "estimated_fdr",
                "min",
            ),
        )
        .sort_values(
            [
                "molecule_id",
                "best_rank",
                "supporting_spectra",
                "candidate_formula",
            ],
            ascending=[
                True,
                True,
                False,
                True,
            ],
            kind="stable",
        )
        .reset_index(
            drop=True
        )
    )

    aggregate["molecule_rank"] = (
        aggregate.groupby(
            "molecule_id",
            sort=False,
        )
        .cumcount()
        + 1
    )

    return aggregate


def build_fusion_candidates(
    production: pd.DataFrame,
    msbuddy: pd.DataFrame,
) -> pd.DataFrame:
    production = production.copy()

    production["spectrum_support_fraction"] = pd.to_numeric(
        production["spectrum_support_fraction"],
        errors="coerce",
    ).fillna(0.0)

    msbuddy = (
        msbuddy.loc[
            msbuddy["molecule_rank"] <= TOP_K_SOURCE
        ]
        .rename(
            columns={
                "supporting_spectra": "msbuddy_supporting_spectra",
            }
        )
        .copy()
    )

    candidates = production.merge(
        msbuddy[
            [
                "molecule_id",
                "candidate_formula",
                "molecule_rank",
                "msbuddy_supporting_spectra",
            ]
        ],
        on=[
            "molecule_id",
            "candidate_formula",
        ],
        how="outer",
        validate="one_to_one",
    )

    candidates["in_production"] = (
        candidates["production_rank"].notna()
    )
    candidates["in_msbuddy"] = (
        candidates["molecule_rank"].notna()
    )
    candidates["agreement"] = (
        candidates["in_production"]
        & candidates["in_msbuddy"]
    )

    max_support = (
        candidates.groupby("molecule_id")[
            "msbuddy_supporting_spectra"
        ]
        .transform("max")
    )

    candidates["msbuddy_support_fraction"] = (
        pd.to_numeric(
            candidates["msbuddy_supporting_spectra"],
            errors="coerce",
        )
        .fillna(0.0)
        / max_support.fillna(1.0).clip(lower=1.0)
    )

    production_rr = np.where(
        candidates["production_rank"].notna(),
        1.0
        / candidates["production_rank"]
        .fillna(np.inf)
        .astype(float),
        0.0,
    )

    msbuddy_rr = np.where(
        candidates["molecule_rank"].notna(),
        1.0
        / candidates["molecule_rank"]
        .fillna(np.inf)
        .astype(float),
        0.0,
    )

    candidates["fusion_score"] = (
        PRODUCTION_WEIGHT * production_rr
        + MSBUDDY_WEIGHT * msbuddy_rr
        + AGREEMENT_BONUS
        * candidates["agreement"].astype(float)
        + PRODUCTION_SUPPORT_WEIGHT
        * candidates["spectrum_support_fraction"].fillna(0.0)
        + MSBUDDY_SUPPORT_WEIGHT
        * candidates["msbuddy_support_fraction"]
    )

    candidates = (
        candidates.sort_values(
            [
                "molecule_id",
                "fusion_score",
                "agreement",
                "production_rank",
                "molecule_rank",
                "candidate_formula",
            ],
            ascending=[
                True,
                False,
                False,
                True,
                True,
                True,
            ],
            kind="stable",
            na_position="last",
        )
        .reset_index(
            drop=True
        )
    )

    candidates["fusion_rank"] = (
        candidates.groupby(
            "molecule_id",
            sort=False,
        )
        .cumcount()
        + 1
    )

    return candidates


def prepare_selected_spectra(
    test: pd.DataFrame,
) -> pd.DataFrame:
    selected = select_query_spectra(
        test
    ).copy()

    selected["neutral_mass"] = [
        neutral_mass_from_adduct(
            precursor_mz,
            adduct,
        )
        for precursor_mz, adduct
        in zip(
            selected["precursor_mz"],
            selected["adduct"],
            strict=True,
        )
    ]

    processability = [
        is_mist_processable(
            mzs,
            intensities,
            precursor_mz,
        )
        for mzs, intensities, precursor_mz
        in zip(
            selected["ms2_mzs"],
            selected["ms2_normalized_intensities"],
            selected["precursor_mz"],
            strict=True,
        )
    ]

    selected["mist_processable"] = [
        item[0]
        for item in processability
    ]
    selected["mist_usable_peak_count"] = [
        item[1]
        for item in processability
    ]

    return selected


def reconstruct_hypothesis(
    molecule_id: str,
    formula: str,
    fusion_rank: int,
    selected: pd.DataFrame,
) -> dict | None:
    try:
        theoretical_mass = float(
            monoisotopic_mass(formula)
        )
    except ValueError:
        return None

    candidates = selected.loc[
        selected["ionization_mode"].astype(str).eq("positive")
        & selected["adduct"].isin(MIST_SUPPORTED_ADDUCTS)
    ].copy()

    if candidates.empty:
        return None

    candidates["selected_mass_error_ppm"] = [
        ppm_error(
            observed,
            theoretical_mass,
        )
        for observed in candidates["neutral_mass"]
    ]

    candidates["selected_abs_mass_error_ppm"] = (
        candidates["selected_mass_error_ppm"].abs()
    )

    within_tolerance = candidates.loc[
        candidates["selected_abs_mass_error_ppm"]
        <= FORMULA_TOLERANCE_PPM
    ].copy()

    if within_tolerance.empty:
        return None

    representative = (
        within_tolerance.sort_values(
            [
                "mist_processable",
                "selected_abs_mass_error_ppm",
                "mist_usable_peak_count",
                "source_test_row_id",
            ],
            ascending=[
                False,
                True,
                False,
                True,
            ],
            kind="stable",
        )
        .iloc[0]
    )

    support_count = int(
        within_tolerance["spectrum_id"].nunique()
    )
    selected_count = int(
        candidates["spectrum_id"].nunique()
    )

    return {
        "molecule_id": molecule_id,
        "source_test_row_id": int(
            representative["source_test_row_id"]
        ),
        "source_spectrum_id": str(
            representative["spectrum_id"]
        ),
        "selected_spectrum_id": str(
            representative["spectrum_id"]
        ),
        "selected_adduct": str(
            representative["adduct"]
        ),
        "selected_ionization_mode": str(
            representative["ionization_mode"]
        ),
        "selection_reason": str(
            representative["selection_reason"]
        ),
        "precursor_mz": float(
            representative["precursor_mz"]
        ),
        "selected_precursor_mz": float(
            representative["precursor_mz"]
        ),
        "neutral_mass": float(
            representative["neutral_mass"]
        ),
        "molecular_formula": formula,
        "candidate_formula": formula,
        "formula_monoisotopic_mass": theoretical_mass,
        "selected_mass_error_ppm": float(
            representative["selected_mass_error_ppm"]
        ),
        "selected_abs_mass_error_ppm": float(
            representative["selected_abs_mass_error_ppm"]
        ),
        "mist_processable": bool(
            representative["mist_processable"]
        ),
        "mist_usable_peak_count": int(
            representative["mist_usable_peak_count"]
        ),
        "spectrum_support_count": support_count,
        "selected_spectrum_count": selected_count,
        "spectrum_support_fraction": (
            support_count / selected_count
            if selected_count
            else 0.0
        ),
        "min_abs_mass_error_ppm": float(
            within_tolerance[
                "selected_abs_mass_error_ppm"
            ].min()
        ),
        "hypothesis_spec": (
            f"{molecule_id}__fused_{fusion_rank:04d}"
        ),
    }


def build_fused_hypotheses(
    test: pd.DataFrame,
    production_hypotheses: pd.DataFrame,
    fused_candidates: pd.DataFrame,
) -> pd.DataFrame:
    selected = prepare_selected_spectra(
        test
    )

    selected_by_molecule = {
        str(molecule_id): group.copy()
        for molecule_id, group
        in selected.groupby(
            "molecule_id",
            sort=False,
        )
    }

    production_lookup = {}

    for row in production_hypotheses.itertuples(
        index=False
    ):
        production_lookup[
            (
                str(row.molecule_id),
                str(row.candidate_formula),
            )
        ] = row._asdict()

    top = fused_candidates.loc[
        fused_candidates["fusion_rank"] <= TOP_K_FINAL
    ].copy()

    rows = []
    reused = 0
    reconstructed = 0
    skipped = 0

    for row in top.itertuples(
        index=False
    ):
        molecule_id = str(
            row.molecule_id
        )
        formula = str(
            row.candidate_formula
        )
        fusion_rank = int(
            row.fusion_rank
        )

        existing = production_lookup.get(
            (
                molecule_id,
                formula,
            )
        )

        if (
            existing is not None
            and str(existing.get("selected_ionization_mode", ""))
            == "positive"
            and str(existing.get("selected_adduct", ""))
            in MIST_SUPPORTED_ADDUCTS
        ):
            hypothesis = existing.copy()
            hypothesis["hypothesis_spec"] = (
                f"{molecule_id}__fused_{fusion_rank:04d}"
            )
            hypothesis["formula_rank"] = fusion_rank
            rows.append(hypothesis)
            reused += 1
            continue

        spectra = selected_by_molecule.get(
            molecule_id
        )

        if spectra is None or spectra.empty:
            skipped += 1
            continue

        hypothesis = reconstruct_hypothesis(
            molecule_id,
            formula,
            fusion_rank,
            spectra,
        )

        if hypothesis is None:
            skipped += 1
            continue

        hypothesis["formula_rank"] = fusion_rank
        rows.append(hypothesis)
        reconstructed += 1

    output = pd.DataFrame(
        rows
    )

    if output.empty:
        raise RuntimeError(
            "MSBuddy fusion produced no usable formula hypotheses."
        )

    output = (
        output.sort_values(
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

    output["formula_rank"] = (
        output.groupby(
            "molecule_id",
            sort=False,
        )
        .cumcount()
        + 1
    )

    if output["hypothesis_spec"].duplicated().any():
        raise RuntimeError(
            "Fused hypothesis_spec values are not unique."
        )

    expected_molecules = set(
        top["molecule_id"].astype(str)
    )
    actual_molecules = set(
        output["molecule_id"].astype(str)
    )
    missing_molecules = sorted(
        expected_molecules - actual_molecules
    )

    print()
    print("Fused formula hypotheses")
    print("=" * 72)
    print(
        "Top-3 fused formula rows:",
        f"{len(top):,}",
    )
    print(
        "Output hypotheses:",
        f"{len(output):,}",
    )
    print(
        "Production hypotheses reused:",
        f"{reused:,}",
    )
    print(
        "MSBuddy-only hypotheses reconstructed:",
        f"{reconstructed:,}",
    )
    print(
        "Fused formulas skipped:",
        f"{skipped:,}",
    )
    print(
        "Molecules represented:",
        f"{len(actual_molecules):,}",
    )
    print(
        "Molecules missing entirely:",
        f"{len(missing_molecules):,}",
    )

    if missing_molecules:
        print(
            "First missing molecule IDs:",
            missing_molecules[:20],
        )

    return output


def main() -> None:
    for path in [
        TEST_PATH,
        PRODUCTION_HYPOTHESES_PATH,
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"Missing required artifact: {path}"
            )

    TEST_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    test = (
        pd.read_parquet(TEST_PATH)
        .reset_index(drop=True)
        .copy()
    )

    required_test = {
        "molecule_id",
        "spectrum_id",
        "ms2_mzs",
        "ms2_normalized_intensities",
        "adduct",
        "ionization_mode",
        "precursor_mz",
    }

    missing_test = required_test - set(
        test.columns
    )

    if missing_test:
        raise ValueError(
            "Test parquet is missing columns: "
            f"{sorted(missing_test)}"
        )

    test["molecule_id"] = (
        test["molecule_id"].astype(str)
    )
    test["spectrum_id"] = (
        test["spectrum_id"].astype(str)
    )
    test["source_test_row_id"] = np.arange(
        len(test),
        dtype=np.int64,
    )

    production_hypotheses = pd.read_parquet(
        PRODUCTION_HYPOTHESES_PATH
    )

    production = prepare_production(
        production_hypotheses
    )

    selected = select_query_spectra(
        test
    )

    print(
        "Test molecules:",
        f"{test['molecule_id'].nunique():,}",
    )
    print(
        "Production formula hypotheses:",
        f"{len(production_hypotheses):,}",
    )
    print(
        "Production top-5 formula candidates:",
        f"{len(production):,}",
    )

    msbuddy = run_msbuddy(
        selected
    )

    msbuddy.to_parquet(
        MSBUDDY_OUTPUT_PATH,
        index=False,
        compression="zstd",
    )

    candidates = build_fusion_candidates(
        production,
        msbuddy,
    )

    fused = build_fused_hypotheses(
        test,
        production_hypotheses,
        candidates,
    )

    fused.to_parquet(
        FUSED_HYPOTHESES_PATH,
        index=False,
        compression="zstd",
    )

    print()
    print("Frozen formula-fusion settings")
    print("=" * 72)
    print(
        "Production reciprocal-rank weight:",
        PRODUCTION_WEIGHT,
    )
    print(
        "MSBuddy reciprocal-rank weight:",
        MSBUDDY_WEIGHT,
    )
    print(
        "Agreement bonus:",
        AGREEMENT_BONUS,
    )
    print(
        "Production support weight:",
        PRODUCTION_SUPPORT_WEIGHT,
    )
    print(
        "MSBuddy support weight:",
        MSBUDDY_SUPPORT_WEIGHT,
    )
    print(
        "Final formula shortlist:",
        TOP_K_FINAL,
    )
    print()
    print(
        "Wrote MSBuddy gate:",
        MSBUDDY_OUTPUT_PATH,
    )
    print(
        "Wrote fused formula hypotheses:",
        FUSED_HYPOTHESES_PATH,
    )


if __name__ == "__main__":
    main()
