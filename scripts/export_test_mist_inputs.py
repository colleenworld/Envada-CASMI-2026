from __future__ import annotations

import os
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

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

HYPOTHESES_PATH = (
    PROJECT_ROOT
    / "data/processed/test/"
      "multispectrum_formula_hypotheses.parquet"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data/processed/test/mist"
)

MGF_PATH = (
    OUTPUT_DIR
    / "test_multiformula.mgf"
)

LABELS_PATH = (
    OUTPUT_DIR
    / "test_multiformula_labels.tsv"
)


# ---------------------------------------------------------------------------
# Private-rerun diagnostics
# ---------------------------------------------------------------------------

DEBUG_STAGE = (
    os.environ.get(
        "CASMI26_EXPORT_DEBUG_STAGE",
        "",
    )
    .strip()
)


def stop_after(
    stage: str,
) -> bool:
    if DEBUG_STAGE == stage:
        print()
        print(
            f"DEBUG: stopping successfully after {stage}"
        )
        return True

    return False


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def format_peaks(
    mzs,
    intensities,
) -> List[str]:
    mzs = np.asarray(
        mzs,
        dtype=float,
    )

    intensities = np.asarray(
        intensities,
        dtype=float,
    )

    if len(mzs) != len(intensities):
        raise ValueError(
            "m/z and intensity arrays differ in length"
        )

    lines = []

    for mz, intensity in zip(
        mzs,
        intensities,
    ):
        if not (
            np.isfinite(mz)
            and np.isfinite(intensity)
        ):
            continue

        if intensity <= 0:
            continue

        lines.append(
            f"{mz:.8f} {intensity:.8f}"
        )

    return lines


def write_empty_outputs() -> None:
    """
    Produce valid empty MIST artifacts.

    This allows the later pipeline to recognize that the positive
    MIST branch has no work rather than failing during export.
    """

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    MGF_PATH.write_text("")

    pd.DataFrame(
        columns=[
            "spec",
            "formula",
            "ionization",
        ]
    ).to_csv(
        LABELS_PATH,
        sep="\t",
        index=False,
    )

    print()
    print(
        "No hypotheses available for MIST export."
    )

    print(
        f"Wrote empty {MGF_PATH}"
    )

    print(
        f"Wrote empty {LABELS_PATH}"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "Project root:",
        PROJECT_ROOT,
    )

    print(
        "Test:",
        TEST_PATH,
    )

    print(
        "Hypotheses:",
        HYPOTHESES_PATH,
    )

    if DEBUG_STAGE:
        print(
            "Export debug stage:",
            DEBUG_STAGE,
        )

    # ------------------------------------------------------------------
    # Stage 1: load test
    # ------------------------------------------------------------------

    test = (
        pd.read_parquet(
            TEST_PATH
        )
        .reset_index(
            drop=True
        )
    )

    required_test_columns = {
        "molecule_id",
        "spectrum_id",
        "precursor_mz",
        "adduct",
        "ionization_mode",
        "ms2_mzs",
        "ms2_normalized_intensities",
    }

    missing_test_columns = (
        required_test_columns
        - set(
            test.columns
        )
    )

    if missing_test_columns:
        raise ValueError(
            "Test parquet is missing required columns: "
            f"{sorted(missing_test_columns)}"
        )

    test[
        "source_test_row_id"
    ] = np.arange(
        len(test),
        dtype=np.int64,
    )

    print()
    print(
        "Test spectra:",
        len(test),
    )

    print(
        "Test molecules:",
        test[
            "molecule_id"
        ].nunique(),
    )

    if stop_after(
        "load_test"
    ):
        return

    # ------------------------------------------------------------------
    # Stage 2: load hypotheses
    # ------------------------------------------------------------------

    hypotheses = pd.read_parquet(
        HYPOTHESES_PATH
    )

    required_hypothesis_columns = {
        "molecule_id",
        "source_test_row_id",
        "selected_spectrum_id",
        "selected_precursor_mz",
        "selected_adduct",
        "selected_ionization_mode",
        "hypothesis_spec",
        "candidate_formula",
    }

    missing_hypothesis_columns = (
        required_hypothesis_columns
        - set(
            hypotheses.columns
        )
    )

    if missing_hypothesis_columns:
        raise ValueError(
            "Hypothesis parquet is missing required columns: "
            f"{sorted(missing_hypothesis_columns)}"
        )

    print()
    print(
        "Hypotheses:",
        len(hypotheses),
    )

    print(
        "Molecules:",
        hypotheses[
            "molecule_id"
        ].nunique(),
    )

    if stop_after(
        "load_hypotheses"
    ):
        return

    #
    # An empty positive branch is not inherently an error.
    #
    if hypotheses.empty:
        write_empty_outputs()
        return

    # ------------------------------------------------------------------
    # Stage 3: validate hypothesis identifiers
    # ------------------------------------------------------------------

    duplicate_specs = (
        hypotheses[
            "hypothesis_spec"
        ]
        .duplicated()
    )

    print()
    print(
        "Duplicate hypothesis_spec values:",
        int(
            duplicate_specs.sum()
        ),
    )

    if duplicate_specs.any():
        duplicates = (
            hypotheses.loc[
                duplicate_specs,
                [
                    "molecule_id",
                    "hypothesis_spec",
                ],
            ]
            .head(20)
        )

        print(
            duplicates.to_string(
                index=False
            )
        )

        raise ValueError(
            "Duplicate hypothesis_spec values"
        )

    if stop_after(
        "validate_hypotheses"
    ):
        return

    # ------------------------------------------------------------------
    # Stage 4: source-row lookup
    # ------------------------------------------------------------------

    source_rows = (
        test.set_index(
            "source_test_row_id"
        )
    )

    source_ids = set(
        source_rows.index
    )

    hypothesis_source_ids = set(
        hypotheses[
            "source_test_row_id"
        ]
        .astype(int)
    )

    missing_source_ids = sorted(
        hypothesis_source_ids
        - source_ids
    )

    print()
    print(
        "Hypothesis source row IDs:",
        len(
            hypothesis_source_ids
        ),
    )

    print(
        "Missing source row IDs:",
        len(
            missing_source_ids
        ),
    )

    if missing_source_ids:
        print(
            "First missing source row IDs:",
            missing_source_ids[:20],
        )

        raise ValueError(
            "Hypothesis artifact references source rows "
            "that are not present in test.parquet"
        )

    if stop_after(
        "source_lookup"
    ):
        return

    # ------------------------------------------------------------------
    # Stage 5: integrity checks
    # ------------------------------------------------------------------

    spectrum_id_mismatches = 0
    precursor_mismatches = 0
    adduct_mismatches = 0

    for row in hypotheses.itertuples(
        index=False
    ):
        source_id = int(
            row.source_test_row_id
        )

        source = source_rows.loc[
            source_id
        ]

        if (
            str(
                source.spectrum_id
            )
            != str(
                row.selected_spectrum_id
            )
        ):
            spectrum_id_mismatches += 1

        if not np.isclose(
            float(
                source.precursor_mz
            ),
            float(
                row.selected_precursor_mz
            ),
        ):
            precursor_mismatches += 1

        if (
            str(
                source.adduct
            )
            != str(
                row.selected_adduct
            )
        ):
            adduct_mismatches += 1

    mismatches = (
        spectrum_id_mismatches
        + precursor_mismatches
        + adduct_mismatches
    )

    print()
    print(
        "Spectrum ID mismatches:",
        spectrum_id_mismatches,
    )

    print(
        "Precursor m/z mismatches:",
        precursor_mismatches,
    )

    print(
        "Adduct mismatches:",
        adduct_mismatches,
    )

    print(
        "Total source integrity mismatches:",
        mismatches,
    )

    #
    # Preserve previous behavior: mismatches are diagnostic, not fatal.
    #

    if stop_after(
        "integrity"
    ):
        return

    # ------------------------------------------------------------------
    # Stage 6: check peak exportability
    # ------------------------------------------------------------------

    no_valid_peak_specs = []

    mismatched_peak_array_specs = []

    for row in hypotheses.itertuples(
        index=False
    ):
        source = source_rows.loc[
            int(
                row.source_test_row_id
            )
        ]

        try:
            peak_lines = format_peaks(
                source.ms2_mzs,
                source.ms2_normalized_intensities,
            )

        except ValueError:
            mismatched_peak_array_specs.append(
                str(
                    row.hypothesis_spec
                )
            )

            continue

        if not peak_lines:
            no_valid_peak_specs.append(
                str(
                    row.hypothesis_spec
                )
            )

    print()
    print(
        "Hypotheses with mismatched peak-array lengths:",
        len(
            mismatched_peak_array_specs
        ),
    )

    if mismatched_peak_array_specs:
        print(
            "First examples:",
            mismatched_peak_array_specs[
                :20
            ],
        )

    print(
        "Hypotheses with no valid positive-intensity peaks:",
        len(
            no_valid_peak_specs
        ),
    )

    if no_valid_peak_specs:
        print(
            "First examples:",
            no_valid_peak_specs[
                :20
            ],
        )

    if mismatched_peak_array_specs:
        raise ValueError(
            f"{len(mismatched_peak_array_specs)} "
            "hypotheses have m/z/intensity arrays "
            "with different lengths."
        )

    #
    # Do not throw yet for no-valid-peaks during debugging.
    # We want to know whether this is what distinguishes the
    # private test.
    #
    if stop_after(
        "peak_validation"
    ):
        return

    # ------------------------------------------------------------------
    # Stage 7: construct MGF and labels
    # ------------------------------------------------------------------

    mgf_blocks = []

    label_rows = []

    skipped_no_peak = 0

    for row in hypotheses.itertuples(
        index=False
    ):
        source_id = int(
            row.source_test_row_id
        )

        source = source_rows.loc[
            source_id
        ]

        peak_lines = format_peaks(
            source.ms2_mzs,
            source.ms2_normalized_intensities,
        )

        #
        # Hidden-test robustness:
        #
        # If a spectrum has no exportable peaks, it cannot be passed
        # to MIST. Skip this hypothesis rather than killing the entire
        # submission. Molecules left without a positive MIST ranking
        # will be handled by the spectral-retrieval fallback.
        #
        if not peak_lines:
            skipped_no_peak += 1
            continue

        block = [
            "BEGIN IONS",
            (
                f"TITLE="
                f"{row.hypothesis_spec}"
            ),
            (
                f"PEPMASS="
                f"{float(source.precursor_mz):.8f}"
            ),
        ]

        mode = str(
            row.selected_ionization_mode
        ).lower()

        if mode == "positive":
            block.append(
                "IONMODE=Positive"
            )

        elif mode == "negative":
            block.append(
                "IONMODE=Negative"
            )

        mgf_blocks.append(
            "\n".join(
                block
                + peak_lines
                + [
                    "END IONS",
                ]
            )
        )

        label_rows.append(
            {
                "spec":
                    row.hypothesis_spec,

                "formula":
                    row.candidate_formula,

                "ionization":
                    row.selected_adduct,
            }
        )

    print()
    print(
        "MGF blocks constructed:",
        len(
            mgf_blocks
        ),
    )

    print(
        "Label rows constructed:",
        len(
            label_rows
        ),
    )

    print(
        "Hypotheses skipped for no valid peaks:",
        skipped_no_peak,
    )

    if stop_after(
        "build_outputs"
    ):
        return

    # ------------------------------------------------------------------
    # Stage 8: validate generated labels
    # ------------------------------------------------------------------

    labels = pd.DataFrame(
        label_rows,
        columns=[
            "spec",
            "formula",
            "ionization",
        ],
    )

    if not labels.empty:
        duplicate_labels = (
            labels[
                "spec"
            ]
            .duplicated()
        )

        if duplicate_labels.any():
            print(
                labels.loc[
                    duplicate_labels,
                    [
                        "spec",
                        "formula",
                        "ionization",
                    ],
                ]
                .head(20)
                .to_string(
                    index=False
                )
            )

            raise ValueError(
                "Duplicate label spec values"
            )

    print()
    print(
        "Duplicate generated label specs:",
        int(
            labels[
                "spec"
            ]
            .duplicated()
            .sum()
        )
        if not labels.empty
        else 0,
    )

    if stop_after(
        "validate_outputs"
    ):
        return

    # ------------------------------------------------------------------
    # Stage 9: write files
    # ------------------------------------------------------------------

    if mgf_blocks:
        MGF_PATH.write_text(
            "\n\n".join(
                mgf_blocks
            )
            + "\n"
        )

    else:
        MGF_PATH.write_text(
            ""
        )

    labels.to_csv(
        LABELS_PATH,
        sep="\t",
        index=False,
    )

    print()
    print(
        "Export complete"
    )

    print(
        "---------------"
    )

    print(
        "MGF blocks:",
        len(
            mgf_blocks
        ),
    )

    print(
        "Label rows:",
        len(
            labels
        ),
    )

    print(
        "Source integrity mismatches:",
        mismatches,
    )

    print(
        "Skipped hypotheses with no valid peaks:",
        skipped_no_peak,
    )

    print()

    print(
        "Label adduct counts:"
    )

    if labels.empty:
        print(
            "(none)"
        )

    else:
        print(
            labels[
                "ionization"
            ]
            .value_counts(
                dropna=False
            )
            .to_string()
        )

    print()
    print(
        f"Wrote {MGF_PATH}"
    )

    print(
        f"Wrote {LABELS_PATH}"
    )

    if stop_after(
        "write"
    ):
        return


if __name__ == "__main__":
    main()