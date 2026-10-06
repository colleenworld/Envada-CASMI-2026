from pathlib import Path
from typing import List
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(
    "/home/colleen/PycharmProjects/Envada-CASMI-2026"
)

TEST_PATH = (
    PROJECT_ROOT
    / "data/raw/test.parquet"
)

HYPOTHESES_PATH = (
    PROJECT_ROOT
    / "data/processed/test/multispectrum_formula_hypotheses.parquet"
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


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    test = (
        pd.read_parquet(
            TEST_PATH
        )
        .reset_index(
            drop=True
        )
    )

    test[
        "source_test_row_id"
    ] = np.arange(
        len(test),
        dtype=np.int64,
    )

    hypotheses = pd.read_parquet(
        HYPOTHESES_PATH
    )

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

    if hypotheses[
        "hypothesis_spec"
    ].duplicated().any():
        raise ValueError(
            "Duplicate hypothesis_spec values"
        )

    source_rows = (
        test.set_index(
            "source_test_row_id"
        )
    )

    mgf_blocks = []
    label_rows = []

    mismatches = 0

    for row in hypotheses.itertuples(
        index=False
    ):
        source_id = int(
            row.source_test_row_id
        )

        if source_id not in source_rows.index:
            raise ValueError(
                f"Missing source row {source_id}"
            )

        source = source_rows.loc[
            source_id
        ]

        #
        # Integrity checks against the hypothesis artifact.
        #
        if (
            str(source.spectrum_id)
            != str(
                row.selected_spectrum_id
            )
        ):
            mismatches += 1

        if not np.isclose(
            float(source.precursor_mz),
            float(
                row.selected_precursor_mz
            ),
        ):
            mismatches += 1

        if (
            str(source.adduct)
            != str(
                row.selected_adduct
            )
        ):
            mismatches += 1

        peak_lines = format_peaks(
            source.ms2_mzs,
            source.ms2_normalized_intensities,
        )

        if not peak_lines:
            raise ValueError(
                f"No valid peaks for {row.hypothesis_spec}"
            )

        #
        # MGF block.
        #
        block = [
            "BEGIN IONS",
            f"TITLE={row.hypothesis_spec}",
            f"PEPMASS={float(source.precursor_mz):.8f}",
        ]

        #
        # Keep polarity explicit.
        #
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

        block.extend(
            peak_lines
        )

        block.append(
            "END IONS"
        )

        mgf_blocks.append(
            "\n".join(block)
        )

        #
        # MIST labels.
        #
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

    labels = pd.DataFrame(
        label_rows
    )

    if labels[
        "spec"
    ].duplicated().any():
        raise ValueError(
            "Duplicate label spec values"
        )

    MGF_PATH.write_text(
        "\n\n".join(
            mgf_blocks
        )
        + "\n"
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

    print()
    print(
        "Label adduct counts:"
    )

    print(
        labels[
            "ionization"
        ]
        .value_counts()
        .to_string()
    )

    print()
    print(
        f"Wrote {MGF_PATH}"
    )

    print(
        f"Wrote {LABELS_PATH}"
    )


if __name__ == "__main__":
    main()