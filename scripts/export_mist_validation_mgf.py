from pathlib import Path

import pandas as pd


ROOT = Path("data/processed/mist/validation_500")

MANIFEST_PATH = ROOT / "manifest.parquet"
MAPPING_PATH = ROOT / "mist_multiformula_mapping.parquet"
OUTPUT_MGF = ROOT / "spectra_multiformula.mgf"


def write_block(f, row, hypothesis_spec, candidate_formula):
    f.write("BEGIN IONS\n")
    f.write(f"TITLE={hypothesis_spec}\n")
    f.write(f"FEATURE_ID={hypothesis_spec}\n")
    f.write(f"PEPMASS={row.precursor_mz:.8f}\n")
    f.write("CHARGE=1+\n")
    f.write(f"FORMULA={candidate_formula}\n")
    f.write(f"ADDUCT={row.adduct}\n")
    f.write(f"IONMODE={row.ionization_mode}\n")

    for mz, intensity in zip(
        row.ms2_mzs,
        row.ms2_normalized_intensities,
    ):
        f.write(f"{mz:.8f} {intensity:.8f}\n")

    f.write("END IONS\n\n")


def main():
    manifest = pd.read_parquet(MANIFEST_PATH)
    mapping = pd.read_parquet(MAPPING_PATH)

    manifest_by_spec = (
        manifest
        .set_index("spec")
    )

    if mapping["hypothesis_spec"].duplicated().any():
        raise ValueError("Duplicate hypothesis IDs")

    written = 0

    with OUTPUT_MGF.open("w") as f:
        for hypothesis in mapping.itertuples(index=False):
            row = manifest_by_spec.loc[
                hypothesis.original_spec
            ]

            write_block(
                f,
                row,
                hypothesis.hypothesis_spec,
                hypothesis.candidate_formula,
            )
            written += 1

    print("Manifest queries:", len(manifest))
    print("Hypotheses:", len(mapping))
    print("MGF blocks written:", written)
    print(f"Wrote {OUTPUT_MGF}")


if __name__ == "__main__":
    main()