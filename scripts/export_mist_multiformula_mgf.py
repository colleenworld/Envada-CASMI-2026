from pathlib import Path

import pandas as pd


ROOT = Path("data/processed/mist/benchmark_100")

INPUT_MGF = ROOT / "spectra.mgf"
MAPPING_PATH = ROOT / "mist_multiformula_mapping.parquet"
OUTPUT_MGF = ROOT / "spectra_multiformula.mgf"


def parse_mgf(path: Path) -> dict[str, list[str]]:
    """
    Return MGF blocks keyed by FEATURE_ID.
    """
    blocks = {}
    current = []

    with path.open() as f:
        for raw_line in f:
            line = raw_line.rstrip("\n")

            if line == "BEGIN IONS":
                if current:
                    raise ValueError("Nested BEGIN IONS encountered")
                current = [line]
                continue

            if current:
                current.append(line)

                if line == "END IONS":
                    feature_id = None

                    for block_line in current:
                        if block_line.startswith("FEATURE_ID="):
                            feature_id = block_line.split("=", 1)[1]
                            break

                    if feature_id is None:
                        raise ValueError(
                            "MGF block is missing FEATURE_ID"
                        )

                    if feature_id in blocks:
                        raise ValueError(
                            f"Duplicate FEATURE_ID: {feature_id}"
                        )

                    blocks[feature_id] = current
                    current = []

    if current:
        raise ValueError("Unterminated MGF block")

    return blocks


def rewrite_block(
    block: list[str],
    hypothesis_spec: str,
    candidate_formula: str,
) -> list[str]:
    """
    Copy a spectrum block while replacing any identity/formula fields
    that could contain oracle information.
    """
    output = []

    saw_title = False
    saw_feature_id = False
    saw_formula = False

    for line in block:
        if line.startswith("TITLE="):
            output.append(f"TITLE={hypothesis_spec}")
            saw_title = True

        elif line.startswith("FEATURE_ID="):
            output.append(f"FEATURE_ID={hypothesis_spec}")
            saw_feature_id = True

        elif line.startswith("FORMULA="):
            output.append(f"FORMULA={candidate_formula}")
            saw_formula = True

        else:
            output.append(line)

    if not saw_title:
        raise ValueError("MGF block is missing TITLE")

    if not saw_feature_id:
        raise ValueError("MGF block is missing FEATURE_ID")

    if not saw_formula:
        raise ValueError("MGF block is missing FORMULA")

    return output


def main():
    mapping = pd.read_parquet(MAPPING_PATH)
    source_blocks = parse_mgf(INPUT_MGF)

    expected_original_specs = set(mapping["original_spec"])
    available_specs = set(source_blocks)

    missing = expected_original_specs - available_specs
    if missing:
        raise ValueError(
            f"{len(missing)} source spectra missing from MGF: "
            f"{sorted(missing)[:10]}"
        )

    if mapping["hypothesis_spec"].duplicated().any():
        duplicates = (
            mapping.loc[
                mapping["hypothesis_spec"].duplicated(keep=False),
                "hypothesis_spec",
            ]
            .unique()
            .tolist()
        )
        raise ValueError(
            f"Duplicate hypothesis IDs: {duplicates[:10]}"
        )

    output_blocks = []

    for row in mapping.itertuples(index=False):
        source = source_blocks[row.original_spec]

        new_block = rewrite_block(
            source,
            hypothesis_spec=row.hypothesis_spec,
            candidate_formula=row.candidate_formula,
        )

        output_blocks.append(new_block)

    with OUTPUT_MGF.open("w") as f:
        for block in output_blocks:
            f.write("\n".join(block))
            f.write("\n\n")

    print(f"Original spectra available: {len(source_blocks)}")
    print(f"Original spectra used: {len(expected_original_specs)}")
    print(f"Expanded spectra written: {len(output_blocks)}")
    print(f"Unique hypothesis IDs: {mapping['hypothesis_spec'].nunique()}")
    print(f"Wrote {OUTPUT_MGF}")


if __name__ == "__main__":
    main()