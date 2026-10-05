from pathlib import Path

import pandas as pd


ROOT = Path("data/processed/mist/validation_500")

MAPPING_IN = ROOT / "mist_multiformula_mapping.parquet"
LABELS_IN = ROOT / "mist_multiformula_labels.tsv"
MGF_IN = ROOT / "spectra_multiformula.mgf"

MAPPING_OUT = ROOT / "mist_multiformula_mapping_processable.parquet"
LABELS_OUT = ROOT / "mist_multiformula_labels_processable.tsv"
MGF_OUT = ROOT / "spectra_multiformula_processable.mgf"


UNPROCESSABLE_QUERIES = {
    "mistval_KXGHHSIMRWPVQM",
    "mistval_ZIBSUBYTARIENZ",
}


def parse_blocks(text: str):
    blocks = []

    for chunk in text.split("BEGIN IONS"):
        chunk = chunk.strip()
        if not chunk:
            continue

        block = "BEGIN IONS\n" + chunk.strip() + "\n"

        feature_id = None

        for line in block.splitlines():
            if line.startswith("FEATURE_ID="):
                feature_id = line.split("=", 1)[1]
                break

        if feature_id is None:
            raise ValueError("MGF block missing FEATURE_ID")

        blocks.append((feature_id, block))

    return blocks


def main():
    mapping = pd.read_parquet(MAPPING_IN)
    labels = pd.read_csv(LABELS_IN, sep="\t")
    mgf_text = MGF_IN.read_text()

    remove_mask = mapping["original_spec"].isin(
        UNPROCESSABLE_QUERIES
    )

    removed = mapping.loc[remove_mask].copy()
    kept_mapping = mapping.loc[~remove_mask].copy()

    kept_ids = set(kept_mapping["hypothesis_spec"])

    kept_labels = labels[
        labels["spec"].isin(kept_ids)
    ].copy()

    blocks = parse_blocks(mgf_text)

    kept_blocks = [
        block
        for feature_id, block in blocks
        if feature_id in kept_ids
    ]

    kept_mapping.to_parquet(
        MAPPING_OUT,
        index=False,
    )

    kept_labels.to_csv(
        LABELS_OUT,
        sep="\t",
        index=False,
    )

    MGF_OUT.write_text(
        "\n".join(kept_blocks)
    )

    print("Original hypotheses:", len(mapping))
    print("Removed hypotheses:", len(removed))
    print("Remaining hypotheses:", len(kept_mapping))
    print(
        "Removed queries:",
        sorted(removed["original_spec"].unique()),
    )

    print("\nOutput validation:")
    print("mapping:", len(kept_mapping))
    print("labels:", len(kept_labels))
    print("MGF blocks:", len(kept_blocks))

    assert len(kept_mapping) == 1635
    assert len(kept_labels) == 1635
    assert len(kept_blocks) == 1635
    assert set(kept_mapping["hypothesis_spec"]) == set(
        kept_labels["spec"]
    )

    print("\nWrote:")
    print(MAPPING_OUT)
    print(LABELS_OUT)
    print(MGF_OUT)


if __name__ == "__main__":
    main()