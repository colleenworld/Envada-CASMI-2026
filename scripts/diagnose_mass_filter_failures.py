from pathlib import Path

import numpy as np
import pandas as pd

from casmi26.chemistry.adducts import neutral_mass
from casmi26.retrieval.index import load_retrieval_index


INDEX_DIR = Path(
    "data/processed/retrieval_index"
)

MANIFEST_PATH = Path(
    "data/processed/splits/retrieval_dev.parquet"
)

FAILURES = [
    "DYDCUQKUCUHJBH",
    "MLSAQOINCGAULQ",
    "NQRKYASMKDDGHT",
    "QLPHOXTXAKOFMU",
]


def infer_mass(
    precursor_mz,
    adduct,
) -> float | None:
    try:
        mz = float(
            precursor_mz
        )
    except (
        TypeError,
        ValueError,
    ):
        return None

    if not np.isfinite(
        mz
    ):
        return None

    value = neutral_mass(
        mz,
        str(adduct),
    )

    if (
        value is None
        or not np.isfinite(
            value
        )
    ):
        return None

    return float(
        value
    )


def main() -> None:
    print(
        "Loading retrieval metadata..."
    )

    metadata, _ = (
        load_retrieval_index(
            INDEX_DIR
        )
    )

    manifest = pd.read_parquet(
        MANIFEST_PATH
    )

    for truth_key in FAILURES:
        manifest_row = (
            manifest[
                manifest[
                    "inchikey14"
                ].astype(str)
                == truth_key
            ]
        )

        if len(manifest_row) != 1:
            raise RuntimeError(
                f"Expected one manifest row "
                f"for {truth_key}, found "
                f"{len(manifest_row)}"
            )

        query_library = str(
            manifest_row.iloc[
                0
            ][
                "query_library"
            ]
        )

        structure_rows = (
            metadata[
                metadata[
                    "inchikey14"
                ].astype(str)
                == truth_key
            ]
            .copy()
        )

        query_rows = (
            structure_rows[
                structure_rows[
                    "ingest_lib"
                ].astype(str)
                == query_library
            ]
            .copy()
        )

        reference_rows = (
            structure_rows[
                structure_rows[
                    "ingest_lib"
                ].astype(str)
                != query_library
            ]
            .copy()
        )

        print()
        print(
            "=" * 72
        )

        print(
            f"{truth_key}  "
            f"query_library="
            f"{query_library}"
        )

        print(
            "=" * 72
        )

        print()
        print(
            "Query spectra"
        )
        print(
            "-------------"
        )

        for row in (
            query_rows.itertuples(
                index=False
            )
        ):
            mass = infer_mass(
                row.precursor_mz,
                row.adduct,
            )

            print(
                f"library="
                f"{row.ingest_lib:<12} "
                f"mode="
                f"{row.ionization_mode:<8} "
                f"precursor="
                f"{row.precursor_mz!s:<14} "
                f"adduct="
                f"{row.adduct!s:<20} "
                f"neutral_mass="
                f"{mass}"
            )

        print()
        print(
            "Truth reference spectra"
        )
        print(
            "-----------------------"
        )

        reference_masses: list[
            tuple[
                str,
                str,
                float,
            ]
        ] = []

        for row in (
            reference_rows.itertuples(
                index=False
            )
        ):
            mass = infer_mass(
                row.precursor_mz,
                row.adduct,
            )

            print(
                f"library="
                f"{row.ingest_lib:<12} "
                f"mode="
                f"{row.ionization_mode:<8} "
                f"precursor="
                f"{row.precursor_mz!s:<14} "
                f"adduct="
                f"{row.adduct!s:<20} "
                f"neutral_mass="
                f"{mass}"
            )

            if mass is not None:
                reference_masses.append(
                    (
                        str(
                            row.ingest_lib
                        ),
                        str(
                            row.ionization_mode
                        ),
                        mass,
                    )
                )

        query_masses = []

        for row in (
            query_rows.itertuples(
                index=False
            )
        ):
            mass = infer_mass(
                row.precursor_mz,
                row.adduct,
            )

            if mass is not None:
                query_masses.append(
                    (
                        str(
                            row.ionization_mode
                        ),
                        mass,
                    )
                )

        print()
        print(
            "Same-polarity neutral-mass "
            "differences"
        )
        print(
            "--------------------------------------"
        )

        differences = []

        for (
            query_mode,
            query_mass,
        ) in query_masses:
            for (
                library,
                reference_mode,
                reference_mass,
            ) in reference_masses:
                if (
                    query_mode
                    != reference_mode
                ):
                    continue

                difference = abs(
                    query_mass
                    - reference_mass
                )

                differences.append(
                    (
                        difference,
                        library,
                        query_mode,
                        query_mass,
                        reference_mass,
                    )
                )

        differences.sort(
            key=lambda item:
            item[0]
        )

        if not differences:
            print(
                "No comparable same-polarity "
                "neutral masses."
            )
        else:
            for (
                difference,
                library,
                mode,
                query_mass,
                reference_mass,
            ) in differences[
                :10
            ]:
                print(
                    f"delta="
                    f"{difference:.6f} Da  "
                    f"mode="
                    f"{mode:<8} "
                    f"library="
                    f"{library:<12} "
                    f"query="
                    f"{query_mass:.6f}  "
                    f"reference="
                    f"{reference_mass:.6f}"
                )


if __name__ == "__main__":
    main()