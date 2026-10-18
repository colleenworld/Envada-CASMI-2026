from __future__ import annotations

import json
import time
from pathlib import Path
from urllib import error, parse, request

import pandas as pd
from rdkit import Chem, RDLogger


PROJECT_ROOT = (
    Path(__file__).resolve().parent.parent
)

TRAIN_PATH = (
    PROJECT_ROOT
    / "data/raw/train.parquet"
)

VALIDATION_STRUCTURES_PATH = (
    PROJECT_ROOT
    / "data/processed/splits/validation_structures.parquet"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data/processed/diagnostics"
)

OUTPUT_PATH = (
    OUTPUT_DIR
    / "pubchem_validation_coverage.parquet"
)

SUMMARY_PATH = (
    OUTPUT_DIR
    / "pubchem_validation_coverage_summary.json"
)

PUG_REST_URL = (
    "https://pubchem.ncbi.nlm.nih.gov/rest/pug/"
    "compound/inchikey/property/"
    "InChIKey,CanonicalSMILES,IsomericSMILES,MolecularFormula/JSON"
)

BATCH_SIZE = 100

# PubChem asks clients to stay below five requests per second.
MIN_REQUEST_INTERVAL_SECONDS = 0.30

MAX_RETRIES = 6

RETRYABLE_STATUS_CODES = {
    429,
    500,
    502,
    503,
    504,
}


RDLogger.DisableLog(
    "rdApp.*"
)


def smiles_to_full_inchikey(
    smiles: str,
) -> str | None:
    mol = Chem.MolFromSmiles(
        str(smiles)
    )

    if mol is None:
        return None

    try:
        return Chem.MolToInchiKey(
            mol
        )
    except Exception:
        return None


def build_validation_truth() -> pd.DataFrame:
    validation_keys = set(
        pd.read_parquet(
            VALIDATION_STRUCTURES_PATH,
            columns=[
                "inchikey14",
            ],
        )[
            "inchikey14"
        ]
        .astype(str)
    )

    train = pd.read_parquet(
        TRAIN_PATH,
        columns=[
            "inchikey14",
            "normalized_smiles",
            "molecular_formula",
        ],
    )

    train[
        "inchikey14"
    ] = (
        train[
            "inchikey14"
        ].astype(str)
    )

    validation = (
        train.loc[
            train[
                "inchikey14"
            ].isin(
                validation_keys
            ),
            [
                "inchikey14",
                "normalized_smiles",
                "molecular_formula",
            ],
        ]
        .dropna()
        .drop_duplicates()
        .copy()
    )

    counts = (
        validation.groupby(
            [
                "inchikey14",
                "normalized_smiles",
                "molecular_formula",
            ],
            as_index=False,
        )
        .size()
        .sort_values(
            [
                "inchikey14",
                "size",
                "molecular_formula",
                "normalized_smiles",
            ],
            ascending=[
                True,
                False,
                True,
                True,
            ],
        )
    )

    truth = (
        counts
        .drop_duplicates(
            subset=[
                "inchikey14",
            ],
            keep="first",
        )
        .drop(
            columns=[
                "size",
            ]
        )
        .sort_values(
            "inchikey14"
        )
        .reset_index(
            drop=True
        )
    )

    missing = (
        validation_keys
        - set(
            truth[
                "inchikey14"
            ]
        )
    )

    if missing:
        raise RuntimeError(
            "Validation structures are missing usable "
            "SMILES/formula metadata. "
            f"Examples: {sorted(missing)[:20]}"
        )

    truth[
        "full_inchikey"
    ] = [
        smiles_to_full_inchikey(
            smiles
        )
        for smiles in truth[
            "normalized_smiles"
        ]
    ]

    return truth


def fetch_pubchem_batch(
    inchikeys: list[str],
) -> dict[
    str,
    dict,
]:
    if not inchikeys:
        return {}

    payload = parse.urlencode(
        {
            "inchikey":
                ",".join(
                    inchikeys
                ),
        }
    ).encode(
        "utf-8"
    )

    last_error = None

    for attempt in range(
        1,
        MAX_RETRIES + 1,
    ):
        req = request.Request(
            PUG_REST_URL,
            data=payload,
            headers={
                "Content-Type":
                    "application/x-www-form-urlencoded",
                "User-Agent":
                    "CASMI26-validation-diagnostic/1.0",
            },
            method="POST",
        )

        try:
            with request.urlopen(
                req,
                timeout=60,
            ) as response:
                body = response.read()

            parsed = json.loads(
                body.decode(
                    "utf-8"
                )
            )

            properties = (
                parsed
                .get(
                    "PropertyTable",
                    {},
                )
                .get(
                    "Properties",
                    [],
                )
            )

            result = {}

            for row in properties:
                key = row.get(
                    "InChIKey"
                )

                if not key:
                    continue

                result[
                    str(key)
                ] = row

            return result

        except error.HTTPError as exc:
            last_error = exc

            if exc.code == 404:
                return {}

            if (
                exc.code
                not in RETRYABLE_STATUS_CODES
            ):
                raise

        except (
            error.URLError,
            TimeoutError,
            json.JSONDecodeError,
        ) as exc:
            last_error = exc

        if attempt < MAX_RETRIES:
            delay = min(
                30.0,
                2.0 ** (
                    attempt - 1
                ),
            )

            print(
                f"  request failed; retrying in "
                f"{delay:.1f}s: {last_error}"
            )

            time.sleep(
                delay
            )

    raise RuntimeError(
        "PubChem request failed after retries"
    ) from last_error


def main() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    truth = (
        build_validation_truth()
    )

    print(
        "Validation structures:",
        f"{len(truth):,}",
    )

    invalid_key_mask = (
        truth[
            "full_inchikey"
        ].isna()
    )

    print(
        "Structures without derivable full InChIKey:",
        int(
            invalid_key_mask.sum()
        ),
    )

    lookup = {}

    if OUTPUT_PATH.exists():
        previous = pd.read_parquet(
            OUTPUT_PATH
        )

        if {
            "full_inchikey",
            "pubchem_found",
        } <= set(
            previous.columns
        ):
            previous = previous[
                previous[
                    "pubchem_found"
                ].notna()
            ]

            for row in previous.itertuples(
                index=False
            ):
                key = getattr(
                    row,
                    "full_inchikey",
                    None,
                )

                if key:
                    lookup[
                        str(key)
                    ] = {
                        "pubchem_found":
                            bool(
                                row.pubchem_found
                            ),
                        "pubchem_canonical_smiles":
                            getattr(
                                row,
                                "pubchem_canonical_smiles",
                                None,
                            ),
                        "pubchem_isomeric_smiles":
                            getattr(
                                row,
                                "pubchem_isomeric_smiles",
                                None,
                            ),
                        "pubchem_molecular_formula":
                            getattr(
                                row,
                                "pubchem_molecular_formula",
                                None,
                            ),
                    }

            print(
                "Resuming cached lookups:",
                f"{len(lookup):,}",
            )

    keys = [
        str(key)
        for key in truth[
            "full_inchikey"
        ].dropna()
        if str(key)
        not in lookup
    ]

    print(
        "PubChem lookups required:",
        f"{len(keys):,}",
    )

    batches = [
        keys[
            start:
            start + BATCH_SIZE
        ]
        for start in range(
            0,
            len(keys),
            BATCH_SIZE,
        )
    ]

    for batch_number, batch in enumerate(
        batches,
        start=1,
    ):
        rows = fetch_pubchem_batch(
            batch
        )

        for key in batch:
            row = rows.get(
                key
            )

            lookup[
                key
            ] = {
                "pubchem_found":
                    row is not None,
                "pubchem_canonical_smiles":
                    (
                        row.get(
                            "ConnectivitySMILES"
                        )
                        or row.get(
                            "CanonicalSMILES"
                        )
                    )
                    if row
                    else None,
                "pubchem_isomeric_smiles":
                    (
                        row.get(
                            "SMILES"
                        )
                        or row.get(
                            "IsomericSMILES"
                        )
                    )
                    if row
                    else None,
                "pubchem_molecular_formula":
                    row.get(
                        "MolecularFormula"
                    )
                    if row
                    else None,
            }

        completed = (
            batch_number
            * BATCH_SIZE
        )

        print(
            "  processed "
            f"{min(completed, len(keys)):,}/"
            f"{len(keys):,} PubChem lookups"
        )

        # Save after every batch so the run is resumable.
        checkpoint = truth.copy()

        checkpoint[
            "pubchem_found"
        ] = [
            (
                lookup.get(
                    str(key),
                    {},
                ).get(
                    "pubchem_found"
                )
                if pd.notna(key)
                else False
            )
            for key in checkpoint[
                "full_inchikey"
            ]
        ]

        checkpoint[
            "pubchem_canonical_smiles"
        ] = [
            lookup.get(
                str(key),
                {},
            ).get(
                "pubchem_canonical_smiles"
            )
            if pd.notna(key)
            else None
            for key in checkpoint[
                "full_inchikey"
            ]
        ]

        checkpoint[
            "pubchem_isomeric_smiles"
        ] = [
            lookup.get(
                str(key),
                {},
            ).get(
                "pubchem_isomeric_smiles"
            )
            if pd.notna(key)
            else None
            for key in checkpoint[
                "full_inchikey"
            ]
        ]

        checkpoint[
            "pubchem_molecular_formula"
        ] = [
            lookup.get(
                str(key),
                {},
            ).get(
                "pubchem_molecular_formula"
            )
            if pd.notna(key)
            else None
            for key in checkpoint[
                "full_inchikey"
            ]
        ]

        checkpoint.to_parquet(
            OUTPUT_PATH,
            index=False,
        )

        time.sleep(
            MIN_REQUEST_INTERVAL_SECONDS
        )

    result = truth.copy()

    result[
        "pubchem_found"
    ] = [
        bool(
            lookup.get(
                str(key),
                {},
            ).get(
                "pubchem_found",
                False,
            )
        )
        if pd.notna(key)
        else False
        for key in result[
            "full_inchikey"
        ]
    ]

    result[
        "pubchem_molecular_formula"
    ] = [
        lookup.get(
            str(key),
            {},
        ).get(
            "pubchem_molecular_formula"
        )
        if pd.notna(key)
        else None
        for key in result[
            "full_inchikey"
        ]
    ]

    result[
        "pubchem_formula_matches"
    ] = (
        result[
            "pubchem_found"
        ]
        & (
            result[
                "molecular_formula"
            ].astype(str)
            == result[
                "pubchem_molecular_formula"
            ].astype(str)
        )
    )

    result.to_parquet(
        OUTPUT_PATH,
        index=False,
    )

    found = int(
        result[
            "pubchem_found"
        ].sum()
    )

    formula_matches = int(
        result[
            "pubchem_formula_matches"
        ].sum()
    )

    total = len(
        result
    )

    summary = {
        "validation_structures":
            total,
        "pubchem_found":
            found,
        "pubchem_coverage":
            (
                found / total
                if total
                else 0.0
            ),
        "pubchem_formula_matches":
            formula_matches,
        "pubchem_exact_formula_coverage":
            (
                formula_matches / total
                if total
                else 0.0
            ),
    }

    SUMMARY_PATH.write_text(
        json.dumps(
            summary,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    print()
    print(
        "PubChem strict-validation coverage"
    )
    print(
        "=================================="
    )
    print(
        "Validation structures:",
        f"{total:,}",
    )
    print(
        "Truth structure found in PubChem:",
        f"{found:,} / {total:,} "
        f"({found / total:.2%})",
    )
    print(
        "Truth structure + formula match:",
        f"{formula_matches:,} / {total:,} "
        f"({formula_matches / total:.2%})",
    )
    print()
    print(
        f"Wrote {OUTPUT_PATH}"
    )
    print(
        f"Wrote {SUMMARY_PATH}"
    )


if __name__ == "__main__":
    main()
