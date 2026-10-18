from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from urllib import error, parse, request

import numpy as np
import pandas as pd

from casmi26.pubchem.formula_candidates import (
    CANDIDATE_COLUMNS,
    candidate_burden,
    parse_formula_properties,
)


PROJECT_ROOT = (
    Path(__file__).resolve().parent.parent
)

VALIDATION_COVERAGE_PATH = (
    PROJECT_ROOT
    / "data/processed/diagnostics/"
    "pubchem_validation_coverage.parquet"
)

FORMULA_INDEX_PATH = (
    PROJECT_ROOT
    / "data/processed/formula_index/"
    "training_coconut_formulas.parquet"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "data/processed/diagnostics"
)

CACHE_ROOT = (
    PROJECT_ROOT
    / "data/external/pubchem/formula_cache"
)

MIN_REQUEST_INTERVAL_SECONDS = 0.30

MAX_RETRIES = 6

RETRYABLE_STATUS_CODES = {
    429,
    500,
    502,
    503,
    504,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Build a resumable PubChem exact-formula candidate "
            "catalog and report candidate burden."
        )
    )

    parser.add_argument(
        "--source",
        choices=[
            "validation",
            "formula-index",
        ],
        default="validation",
        help=(
            "validation: formulas in the strict validation truth set; "
            "formula-index: every formula in the current inference index"
        ),
    )

    parser.add_argument(
        "--max-records",
        type=int,
        default=5000,
        help=(
            "Maximum PubChem records requested per formula. "
            "Formulas that reach this limit are explicitly flagged."
        ),
    )

    return parser.parse_args()


def load_formula_set(
    source: str,
) -> list[str]:
    if source == "validation":
        frame = pd.read_parquet(
            VALIDATION_COVERAGE_PATH,
            columns=[
                "molecular_formula",
            ],
        )

    else:
        frame = pd.read_parquet(
            FORMULA_INDEX_PATH
        )

        if (
            "molecular_formula"
            not in frame.columns
        ):
            raise ValueError(
                "Formula index does not contain molecular_formula"
            )

    formulas = sorted(
        set(
            frame[
                "molecular_formula"
            ]
            .dropna()
            .astype(str)
        )
    )

    if not formulas:
        raise RuntimeError(
            "No formulas were found"
        )

    return formulas


def cache_path_for_formula(
    source: str,
    molecular_formula: str,
    max_records: int,
) -> Path:
    digest = hashlib.sha256(
        molecular_formula.encode(
            "utf-8"
        )
    ).hexdigest()[:20]

    return (
        CACHE_ROOT
        / source
        / f"max-{max_records}"
        / f"{digest}.json"
    )


def fetch_formula(
    molecular_formula: str,
    max_records: int,
) -> dict:
    encoded_formula = parse.quote(
        molecular_formula,
        safe="",
    )

    url = (
        "https://pubchem.ncbi.nlm.nih.gov/rest/pug/"
        f"compound/fastformula/{encoded_formula}/property/"
        "InChIKey,ConnectivitySMILES,SMILES,MolecularFormula/JSON"
        f"?MaxRecords={max_records}"
    )

    last_error = None

    for attempt in range(
        1,
        MAX_RETRIES + 1,
    ):
        req = request.Request(
            url,
            headers={
                "User-Agent":
                    "CASMI26-formula-candidate-diagnostic/1.0",
            },
            method="GET",
        )

        try:
            with request.urlopen(
                req,
                timeout=120,
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

            return {
                "molecular_formula":
                    molecular_formula,
                "properties":
                    properties,
                "raw_record_count":
                    len(properties),
                "hit_max_records":
                    len(properties)
                    >= max_records,
            }

        except error.HTTPError as exc:
            last_error = exc

            if exc.code == 404:
                return {
                    "molecular_formula":
                        molecular_formula,
                    "properties":
                        [],
                    "raw_record_count":
                        0,
                    "hit_max_records":
                        False,
                }

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
        "PubChem formula request failed after retries: "
        f"{molecular_formula}"
    ) from last_error


def load_or_fetch_formula(
    source: str,
    molecular_formula: str,
    max_records: int,
) -> tuple[dict, bool]:
    cache_path = cache_path_for_formula(
        source,
        molecular_formula,
        max_records,
    )

    if cache_path.exists():
        return (
            json.loads(
                cache_path.read_text()
            ),
            True,
        )

    result = fetch_formula(
        molecular_formula,
        max_records,
    )

    cache_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    cache_path.write_text(
        json.dumps(
            result,
            separators=(
                ",",
                ":",
            ),
        )
        + "\n"
    )

    return result, False


def percentile(
    values: pd.Series,
    q: float,
) -> float:
    if values.empty:
        return 0.0

    return float(
        np.quantile(
            values.to_numpy(
                dtype=np.float64
            ),
            q,
        )
    )


def main() -> None:
    args = parse_args()

    if args.max_records <= 0:
        raise ValueError(
            "--max-records must be positive"
        )

    formulas = load_formula_set(
        args.source
    )

    print(
        "Formula source:",
        args.source,
    )
    print(
        "Unique formulas:",
        f"{len(formulas):,}",
    )
    print(
        "Max records/formula:",
        f"{args.max_records:,}",
    )

    candidate_frames = []
    status_rows = []

    for number, formula in enumerate(
        formulas,
        start=1,
    ):
        result, cached = (
            load_or_fetch_formula(
                args.source,
                formula,
                args.max_records,
            )
        )

        parsed = parse_formula_properties(
            formula,
            result.get(
                "properties",
                [],
            ),
        )

        if not parsed.empty:
            candidate_frames.append(
                parsed
            )

        status_rows.append(
            {
                "molecular_formula":
                    formula,
                "raw_record_count":
                    int(
                        result.get(
                            "raw_record_count",
                            0,
                        )
                    ),
                "candidate_count":
                    len(parsed),
                "hit_max_records":
                    bool(
                        result.get(
                            "hit_max_records",
                            False,
                        )
                    ),
            }
        )

        marker = (
            "cached"
            if cached
            else "fetched"
        )

        print(
            f"  {number:,}/{len(formulas):,} "
            f"{formula}: "
            f"{len(parsed):,} unique structures "
            f"({marker})"
        )

        if not cached:
            time.sleep(
                MIN_REQUEST_INTERVAL_SECONDS
            )

    if candidate_frames:
        candidates = pd.concat(
            candidate_frames,
            ignore_index=True,
        )

        candidates = (
            candidates
            .drop_duplicates(
                subset=[
                    "molecular_formula",
                    "inchikey14",
                ]
            )
            .sort_values(
                [
                    "molecular_formula",
                    "inchikey14",
                ]
            )
            .reset_index(
                drop=True
            )
        )

    else:
        candidates = pd.DataFrame(
            columns=CANDIDATE_COLUMNS
        )

    status = pd.DataFrame(
        status_rows
    )

    burden = candidate_burden(
        candidates,
        formulas,
    )

    burden = burden.merge(
        status[
            [
                "molecular_formula",
                "raw_record_count",
                "hit_max_records",
            ]
        ],
        on="molecular_formula",
        how="left",
        validate="one_to_one",
    )

    prefix = (
        "pubchem_validation_formula_candidates"
        if args.source
        == "validation"
        else
        "pubchem_formula_index_candidates"
    )

    candidate_path = (
        OUTPUT_ROOT
        / f"{prefix}.parquet"
    )

    burden_path = (
        OUTPUT_ROOT
        / f"{prefix}_burden.parquet"
    )

    summary_path = (
        OUTPUT_ROOT
        / f"{prefix}_summary.json"
    )

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    candidates.to_parquet(
        candidate_path,
        index=False,
    )

    burden.to_parquet(
        burden_path,
        index=False,
    )

    counts = burden[
        "candidate_count"
    ]

    capped_count = int(
        burden[
            "hit_max_records"
        ].sum()
    )

    summary = {
        "source":
            args.source,
        "formula_count":
            len(formulas),
        "candidate_rows":
            len(candidates),
        "unique_candidate_structures":
            int(
                candidates[
                    "inchikey14"
                ].nunique()
            )
            if not candidates.empty
            else 0,
        "max_records":
            args.max_records,
        "formulas_hitting_max_records":
            capped_count,
        "candidate_count_mean":
            float(
                counts.mean()
            ),
        "candidate_count_median":
            float(
                counts.median()
            ),
        "candidate_count_p90":
            percentile(
                counts,
                0.90,
            ),
        "candidate_count_p95":
            percentile(
                counts,
                0.95,
            ),
        "candidate_count_p99":
            percentile(
                counts,
                0.99,
            ),
        "candidate_count_max_observed":
            int(
                counts.max()
            ),
    }

    if args.source == "validation":
        truth = pd.read_parquet(
            VALIDATION_COVERAGE_PATH,
            columns=[
                "inchikey14",
                "molecular_formula",
            ],
        )

        truth[
            "inchikey14"
        ] = (
            truth[
                "inchikey14"
            ].astype(str)
        )

        truth_pairs = set(
            zip(
                truth[
                    "molecular_formula"
                ].astype(str),
                truth[
                    "inchikey14"
                ],
                strict=True,
            )
        )

        candidate_pairs = set(
            zip(
                candidates[
                    "molecular_formula"
                ].astype(str),
                candidates[
                    "inchikey14"
                ].astype(str),
                strict=True,
            )
        )

        covered_pairs = (
            truth_pairs
            & candidate_pairs
        )

        truth[
            "truth_candidate_present"
        ] = [
            (
                str(formula),
                str(inchikey14),
            )
            in candidate_pairs
            for formula, inchikey14
            in zip(
                truth[
                    "molecular_formula"
                ],
                truth[
                    "inchikey14"
                ],
                strict=True,
            )
        ]

        truth = truth.merge(
            burden[
                [
                    "molecular_formula",
                    "candidate_count",
                    "hit_max_records",
                ]
            ],
            on="molecular_formula",
            how="left",
            validate="many_to_one",
        )

        total_truth = len(
            truth
        )

        covered_truth = int(
            truth[
                "truth_candidate_present"
            ].sum()
        )

        uncapped_truth = truth.loc[
            ~truth[
                "hit_max_records"
            ].fillna(False)
        ]

        uncapped_covered = int(
            uncapped_truth[
                "truth_candidate_present"
            ].sum()
        )

        summary.update(
            {
                "validation_structures":
                    total_truth,
                "truth_structures_covered":
                    covered_truth,
                "truth_structure_coverage":
                    (
                        covered_truth
                        / total_truth
                        if total_truth
                        else 0.0
                    ),
                "validation_structures_on_uncapped_formulas":
                    len(
                        uncapped_truth
                    ),
                "truth_structures_covered_on_uncapped_formulas":
                    uncapped_covered,
                "truth_coverage_on_uncapped_formulas":
                    (
                        uncapped_covered
                        / len(
                            uncapped_truth
                        )
                        if len(
                            uncapped_truth
                        )
                        else 0.0
                    ),
            }
        )

        for threshold in [
            25,
            100,
            500,
            1000,
            5000,
        ]:
            within = truth[
                truth[
                    "candidate_count"
                ]
                <= threshold
            ]

            summary[
                f"validation_structures_with_candidate_count_le_{threshold}"
            ] = len(
                within
            )

            summary[
                f"covered_validation_structures_with_candidate_count_le_{threshold}"
            ] = int(
                within[
                    "truth_candidate_present"
                ].sum()
            )

    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    print()
    print(
        "PubChem formula candidate burden"
    )
    print(
        "================================"
    )
    print(
        "Formulas:",
        f"{len(formulas):,}",
    )
    print(
        "Candidate rows:",
        f"{len(candidates):,}",
    )
    print(
        "Unique candidate structures:",
        f"{summary['unique_candidate_structures']:,}",
    )
    print(
        "Formulas hitting MaxRecords:",
        f"{capped_count:,}",
    )
    print()
    print(
        "Candidates/formula mean:",
        f"{summary['candidate_count_mean']:.1f}",
    )
    print(
        "Candidates/formula median:",
        f"{summary['candidate_count_median']:.1f}",
    )
    print(
        "Candidates/formula p90:",
        f"{summary['candidate_count_p90']:.1f}",
    )
    print(
        "Candidates/formula p95:",
        f"{summary['candidate_count_p95']:.1f}",
    )
    print(
        "Candidates/formula p99:",
        f"{summary['candidate_count_p99']:.1f}",
    )
    print(
        "Candidates/formula max observed:",
        f"{summary['candidate_count_max_observed']:,}",
    )

    if args.source == "validation":
        print()
        print(
            "Truth structures covered:",
            f"{summary['truth_structures_covered']:,} / "
            f"{summary['validation_structures']:,} "
            f"({summary['truth_structure_coverage']:.2%})",
        )
        print(
            "Truth coverage on uncapped formulas:",
            f"{summary['truth_coverage_on_uncapped_formulas']:.2%}",
        )

        print()
        print(
            "Validation structures by candidate burden:"
        )

        for threshold in [
            25,
            100,
            500,
            1000,
            5000,
        ]:
            total_key = (
                "validation_structures_with_"
                f"candidate_count_le_{threshold}"
            )
            covered_key = (
                "covered_validation_structures_with_"
                f"candidate_count_le_{threshold}"
            )

            print(
                f"  <= {threshold:>4}: "
                f"{summary[total_key]:,} structures, "
                f"{summary[covered_key]:,} truths covered"
            )

    print()
    print(
        f"Wrote {candidate_path}"
    )
    print(
        f"Wrote {burden_path}"
    )
    print(
        f"Wrote {summary_path}"
    )


if __name__ == "__main__":
    main()
