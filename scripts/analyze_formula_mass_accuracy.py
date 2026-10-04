from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from casmi26.chemistry.adducts import neutral_mass
from casmi26.chemistry.formula import (
    mass_error_da,
    mass_error_ppm,
    monoisotopic_mass,
)


TRAIN_PATH = Path("data/raw/train.parquet")

COLUMNS = [
    "molecular_formula",
    "ionization_mode",
    "adduct",
    "precursor_mz",
    "precursor_error_ppm",
]


def percentile(values: list[float], p: float) -> float:
    return float(np.percentile(values, p))


def print_distribution(
    name: str,
    values: list[float],
) -> None:
    if not values:
        print(f"{name}: no values")
        return

    arr = np.asarray(values, dtype=np.float64)

    print(f"{name}:")
    print(f"  n       {len(arr):,}")
    print(f"  median  {np.median(arr):.6f}")
    print(f"  p90     {np.percentile(arr, 90):.6f}")
    print(f"  p95     {np.percentile(arr, 95):.6f}")
    print(f"  p99     {np.percentile(arr, 99):.6f}")
    print(f"  p99.9   {np.percentile(arr, 99.9):.6f}")
    print(f"  max     {np.max(arr):.6f}")


def main() -> None:
    parquet = pq.ParquetFile(TRAIN_PATH)

    total_rows = 0
    usable_rows = 0

    missing_formula = 0
    invalid_formula = 0
    missing_precursor = 0
    unsupported_adduct = 0

    unsupported_formulas: Counter[str] = Counter()
    unsupported_adducts: Counter[str] = Counter()

    abs_error_da: list[float] = []
    abs_error_ppm: list[float] = []
    supplied_abs_error_ppm: list[float] = []
    supplied_difference_ppm: list[float] = []

    by_adduct: dict[str, list[float]] = defaultdict(list)
    by_mode: dict[str, list[float]] = defaultdict(list)

    for batch in parquet.iter_batches(
        columns=COLUMNS,
        batch_size=100_000,
    ):
        data = batch.to_pydict()

        row_count = len(data["molecular_formula"])
        total_rows += row_count

        for i in range(row_count):
            formula = data["molecular_formula"][i]
            mode = data["ionization_mode"][i]
            adduct = data["adduct"][i]
            precursor_mz = data["precursor_mz"][i]
            supplied_ppm = data["precursor_error_ppm"][i]

            if not formula:
                missing_formula += 1
                continue

            if precursor_mz is None or not np.isfinite(precursor_mz):
                missing_precursor += 1
                continue

            try:
                theoretical_mass = monoisotopic_mass(formula)
            except ValueError:
                invalid_formula += 1
                unsupported_formulas[str(formula)] += 1
                continue

            inferred_mass = neutral_mass(
                float(precursor_mz),
                str(adduct),
            )

            if inferred_mass is None:
                unsupported_adduct += 1
                unsupported_adducts[str(adduct)] += 1
                continue

            error_da = mass_error_da(
                inferred_mass,
                theoretical_mass,
            )

            error_ppm = mass_error_ppm(
                inferred_mass,
                theoretical_mass,
            )

            abs_da = abs(error_da)
            abs_ppm = abs(error_ppm)

            abs_error_da.append(abs_da)
            abs_error_ppm.append(abs_ppm)

            by_adduct[str(adduct)].append(abs_ppm)
            by_mode[str(mode)].append(abs_ppm)

            if supplied_ppm is not None and np.isfinite(supplied_ppm):
                supplied_abs_error_ppm.append(
                    abs(float(supplied_ppm))
                )

                # We don't assume the sign convention of the supplied
                # value is the same as ours. Compare magnitudes.
                supplied_difference_ppm.append(
                    abs(
                        abs(error_ppm)
                        - abs(float(supplied_ppm))
                    )
                )

            usable_rows += 1

    print()
    print("Formula / precursor mass validation")
    print("=" * 40)
    print(f"Rows:                {total_rows:,}")
    print(f"Usable:              {usable_rows:,}")
    print(f"Missing formula:     {missing_formula:,}")
    print(f"Invalid formula:     {invalid_formula:,}")
    print(f"Missing precursor:   {missing_precursor:,}")
    print(f"Unsupported adduct:  {unsupported_adduct:,}")
    print()

    print_distribution(
        "Absolute neutral-mass error (Da)",
        abs_error_da,
    )
    print()

    print_distribution(
        "Absolute neutral-mass error (ppm)",
        abs_error_ppm,
    )
    print()

    print("Retention by ppm tolerance:")
    for tolerance in [
        1,
        2,
        5,
        10,
        20,
        50,
        100,
    ]:
        retained = sum(
            error <= tolerance
            for error in abs_error_ppm
        )

        fraction = (
            retained / len(abs_error_ppm)
            if abs_error_ppm
            else 0.0
        )

        print(
            f"  <= {tolerance:3d} ppm: "
            f"{retained:>9,} "
            f"({fraction:7.3%})"
        )

    print()
    print("By ionization mode:")

    for mode in sorted(by_mode):
        values = np.asarray(
            by_mode[mode],
            dtype=np.float64,
        )

        print(
            f"  {mode:10s} "
            f"n={len(values):>9,} "
            f"median={np.median(values):9.4f} ppm "
            f"p95={np.percentile(values, 95):9.4f} "
            f"p99={np.percentile(values, 99):9.4f}"
        )

    print()
    print("By adduct:")

    for adduct, values_list in sorted(
        by_adduct.items(),
        key=lambda item: -len(item[1]),
    ):
        values = np.asarray(
            values_list,
            dtype=np.float64,
        )

        print(
            f"  {adduct:20s} "
            f"n={len(values):>9,} "
            f"median={np.median(values):9.4f} ppm "
            f"p95={np.percentile(values, 95):9.4f} "
            f"p99={np.percentile(values, 99):9.4f}"
        )

    if supplied_abs_error_ppm:
        print()

        print_distribution(
            "Provided precursor_error_ppm (absolute)",
            supplied_abs_error_ppm,
        )

        print()

        print_distribution(
            "Difference between calculated and provided "
            "absolute ppm error",
            supplied_difference_ppm,
        )

    if unsupported_adducts:
        print()
        print("Unsupported adducts:")

        for adduct, count in unsupported_adducts.most_common():
            print(
                f"  {adduct:30s} {count:>9,}"
            )

    if unsupported_formulas:
        print()
        print("Most common unsupported formulas:")

        for formula, count in unsupported_formulas.most_common(20):
            print(
                f"  {formula:30s} {count:>9,}"
            )


if __name__ == "__main__":
    main()