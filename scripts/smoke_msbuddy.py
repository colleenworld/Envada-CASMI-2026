from __future__ import annotations

import sys

from msbuddy import Msbuddy


def main() -> None:
    print("Python:", sys.version)
    print("Creating MSBuddy engine...")

    engine = Msbuddy()

    print("Enumerating formulas for m/z 300.0, [M+H]+...")

    formulas = engine.mz_to_formula(
        300.0,
        "[M+H]+",
        10.0,
        True,
        True,
        0.0,
        True,
    )

    print("Formula candidates:", len(formulas))

    for item in formulas[:5]:
        print(
            item.formula,
            item.mass_error,
            item.mass_error_ppm,
        )

    if not formulas:
        raise RuntimeError(
            "MSBuddy imported but returned no formula candidates"
        )

    print("MSBuddy smoke test OK")


if __name__ == "__main__":
    main()
