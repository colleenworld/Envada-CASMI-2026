from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class FormulaIndex:
    formulas: np.ndarray
    masses: np.ndarray


def build_formula_index(
    formulas: Sequence[str],
    masses: Sequence[float],
) -> FormulaIndex:
    if len(formulas) != len(masses):
        raise ValueError(
            "formulas and masses must have equal length"
        )

    unique: dict[str, float] = {}

    for formula, mass in zip(
        formulas,
        masses,
        strict=True,
    ):
        mass = float(mass)

        if not np.isfinite(mass):
            continue

        existing = unique.get(formula)

        if existing is not None:
            if not np.isclose(
                existing,
                mass,
                rtol=0.0,
                atol=1e-9,
            ):
                raise ValueError(
                    f"Inconsistent mass for formula {formula!r}"
                )
            continue

        unique[formula] = mass

    ordered = sorted(
        unique.items(),
        key=lambda item: (
            item[1],
            item[0],
        ),
    )

    return FormulaIndex(
        formulas=np.asarray(
            [formula for formula, _ in ordered],
            dtype=object,
        ),
        masses=np.asarray(
            [mass for _, mass in ordered],
            dtype=np.float64,
        ),
    )


def search_formula_index(
    index: FormulaIndex,
    neutral_mass: float,
    tolerance_ppm: float,
) -> np.ndarray:
    if tolerance_ppm < 0:
        raise ValueError(
            "tolerance_ppm must be non-negative"
        )

    if not np.isfinite(neutral_mass):
        return np.empty(0, dtype=np.int64)

    tolerance_da = (
        neutral_mass
        * tolerance_ppm
        / 1_000_000.0
    )

    lower = neutral_mass - tolerance_da
    upper = neutral_mass + tolerance_da

    start = np.searchsorted(
        index.masses,
        lower,
        side="left",
    )

    end = np.searchsorted(
        index.masses,
        upper,
        side="right",
    )

    return np.arange(
        start,
        end,
        dtype=np.int64,
    )