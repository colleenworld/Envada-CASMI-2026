from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class StructureIndex:
    inchikey14: np.ndarray
    smiles: np.ndarray
    formulas: np.ndarray
    masses: np.ndarray

    def __len__(self) -> int:
        return len(self.inchikey14)


def build_structure_index(
    inchikey14: Sequence[str],
    smiles: Sequence[str],
    formulas: Sequence[str],
    masses: Sequence[float],
) -> StructureIndex:
    lengths = {
        len(inchikey14),
        len(smiles),
        len(formulas),
        len(masses),
    }

    if len(lengths) != 1:
        raise ValueError(
            "All structure index fields must have equal length"
        )

    #
    # One record per structure.
    #
    # If the same InChIKey14 occurs more than once, require
    # its structure metadata to be internally consistent.
    #

    structures: dict[
        str,
        tuple[str, str, float],
    ] = {}

    for (
        structure_key,
        structure_smiles,
        formula,
        mass,
    ) in zip(
        inchikey14,
        smiles,
        formulas,
        masses,
        strict=True,
    ):
        mass = float(mass)

        if not np.isfinite(mass):
            continue

        record = (
            str(structure_smiles),
            str(formula),
            mass,
        )

        existing = structures.get(
            str(structure_key)
        )

        if existing is not None:
            if (
                existing[0] != record[0]
                or existing[1] != record[1]
                or not np.isclose(
                    existing[2],
                    record[2],
                    rtol=0.0,
                    atol=1e-9,
                )
            ):
                raise ValueError(
                    "Inconsistent metadata for structure "
                    f"{structure_key!r}"
                )

            continue

        structures[str(structure_key)] = record

    #
    # Sorting by formula first makes structures belonging
    # to the same molecular formula contiguous.
    #

    ordered = sorted(
        structures.items(),
        key=lambda item: (
            item[1][1],
            item[0],
        ),
    )

    return StructureIndex(
        inchikey14=np.asarray(
            [
                structure_key
                for structure_key, _ in ordered
            ],
            dtype=object,
        ),
        smiles=np.asarray(
            [
                record[0]
                for _, record in ordered
            ],
            dtype=object,
        ),
        formulas=np.asarray(
            [
                record[1]
                for _, record in ordered
            ],
            dtype=object,
        ),
        masses=np.asarray(
            [
                record[2]
                for _, record in ordered
            ],
            dtype=np.float64,
        ),
    )


def structures_for_formulas(
    index: StructureIndex,
    formulas: Iterable[str],
) -> np.ndarray:
    requested = set(formulas)

    if not requested:
        return np.empty(
            0,
            dtype=np.int64,
        )

    mask = np.fromiter(
        (
            formula in requested
            for formula in index.formulas
        ),
        dtype=bool,
        count=len(index),
    )

    return np.flatnonzero(mask)