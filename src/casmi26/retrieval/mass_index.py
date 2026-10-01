from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class MassIndexPartition:
    """
    Sorted neutral masses and their corresponding reference row IDs.
    """

    masses: np.ndarray
    row_ids: np.ndarray


@dataclass(frozen=True)
class NeutralMassIndex:
    """
    Neutral-mass lookup index partitioned by ionization mode.
    """

    partitions: dict[str, MassIndexPartition]


def build_neutral_mass_index(
    neutral_masses: Sequence[float],
    modes: Sequence[str],
) -> NeutralMassIndex:
    """
    Build a sorted neutral-mass index for each ionization mode.

    Rows whose neutral mass is not finite are excluded.
    """
    neutral_masses = np.asarray(
        neutral_masses,
        dtype=np.float64,
    )

    modes = np.asarray(
        modes,
        dtype=object,
    )

    if len(neutral_masses) != len(modes):
        raise ValueError(
            "neutral_masses and modes must have the same length"
        )

    partitions: dict[
        str,
        MassIndexPartition,
    ] = {}

    for mode in np.unique(modes):
        mode_mask = (
            modes == mode
        )

        row_ids = np.flatnonzero(
            mode_mask
            & np.isfinite(
                neutral_masses
            )
        )

        if len(row_ids) == 0:
            continue

        masses = neutral_masses[
            row_ids
        ]

        order = np.argsort(
            masses,
            kind="stable",
        )

        partitions[str(mode)] = (
            MassIndexPartition(
                masses=masses[
                    order
                ],
                row_ids=row_ids[
                    order
                ].astype(
                    np.int64,
                    copy=False,
                ),
            )
        )

    return NeutralMassIndex(
        partitions=partitions
    )


def search_neutral_mass(
    index: NeutralMassIndex,
    neutral_mass: float,
    mode: str,
    tolerance_da: float,
) -> np.ndarray:
    """
    Return reference row IDs within the requested neutral-mass
    tolerance and ionization mode.
    """
    if tolerance_da < 0:
        raise ValueError(
            "tolerance_da must be non-negative"
        )

    if not np.isfinite(
        neutral_mass
    ):
        return np.empty(
            0,
            dtype=np.int64,
        )

    partition = (
        index.partitions.get(
            str(mode)
        )
    )

    if partition is None:
        return np.empty(
            0,
            dtype=np.int64,
        )

    lower = (
        neutral_mass
        - tolerance_da
    )

    upper = (
        neutral_mass
        + tolerance_da
    )

    left = np.searchsorted(
        partition.masses,
        lower,
        side="left",
    )

    right = np.searchsorted(
        partition.masses,
        upper,
        side="right",
    )

    return partition.row_ids[
        left:right
    ]


def search_neutral_masses(
    index: NeutralMassIndex,
    neutral_masses: Sequence[float],
    modes: Sequence[str],
    tolerance_da: float,
) -> np.ndarray:
    """
    Return the union of reference row IDs matching any supplied
    neutral-mass/mode pair.
    """
    neutral_masses = np.asarray(
        neutral_masses,
        dtype=np.float64,
    )

    modes = np.asarray(
        modes,
        dtype=object,
    )

    if len(neutral_masses) != len(modes):
        raise ValueError(
            "neutral_masses and modes must have the same length"
        )

    matches: list[np.ndarray] = []

    for neutral_mass, mode in zip(
        neutral_masses,
        modes,
        strict=True,
    ):
        row_ids = search_neutral_mass(
            index=index,
            neutral_mass=float(
                neutral_mass
            ),
            mode=str(mode),
            tolerance_da=tolerance_da,
        )

        if len(row_ids):
            matches.append(
                row_ids
            )

    if not matches:
        return np.empty(
            0,
            dtype=np.int64,
        )

    return np.unique(
        np.concatenate(
            matches
        )
    )