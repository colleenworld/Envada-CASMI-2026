from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class PubChemMassTier:
    mass: np.memmap
    offsets: np.memmap
    smiles_buffer: np.memmap

    @classmethod
    def load(
        cls,
        root: Path,
    ) -> "PubChemMassTier":
        mass_path = root / "pc_mass.npy"
        off_path = root / "pc_off.npy"
        smiles_path = root / "pc_smiles.npy"

        for path in [
            mass_path,
            off_path,
            smiles_path,
        ]:
            if not path.exists():
                raise FileNotFoundError(
                    f"Missing PubChem tier asset: {path}"
                )

        mass = np.load(
            mass_path,
            mmap_mode="r",
        )
        offsets = np.load(
            off_path,
            mmap_mode="r",
        )
        smiles_buffer = np.load(
            smiles_path,
            mmap_mode="r",
        )

        if mass.ndim != 1:
            raise ValueError(
                "pc_mass.npy must be one-dimensional"
            )

        if offsets.ndim != 1:
            raise ValueError(
                "pc_off.npy must be one-dimensional"
            )

        if smiles_buffer.ndim != 1:
            raise ValueError(
                "pc_smiles.npy must be one-dimensional"
            )

        if len(offsets) != len(mass) + 1:
            raise ValueError(
                "Expected len(pc_off) == len(pc_mass) + 1; "
                f"got {len(offsets):,} and {len(mass):,}"
            )

        return cls(
            mass=mass,
            offsets=offsets,
            smiles_buffer=smiles_buffer,
        )

    def window_bounds(
        self,
        neutral_mass: float,
        *,
        ppm: float = 10.0,
    ) -> tuple[int, int]:
        if (
            not np.isfinite(neutral_mass)
            or neutral_mass <= 0
        ):
            return 0, 0

        tolerance = (
            float(neutral_mass)
            * float(ppm)
            / 1_000_000.0
        )

        left = int(
            np.searchsorted(
                self.mass,
                neutral_mass - tolerance,
                side="left",
            )
        )

        right = int(
            np.searchsorted(
                self.mass,
                neutral_mass + tolerance,
                side="right",
            )
        )

        return left, right

    def window_size(
        self,
        neutral_mass: float,
        *,
        ppm: float = 10.0,
    ) -> int:
        left, right = self.window_bounds(
            neutral_mass,
            ppm=ppm,
        )

        return max(
            0,
            right - left,
        )

    def iter_smiles_window(
        self,
        neutral_mass: float,
        *,
        ppm: float = 10.0,
    ) -> Iterator[str]:
        left, right = self.window_bounds(
            neutral_mass,
            ppm=ppm,
        )

        if right <= left:
            return

        offsets = self.offsets[
            left:right + 1
        ]

        for index in range(
            right - left
        ):
            start = int(
                offsets[index]
            )
            end = int(
                offsets[index + 1]
            )

            yield bytes(
                self.smiles_buffer[
                    start:end
                ]
            ).decode(
                "ascii"
            )

    def smiles_window(
        self,
        neutral_mass: float,
        *,
        ppm: float = 10.0,
    ) -> list[str]:
        return list(
            self.iter_smiles_window(
                neutral_mass,
                ppm=ppm,
            )
        )
