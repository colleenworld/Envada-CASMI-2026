from __future__ import annotations

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
        mass_path = root / "pubchem_tier__pc_mass.npy"
        off_path = root / "pubchem_tier__pc_off.npy"
        smiles_path = root / "pubchem_tier__pc_smiles.npy"

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

    def smiles_window(
        self,
        neutral_mass: float,
        *,
        ppm: float = 10.0,
    ) -> list[str]:
        left, right = self.window_bounds(
            neutral_mass,
            ppm=ppm,
        )

        if right <= left:
            return []

        offsets = np.asarray(
            self.offsets[
                left:right + 1
            ],
            dtype=np.int64,
        )

        start = int(
            offsets[0]
        )
        end = int(
            offsets[-1]
        )

        raw = bytes(
            self.smiles_buffer[
                start:end
            ]
        )

        base = start

        return [
            raw[
                int(offsets[i]) - base:
                int(offsets[i + 1]) - base
            ].decode(
                "ascii"
            )
            for i in range(
                right - left
            )
        ]
