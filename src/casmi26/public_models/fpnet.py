from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


FP_MAX_PEAKS = 128

ADDUCT_LIST = [
    "[M+H]+",
    "[M+NH4]+",
    "[M+Na]+",
    "[M+K]+",
    "[M-H2O+H]+",
    "[M-2H2O+H]+",
    "[M]+",
    "[M-H]-",
    "[M-H2O-H]-",
    "[M+CH2O2-H]-",
    "[M+C2H4O2-H]-",
    "[M+Cl]-",
    "[M]-",
    "[M+2H]2+",
    "[M-2H]-",
    "[2M+H]+",
    "[2M+Na]+",
    "[2M+NH4]+",
    "[2M-H]-",
    "[2M+K]+",
    "[2M+CH2O2-H]-",
    "[2M+C2H4O2-H]-",
    "[2M+Na-2H]-",
    "[M+Na-2H]-",
    "[M-H2O]+",
    "<unk>",
]

ADDUCT_INDEX = {
    adduct: index
    for index, adduct in enumerate(
        ADDUCT_LIST
    )
}

INSTRUMENT_LIST = [
    "timsTOF",
    "Orbitrap",
    "QTOF",
    "IT",
    "other",
]


def instrument_family(
    value,
) -> int:
    if value is None:
        return 4

    text = str(
        value
    ).lower()

    if "timstof" in text:
        return 0

    if any(
        token in text
        for token in [
            "orbitrap",
            "qft",
            "ftms",
            "hybrid ft",
            "itft",
            "exactive",
        ]
    ):
        return 1

    if "tof" in text:
        return 2

    if (
        "trap" in text
        or "qq" in text
    ):
        return 3

    return 4


def prep_peaks(
    mz,
    intensity,
    precursor_mz: float,
    *,
    max_peaks: int = FP_MAX_PEAKS,
    floor: float = 1e-3,
    window: float = 50.0,
    per_window: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    mz = np.asarray(
        mz,
        dtype=np.float64,
    )

    intensity = np.asarray(
        intensity,
        dtype=np.float64,
    )

    if (
        mz.size == 0
        or intensity.size == 0
    ):
        return (
            np.zeros(
                0,
                dtype=np.float32,
            ),
            np.zeros(
                0,
                dtype=np.float32,
            ),
        )

    keep = (
        mz
        <= float(
            precursor_mz
        )
        + 1.5
    )

    mz = mz[
        keep
    ]
    intensity = intensity[
        keep
    ]

    if mz.size == 0:
        return (
            np.zeros(
                0,
                dtype=np.float32,
            ),
            np.zeros(
                0,
                dtype=np.float32,
            ),
        )

    maximum = float(
        intensity.max()
    )

    if maximum <= 0:
        return (
            np.zeros(
                0,
                dtype=np.float32,
            ),
            np.zeros(
                0,
                dtype=np.float32,
            ),
        )

    keep = (
        intensity
        >= floor
        * maximum
    )

    mz = mz[
        keep
    ]
    intensity = intensity[
        keep
    ]

    if mz.size > max_peaks:
        order = np.argsort(
            -intensity
        )

        buckets = (
            mz
            // window
        ).astype(
            np.int64
        )

        counts: dict[
            int,
            int,
        ] = {}

        selected: list[
            int
        ] = []

        for index in order:
            bucket = int(
                buckets[
                    index
                ]
            )

            count = counts.get(
                bucket,
                0,
            )

            if count < per_window:
                counts[
                    bucket
                ] = (
                    count
                    + 1
                )

                selected.append(
                    int(
                        index
                    )
                )

        selected_array = np.asarray(
            selected,
            dtype=np.int64,
        )

        if (
            selected_array.size
            > max_peaks
        ):
            selected_array = (
                selected_array[
                    np.argsort(
                        -intensity[
                            selected_array
                        ]
                    )[
                        :max_peaks
                    ]
                ]
            )

        elif (
            selected_array.size
            < max_peaks
        ):
            selected_set = set(
                selected
            )

            rest = np.asarray(
                [
                    int(index)
                    for index in order
                    if int(index)
                    not in selected_set
                ],
                dtype=np.int64,
            )

            need = (
                max_peaks
                - selected_array.size
            )

            if rest.size:
                selected_array = (
                    np.concatenate(
                        [
                            selected_array,
                            rest[
                                :need
                            ],
                        ]
                    )
                )

        mz = mz[
            selected_array
        ]

        intensity = intensity[
            selected_array
        ]

    order = np.argsort(
        mz
    )

    mz = mz[
        order
    ]

    intensity = intensity[
        order
    ]

    scaled = np.sqrt(
        intensity
        / intensity.max()
    )

    return (
        mz.astype(
            np.float32
        ),
        scaled.astype(
            np.float32
        ),
    )


class SinEmbedding(
    nn.Module
):
    def __init__(
        self,
        dim: int,
        low: float = -2.0,
        high: float = 3.2,
        power: float = 1.0,
    ):
        super().__init__()

        count = (
            dim
            // 2
        )

        wavelength = torch.pow(
            torch.tensor(
                10.0
            ),
            (
                high
                - low
            )
            * torch.pow(
                torch.linspace(
                    0,
                    1,
                    count,
                ),
                power,
            )
            + low,
        )

        self.register_buffer(
            "inv",
            (
                2
                * math.pi
            )
            / wavelength,
        )

    def forward(
        self,
        value: torch.Tensor,
    ) -> torch.Tensor:
        angle = (
            value.unsqueeze(
                -1
            )
            * self.inv
        )

        return torch.cat(
            [
                torch.sin(
                    angle
                ),
                torch.cos(
                    angle
                ),
            ],
            dim=-1,
        )


class TransformerBlock(
    nn.Module
):
    def __init__(
        self,
        dim: int,
        heads: int,
        dropout: float,
    ):
        super().__init__()

        self.heads = heads

        self.norm1 = nn.LayerNorm(
            dim
        )

        self.qkv = nn.Linear(
            dim,
            3
            * dim,
        )

        self.output = nn.Linear(
            dim,
            dim,
        )

        self.norm2 = nn.LayerNorm(
            dim
        )

        self.feed_forward = nn.Sequential(
            nn.Linear(
                dim,
                4
                * dim,
            ),
            nn.GELU(),
            nn.Dropout(
                dropout
            ),
            nn.Linear(
                4
                * dim,
                dim,
            ),
        )

        self.dropout = nn.Dropout(
            dropout
        )

    def forward(
        self,
        value: torch.Tensor,
        padding: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, length, dim = (
            value.shape
        )

        normalized = self.norm1(
            value
        )

        query, key, item = (
            self.qkv(
                normalized
            )
            .view(
                batch_size,
                length,
                3,
                self.heads,
                dim
                // self.heads,
            )
            .permute(
                2,
                0,
                3,
                1,
                4,
            )
        )

        mask = (
            ~padding
        )[
            :,
            None,
            None,
            :,
        ]

        attention = (
            F.scaled_dot_product_attention(
                query,
                key,
                item,
                attn_mask=mask,
            )
        )

        value = (
            value
            + self.dropout(
                self.output(
                    attention
                    .transpose(
                        1,
                        2,
                    )
                    .reshape(
                        batch_size,
                        length,
                        dim,
                    )
                )
            )
        )

        return (
            value
            + self.dropout(
                self.feed_forward(
                    self.norm2(
                        value
                    )
                )
            )
        )


class FPNet(
    nn.Module
):
    def __init__(
        self,
        nbits: int,
        *,
        dim: int = 512,
        layers: int = 6,
        heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.dim = dim

        self.mz_embedding = (
            SinEmbedding(
                dim
            )
        )

        self.neutral_loss_embedding = (
            SinEmbedding(
                dim
            )
        )

        self.peak_projection = (
            nn.Linear(
                (
                    2
                    * dim
                )
                + 1,
                dim,
            )
        )

        self.precursor_embedding = (
            SinEmbedding(
                dim
            )
        )

        self.adduct_embedding = (
            nn.Embedding(
                len(
                    ADDUCT_LIST
                ),
                dim,
            )
        )

        self.instrument_embedding = (
            nn.Embedding(
                len(
                    INSTRUMENT_LIST
                ),
                dim,
            )
        )

        self.global_projection = (
            nn.Linear(
                dim
                + 3,
                dim,
            )
        )

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    dim,
                    heads,
                    dropout,
                )
                for _ in range(
                    layers
                )
            ]
        )

        self.norm = nn.LayerNorm(
            dim
        )

        self.head = nn.Sequential(
            nn.Linear(
                2
                * dim,
                2048,
            ),
            nn.GELU(),
            nn.Dropout(
                dropout
            ),
            nn.Linear(
                2048,
                nbits,
            ),
        )

    def forward(
        self,
        mz: torch.Tensor,
        intensity: torch.Tensor,
        padding: torch.Tensor,
        precursor_mz: torch.Tensor,
        adduct: torch.Tensor,
        instrument: torch.Tensor,
        collision_energy: torch.Tensor,
        mode: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, peak_count = (
            mz.shape
        )

        neutral_loss = (
            precursor_mz[
                :,
                None,
            ]
            - mz
        ).clamp(
            min=0
        )

        peak = (
            self.peak_projection(
                torch.cat(
                    [
                        self.mz_embedding(
                            mz
                        ),
                        self.neutral_loss_embedding(
                            neutral_loss
                        ),
                        intensity.unsqueeze(
                            -1
                        ),
                    ],
                    dim=-1,
                )
            )
        )

        global_token = (
            self.global_projection(
                torch.cat(
                    [
                        self.precursor_embedding(
                            precursor_mz
                        ),
                        (
                            collision_energy
                            / 100.0
                        ).unsqueeze(
                            -1
                        ),
                        mode.unsqueeze(
                            -1
                        ),
                        (
                            torch.log1p(
                                precursor_mz
                            )
                            .unsqueeze(
                                -1
                            )
                            / 10.0
                        ),
                    ],
                    dim=-1,
                )
            )
            + self.adduct_embedding(
                adduct
            )
            + self.instrument_embedding(
                instrument
            )
        )

        value = torch.cat(
            [
                global_token.unsqueeze(
                    1
                ),
                peak,
            ],
            dim=1,
        )

        padding = torch.cat(
            [
                torch.zeros(
                    batch_size,
                    1,
                    dtype=torch.bool,
                    device=padding.device,
                ),
                padding,
            ],
            dim=1,
        )

        for block in self.blocks:
            value = block(
                value,
                padding,
            )

        value = self.norm(
            value
        )

        cls = value[
            :,
            0,
        ]

        mask = (
            ~padding[
                :,
                1:,
            ]
        ).float().unsqueeze(
            -1
        )

        mean = (
            (
                value[
                    :,
                    1:,
                ]
                * mask
            ).sum(
                dim=1
            )
            / mask.sum(
                dim=1
            ).clamp(
                min=1
            )
        )

        return self.head(
            torch.cat(
                [
                    cls,
                    mean,
                ],
                dim=-1,
            )
        )


def load_checkpoint(
    path: Path,
    *,
    device: str | torch.device = "cpu",
) -> tuple[
    FPNet,
    dict,
]:
    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )

    required = {
        "nbits",
        "d",
        "layers",
        "model",
    }

    missing = (
        required
        - set(
            checkpoint
        )
    )

    if missing:
        raise ValueError(
            "Unexpected FPNet checkpoint; "
            f"missing keys: {sorted(missing)}"
        )

    model = FPNet(
        int(
            checkpoint[
                "nbits"
            ]
        ),
        dim=int(
            checkpoint[
                "d"
            ]
        ),
        layers=int(
            checkpoint[
                "layers"
            ]
        ),
    )

    model.load_state_dict(
        checkpoint[
            "model"
        ]
    )

    model = model.to(
        device
    ).eval()

    return (
        model,
        checkpoint,
    )
