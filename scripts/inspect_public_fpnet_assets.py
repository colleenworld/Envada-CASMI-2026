from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from casmi26.public_models.fpnet import (
    load_checkpoint,
)


ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)

ASSET_ROOT = (
    ROOT
    / "data/external/"
      "casmi26-public-assets"
)

POOL_ROOT = (
    ASSET_ROOT
    / "casmi26-v2-pool"
)

MODEL_PATH = (
    ASSET_ROOT
    / "casmi26-fpnet-full1/"
      "fpnet_full1.pt"
)


def describe_array(
    path: Path,
) -> None:
    array = np.load(
        path,
        mmap_mode="r",
    )

    print(
        path.name,
        "shape=",
        array.shape,
        "dtype=",
        array.dtype,
        "size=",
        f"{path.stat().st_size / 1e6:.1f} MB",
    )

    if array.size:
        flat = array.reshape(
            -1
        )

        print(
            "  min=",
            flat.min(),
            "max=",
            flat.max(),
        )


def main() -> None:
    for path in [
        MODEL_PATH,
        POOL_ROOT
        / "fp_bits.npy",
        POOL_ROOT
        / "pool_fp.npy",
        POOL_ROOT
        / "train_fp_sel.npy",
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"Missing public asset: {path}"
            )

    print(
        "Torch:",
        torch.__version__,
    )

    print()

    for filename in [
        "fp_bits.npy",
        "pool_fp.npy",
        "train_fp_sel.npy",
    ]:
        describe_array(
            POOL_ROOT
            / filename
        )

    print()

    model, checkpoint = (
        load_checkpoint(
            MODEL_PATH
        )
    )

    print(
        "FPNet checkpoint keys:",
        sorted(
            checkpoint.keys()
        ),
    )

    for key in [
        "nbits",
        "d",
        "layers",
    ]:
        print(
            f"{key}:",
            checkpoint.get(
                key
            ),
        )

    print(
        "model parameters:",
        f"{sum(parameter.numel() for parameter in model.parameters()):,}",
    )

    bits = np.load(
        POOL_ROOT
        / "fp_bits.npy"
    )

    pool_fp = np.load(
        POOL_ROOT
        / "pool_fp.npy",
        mmap_mode="r",
    )

    nbits = int(
        checkpoint[
            "nbits"
        ]
    )

    print()
    print(
        "Cross-checks"
    )
    print(
        "=" * 72
    )
    print(
        "checkpoint nbits == len(fp_bits):",
        nbits
        == len(
            bits
        ),
    )

    if pool_fp.ndim == 2:
        print(
            "pool rows:",
            f"{pool_fp.shape[0]:,}",
        )

        print(
            "pool fingerprint width:",
            pool_fp.shape[1],
        )

        print(
            "width equals nbits:",
            pool_fp.shape[1]
            == nbits,
        )

        print(
            "width equals packed nbits:",
            (
                pool_fp.shape[1]
                * 8
            )
            >= nbits,
        )


if __name__ == "__main__":
    main()
