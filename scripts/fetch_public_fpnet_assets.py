from __future__ import annotations

from pathlib import Path


DATASET = (
    "dmitriigluzdov/"
    "casmi26-attributed-v4g-reproduction-assets"
)

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)

OUTPUT = (
    PROJECT_ROOT
    / "data/external/"
      "casmi26-public-assets"
)

ASSETS = [
    "casmi26-fpnet-full1/fpnet_full1.pt",
    "casmi26-v2-pool/fp_bits.npy",
    "casmi26-v2-pool/pool_fp.npy",
    "casmi26-v2-pool/train_fp_sel.npy",
    "casmi26-v4b-models",
]


def main() -> None:
    try:
        import kagglehub
    except ImportError as error:
        raise RuntimeError(
            "kagglehub is required. Install it with: "
            "uv add kagglehub"
        ) from error

    OUTPUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    for asset in ASSETS:
        destination = (
            OUTPUT
            / asset
        )

        if destination.exists():
            print(
                "Already present:",
                destination,
            )
            continue

        print(
            "Fetching:",
            asset,
        )

        try:
            path = kagglehub.dataset_download(
                DATASET,
                path=asset,
                output_dir=str(
                    OUTPUT
                ),
            )
        except FileExistsError:
            # Directory-style assets may already have been created by a
            # previous partial or successful download. Treat an existing
            # destination as reusable and let the inspection step verify it.
            if destination.exists():
                print(
                    "Already present:",
                    destination,
                )
                continue
            raise

        print(
            " ->",
            path,
        )


if __name__ == "__main__":
    main()
