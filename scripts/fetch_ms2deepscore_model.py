from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path


URL = (
    "https://zenodo.org/records/16962640/files/"
    "ms2deepscore_model.pt?download=1"
)

OUTPUT = Path(
    "data/external/ms2deepscore/ms2deepscore_model.pt"
)

EXPECTED_MD5 = (
    "26502fb9b110f457e15b395a23c578ba"
)


def md5sum(
    path: Path,
) -> str:
    digest = hashlib.md5()

    with path.open(
        "rb"
    ) as handle:
        while True:
            chunk = handle.read(
                1024 * 1024
            )

            if not chunk:
                break

            digest.update(
                chunk
            )

    return digest.hexdigest()


def main() -> None:
    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if OUTPUT.exists():
        current = md5sum(
            OUTPUT
        )

        if current == EXPECTED_MD5:
            print(
                "MS2DeepScore model already present:",
                OUTPUT,
            )
            return

        raise RuntimeError(
            "Existing MS2DeepScore model has unexpected MD5: "
            f"{current}"
        )

    print(
        "Downloading MS2DeepScore model (~416 MB)..."
    )

    urllib.request.urlretrieve(
        URL,
        OUTPUT,
    )

    current = md5sum(
        OUTPUT
    )

    if current != EXPECTED_MD5:
        OUTPUT.unlink(
            missing_ok=True
        )

        raise RuntimeError(
            "Downloaded MS2DeepScore model failed MD5 validation: "
            f"{current}"
        )

    print(
        "Wrote:",
        OUTPUT,
    )


if __name__ == "__main__":
    main()
