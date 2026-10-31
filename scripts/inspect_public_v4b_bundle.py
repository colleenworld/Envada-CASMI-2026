from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent

ASSET_ROOT = (
    ROOT
    / "data/external/casmi26-public-assets"
)

V4_ROOT = (
    ASSET_ROOT
    / "casmi26-v4b-models"
)


def main() -> None:
    if not V4_ROOT.exists():
        raise FileNotFoundError(
            f"Missing v4b bundle: {V4_ROOT}\n"
            "Run: python scripts/fetch_public_fpnet_assets.py"
        )

    files = sorted(
        path
        for path in V4_ROOT.rglob("*")
        if path.is_file()
    )

    print(
        "v4b files:",
        len(files),
    )

    print()

    interesting = [
        path
        for path in files
        if (
            path.name
            in {
                "fpnet.py",
                "chem.py",
                "engine.py",
                "pool.py",
                "library.py",
                "MANIFEST.json",
                "manifest.json",
            }
            or "fpnet" in path.name.lower()
        )
    ]

    print(
        "Interesting files"
    )
    print(
        "=" * 80
    )

    for path in interesting:
        print(
            path.relative_to(
                ASSET_ROOT
            )
        )

    print()
    print(
        "Python files"
    )
    print(
        "=" * 80
    )

    for path in files:
        if path.suffix == ".py":
            print(
                path.relative_to(
                    ASSET_ROOT
                )
            )


if __name__ == "__main__":
    main()
