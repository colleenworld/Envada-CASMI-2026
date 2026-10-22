from __future__ import annotations

import argparse
from pathlib import Path


OLD_IMPORT = (
    "from pytorch_lightning.loggers.base import "
    "LightningLoggerBase, rank_zero_experiment"
)

NEW_IMPORT = """try:
    from pytorch_lightning.loggers.base import (
        LightningLoggerBase,
        rank_zero_experiment,
    )
except ImportError:
    try:
        from pytorch_lightning.loggers.logger import (
            Logger as LightningLoggerBase,
            rank_zero_experiment,
        )
    except ImportError:
        from lightning.pytorch.loggers.logger import (
            Logger as LightningLoggerBase,
            rank_zero_experiment,
        )"""


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mist_root",
        type=Path,
        help="Root of the MIST copy to patch",
    )
    return parser.parse_args()


RDKIT_SUPPRESSION_IMPORT = "from rdkit import RDLogger"
RDKIT_SUPPRESSION_CALL = 'RDLogger.DisableLog("rdApp.*")'


def patch_lightning(
    mist_root: Path,
) -> None:
    misc_utils = (
        mist_root
        / "src/mist/utils/misc_utils.py"
    )

    if not misc_utils.exists():
        raise FileNotFoundError(
            f"MIST misc_utils.py not found: {misc_utils}"
        )

    text = misc_utils.read_text()

    if NEW_IMPORT in text:
        print(
            "MIST Lightning compatibility patch already present:",
            misc_utils,
        )
        return

    if OLD_IMPORT not in text:
        raise RuntimeError(
            "Expected legacy Lightning logger import was not found in "
            f"{misc_utils}. Refusing to patch an unknown MIST source."
        )

    misc_utils.write_text(
        text.replace(
            OLD_IMPORT,
            NEW_IMPORT,
            1,
        )
    )

    print(
        "Patched MIST Lightning logger compatibility:",
        misc_utils,
    )


def patch_rdkit_logging(
    path: Path,
) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"MIST RDKit module not found: {path}"
        )

    text = path.read_text()

    if (
        RDKIT_SUPPRESSION_IMPORT in text
        and RDKIT_SUPPRESSION_CALL in text
    ):
        print(
            "MIST RDKit logging suppression already present:",
            path,
        )
        return

    lines = text.splitlines()

    insert_at = 0

    if lines and lines[0].startswith('"""'):
        if lines[0].count('"""') >= 2:
            insert_at = 1
        else:
            for index in range(1, len(lines)):
                if '"""' in lines[index]:
                    insert_at = index + 1
                    break

    suppression = [
        "",
        "from rdkit import RDLogger",
        "",
        'RDLogger.DisableLog("rdApp.*")',
        "",
    ]

    lines[
        insert_at:insert_at
    ] = suppression

    path.write_text(
        "\n".join(lines)
        + "\n"
    )

    print(
        "Patched MIST RDKit logging suppression:",
        path,
    )


def main() -> None:
    args = parse_args()

    patch_lightning(
        args.mist_root
    )

    for relative_path in [
        "src/mist/data/featurizers.py",
        "src/mist/utils/chem_utils.py",
    ]:
        patch_rdkit_logging(
            args.mist_root
            / relative_path
        )


if __name__ == "__main__":
    main()
