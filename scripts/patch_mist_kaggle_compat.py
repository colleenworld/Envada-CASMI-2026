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


def main() -> None:
    args = parse_args()

    misc_utils = (
        args.mist_root
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

    patched = text.replace(
        OLD_IMPORT,
        NEW_IMPORT,
        1,
    )

    misc_utils.write_text(
        patched
    )

    print(
        "Patched MIST Lightning logger compatibility:",
        misc_utils,
    )


if __name__ == "__main__":
    main()
