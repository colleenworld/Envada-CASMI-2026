from __future__ import annotations

from collections.abc import Collection

import pandas as pd


def filter_to_training_structures(
    frame: pd.DataFrame,
    training_keys: Collection[str],
) -> pd.DataFrame:
    """
    Return only spectra whose InChIKey14 belongs to the training split.

    The ordinary validation manifest is structure-disjoint, so filtering
    every retrieval-index batch this way guarantees that validation
    structures cannot leak into the spectral reference index.
    """
    if "inchikey14" not in frame.columns:
        raise ValueError(
            "frame must contain an inchikey14 column"
        )

    keys = {
        str(key)
        for key in training_keys
    }

    if not keys:
        raise ValueError(
            "training_keys must not be empty"
        )

    mask = (
        frame["inchikey14"]
        .astype(str)
        .isin(keys)
    )

    return (
        frame.loc[mask]
        .reset_index(drop=True)
    )
