import numpy as np
from numpy.typing import NDArray

def neutral_mass_mask(
    *,
    query_neutral_mass: float,
    candidate_neutral_masses: NDArray[np.floating],
    tolerance_da: float = 0.01,
) -> NDArray[np.bool_]:
    """
    Select candidates whose inferred neutral mass is within the
    specified absolute tolerance of the query neutral mass.
    """
    if tolerance_da < 0:
        raise ValueError(
            "tolerance_da must be non-negative"
        )

    return (
        np.isfinite(candidate_neutral_masses)
        & np.isfinite(query_neutral_mass)
        & (
            np.abs(
                candidate_neutral_masses
                - query_neutral_mass
            )
            <= tolerance_da
        )
    )

def precursor_adduct_mask(
    *,
    query_adduct: str,
    query_precursor_mz: float,
    candidate_adducts: NDArray[np.str_],
    candidate_precursor_mzs: NDArray[np.floating],
    tolerance_da: float = 0.01,
) -> NDArray[np.bool_]:
    """
    Select candidates having the same adduct and a precursor m/z
    within the requested absolute tolerance.
    """
    if tolerance_da < 0:
        raise ValueError("tolerance_da must be non-negative")

    if len(candidate_adducts) != len(candidate_precursor_mzs):
        raise ValueError(
            "candidate_adducts and candidate_precursor_mzs "
            "must have the same length"
        )

    return (
        (candidate_adducts == query_adduct)
        & np.isfinite(candidate_precursor_mzs)
        & np.isfinite(query_precursor_mz)
        & (
            np.abs(
                candidate_precursor_mzs
                - query_precursor_mz
            )
            <= tolerance_da
        )
    )