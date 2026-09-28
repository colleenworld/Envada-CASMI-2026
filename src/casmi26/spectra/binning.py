import numpy as np
from numpy.typing import ArrayLike, NDArray


def bin_spectrum(
    mzs: ArrayLike,
    intensities: ArrayLike,
    *,
    min_mz: float = 0.0,
    max_mz: float = 1500.0,
    bin_width: float = 1.0,
    normalize: bool = True,
) -> NDArray[np.float32]:
    """
    Convert an MS/MS spectrum into a fixed-width binned vector.

    Peaks outside [min_mz, max_mz) are discarded. Intensities of peaks
    falling into the same bin are summed.

    When normalize=True, the resulting vector is L2 normalized.
    """
    if max_mz <= min_mz:
        raise ValueError("max_mz must be greater than min_mz")

    if bin_width <= 0:
        raise ValueError("bin_width must be positive")

    mzs = np.asarray(mzs, dtype=np.float64)
    intensities = np.asarray(intensities, dtype=np.float64)

    if mzs.shape != intensities.shape:
        raise ValueError("mzs and intensities must have the same shape")

    n_bins = int(np.ceil((max_mz - min_mz) / bin_width))
    result = np.zeros(n_bins, dtype=np.float32)

    if mzs.size == 0:
        return result

    valid = (
        np.isfinite(mzs)
        & np.isfinite(intensities)
        & (mzs >= min_mz)
        & (mzs < max_mz)
    )

    if not np.any(valid):
        return result

    valid_mzs = mzs[valid]
    valid_intensities = intensities[valid]

    indices = ((valid_mzs - min_mz) / bin_width).astype(np.int64)

    np.add.at(result, indices, valid_intensities)

    if normalize:
        norm = np.linalg.norm(result)

        if norm > 0:
            result /= norm

    return result