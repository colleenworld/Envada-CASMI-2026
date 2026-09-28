import numpy as np
import pytest

from casmi26.spectra.binning import bin_spectrum


def test_creates_expected_number_of_bins():
    result = bin_spectrum(
        [10.0],
        [1.0],
        max_mz=100.0,
        bin_width=1.0,
    )

    assert result.shape == (100,)


def test_places_peaks_into_correct_bins():
    result = bin_spectrum(
        [10.2, 20.8],
        [1.0, 2.0],
        max_mz=100.0,
        normalize=False,
    )

    assert result[10] == pytest.approx(1.0)
    assert result[20] == pytest.approx(2.0)


def test_sums_peaks_in_same_bin():
    result = bin_spectrum(
        [10.1, 10.7],
        [2.0, 3.0],
        max_mz=100.0,
        normalize=False,
    )

    assert result[10] == pytest.approx(5.0)


def test_discards_out_of_range_peaks():
    result = bin_spectrum(
        [-1.0, 10.0, 1500.0, 2000.0],
        [10.0, 1.0, 20.0, 30.0],
        normalize=False,
    )

    assert result.sum() == pytest.approx(1.0)
    assert result[10] == pytest.approx(1.0)


def test_normalizes_vector():
    result = bin_spectrum(
        [10.0, 20.0],
        [3.0, 4.0],
        max_mz=100.0,
    )

    assert np.linalg.norm(result) == pytest.approx(1.0)


def test_empty_spectrum_returns_zero_vector():
    result = bin_spectrum(
        [],
        [],
        max_mz=100.0,
    )

    assert np.all(result == 0)


def test_rejects_different_length_arrays():
    with pytest.raises(
        ValueError,
        match="same shape",
    ):
        bin_spectrum(
            [10.0, 20.0],
            [1.0],
        )


def test_rejects_invalid_range():
    with pytest.raises(
        ValueError,
        match="max_mz",
    ):
        bin_spectrum(
            [10.0],
            [1.0],
            min_mz=100.0,
            max_mz=50.0,
        )