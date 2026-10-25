"""Calendar labels, rather than row numbers, determine sample pairing."""
import numpy as np
import pytest

from climate_statistics.alignment import align_years


def test_missing_years_align_without_rescaling_or_imputation():
    years, (index, field, control) = align_years(
        ([2000, 2002, 2003], [20., 40., 60.]),
        ([2001, 2002, 2003], [[1., 2.], [3., np.nan], [5., 6.]]),
        ([1999, 2002, 2003, 2004], [7., 8., 9., 10.]),
    )
    np.testing.assert_array_equal(years, [2002, 2003])
    np.testing.assert_array_equal(index, [40., 60.])
    np.testing.assert_array_equal(field, [[3., np.nan], [5., 6.]])
    np.testing.assert_array_equal(control, [8., 9.])


def test_masked_samples_become_nan():
    _, (values, _) = align_years(([1, 2], np.ma.array([1, 2], mask=[0, 1])), ([1, 2], [3, 4]))
    assert values[0] == 1 and np.isnan(values[1])


@pytest.mark.parametrize("bad", [[1, 1], [2, 1], [1, np.nan], [1, 2.5], []])
def test_invalid_year_labels_are_rejected(bad):
    with pytest.raises(ValueError, match="Years"):
        align_years((bad, np.zeros(len(bad))), ([1, 2], [3, 4]))


def test_no_common_years_and_wrong_time_dimension_are_rejected():
    with pytest.raises(ValueError, match="no common"):
        align_years(([1, 2], [3, 4]), ([3, 4], [5, 6]))
    with pytest.raises(ValueError, match="dimension"):
        align_years(([1, 2], [3]), ([1, 2], [5, 6]))
    with pytest.raises(ValueError, match="at least two"):
        align_years(([1, 2], [3, 4]))


def test_complex_and_infinite_values_are_rejected():
    for bad in (np.array([1 + 2j, 3 + 4j]), [1, np.inf], [1, -np.inf]):
        with pytest.raises(ValueError, match="real|infinities"):
            align_years(([1, 2], bad), ([1, 2], [3, 4]))
    with pytest.raises(ValueError, match="real"):
        align_years((np.array([1 + 2j, 2 + 3j]), [1, 2]), ([1, 2], [3, 4]))
