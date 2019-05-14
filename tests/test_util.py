import numpy as np
import cherenkovdeconvolution.util as util
from pytest import mark, raises, approx


@mark.parametrize('i', range(10))
def test_equidistant_bin_edges(i):
    """Test the function cherenkovdeconvolution.util.equidistant_bin_edges."""
    # test on random arguments
    minimum = np.random.uniform(-1)      # in [-1, 1)
    maximum = np.random.uniform(minimum) # in [minimum, 1)
    num_bins = np.random.randint(1, 1000)
    bin_edges = util.equidistant_bin_edges(minimum, maximum, num_bins)
    assert len(bin_edges) == num_bins + 1
    assert bin_edges.min() == minimum
    assert bin_edges.max() == maximum

    # test exceptions
    minimum = np.random.uniform(-1)
    maximum = np.random.uniform(minimum)
    num_bins = np.random.randint(1, 1000)
    with raises(ValueError):
        util.equidistant_bin_edges(minimum, maximum, 0)  # num_bins has to be > 1
    with raises(ValueError):
        util.equidistant_bin_edges(maximum, minimum, num_bins)  # max > min is required
    with raises(ValueError):
        util.equidistant_bin_edges(minimum, minimum, num_bins)  # max > min is required (equality)


@mark.parametrize('i', range(10))
def test_normalizepdf(i):
    """Test the function cherenkovdeconvolution.util.normalizepdf."""
    # test on random arguments
    num_bins = np.random.randint(1, 1000)
    arr      = np.random.uniform(size = num_bins)
    narr = util.normalizepdf(arr)
    assert np.any(arr != narr)  # not performed in place
    assert len(narr) == num_bins
    assert sum(narr) == approx(1) # total equality violated by rounding
    util.normalizepdf(arr, copy = False)
    assert np.all(arr == narr) # in place version

    # test exceptions
    intarr = np.random.randint(1000, size = 10) # integer array
    assert sum(util.normalizepdf(intarr)) == approx(1)
    with raises(ValueError):
        util.normalizepdf(intarr, copy = False) # in place only allowed on floats


@mark.parametrize('i', range(10))
def test_smooth_polynomial(i):
    """Test the function cherenkovdeconvolution.util.smooth_polynomial."""
    # test on random arguments
    num_bins = np.random.randint(100, 1000)
    arr = np.random.uniform(size = num_bins)

    sarr = util.smooth_polynomial(arr, order = 1)
    diffs = sarr[1:] - sarr[:-1] # array of finite differences
    mdiff = np.mean(diffs)       # mean difference
    assert np.allclose(diffs, mdiff) # all differences approx. equal

    # multiple smoothings return approximately same array
    order = i + 1
    sarr1 = util.smooth_polynomial(arr,   order = order)
    sarr2 = util.smooth_polynomial(sarr1, order = order)
    assert np.allclose(sarr1, sarr2)

    # test exceptions
    with raises(ValueError):
        util.smooth_polynomial(np.random.uniform(size = 3), order = 3) # order < len(arr)


@mark.parametrize('i', range(10))
def test_chi2s(i):
    """Test the function cherenkovdeconvolution.util.chi2s."""
    # test on random arguments
    num_bins = np.random.randint(1, 1000)
    a = np.random.randint(1000, size = num_bins)
    b = np.random.randint(1000, size = num_bins)
    chi2s = util.chi2s(a, b)
    assert chi2s >= 0
    result = util.chi2s(util.normalizepdf(a), util.normalizepdf(b), normalize = False)
    assert result == chi2s

    # test increase on diverging arrays
    num_bins = np.random.randint(2, 1000)
    a = np.zeros(num_bins)
    b = np.ones(num_bins)
    a[1] = 1
    last_chi2s = util.chi2s(a, b)

    for i in range(10):
        b[2] += 1 - np.random.uniform()  # in (0, 1]
        chi2s = util.chi2s(a, b)
        assert chi2s >= last_chi2s
        last_chi2s = chi2s

    # test exceptions
    with raises(ValueError):
        util.chi2s(np.random.uniform(size = 3), np.random.uniform(size = 4))
