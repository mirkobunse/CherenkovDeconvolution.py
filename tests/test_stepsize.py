import numpy as np
import cherenkovdeconvolution.stepsize as stepsize
from pytest import mark
from itertools import product


@mark.parametrize('i,k', product(range(10), range(1, 10)))
def test_decay_mul(i, k):
    """Test the function cherenkovdeconvolution.stepsize.decay_mul."""
    num_bins = np.random.randint(1, 100)
    start = np.random.uniform()
    eta = np.random.uniform()

    alpha_fun = stepsize.decay_mul(eta, start)

    f_prev = np.random.rand(num_bins)
    pk = np.random.rand(num_bins)
    assert alpha_fun(k, pk, f_prev) == start * k**(eta - 1)


@mark.parametrize('i,k', product(range(10), range(1, 10)))
def test_decay_exp(i, k):
    """Test the function cherenkovdeconvolution.stepsize.decay_exp."""
    num_bins = np.random.randint(1, 100)
    start = np.random.uniform()
    eta = np.random.uniform()

    alpha_fun = stepsize.decay_exp(eta, start)
    f_prev = np.random.rand(num_bins)
    pk = np.random.rand(num_bins)
    assert alpha_fun(k, pk, f_prev) == start * eta**(k - 1)
