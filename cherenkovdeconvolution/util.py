# 
# CherenkovDeconvolution.py
# Copyright 2018, 2019 Mirko Bunse
# 
# 
# Deconvolution methods for Cherenkov astronomy and other use cases in experimental physics.
# 
# 
# CherenkovDeconvolution.py is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with CherenkovDeconvolution.py.  If not, see <http://www.gnu.org/licenses/>.
# 
import numpy as np
from warnings import warn


def equidistant_bin_edges(minimum, maximum, num_bins):
    """Shorthand obtaining equidistant bin edges from numpy.histogram.
    
    The returned edges can be used in numpy.digitize, to obtain the bin indices of data
    points. These indices, in turn, can be used to obtain a histogram with numpy.bincount.
    
    Parameters
    ----------
    minimum : float
        The lower range of the binning, i.e. the left-most bin edge.
    
    maximum : float
        The upper range of the binning, i.e. the right-most bin edge, which has to be
        greater than the minimum.
    
    num_bins: int
        The number of equidistant bins, which has to be greater than zero.
    
    Returns
    ----------
    bin_edges : array-like, shape (num_bins+1,), floats
        The bin edges.
    """
    if maximum > minimum: # not checked by numpy.histogram
        return np.histogram(np.empty(0), bins = num_bins, range = (minimum, maximum))[1]
    else:
        raise ValueError("maximum is not greater than minimum")


def fit_pdf(x, bins = None, normalize = True):
    """Estimate the discrete probability density function (pdf) g of the observed values x.
    
    Parameters
    ----------
    x : array-like, shape (n_samples,), nonnegative ints
        The indices of the J observed clusters.
    
    bins : array-like, shape (J,), nonnegative ints, optional
        The J indices of the observed clusters, i.e. the unique values of x.
    
    normalize : boolean, optional
        True, if the result g should be normalized to a probability density function.
        Otherwise, the integer counts of each bin are returned.
    
    Returns
    ----------
    g : array-like, shape (J,), floats
       The empirical discrete pdf of the observed values x.
    """
    if bins == None:
        bins = range(np.min(x), np.max(x)+1)
    bincounts = np.bincount(x, minlength = np.max(bins)+1)[bins]
    return normalizepdf(bincounts) if normalize else bincounts

def fit_R(y, x, bins_y = None, bins_x = None, normalize = True):
    """Estimate the detector response matrix R from the observed cluster indices x and the
    target quantity indices y.
    
    Parameters
    ----------
    y : array-like, shape (n_samples,), nonnegative ints
        The indices of the I target quantity values.
    
    x : array-like, shape (n_samples,), nonnegative ints
        The indices of the J observed clusters.
    
    bins_y : array-like, shape (I,), nonnegative ints, optional
        The I indices of the target quantity values, i.e. the unique values of y.
    
    bins_x : array-like, shape (J,), nonnegative ints, optional
        The J indices of the observed clusters, i.e. the unique values of x.
    
    normalize : boolean, optional
        True, if the columns of the result R should be normalized to probability densities.
        Otherwise, the integer counts of each bin are returned.
    
    Returns
    ----------
    R : array-like, shape (J, I), floats
        The empirical detector response matrix.
    """
    if bins_y == None:
        bins_y = range(np.min(y), np.max(y)+1)
    if bins_x == None:
        bins_x = range(np.min(x), np.max(x)+1)
    R = np.zeros((len(bins_x), len(bins_y)))
    for i in range(len(bins_y)):
        R[:, i] = fit_pdf(x[y == bins_y[i]], bins_x, normalize)
    return R


def normalizepdf(arr, copy = True):
    """Normalize the array so that it represents a probability density function.
    
    Parameters
    ----------
    arr : array-like, shape (I,)
        The array to normalize.
    
    copy : bool, optional
        Whether to create a copy of arr (True) or to work in place (False). Computation in
        place is only possible, if the dtype of arr is float.
    
    Returns
    ----------
    out : array-like, shape (I,), floats
        The normalized array, which may be a copy of arr.
    """
    if copy:
        arr = np.array(arr, dtype = float)
    elif arr.dtype != 'float':
        raise ValueError("dtype of arr has to be float, if copy is True")
    
    # replace NaNs and Infs by zero
    np.put(arr, np.argwhere(~np.isfinite(arr)), 0.0) # in-place replacement
    
    # divide by sum
    arrsum = np.sum(arr)
    if arrsum != 0:
        arr /= arrsum
    else:
        warn("Sum of array to be normalized is zero - returning uniform distribution")
        arr[:] = np.ones_like(arr) / len(arr)
    return arr


def smooth_polynomial(arr, order = 2):
    """Smooth arr with a polynomial fit, i.e. fit a polynomial to arr and return the values
    of the polynomial at the indices of arr.
    
    Parameters
    ----------
    arr : array-like, shape (I,), floats
        The array to smooth.
    
    order : int, optional
        The order of the polynomial used for smoothing.
    
    Returns
    ----------
    out : array-like, shape (I,), floats
        The smoothed array.
    """
    if order < len(arr):                                # pre-condition
        x = np.arange(len(arr))                         # values on x axis
        return np.polyval(np.polyfit(x, arr, order), x) # values of fitted polynomial
    else:
        raise ValueError("Order in polynomial smoothing has to be smaller than the array dimension")


def chi2s(a, b, normalize = True):
    """Compute the Chi Square distance between a and b.
    
    Parameters
    ----------
    a, b : array-like, shape (I,), floats
        The two arrays.
    
    normalize : bool, optional
        Whether to normalize a and b to probability density functions before computing the
        distance.
    
    Returns
    ----------
    out : float
        The probabilistic symmetric Chi Square distance between a and b.
    """
    if normalize:
        a = normalizepdf(a)
        b = normalizepdf(b)
    selection = (a > 0) | (b > 0) # limit computation to denominators > 0
    a = a[selection]
    b = b[selection]
    return 2 * np.sum(np.power(a - b, 2) / (a + b))

