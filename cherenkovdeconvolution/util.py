# 
# CherenkovDeconvolution.py
# Copyright 2018 Mirko Bunse
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


def histogram(arr, levels = None):
    """Return a histogram of arr, in which the unique values are optionally defined.
    
    Parameters
    ----------
    arr : array-like, shape (n_samples,)
        The array to obtain the histogram of.
    
    levels : array-like, shape (m,)
        The m unique values in arr, optionally specified to ensure that each expected unique
        value is considered in the histogram. If not explicitly given, the unique values
        already present in arr are used.
    
    Returns
    ----------
    hist : array-like, shape (m,)
        The histogram of arr.
    """
    if levels is None:
        # return counts of sorted unique elements
        return np.unique(arr, return_counts = True)[1] # TODO np.bincount, only indices
    else:
        # concatenate levels to ensure existence, then substract 1 from each count
        return np.unique(np.concatenate((arr, levels)), return_counts = True)[1] - 1


def empiricaltransfer():
    raise NotImplementedError


def normalizepdf(arr, copy = True):
    """Normalize the array so that it represents a probability density function.
    
    Parameters
    ----------
    arr : array-like, shape (m,)
        The array to normalize.
    
    copy : bool, optional
        Whether to create a copy of arr (True) or to replace values in-place (False).
    
    Returns
    ----------
    out : array-like, shape (m,)
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
    arr : array-like, shape (m,)
        The array to smooth.
    
    order : int, optional
        The order of the polynomial used for smoothing.
    
    Returns
    ----------
    out : array-like, shape (m,)
        The smoothed array.
    """
    if order < len(arr):                                # pre-condition
        x = np.arange(len(arr))                         # values on x axis
        return np.polyval(np.polyfit(x, arr, order), x) # values of fitted polynomial
    else:
        ValueError("Order in polynomial smoothing has to be smaller than the array dimension")


def chi2s(a, b, normalize = True):
    """Compute the Chi Square distance between a and b.
    
    Parameters
    ----------
    a, b : array-like, shape (m,)
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

