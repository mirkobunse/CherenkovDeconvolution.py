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
import numpy

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
        return numpy.unique(arr, return_counts = True)[1]
    else:
        # concatenate levels to ensure existence, then substract 1 from each count
        return numpy.unique(numpy.concatenate((arr, levels)), return_counts = True)[1] - 1

def empiricaltransfer():
    raise NotImplementedError

def normalizepdf(arr):
    # check for NaNs and Infs
    # check for negative values
    # check for zero sums
    raise NotImplementedError

def chi2s(a, b):
    raise NotImplementedError # 2 * Distances.chisq_dist(a, b)

