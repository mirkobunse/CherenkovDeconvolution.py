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
import cherenkovdeconvolution.util as util
from .. import (_discrete_deconvolution, _check_prior)


# compute a 'reverse transfer' matrix with all the entries used by Bayes' theorem
def _ibu_reverse_transfer(R, f_0):
    B = np.zeros((R.shape[1], R.shape[0]))
    for j in range(R.shape[0]):
        B[:, j] = R[j, :] * f_0 / np.dot(R[j, :], f_0)
    return B


def deconvolve(R, g,
               f_0 = None,
               smoothing = None,
               K = 3,
               epsilon = 0.0,
               fit_ratios = False,
               inspect = None):
    """Deconvolve the target distribution f, given R and g, with Iterative Bayesian
    Unfolding.
    
    Parameters
    ----------
    R : array-like, shape (J, I), floats
        The detector response matrix.
    
    g : array-like, shape (J,), floats
        The observed discrete pdf.
    
    f_0 : array-like, shape(I,), floats, optional
        The prior, which is uniform by default.
    
    smoothing : callable, optional
        A function (f) -> (f_smooth) optionally smoothing each estimate before using it as
        the prior of the next iteration.
    
    K : int, optional
        The maximum iteration number.
    
    epsilon : float, optional
        The minimum Chi Square distance between iterations. If the actual distance is below
        this threshold, convergence is assumed and the algorithm stops.
    
    fit_ratios : boolean, optional
        Determines if ratios are fitted (i.e. R has to contain counts so that the ratio
        f_est/f_train is estimated) or if the probability density f_est is fitted directly.
    
    inspect : callable, optional
        A function (f, k, chi2s) -> () optionally called in every iteration.
    
    Returns
    ----------
    f : array-like, shape (I,)
        The estimated target pdf.
    """
    
    # check arguments
    if R.shape[0] != len(g):
        raise ValueError('dim(g) = {} is not equal to the observable dimension {} of R'.format(
          len(g), R.shape[0]))
    
    # initial estimate
    f = _check_prior(f_0, m = R.shape[1], fit_ratios = fit_ratios)
    if inspect is not None:
        inspect(f, 0, np.nan)
    
    # iterative Bayesian deconvolution
    for k in range(1, K+1):
        
        # == smoothing in between iterations ==
        f_prev_smooth = smoothing(f) if smoothing is not None and k > 1 else f
        f_prev = f # unsmoothed estimate is required for convergence check
        # = = = = = = = = = = = = = = = = = = =
        
        # === apply Bayes' rule ===
        f = np.dot(_ibu_reverse_transfer(R, f_prev_smooth), g)
        if not fit_ratios:
            f = util.normalizepdf(f)
        # = = = = = = = = = = = = =
        
        # monitor progress
        chi2s = util.chi2s(f_prev, f, False) # Chi square distance between iterations
        if inspect is not None:
            inspect(f, k, chi2s)
        
        # stop when convergence is assumed
        if chi2s < epsilon:
            break
    
    return f # return the last estimate


def deconvolve_evt(x_data, x_train, y_train, bins_y = None, **kwargs):
    if np.any(bins_y == None):
        bins_y = np.unique(y_train)
    return _discrete_deconvolution(deconvolve, x_data, x_train, y_train, bins_y, dict(kwargs))
