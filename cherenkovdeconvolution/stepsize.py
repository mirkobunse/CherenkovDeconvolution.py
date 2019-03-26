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
from scipy.optimize import minimize_scalar
import cherenkovdeconvolution.util as util
from cherenkovdeconvolution.methods.run import (_C_l, _maxl_l, _tikhonov_binning)


def decay_mul(eta, start = 1.0):
    """Construct a function object for a decaying stepsize in DSEA.
    
    The returned function describes a slow decay  alpha_k = start * k**(eta-1),  where k is
    the iteration number.
    
    Parameters
    ----------
    eta : float
        The decay rate. eta = 1 means no decay, eta = 0 means decay with medium speed 1/k,
        and eta = .5 means alpha_k = 1/sqrt(k), for example.
    
    start : float, optional
        The initial step size, which is 1, by default.
    
    Returns
    ----------
    alpha_fun : callable
        The stepsize function (k, pk, f_prev) -> float, which can be used as the alpha
        argument in DSEA.
    """
    def alpha_fun(k, pk, f_prev):
        return start * k**(eta-1) # pk and f_prev are not used, here
    return alpha_fun


def decay_exp(eta, start = 1.0):
    """Construct a function object for a decaying stepsize in DSEA.
    
    The returned function describes a fast decay  alpha_k = start * eta**(k-1),  where k is
    the iteration number.
    
    Parameters
    ----------
    eta : float
        The decay rate. eta = 1 means no decay and eta > 0 is recommended because DSEA would
        stop directly, otherwise.
    
    start : float, optional
        The initial step size, which is 1, by default.
    
    Returns
    ----------
    alpha_fun : callable
        The stepsize function (k, pk, f_prev) -> float, which can be used as the alpha
        argument in DSEA.
    """
    def alpha_fun(k, pk, f_prev):
        return start * eta**(k-1) # pk and f_prev are not used, here
    return alpha_fun


def alpha_adaptive_run(x_data, x_train, y_train, tau = 0, bins_y = None, bins_x = None):
    """Return a function object with the signature required by the alpha parameter in DSEA.
    
    This object adapts the DSEA step size to the current estimate by maximizing the likelihood
    of the next estimate in the search direction of the current iteration.
    
    Parameters
    ----------
    x_data : array-like, shape (n_samples,), nonnegative ints
        The observable quantity of the observed data set.
    
    x_train : array-like, shape (n_training_samples,), nonnegative ints
        The observable quantity of the training set.
    
    y_train : array-like, shape (n_training_samples,), nonnegative ints
        The labels belonging to x_train.
    
    tau : float, optional
        The regularization strength used while maximizing the likelihood.
    
    bins_y : array-like, shape (I,), nonnegative ints, optional
        The I indices of the target quantity values, i.e. the unique values of y.
    
    bins_x : array-like, shape (J,), nonnegative ints, optional
        The J indices of the observed clusters, i.e. the unique values of x.
    """
    if bins_y is None:
        bins_y = range(np.max(y_train))
    if bins_x is None:
        bins_x = range(np.max(np.concatenate((x_data, x_train))))
    
    # set up the discrete deconvolution problem
    R = util.fit_R(y_train, x_train, bins_y = bins_y, bins_x = bins_x)
    g = util.fit_pdf(x_data, bins_x, normalize = False) # absolute counts instead of pdf
    
    # set up a negative log likelihood function to be minimized
    C = _tikhonov_binning(R.shape[1]) # regularization matrix (from methods.run)
    maxl_l = _maxl_l(R, g)            # function of f (from methods.run)
    maxl_C = _C_l(tau, C)             # regularization term (from methods.run)
    def negloglike(f): # regularized objective function
        return maxl_l(f) + maxl_C(f)
    
    # return a step size function
    def alpha_adaptive_run(k, pk, f):
        a_min, a_max = _alpha_range(pk, f)
        return minimize_scalar(
          lambda a : negloglike(f + a*pk),
          bounds = (a_min, a_max),
          method = 'Bounded'
        ).x # only return the minimizer x (minimize_scalar is from scipy.optimize)
    return alpha_adaptive_run

# range of admissible alpha values
def _alpha_range(pk, f):
    # find alpha values for which the next estimate would be zero in one dimension
    a_zero = - (f[pk!=0] / pk[pk!=0]) # ignore zeros in pk, for which alpha is arbitrary
    
    # for positive pk[i] (negative a_zero[i]), alpha has to be larger than a_zero[i]
    # for negative pk[i] (positive a_zero[i]), alpha has to be smaller than a_zero[i]
    a_min = np.max(np.concatenate((a_zero[a_zero<0], [0]))) # select a_min=0 if no pk[i]>0 exists
    a_max = np.min(a_zero[a_zero>=0])
    return a_min, a_max
