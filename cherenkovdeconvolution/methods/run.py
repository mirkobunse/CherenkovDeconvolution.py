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
import cherenkovdeconvolution.util as util
from .. import _discrete_deconvolution


# objective function: negative log-likelihood
def _maxl_l(R, g):
    def maxl_lj(j, f): # map each index of g to its element-wise loss
        fj = np.dot(R[j,:], f)
        return fj - g[j]*np.log(fj)
    def maxl_l(f): # compute the loss of an estimate f
        return np.sum([ maxl_lj(j, f) for j in range(len(g)) ])
    return maxl_l # return a function object f -> l(f)

# gradient of objective
def _maxl_g(R, g):
    def maxl_gi(i, f): # map each dimension of f to a gradient dimension
        return np.sum([ R[j,i] - g[j]*R[j,i] / np.dot(R[j,:], f) for j in range(len(g)) ])
    def maxl_g(f): # compute the gradient at an estimate f
        return np.array([ maxl_gi(i, f) for i in range(len(f)) ])
    return maxl_g # return a function object f -> gradient(f)

# Hessian of objective
def _maxl_H(R, g):
    def maxl_H(f): # compute the gradient at an estimate f
        res = np.zeros((len(f), len(f)))
        for i1 in range(len(f)):
          for i2 in range(len(f)):
            res[i1,i2] = np.sum(
              [ g[j]*R[j,i1]*R[j,i2] / pow(np.dot(R[j,:], f), 2) for j in range(len(g)) ]
            )
        return res
    return maxl_H # return a function object f -> Hessian(f)


# objective function: least squares
def _lsq_l(R, g):
    def lsq_l(f): # compute the loss of an estimate f
        return np.sum([ pow(g[j] - np.dot(R[j,:], f), 2) / g[j] for j in range(len(g)) ])/2
    return lsq_l # return a function object f -> l(f)

# gradient of least squares objective
def _lsq_g(R, g):
    def lsq_gi(i, f): # map each dimension of f to a gradient dimension
        return np.sum([ -R[j,i] * (g[j] - np.dot(R[j,:], f)) / g[j] for j in range(len(g)) ])
    def lsq_g(f): # compute the gradient at an estimate f
        return np.array([ lsq_gi(i, f) for i in range(len(f)) ])
    return lsq_g # return a function object f -> gradient(f)

# hessian of least squares objective
def _lsq_H(R, g):
    def lsq_H(f): # compute the gradient at an estimate f
        res = np.zeros((len(f), len(f)))
        for i1 in range(len(f)):
          for i2 in range(len(f)):
            res[i1,i2] = np.sum([ R[j,i1]*R[j,i2] / g[j] for j in range(len(g)) ])
        return res
    return lsq_H # return a function object f -> Hessian(f)


# regularization term in objective function (both LSq and MaxL)
def _C_l(tau, C):
    def C_l(f):
        return tau/2 * np.dot(f, np.dot(C, f))
    return C_l # return a function object f -> C_l(f)

# regularization term in gradient of objective
def _C_g(tau, C):
    def C_g(f):
        return tau * np.dot(C, f)
    return C_g # return a function object f -> C_g(f)

# regularization term in the Hessian of objective
def _C_H(tau, C):
    def C_H(f):
        return tau * C
    return C_H # return a function object f -> C_H(f)

# A Tikhonov matrix for binned discretization, as given in [cowan1998statistical, p. 169].
# This is equivalent to the notation in [blobel2002unfolding_long]!
def _tikhonov_binning(m):
    if m < 1:
        raise ValueError('m has to be greater than zero')
    elif m < 3: # stupid case
        return np.eye(m)
    elif m == 3: # not quite intelligent case
        return np.eye(m) + np.diag([-1, -1], 1) + np.diag([-1, -1], -1)
    else: # usual case
        return np.diag(np.concatenate(([1, 5], np.repeat(6, np.max([0, m-4])), [5, 1]))) + \
          np.diag(np.concatenate(([-2], np.repeat(-4, m-3), [-2])),  1) + \
          np.diag(np.concatenate(([-2], np.repeat(-4, m-3), [-2])), -1) + \
          np.diag(np.repeat(1, m-2),  2) + \
          np.diag(np.repeat(1, m-2), -2)


def _tau(n_df, eigvals_C):
    warn('cherenkovdeconvolution.methods.run._tau needs implementation - always returns zero')
    return 0 # TODO implement cherenkovdeconvolution.methods.run._tau


def deconvolve(R, g,
               n_df = None,
               K = 100,
               epsilon = 1e-6,
               inspect = None):
    """Deconvolve the target distribution f, given R and g, with Regularized Unfolding.
    
    Parameters
    ----------
    R : array-like, shape (J, I), floats
        The detector response matrix.
    
    g : array-like, shape (J,), floats
        The observed discrete pdf.
    
    n_df : int
        The desired number of degrees of freedom in the solution f.
    
    K : int
        The maximum iteration number.
    
    epsilon : float
        The minimum Chi Square distance between iterations. If the actual distance is below
        this threshold, convergence is assumed and the algorithm stops.
    
    inspect : callable
        A function (f, k, tau, ldiff) -> () called in every iteration.
    
    Returns
    ----------
    f : array-like, shape (I,)
        The estimated target pdf.
    """
    # check arguments
    if np.any(g <= 0): # limit deconvolution to non-zero bins
        nonzero = g > 0
        warn('Limiting RUN to {} of {} observable non-zero bins'.format(np.sum(nonzero), len(g)))
        g = g[nonzero]
        R = R[nonzero, :]
    m = R.shape[1] # dimension of f
    if n_df is None:
        n_df = m
    if R.shape[0] != len(g):
        raise ValueError('dim(g) = {} is not equal to the observable dimension {} of R'.format(
          len(g), R.shape[0]))
    if m > R.shape[0]:
        warn('RUN is given more target than observable bins - results may be unsatisfactory')
    
    # set up the loss function
    l   = _maxl_l(R, g) # the objective function,
    g_l = _maxl_g(R, g) # ..its gradient,
    H_l = _maxl_H(R, g) # ..and its Hessian
    C   = _tikhonov_binning(m) # the Tikhonov matrix (not in l and its derivatives)
    
    # the initial estimate is the zero vector
    f = np.zeros(m)
    
    # the first iteration is a least-squares fit
    H_lsq = _lsq_H(R, g)(f) # Hessian at the zero vector
    if not np.all(np.isfinite(H_lsq)):
        warn('LSq Hessian contains Infs or NaNs - replacing these by zero')
        H_lsq[np.logical_not(np.isfinite(H_lsq))] = 0
    try:
        f -= np.dot(np.linalg.inv(H_lsq), _lsq_g(R, g)(f))
    except np.linalg.LinAlgError:
        warn('LSq Hessian is singular - using pseudo inverse in RUN')
        f -= np.dot(np.linalg.pinv(H_lsq), _lsq_g(R, g)(f))
    if inspect is not None:
        inspect(f, 1, np.nan, np.nan)
    
    # subsequent iterations maximize the likelihood
    l_prev = l(f) # loss from the previous iteration
    for k in range(2, K+1):
        
        # gradient and Hessian at the last estimate
        g_f = g_l(f)
        H_f = H_l(f)
        if not np.all(np.isfinite(H_f)):
            warn('MaxL Hessian contains Infs or NaNs - replacing these by zero')
            H_f[np.logical_not(np.isfinite(H_f))] = 0
        
        # eigendecomposition of the Hessian: H_f == U*D*U' (complex conversion if I>J)
        eigvals_H, U = np.linalg.eig(H_f)
        D = np.diag(np.power(eigvals_H, -1/2)) # D^(-1/2)
        
        # eigendecomposition of transformed Tikhonov matrix: C2 == U_C*S*U_C'
        C2 = np.matmul(np.matmul(np.matmul(np.matmul(D, U.transpose()), C), U), D)
        C2[np.logical_not(np.isfinite(C2))] = 0
        eigvals_C, U_C = np.linalg.eig(C2)
        
        # select tau (special case: no regularization if n_df == m)
        tau = _tau(n_df, eigvals_C) if n_df < m else 0
        
        # 
        # Taking a step in the transformed problem and transforming back to the actual
        # solution is numerically difficult because the eigendecomposition introduces some
        # error. In the transformed problem, therefore only tau is chosen. The step is taken 
        # in the original problem instead of the commented-out solution.
        # 
        # S   = np.diag(eigvals_C)
        # f_2 = 1/2 * np.linalg.inv(np.eye(S) + tau*S) * (U*D*U_C)' * (H_f * f - g_f)
        # f   = (U*D*U_C) * f_2
        # 
        g_f += _C_g(tau, C)(f) # regularized gradient
        H_f += _C_H(tau, C)(f) # regularized Hessian
        try:
            f -= np.dot(np.linalg.inv(H_f), g_f)
        except np.linalg.LinAlgError:
            warn('MaxL Hessian is singular - using pseudo inverse in RUN')
            f -= np.dot(np.linalg.pinv(H_f), g_f) # try again with pseudo inverse
        
        # monitor progress
        l_now = l(f) + _C_l(tau, C)(f)
        ldiff = l_prev - l_now
        if inspect is not None:
            inspect(f, k, ldiff, tau)
        
        # stop when convergence is assumed
        if abs(ldiff) < epsilon:
            break
        l_prev = l_now
    
    return f


def deconvolve_evt(x_data, x_train, y_train, bins_y = None, **kwargs):
    if np.any(bins_y == None):
        bins_y = np.unique(y_train)
    return _discrete_deconvolution(deconvolve, x_data, x_train, y_train, bins_y, dict(kwargs))
