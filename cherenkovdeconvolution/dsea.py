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
import cherenkovdeconvolution.util as util

def deconvolve(X_data, X_train, y_train,
               K = 1,
               epsilon = 0.0,
               fixweighting = True,
               ylevels = None,
               return_contributions = False):
    """Deconvolve the target distribution of X_data, as learned from X_train and y_train.

    y_train has to be discrete, i.e., it has to have a limited number of unique values that
    are used as labels for the classifier.
    
    Parameters
    ----------
    X_data : array-like, shape (n_samples, n_features)
        The data from which the target distribution is deconvolved.
    
    X_train : array-like, shape (n_samples_train, n_features)
        The data from which the classifier is trained.
    
    y_train : array-like, shape (n_samples_train,)
        The target quantity values belonging to X_train.
    
    K : int
        The maximum iteration number.
    
    epsilon : float
        The minimum Chi Square distance between iterations. If the actual distance is below
        this threshold, convergence is assumed and the algorithm stops.
    
    fixweighting : bool
        Whether or not the weight update fix is applied, which is proposed in my Master's
        thesis and the corresponding paper.
    
    ylevels : array-like, shape (m,)
        The m unique values in y_train, optionally specified to ensure that each expected
        unique value is considered in the deconvolution result. If not explicitly given, the
        unique values actually present in y_train are used.
    
    return_contributions : bool
        Whether or not to return the contributions of individual examples in X_data along
        with the deconvolution result.
    
    Returns
    ----------
    out : array-like, shape (m,)
        The estimated target distribution X_data.
    
    contributions : array-like, shape (n_samples, m)
        The contributions of individual items in X_data.
    """
    
    # default arguments
    if ylevels is None:
        ylevels = np.unique(y_train)
    m = len(ylevels) # number of classes
    
    if f_0 is None:
        np.ones(m) / m # uniform prior
    
    # check arguments
    
    # initial estimate (uniform by default)
    f       = f_0
    f_train = util.histogram(y_train, ylevels) / m                                # training distribution
    w_train = _dsea_weights(y_train, f / f_train if fixweighting else f, ylevels) # instance weights
    # inspection
    
    # iterative deconvolution
    for k in range(1, K):
        f_prev = f.copy() # previous estimate
        
        # predict data and reconstruct spectrum
        # find and apply step size
        # inspection
        # stop when convergence is assumed
        
        # == smoothing and reweighting in between iterations ==
        if k < K:
            # f = smoothing(f)
            _dsea_weights(y_train, f / f_train if fixweighting else f, ylevels, w_train) # in place
        # = = = = = = = = = = = = = = = = = = = = = = = = = = =
    
    if not return_contributions:
        return f
    else:
        # return f, contributions
        raise NotImplementedError

# the weights of training instances are based on the bin weights in w_bin
def _dsea_weights(y_train, w_bin, ylevels, out = None):
    w_bin = util.normalizepdf(w_bin) # normalized copy
    if out is None:
        out = np.zeros(len(y_train))
    
    # fill out array with bin weights
    for y, w in zip(ylevels, w_bin):
        np.put(out, np.argwhere(np.equal(y_train, y)), max(w, 1/len(y_train)))
    return out

