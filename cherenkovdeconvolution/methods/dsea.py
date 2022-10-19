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
from .. import (_recode_indices, _recode_result, _check_prior)


def deconvolve(X_data, X_train, y_train, classifier,
               bins = None,
               f_0 = None,
               fixweighting = True,
               alpha = 1,
               smoothing = None,
               K = 1,
               epsilon = 0.0,
               inspect = None,
               return_contributions = False):
    """Deconvolve the target distribution of X_data with DSEA, learning from X_train and
    y_train.
    
    Parameters
    ----------
    X_data : array-like, shape (n_samples, n_features), floats
        The data from which the target distribution is deconvolved.
    
    X_train : array-like, shape (n_samples_train, n_features), floats
        The data from which the classifier is trained.
    
    y_train : array-like, shape (n_samples_train,), nonnegative ints
        The indices of target quantity values belonging to X_train.
    
    classifier: object
        A classifier that is trained with classifier.fit(X_train, y_train, w_train) to
        obtain a matrix of probabilities with classifier.predict_proba(X_data).
        Any sklearn classifier is perfectly suited.
    
    bins: array-like, shape(I,), nonnegative ints, optional
        The indices of target quantity values. These values are allowed in y_train.
    
    f_0 : array-like, shape(I,), floats, optional
        The prior, which is uniform by default.
    
    fixweighting : bool, optional
        Whether or not the weight update fix is applied, which is proposed in my Master's
        thesis and the corresponding paper.
    
    alpha : float or function, optional
        A constant value or a function (k, pk, f_prev) -> float, which is used to choose
        the step size depending on the current estimate.
    
    smoothing : callable, optional
        A function (f) -> (f_smooth) optionally smoothing each estimate before using it as
        the prior of the next iteration.
    
    K : int, optional
        The maximum iteration number.
    
    epsilon : float, optional
        The minimum Chi Square distance between iterations. If the actual distance is below
        this threshold, convergence is assumed and the algorithm stops.
    
    inspect : callable, optional
        A function (f, k, alpha, chi2s) -> () optionally called in every iteration.
    
    return_contributions : bool, optional
        Whether or not to return the contributions of individual examples in X_data along
        with the deconvolution result.
    
    Returns
    ----------
    f : array-like, shape (I,)
        The estimated target pdf of X_data.
    
    contributions : array-like, shape (n_samples, I)
        The contributions of individual samples in X_data.
    """
    
    # check input data
    if bins is None:
        bins = np.unique(y_train)
    recode_dict, y_train = _recode_indices(bins, y_train)
    if X_data.shape[1] != X_train.shape[1]:
        raise ValueError("X_data and X_train have different numbers of features")
    elif not issubclass(y_train.dtype.type, np.integer):
        raise ValueError("dtype of y_train is not int")
    elif np.any(bins < 0):
        raise ValueError("y_train contains negative values")
    f_0 = _check_prior(f_0, recode_dict)
    if inspect is not None:
        inspect(_recode_result(f_0, recode_dict), 0, np.nan, np.nan)
    
    # initial estimate, training histogram, and instance weights
    f_prev = f_0
    f_train = np.bincount(y_train) / len(f_0)
    w_train = _dsea_weights(y_train, f_prev / f_train if fixweighting else f_prev)
    
    # iterative deconvolution
    for k in range(1, K+1):
        # === update the estimate ===
        proba = _train_and_predict_proba(classifier, X_data, X_train, y_train, w_train)
        f_dsea = _recode_result(_dsea_reconstruct(proba), recode_dict) # original DSEA
        f_next, alphak = _dsea_step(
          k,
          f_dsea,
          _recode_result(f_prev, recode_dict),
          alpha
        ) # next prior; step size function assumes original coding
        f_next = _check_prior(f_next, recode_dict) # re-code the next prior
        # = = = = = = = = = = = = = =
        
        # monitor progress
        chi2s = util.chi2s(f_prev, f_next) # Chi Square distance between iteration priors
        if inspect is not None:
            inspect(f_dsea, k, alphak, chi2s) # always inspect original DSEA reconstruction
        
        # stop when convergence is assumed
        if chi2s < epsilon:
            break
        
        # == smoothing and reweighting in between iterations ==
        if k < K:
            f_prev = f_next # update the prior
            if smoothing is not None:
                f_prev = smoothing(f_prev)
            w_train = _dsea_weights(y_train, f_prev / f_train if fixweighting else f_prev)
        # = = = = = = = = = = = = = = = = = = = = = = = = = = =
    
    proba = _recode_result(proba, recode_dict)
    return (f_dsea, proba) if return_contributions else f_dsea


# the weights of training instances are based on the bin weights in w_bin
def _dsea_weights(y_train, w_bin, normalize = True):
    if normalize:
        w_bin = util.normalizepdf(w_bin) # normalized copy
    return np.maximum(w_bin[y_train], 1/len(y_train)) # Laplace correction


# train and apply the classifier to obtain a matrix of confidence values
def _train_and_predict_proba(classifier, X_data, X_train, y_train, w_train):
    classifier.fit(X_train, y_train, w_train)
    return classifier.predict_proba(X_data)


# the reconstructed estimate is the normalized sum of confidences in each bin
def _dsea_reconstruct(proba):
    return util.normalizepdf(np.apply_along_axis(np.sum, 0, proba))


# the step taken by DSEA+, where alpha may be a constant or a function
def _dsea_step(k, f, f_prev, alpha):
    pk     = f - f_prev                                         # search direction
    alphak = alpha(k, pk, f_prev) if callable(alpha) else alpha # function or constant
    return f_prev + alphak * pk,  alphak                        # estimate and step size

