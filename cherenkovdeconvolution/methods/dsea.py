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


def deconvolve(X_data, X_train, y_train, classifier,
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
    
    f_0 : array-like, shape(I,), floats
        The prior, which is uniform by default.
    
    fixweighting : bool
        Whether or not the weight update fix is applied, which is proposed in my Master's
        thesis and the corresponding paper.
    
    alpha : float or function
        A constant value or a function (k, pk, f_prev) -> float, which is used to choose
        the step size depending on the current estimate.
    
    smoothing : callable
        A function (f) -> (f_smooth) optionally smoothing each estimate before using it as
        the prior of the next iteration.
    
    K : int
        The maximum iteration number.
    
    epsilon : float
        The minimum Chi Square distance between iterations. If the actual distance is below
        this threshold, convergence is assumed and the algorithm stops.
    
    inspect : callable
        A function (k, alpha, chi2s, f) -> () optionally called in every iteration.
    
    return_contributions : bool
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
    classes = np.unique(y_train)
    if X_data.shape[1] != X_train.shape[1]:
        raise ValueError("X_data and X_train have different numbers of features")
    elif not issubclass(y_train.dtype.type, np.integer):
        raise ValueError("dtype of y_train is not int")
    elif np.any(classes < 0):
        raise ValueError("y_train contains negative values")
    
    # default prior is uniform
    I = len(classes)
    if f_0 is None:
        f_0 = np.ones(I) / I
    elif len(f_0) != I:
        raise ValueError("f_0 has a wrong dimension")
    else:
        util.normalizepdf(f_0, copy = False) # make sure that f_0 is a pdf
    
    # weight the training set and inspect the prior
    f       = f_0
    f_train = np.bincount(y_train) / I                                   # training histogram
    w_train = _dsea_weights(y_train, f / f_train if fixweighting else f) # instance weights
    if inspect is not None:
        inspect(0, np.nan, np.nan, f)
    
    # iterative deconvolution
    for k in range(1, K+1):
        f_prev = f.copy() # previous estimate
        
        # === update the estimate ===
        proba     = _train_and_predict_proba(classifier, X_data, X_train, y_train, w_train)
        f, alphak = _dsea_step(k, _dsea_reconstruct(proba), f_prev, alpha)
        # = = = = = = = = = = = = = =
        
        # monitor progress
        chi2s = util.chi2s(f_prev, f) # Chi Square distance between iterations
        if inspect is not None:
            inspect(k, alphak, chi2s, f)
        
        # stop when convergence is assumed
        if chi2s < epsilon:
            break
        
        # == smoothing and reweighting in between iterations ==
        if k < K:
            if smoothing is not None:
                f = smoothing(f)
            w_train = _dsea_weights(y_train, f / f_train if fixweighting else f)
        # = = = = = = = = = = = = = = = = = = = = = = = = = = =
    
    return (f, proba) if return_contributions else f


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

