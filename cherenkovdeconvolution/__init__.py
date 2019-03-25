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

# recode indices to resemble a unit range (no missing labels in between)
def _recode_indices(bins, *inds):
    # recode the training set
    inds_bins = np.unique(np.concatenate(tuple(inds)))
    inds_dict = dict(zip(inds_bins, range(len(inds_bins))))
    inds_rec  = [ np.array([inds_dict[i] for i in ind]) for ind in inds ]
    
    # set up reverse recoding applied in _recode_result
    recode_dict = dict(zip(inds_dict.values(), inds_dict.keys())) # map from values to keys
    recode_dict[-1] = np.max(bins) # the highest original bin (may not be in y_train)
    
    return tuple([recode_dict, *inds_rec]) # return all recoded indices

# recode a deconvolution result by reverting the initial recoding of the data
def _recode_result(M, recode_dict):
    is_vector = len(M.shape) == 1 # else, we are recoding a probability matrix
    if is_vector:
        M = M.reshape(1, len(M)) # treat the vector M as a matrix
    r = np.zeros(( M.shape[0], np.max(list(recode_dict.values()))+1 ))
    for key, val in recode_dict.items():
        if key != -1:
            r[:, val] = M[:, key] # else, the key was just included to store the maximum value
    if is_vector:
        r = r.reshape(r.shape[1]) # treat the vector M like a vector again
    return r

# check and repair the f_0 argument of deconvolution methods
def _check_prior(f_0, recode_dict=None, m=None, fit_ratios=False):
    if recode_dict != None:
        m = len(recode_dict)-1
    if np.any(f_0 == None) or len(f_0) == 0:
        return np.ones(m) if fit_ratios else np.ones(m) / m
    elif len(f_0) != m:
        raise ValueError('dim(f_0) = {} != {}, the number of classes'.format(len(f_0), m))
    else:
        f_0 = f_0[np.sort(np.setdiff1d(list(recode_dict.values()), [-1]))] # recode argument
        if fit_ratios:
            return f_0 # f_0 is provided and alright (after recoding)
        else:
            return util.normalizepdf(f_0) # ensure pdf

from cherenkovdeconvolution.methods.dsea import deconvolve as dsea
