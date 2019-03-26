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
    if f_0 is None or len(f_0) == 0:
        return np.ones(m) if fit_ratios else np.ones(m) / m
    elif len(f_0) != m:
        raise ValueError('dim(f_0) = {} != {}, the number of classes'.format(len(f_0), m))
    else:
        if recode_dict != None:
            f_0 = f_0[np.sort(np.setdiff1d(list(recode_dict.values()), [-1]))] # recode argument
        return f_0 if fit_ratios else util.normalizepdf(f_0)


# wrapper for classical algorithms (ibu, run, ...) to set up R and g and then call the solver
def _discrete_deconvolution(solver, x_data, x_train, y_train, bins_y, kw_dict, normalize_g=True):
    # recode indices
    recode_dict, y_train = _recode_indices(bins_y, y_train)
    _, x_data, x_train   = _recode_indices(
      range(np.max(np.concatenate((x_data, x_train)))+1),
      x_data,
      x_train
    )

    # prepare the arguments for the solver
    bins_x = range(np.max(np.concatenate((x_data, x_train)))+1)
    fit_ratios = kw_dict.get('fit_ratios', False)
    R = util.fit_R(y_train, x_train, bins_x = bins_x, normalize = not fit_ratios)
    g = util.fit_pdf(x_data, bins_x, normalize = normalize_g)
    
    if 'f_0' in kw_dict:
        f_0 = _check_prior(kw_dict['f_0'], recode_dict) # also normalizes f_0
        if fit_ratios:
            f_0 = f_0 / util.fit_pdf(y_train) # pdf prior -> ratio prior
        kw_dict['f_0'] = f_0
    elif fit_ratios:
        kw_dict['f_0'] = np.ones(R.shape[1]) / util.fit_pdf(y_train) # uniform prior, not f_train
    
    # inspect with original coding of labels
    if 'inspect' in kw_dict:
        original = kw_dict['inspect']
        def inspect(f_est, *args):
            if fit_ratios:
                f_est = f_est * util.fit_pdf(y_train) # ratio solution -> pdf solution
            original(util.normalizepdf(_recode_result(f_est, recode_dict)), *args)
        kw_dict['inspect'] = inspect
    
    # call the solver (ibu, run, ...)
    f_est = solver(R, g, **kw_dict)
    if fit_ratios:
        f_est = f_est * util.fit_pdf(y_train) # ratio solution -> pdf solution
    return util.normalizepdf(_recode_result(f_est, recode_dict)) # revert recoding of labels


from cherenkovdeconvolution.methods.dsea import deconvolve as dsea
from cherenkovdeconvolution.methods.ibu  import deconvolve as ibu
from cherenkovdeconvolution.methods.run  import deconvolve as run
from cherenkovdeconvolution.methods.ibu  import deconvolve_evt as ibu_evt
from cherenkovdeconvolution.methods.run  import deconvolve_evt as run_evt
