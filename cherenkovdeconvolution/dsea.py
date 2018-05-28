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
import util

def deconvolve(X_data, X_train, y_train,
               maxiter = 1,
               epsilon = 0.0,
               fixweighting = True,
               ylevels = None): # TODO other kwargs
    
    # default arguments
    if ylevels is None:
        ylevels = numpy.unique(y_train)
    
    m = len(ylevels) # number of classes
    n = len(y_train) # number of training examples
    
    # initial estimate is uniform prior
    f = numpy.ones(m) / m # TODO optional argument
    f_train = util.histogram(y_train, ylevels) / m # training set distribution (fixweighting)
    
    # inspection
    # initial example weights
    # loop with training and prediction and chi2s
    
    return f_train

