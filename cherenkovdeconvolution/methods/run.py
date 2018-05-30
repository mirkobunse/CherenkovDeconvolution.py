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
        A function (k, tau, ldiff, f) -> () called in every iteration.
    
    Returns
    ----------
    f : array-like, shape (I,)
        The estimated target pdf.
    """
    raise NotImplementedError

