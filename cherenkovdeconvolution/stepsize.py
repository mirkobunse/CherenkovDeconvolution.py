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
from warnings import warn


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

