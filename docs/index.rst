.. Documentation master file, created by
   sphinx-quickstart on Mon Apr 16 21:22:43 2012.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

CherenkovDeconvolution.py
==================================

Deconvolution methods for Cherenkov astronomy and other use cases in experimental physics.


DSEA
----------

The Dortmund Spectrum Estimation Algorithm (DSEA) reconstructs the target distribution
from classifier predictions on the target quantity of individual examples.

CherenkovDeconvolution.py implements DSEA+, an improved version which employs an adaptive
step size for fast and reliable convergence.

.. autofunction:: cherenkovdeconvolution.dsea


Utilities
----------

The following set of utility functions is provided.

.. automodule:: cherenkovdeconvolution.util
    :members:
    :undoc-members:

