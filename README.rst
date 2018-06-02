.. image:: https://travis-ci.org/mirkobunse/CherenkovDeconvolution.py.svg?branch=master
    :target: https://travis-ci.org/mirkobunse/CherenkovDeconvolution.py

=========================
CherenkovDeconvolution.py
=========================

Deconvolution methods for Cherenkov astronomy and other use cases in experimental physics.



Deconvolution in Cherenkov Astronomy
------------------------------------

One frequent objective in experimental physics is to obtain the distribution of a physical quantity.
In cases where the distribution of the relevant quantity cannot be accessed experimentally,
it has to be reconstructed from distributions of correlated quantities that are measured, instead.
This reconstruction is called *deconvolution*.

Cherenkov astronomy is a deconvolution use case which studies the energy distribution of cosmic gamma radiation
to reason about the characteristics of celestial objects emitting such radiation.
Since the gamma radiation is not directly measured by the ground-based telescopes employed in Cherenkov astronomy,
deconvolution is applied to reconstruct the gamma particle distribution from the related Cherenkov light recorded by these telescopes.

.. figure:: docs/air-shower.png

*A gamma particle interacting in Earth's atmosphere produces a cascade of secondary particles, the air shower. This shower emits Cherenkov light, which is measured by a telescope. The energy distribution of gamma particles can be reconstructed from IACT measurements.*

CherenkovDeconvolution.py provides functions for reconstructing the distribution of a target quantity
from measurements of correlated quantities.



Getting Started
---------------

You can install CherenkovDeconvolution.py via the provided Makefile.
Running ``make`` without arguments will also run the test cases.

For a quick start, take a look at the Jupyter notebooks in the ``examples`` directory.
Running ``make doc`` will generate the API documentation with Sphinx.



Current Status
--------------

CherenkovDeconvolution.py currently implements the Dortmund Spectrum Estimation Algorithm
(DSEA). The other major algorithms will soon be added.

CherenkovDeconvolution.py was originally developed in Julia,
as `CherenkovDeconvolution.jl <https://github.com/mirkobunse/CherenkovDeconvolution.jl>`_.


