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
from sklearn.tree import DecisionTreeClassifier


class TreeDiscretizer:
    """A DecisionTreeClassifier of which the leaf indices are used as cluster indices.
    
    Parameters
    ----------
    X_train : array-like, shape (n_samples, n_features), floats
        The feature matrix of the training set.
    
    y_train : array-like, shape (n_samples), nonnegative ints
        The labels of X_train.
    
    J : int
        The maximum number of desired clusters.
    
    criterion : string, optional
        The split criterion of the underlying DecisionTreeClassifier, 'gini' by default.
    
    seed : nonnegative int, optional
        This seed for the random number generator makes random splits reproducible.
    """
    def __init__(self, X_train, y_train, J, criterion = 'gini', seed = None):
        self.classifier = DecisionTreeClassifier(
          max_leaf_nodes = J,
          criterion = criterion,
          random_state = seed
        )
        self.classifier.fit(X_train, y_train)
        x_train = self.classifier.apply(X_train)
        self.indexmap = dict(zip(np.unique(x_train), range(len(np.unique(x_train)))))
    def discretize(self, X):
        x_raw = self.classifier.apply(X) # return the raw leaf indices of X
        return np.array([ self.indexmap[x] for x in x_raw ])


def fit_pdf(x, bins = None, normalize = True):
    """Estimate the discrete probability density function (pdf) g of the observed values x.
    
    Parameters
    ----------
    x : array-like, shape (n_samples,), nonnegative ints
        The indices of the J observed clusters.
    
    bins : array-like, shape (J,), nonnegative ints, optional
        The J indices of the observed clusters, i.e. the unique values of x.
    
    normalize : boolean, optional
        True, if the result g should be normalized to a probability density function.
        Otherwise, the integer counts of each bin are returned.
    
    Returns
    ----------
    g : array-like, shape (J,), floats
       The empirical discrete pdf of the observed values x.
    """
    if bins is None:
        bins = range(np.min(x), np.max(x)+1)
    bincounts = np.bincount(x, minlength = np.max(bins)+1)[bins]
    return normalizepdf(bincounts) if normalize else bincounts

