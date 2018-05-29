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

def train_and_predict_proba(classifier):
    """Create a function object which trains and applies the given classifier. This function
    object can be used as an argument to the dsea.deconvolve function.
    
    Parameters
    ----------
    classifier : sklearn object
        The classifier to train and apply.
    
    Returns
    ----------
    lambda X_data, X_train, y_train, w_train, ylevels : proba
        A function object which trains the classifier on X_train and y_train, given the
        weights w_train and the labels in ylevels. The trained classifier is directly
        applied to X_data, for which a probability matrix is returned by this function
        object.
    """
    return lambda X_d, X_t, y_t, w_t, ys: _train_and_predict_proba(classifier, X_d, X_t, y_t, w_t, ys)

# the work horse of the lambda expression in train_and_predict_proba(classifier)
def _train_and_predict_proba(classifier, X_data, X_train, y_train, w_train, ylevels):
    # train classifier and obtain confidence values
    classifier.fit(X_train, y_train, w_train)
    proba = classifier.predict_proba(X_data) # matrix of probabilities
    
    # permute columns in order of ylevels, i.e. match order of columns
    proba[:, np.argsort(classifier.classes_)] = proba[:, np.argsort(ylevels)]
    return proba

