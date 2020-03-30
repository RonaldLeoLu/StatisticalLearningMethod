import numpy as np
import pandas as pd

class BaseBuilder:
    '''

    The base structure of future models.

    Param:
    - learning_rate : the step of each updating. Defaultly 1e-2.
    - with_bias     : whether there is an intercept in the hypothesis plane. Defaultly True
    - threshold     : the threshold of probability. Defaultly 0.5

    Attr:
    - coef_      : the weights or the coeficients of linear model 
    - intercept_ : the bias or the intercept of linear model
    
    Method:
    - fit           : fit training data
    - predict       : predict the final class
    - predict_proba : predict the probability of each class
    '''
    def __init__(self, learning_rate=1e-2, with_bias=True, threshold=0.5, max_iters=10000):
        self.lr = learning_rate
        self.with_bias = with_bias
        self.td = threshold
        self.coef_ = None
        self.intercept_ = None
        self.max_iters = max_iters

    def _fit(self,X,y):
        return self

    def _eval(self, X):
        return np.dot(X, self.coef_.T)+self.intercept_

    def fit(self, X, y):
        if type(X) != type(y):
            raise TypeError(
                "'X' and 'y' should be the same type.")

        if isinstance(X, pd.DataFrame):
            X = X.values
            y = y.values
        elif isinstance(X, np.ndarray):
            pass
        elif isinstance(X, list):
            X = np.array(X)
            y = np.array(y)
        else:
            raise TypeError(
                "Invalid data type. It should be 'pandas.DataFrame', 'numpy.ndarray' or python 'list'.")

        self._fit(X, y)
        return self

    def predict(self, X):
        y_proba = self._eval(X)

        if y_proba.shape[1] == 1:
            return np.where(y_proba < self.td, 0, 1).squeeze()
        else:
            return np.argmax(y_proba, axis=1)

    def predict_proba(self, X):
        y_proba = self._eval(X)

        if y_proba.shape[1] == 1:
            return self.sigmoid(y_proba).squeeze()
        else:
            return y_proba

    def sigmoid(self,x):
        return 1/(1+np.exp(-x))