from ..base import BaseBuilder
from KDTree import KDTree

import numpy as np


class KNNClassifier(BaseBuilder):
    """
    
    K-nearest neighbors algorithm deposited in k-d tree structure.

    Param:
    - n_neighbors : the number of neighbors used for classification. Defaultly 10.
    - distance    : what kind of distance the model will use to calc distance. 
                    Defaultly use Euclidean distance. Options are 'Euclidean' or 'Manhattan'.
    """
    def __init__(self, n_neighbors=10, distance='Euclidean'):
        super(KNNClassifier, self).__init__()

        self.K = n_neighbors

        self.metrics = distance

    def _fit(self, X, y):
        pass

    def _eval(self, X):
        pass