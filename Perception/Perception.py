from ..base import BaseBuilder

import numpy as np

class PerceptionClassifier(BaseBuilder):
    '''
    
    The dual algorithm of Perception method.
    More details about the function can be found 
    from the codelib.base.BaseBuilder

    Attention:
    This model can only work well on toy dataset. So it cannot be put
    into application. I just write it to test the algorithm.

    '''
    def __init__(self, learning_rate=1e-2, with_bias=True, threshold=0.5, max_iters=10000):
        super(PerceptionClassifier,self).__init__()
        self.lr = learning_rate
        self.with_bias = with_bias
        self.td = threshold
        self.coef_ = None
        self.intercept_ = None
        self.max_iters = max_iters

        warn_text = """
        This model can only work well on toy dataset. So it cannot be put
    into application. I just write it to test the algorithm.
        """

        print('Warning:', warn_text)

    def _fit(self, X, y):
        # new y to suit the loss function
        y = np.where(y<1, -1, 1)
        # init
        N,c = X.shape
        # a: (a1,a2,...,aN).T
        # b: bias
        a = np.zeros((1,N))
        b = 0
        # gram matrix
        g_m = np.dot(X, X.T)

        idx = 0
        iter_cnt = 0
        while idx < N:
            pre_dot = a * y

            if iter_cnt > self.max_iters:
                break
            else:
                iter_cnt += 1

            if (np.dot(pre_dot, g_m[idx].T)+b)*y[idx] <= 0:
                # update params
                a[0,idx] += self.lr
                b += self.lr * y[idx]

                idx = 0
            else:
                idx += 1

        self.coef_ = np.dot(a*y, X)
        self.intercept_ = b

