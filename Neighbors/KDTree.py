import numpy as np
from collections import deque
import heapq as hq

class K_heap:
    def __init__(self, maxlen):
        self.maxlen = maxlen
        self.deposit = []

    def add(self, value):
        d, idx = value

        if len(self.deposit) >= self.maxlen:
            if d > self.deposit[0][0]:
                hq.heappop(self.deposit)
                hq.heappush(self.deposit, value)
            else:
                pass
        else:
            hq.headppush(self.deposit, value)

        self.min_value = self.deposit[0][0]

        return self




class LeafNode:
    def __init__(self, root):
        self.root = root
        self.isLeafNode = True

    def __str__(self):
        return "This leaf node contain %d points."%len(self.root)


class InnerNode:
    def __init__(self, root, split_dim, split_value, left_node, right_node):
        self.root = root
        #self.root_idx = root_idx
        self.left_node = left_node
        self.right_node = right_node
        self.split_dim = split_dim
        self.split_value = split_value
        self.isLeafNode = False



class KDTree:
    def __init__(self, X, y, K=1, leafsize=10, metrics='Euclidean'):
        self.data = X
        self.label = y
        self.leafsize = leafsize
        self.K = K

        if metrics == 'Euclidean':
            self._calc_d = self._calc_eu_distance
        elif metrics == 'Manhattan':
            self._calc_d = self._calc_ma_distance
        else:
            raise ValueError("Invalid value for parameter 'metrics'. It should be either 'Euclidean' or 'Manhattan'.")

        self.tree = self._build_tree(np.arange(self.data.shape[0]))


    def _build_tree(self, idx):
        if len(idx) <= self.leafsize:
            return LeafNode(idx)
        else:
            tmp_dt = self.data[idx]

            split_dim = np.argmax(np.var(tmp_dt, axis=0))
            split_value = self._get_midpoint(tmp_dt[:,split_dim])

            """

            Here is a very confused and complicated part of index slice for me.

            Example:
            >>> idx
            >>> array([1,4,6,8,9])
            >>> tmp_dt = self.data[idx]; tmp_dt.index # This is a invaild syntax, just for explanation.
            >>> array([0,1,2,3,4])

            Suppose we get the split_dim 2 and split_value 4
            >>> np.where(tmp_dt[:,2] == 4)
            >>> (array([1],dtype=int64), )

            Then root_idx = 1 because of the result of np.where(tmp_dt[:,2] == 4)[0][0]

            Here we must notice that this 1 is the index of tmp_dt, and it also represents idx[1] of
            which value is 4. Then idx becomes array([1,6,8,9])

            So the delete-thing are supposed to delete 4 from idx and delete the second row from tmp_dt.

            This explanation is not quite clear and I use it to remind me one day I review these codes.

            """
            root_idx = np.where(tmp_dt[:,split_dim] == split_value)[0][0]

            tmp_root = idx[root_idx]
            idx = np.delete(idx, root_idx)
            tmp_dt = np.delete(tmp_dt, root_idx, axis=0)

            left_idx = np.where(tmp_dt[:,split_dim] < split_value)[0]
            right_idx = np.where(tmp_dt[:,split_dim] >= split_value)[0]

            return InnerNode(tmp_root, split_dim, split_value,
                            self._build_tree(idx[left_idx]),
                            self._build_tree(idx[right_idx]))

    def find_K_nn_pos(self, X):
        if len(X.shape) == 1:
            X = X.reshape(1,-1)

        for x in X:
            self.single_path = deque()
            self.knn_points = K_heap(maxlen=self.K)
            self._search_route(x, self.tree)
            self._trace_back_for_Knn(x)


    def _search_route(self, X, current_node):
        self.single_path.appendleft(current_node)
        if current_node.isLeafNode:
            self.current_nearest_id = current_node.root
            return self.single_path
        else:
            s_dim, s_value = current_node.split_dim, current_node.split_value
            if X[s_dim] < s_value:
                return self._search_route(X, current_node.left_node)
            else:
                return self._search_route(X, current_node.right_node)

    def _trace_back_for_Knn(self, X):
        for nodes in self.single_path:
            if nodes.isLeafNode:
                current_root = self.data[nodes.root]

                dist = self._calc_d(X, current_root)

                for pairs in zip(1/dist, nodes.root):
                    self.knn_points.add(pairs)

            else:
                current_root = self.data[nodes.root]
                dist = self._calc_d(X, current_root)





    def _get_midpoint(self, ndarray):
        K = len(ndarray) // 2

        return np.sort(ndarray)[K]

    def _calc_eu_distance(self, x1, x2):
        if len(x1.shape) == 1:
            x1 = x1.reshape(1,-1)
        return np.square(x1-x2).sum(axis=1)

    def _calc_ma_distance(self, x1, x2):
        if len(x1.shape) == 1:
            x1 = x1.reshape(1,-1)
        return np.abs(x1-x2).sum(axis=1)



if __name__ == '__main__':
    test_data = np.array([[2,3],[5,4],[9,6],[4,7],[8,1],[7,2]])

    kt = KDTree(test_data, y=None, leafsize=1, metrics='Euclidean')

    tree = kt.tree

    print(tree.root)
    print(tree.left_node.root)
    print(tree.right_node.root)

    try_data = np.random.randn(2040,20)
    kt1 = KDTree(try_data, y=None, leafsize=10)
    print(kt1.tree.left_node.left_node.root)