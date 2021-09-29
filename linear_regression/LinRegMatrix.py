import numpy as np

from linear_regression.LinReg import LinReg


class LinRegMatrix(LinReg):

    def __init__(self, dataset):
        super().__init__(dataset)
        self.feature_matrix = np.ones((self.m, 2))
        self.target_matrix = np.zeros((self.m, 1), dtype=float)
        for x in range(0, self.m):
            self.feature_matrix[x][1] = self.dataset[x][0]
            self.target_matrix[x] = self.dataset[x][1]

    def calc_sum(self, k, b):
        s = 0

        hypothesis_matrix = np.array([b, k]).reshape((2, 1))

        result_matrix = np.matmul(self.feature_matrix, hypothesis_matrix)

        diff_array = np.subtract(result_matrix, self.target_matrix)

        for diff in diff_array:
            s += diff ** 2

        print('matrix')
        return s[0]
