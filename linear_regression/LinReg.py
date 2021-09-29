import numpy as np

from linear_regression.GradientDescent import GradientDescent


class LinReg:
    def __init__(self, dataset):
        self.dataset = dataset
        self.m = len(self.dataset)
        self.grad_desc = GradientDescent(dataset)

    def calc_linreg(self, x, k, b):
        return k * x + b

    def calc_sum(self, k, b):
        s = 0

        for entry in self.dataset:
            Xi = int(entry[0])
            Yi = float(entry[1])
            s += (self.calc_linreg(Xi, k, b) - Yi) ** 2

        return s

    def calc_lost(self, k, b):
        return 1 / (2 * self.m) * self.calc_sum(k, b)

    def calculate_brute(self, step, range):
        max_range = abs(range)
        min_range = -max_range
        min_lost = 10000000
        best_params = []
        for i in np.arange(min_range, max_range, step):
            for j in np.arange(min_range, max_range, step):
                lost = self.calc_lost(i, j)
                print('lost function for i: {}, j: {} returns {}'.format(i, j, lost))
                if lost < min_lost:
                    min_lost = lost
                    best_params = [i, j]

        print('{} : k = {}, b = {}'.format(min_lost, best_params[0], best_params[1]))
        return best_params[0], best_params[1]

    def calculate_grad_desc(self, learning_rate, threshold):
        return self.grad_desc.calculate(learning_rate, threshold)

    def calculate_norm_eq(self):
        pass
