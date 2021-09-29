import matplotlib.pyplot as plt
import numpy as np


class LRPlot:

    def __init__(self, dataset):
        self.dataset = dataset
        self.X = []
        self.Y = []
        for entry in self.dataset:
            self.X.append(int(entry[0]))
            self.Y.append(float(entry[1]))

    def compare_on_plot(self, k, b):
        (lib_k, lib_b) = np.polyfit(self.X, self.Y, 1)
        lib_result = np.polyval([lib_k, lib_b], self.X)
        result = np.polyval([k, b], self.X)
        plt.plot(self.X, lib_result, color='red', label='lib result')
        plt.plot(self.X, result, label='my result')
        plt.scatter(self.X, self.Y)
        plt.legend()
        plt.show()
