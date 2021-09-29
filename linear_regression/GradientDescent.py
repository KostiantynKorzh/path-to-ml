class GradientDescent:

    def __init__(self, dataset):
        self.dataset = dataset
        self.m = len(dataset)

    def calculate(self, learning_rate, threshold):
        k, b = 0, 0

        while True:
            new_k, new_b = self.calc_parameters(learning_rate, k, b)
            print(new_k, new_b)

            if abs(new_k - k) < threshold and abs(new_b - b) < threshold:
                print(str(k) + " : " + str(b))
                return k, b

            k = new_k
            b = new_b

    def calc_parameters(self, learning_rate, k, b):
        sum_k = 0
        sum_b = 0

        for entry in self.dataset:
            x = int(entry[0])  # feature
            y = float(entry[1])  # target

            iter_sum = (self.lin_reg(x, k, b) - y)
            # o0 -> derivative when j = 0
            sum_k += iter_sum * x
            # o1 - derivative when j = 1
            sum_b += iter_sum

        # (1 / self.m) * sum_k(b) <--- derivative of Cost function
        new_k = k - (learning_rate / self.m) * sum_k
        new_b = b - (learning_rate / self.m) * sum_b

        return new_k, new_b

    def lin_reg(self, x, k, b):
        return k * x + b
