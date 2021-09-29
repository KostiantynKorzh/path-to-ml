from linear_regression.GradientDescent import GradientDescent
from linear_regression.LRPlot import LRPlot
from linear_regression.LinReg import LinReg
from linear_regression.testing import tester

lr_plot = LRPlot(tester.dataset)

linreg = LinReg(tester.dataset)

grad_desc = GradientDescent(linreg.dataset)
k, b = grad_desc.calculate(learning_rate=0.001, threshold=0.0001)
lr_plot.compare_on_plot(k, b)
