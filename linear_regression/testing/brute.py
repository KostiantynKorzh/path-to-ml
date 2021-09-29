from linear_regression.LRPlot import LRPlot
from linear_regression.LinReg import LinReg
from linear_regression.testing import tester

linreg = LinReg(tester.dataset)
lr_plot = LRPlot(tester.dataset)


k, b = linreg.calculate_brute(step=0.1, range=20)
lr_plot.compare_on_plot(k, b)
