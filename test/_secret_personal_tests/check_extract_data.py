from test_1D_fitted import x
import numpy as np
from matplotlib import pyplot as plt

i = np.arange(x.shape[0])
X = np.logspace(-8, -2, x.shape[0])

plt.yscale("log")
plt.plot(i, x)
plt.plot(i, X)
plt.show()