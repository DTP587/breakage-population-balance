from breakage_population_balance import breakageModel
from breakage_population_balance.arrayManipulation import apply_function
from test import plot_simple, plot_together
import numpy as np

# Currently not working - need to extend to higher dimensional integrals.
# 1D - simple ODE
# 2D - 4D higher dimension ODE
# 5D monte-carlo

CLASSES_x = 100
CLASSES_a = 20
TIME_STEPS = 21
TIME_FINAL = 20

x = np.logspace(-6, -2, CLASSES_x, dtype=np.float64)
a = np.logspace(0, -1, CLASSES_a, dtype=np.float64)
t = np.linspace(0, TIME_FINAL, TIME_STEPS+1, dtype=np.float64)

k = lambda x, a: (x/(a*0.0043))**2.2

# kernel function
# the trailing _ indicates the parent particle
def Phi(x, a, x_, a_):
    v_ = (1/6) * np.pi * x_ ** 3
    v  = (1/6) * np.pi * x ** 3
    if v < v_:
        return ( v / v_ )**(0.43)
    else:
        return 0
Phi = np.vectorize(Phi, otypes=[np.float64])

f_ic = lambda x, a: np.exp(-x)* ((1/6) * np.pi * x ** 3) if a < 0.2 else 0
f_ic = np.vectorize(f_ic, otypes=[np.float64])

IC = apply_function([x, a], f_ic)
IC = IC/np.sum(IC)

# Run and solve the breakage problem
model1 = breakageModel(IC, t, [x, a], k, kernel=Phi)
# solution1 = model1.solve("cyf_2D_breakage")

from matplotlib import pyplot as plt
from matplotlib import cm

fig = plt.figure()

ax = fig.add_subplot(111)

xx, aa = np.meshgrid(x, a, indexing="ij")

plt.pcolor(xx, aa, IC, cmap=cm.viridis)
# ax.plot_surface(xx, aa, IC, cmap=cm.viridis, linewidth=0, antialiased=False)

ax.set_xlabel("x")
ax.set_ylabel("a")

plt.tight_layout()
plt.show()