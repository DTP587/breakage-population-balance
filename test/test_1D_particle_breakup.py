from breakage_population_balance import breakageModel
from plotting import plot_simple, plot_together
import numpy as np

classes = 100
nt = 501
Tf = 1800

x = np.logspace(-6, -2, classes, dtype=np.float64)
t = np.linspace(0, Tf, nt+1, dtype=np.float64)

k = lambda x: (x/0.0043)**2.2

# kernel function
def Phi(x, y):
    if x < y:
        # Spherical chickens in a vacuum
        v_ = (1/6) * np.pi * y ** 3
        v  = (1/6) * np.pi * x ** 3
        return ( v / v_ )**(0.43)
    else:
        return 0
Phi = np.vectorize(Phi, otypes=[np.float64])

IC = np.exp(-x)* ((1/6) * np.pi * x ** 3)

# Run and solve the breakage problem
model1 = breakageModel(IC, t, x, k, kernel=Phi)
solution1 = model1.solve("pyf_simple_breakage")

t = np.logspace(0, np.log10(Tf), nt+1, dtype=np.float64)

model2 = breakageModel(IC, t, x, k, kernel=Phi)
solution2 = model2.solve("pyf_simple_breakage")

# =============================================================================

print(
	f"Mass before:\t{np.sum(IC)}\n" +
	f"------------\n" +
	f"Mass after:\t{np.sum(solution1[-1])}"
)

plot_together([solution1, solution2], [model1, model2], n_plots=5, plot_as="fraction")
# plot_simple(solution1, model1, n_plots=5, plot_as="fraction")
# plot_simple(solution2, model2, n_plots=5, plot_as="fraction")