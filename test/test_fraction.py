from breakage_population_balance import breakageModel
from test import plot_against_analytical
import numpy as np

classes = 100
nt = 100
Tf = 5

x = np.logspace(-2, 1, classes, dtype=np.float64)
t = np.linspace(0, Tf, nt+1, dtype=np.float64)

k = lambda x: x**2

# kernel function
def Phi(x, y):
    if x < y:
        return x / y
    else:
        return 0.
Phi = np.vectorize(Phi, otypes=[np.float64])

IC = np.exp(-x)*x

# Run and solve the breakage problem
model = breakageModel(IC, t, x, k, kernel=Phi)
solution = model.solve("cyf_simple_breakage")

# =============================================================================

print(
	f"Mass before:\t{np.sum(IC)}\n" +
	f"------------\n" +
	f"Mass after:\t{np.sum(solution[-1])}"
)

plot_against_analytical(solution, model, plot_as="fraction")
# plot_against_analytical(solution/x, model, plot_as="number")