from breakage_population_balance import breakageModel
from test import plot_against_analytical
import numpy as np

classes = 100
nt = 100
Tf = 5

x = np.logspace(-2, 1, classes)
t = np.linspace(0, Tf, nt+1)

k = lambda x: x**2

# kernel function
def Phi(x, y):
    if x < y:
        pass
    else:
        return 0
    return x / y

# Works on a fraction based approach, so number needs to be converted to
# fraction.
IC = np.exp(-x)*x

# Run and solve the breakage problem
model = breakageModel(IC, t, x, k, kernel=Phi)
solution = model.solve_fraction()

# =============================================================================

plot_against_analytical(solution, model, plot_as="fraction")
# plot_against_analytical(solution/x, model, plot_as="number")