from breakage_population_balance import breakageModel
from test import plot_against_analytical
import numpy as np

classes = 100
nt = 100
Tf = 5

x = np.logspace(-2, 1, classes, dtype=np.float64)
t = np.linspace(0, Tf, nt+1, dtype=np.float64)

k = lambda x: x**2

# daughter particle distribution function
def Beta(x, y):
	if x < y:
		return 2./y
	else:
		return 0.
Beta = np.vectorize(Beta, otypes=[np.float64])

IC = np.exp(-x)

# Run and solve the breakage problem
model = breakageModel(IC, t, x, k, beta=Beta)
solution = model.solve_number()

# =============================================================================

print(
	f"Mass before:\t{np.sum(IC*x)}\n" +
	f"------------\n" +
	f"Mass after:\t{np.sum(solution[-1]*x)}"
)

plot_against_analytical(solution, model, plot_as="number")
# plot_against_analytical(solution*x, model, plot_as="fraction")