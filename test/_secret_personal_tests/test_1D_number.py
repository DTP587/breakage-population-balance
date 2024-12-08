from breakage_population_balance import breakageModel
from breakage_population_balance.arrayManipulation import normalise_to_x
import sys, pickle, os
import numpy as np

def volume(x):
  """
  Assumes particles are spherical
  """
  return (1/6) * np.pi * x ** 3

# Importing test plotting functions
dirname = os.path.dirname(__file__)
sys.path.append(os.path.join(dirname, '..'))
from plotting import plot_simple

nt = 700
Tf = 1800

with open(os.path.join(dirname, "IC.pkl"), 'rb') as file:
  x, IC = pickle.load(file)

vols = volume(x)

t = np.linspace(0, Tf, nt+1, dtype=np.float64)

k = (x/0.0043)**2.2

# daughter particle distribution function
def Beta(v, v_, a=0.43):
  if v < v_:
    return (2/v_)*((v_/v)**(1-a))
  else:
    return 0
Beta = np.vectorize(Beta, otypes=[np.float64])


# # Run and solve the breakage problem
model = breakageModel(IC/vols, t, vols, k, beta=Beta)
uncorrected_solution = model.solve("cyn_simple_breakage")
solution = normalise_to_x(uncorrected_solution, vols)

# =============================================================================

if __name__=="__main__":
  print(
    f"============\n" +
    f"Uncorrected\n" +
    f"Mass before:\t{np.sum(uncorrected_solution[0]*vols)}\n" +
    f"------------\n" +
    f"Mass after:\t{np.sum(uncorrected_solution[-1]*vols)}"
  )
  print(
    f"============\n" +
    f"Corrected\n" +
    f"Mass before:\t{np.sum(solution[0])}\n" +
    f"------------\n" +
    f"Mass after:\t{np.sum(solution[-1])}"
  )
  with open(os.path.join(dirname, '11500_solution.pkl'), 'wb') as f:
    pickle.dump((t, x, solution), f)

  plot_simple(solution, model, n_plots=5, plot_as="fraction")