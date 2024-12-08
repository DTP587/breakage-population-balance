import os, sys
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider
from scipy.optimize import minimize as minimise
from breakage_population_balance import breakageModel
from breakage_population_balance.arrayManipulation import \
	fractional_percentage, fractional_percentage_array

from test_fitting import \
	x, zero, PSMs, time_to_step

CLASSES = 100
NT = 600
TF = 3000

t = np.logspace(-2, 3.3, NT+1, dtype=np.float64) # np.linspace(0, TF, NT+1, dtype=np.float64) 

if __name__ == "__main__":
	b = 3.007
	e = 2.396
	a = 4.816
	# rate function
	k = lambda x: (x/(b*1e-3))**e

	# kernel function
	def Phi(x, y):
		if x < y:
			# Spherical chickens in a vacuum
			v_ = (1/6) * np.pi * y ** 3
			v  = (1/6) * np.pi * x ** 3
			return ( v / v_ )**(a*1e-1)
		else:
			return 0
	Phi = np.vectorize(Phi, otypes=[np.float64])

	# Run and solve the breakage problem
	model1 = breakageModel(zero, t, x, k, kernel=Phi)
	solution1 = model1.solve("pyf_simple_breakage")

	zero2 = np.roll(zero, 15)

	# Run and solve the breakage problem
	model2 = breakageModel(zero2, t, x, k, kernel=Phi)
	solution2 = model2.solve("pyf_simple_breakage")

	fig, ax = plt.subplots(figsize=[5,7])

	mtoum = 1e6

	# 0.5*(solution1[1:].T + solution2[1:].T)

	ax.pcolor(
		x*1e6, t[1:], np.maximum(solution1[1:], solution2[1:]),
		cmap='cividis', label="Numerical Solution", rasterized=True
	)

	plt.xscale("log")
	plt.yscale("log")

	ax.set_ylabel("Time [s]")
	ax.set_xlabel(r"Size [$\mu m$]")

	plt.ylim([1e-2, 1.8e3])
	plt.xlim([1e1, 1e4])

	ax.invert_xaxis()
	ax.invert_yaxis()

	plt.tight_layout()
	# plt.show()
	plt.savefig(os.path.splitext(__file__)[0] + '.pdf', dpi=400)