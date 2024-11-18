import os, sys
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.style as mplstyle
from matplotlib import cm, tri
from scipy.optimize import minimize as minimise
from breakage_population_balance import breakageModel
from breakage_population_balance.arrayManipulation import \
	fractional_percentage, fractional_percentage_array

from test_fitting import \
	x, zero, PSMs, time_to_step

mplstyle.use("fast")

CLASSES = 100
NT = 600
TF = 3000

t = np.logspace(-2, 3.3, NT+1, dtype=np.float64) # np.linspace(0, TF, NT+1, dtype=np.float64) 

if __name__ == "__main__":
	b = 3.85815191
	e = 2.15005062
	a = 4.83796328
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
	solution1 = model1.solve("pyf_simple_breakage") #[1:]

	ids = np.where(t >= 10)
	idx = np.where((1e-6 < x) & (x < 2e-3))

	xx, tt = np.meshgrid(x[idx], t[ids][::10])

	solution1 = np.take_along_axis(solution1[ids][::10], np.array(idx), 1)

	fig = plt.figure()

	ax = fig.add_subplot(111, projection="3d")

	ax.set_box_aspect([1.0, 1.0, 0.5])

	xrav = np.log(xx).ravel()
	trav = np.log(tt).ravel()
	solrav = solution1.ravel()

	# Create the Triangulation; no triangles so Delaunay triangulation created.
	# triang = tri.Triangulation(xrav, trav)

	# # Mask off unwanted triangles.
	# xmid = xrav[triang.triangles].mean(axis=1)
	# ymid = trav[triang.triangles].mean(axis=1)
	# mask = np.abs(xmid**2 + ymid**2) < 10
	# triang.set_mask(mask)

	# tri.Triangulation(xrav, trav)
	# ax.plot_trisurf(trav, xrav, solrav[::-1], alpha=1, linewidth=0, cmap=cm.coolwarm, antialiased=False)


	# logx = np.log(x)
	# lenx = x.shape[0]

	# np.linspace(x[0], x[1])

	ax.plot_surface(np.log(tt), np.log(xx), solution1[:, ::-1], linewidth=0.2, alpha=0.7, rstride=1, cstride=1, cmap=cm.coolwarm, antialiased=False)
	ax.plot(np.log(tt[3, :]), np.log(xx[3, :]), solution1[:, ::-1][3]+ 0.001, 'r-', alpha=0.5)
	ax.plot(np.log(tt[-3, :]), np.log(xx[-3, :]), solution1[:, ::-1][-3]+ 0.001, 'r-', alpha=0.5)

	ax.set_ylabel("y")
	# ax.yaxis.set_scale('log')

	# plt.xscale("log")
	# plt.yscale("log")

	# plt.ylim([1e-5, 1e-2])
	ax.grid(False)
	ax.set_xticks([])
	ax.set_yticks([])
	ax.set_zticks([])
	ax.view_init(40, 0) 

	plt.savefig(os.path.splitext(__file__)[0] + '.png')
	# plt.show()